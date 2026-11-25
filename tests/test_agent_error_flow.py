import asyncio
import copy
import json
import uuid

import pytest
from agents import SQLiteSession, set_tracing_disabled
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

import agentLite
import agentPrompts
import tools
from baseModel import AgentOutput


def tool_call(name, **arguments):
    import json

    return [ResponseFunctionToolCall(
        id=uuid.uuid4().hex, call_id=uuid.uuid4().hex,
        name=name, arguments=json.dumps(arguments), type="function_call",
    )]


def message(text):
    return [ResponseOutputMessage(
        id=uuid.uuid4().hex, role="assistant", status="completed", type="message",
        content=[ResponseOutputText(text=text, annotations=[], type="output_text")],
    )]


def final_message(summary):
    """Supervisor 的预设响应必须符合真实运行使用的输出 Schema。"""
    return message(AgentOutput(
        summary=summary, changed_files=[], test_status="not_run", unfinished_work=[],
    ).model_dump_json())


class ScriptedModel(Model):
    """仅返回预设响应，让真实 SDK 运行工具链，不访问模型服务。"""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.inputs = []
        self.output_schemas = []

    async def get_response(self, *args, **kwargs):
        # 下一次模型请求不能携带没有对应结果的工具调用。
        history = kwargs.get("input", args[1] if len(args) > 1 else [])
        pending = set()
        if isinstance(history, list):
            for item in history:
                if item.get("type") == "function_call":
                    pending.add(item["call_id"])
                elif item.get("type") == "function_call_output":
                    pending.discard(item["call_id"])
        assert not pending, f"Unresolved tool calls in model input: {pending}"
        self.calls += 1
        self.inputs.append(copy.deepcopy(history))
        self.output_schemas.append(kwargs.get("output_schema", args[4] if len(args) > 4 else None))
        return ModelResponse(
            output=next(self.responses), usage=Usage(requests=1),
            response_id=uuid.uuid4().hex,
        )

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


class OfflineMCPServer:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def list_tools(self):
        return []


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    # 禁用远程 tracing，同时避免初始化 exporter 的网络客户端。
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    set_tracing_disabled(True)
    root = tmp_path / "authorized"
    root.mkdir()
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", root)
    monkeypatch.setenv("THREAD_ID", "offline-test")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(agentLite, "AsyncOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(agentLite, "MCPServerStdio", OfflineMCPServer)
    monkeypatch.setattr(agentLite, "SQLiteSession", lambda name, path: SQLiteSession(name, str(tmp_path / "session.db")))

    async def build(responses):
        model = ScriptedModel(responses)
        monkeypatch.setattr(agentLite, "OpenAIResponsesModel", lambda **kwargs: model)
        lite = agentLite.AgentLite()
        await lite.initDSClientWithMCP()
        # 本次验证本地工具异常传播，MCP 服务不参与工具执行。
        for agent in (lite._client, lite._python_agent, lite._cpp_agent):
            agent.mcp_servers = []
        return lite, model, root

    return build


@pytest.mark.parametrize("expert", ["PythonExpert", "CppExpert"])
def test_nested_tool_error_reaches_supervisor_catch(make_agent, expert, capsys):
    async def run():
        lite, model, root = await make_agent([
            tool_call(expert, input="read missing files"),
            tool_call("readFile", path="missing_a.py"),
            tool_call("readFile", path="missing_b.py"),
        ])
        state = tools.EvalContext()
        assert await lite.simpleCallLLM("task", eval_context=state) is False
        assert state.failure_counts == {"readFile_NOT_FOUND": 2}
        assert model.calls == 3
        await lite.close()

    asyncio.run(run())
    assert "工具调用失败：NOT_FOUND_LIMIT" in capsys.readouterr().out


def test_retry_then_success_preserves_boolean_interface(make_agent):
    async def run():
        lite, model, root = await make_agent([
            tool_call("PythonExpert", input="read file"),
            tool_call("readFile", path="missing.py"),
            tool_call("readFile", path="sample.py"),
            message("expert done"), final_message("supervisor done"),
        ])
        (root / "sample.py").write_text("data")
        state = tools.EvalContext()
        assert await lite.simpleCallLLM("task", eval_context=state) is True
        assert state.failure_counts == {}
        assert model.calls == 5
        assert lite._client.instructions == agentPrompts.SUPERVISOR_INSTRUCTIONS
        assert lite._client.output_type is AgentOutput
        assert lite._python_agent.output_type is None
        assert lite._cpp_agent.output_type is None
        await lite.close()

    asyncio.run(run())


def test_missing_query_feedback_reaches_model_and_allows_correction(make_agent):
    """实际 Runner 将参数错误作为对应 call_id 的结果送回专家，随后允许修正。"""
    async def run():
        lite, model, root = await make_agent([
            tool_call("PythonExpert", input="search file"),
            tool_call("searchFile", path="sample.txt", offset=0, max_matches=2),
            tool_call("searchFile", path="sample.txt", query="target", offset=0, max_matches=2),
            message("expert done"), final_message("supervisor done"),
        ])
        (root / "sample.txt").write_text("target\nother\n", encoding="utf-8")
        state = tools.EvalContext()
        try:
            assert await lite.simpleCallLLM("task", eval_context=state) is True
            outputs = [
                json.loads(item["output"])
                for history in model.inputs if isinstance(history, list)
                for item in history if item.get("type") == "function_call_output"
            ]
            assert any(result.get("error_code") == "INVALID_ARGUMENT"
                       and "query" in result["message"] for result in outputs)
            assert any(result.get("total_matches") == 1 for result in outputs)
            assert state.failure_counts == {}  # 搜索成功后清除该工具的参数错误计数。
            assert model.calls == 5
        finally:
            await lite.close()

    asyncio.run(run())


def test_repeated_missing_query_reaches_supervisor_fatal_handler(make_agent):
    async def run():
        lite, model, root = await make_agent([
            tool_call("PythonExpert", input="search file"),
            tool_call("searchFile", path="sample.txt", offset=0, max_matches=2),
            tool_call("searchFile", path="sample.txt", offset=0, max_matches=2),
        ])
        state = tools.EvalContext()
        try:
            assert await lite.simpleCallLLM("task", eval_context=state) is False
            assert state.failure_counts == {"searchFile_INVALID_ARGUMENT": 2}
            assert model.calls == 3
        finally:
            await lite.close()

    asyncio.run(run())


def test_default_context_is_new_for_each_task(make_agent, monkeypatch):
    async def run():
        one_task = lambda: [
            tool_call("PythonExpert", input="read missing"),
            tool_call("readFile", path="missing.py"),
            message("expert result"), final_message("supervisor result"),
        ]
        lite, model, root = await make_agent(one_task() + one_task())
        contexts = []
        original = tools._retryable_error

        def record(state, *args):
            contexts.append(state)
            return original(state, *args)

        monkeypatch.setattr(tools, "_retryable_error", record)
        assert await lite.simpleCallLLM("first task") is True
        assert await lite.simpleCallLLM("second task") is True
        assert contexts[0] is not contexts[1]
        assert [state.failure_counts for state in contexts] == [{"readFile_NOT_FOUND": 1}] * 2
        await lite.close()

    asyncio.run(run())


def test_interactive_entry_catches_error_and_continues(make_agent, monkeypatch, capsys):
    async def run():
        lite, model, root = await make_agent([
            tool_call("PythonExpert", input="read outside"),
            tool_call("readFile", path="../outside.py"),
            final_message("next task done"),
        ])
        prompts = iter(["first task", "next task", "/clear", "/exit"])
        monkeypatch.setattr("builtins.input", lambda: next(prompts))
        assert await lite.callLLMWithMcpServer() == "退出对话"
        assert model.calls == 3
        assert await lite._session.get_items() == []
        await lite.close()

    asyncio.run(run())
    output = capsys.readouterr().out
    assert "工具调用失败：PERMISSION_DENIED" in output
    assert "next task done" in output


@pytest.mark.parametrize("entry", ["callLLMWithLocalSession", "callLLMWithPreRespId"])
def test_sync_entries_catch_error_and_continue(make_agent, monkeypatch, capsys, entry):
    lite, model, root = asyncio.run(make_agent([
        tool_call("CppExpert", input="read outside"),
        tool_call("readFile", path="../outside.py"),
        final_message("next task done"),
    ]))
    prompts = iter(["first task", "next task", "/exit"])
    monkeypatch.setattr("builtins.input", lambda: next(prompts))
    assert getattr(lite, entry)() == "退出对话"
    assert model.calls == 3
    output = capsys.readouterr().out
    assert "工具调用失败：PERMISSION_DENIED" in output
    assert "next task done" in output
    asyncio.run(lite.close())


def test_real_runner_receives_bounded_file_and_expert_outputs(make_agent):
    """检查真正传到下一次 Model.get_response 的结果，不以打印长度代替。"""
    async def run():
        expert_text = "结论与证据。" * 1000
        lite, model, root = await make_agent([
            tool_call("PythonExpert", input="read large file"),
            tool_call("readFile", path="large.log"),
            tool_call("searchFile", path="large.log", query="target_marker"),
            tool_call("readFileRange", path="large.log", start_line=1, end_line=1, start_column=6000),
            message(expert_text), final_message("supervisor done"),
        ])
        (root / "large.log").write_text("padding" * 900 + "target_marker\nline2\n", encoding="utf-8")
        assert await lite.simpleCallLLM("task") is True
        seen_calls = set()
        for history in model.inputs:
            if not isinstance(history, list):
                continue
            for item in history:
                if item.get("type") == "function_call_output":
                    assert isinstance(item["output"], str)
                    assert len(item["output"]) <= 2000
                    result = json.loads(item["output"])
                    if result.get("source", "").endswith(".log") and "expert-" in result["source"]:
                        assert result["truncated"] is True
                        assert tools.resolve_path(tools.EvalContext(), result["source"]).read_text(encoding="utf-8") == expert_text
                    seen_calls.add(item["call_id"])
        assert len(seen_calls) == 4
        assert {"readFile", "searchFile", "readFileRange"} <= {tool.name for tool in lite._tools}
        await lite.close()

    asyncio.run(run())


def test_structured_schema_reaches_model_and_debug_keeps_json_text(make_agent):
    """真实 SDK 接收 Schema、保留专家文本，并校验最终结果后写入调试信息。"""
    async def run():
        lite, model, root = await make_agent([
            tool_call("PythonExpert", input="分析任务"),
            message("专家完成分析，未修改文件，也未运行测试。"),
            final_message("已完成分析"),
        ])
        debug = {}
        try:
            assert await lite.simpleCallLLM("分析任务", debug_info=debug) is True
            assert model.output_schemas[0].output_type is AgentOutput
            assert model.output_schemas[1] is None
            assert model.output_schemas[2].output_type is AgentOutput
            schema = model.output_schemas[0].json_schema()
            assert set(schema["required"]) == {"summary", "changed_files", "test_status", "unfinished_work"}
            assert schema["additionalProperties"] is False
            assert isinstance(debug["final_output"], str)
            assert json.loads(debug["final_output"]) == {
                "summary": "已完成分析", "changed_files": [], "test_status": "not_run", "unfinished_work": [],
            }
            assert debug["usage"]["requests"] == 3
        finally:
            await lite.close()

    asyncio.run(run())


@pytest.mark.parametrize("raw_output", [
    "普通文本，未按 Schema 返回",
    '{"summary":"缺少必填字段"}',
    '{"summary":"结论","changed_files":[],"test_status":"success","unfinished_work":[]}',
    '{"summary":"结论","changed_files":[],"test_status":"not_run","unfinished_work":[],"extra":1}',
    '{"summary":"结论","changed_files":["../outside.py"],"test_status":"not_run","unfinished_work":[]}',
])
def test_invalid_final_output_is_rejected_by_sdk(make_agent, raw_output):
    """结构错误及本地自定义校验失败都必须阻止任务被记录为成功。"""
    from agents.exceptions import ModelBehaviorError

    async def run():
        lite, model, root = await make_agent([message(raw_output)])
        debug = {}
        try:
            with pytest.raises(ModelBehaviorError):
                await lite.simpleCallLLM("分析任务", debug_info=debug)
            assert debug["final_output"] is None
            assert debug["error"].startswith("ModelBehaviorError:")
            assert model.calls == 1
        finally:
            await lite.close()

    asyncio.run(run())
