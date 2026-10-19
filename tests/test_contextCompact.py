import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import asyncio
import copy
import hashlib
import os
import shutil
import uuid
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace
from agents import Agent, AgentHooks, SQLiteSession, OpenAIChatCompletionsModel
from agents.models.interface import Model
from agents.tool_context import ToolContext
from openai import AsyncOpenAI

import agentLite
from contextCompact import compact_session, select_history, _is_summary, summarize_history
from tools import _read_text_range

_WORKSPACE = "/media/xumaozhou/data/AI/local_agent_test/week5_day3_context_pruning"
# 真实对话预算与固定样本分开；这里只限制历史 JSON，不代表整个模型 Context。
_DEFAULT_REAL_HISTORY_CHAR_LIMIT = 48000


def _real_history_char_limit():
    """读取真实四轮验证的独立字符预算，可在 PyCharm 环境变量中配置。"""
    limit = int(os.environ.get("CONTEXT_TEST_HISTORY_CHAR_LIMIT", str(_DEFAULT_REAL_HISTORY_CHAR_LIMIT)))
    if limit < 3:
        raise ValueError("CONTEXT_TEST_HISTORY_CHAR_LIMIT 必须至少为 3")
    return limit


def _directory_instructions(workspace, allowed_root):
    """明确本次验证的项目目录和工具授权根，供三个 Agent 一致使用。"""
    relative_project = Path(workspace).relative_to(allowed_root).as_posix()
    return (
        f"\n本次项目相对目录：{relative_project}\n"
        f"工具授权根目录：{allowed_root}\n"
        f"项目扫描：{relative_project}；源码：{relative_project}/src/pipeline.py；"
        f"数据：{relative_project}/data/tasks_large.json、{relative_project}/data/processing_large.log。\n"
        "固定测试目录：eval_tests；专家日志：.tool_results/文件名。\n"
    )


class FakeSession:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = Path(input_file_path)
        self.output_file_path = Path(output_file_path)

    async def get_items(self, limit=None):
        # 每次调用都从本地文件加载数据
        content = json.loads(
            self.input_file_path.read_text(encoding="utf-8")
        )
        items = content["items"]

        if not isinstance(items, list):
            raise ValueError("Session 数据必须是列表")

        if limit is None:
            return items

        if limit <= 0:
            return []

        return items[-limit:]


    async def add_items(self, item):
        if self.output_file_path.exists():
            content = json.loads(
                self.output_file_path.read_text(encoding="utf-8")
            )
        else:
            content = {"items": []}

        items = content.get("items", [])

        if not isinstance(items, list):
            raise ValueError("Session 数据必须是列表")

        if items is None or len(items) == 0:
            items = []
        if not isinstance(item, list):
            raise ValueError("item 必须是列表")

        items.extend(item)
        content["items"] = items

        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))


    async def clear_session(self):
        content = {"items": []}
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))


@pytest.mark.asyncio
async def test_compact_session():
    fake_summarize_fn = AsyncMock()
    fake_summarize_fn.return_value = "[摘要信息]"

    input_session_json = f'{_WORKSPACE}/session_history.json'
    output_session_json = f'{_WORKSPACE}/selected_context.json'
    session = FakeSession(input_session_json, output_session_json)

    result = await compact_session(session, fake_summarize_fn)
    assert result.get("compacted", False) == True



@pytest.mark.asyncio
async def test_select_history():
    input_session_json = f'{_WORKSPACE}/session_history.json'
    output_session_json = f'{_WORKSPACE}/selected_context.json'
    session = FakeSession(input_session_json, output_session_json)

    select_items = await select_history(session=session, priority_indices=(0,))
    assert len(select_items) > 0

    await session.clear_session()
    await session.add_items(select_items)
    assert Path(output_session_json).exists()



@pytest.mark.parametrize(
    "path, limit",
    [
        (f'{_WORKSPACE}/workspace/data/processing_large.log', 2000),
        (f'{_WORKSPACE}/workspace/data/tasks_large.json', 2000),
    ]
)
def test_read_text_range(path, limit):
    encoded_result = _read_text_range(target=Path(path))
    assert len(encoded_result) <= limit


def _save_debug_json(path, value):
    """保存调试记录，不保存客户端对象或 API Key。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )


def _file_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(root).rglob("*") if path.is_file()
    }


def _recent_two_turns(items):
    """从实际历史提取预期记录，不依赖固定样本的 call_id 或条目数量。"""
    starts = [index for index, item in enumerate(items)
              if item.get("role") == "user" and not _is_summary(item)]
    return items[starts[-2] if len(starts) >= 2 else starts[0]:] if starts else []


def _contains_in_order(items, expected):
    position = 0
    for item in expected:
        try:
            position = items.index(item, position) + 1
        except ValueError:
            return False
    return True


def _tool_records_paired(items):
    pending = set()
    for item in items:
        if item.get("type") == "function_call":
            pending.add(item["call_id"])
        elif item.get("type") == "function_call_output":
            if item["call_id"] not in pending:
                return False
            pending.remove(item["call_id"])
    return not pending


class RecordingModel(Model):
    """转发真实模型调用，记录 SDK 输入；这些记录不是 HTTP 原始报文。"""

    def __init__(self, model, agent_name, requests):
        self.model = model
        self.agent_name = agent_name
        self.requests = requests
        self.round_number = 0

    async def get_response(self, *args, **kwargs):
        record = {
            "round": self.round_number,
            "agent": self.agent_name,
            "instructions": kwargs.get("system_instructions"),
            "input": copy.deepcopy(kwargs["input"]),
            "tools": [{"name": tool.name, "schema": tool.params_json_schema}
                      for tool in kwargs["tools"]],
        }
        self.requests.append(record)
        try:
            response = await self.model.get_response(*args, **kwargs)
            record["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def stream_response(self, *args, **kwargs):
        # 本验证使用非流式 Runner；避免流式调用漏掉调试记录。
        raise NotImplementedError("四轮验证仅支持非流式调用")


class RecordingHooks(AgentHooks):
    """开始事件记录参数，结束事件记录结果；失败调用也能留下开始记录。"""

    def __init__(self, events, starts=None):
        self.events = events
        # 分开保存，避免只有开始事件的失败调用干扰工具结果限长检查。
        self.starts = starts if starts is not None else []
        self.round_number = 0

    async def on_tool_start(self, context, agent, tool):
        record = {
            "round": self.round_number,
            "agent": agent.name,
            "tool": tool.name,
            "call_id": getattr(context, "tool_call_id", None),
            "arguments": getattr(context, "tool_arguments", ""),
        }
        self.starts.append(record)
        print(
            f"[{agent.name}] 开始调用 {tool.name}, "
            f"call_id={record['call_id']}, arguments={record['arguments']}"
        )

    async def on_tool_end(self, context, agent, tool, result):
        self.events.append({
            "round": self.round_number,
            "agent": agent.name,
            "tool": tool.name,
            "call_id": getattr(context, "tool_call_id", None),
            "arguments": getattr(context, "tool_arguments", ""),
            "output": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
        })


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_CONTEXT_TEST") != "1",
    reason="真实 LLM 验证需显式设置 RUN_REAL_LLM_CONTEXT_TEST=1",
)
@pytest.mark.asyncio
async def test_four_round_real_llm():
    """四轮真实对话验收；默认跳过，禁止在普通测试中自动调用 LLM。

    显式启用后，在练习目录下创建独立副本和 SQLiteSession，保留调试产物。
    每轮先选择历史，再调用 AgentLite；第4轮降低选择阈值以覆盖裁剪分支。
    专家不共用 Supervisor Session，避免委派输入干扰真实用户轮次。
    本测试只使用已有本地文件工具，不启动与本练习无关的 MCP 服务。
    历史预算通过 CONTEXT_TEST_HISTORY_CHAR_LIMIT 配置，默认 48000 字符，
    不使用固定样本 manifest 中的 history_char_limit。
    需要可用的 DeepSeek 配置、Docker 和 agent-python:3.10 镜像。
    """
    # 装饰器之外再检查一次，直接调用函数也必须显式启用。
    if os.environ.get("RUN_REAL_LLM_CONTEXT_TEST") != "1":
        pytest.skip("请先显式启用真实 LLM 验证")
    history_char_limit = _real_history_char_limit()
    assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY"

    from agentLite import AgentLite, _limited_expert_output
    from agentPrompts import (
        SUPERVISOR_INSTRUCTIONS, PYTHON_EXPERT_INSTRUCTIONS, CPP_EXPERT_INSTRUCTIONS,
        PYTHON_EXPERT_DESCRIPTION, CPP_EXPERT_DESCRIPTION,
    )
    from tools import EvalContext, runSafeTest

    source = Path(_WORKSPACE)
    manifest = json.loads((source / "eval_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((source / "expected_results.json").read_text(encoding="utf-8"))
    required_texts = json.loads((source / "session_history.json").read_text(encoding="utf-8"))["required_texts"]
    prompts = [line.strip() for line in (source / "prompt.txt").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(prompts) == 4, "练习必须包含四轮提示词"

    run_id = "real_llm_" + uuid.uuid4().hex[:12]
    root = source / "real_llm_runs" / run_id
    root.mkdir(parents=True)
    workspace = root / "workspace"
    test_dir = root / "eval_tests"
    shutil.copytree(source / "workspace", workspace)
    shutil.copytree(source / "eval_tests", test_dir)
    before_files = _file_hashes(workspace)
    fixed_test_hashes = _file_hashes(test_dir)
    report = {
        "passed": False, "run_id": run_id, "history_char_limit": history_char_limit,
        "rounds": [], "checks": {},
    }
    requests, events, starts = [], [], []
    session = SQLiteSession(run_id, str(root / "session.db"))
    client = None

    async def fixed_tests():
        # 使用既有 Docker 工具独立验收，不依据 Agent 的“测试通过”自评。
        payload = json.dumps({"project_path": str(workspace), "test_path": str(test_dir)})
        result = await runSafeTest.on_invoke_tool(
            ToolContext(
                context=EvalContext(workspace=root), tool_name="runSafeTest",
                tool_call_id=uuid.uuid4().hex, tool_arguments=payload,
            ),
            payload,
        )
        return json.loads(result)

    try:
        report["baseline_test"] = await fixed_tests()
        assert report["baseline_test"]["exit_code"] == manifest["expected_before_pytest_exit_code"], \
            "基线测试不符合预期，请检查练习副本或 Docker 环境"

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com")
        real_model = OpenAIChatCompletionsModel(
            model=os.environ.get("CONTEXT_TEST_MODEL", "deepseek-flash"), openai_client=client,
        )
        models = {name: RecordingModel(real_model, name, requests)
                  for name in ("Supervisor", "PythonExpert", "CppExpert")}
        hooks = RecordingHooks(events, starts)
        directory_instructions = _directory_instructions(workspace, root)
        with patch.dict(os.environ, {"THREAD_ID": run_id}):
            lite = AgentLite(session=session)
        python_expert = Agent(
            name="PythonExpert", instructions=PYTHON_EXPERT_INSTRUCTIONS + directory_instructions,
            model=models["PythonExpert"], tools=lite._tools, hooks=hooks,
        )
        cpp_expert = Agent(
            name="CppExpert", instructions=CPP_EXPERT_INSTRUCTIONS + directory_instructions,
            model=models["CppExpert"], tools=lite._tools, hooks=hooks,
        )
        lite._client = Agent(
            name="Supervisor", instructions=SUPERVISOR_INSTRUCTIONS + directory_instructions,
            model=models["Supervisor"], hooks=hooks,
            tools=[
                python_expert.as_tool(
                    tool_name="PythonExpert", tool_description=PYTHON_EXPERT_DESCRIPTION,
                    custom_output_extractor=_limited_expert_output, failure_error_function=None,
                    max_turns=30,
                ),
                cpp_expert.as_tool(
                    tool_name="CppExpert", tool_description=CPP_EXPERT_DESCRIPTION,
                    custom_output_extractor=_limited_expert_output, failure_error_function=None,
                    max_turns=30,
                ),
            ],
        )
        context = EvalContext(workspace=root)

        for number, prompt in enumerate(prompts, 1):
            for model in models.values():
                model.round_number = number
            hooks.round_number = number
            before = await session.get_items()
            recent = copy.deepcopy(_recent_two_turns(before))
            # 独立预算以字符计；预留一个字符，兼容 // 2 估算的取整边界。
            budget = (history_char_limit - 1) // 2
            if number == 4 and before:
                minimal = [before[0], *recent]
                budget = min(budget, len(json.dumps(minimal, ensure_ascii=False)) // 2)
            selected = await select_history(session, history_tokens_thre=budget)
            # 请求之间顺序替换；写入失败时恢复，避免留下半份历史。
            if selected != before:
                try:
                    await session.clear_session()
                    await session.add_items(selected)
                except Exception:
                    await session.clear_session()
                    await session.add_items(before)
                    raise
            # 工具以 context.workspace（运行根目录）解析相对路径，不让模型重写长前缀。
            prompt = prompt.replace(str(source / "workspace"), "workspace")
            prompt = prompt.replace(str(source / "eval_tests"), "eval_tests")
            if number == 4:
                prompt += (
                    "\n最终回答只输出 JSON 对象，字段为 true_error_count、error_line_numbers、"
                    "matching_task_count、matching_task_ids、changed_files、test_status。"
                )
            row = {
                "round": number, "prompt": prompt, "history_tokens_threshold": budget,
                "selected_history_chars": len(json.dumps(selected, ensure_ascii=False)),
                "history_reduced": selected != before,
                "recent_turns_complete": _contains_in_order(selected, recent),
                "priority_constraints_kept": not before or all(
                    text in json.dumps(selected, ensure_ascii=False) for text in required_texts
                ),
            }
            report["rounds"].append(row)
            _save_debug_json(root / f"round_{number}_before.json", {"items": before})
            _save_debug_json(root / f"round_{number}_selected.json", {"items": selected})
            row["runner_completed"] = await asyncio.wait_for(
                lite.simpleCallLLM(prompt, context, debug_info=row, max_turns=30), timeout=300,
            )
            after = await session.get_items()
            _save_debug_json(root / f"round_{number}_after.json", {"items": after})
            outer_requests = [request for request in requests
                              if request["round"] == number and request["agent"] == "Supervisor"]
            row["selected_history_in_model_input"] = bool(outer_requests) and _contains_in_order(
                outer_requests[0]["input"], selected,
            )
            current_files = _file_hashes(workspace)
            row["changed_files"] = sorted(
                path for path in before_files.keys() | current_files.keys()
                if before_files.get(path) != current_files.get(path)
            )
            assert row["runner_completed"], f"第{number}轮执行失败：{row['error']}"
            assert _file_hashes(test_dir) == fixed_test_hashes, "禁止修改固定测试"
            if number <= 2:
                assert not row["changed_files"], "前两轮只允许分析"
            assert set(row["changed_files"]) <= set(manifest["allowed_changed_paths"]), "修改范围超出约定"

        report["final_test"] = await fixed_tests()
        try:
            answer = json.loads(report["rounds"][-1]["final_output"])
        except (TypeError, ValueError):
            answer = {}
        model_tool_results = [
            item["output"] for request in requests for item in request["input"]
            if item.get("type") == "function_call_output"
        ]
        report["checks"] = {
            "four_rounds_completed": len(report["rounds"]) == 4,
            "history_selection_exercised": any(row["history_reduced"] for row in report["rounds"]),
            "history_within_budget": all(row["selected_history_chars"] <= history_char_limit
                                         for row in report["rounds"]),
            "priority_constraints_kept": all(row["priority_constraints_kept"] for row in report["rounds"]),
            "recent_turns_complete": all(row["recent_turns_complete"] for row in report["rounds"]),
            "selected_history_in_model_input": all(row["selected_history_in_model_input"]
                                                   for row in report["rounds"]),
            "tool_calls_and_outputs_paired": all(_tool_records_paired(request["input"]) for request in requests),
            "tool_results_within_budget": bool(events) and all(
                len(event["output"]) <= manifest["per_tool_result_char_limit"] for event in events
            ),
            "model_tool_results_within_budget": bool(model_tool_results) and all(
                isinstance(output, str) and len(output) <= manifest["per_tool_result_char_limit"]
                for output in model_tool_results
            ),
            "final_tests_passed": report["final_test"]["exit_code"] == manifest["expected_after_pytest_exit_code"],
            "final_facts_correct": isinstance(answer, dict) and all(
                answer.get(key) == expected[key] for key in (
                    "true_error_count", "error_line_numbers", "matching_task_count", "matching_task_ids",
                )
            ),
        }
        for filename in ("processing_large.log", "tasks_large.json"):
            matched = []
            for event in events:
                if event["tool"] != "readFile":
                    continue
                arguments = json.loads(event["arguments"])
                if str(arguments.get("path", "")).endswith("data/" + filename):
                    matched.append(json.loads(event["output"]))
            report["checks"][filename + "_truncation_and_source"] = any(
                body.get("truncated") is True and bool(body.get("excerpt"))
                and str(body.get("source", "")).endswith("data/" + filename)
                for body in matched
            )
        report["passed"] = all(report["checks"].values())
        assert report["passed"], "存在未通过的验收项，查看 report.json"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # 无论成功或失败都保存已收集数据，保留副本供人工调试。
        try:
            _save_debug_json(root / "model_requests.json", requests)
            _save_debug_json(root / "tool_starts.json", starts)
            _save_debug_json(root / "tool_results.json", events)
            _save_debug_json(root / "report.json", report)
            print(f"四轮验证产物：{root}")
        finally:
            session.close()
            if client is not None:
                await client.close()


@pytest.mark.parametrize("problem,failed_check", [
    ("old_history", "first_model_input_is_compacted_history_plus_one_prompt"),
    ("split_prompt", "first_model_input_is_compacted_history_plus_one_prompt"),
    ("discard_marker", "no_irrelevant_old_history_in_model_inputs"),
    ("unpaired_tool", "model_tool_records_paired"),
])
def test_day5_support_rejects_invalid_model_inputs(problem, failed_check):
    """只验证输入检查器，不执行真实摘要或任务续接函数。"""
    compacted = [{"role": "user", "content": "历史摘要"}]
    prompt = "继续任务\n完成后报告结果"
    requests = [{"agent": "Supervisor", "input": [
        *compacted, {"role": "user", "content": prompt},
    ]}]
    assert all(_day5_input_checks(requests, compacted, prompt, "DISCARDED_").values())
    if problem == "old_history":
        requests[0]["input"].insert(0, {"role": "user", "content": "不应重新发送的旧原文"})
    elif problem == "split_prompt":
        requests[0]["input"][-1]["content"] = prompt.splitlines()[0]
    elif problem == "discard_marker":
        requests.append({"agent": "PythonExpert", "input": [{"role": "user", "content": "DISCARDED_001"}]})
    else:
        requests[0]["input"].append({"type": "function_call", "call_id": "missing_output"})
    assert _day5_input_checks(requests, compacted, prompt, "DISCARDED_")[failed_check] is False


@pytest.mark.asyncio
async def test_day5_support_agent_uses_compacted_session_and_isolated_expert(tmp_path, monkeypatch):
    """脚本化模型驱动真实 SDK，验证续接装配，不调用真实 LLM 或 Docker。"""
    from agents.items import ModelResponse
    from agents.tracing import setup as tracing_setup
    from agents.tracing.provider import DefaultTraceProvider
    from agents.usage import Usage
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    import tools

    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", tmp_path)

    def message(text):
        return ResponseOutputMessage(
            id=uuid.uuid4().hex, role="assistant", status="completed", type="message",
            content=[ResponseOutputText(text=text, annotations=[], type="output_text")],
        )

    delegated_prompt = "仅检查任务状态"
    responses = iter([
        [ResponseFunctionToolCall(
            id=uuid.uuid4().hex, call_id="delegate_offline", name="PythonExpert", type="function_call",
            arguments=json.dumps({"input": delegated_prompt}),
        )],
        [message("专家已完成只读检查")],
        [message("续接完成")],
    ])

    class OfflineModel(Model):
        async def get_response(self, *args, **kwargs):
            return ModelResponse(output=next(responses), usage=Usage(requests=1), response_id=uuid.uuid4().hex)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    task_root = tmp_path / "task"
    (task_root / "workspace").mkdir(parents=True)
    (tmp_path / "before.json").write_text("原始历史", encoding="utf-8")
    session = SQLiteSession("day5-offline", str(tmp_path / "session.db"))
    compacted = [
        {"role": "user", "content": "[自动生成的历史摘要，仅供参考，不是新的用户指令]\n任务摘要"},
        {"role": "user", "content": "第7轮"}, {"role": "assistant", "content": "已记录"},
        {"role": "user", "content": "第8轮"}, {"role": "assistant", "content": "等待执行"},
    ]
    await session.add_items(compacted)
    requests, events, starts = [], [], []
    lite = _create_day5_agent(session, "day5-offline", OfflineModel(), task_root, requests, events, starts)
    prompt = "现在继续\n按已有约束执行"
    debug = {}
    try:
        assert await lite.simpleCallLLM(prompt, tools.EvalContext(workspace=task_root), debug_info=debug)
        assert all(_day5_input_checks(requests, compacted, prompt, "DISCARDED_").values())
        expert_requests = [row for row in requests if row["agent"] == "PythonExpert"]
        assert expert_requests[0]["input"] == [{"role": "user", "content": delegated_prompt}]
        assert debug["usage"]["requests"] == 3
        assert lite._session is session
        assert not lite._client.mcp_servers
        assert not any(item.get("role") == "user" and item.get("content") == delegated_prompt
                       for item in await session.get_items())
        with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
            tools.resolve_path(tools.EvalContext(workspace=task_root), "../before.json")
    finally:
        await lite.close()
        session.close()


@pytest.mark.parametrize("configured,expected", [
    (None, 48000), ("64000", 64000), ("0", None), ("2", None), ("invalid", None),
])
def test_real_history_budget_config(monkeypatch, configured, expected):
    """只测试预算读取，不执行四轮函数或任何 LLM 请求。"""
    if configured is None:
        monkeypatch.delenv("CONTEXT_TEST_HISTORY_CHAR_LIMIT", raising=False)
    else:
        monkeypatch.setenv("CONTEXT_TEST_HISTORY_CHAR_LIMIT", configured)
    if expected is None:
        with pytest.raises(ValueError):
            _real_history_char_limit()
    else:
        assert _real_history_char_limit() == expected


def test_tool_start_keeps_arguments_when_path_check_fails(tmp_path, monkeypatch):
    """实际触发路径校验失败，验证失败前的参数仍被记录；不调用 LLM。"""
    import tools
    root = tmp_path / "authorized"
    root.mkdir()
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", root)
    payload = json.dumps({"path": "../outside.txt"})
    context = ToolContext(
        context=tools.EvalContext(workspace=root), tool_name="readFile",
        tool_call_id="failed_call", tool_arguments=payload,
    )
    starts, events = [], []
    hooks = RecordingHooks(events, starts)
    hooks.round_number = 2
    agent, tool = SimpleNamespace(name="PythonExpert"), SimpleNamespace(name="readFile")

    async def run():
        await hooks.on_tool_start(context, agent, tool)
        tools.resolve_path(context.context, "../outside.txt")
        await hooks.on_tool_end(context, agent, tool, "不应执行到这里")

    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        asyncio.run(run())
    assert starts == [{
        "round": 2, "agent": "PythonExpert", "tool": "readFile",
        "call_id": "failed_call", "arguments": payload,
    }]
    assert events == []


def test_successful_tool_has_separate_start_and_result_records():
    payload = '{"path":"workspace/src/pipeline.py"}'
    context = ToolContext(
        context=None, tool_name="readFile", tool_call_id="successful_call", tool_arguments=payload,
    )
    starts, events = [], []
    hooks = RecordingHooks(events, starts)
    agent, tool = SimpleNamespace(name="PythonExpert"), SimpleNamespace(name="readFile")

    async def run():
        await hooks.on_tool_start(context, agent, tool)
        await hooks.on_tool_end(context, agent, tool, '{"excerpt":"代码内容"}')

    asyncio.run(run())
    assert starts[0]["call_id"] == events[0]["call_id"] == "successful_call"
    assert "output" not in starts[0]
    assert events[0]["output"] == '{"excerpt":"代码内容"}'


@pytest.mark.skipif(
    os.environ.get("RUN_DAY5_REAL_SUMMARY") != "1",
    reason="需要手动启用真实 LLM 摘要测试",
)
@pytest.mark.asyncio
async def test_day5_real_summary():
    from contextCompact import summarize_history

    case_dir = Path(
        "/media/xumaozhou/data/AI/local_agent_test/week5_day5_compaction"
    )
    run_id = "summary_" + uuid.uuid4().hex[:12]
    run_dir = case_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    def save_json(filename, data):
        (run_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 每次使用独立 Session，避免混入上次实验的数据。
    session = SQLiteSession(run_id, str(run_dir / "session.db"))
    history = json.loads(
        (case_dir / "session_history.json").read_text(encoding="utf-8")
    )["items"]
    await session.add_items(history)

    before = await session.get_items()
    save_json("before.json", {"items": before})

    async def summarize_and_record(old_history):
        # 保存真正交给摘要 Agent 的旧历史，方便确认压缩范围。
        save_json("summary_input.json", {"items": json.loads(old_history)})

        summary = await summarize_history(old_history)
        (run_dir / "summary.txt").write_text(summary.get("summary", ""), encoding="utf-8")
        return summary

    result = await compact_session(
        session,
        summarize_fn=summarize_and_record,
        threshold=8000,
        reserved_tokens=2500,
        keep_recent_turns=2,
    )

    after = await session.get_items()
    save_json("compacted_context.json", {"items": after})
    save_json("compaction_result.json", result)
    print(f"本次测试产物：{run_dir}")

    assert result["compacted"], result
    assert result["after"] < result["before"]



def _create_day5_agent(session, run_id, model, task_root, requests, events, starts):
    """复用 AgentLite 和通用提示词，只在测试中注入本次目录与记录器。"""
    from agentLite import AgentLite, _limited_expert_output
    from agentPrompts import (
        SUPERVISOR_INSTRUCTIONS, PYTHON_EXPERT_INSTRUCTIONS, CPP_EXPERT_INSTRUCTIONS,
        PYTHON_EXPERT_DESCRIPTION, CPP_EXPERT_DESCRIPTION,
    )

    # task_root 只包含工作副本和固定测试，原历史、答案和验收配置放在其外。
    directory_instructions = (
        f"\n本次工具授权根目录：{task_root}\n"
        "项目相对目录：workspace；源码入口：workspace/src/report.py；"
        "数据文件：workspace/data/tasks.json；固定测试目录：eval_tests。\n"
        "工具参数优先使用上述相对路径；changed_files 使用相对于项目目录的路径。\n"
    )
    with patch.dict(os.environ, {"THREAD_ID": run_id}):
        lite = AgentLite(session=session)
    models = {name: RecordingModel(model, name, requests)
              for name in ("Supervisor", "PythonExpert", "CppExpert")}
    hooks = RecordingHooks(events, starts)
    lite._python_agent = Agent(
        name="PythonExpert", instructions=PYTHON_EXPERT_INSTRUCTIONS + directory_instructions,
        model=models["PythonExpert"], tools=lite._tools, hooks=hooks,
    )
    lite._cpp_agent = Agent(
        name="CppExpert", instructions=CPP_EXPERT_INSTRUCTIONS + directory_instructions,
        model=models["CppExpert"], tools=lite._tools, hooks=hooks,
    )
    lite._client = Agent(
        name="Supervisor", instructions=SUPERVISOR_INSTRUCTIONS + directory_instructions,
        model=models["Supervisor"], hooks=hooks,
        tools=[
            expert.as_tool(
                tool_name=expert.name, tool_description=description,
                custom_output_extractor=_limited_expert_output, failure_error_function=None,
                max_turns=30,
                # 专家不共用 Supervisor 的 Session，避免委派输入混入用户历史。
            )
            for expert, description in (
                (lite._python_agent, PYTHON_EXPERT_DESCRIPTION),
                (lite._cpp_agent, CPP_EXPERT_DESCRIPTION),
            )
        ],
    )
    return lite


def _day5_input_checks(requests, compacted, prompt, discard_marker):
    """核对 SDK 模型入口，避免只检查 Session 文件却重新发送了完整旧历史。"""
    supervisor_requests = [row for row in requests if row["agent"] == "Supervisor"]
    expected_input = [*compacted, {"role": "user", "content": prompt}]
    return {
        "first_model_input_is_compacted_history_plus_one_prompt": bool(supervisor_requests)
        and supervisor_requests[0]["input"] == expected_input,
        "no_irrelevant_old_history_in_model_inputs": bool(requests) and all(
            discard_marker not in json.dumps(row["input"], ensure_ascii=False) for row in requests
        ),
        "model_tool_records_paired": bool(requests) and all(
            _tool_records_paired(row["input"]) for row in requests
        ),
    }


@pytest.mark.skipif(
    os.environ.get("RUN_DAY5_FINAL_TEST") != "1",
    reason="真实摘要与任务续接验收需显式设置 RUN_DAY5_FINAL_TEST=1",
)
@pytest.mark.asyncio
async def test_day5_final_test():
    """Day5 真实续接：摘要压缩、一次完整续接输入、独立产物验收。

    默认跳过；不依赖 Day5 第二步的旧产物，不使用 TaskState 或 Memory 补回历史。
    每次创建独立副本，调用现有 DeepSeek 配置与本地工具，不启动 MCP 服务。
    需要 Docker、agent-python:3.10 镜像和项目虚拟环境中的 pytest。
    """
    if os.environ.get("RUN_DAY5_FINAL_TEST") != "1":
        pytest.skip("请先显式启用 Day5 真实任务续接验收")
    assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY"
    from tools import EvalContext, runSafeTest

    case_dir = Path("/media/xumaozhou/data/AI/local_agent_test/week5_day5_compaction")
    manifest = json.loads((case_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    context_checker = runpy.run_path(str(case_dir / "check_compaction.py"))["check"]
    run_id = "final_" + uuid.uuid4().hex[:12]
    run_dir = case_dir / "runs" / run_id
    # 将可供工具访问的目录与调试产物分开，不能补读原历史或参考答案。
    task_root = run_dir / "task"
    workspace = task_root / "workspace"
    test_dir = task_root / "eval_tests"
    test_dir.mkdir(parents=True)
    shutil.copytree(case_dir / "workspace", workspace)
    shutil.copy2(case_dir / "eval_tests/test_report.py", test_dir / "test_report.py")
    baseline_files = _file_hashes(workspace)
    fixed_test_hashes = _file_hashes(test_dir)
    session = SQLiteSession(run_id, str(run_dir / "session.db"))
    report = {"passed": False, "run_id": run_id, "checks": {}}
    requests, events, starts = [], [], []
    client, lite = None, None

    try:
        assert baseline_files == manifest["workspace_hashes"], \
            "练习 workspace 已偏离初始版本，请先恢复初始文件"
        # 在调用 LLM 前检查 Docker 和基线，环境问题不消耗真实摘要请求。
        payload = json.dumps({"project_path": "workspace", "test_path": "eval_tests"})
        baseline = await runSafeTest.on_invoke_tool(
            ToolContext(
                context=EvalContext(workspace=task_root), tool_name="runSafeTest",
                tool_call_id=uuid.uuid4().hex, tool_arguments=payload,
            ), payload,
        )
        report["baseline_test"] = json.loads(baseline)
        assert report["baseline_test"]["exit_code"] == manifest["expected_baseline_pytest_exit_code"], \
            "基线结果不符合预期，请检查 Docker、依赖或初始文件"

        history = json.loads((case_dir / "session_history.json").read_text(encoding="utf-8"))["items"]
        await session.add_items(history)
        before = await session.get_items()
        recent = copy.deepcopy(_recent_two_turns(before))
        summary_inputs = []
        _save_debug_json(run_dir / "before.json", {"items": before})

        async def summarize_and_record(old_history):
            summary_inputs.append(json.loads(old_history))
            _save_debug_json(run_dir / "summary_input.json", {"items": summary_inputs[-1]})
            # 当前接口返回 {summary, tokens}，直接交回 compact_session。
            summary = await summarize_history(old_history)
            (run_dir / "summary.txt").write_text(summary["summary"], encoding="utf-8")
            return summary

        started = time.perf_counter()
        report["compaction"] = await asyncio.wait_for(compact_session(
            session, summarize_fn=summarize_and_record,
            threshold=manifest["threshold"], reserved_tokens=manifest["reserved_tokens"],
            keep_recent_turns=manifest["keep_recent_turns"],
        ), timeout=180)
        report["summary_seconds"] = time.perf_counter() - started
        report["summary_total_tokens"] = report["compaction"].get("summary_tokens", 0)
        compacted = await session.get_items()
        _save_debug_json(run_dir / "compaction_result.json", report["compaction"])
        _save_debug_json(run_dir / "compacted_context.json", {"items": compacted})
        assert report["compaction"]["compacted"], report["compaction"]
        report["context_check"] = context_checker(compacted, before, manifest)
        _save_debug_json(run_dir / "context_check.json", report["context_check"])
        report["checks"]["summary_receives_only_old_history"] = summary_inputs == [before[:len(before) - len(recent)]]
        report["checks"]["compacted_context_passes"] = report["context_check"]["passed"]
        assert all(report["checks"].values()), "压缩结果未通过检查，查看 context_check.json 和 summary.txt"

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com")
        model = OpenAIChatCompletionsModel(
            model=os.environ.get("CONTEXT_TEST_MODEL", "deepseek-flash"), openai_client=client,
        )
        lite = _create_day5_agent(session, run_id, model, task_root, requests, events, starts)
        # continuation_prompt.txt 是一条多行消息，不能按行拆成多轮请求。
        prompt = (case_dir / "continuation_prompt.txt").read_text(encoding="utf-8").strip()
        report["continuation"] = {"prompt": prompt}
        started = time.perf_counter()
        completed = await asyncio.wait_for(lite.simpleCallLLM(
            prompt, EvalContext(workspace=task_root), debug_info=report["continuation"], max_turns=30,
        ), timeout=600)
        report["continuation_seconds"] = time.perf_counter() - started
        report["checks"].update(_day5_input_checks(requests, compacted, prompt, manifest["discard_marker"]))
        report["checks"]["runner_completed"] = completed
        assert completed, report["continuation"]["error"]
        assert all(report["checks"].values()), "模型输入不符合压缩续接要求，查看 model_requests.json"

        answer = json.loads(report["continuation"]["final_output"])
        _save_debug_json(run_dir / "answer.json", answer)
        # 调用已有检查器独立运行固定 pytest、比较实际文件哈希与最终 JSON。
        evaluation = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-B", str(case_dir / "check_continuation.py"),
             str(run_dir / "answer.json"), "--workspace", str(workspace)],
            capture_output=True, text=True, timeout=60,
        )
        report["evaluation_process"] = {
            "returncode": evaluation.returncode, "stdout": evaluation.stdout, "stderr": evaluation.stderr,
        }
        report["continuation_check"] = json.loads(evaluation.stdout)
        _save_debug_json(run_dir / "continuation_check.json", report["continuation_check"])
        report["checks"]["independent_continuation_check_passed"] = (
            evaluation.returncode == 0 and report["continuation_check"]["passed"]
        )
        report["checks"]["fixed_tests_unchanged"] = _file_hashes(test_dir) == fixed_test_hashes
        report["checks"]["original_workspace_unchanged"] = _file_hashes(case_dir / "workspace") == baseline_files
        report["passed"] = all(report["checks"].values())
        assert report["passed"], "续接产物未通过独立验收，查看 report.json"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # 无论在哪个阶段失败都保留已有证据；原始历史不放入工具授权目录。
        try:
            _save_debug_json(run_dir / "session_after.json", {"items": await session.get_items()})
            _save_debug_json(run_dir / "model_requests.json", requests)
            _save_debug_json(run_dir / "tool_starts.json", starts)
            _save_debug_json(run_dir / "tool_results.json", events)
            _save_debug_json(run_dir / "report.json", report)
            print(f"Day5 真实续接验收产物：{run_dir}")
        finally:
            session.close()
            if lite is not None:
                await lite.close()
            if client is not None:
                await client.close()
