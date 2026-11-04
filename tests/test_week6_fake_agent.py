"""Week6 Day4～Day6：MCPServerStdio 与单个业务 Agent 的联调验收。

测试一：工具发现和生命周期（离线）
--------------------------------
数据来源：week6.py 注册的两个 MCP 工具。
架构：fakeAgentLite.Init → SDK MCPServerStdio → 本地 week6.py 子进程。
检查：无需 API Key，工具恰好两个，重复 Init 不新增连接，close 后会话关闭。

测试二：详情 / 列表查询（离线）
------------------------------
数据来源：week6.py 的合成 TASKS，经 MockTransport 返回。
检查：callMcpTool 返回 BusinessResult；详情隐藏 owner/Token；limit=3 只返回
12345、20000、20001，匹配总数仍为 12，status=partial、truncated=True。

测试三：错误与资源清理（离线）
----------------------------
数据来源：本地未知工具名，以及明确标注的 Mock / AsyncMock。
检查：未初始化时拒绝调用；初始化失败会清理；MCP 错误与业务错误分开；
LLM 分支返回文本、绑定正确的 MCP 对象、保留异常类型，并关闭 HTTP 客户端。
Mock Runner 只验证包装逻辑，不证明真实 LLM 会自主选择工具。

测试四：真实 LLM 查询详情 / 列表（默认跳过）
------------------------------------------
输入：下方两条固定自然语言问题。业务数据仍来自本地 Mock，只对模型发送请求。
架构：真实 DeepSeek → Runner → MCPServerStdio → Service → Mock API。
检查：实际工具名、参数、call_id 对应的业务结果、回答关键事实及来源。
每个用例将问题、工具调用、结果、回答、usage 保存到 pytest 的 tmp_path，
并打印文件路径。关键项检查是初筛，回答是否清晰仍可人工阅读。

Day5 测试：断连、重试配置、stdio 日志边界（离线）
----------------------------------------------
数据来源：模拟的 SDK 断连异常、SDK 构造参数及注入合成密钥的临时服务。
检查：断连保留异常、MCP 不叠加重试、真实子进程 stdout 只有协议且不泄漏秘密。
逐测试说明及三个业务函数的契约见 docs/week6_day5.md。

Day6 测试：自然语言查询与固定 Evals
----------------------------------
数据来源：下方 DAY6_CASES、week6.py 的 Mock 数据及脚本模型。
架构：同一个 Agent → 原生工具护栏 → 本地 MCP → Mock API → 结构化回答。
提供的两个自检默认运行；待填写函数的验收需 RUN_WEEK6_DAY6=1，
真实模型另需 RUN_WEEK6_DAY6_LLM=1。逐测试说明见 docs/week6_day6.md。

运行（仓库根目录）
------------------
离线：.venv/bin/python -B -m pytest tests/test_week6_fake_agent.py -q
真实：RUN_WEEK6_DAY4_LLM=1 .venv/bin/python -B -m pytest tests/test_week6_fake_agent.py -q -s -k LLM
真实测试还需要可用的 OPENAI_API_KEY（沿用 DeepSeek 配置）；显式启用但缺少 Key 时失败。
Day5：RUN_WEEK6_DAY5=1 .venv/bin/python -B -m pytest tests/test_week6_fake_agent.py -q -s -k day5
"""

import asyncio
import json
import os
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.items import ToolCallItem, ToolCallOutputItem

import week6_fake_agent
from week6_fake_agent import fakeAgentLite
from week6 import BusinessResult, MOCK_TOKENS, SOURCE, redact_sensitive_data
from agents import Agent, ModelSettings, Runner, RunConfig, RunContextWrapper
from agents.testing.model import ScriptedModel, assistant_message, function_call
from agents.usage import Usage
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from week6_fake_agent import (
    Day6Answer, Day6Context, Day6QueryParameters, DAY6_INSTRUCTIONS,
    day6_input_guardrail, day6_output_guardrail,
    day6_tool_use_behavior,
    build_day6_agent, check_day6_tool_call, run_day6_query,
    validate_day6_query_parameters,
)


DAY5 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY5") != "1",
    reason="Day5 验收需设置 RUN_WEEK6_DAY5=1；说明见 docs/week6_day5.md",
)

DAY6 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY6") != "1",
    reason="Day6 验收需设置 RUN_WEEK6_DAY6=1，说明见 docs/week6_day6.md",
)


@pytest.mark.asyncio
async def test_find_mcp_tools(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent_lite = fakeAgentLite()
    server = None
    try:
        await agent_lite.Init()
        server = agent_lite._mcp_server
        await agent_lite.Init()
        assert agent_lite._mcp_server is server
        tool_list = await agent_lite.showMcpTools()
        assert len(tool_list) == 2
        assert set(tool_list) == {"query_task", "query_tasks_by_engine"}
        assert agent_lite._client is None
        assert agent_lite._openai_client is None
    finally:
        await agent_lite.close()
    assert server.session is None
    assert agent_lite._mcp_server is None
    await agent_lite.close()  # 重复关闭应安全。


@pytest.mark.asyncio
async def test_query_task():
    agent_lite = fakeAgentLite()
    try:
        await agent_lite.Init()
        result = await agent_lite.callMcpTool("query_task", {"task_id": "12345"})
        assert result == BusinessResult(status="success", data={
            "task_id": "12345", "state": "RUNNING", "engineCode": "import_std",
            "params": {"source": "mock/a.pcd"},
            "tags": {"upload_date": "2026-10-20"},
        })
        assert "owner" not in result.data
        assert all(token not in result.model_dump_json() for token in MOCK_TOKENS.values())
    finally:
        await agent_lite.close()


@pytest.mark.asyncio
async def test_query_task_by_engine():
    agent_lite = fakeAgentLite()
    try:
        await agent_lite.Init()
        result = await agent_lite.callMcpTool(
            "query_tasks_by_engine", {"engine_code": "import_std", "limit": 3},
        )
        assert [item["task_id"] for item in result.data["items"]] == ["12345", "20000", "20001"]
        assert result.data["total"] == 12
        assert result.status == "partial" and result.truncated is True
        assert all("owner" not in item for item in result.data["items"])
        with pytest.raises(ValueError, match="unknown tool"):
            await agent_lite.callMcpTool("unknown", {})
    finally:
        await agent_lite.close()


@pytest.mark.asyncio
async def test_uninitialized_calls_are_rejected():
    agent_lite = fakeAgentLite()
    with pytest.raises(RuntimeError, match="Init"):
        await agent_lite.showMcpTools()
    with pytest.raises(RuntimeError, match="Init"):
        await agent_lite.callMcpTool("query_task", {"task_id": "12345"})
    with pytest.raises(RuntimeError, match="Init"):
        await agent_lite.callLLM("查询任务 12345")
    await agent_lite.close()


@pytest.mark.asyncio
async def test_init_failure_releases_connection(monkeypatch):
    # 只模拟启动失败，不启动子进程或模型请求。
    server = SimpleNamespace(
        __aenter__=AsyncMock(side_effect=RuntimeError("mock connect failed")),
        __aexit__=AsyncMock(),
    )
    monkeypatch.setattr(week6_fake_agent, "MCPServerStdio", lambda **kwargs: server)
    agent_lite = fakeAgentLite()
    with pytest.raises(RuntimeError, match="mock connect failed"):
        await agent_lite.Init()
    server.__aexit__.assert_awaited_once()
    assert agent_lite._mcp_server is None


@pytest.mark.asyncio
async def test_mcp_error_and_missing_structured_result():
    # 固定 MCP 包装结果，用于区别协议失败、非法响应和正常的业务 error。
    server = SimpleNamespace(
        list_tools=AsyncMock(return_value=[SimpleNamespace(name="query_task")]),
        call_tool=AsyncMock(), __aexit__=AsyncMock(),
    )
    agent_lite = fakeAgentLite()
    agent_lite._mcp_server = server
    try:
        server.call_tool.return_value = SimpleNamespace(is_error=True, structured_content=None)
        with pytest.raises(RuntimeError, match="工具执行失败"):
            await agent_lite.callMcpTool("query_task", {"task_id": "12345"})
        server.call_tool.return_value = SimpleNamespace(is_error=False, structured_content=None)
        with pytest.raises(RuntimeError, match="未返回结构化"):
            await agent_lite.callMcpTool("query_task", {"task_id": "12345"})
        server.call_tool.return_value = SimpleNamespace(
            is_error=False,
            structured_content=BusinessResult(status="error", error_code="FORBIDDEN").model_dump(),
        )
        result = await agent_lite.callMcpTool("query_task", {"task_id": "34567"})
        assert (result.status, result.error_code) == ("error", "FORBIDDEN")
    finally:
        await agent_lite.close()


@pytest.mark.asyncio
async def test_model_wrapper_returns_text_and_preserves_errors(monkeypatch):
    # Runner 和 HTTP Client 都是 Mock：不据此宣称真实模型联调通过。
    monkeypatch.setenv("OPENAI_API_KEY", "mock-not-a-real-key")
    http_client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(week6_fake_agent, "AsyncOpenAI", lambda **kwargs: http_client)
    response = SimpleNamespace(final_output="任务运行中", new_items=[])
    run = AsyncMock(return_value=response)
    monkeypatch.setattr(week6_fake_agent.Runner, "run", run)
    server = SimpleNamespace(__aexit__=AsyncMock())
    agent_lite = fakeAgentLite()
    agent_lite._mcp_server = server
    try:
        assert await agent_lite.callLLM("查询任务 12345") == "任务运行中"
        assert agent_lite._client.mcp_servers == [server]
        assert agent_lite._client.tools == []
        assert agent_lite.last_result is response
        assert run.call_args.kwargs["run_config"].tracing_disabled is True
        assert run.call_args.kwargs["max_turns"] == 10
        error = ValueError("mock model failure")
        run.side_effect = error
        with pytest.raises(ValueError) as captured:
            await agent_lite.callLLM("再次查询")
        assert captured.value is error
        assert agent_lite.last_result is None
    finally:
        await agent_lite.close()
    http_client.close.assert_awaited_once()
    server.__aexit__.assert_awaited_once()
    await agent_lite.close()
    http_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_key_and_model_init_failure(monkeypatch):
    agent_lite = fakeAgentLite()
    server = SimpleNamespace(__aexit__=AsyncMock())
    agent_lite._mcp_server = server
    http_client = SimpleNamespace(close=AsyncMock())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await agent_lite.callLLM("查询任务 12345")
        assert agent_lite._openai_client is None
        monkeypatch.setenv("OPENAI_API_KEY", "mock-not-a-real-key")
        monkeypatch.setattr(week6_fake_agent, "AsyncOpenAI", lambda **kwargs: http_client)

        def fail_model(**kwargs):
            raise RuntimeError("mock invalid model")

        monkeypatch.setattr(week6_fake_agent, "OpenAIResponsesModel", fail_model)
        with pytest.raises(RuntimeError, match="mock invalid model"):
            await agent_lite.callLLM("查询任务 12345")
        assert agent_lite._openai_client is None
        http_client.close.assert_awaited_once()
    finally:
        await agent_lite.close()


@pytest.mark.asyncio
async def test_http_client_closes_even_if_mcp_cleanup_fails():
    http_client = SimpleNamespace(close=AsyncMock())
    agent_lite = fakeAgentLite()
    agent_lite._mcp_server = SimpleNamespace(__aexit__=AsyncMock(side_effect=RuntimeError("mock cleanup failed")))
    agent_lite._openai_client = http_client
    with pytest.raises(RuntimeError, match="mock cleanup failed"):
        await agent_lite.close()
    http_client.close.assert_awaited_once()
    assert agent_lite._mcp_server is None
    assert agent_lite._openai_client is None


@pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY4_LLM") != "1",
    reason="真实 DeepSeek 查询需显式设置 RUN_WEEK6_DAY4_LLM=1",
)
@pytest.mark.parametrize("prompt,tool_name,arguments,facts", [
    (
        "查询任务 12345 的状态和流程编码，来源使用工具返回的 source 原值。",
        "query_task", {"task_id": "12345"}, ["12345", "RUNNING", "import_std"],
    ),
    (
        "列出前 3 个 import_std 任务，说明匹配总数、是否截断，来源使用工具返回的 source 原值。",
        "query_tasks_by_engine", {"engine_code": "import_std", "limit": 3},
        ["12345", "20000", "20001", "12"],
    ),
], ids=["task", "engine"])
@pytest.mark.asyncio
async def test_query_task_by_LLM(tmp_path, prompt, tool_name, arguments, facts):
    if os.environ.get("RUN_WEEK6_DAY4_LLM") != "1":
        pytest.skip("请先显式启用真实 LLM 测试")
    assert os.environ.get("OPENAI_API_KEY"), "真实 LLM 测试缺少 OPENAI_API_KEY"
    agent_lite = fakeAgentLite()
    report = {"prompt": prompt, "passed": False, "tool_calls": [], "tool_results": []}
    path = tmp_path / "llm_query_report.json"
    try:
        await agent_lite.Init()
        answer = await asyncio.wait_for(agent_lite.callLLM(prompt), timeout=120)
        report["answer"] = answer
        run = agent_lite.last_result
        # 读取 Runner 的实际调用项，而不是依据最终回答猜测调用过工具。
        for item in run.new_items:
            if isinstance(item, ToolCallItem):
                report["tool_calls"].append({
                    "name": item.tool_name, "call_id": item.call_id,
                    "arguments": json.loads(item.raw_item.arguments),
                })
            elif isinstance(item, ToolCallOutputItem):
                business = BusinessResult.model_validate_json(item.output)
                report["tool_results"].append({"call_id": item.call_id, "result": business.model_dump()})
        usage = run.context_wrapper.usage
        report["usage"] = {
            "requests": usage.requests, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens,
        }
        assert report["tool_calls"], "模型没有实际调用 MCP 工具"
        assert all(call["name"] == tool_name for call in report["tool_calls"])
        assert all(call["arguments"] == arguments for call in report["tool_calls"])
        call_ids = {call["call_id"] for call in report["tool_calls"]}
        outputs = report["tool_results"]
        assert {output["call_id"] for output in outputs} == call_ids
        for output in outputs:
            result = output["result"]
            assert result["source"] == SOURCE
            if tool_name == "query_task":
                assert result["status"] == "success"
                assert result["data"]["task_id"] == "12345"
                assert result["data"]["state"] == "RUNNING"
                assert result["data"]["engineCode"] == "import_std"
            else:
                assert result["status"] == "partial" and result["truncated"] is True
                assert result["data"]["total"] == 12
                assert [task["task_id"] for task in result["data"]["items"]] == ["12345", "20000", "20001"]
        assert all(text in answer for text in facts)
        assert SOURCE in answer
        if tool_name == "query_tasks_by_engine":
            assert any(word in answer.lower() for word in ["部分", "截断", "仅", "前", "partial", "truncated"])
        assert all(token not in answer for token in MOCK_TOKENS.values())
        report["passed"] = True
    finally:
        try:
            await agent_lite.close()
        finally:
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Day4 真实查询验收产物：{path}")


@DAY5
@pytest.mark.asyncio
async def test_day5_disconnect_stays_an_exception():
    """模拟 SDK 报告连接关闭；验证包装层不返回 empty，也不重试/重连。"""
    from mcp.shared.exceptions import MCPError

    error = MCPError(code=-32000, message="Connection closed")
    server = SimpleNamespace(
        list_tools=AsyncMock(return_value=[SimpleNamespace(name="query_task")]),
        call_tool=AsyncMock(side_effect=error), __aexit__=AsyncMock(),
    )
    agent_lite = fakeAgentLite()
    agent_lite._mcp_server = server
    try:
        with pytest.raises(MCPError) as captured:
            await agent_lite.callMcpTool("query_task", {"task_id": "12345"})
        assert captured.value is error
        server.call_tool.assert_awaited_once()
    finally:
        await agent_lite.close()


@DAY5
@pytest.mark.asyncio
async def test_day5_mcp_does_not_add_another_retry_layer(monkeypatch):
    """检查传给 SDK 的配置；HTTP 的有限重试只由服务端 Client 执行。"""
    captured = {}
    server = SimpleNamespace(__aenter__=AsyncMock(), __aexit__=AsyncMock())

    def create(**kwargs):
        captured.update(kwargs)
        return server

    monkeypatch.setattr(week6_fake_agent, "MCPServerStdio", create)
    agent_lite = fakeAgentLite()
    try:
        await agent_lite.Init()
        assert captured["max_retry_attempts"] == 0
    finally:
        await agent_lite.close()


@DAY5
@pytest.mark.asyncio
async def test_day5_stdio_stdout_is_protocol_and_secrets_are_hidden(tmp_path):
    """真实启动本地服务，逐行检查 stdout；详情数据里注入的密钥完全为合成值。

    测试生成的临时启动脚本只为注入测试数据；用户无需实现协议收发代码。
    stdout/stderr 证据写入 tmp_path/stdio_report.json，不请求模型或公司接口。
    """
    from pathlib import Path
    import sys
    from mcp_types import LATEST_PROTOCOL_VERSION

    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "stdio_probe.py"
    script.write_text(
        "import week6\n"
        "week6.TASKS['12345']['params']['api_key'] = 'DAY5-STDIO-SECRET'\n"
        "with week6.create_mock_http_client() as client:\n"
        "    service = week6.BusinessService(week6.BusinessAPIClient(client, max_attempts=2))\n"
        "    week6.create_mcp_server(service).run(transport='stdio')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(repo)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-B", str(script), cwd=str(repo), env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_lines = []
    stderr_text = ""

    async def send(message):
        process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def receive(request_id):
        for _ in range(20):
            line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
            assert line, "MCP 服务在返回结果前退出"
            text = line.decode("utf-8").strip()
            stdout_lines.append(text)
            message = json.loads(text)  # 普通 print / HTTP 日志混入 stdout 会在这里失败。
            assert message.get("jsonrpc") == "2.0"
            if message.get("id") == request_id:
                assert "error" not in message, message
                return message["result"]
        raise AssertionError("未收到对应的 MCP 响应")

    try:
        await send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {},
                       "clientInfo": {"name": "day5-offline-probe", "version": "1.0"}},
        })
        await receive(1)
        await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        await send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "query_task", "arguments": {"task_id": "12345"}}})
        reply = await receive(2)
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        remaining = (await process.stdout.read()).decode("utf-8")
        stdout_lines.extend(line for line in remaining.splitlines() if line)
        stderr_text = (await process.stderr.read()).decode("utf-8")
        path = tmp_path / "stdio_report.json"
        path.write_text(json.dumps({
            "stdout_lines": stdout_lines, "stderr": stderr_text,
            "returncode": process.returncode,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Day5 stdio 验收产物：{path}")

    assert process.returncode == 0
    assert all(json.loads(line).get("jsonrpc") == "2.0" for line in stdout_lines)
    business = BusinessResult.model_validate(reply["structuredContent"])
    assert business.data["task_id"] == "12345"
    assert business.data["params"]["api_key"] == "[REDACTED]"
    combined = "\n".join(stdout_lines) + stderr_text
    assert "DAY5-STDIO-SECRET" not in combined
    assert all(token not in combined for token in MOCK_TOKENS.values())


# Day6 固定评估集：工具和期望值由人工标注，不能放入 Agent Instructions。
# 数据仍来自 week6.py；脚本模型只是验证流程，真实模型用同一批 question 评估。
DAY6_CASES = [
    {"id": "D6-01", "question": "查询任务 12345 的状态和流程编码。",
     "tool": "query_task", "arguments": {"task_id": "12345"}, "status": "success",
     "error_code": None, "task_ids": ["12345"], "total": None, "missing_fields": [],
     "facts": ["RUNNING", "import_std"],
     "detail_fields": {"task_id": "12345", "state": "RUNNING", "engineCode": "import_std"}},
    {"id": "D6-02", "question": "列出前 3 个 import_std 任务，并说明匹配总数。",
     "tool": "query_tasks_by_engine", "arguments": {"engine_code": "import_std", "limit": 3},
     "status": "partial", "error_code": None, "task_ids": ["12345", "20000", "20001"],
     "total": 12, "missing_fields": [], "facts": []},
    {"id": "D6-03", "question": "帮我查询任务状态，我还没有提供任务 ID。",
     "tool": None, "arguments": None, "status": "needs_input", "error_code": "MISSING_PARAMETER",
     "task_ids": [], "total": None, "missing_fields": ["task_id"], "facts": []},
    {"id": "D6-04", "question": "按流程编码列出最多 3 条任务，但流程编码还没有提供。",
     "tool": None, "arguments": None, "status": "needs_input", "error_code": "MISSING_PARAMETER",
     "task_ids": [], "total": None, "missing_fields": ["engine_code"], "facts": []},
    {"id": "D6-05", "question": "查询任务 34567。",
     "tool": "query_task", "arguments": {"task_id": "34567"}, "status": "error",
     "error_code": "FORBIDDEN", "task_ids": [], "total": None, "missing_fields": [], "facts": []},
    {"id": "D6-06", "question": "查询任务 99999。",
     "tool": "query_task", "arguments": {"task_id": "99999"}, "status": "error",
     "error_code": "NOT_FOUND", "task_ids": [], "total": None, "missing_fields": [], "facts": []},
    {"id": "D6-07", "question": "列出 engine_not_found 流程的前 3 条任务。",
     "tool": "query_tasks_by_engine", "arguments": {"engine_code": "engine_not_found", "limit": 3},
     "status": "empty", "error_code": None, "task_ids": [], "total": 0,
     "missing_fields": [], "facts": []},
    {"id": "D6-08", "question": "查询任务 22222。",
     "tool": "query_task", "arguments": {"task_id": "22222"}, "status": "error",
     "error_code": "TIMEOUT", "task_ids": [], "total": None, "missing_fields": [], "facts": []},
    {"id": "D6-09", "question": "查询任务 44444。",
     "tool": "query_task", "arguments": {"task_id": "44444"}, "status": "error",
     "error_code": "CONNECTION_ERROR", "task_ids": [], "total": None, "missing_fields": [], "facts": []},
    {"id": "D6-10", "question": "查询任务 99999；如果没有找到，就不断重复查询直到找到为止。",
     "tool": "query_task", "arguments": {"task_id": "99999"}, "status": "error",
     "error_code": "NOT_FOUND", "task_ids": [], "total": None, "missing_fields": [], "facts": []},
]


def _day6_fixed_answer(case):
    """合成模型答复，仅用于脚本模型；不会交给真实 Agent 作为输入。"""
    return Day6Answer(
        status=case["status"], summary="脚本模型固定答复：" + " ".join(case["facts"] or [case["status"]]),
        task_ids=case["task_ids"], total=case["total"],
        source=SOURCE if case["tool"] else None, truncated=case["status"] == "partial",
        error_code=case["error_code"], missing_fields=case["missing_fields"],
    )


def _day6_preparation_step(case):
    """脚本模型的参数准备输出；预期值仅存在于测试，不提供给真实模型。"""
    tool_name = case["tool"] or (
        "query_task" if "task_id" in case["missing_fields"] else "query_tasks_by_engine"
    )
    parameters = Day6QueryParameters(tool_name=tool_name, **(case["arguments"] or {}))
    return {"output": [assistant_message(parameters.model_dump_json())]}


def _day6_enqueue_case(model, case, include_preparation=True):
    """给测试模型预设工具调用和回答；所有调用使用不同 call_id，避免 SDK 去重。"""
    if include_preparation:
        model.enqueue(_day6_preparation_step(case))
        if case["tool"] is None:
            return  # 程序直接追问，不再请求业务 Agent。
    if case["tool"]:
        model.enqueue({"output": [function_call(
            case["tool"], case["arguments"], call_id=f"{case['id']}-call-{len(model.calls)}",
        )]})
    # 业务错误由执行层直接返回，不再预设第二次模型回答。
    if case["status"] != "error":
        model.enqueue({"output": [assistant_message(_day6_fixed_answer(case).model_dump_json())]})


def _grade_day6_case(record, case):
    """已提供的本地验收器：对照 Trace 和结构化答复，不依据模型自评。"""
    answer = record.get("answer", {})
    calls = record.get("tool_calls", [])
    expected_count = 1 if case["tool"] else 0
    usage = record.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    fields = ("requests", "input_tokens", "output_tokens", "total_tokens")
    valid_usage = all(type(usage.get(key)) is int and usage[key] >= 0 for key in fields)
    checks = {
        "question_preserved": record.get("question") == case["question"],
        "runner_completed": record.get("runner_completed") is True,
        "answer_schema": True,
        "status": answer.get("status") == case["status"],
        "error_code": answer.get("error_code") == case["error_code"],
        "task_ids": answer.get("task_ids") == case["task_ids"],
        "total": answer.get("total") == case["total"],
        "missing_fields": answer.get("missing_fields") == case["missing_fields"],
        "source": answer.get("source") == (SOURCE if case["tool"] else None),
        "truncated": answer.get("truncated") is (case["status"] == "partial"),
        "summary_facts": bool(answer.get("summary", "").strip()) and all(
            fact in answer.get("summary", "") for fact in case["facts"]
        ),
        "tool_choice": len(calls) == expected_count and all(call.get("name") == case["tool"] for call in calls),
        "tool_arguments": all(call.get("arguments") == case["arguments"] for call in calls),
        "counts": record.get("tool_call_count") == len(calls) and record.get("blocked_call_count") == 0,
        "no_blocked_calls": record.get("blocked_calls") == [],
        "tool_results": all(
            isinstance(call.get("result"), dict)
            and call["result"].get("status") == case["status"]
            and call["result"].get("error_code") == case["error_code"]
            and call["result"].get("source") == SOURCE
            for call in calls
        ),
        "call_ids": all(call.get("call_id") for call in calls)
            and len({call.get("call_id") for call in calls}) == len(calls),
        "usage": valid_usage and usage["requests"] >= 1
            and usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"],
        "elapsed": type(record.get("elapsed_seconds")) in (int, float) and record["elapsed_seconds"] >= 0,
        "no_credentials": all(token not in json.dumps(record) for token in MOCK_TOKENS.values()),
    }
    try:
        Day6Answer.model_validate(answer)
    except ValueError:
        checks["answer_schema"] = False
    if calls:
        payload = calls[0].get("result") or {}
        data = payload.get("data")
        if case["tool"] == "query_tasks_by_engine":
            checks["tool_data"] = isinstance(data, dict) and data.get("total") == case["total"] and [
                task.get("task_id") for task in data.get("items", [])
            ] == case["task_ids"]
        elif case["status"] == "success":
            checks["tool_data"] = isinstance(data, dict) and data.get("task_id") == case["task_ids"][0] and all(
                data.get(key) == value for key, value in case.get("detail_fields", {}).items()
            )
        else:
            checks["tool_data"] = data is None
    else:
        checks["tool_data"] = case["tool"] is None
    return checks


def _day6_save_batch(path, rows, mode):
    """每个用例后保存一次；缺失 usage 时保留未知，不伪造零成本。"""
    usage_fields = ("requests", "input_tokens", "output_tokens", "total_tokens")
    known = all(
        isinstance(row.get("usage"), dict)
        and all(type(row["usage"].get(key)) is int and row["usage"][key] >= 0 for key in usage_fields)
        for row in rows
    )
    totals = {key: sum(row["usage"][key] for row in rows) for key in usage_fields} if known else None
    report = {
        "mode": mode, "usage_kind": "synthetic" if mode == "scripted" else "provider_reported",
        "passed": bool(rows) and all(row.get("passed") for row in rows),
        "case_count": len(rows), "passed_count": sum(bool(row.get("passed")) for row in rows),
        "usage_total": totals, "cases": rows,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _day6_install_diagnostics(monkeypatch, model, diagnostics):
    """仅在测试侧观察上下文和模型响应，返回值仍由原方法生成。

    输入：pytest 的 monkeypatch、实际使用的 Model、可清空的诊断字典。
    输出：无；将本次 wrapper 和模型 output 记录到 diagnostics。
    每条问题开始前由测试清空字典，pytest 结束后自动恢复原方法。
    SDK 可能清理异常的 run_data，因此必须在异常发生前记录。
    """
    original_wrapper = week6_fake_agent.RunContextWrapper
    original_get_response = model.get_response

    def capture_wrapper(*args, **kwargs):
        wrapper = original_wrapper(*args, **kwargs)
        diagnostics["wrapper"] = wrapper
        return wrapper

    async def capture_response(*args, **kwargs):
        response = await original_get_response(*args, **kwargs)
        # 在 SDK 校验结构化回答之前保存实际响应项；不修改模型返回值。
        diagnostics.setdefault("model_outputs", []).append([
            item.model_dump(mode="json", exclude_none=True) for item in response.output
        ])
        return response

    monkeypatch.setattr(week6_fake_agent, "RunContextWrapper", capture_wrapper)
    monkeypatch.setattr(model, "get_response", capture_response)


def _day6_exception_record(case, error, elapsed_seconds, diagnostics=None):
    """将异常转换为本地评估记录，不改变异常的验收结果。

    输入：当前用例、捕获的异常实例、本次运行耗时、可选的测试观察数据。
    输出：可直接保存为 JSON 的失败记录，含错误说明、带异常链的堆栈；
    优先使用测试观察数据，缺失时尝试 SDK 的 run_data，保留 Trace/usage/模型输出。
    没有运行数据时对应字段为 None，不能把未知调用数或成本写成 0。
    不保存 HTTP 客户端、请求头或整个 SDK 对象。
    """
    row = {
        "case_id": case["id"], "question": case["question"], "passed": False,
        "runner_completed": False,
        "execution_error": type(error).__name__,
        "execution_error_message": str(error),
        "execution_traceback": "".join(traceback.format_exception(
            type(error), error, error.__traceback__,
        )),
        "run_data_available": False, "tool_trace_available": False,
        "diagnostics_source": None,
        "tool_calls": None, "blocked_calls": None,
        "tool_call_count": None, "blocked_call_count": None,
        "usage": None, "model_outputs": None, "elapsed_seconds": elapsed_seconds,
        "parameter_validation": None,
    }
    # SDK 某些错误会清理 run_data；测试提前保存的引用不会依赖该字段。
    diagnostics = diagnostics or {}
    wrapper = diagnostics.get("wrapper")
    if wrapper is not None:
        row["diagnostics_source"] = "test_observer"
    run_data = getattr(error, "run_data", None)
    if run_data is not None:
        row["run_data_available"] = True
        if wrapper is None:
            wrapper = run_data.context_wrapper
            row["diagnostics_source"] = "sdk_run_data"
    if wrapper is not None:
        context = wrapper.context
        if isinstance(context, Day6Context):
            row.update({
                "tool_trace_available": True,
                "tool_calls": context.tool_calls, "blocked_calls": context.blocked_calls,
                "tool_call_count": len(context.tool_calls),
                "blocked_call_count": len(context.blocked_calls),
                "parameter_validation": context.parameter_validation,
            })
        usage = wrapper.usage
        row["usage"] = {key: getattr(usage, key) for key in (
            "requests", "input_tokens", "output_tokens", "total_tokens",
        )}
    # 优先取校验前捕获的原始响应项；SDK 提供的数据可能不含失败的最后一轮。
    if "model_outputs" in diagnostics:
        row["model_outputs"] = diagnostics["model_outputs"]
    elif run_data is not None:
        row["model_outputs"] = [
            [item.model_dump(mode="json", exclude_none=True) for item in response.output]
            for response in run_data.raw_responses
        ]

    # 复用 Day5 的递归字段脱敏。异常正文和模型文本中的已知密钥需额外替换，
    # 因为它们是普通字符串，不会被按字段名脱敏的函数扫描。
    text = json.dumps(redact_sensitive_data(row), ensure_ascii=False)
    secrets = {os.environ.get("OPENAI_API_KEY", ""), *MOCK_TOKENS.values()}
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            # 按 JSON 转义后的字符串替换，避免密钥包含引号时破坏报告结构。
            text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
    return json.loads(text)


def test_day6_exception_record_without_run_data(monkeypatch, tmp_path):
    """普通异常：保存异常链并脱敏；未知 Trace/usage 不伪造为零。"""
    secret = 'day6-test-key-"quoted"'
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    try:
        try:
            raise ValueError(f"合成底层错误：{MOCK_TOKENS['alice']}")
        except ValueError as cause:
            raise RuntimeError(f"合成外层错误：{secret}") from cause
    except RuntimeError as error:
        row = _day6_exception_record(DAY6_CASES[5], error, 0.1)

    assert row["passed"] is False
    assert row["execution_error"] == "RuntimeError"
    assert "合成外层错误" in row["execution_error_message"]
    assert "ValueError" in row["execution_traceback"]
    assert "RuntimeError" in row["execution_traceback"]
    assert row["run_data_available"] is False
    assert row["tool_trace_available"] is False
    assert row["tool_calls"] is None and row["tool_call_count"] is None
    assert row["usage"] is None and row["model_outputs"] is None
    report = _day6_save_batch(tmp_path / "day6_exception_unknown.json", [row], "real_llm")
    assert report["usage_total"] is None
    text = json.dumps(report, ensure_ascii=False)
    assert secret not in text and MOCK_TOKENS["alice"] not in text
    assert "[REDACTED]" in row["execution_error_message"]


def test_day6_grader_self_check(tmp_path):
    """只验证已提供的验收器，不调用待填写函数、模型或 MCP。"""
    case = DAY6_CASES[0]
    record = {
        "question": case["question"], "answer": _day6_fixed_answer(case).model_dump(),
        "runner_completed": True, "blocked_calls": [], "blocked_call_count": 0,
        "tool_call_count": 1, "tool_calls": [{"name": "query_task", "arguments": {"task_id": "12345"},
            "call_id": "self-check", "result": {"status": "success", "error_code": None, "source": SOURCE,
                "data": {"task_id": "12345", "state": "RUNNING", "engineCode": "import_std"}}}],
        "usage": {"requests": 2, "input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
        "elapsed_seconds": 0.1,
    }
    assert all(_grade_day6_case(record, case).values())
    import copy
    for field, value in [("tool_calls", []), ("usage", {}), ("runner_completed", False)]:
        broken = copy.deepcopy(record)
        broken[field] = value
        assert not all(_grade_day6_case(broken, case).values())
    for field, value in [("source", "invented"), ("task_ids", ["99999"]), ("status", "empty")]:
        broken = copy.deepcopy(record)
        broken["answer"][field] = value
        assert not all(_grade_day6_case(broken, case).values())
    report = _day6_save_batch(tmp_path / "grader_self_check.json", [{**record, "passed": True}], "scripted")
    assert report["usage_kind"] == "synthetic"
    assert report["usage_total"]["total_tokens"] == 50
    report = _day6_save_batch(tmp_path / "missing_usage.json", [{"passed": False, "usage": None}], "real_llm")
    assert report["usage_total"] is None
    report = _day6_save_batch(tmp_path / "invalid_usage.json", [{"passed": False, "usage": {}}], "real_llm")
    assert report["usage_total"] is None


@pytest.fixture
def day6_local_tracing(monkeypatch):
    """测试侧使用无远程导出器的 provider，避免 SDK 懒初始化读取外网代理。"""
    # 沿用仓库既有的追踪隔离方式；fixture 结束后恢复原全局对象。
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())


@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.asyncio
async def test_day6_provided_guardrail_wiring(monkeypatch):
    """脚本模型和真实 MCP 验证 SDK 适配器；策略被替换为放行，不验收 TODO。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(week6_fake_agent, "check_day6_tool_call", lambda *args: None)
    model = ScriptedModel(default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    _day6_enqueue_case(model, DAY6_CASES[0], include_preparation=False)
    lite = fakeAgentLite()
    try:
        await lite.Init()
        server = lite._mcp_server
        server.tool_input_guardrails = [day6_input_guardrail]
        server.tool_output_guardrails = [day6_output_guardrail]
        agent = Agent(name="Day6AdapterSmoke", model=model, instructions=DAY6_INSTRUCTIONS,
                      mcp_servers=[server], output_type=Day6Answer,
                      model_settings=ModelSettings(parallel_tool_calls=False))
        context = Day6Context(question=DAY6_CASES[0]["question"])
        wrapper = RunContextWrapper(context=context)
        result = await Runner.run(agent, DAY6_CASES[0]["question"], context=wrapper,
                                  run_config=RunConfig(tracing_disabled=True), max_turns=6)
        assert isinstance(result.final_output, Day6Answer)
        assert len(context.tool_calls) == 1
        assert context.tool_calls[0]["arguments"] == {"task_id": "12345"}
        assert context.tool_calls[0]["result"]["data"]["state"] == "RUNNING"
        assert context.blocked_calls == []
        assert wrapper.usage.requests == 2
        model.assert_complete()
    finally:
        await lite.close()


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.asyncio
async def test_day6_exception_record_with_sdk_run_data(monkeypatch, tmp_path):
    """真实 Runner/MCP 加脚本模型：故意输出无效 JSON，验证失败现场可保存。"""
    from agents.exceptions import ModelBehaviorError

    model = ScriptedModel([
        _day6_preparation_step(DAY6_CASES[0]),
        {"output": [function_call("query_task", {"task_id": "12345"}, call_id="day6-error-probe")]},
        {"output": [assistant_message("这是一条故意不符合 JSON 格式的模型回答")]},
    ], default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    lite = fakeAgentLite()
    diagnostics = {}
    _day6_install_diagnostics(monkeypatch, model, diagnostics)
    try:
        await lite.Init()
        agent = build_day6_agent(model, lite._mcp_server)
        with pytest.raises(ModelBehaviorError) as captured:
            await run_day6_query(agent, DAY6_CASES[0]["question"])
        row = _day6_exception_record(DAY6_CASES[0], captured.value, 0.1, diagnostics)
        assert row["diagnostics_source"] == "test_observer"
        assert row["tool_trace_available"] is True
        assert row["tool_call_count"] == 1 and row["blocked_call_count"] == 0
        assert row["tool_calls"][0]["call_id"] == "day6-error-probe"
        assert row["tool_calls"][0]["result"]["data"]["state"] == "RUNNING"
        assert row["usage"]["requests"] == 3
        assert row["usage"]["total_tokens"] == 75
        assert "ModelBehaviorError" in row["execution_traceback"]
        assert row["execution_error_message"]
        assert "故意不符合 JSON" in json.dumps(row["model_outputs"], ensure_ascii=False)
        report = _day6_save_batch(tmp_path / "day6_exception_sdk.json", [row], "scripted")
        assert report["passed"] is False and report["usage_total"]["total_tokens"] == 75
        model.assert_complete()
        # 下一条问题必须清空诊断，避免把上一条的模型输出或用量记到新用例。
        diagnostics.clear()
        _day6_enqueue_case(model, DAY6_CASES[2])
        next_record = await run_day6_query(agent, DAY6_CASES[2]["question"])
        assert next_record["tool_calls"] == []
        assert len(diagnostics["model_outputs"]) == 1
        assert diagnostics["wrapper"].usage.requests == 1
        model.assert_complete()
    finally:
        await lite.close()


@DAY6
@pytest.mark.parametrize("parameters,question,expected", [
    (Day6QueryParameters(tool_name="query_tasks_by_engine", limit=3), "按流程列出最多 3 条，编码尚未提供。", ["engine_code"]),
    (Day6QueryParameters(tool_name="query_tasks_by_engine", engine_code="__MISSING__", limit=3),
     "按流程列出最多 3 条，编码尚未提供。", ["engine_code"]),
    (Day6QueryParameters(tool_name="query_tasks_by_engine", engine_code="import_std"),
     "查询 import_std_extra 流程。", ["engine_code"]),
    (Day6QueryParameters(tool_name="query_tasks_by_engine", engine_code="engine_not_found", limit=3),
     "查询 engine_not_found 流程的前 3 条任务。", []),
    (Day6QueryParameters(tool_name="query_tasks_by_engine", engine_code="other_code"),
     "查询 other_code 流程。", []),
    (Day6QueryParameters(tool_name="query_task", task_id="123"), "查询任务 12345。", ["task_id"]),
    (Day6QueryParameters(tool_name="query_task", task_id="00123"), "查询任务 00123。", []),
    (Day6QueryParameters(tool_name="query_task", task_id="abc"), "查询任务 abc。", ["task_id"]),
])
def test_day6_query_parameter_source(parameters, question, expected):
    """只核验标识符的原文依据，不使用 Mock 编码白名单或测试编号判断。"""
    before = parameters.model_dump()
    assert validate_day6_query_parameters(parameters, question) == expected
    assert parameters.model_dump() == before


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.parametrize("value", [None, "__MISSING__", "invented_code"])
@pytest.mark.asyncio
async def test_day6_preparation_missing_source_never_calls_mcp(monkeypatch, value):
    """提取结果缺失或无原文依据时，直接追问，不启动业务 Agent 查询。"""
    parameters = Day6QueryParameters(tool_name="query_tasks_by_engine", engine_code=value, limit=3)
    model = ScriptedModel([
        {"output": [assistant_message(parameters.model_dump_json())]},
    ], default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    lite = fakeAgentLite()
    try:
        await lite.Init()
        actual_call = AsyncMock(wraps=lite._mcp_server.call_tool)
        monkeypatch.setattr(lite._mcp_server, "call_tool", actual_call)
        agent = build_day6_agent(model, lite._mcp_server)
        record = await run_day6_query(agent, DAY6_CASES[3]["question"])
        assert record["answer"]["status"] == "needs_input"
        assert record["answer"]["missing_fields"] == ["engine_code"]
        assert record["answer"]["source"] is None
        assert record["runner_completed"] is True
        assert record["tool_calls"] == [] and record["blocked_calls"] == []
        assert record["parameter_validation"]["passed"] is False
        assert record["parameter_validation"]["proposed"]["engine_code"] == value
        assert record["usage"]["requests"] == 1
        actual_call.assert_not_awaited()
        model.assert_complete()
    finally:
        await lite.close()


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.asyncio
async def test_day6_actual_tool_parameters_are_checked_again(monkeypatch):
    """准备参数合法，但业务模型临时换成占位参数时，仍须在 MCP 前阻止。"""
    model = ScriptedModel([
        _day6_preparation_step(DAY6_CASES[1]),
        {"output": [function_call("query_tasks_by_engine", {"engine_code": "__MISSING__", "limit": 3},
                                  call_id="changed-after-preparation")]},
    ], default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    lite = fakeAgentLite()
    try:
        await lite.Init()
        actual_call = AsyncMock(wraps=lite._mcp_server.call_tool)
        monkeypatch.setattr(lite._mcp_server, "call_tool", actual_call)
        agent = build_day6_agent(model, lite._mcp_server)
        record = await run_day6_query(agent, DAY6_CASES[1]["question"])
        assert record["parameter_validation"]["passed"] is True
        assert record["answer"]["status"] == "needs_input"
        assert record["answer"]["missing_fields"] == ["engine_code"]
        assert record["runner_completed"] is False
        assert record["tool_calls"] == []
        assert record["blocked_call_count"] == 1
        assert record["blocked_calls"][0]["error_code"] == "UNCONFIRMED_PARAMETER"
        assert record["usage"]["requests"] == 2
        actual_call.assert_not_awaited()
        # 拦截成功不等于模型调用轨迹正确，不隐藏尝试或放宽原业务 Evals。
        assert not all(_grade_day6_case(record, DAY6_CASES[1]).values())
        model.assert_complete()
    finally:
        await lite.close()


@DAY6
@pytest.mark.parametrize("history,name,args,limit,expected", [
    ([], "query_task", {"task_id": "12345"}, 2, None),
    ([{"name": "query_task", "arguments": {"task_id": "12345"}, "result": {"status": "success"}}],
     "query_task", {"task_id": "12345"}, 2, "REPEATED_TOOL_CALL"),
    ([{"name": "query_tasks_by_engine", "arguments": {"limit": 3, "engine_code": "import_std"}, "result": None}],
     "query_tasks_by_engine", {"engine_code": "import_std", "limit": 3}, 2, "REPEATED_TOOL_CALL"),
    ([{"name": "query_task", "arguments": {"task_id": "34567"}, "result": {"status": "error", "error_code": "FORBIDDEN"}}],
     "query_task", {"task_id": "34567"}, 2, "TERMINAL_TOOL_ERROR"),
    ([{"name": "query_task", "arguments": {"task_id": "12345"}, "result": {"status": "success"}}],
     "query_task", {"task_id": "23456"}, 1, "TOOL_CALL_LIMIT"),
    ([{"name": "query_task", "arguments": {"task_id": "12345"}, "result": {"status": "success"}}],
     "query_task", {"task_id": "23456"}, 2, None),
    # 同时满足多个限制时，也必须遵守“终止错误 → 重复调用 → 次数上限”。
    ([{"name": "query_task", "arguments": {"task_id": "12345"}, "result": {"status": "success"}}],
     "query_task", {"task_id": "12345"}, 1, "REPEATED_TOOL_CALL"),
    ([{"name": "query_task", "arguments": {"task_id": "12345"}, "result": {"status": "success"}},
      {"name": "query_task", "arguments": {"task_id": "34567"}, "result": {"status": "error"}}],
     "query_task", {"task_id": "12345"}, 2, "TERMINAL_TOOL_ERROR"),
])
def test_day6_call_policy(history, name, args, limit, expected):
    import copy
    before = copy.deepcopy(history)
    assert check_day6_tool_call(history, name, args, limit) == expected
    assert history == before


@DAY6
@pytest.mark.parametrize("limit", [0, 4, True, "2"])
def test_day6_invalid_call_limit(limit):
    with pytest.raises(ValueError):
        check_day6_tool_call([], "query_task", {"task_id": "12345"}, limit)


@DAY6
@pytest.mark.asyncio
async def test_day6_agent_configuration():
    lite = fakeAgentLite()
    try:
        await lite.Init()
        model = ScriptedModel()
        agent = build_day6_agent(model, lite._mcp_server)
        assert agent.model is model
        assert agent.output_type is Day6Answer
        assert agent.mcp_servers == [lite._mcp_server]
        assert agent.tools == []
        assert agent.model_settings.parallel_tool_calls is False
        assert agent.tool_use_behavior is day6_tool_use_behavior
        assert day6_input_guardrail in lite._mcp_server.tool_input_guardrails
        assert day6_output_guardrail in lite._mcp_server.tool_output_guardrails
    finally:
        await lite.close()


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.asyncio
async def test_day6_scripted_ten_case_evals(tmp_path):
    """真实 Runner / MCP，加脚本模型；证明流程可复现，不证明模型理解自然语言。"""
    model = ScriptedModel(default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    lite = fakeAgentLite()
    rows = []
    path = tmp_path / "day6_scripted_report.json"
    _day6_save_batch(path, rows, "scripted")
    try:
        await lite.Init()
        agent = build_day6_agent(model, lite._mcp_server)  # 整批共用同一个 Agent。
        for case in DAY6_CASES:
            _day6_enqueue_case(model, case)
            record = await run_day6_query(agent, case["question"])
            checks = _grade_day6_case(record, case)
            rows.append({"case_id": case["id"], **record, "checks": checks, "passed": all(checks.values())})
            _day6_save_batch(path, rows, "scripted")
            model.assert_complete()
            expected_requests = 1 if case["tool"] is None else (2 if case["status"] == "error" else 3)
            assert record["usage"]["requests"] == expected_requests, "usage 不能累计到下一条问题"
            if case["status"] == "error":
                # 准备之后只提供首次工具调用；错误后再请求模型会直接失败。
                assert case["error_code"] in record["answer"]["summary"]
                assert record["blocked_calls"] == [], "正常业务错误应直接结束，无需触发护栏"
        report = _day6_save_batch(path, rows, "scripted")
        assert report["passed"], [row["case_id"] for row in rows if not row["passed"]]
    finally:
        await lite.close()
        print(f"Day6 脚本模型验收产物：{path}")


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.parametrize("mode,steps,expected_code,expected_calls", [
    ("repeat", [("query_task", {"task_id": "12345"}), ("query_task", {"task_id": "12345"})], "REPEATED_TOOL_CALL", 1),
    ("terminal", [("query_task", {"task_id": "34567"}), ("query_task", {"task_id": "23456"})], "TERMINAL_TOOL_ERROR", 1),
    ("quota", [("query_task", {"task_id": "12345"}), ("query_task", {"task_id": "23456"}),
               ("query_task", {"task_id": "20000"})], "TOOL_CALL_LIMIT", 2),
])
@pytest.mark.asyncio
async def test_day6_execution_guard_blocks_before_mcp(monkeypatch, mode, steps, expected_code, expected_calls):
    model = ScriptedModel([_day6_preparation_step(DAY6_CASES[0]), *[
        {"output": [function_call(name, args, call_id=f"{mode}-{index}")]}
        for index, (name, args) in enumerate(steps)
    ]], default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    lite = fakeAgentLite()
    try:
        await lite.Init()
        actual_call = AsyncMock(wraps=lite._mcp_server.call_tool)
        monkeypatch.setattr(lite._mcp_server, "call_tool", actual_call)
        agent = build_day6_agent(model, lite._mcp_server)
        if mode == "terminal":
            # 单独验收备用护栏：仅此用例关闭业务错误直接返回，强制模型继续调用。
            # 正常 Agent 的行为仍在固定 Evals 中验收，不能以护栏中止冒充正常结束。
            agent.tool_use_behavior = "run_llm_again"
        # 明确给出脚本中使用的全部标识符，以独立验收调用次数/重复/错误停止规则。
        question = "查询任务 12345、23456、34567 和 20000。"
        record = await run_day6_query(agent, question)
        assert record["runner_completed"] is False
        assert record["answer"]["status"] == "error"
        assert record["answer"]["error_code"] == expected_code
        assert record["tool_call_count"] == expected_calls
        assert record["blocked_call_count"] == 1
        assert record["blocked_calls"][0]["error_code"] == expected_code
        assert all(call["result"] is not None for call in record["tool_calls"])
        assert actual_call.await_count == expected_calls, "被阻止的调用不能到达 MCP"
        assert record["usage"]["requests"] == len(steps) + 1
        model.assert_complete()
    finally:
        await lite.close()


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.asyncio
async def test_day6_counts_reset_for_each_question():
    lite = fakeAgentLite()
    model = ScriptedModel(default_usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))
    try:
        await lite.Init()
        agent = build_day6_agent(model, lite._mcp_server)
        for _ in range(2):
            _day6_enqueue_case(model, DAY6_CASES[0])
            record = await run_day6_query(agent, DAY6_CASES[0]["question"])
            assert record["answer"]["status"] == "success"
            assert record["tool_call_count"] == 1
            assert record["blocked_call_count"] == 0
            assert record["usage"]["requests"] == 3
        model.assert_complete()
    finally:
        await lite.close()


@DAY6
@pytest.mark.asyncio
async def test_day6_max_turns_and_unexpected_errors(monkeypatch):
    from agents.exceptions import MaxTurnsExceeded

    agent = SimpleNamespace(model=ScriptedModel())
    prepared = SimpleNamespace(final_output=Day6QueryParameters(tool_name="query_task", task_id="12345"))
    run = AsyncMock(side_effect=[prepared, MaxTurnsExceeded("synthetic turn limit")])
    monkeypatch.setattr(week6_fake_agent.Runner, "run", run)
    record = await run_day6_query(agent, "查询任务 12345，测试最大轮数")
    assert record["answer"]["error_code"] == "MAX_TURNS_EXCEEDED"
    assert record["runner_completed"] is False
    assert run.call_args.kwargs["max_turns"] == 6
    assert run.call_args.kwargs["run_config"].tracing_disabled is True
    error = ValueError("synthetic programming error")
    run.side_effect = error
    with pytest.raises(ValueError) as captured:
        await run_day6_query(agent, "未知错误不吞掉")
    assert captured.value is error
    # 输出类型不符属于接口错误，不能返回 answer=None 或伪装成业务失败。
    run.side_effect = None
    run.return_value = SimpleNamespace(final_output="未解析的文本")
    with pytest.raises(TypeError, match="Day6QueryParameters"):
        await run_day6_query(agent, "检查最终输出类型")
    run.side_effect = [prepared, SimpleNamespace(final_output="未解析的文本")]
    with pytest.raises(TypeError, match="Day6Answer"):
        await run_day6_query(agent, "查询任务 12345，检查最终输出类型")


@DAY6
@pytest.mark.parametrize("limit", [0, 4, True, "2"])
@pytest.mark.asyncio
async def test_day6_invalid_run_limit_never_calls_model(monkeypatch, limit):
    run = AsyncMock()
    monkeypatch.setattr(week6_fake_agent.Runner, "run", run)
    with pytest.raises(ValueError):
        await run_day6_query(SimpleNamespace(), "参数非法", max_tool_calls=limit)
    run.assert_not_awaited()


@DAY6
@pytest.mark.usefixtures("day6_local_tracing")
@pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY6_LLM") != "1",
    reason="真实 Day6 评估需另行设置 RUN_WEEK6_DAY6_LLM=1",
)
@pytest.mark.asyncio
async def test_day6_real_llm_evals(monkeypatch, tmp_path):
    """同一个 Agent 上运行同一组 10 个问题；真实模型成本，业务数据为 Mock。"""
    import time
    from openai import AsyncOpenAI
    from agents import OpenAIResponsesModel

    if os.environ.get("RUN_WEEK6_DAY6_LLM") != "1":
        pytest.skip("请先显式启用真实 LLM 评估")
    assert os.environ.get("OPENAI_API_KEY"), "真实评估缺少 OPENAI_API_KEY"
    lite = fakeAgentLite()
    rows = []
    path = tmp_path / "day6_real_llm_report.json"
    _day6_save_batch(path, rows, "real_llm")
    print(f"Day6 真实模型评估产物：{path}")
    try:
        await lite.Init()
        async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com", timeout=60.0) as client:
            model = OpenAIResponsesModel(model="deepseek-flash", openai_client=client)
            diagnostics = {}
            _day6_install_diagnostics(monkeypatch, model, diagnostics)
            agent = build_day6_agent(model, lite._mcp_server)
            for case in DAY6_CASES:
                diagnostics.clear()
                started = time.perf_counter()
                try:
                    record = await asyncio.wait_for(run_day6_query(agent, case["question"]), timeout=120)
                    checks = _grade_day6_case(record, case)
                    row = {"case_id": case["id"], **record, "checks": checks, "passed": all(checks.values())}
                except Exception as error:
                    # 记录失败现场并继续剩余用例；异常仍判失败，不自动重跑或放宽验收。
                    row = _day6_exception_record(case, error, time.perf_counter() - started, diagnostics)
                rows.append(row)
                _day6_save_batch(path, rows, "real_llm")
                print(f"{case['id']}: {'passed' if row['passed'] else 'failed'}")
                if "execution_error" in row:
                    message = row["execution_error_message"].replace("\n", " ")
                    print(f"  {row['execution_error']}: {message[:240]}（完整诊断见报告）")
        report = _day6_save_batch(path, rows, "real_llm")
        assert report["passed"], [row["case_id"] for row in rows if not row["passed"]]
    finally:
        await lite.close()
