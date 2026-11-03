"""Week6 Day4～Day5：MCPServerStdio 与单个业务 Agent 的联调验收。

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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.items import ToolCallItem, ToolCallOutputItem

import week6_fake_agent
from week6_fake_agent import fakeAgentLite
from week6 import BusinessResult, MOCK_TOKENS, SOURCE


DAY5 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY5") != "1",
    reason="Day5 验收需设置 RUN_WEEK6_DAY5=1；说明见 docs/week6_day5.md",
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
