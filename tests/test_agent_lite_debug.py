"""只验证 AgentLite 调试接口，不运行四轮验证、不连接模型或 Docker。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

import agentLite
from tools import EvalContext, FatalToolError
from baseModel import AgentOutput


@pytest.fixture
def debug_agent(monkeypatch):
    monkeypatch.setenv("THREAD_ID", "offline-debug-test")
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    # 注入 Session 后不应再打开日常对话数据库。
    def unexpected_session(*args, **kwargs):
        raise AssertionError("不应创建默认 SQLiteSession")
    monkeypatch.setattr(agentLite, "SQLiteSession", unexpected_session)
    session = SimpleNamespace(session_id="independent-session")
    lite = agentLite.AgentLite(session=session)
    lite._client = object()
    usage = SimpleNamespace(requests=2, input_tokens=100, output_tokens=20, total_tokens=120)
    response = SimpleNamespace(final_output="真实 Runner 的结果由此处模拟", context_wrapper=SimpleNamespace(usage=usage))
    run = AsyncMock(return_value=response)
    monkeypatch.setattr(agentLite.Runner, "run", run)
    return lite, run


def test_default_call_still_returns_bool_and_uses_sdk_defaults(debug_agent):
    lite, run = debug_agent
    assert asyncio.run(lite.simpleCallLLM("任务")) is True
    assert run.call_args.kwargs["session"] is lite._session
    assert "max_turns" not in run.call_args.kwargs
    assert "run_config" not in run.call_args.kwargs


def test_debug_call_records_output_usage_and_preserves_task_context(debug_agent):
    lite, run = debug_agent
    debug = {"error": "上一轮错误"}
    context = EvalContext()
    assert asyncio.run(lite.simpleCallLLM("任务", context, debug_info=debug, max_turns=30)) is True
    assert debug["error"] is None
    assert debug["final_output"] == "真实 Runner 的结果由此处模拟"
    assert debug["usage"] == {"requests": 2, "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert run.call_args.kwargs["context"] is context
    assert run.call_args.kwargs["max_turns"] == 30
    assert run.call_args.kwargs["run_config"].tracing_disabled is True


def test_structured_output_is_serialized_for_debug_and_console(debug_agent, capsys):
    lite, run = debug_agent
    output = AgentOutput(
        summary="已修复代码，仍需运行测试。", changed_files=["src/a.py"],
        test_status="not_run", unfinished_work=["运行固定测试"],
    )
    run.return_value.final_output = output
    debug = {}
    assert asyncio.run(lite.simpleCallLLM("任务", debug_info=debug)) is True
    assert json.loads(debug["final_output"]) == output.model_dump()
    answer = capsys.readouterr().out.splitlines()[0].removeprefix("answer = ")
    assert json.loads(answer) == output.model_dump()


def test_known_tool_failure_returns_false_and_records_error(debug_agent):
    lite, run = debug_agent
    run.side_effect = FatalToolError("PERMISSION_DENIED: 无权访问")
    debug = {}
    assert asyncio.run(lite.simpleCallLLM("任务", debug_info=debug)) is False
    assert debug["error"] == "FatalToolError: PERMISSION_DENIED: 无权访问"
    assert debug["final_output"] is None
    assert debug["usage"] is None


def test_unexpected_error_is_recorded_and_reraised(debug_agent):
    lite, run = debug_agent
    run.side_effect = RuntimeError("模型配置错误")
    debug = {}
    with pytest.raises(RuntimeError, match="模型配置错误"):
        asyncio.run(lite.simpleCallLLM("任务", debug_info=debug))
    assert debug["error"] == "RuntimeError: 模型配置错误"