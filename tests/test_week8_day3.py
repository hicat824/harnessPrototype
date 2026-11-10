"""Week8 Day3：已有业务层 → 参数 Schema → Tool → Agent 的分层验收。

默认检查基础部分；完整离线流程需设置 RUN_WEEK8_DAY3=1。
真实模型测试另需 RUN_WEEK8_DAY3_LLM=1；所有业务 HTTP 仍来自 Week6 Mock。
逐项数据来源、输入输出和报告位置见 langchain_learning/week8.md 的 Day3 章节。
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("langchain", reason="请使用 Week8 独立环境并安装最新 requirements.txt")
pytest.importorskip("mcp", reason="复用 Week6 业务层需要 requirements.txt 中的 mcp")

import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import Field, ValidationError

from langchain_learning.day1 import build_model
from langchain_learning.day3 import (
    DAY3_INSTRUCTIONS, QueryTaskInput, QueryTasksInput, build_agent, build_tools,
    describe_tools, mock_service, run_agent,
)


REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("WEEK8_DAY3_OUTPUT", str(REPO / "langchain_learning/artifacts/day3")))
DAY3 = pytest.mark.skipif(os.environ.get("RUN_WEEK8_DAY3") != "1", reason="完整流程需设置 RUN_WEEK8_DAY3=1")
REAL = pytest.mark.skipif(
    os.environ.get("RUN_WEEK8_DAY3") != "1" or os.environ.get("RUN_WEEK8_DAY3_LLM") != "1",
    reason="真实调用需额外设置 RUN_WEEK8_DAY3_LLM=1",
)


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def business():
    with mock_service() as service:
        requests = []
        # 仅记录方法、路径和查询参数，不记录合成 Token 或任何请求头。
        service.api_client.client.event_hooks["request"].append(
            lambda request: requests.append({"method": request.method, "path": request.url.path,
                                             "query": dict(request.url.params)}))
        yield service, requests
    assert service.api_client.client.is_closed


class ScriptedModel(GenericFakeChatModel):
    """支持 Agent 工具绑定的假模型；脚本返回工具调用/最终回答，不做语义推理。"""

    observed_inputs: list = Field(default_factory=list)
    advertised_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.advertised_names = [item.name for item in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.observed_inputs.append(copy.deepcopy(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def call(name="query_task", args=None, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": {"task_id": "12345"} if args is None else args,
        "id": call_id, "type": "tool_call",
    }])


def scripted(*responses):
    return ScriptedModel(messages=iter(responses))


@pytest.mark.parametrize("values", [
    {"engine_code": ""}, {"engine_code": " "}, {"engine_code": 123},
    {"engine_code": "import_std", "limit": 0}, {"engine_code": "import_std", "limit": 101},
    {"engine_code": "import_std", "limit": True}, {"engine_code": "import_std", "limit": "3"},
    {"engine_code": "import_std", "token": "不能是工具参数"},
])
def test_provided_engine_schema_rejects_invalid_arguments(values):
    with pytest.raises(ValidationError):
        QueryTasksInput.model_validate(values)
    assert QueryTasksInput(engine_code=" import_std ").engine_code == " import_std "
    assert QueryTasksInput(engine_code="import_std").limit == 100


def test_provided_business_data_is_same_week6_mock(business):
    service, requests = business
    assert service.query_task("12345").data["state"] == "RUNNING"
    result = service.query_tasks_by_engine("import_std", 3)
    assert result.status == "partial" and result.truncated
    assert result.data["total"] == 12 and len(result.data["items"]) == 3
    assert [item["task_id"] for item in result.data["items"]] == ["12345", "20000", "20001"]
    assert all(item["engineCode"] == "import_std" for item in result.data["items"])
    assert all(request["method"] == "GET" for request in requests)


def test_provided_cli_requires_explicit_real_flag():
    result = subprocess.run([sys.executable, "-B", "-m", "langchain_learning.day3", "--question", "查询任务"],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--real-llm" in result.stderr


@pytest.mark.asyncio
async def test_provided_runner_saves_safe_failure_without_fake_usage(tmp_path):
    class BrokenAgent:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("private-error-detail")
    path = tmp_path / "failure_report.json"
    with pytest.raises(RuntimeError):
        await run_agent(BrokenAgent(), "查询任务 12345", output_path=path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["error"] == {"type": "RuntimeError"} and saved["execution_status"] == "failed"
    assert saved["usage"] is None and saved["final_answer"] is None
    assert "private-error-detail" not in path.read_text(encoding="utf-8")


@DAY3
@pytest.mark.parametrize("task_id", ["", " ", "abc", " 123 ", "１２３", 123, True, None])
def test_task_id_schema_preserves_string_identity_and_rejects_invalid_values(task_id):
    with pytest.raises(ValidationError):
        QueryTaskInput(task_id=task_id)
    assert QueryTaskInput(task_id="00123").task_id == "00123"


@DAY3
def test_tool_schemas_names_descriptions_and_program_side_dependencies(business):
    service, requests = business
    tools = build_tools(service)
    schemas = describe_tools(tools)
    assert [row["name"] for row in schemas] == ["query_task", "query_tasks_by_engine"]
    assert all(row["description"] for row in schemas)
    task, engine = [row["input_schema"] for row in schemas]
    assert set(task["properties"]) == {"task_id"} and task["required"] == ["task_id"]
    assert task["properties"]["task_id"]["type"] == "string" and "pattern" in task["properties"]["task_id"]
    assert set(engine["properties"]) == {"engine_code", "limit"} and engine["required"] == ["engine_code"]
    assert engine["properties"]["limit"]["default"] == 100
    assert task["additionalProperties"] is False and engine["additionalProperties"] is False
    assert requests == []
    path = OUTPUT / "offline/tool_schemas.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schemas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@DAY3
@pytest.mark.parametrize("name,params,expected", [
    ("query_task", {"task_id": "12345"}, ("success", None)),
    ("query_task", {"task_id": "00123"}, ("error", "NOT_FOUND")),
    ("query_task", {"task_id": "34567"}, ("error", "FORBIDDEN")),
    ("query_task", {"task_id": "99999"}, ("error", "NOT_FOUND")),
    ("query_tasks_by_engine", {"engine_code": "engine_not_found"}, ("empty", None)),
    ("query_tasks_by_engine", {"engine_code": "import_std", "limit": 3}, ("partial", None)),
])
def test_direct_tools_keep_business_contract_and_call_id(business, name, params, expected):
    service, requests = business
    tools = {item.name: item for item in build_tools(service)}
    before = copy.deepcopy(params)
    result = tools[name].invoke({"name": name, "args": params, "id": "direct_1", "type": "tool_call"})
    assert isinstance(result, ToolMessage) and result.tool_call_id == "direct_1"
    payload = json.loads(result.content)
    assert (payload["status"], payload["error_code"]) == expected
    assert payload["source"] == "mock_task_api" and params == before
    assert "mock-alice-token" not in result.content and '"owner"' not in result.content
    assert len(requests) == 1
    if params.get("task_id") == "00123":
        assert requests[0]["path"] == "/tasks/00123"


@DAY3
@pytest.mark.parametrize("name,params", [
    ("query_task", {}), ("query_task", {"task_id": "abc"}),
    ("query_task", {"task_id": 12345}), ("query_task", {"task_id": "12345", "token": "x"}),
    ("query_tasks_by_engine", {"engine_code": "import_std", "limit": True}),
])
def test_direct_tool_validation_stops_before_service_and_http(business, name, params, monkeypatch):
    service, requests = business
    entered = []
    monkeypatch.setattr(service, name, lambda *args, **kwargs: entered.append(args))
    tools = {item.name: item for item in build_tools(service)}
    with pytest.raises(ValidationError):
        tools[name].invoke(params)
    assert entered == [] and requests == []


@DAY3
@pytest.mark.asyncio
async def test_agent_tool_loop_records_result_before_final_answer(business):
    service, requests = business
    model = scripted(call(), AIMessage(content="任务 12345 为 RUNNING，来源 mock_task_api。"))
    agent = build_agent(model, build_tools(service))
    assert requests == [] and model.observed_inputs == []
    report = await run_agent(agent, "查询任务 12345 的状态", output_path=OUTPUT / "offline/query_report.json")
    assert report["execution_status"] == "succeeded" and report["model_responses"] == 2
    assert [row["type"] for row in report["messages"]] == ["human", "ai", "tool", "ai"]
    assert report["tool_calls"][0]["id"] == report["tool_results"][0]["call_id"] == "call_1"
    assert json.loads(report["tool_results"][0]["content"])["data"]["state"] == "RUNNING"
    assert model.observed_inputs[0][0].content == DAY3_INSTRUCTIONS
    assert isinstance(model.observed_inputs[1][-1], ToolMessage)
    assert set(model.advertised_names) == {"query_task", "query_tasks_by_engine"}
    assert report["usage"] is None and len(requests) == 1


@DAY3
@pytest.mark.asyncio
async def test_agent_validation_feedback_allows_a_scripted_correction(business):
    service, requests = business
    model = scripted(call(args={"task_id": 12345}, call_id="bad_1"),
                     call(call_id="good_1"), AIMessage(content="已更正字符串 ID 并查询。"))
    report = await run_agent(build_agent(model, build_tools(service)), "查询任务 12345",
                             output_path=OUTPUT / "offline/correction_report.json")
    assert [row["status"] for row in report["tool_results"]] == ["error", "success"]
    assert [row["call_id"] for row in report["tool_results"]] == ["bad_1", "good_1"]
    assert report["model_responses"] == 3 and len(requests) == 1


@DAY3
@pytest.mark.asyncio
async def test_missing_id_asks_without_calling_business_api(business):
    service, requests = business
    model = scripted(AIMessage(content="请提供需要查询的任务 ID。"))
    report = await run_agent(build_agent(model, build_tools(service)), "帮我查询任务状态")
    assert report["tool_calls"] == [] and report["tool_results"] == [] and requests == []
    assert report["final_answer"] == "请提供需要查询的任务 ID。"


@DAY3
@pytest.mark.asyncio
async def test_repeated_agent_runs_do_not_automatically_share_history(business):
    service, _ = business
    model = scripted(AIMessage(content="请提供任务 ID。"), AIMessage(content="请提供任务 ID。"))
    agent = build_agent(model, build_tools(service))
    await run_agent(agent, "第一个独立问题")
    second = await run_agent(agent, "第二个独立问题")
    assert len(second["messages"]) == 2
    assert [item.content for item in model.observed_inputs[1]] == [DAY3_INSTRUCTIONS, "第二个独立问题"]


@DAY3
@pytest.mark.asyncio
async def test_graph_step_limit_stops_repeated_tool_calls(business, tmp_path):
    service, requests = business
    # 给假模型足够多的合法调用，真正的停止条件应来自图步数限制。
    model = scripted(*(call(call_id=f"loop_{number}") for number in range(20)))
    output = tmp_path / "step_limit_report.json"
    with pytest.raises(GraphRecursionError):
        await run_agent(build_agent(model, build_tools(service)), "查询任务 12345",
                        recursion_limit=3, output_path=output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["error"] == {"type": "GraphRecursionError"}
    assert saved["usage"] is None and len(requests) < 20


@REAL
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,question,tool_name,params,status,error_code", [
    ("task", "查询任务 12345 的状态，并说明来源。", "query_task", {"task_id": "12345"}, "success", None),
    ("engine", "列出 import_std 的前 3 个任务，说明总数与是否为部分记录。",
     "query_tasks_by_engine", {"engine_code": "import_std", "limit": 3}, "partial", None),
    ("missing", "帮我查询任务状态。", None, None, None, None),
    ("not_found", "查询任务 99999 的状态，并说明来源。", "query_task", {"task_id": "99999"}, "error", "NOT_FOUND"),
])
async def test_real_agent_selects_tool_and_preserves_arguments(business, case_id, question,
                                                             tool_name, params, status, error_code):
    assert os.environ.get("OPENAI_API_KEY"), "需要项目既有 DeepSeek OPENAI_API_KEY"
    service, requests = business
    path = OUTPUT / f"real_model/{case_id}_report.json"
    with httpx.Client(trust_env=False) as client:
        async with httpx.AsyncClient(trust_env=False) as async_client:
            model = build_model(http_client=client, http_async_client=async_client)
            report = await run_agent(build_agent(model, build_tools(service)), question, output_path=path)
    assert report["execution_status"] == "succeeded"
    if tool_name is None:
        assert report["tool_calls"] == [] and requests == []
    else:
        assert len(report["tool_calls"]) == len(report["tool_results"]) == len(requests) == 1
        assert report["tool_calls"][0]["name"] == tool_name
        assert report["tool_calls"][0]["args"] == params
        payload = json.loads(report["tool_results"][0]["content"])
        assert (payload["status"], payload["error_code"]) == (status, error_code)
    print(f"Week8 Day3 真实产物：{path}；回答是否正确解释数据需阅读复核")
