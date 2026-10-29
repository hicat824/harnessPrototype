"""Week6 Day1～Day3 合并测试：实现只放在 week6.py，本文件只放测试。

数据来源
--------
沿用原 test_week6_day1.py 的合成任务和 10 个手工标注场景，结合 Day2 的
空结果 / 超时 / 403 样例；另补连接失败、500、非法 JSON。均无真实业务数据。

测试一：Day1 契约
----------------
检查两个 API 只读、参数约定和统一响应字段。自然语言问题是人工标注输入，
不做 LLM 意图识别；10 个场景的实际业务结果在 Day3 Service 测试中验收。

测试二：Day2 HTTP Client
-----------------------
架构：BusinessAPIClient → httpx.Client → MockTransport → mock_handler。
检查两个真实 HTTP 路径、流程精确匹配、顺序、limit、权限、空结果、异常、
字段脱敏及上下文管理。超时通过抛 ReadTimeout 注入，验证异常传播；
这不是对真实网络超时时长的验证。没有 sleep、真实网络或 LLM。

测试三：Day3 Service（自行实现后启用）
------------------------------------
输入：TASKS / SCENARIOS / 非法参数 / 故障 ID，以及程序侧 can_read_tasks。
输出：BusinessResult。检查统一结果、错误映射、权限、拒绝时 API 调用数为 0，
合法查询恰好调用一次；原始异常中的 URL、Token、正文不能进入结果。

测试四：Day3 MCP（自行实现后启用）
--------------------------------
先在进程内检查两个工具的 Schema、调用结果及非法参数拒绝；再以 stdio 启动
week6.py，真实完成 initialize → list_tools → call_tool → 关闭子进程。
不接 Agents SDK / LLM；自然语言自主选工具留到 Day4。

运行（仓库根目录）
------------------
已提供部分：.venv/bin/python -B -m pytest tests/test_week6.py -q
完成四个 TODO 后：RUN_WEEK6_DAY3=1 .venv/bin/python -B -m pytest tests/test_week6.py -q
只看 Day3：RUN_WEEK6_DAY3=1 .venv/bin/python -B -m pytest tests/test_week6.py -q -k day3
Day3 默认 skip 表示尚未验收；显式启用后未完成函数必须失败，不能静默跳过。
"""

import copy
import json
import os
from pathlib import Path
import sys

import httpx
import pytest

from week6 import (
    API_CONTRACTS, MAX_LIMIT, MOCK_TIME, MOCK_TOKENS, SOURCE, TASKS,
    BusinessAPIClient, BusinessResult, BusinessService,
    create_mcp_server, create_mock_http_client, map_api_error, mock_handler,
)


DAY3 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY3") != "1",
    reason="Day3 练习待实现，完成后设置 RUN_WEEK6_DAY3=1 验收",
)

# 人工标注工具和参数；不是已经接入 LLM 的证明。
SCENARIOS = [
    ("S01", "查询任务 12345 的状态", "query_task", {"task_id": "12345"}, "success", None),
    ("S02", "任务 12345 使用哪个引擎？", "query_task", {"task_id": "12345"}, "success", None),
    ("S03", "查看任务 12345 的参数", "query_task", {"task_id": "12345"}, "success", None),
    ("S04", "查询任务 12345 的上传日期", "query_task", {"task_id": "12345"}, "success", None),
    ("S05", "查询所有 import_std 任务", "query_tasks_by_engine", {"engine_code": "import_std"}, "success", None),
    ("S06", "列出前 10 个 import_std 任务", "query_tasks_by_engine", {"engine_code": "import_std", "limit": 10}, "partial", None),
    ("S07", "查询任务 99999", "query_task", {"task_id": "99999"}, "error", "NOT_FOUND"),
    ("S08", "查询无权限任务 34567", "query_task", {"task_id": "34567"}, "error", "FORBIDDEN"),
    ("S09", "查询任务状态但没有 ID", "query_task", {"task_id": None}, "needs_input", "MISSING_PARAMETER"),
    ("S10", "查询任务 88888 时超时", "query_task", {"task_id": "88888"}, "error", "TIMEOUT"),
]


@pytest.fixture
def api_client():
    with create_mock_http_client() as client:
        yield BusinessAPIClient(client)
    assert client.is_closed


def test_day1_read_only_contract_and_result_fields():
    assert set(API_CONTRACTS) == {"query_task", "query_tasks_by_engine"}
    assert API_CONTRACTS["query_task"]["path"] == "/tasks/{task_id}"
    assert API_CONTRACTS["query_tasks_by_engine"]["required_params"] == ["engine_code"]
    for contract in API_CONTRACTS.values():
        assert contract["method"] == "GET"
        assert contract["permission"] == "task.read"
    assert len(SCENARIOS) == 10
    result = BusinessResult(status="empty").model_dump()
    assert set(result) == {
        "status", "source", "data", "fetched_at", "truncated", "error_code", "next_cursor",
    }
    assert result["source"] == SOURCE
    assert result["fetched_at"] == MOCK_TIME


def test_day2_query_task_and_internal_fields(api_client):
    result = api_client.query_task("12345")
    assert result == BusinessResult(status="success", data={
        "task_id": "12345", "state": "RUNNING", "engineCode": "import_std",
        "params": {"source": "mock/a.pcd"},
        "tags": {"upload_date": "2026-10-20"},
    }).model_dump()
    assert "owner" not in result["data"]
    assert all(token not in json.dumps(result) for token in MOCK_TOKENS.values())


def test_day2_engine_exact_match_order_and_limit(api_client):
    before = copy.deepcopy(TASKS)
    result = api_client.query_tasks_by_engine("import_std")
    expected_ids = ["12345", *[str(20000 + number) for number in range(11)]]
    assert [item["task_id"] for item in result["data"]["items"]] == expected_ids
    assert result["data"]["total"] == 12
    assert all("owner" not in item for item in result["data"]["items"])
    limited = api_client.query_tasks_by_engine("import_std", limit=10)
    assert limited["status"] == "partial" and limited["truncated"] is True
    assert limited["data"]["total"] == 12
    assert [item["task_id"] for item in limited["data"]["items"]] == expected_ids[:10]
    assert TASKS == before


def test_day2_request_contract_and_encoded_query():
    requests = []

    def record(request):
        requests.append(request)
        return mock_handler(request)

    with httpx.Client(
        base_url="https://mock.example", transport=httpx.MockTransport(record),
        headers={"Authorization": f"Bearer {MOCK_TOKENS['alice']}"}, trust_env=False,
    ) as client:
        api = BusinessAPIClient(client)
        api.query_task("12345")
        # 若直接拼 URL，&limit=1 会被误认为第二个查询参数。
        api.query_tasks_by_engine("engine&limit=1", limit=3)
    assert [request.method for request in requests] == ["GET", "GET"]
    assert requests[0].url.path == "/tasks/12345"
    assert requests[1].url.path == "/tasks"
    assert requests[1].url.params["engine_code"] == "engine&limit=1"
    assert requests[1].url.params["limit"] == "3"


def test_day2_empty_is_normal(api_client):
    assert api_client.query_task("11111")["status"] == "empty"
    result = api_client.query_tasks_by_engine("engine_not_found")
    assert result["status"] == "empty" and result["error_code"] is None
    assert result["data"] == {"items": [], "total": 0}


@pytest.mark.parametrize("task_id,status", [("33333", 403), ("34567", 403), ("99999", 404), ("55555", 500)])
def test_day2_http_errors_propagate(api_client, task_id, status):
    with pytest.raises(httpx.HTTPStatusError) as error:
        api_client.query_task(task_id)
    assert error.value.response.status_code == status


@pytest.mark.parametrize("task_id,error_type", [("22222", httpx.ReadTimeout), ("44444", httpx.ConnectError)])
def test_day2_network_errors_propagate(api_client, task_id, error_type):
    with pytest.raises(error_type):
        api_client.query_task(task_id)


def test_day2_invalid_json_propagates(api_client):
    with pytest.raises(json.JSONDecodeError):
        api_client.query_task("66666")


@pytest.mark.parametrize("limit", [0, -1, 101, "bad", True])
def test_day2_mock_rejects_invalid_limit(api_client, limit):
    with pytest.raises(httpx.HTTPStatusError) as error:
        api_client.query_tasks_by_engine("import_std", limit=limit)
    assert error.value.response.status_code == 400


def test_day2_identity_and_read_only_boundary():
    with create_mock_http_client("bob") as client:
        api = BusinessAPIClient(client)
        assert api.query_task("34567")["data"]["state"] == "FAILED"
        assert api.query_tasks_by_engine("import_std")["data"]["total"] == 1
    with create_mock_http_client("mallory") as client:
        assert client.get("/tasks/12345").status_code == 403
    with create_mock_http_client() as client:
        assert client.post("/tasks/12345").status_code == 405
        assert client.get("/tasks_extra/12345").status_code == 404


@DAY3
@pytest.mark.parametrize("case_id,question,tool,params,status,error", SCENARIOS, ids=[case[0] for case in SCENARIOS])
def test_day3_ten_business_scenarios(api_client, case_id, question, tool, params, status, error):
    result = getattr(BusinessService(api_client), tool)(**params)
    assert isinstance(result, BusinessResult)
    assert (result.status, result.error_code) == (status, error), (case_id, question)
    assert result.source == SOURCE
    if case_id == "S01":
        assert result.data["state"] == "RUNNING"
    elif case_id == "S02":
        assert result.data["engineCode"] == "import_std"
    elif case_id == "S03":
        assert result.data["params"]["source"] == "mock/a.pcd"
    elif case_id == "S04":
        assert result.data["tags"]["upload_date"] == "2026-10-20"
    elif case_id in {"S05", "S06"}:
        assert result.data["total"] == 12
        assert len(result.data["items"]) == (12 if case_id == "S05" else 10)
        assert result.truncated is (case_id == "S06")
        assert all(item["engineCode"] == "import_std" for item in result.data["items"])


@DAY3
@pytest.mark.parametrize("method,params,status,error", [
    ("query_task", {"task_id": None}, "needs_input", "MISSING_PARAMETER"),
    ("query_task", {"task_id": ""}, "needs_input", "MISSING_PARAMETER"),
    *[("query_task", {"task_id": value}, "error", "INVALID_ARGUMENT")
      for value in (12345, True, "abc", " 12345", "１２３４５")],
    ("query_tasks_by_engine", {"engine_code": None}, "needs_input", "MISSING_PARAMETER"),
    ("query_tasks_by_engine", {"engine_code": ""}, "needs_input", "MISSING_PARAMETER"),
    *[("query_tasks_by_engine", {"engine_code": value}, "error", "INVALID_ARGUMENT")
      for value in (123, True, "   ")],
    *[("query_tasks_by_engine", {"engine_code": "import_std", "limit": value}, "error", "INVALID_ARGUMENT")
      for value in (0, -1, 101, "10", True)],
])
def test_day3_invalid_input_never_requests_api(api_client, monkeypatch, method, params, status, error):
    calls = []
    monkeypatch.setattr(api_client, method, lambda *args, **kwargs: calls.append((args, kwargs)))
    result = getattr(BusinessService(api_client), method)(**params)
    assert (result.status, result.error_code) == (status, error)
    assert calls == []


@DAY3
@pytest.mark.parametrize("method,params", [
    ("query_task", {"task_id": "12345"}),
    ("query_tasks_by_engine", {"engine_code": "import_std"}),
])
def test_day3_permission_denied_before_http(api_client, monkeypatch, method, params):
    calls = []
    monkeypatch.setattr(api_client, method, lambda *args, **kwargs: calls.append((args, kwargs)))
    result = getattr(BusinessService(api_client, can_read_tasks=False), method)(**params)
    assert (result.status, result.error_code) == ("error", "FORBIDDEN")
    assert calls == []


@DAY3
@pytest.mark.parametrize("task_id,expected", [
    ("11111", ("empty", None)), ("33333", ("error", "FORBIDDEN")),
    ("99999", ("error", "NOT_FOUND")), ("22222", ("error", "TIMEOUT")),
    ("44444", ("error", "CONNECTION_ERROR")), ("55555", ("error", "UPSTREAM_HTTP_ERROR")),
    ("66666", ("error", "INVALID_RESPONSE")),
])
def test_day3_result_mapping(api_client, task_id, expected):
    result = BusinessService(api_client).query_task(task_id)
    assert (result.status, result.error_code) == expected
    assert result.data is None


@DAY3
def test_day3_only_one_http_call_and_no_sensitive_error_text():
    requests = []

    def record(request):
        requests.append(request)
        return mock_handler(request)

    with httpx.Client(
        base_url="https://mock.example", transport=httpx.MockTransport(record),
        headers={"Authorization": f"Bearer {MOCK_TOKENS['alice']}"}, trust_env=False,
    ) as client:
        service = BusinessService(BusinessAPIClient(client))
        service.query_task("12345")
        assert len(requests) == 1
        service.query_tasks_by_engine("import_std", limit=3)
        assert len(requests) == 2

    error = httpx.ReadTimeout("private URL and mock-secret-token", request=httpx.Request("GET", "https://private.example"))
    result = map_api_error(error)
    assert (result.status, result.error_code) == ("error", "TIMEOUT")
    assert "private" not in result.model_dump_json()
    assert "mock-secret-token" not in result.model_dump_json()


@DAY3
def test_day3_empty_list_and_response_validation(api_client, monkeypatch):
    service = BusinessService(api_client)
    result = service.query_tasks_by_engine("engine_not_found")
    assert result.status == "empty" and result.error_code is None
    assert result.data == {"items": [], "total": 0}
    monkeypatch.setattr(api_client, "query_task", lambda task_id: {"unexpected": "field"})
    result = service.query_task("12345")
    assert (result.status, result.error_code) == ("error", "INVALID_RESPONSE")


@DAY3
@pytest.mark.asyncio
async def test_day3_mcp_schema_and_calls(api_client, monkeypatch):
    """同进程调用用于检查注册、Schema 与服务层委派；不是 stdio 联通证明。"""
    service = BusinessService(api_client)
    server = create_mcp_server(service)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {"query_task", "query_tasks_by_engine"}
    task_schema = tools["query_task"].input_schema
    list_schema = tools["query_tasks_by_engine"].input_schema
    assert set(task_schema["properties"]) == {"task_id"}
    assert task_schema["required"] == ["task_id"]
    assert task_schema["properties"]["task_id"]["type"] == "string"
    assert set(list_schema["properties"]) == {"engine_code", "limit"}
    assert list_schema["required"] == ["engine_code"]
    assert list_schema["properties"]["engine_code"]["type"] == "string"
    assert list_schema["properties"]["limit"]["type"] == "integer"
    assert list_schema["properties"]["limit"]["default"] == MAX_LIMIT
    assert list_schema["properties"]["limit"]["minimum"] == 1
    assert list_schema["properties"]["limit"]["maximum"] == MAX_LIMIT

    # 替换 Service 方法，确认 MCP 包装函数确实委托到传入的同一个 Service。
    calls = []

    def record(task_id):
        calls.append(task_id)
        return BusinessResult(status="empty")

    monkeypatch.setattr(service, "query_task", record)
    result = await server.call_tool("query_task", {"task_id": "12345"})
    assert result.structured_content["status"] == "empty"
    assert calls == ["12345"]

    from mcp.server.mcpserver.exceptions import ToolError

    for params in ({}, {"task_id": 12345}):
        with pytest.raises(ToolError):
            await server.call_tool("query_task", params)
    for params in ({"engine_code": "import_std", "limit": True}, {"engine_code": "import_std", "limit": 101}):
        with pytest.raises(ToolError):
            await server.call_tool("query_tasks_by_engine", params)
    assert calls == ["12345"]
    result = await server.call_tool("query_tasks_by_engine", {"engine_code": "import_std", "limit": 3})
    assert result.structured_content["status"] == "partial"
    assert len(result.structured_content["data"]["items"]) == 3


@DAY3
@pytest.mark.asyncio
async def test_day3_mcp_stdio_lifecycle():
    """真实启动本地子进程与 stdio 协议；不调用 LLM、外网或公司 API。"""
    import asyncio

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    repo = Path(__file__).resolve().parents[1]
    params = StdioServerParameters(command=sys.executable, args=["-B", str(repo / "week6.py")])

    async def check():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=10) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == {"query_task", "query_tasks_by_engine"}
                result = await session.call_tool("query_task", {"task_id": "12345"})
                assert not result.is_error
                assert result.structured_content["data"]["task_id"] == "12345"
                result = await session.call_tool("query_tasks_by_engine", {"engine_code": "import_std", "limit": 3})
                assert not result.is_error
                assert result.structured_content["status"] == "partial"
                assert len(result.structured_content["data"]["items"]) == 3

    # 包括启动与关闭在内的总时间边界，防止练习服务挂起后 pytest 无限等待。
    await asyncio.wait_for(check(), timeout=20)
