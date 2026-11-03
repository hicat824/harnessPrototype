"""Week6 Day1～Day5 合并测试：实现只放在 week6.py，本文件只放测试。

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

测试五：Day5 结果边界、权限与 HTTP 有限重试
----------------------------------------
数据来源：现有 TASKS，加上测试现场生成的大 JSON、合成密钥与故障响应。
架构：结果准备函数 / Client → MockTransport / MCP → Service。
检查：脱敏、完整响应预算、来源与总数、实际请求次数和服务端权限。
每个测试的独立说明、三个函数的输入输出见 docs/week6_day5.md。

运行（仓库根目录）
------------------
已提供部分：.venv/bin/python -B -m pytest tests/test_week6.py -q
完成四个 TODO 后：RUN_WEEK6_DAY3=1 .venv/bin/python -B -m pytest tests/test_week6.py -q
只看 Day3：RUN_WEEK6_DAY3=1 .venv/bin/python -B -m pytest tests/test_week6.py -q -k day3
Day3 默认 skip 表示尚未验收；显式启用后未完成函数必须失败，不能静默跳过。
Day5：RUN_WEEK6_DAY5=1 .venv/bin/python -B -m pytest tests/test_week6.py -q -k day5
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
    MAX_TOOL_RESULT_CHARS, prepare_tool_result, redact_sensitive_data,
    request_json_with_retry, response, visible_task,
)


DAY3 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY3") != "1",
    reason="Day3 练习待实现，完成后设置 RUN_WEEK6_DAY3=1 验收",
)

DAY5 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK6_DAY5") != "1",
    reason="Day5 验收需显式设置 RUN_WEEK6_DAY5=1",
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


# Day5：测试数据均为本文件生成的合成值；各测试来源、目标及函数契约见
# docs/week6_day5.md。这里不调用 LLM，也不会访问真实业务系统。
def _day5_chars(result):
    """仅测量整个结果的 JSON 字符数，不实现被测的裁剪规则。"""
    return len(json.dumps(result.model_dump(), ensure_ascii=True))


def _day5_large_task(task_id="12345"):
    """人为放大公开任务字段，模拟大 JSON；不修改全局 TASKS。"""
    return {
        "task_id": task_id, "state": "RUNNING", "engineCode": "import_std",
        "params": {"source": "mock/a.pcd", "large_text": "测试数据" * 6000},
        "tags": {"note": "合成备注" * 6000},
    }


@DAY5
def test_day5_redact_sensitive_fields_without_changing_input():
    value = {
        "task_id": "12345", "params": {
            "API_KEY": "synthetic-key", "nested": [{"Password": "synthetic-password"}],
            "source": "mock/a.pcd", "authorization": "Bearer synthetic-token",
        },
        "tags": {"token": "synthetic-token", "enabled": True, "missing": None},
    }
    before = copy.deepcopy(value)
    result = redact_sensitive_data(value)
    assert result["task_id"] == "12345"
    assert result["params"]["API_KEY"] == "[REDACTED]"
    assert result["params"]["nested"][0]["Password"] == "[REDACTED]"
    assert result["params"]["authorization"] == "[REDACTED]"
    assert result["params"]["source"] == "mock/a.pcd"
    assert result["tags"] == {"token": "[REDACTED]", "enabled": True, "missing": None}
    assert value == before
    result["params"]["source"] = "changed"
    assert value == before, "返回值不能共享原输入的嵌套字典"


@DAY5
def test_day5_short_result_preserves_business_data():
    original = BusinessResult(status="success", data=visible_task(TASKS["12345"]))
    result = prepare_tool_result(original)
    assert result == original
    assert result is not original
    result.data["params"]["source"] = "changed"
    assert original.data["params"]["source"] == "mock/a.pcd"


@DAY5
def test_day5_unknown_task_fields_are_removed_and_redacted():
    task = visible_task(TASKS["12345"])
    task.update(owner="alice", internal_debug="private synthetic detail")
    task["params"]["api_key"] = "DAY5-SYNTHETIC-KEY"
    original = BusinessResult(status="success", data=task)
    before = original.model_dump()
    result = prepare_tool_result(original)
    assert "owner" not in result.data and "internal_debug" not in result.data
    assert result.data["params"]["api_key"] == "[REDACTED]"
    assert "DAY5-SYNTHETIC-KEY" not in result.model_dump_json()
    assert result.status == "partial" and result.truncated is True
    assert result.source == SOURCE and result.fetched_at == MOCK_TIME
    assert original.model_dump() == before


@DAY5
def test_day5_large_detail_has_budget_and_original_reference():
    original = BusinessResult(status="success", data=_day5_large_task())
    before = original.model_dump()
    result = prepare_tool_result(original, max_chars=1024)
    assert _day5_chars(result) <= 1024
    assert result.status == "partial" and result.truncated is True
    assert result.source == SOURCE and result.fetched_at == MOCK_TIME
    assert result.data["task_id"] == "12345"
    assert result.data["state"] == "RUNNING"
    assert result.data["engineCode"] == "import_std"
    assert original.model_dump() == before


@DAY5
def test_day5_large_list_keeps_prefix_total_and_truncation():
    items = [_day5_large_task(str(30000 + number)) for number in range(30)]
    original = BusinessResult(status="success", data={"items": items, "total": 30})
    before = original.model_dump()
    result = prepare_tool_result(original, max_chars=1024)
    selected = result.data["items"]
    assert _day5_chars(result) <= 1024
    assert 0 < len(selected) < len(items)
    assert [item["task_id"] for item in selected] == [item["task_id"] for item in items[:len(selected)]]
    assert result.data["total"] == 30
    assert result.status == "partial" and result.truncated is True
    assert result.next_cursor is None  # 本日使用 limit / 前缀截断，不虚构翻页能力。
    assert original.model_dump() == before


@DAY5
def test_day5_no_record_fits_is_partial_instead_of_empty():
    original = BusinessResult(
        status="success", data={"items": [_day5_large_task("7" * 3000)], "total": 1},
    )
    result = prepare_tool_result(original, max_chars=512)
    assert _day5_chars(result) <= 512
    assert result.data == {"items": [], "total": 1}
    assert result.status == "partial" and result.truncated is True


@DAY5
def test_day5_identifier_cannot_be_cut_to_fit():
    original = BusinessResult(status="success", data=_day5_large_task("7" * 3000))
    result = prepare_tool_result(original, max_chars=512)
    assert (result.status, result.error_code) == ("error", "RESULT_TOO_LARGE")
    assert result.data is None and result.truncated is True
    assert result.source == SOURCE and result.fetched_at == MOCK_TIME
    assert _day5_chars(result) <= 512
    assert original.data["task_id"] == "7" * 3000


@DAY5
def test_day5_metadata_itself_cannot_fit():
    original = BusinessResult(status="success", source="synthetic-source/" * 1000, data=_day5_large_task())
    with pytest.raises(ValueError):
        prepare_tool_result(original, max_chars=512)


@DAY5
@pytest.mark.parametrize("max_chars", [0, 100, True, "1024"])
def test_day5_invalid_budget_is_rejected(max_chars):
    with pytest.raises(ValueError):
        prepare_tool_result(BusinessResult(status="empty"), max_chars=max_chars)


@DAY5
@pytest.mark.parametrize("original", [
    BusinessResult(status="empty", data={"items": [], "total": 0}),
    BusinessResult(status="error", error_code="FORBIDDEN"),
    BusinessResult(status="needs_input", error_code="MISSING_PARAMETER"),
    BusinessResult(status="partial", data={"items": [visible_task(TASKS["12345"])], "total": 12}, truncated=True),
])
def test_day5_existing_result_states_and_metadata_are_preserved(original):
    assert prepare_tool_result(original) == original


@DAY5
@pytest.mark.parametrize("failure", ["timeout", "connect", 429, 503])
def test_day5_transient_failure_really_sends_a_second_request(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("synthetic timeout", request=request)
            if failure == "connect":
                raise httpx.ConnectError("synthetic connection failure", request=request)
            return httpx.Response(failure, json=response("error", error_code="UPSTREAM_HTTP_ERROR"))
        return httpx.Response(200, json=response("success", data=visible_task(TASKS["12345"])))

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = BusinessAPIClient(client, max_attempts=2).query_task("12345")
    assert result["status"] == "success" and result["data"]["task_id"] == "12345"
    assert len(requests) == 2
    assert requests[0].url == requests[1].url
    assert all(request.method == "GET" for request in requests)


@DAY5
@pytest.mark.parametrize("max_attempts", [1, 2, 3])
@pytest.mark.parametrize("failure,code", [
    ("timeout", "TIMEOUT"), ("connect", "CONNECTION_ERROR"), (429, "UPSTREAM_HTTP_ERROR"), (503, "UPSTREAM_HTTP_ERROR"),
])
def test_day5_attempt_limit_and_final_service_error(failure, code, max_attempts):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if failure == "connect":
            raise httpx.ConnectError("synthetic connection failure", request=request)
        return httpx.Response(failure, json=response("error", error_code="UPSTREAM_HTTP_ERROR"))

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = BusinessService(BusinessAPIClient(client, max_attempts=max_attempts)).query_task("12345")
    assert (result.status, result.error_code) == ("error", code)
    assert result.data is None
    assert len(requests) == max_attempts, "max_attempts 是总尝试次数，不是额外重试次数"


@DAY5
@pytest.mark.parametrize("status,code", [(400, "UPSTREAM_HTTP_ERROR"), (403, "FORBIDDEN"), (404, "NOT_FOUND"), (422, "UPSTREAM_HTTP_ERROR")])
def test_day5_non_retryable_http_failure_only_once(status, code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json=response("error", error_code=code))

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = BusinessService(BusinessAPIClient(client, max_attempts=3)).query_task("12345")
    assert (result.status, result.error_code) == ("error", code)
    assert len(requests) == 1


@DAY5
def test_day5_invalid_json_does_not_retry_or_expose_body():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="not-json DAY5-SYNTHETIC-SECRET")

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = BusinessService(BusinessAPIClient(client, max_attempts=3)).query_task("12345")
    assert (result.status, result.error_code) == ("error", "INVALID_RESPONSE")
    assert len(requests) == 1
    assert "DAY5-SYNTHETIC-SECRET" not in result.model_dump_json()


@DAY5
@pytest.mark.parametrize("max_attempts", [0, 4, True, "2"])
def test_day5_invalid_attempt_limit_sends_no_request(max_attempts):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response("empty"))

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(ValueError):
            request_json_with_retry(client, "/tasks/12345", max_attempts=max_attempts)
    assert requests == []


@DAY5
@pytest.mark.parametrize("method,params", [
    ("query_task", {"task_id": "12345"}),
    ("query_tasks_by_engine", {"engine_code": "import_std", "limit": 3}),
])
def test_day5_empty_result_is_not_retried(method, params):
    requests = []
    payload = response("empty", data=None if method == "query_task" else {"items": [], "total": 0})

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = getattr(BusinessService(BusinessAPIClient(client, max_attempts=3)), method)(**params)
    assert result.status == "empty" and result.error_code is None
    assert len(requests) == 1


@DAY5
@pytest.mark.parametrize("method,params", [
    ("query_task", {"task_id": "12345"}),
    ("query_tasks_by_engine", {"engine_code": "import_std", "limit": 3}),
])
@pytest.mark.asyncio
async def test_day5_mcp_boundary_uses_prepared_result(api_client, monkeypatch, method, params):
    # 任务含长字段和合成密钥，通过实际注册的 MCP 工具验收出站边界。
    service = BusinessService(api_client)
    task = _day5_large_task()
    task["params"]["api_key"] = "DAY5-SYNTHETIC-KEY"
    payload = task if method == "query_task" else {"items": [task], "total": 1}
    monkeypatch.setattr(api_client, method, lambda *args: response("success", data=payload))
    server = create_mcp_server(service)
    raw = await server.call_tool(method, params)
    result = BusinessResult.model_validate(raw.structured_content)
    assert _day5_chars(result) <= MAX_TOOL_RESULT_CHARS
    assert result.status == "partial" and result.truncated is True
    returned_task = result.data if method == "query_task" else result.data["items"][0]
    assert returned_task["task_id"] == "12345"
    assert "DAY5-SYNTHETIC-KEY" not in result.model_dump_json()


@DAY5
@pytest.mark.asyncio
async def test_day5_server_permissions_are_enforced(api_client, monkeypatch):
    # 工具仍然可见，但服务端权限关闭时不允许执行 API。
    service = BusinessService(api_client, can_read_tasks=False)
    calls = []
    monkeypatch.setattr(api_client, "query_task", lambda *args: calls.append(args))
    server = create_mcp_server(service)
    assert {tool.name for tool in await server.list_tools()} == {"query_task", "query_tasks_by_engine"}
    raw = await server.call_tool("query_task", {"task_id": "12345"})
    result = BusinessResult.model_validate(raw.structured_content)
    assert (result.status, result.error_code) == ("error", "FORBIDDEN")
    assert result.data is None and calls == []


@DAY5
@pytest.mark.asyncio
async def test_day5_upstream_owner_permission_is_not_bypassed(api_client):
    server = create_mcp_server(BusinessService(api_client))
    raw = await server.call_tool("query_task", {"task_id": "34567"})
    result = BusinessResult.model_validate(raw.structured_content)
    assert (result.status, result.error_code) == ("error", "FORBIDDEN")
    assert result.data is None


@DAY5
def test_day5_other_protocol_error_is_not_retried():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.RemoteProtocolError("synthetic protocol failure", request=request)

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(httpx.RemoteProtocolError):
            BusinessAPIClient(client, max_attempts=3).query_task("12345")
    assert len(requests) == 1


@DAY5
def test_day5_exhausted_retry_preserves_original_exception():
    requests = []
    error = httpx.ConnectError("synthetic connection failure")

    def handler(request):
        requests.append(request)
        raise error

    with httpx.Client(base_url="https://mock.example", transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(httpx.ConnectError) as captured:
            request_json_with_retry(client, "/tasks/12345", max_attempts=2)
    assert captured.value is error
    assert len(requests) == 2
