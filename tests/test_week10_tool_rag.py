"""Week10 离线验收。默认检查样例，按天启用 Day1 路由或 Day2 Context 验收。"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from week6 import BusinessAPIClient, BusinessResult, BusinessService, create_mock_http_client
from week10_tool_rag import (
    Day1Request, KnowledgeEvidence, build_day2_context, prepare_knowledge_query, run_day1,
)


DATA_ROOT = Path("/media/xumaozhou/data/AI/local_agent_test")
DATASET = Path(os.environ.get("WEEK10_DATASET", DATA_ROOT / "week10_tool_rag"))
KNOWLEDGE_ROOT = Path(os.environ.get("WEEK9_RAG_DATASET", DATA_ROOT / "week9_rag_quality"))
OUTPUT = Path(os.environ.get("WEEK10_DAY1_OUTPUT", DATASET / "runs/day1"))
DAY1 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY1") != "1",
                         reason="完整路由验收需设置 RUN_WEEK10_DAY1=1")
DAY2_OUTPUT = Path(os.environ.get("WEEK10_DAY2_OUTPUT", DATASET / "runs/day2"))
DAY2 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY2") != "1",
                         reason="完整 Context 验收需设置 RUN_WEEK10_DAY2=1")


@pytest.fixture(scope="module")
def scenarios():
    path = DATASET / "mock_scenarios.json"
    assert path.is_file(), f"找不到 Day1 数据，请检查 WEEK10_DATASET：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence(scenarios):
    config = scenarios["knowledge"]
    path = KNOWLEDGE_ROOT / config["source"]
    line_number = config["line"]
    text = path.read_text(encoding="utf-8").splitlines()[line_number - 1]
    return KnowledgeEvidence(config["doc_id"], f"{config['source']}#L{line_number}", text)


def test_day1_mock_sources_match_fixtures(scenarios, evidence):
    """直接核对 Week6 Mock 和 Week9 证据原文，独立于学生代码。"""
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        for probe in scenarios["task_probes"]:
            result = service.query_task(probe["task_id"])
            assert result.status == probe["status"]
            assert (result.data or {}).get("state") == probe["state"]
            assert result.error_code == probe["error_code"]
    assert all(topic in evidence.text for topic in scenarios["knowledge"]["topics"])


@DAY1
@pytest.mark.parametrize("state", ["SUCCESS", "FAILED", None])
def test_day1_combined_query_uses_the_observed_state(state):
    """改变上游返回值，检查组合路由是否跟随实际状态，以及无状态时是否停下。"""
    request = Day1Request("任务 12345 还是 RUNNING 吗？解释查到的状态。", True, True, "12345")
    query = prepare_knowledge_query(request, "tool_rag", state)
    if state is None:
        assert query is None
    else:
        assert isinstance(query, str) and state in query and "RUNNING" not in query


def _grade_case(case, service, knowledge, topics):
    # Agent 只接收 input。参考答案在本轮执行完成后用于评分。
    request = Day1Request(**case["input"])
    calls = []

    def query_task(task_id):
        calls.append({"tool": "query_task", "task_id": task_id})
        return service.query_task(task_id)

    def search_knowledge(query):
        calls.append({"tool": "search_knowledge", "query": query})
        return (knowledge,) if any(topic in query for topic in topics) else ()

    result = run_day1(request, query_task=query_task, search_knowledge=search_knowledge)
    expected = case["expected"]
    queries = [call["query"] for call in calls if call["tool"] == "search_knowledge"]
    expected_task_ids = [request.task_id] if "query_task" in expected["tool_order"] else []
    query_text = "\n".join(queries)
    checks = {
        "route": result.route == expected["route"],
        "outcome": (result.status, result.reason) == (expected["status"], expected["reason"]),
        "task_fact": (result.task_state, result.task_source) == (
            expected["task_state"], expected["task_source"],
        ),
        "knowledge_sources": [item.doc_id for item in result.evidence] == expected["doc_ids"],
        "tool_order": [call["tool"] for call in calls] == expected["tool_order"],
        "task_arguments": [call["task_id"] for call in calls if call["tool"] == "query_task"]
        == expected_task_ids,
        "knowledge_arguments": (
            (expected["route"] != "rag_only" or queries == [request.question])
            and all(term in query_text for term in expected.get("query_contains", []))
            and all(term not in query_text for term in expected.get("query_excludes", []))
        ),
    }
    return {"case_id": case["case_id"], "input": asdict(request), "result": asdict(result),
            "calls": calls, "checks": checks, "passed": all(checks.values())}


@DAY1
def test_day1_routes_and_stop_conditions(scenarios, evidence):
    """运行三个正常场景和两个停止场景，报告实际工具参数与调用顺序。"""
    data = json.loads((DATASET / "eval_cases.json").read_text(encoding="utf-8"))
    cases = [case for case in data["cases"] if case["day"] == 1]
    assert len(cases) == len({case["case_id"] for case in cases}) == 5
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        rows = [_grade_case(case, service, evidence, scenarios["knowledge"]["topics"]) for case in cases]

    failed = {row["case_id"]: [key for key, passed in row["checks"].items() if not passed]
              for row in rows if not row["passed"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "routing_report.json"
    path.write_text(json.dumps({"passed": not failed, "cases": rows}, ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")
    print(f"Day1 路由验收报告：{path}")
    assert not failed, failed


@pytest.fixture(scope="module")
def day2_cases():
    path = DATASET / "day2_context_cases.json"
    assert path.is_file(), f"找不到 Day2 样例，请检查 WEEK10_DATASET：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def day2_evidence(day2_cases):
    """两条 Week9 原文夹住一条偏长的合成片段，检验跳过中间项后继续选取。"""
    short_chunks = []
    for entry in day2_cases["knowledge"]:
        line = entry["line"]
        source = entry["source"]
        text = (KNOWLEDGE_ROOT / source).read_text(encoding="utf-8").splitlines()[line - 1]
        short_chunks.append(KnowledgeEvidence(entry["doc_id"], f"{source}#L{line}", text))
    stress = day2_cases["budget_stress_chunk"]
    long_chunk = KnowledgeEvidence(stress["doc_id"], "day2_context_cases.json#budget_stress_chunk",
                                   stress["fragment"] * stress["repeat"])
    return (short_chunks[0], long_chunk, short_chunks[1])


@pytest.fixture(scope="module")
def day2_task_results(scenarios, day2_cases):
    """经 Week6 Mock 服务查一次可见任务和一次不存在的任务。"""
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        return (service.query_task(day2_cases["task_id"]),
                service.query_task(day2_cases["missing_task_id"]))


def test_day2_fixture_sources(day2_cases, day2_evidence, day2_task_results):
    """检查样例来源以及长短差异，无需先实现填空。"""
    success, missing = day2_task_results
    assert success.status == "success" and success.data["state"] == "RUNNING"
    assert missing.status == "error" and missing.error_code == "NOT_FOUND"
    first, long_chunk, last = day2_evidence
    assert "RUNNING" in first.text and "实时 API" in last.text
    assert len(long_chunk.text) > day2_cases["budgets"]["tight"]


@DAY2
def test_day2_task_summary_keeps_verified_fields(day2_task_results):
    """保留业务结果和上游截断标记，从成功结果投影可引用的事实。"""
    success, _ = day2_task_results
    upstream = success.model_copy(update={"truncated": True})
    before = upstream.model_dump()
    context = build_day2_context(upstream, ())
    payload = json.loads(context.text)
    assert payload == {
        "task_query": {
            "status": "success", "error_code": None, "source_truncated": True,
            "fact": {
                "evidence_id": "T1", "task_id": "12345", "state": "RUNNING",
                "source": upstream.source, "fetched_at": upstream.fetched_at,
            },
        },
        "knowledge": [], "knowledge_truncated": False,
    }
    assert context.omitted_ids == () and upstream.model_dump() == before


@DAY2
def test_day2_missing_or_invalid_task_fact_has_no_citation(day2_task_results, day2_evidence):
    """没查任务、查不到任务和缺有效字段时，仍可保留各自允许的证据。"""
    _, missing = day2_task_results
    evidence = day2_evidence[:1]
    rag_only = json.loads(build_day2_context(None, evidence).text)
    assert rag_only["task_query"] is None
    assert [row["evidence_id"] for row in rag_only["knowledge"]] == ["K1"]

    failed = json.loads(build_day2_context(missing, ()).text)
    assert failed["task_query"] == {
        "status": "error", "error_code": "NOT_FOUND", "source_truncated": False, "fact": None,
    }

    for data in (
        {"state": "RUNNING"},
        {"task_id": "12345", "state": " "},
        {"task_id": None, "state": "RUNNING"},
        {"task_id": "12345", "state": 1},
    ):
        malformed = BusinessResult(status="success", data=data)
        payload = json.loads(build_day2_context(malformed, ()).text)
        assert payload["task_query"]["fact"] is None


@DAY2
def test_day2_packs_complete_evidence_within_budget(day2_cases, day2_evidence, day2_task_results):
    """同时检查完整、限长、恰好放下和最小结构超限，输出可复核的预算报告。"""
    success, _ = day2_task_results
    limits = day2_cases["budgets"]
    full = build_day2_context(success, day2_evidence, max_chars=limits["full"])
    tight = build_day2_context(success, day2_evidence, max_chars=limits["tight"])
    exact = build_day2_context(success, day2_evidence, max_chars=len(full.text))
    baseline = build_day2_context(success, ())
    with pytest.raises(ValueError, match="预算"):
        build_day2_context(success, (), max_chars=len(baseline.text) - 1)

    expected_chunks = [
        {"evidence_id": f"K{index}", **asdict(item)}
        for index, item in enumerate(day2_evidence, start=1)
    ]
    rows = []
    for name, context, expected, omitted in (
        ("full", full, expected_chunks, ()),
        ("tight", tight, [expected_chunks[0], expected_chunks[2]], ("K2",)),
        ("exact", exact, expected_chunks, ()),
    ):
        payload = json.loads(context.text)
        assert len(context.text) <= context.max_chars
        assert payload["task_query"] == json.loads(baseline.text)["task_query"]
        assert payload["knowledge"] == expected
        assert payload["knowledge_truncated"] is bool(omitted) and context.omitted_ids == omitted
        rows.append({"case": name, "max_chars": context.max_chars, "used_chars": len(context.text),
                     "omitted_ids": omitted, "context": payload})
    assert len(exact.text) == exact.max_chars

    DAY2_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DAY2_OUTPUT / "context_report.json"
    path.write_text(json.dumps({"passed": True, "cases": rows}, ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")
    print(f"Day2 Context 验收报告：{path}")
