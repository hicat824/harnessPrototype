"""Week10 Day1 离线验收。默认检查样例，用 RUN_WEEK10_DAY1=1 验收完整路由。"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from week6 import BusinessAPIClient, BusinessService, create_mock_http_client
from week10_tool_rag import Day1Request, KnowledgeEvidence, prepare_knowledge_query, run_day1


DATA_ROOT = Path("/media/xumaozhou/data/AI/local_agent_test")
DATASET = Path(os.environ.get("WEEK10_DATASET", DATA_ROOT / "week10_tool_rag"))
KNOWLEDGE_ROOT = Path(os.environ.get("WEEK9_RAG_DATASET", DATA_ROOT / "week9_rag_quality"))
OUTPUT = Path(os.environ.get("WEEK10_DAY1_OUTPUT", DATASET / "runs/day1"))
DAY1 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY1") != "1",
                         reason="完整路由验收需设置 RUN_WEEK10_DAY1=1")


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
