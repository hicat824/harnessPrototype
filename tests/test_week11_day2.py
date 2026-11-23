"""验收 Week11 Day2 的选路、检索回环与流程退出。"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from langgraph_learning.day2 import build_day2_graph, route_after_plan, route_after_search
from week6 import BusinessAPIClient, BusinessService, create_mock_http_client
from week10_tool_rag import Day1Request, KnowledgeEvidence, select_route


DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
KNOWLEDGE_ROOT = Path(os.environ.get(
    "WEEK9_RAG_DATASET", DATASET.parent / "week9_rag_quality",
))
DAY2 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY2") != "1",
                         reason="设置 RUN_WEEK11_DAY2=1 验收 Day2 条件分支")


@pytest.fixture(scope="module")
def dataset():
    return json.loads((DATASET / "day2_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence(dataset):
    item = dataset["knowledge"]
    line = item["line"]
    text = (KNOWLEDGE_ROOT / item["path"]).read_text(encoding="utf-8").splitlines()[line - 1]
    return KnowledgeEvidence(item["doc_id"], f"{item['path']}#L{line}", text)


def test_day2_data_reuses_week6_and_week9(dataset, evidence):
    cases = dataset["cases"]
    assert len(cases) == len({case["case_id"] for case in cases}) == 9
    assert {case["expected"]["route"] for case in cases} == {"tool_only", "rag_only", "tool_rag"}
    assert "RUNNING" in evidence.text and "FAILED" in evidence.text
    for case in cases:
        expected = case["expected"]
        request = Day1Request(**case["input"])
        assert select_route(request) == expected["route"], case["case_id"]
        if expected["query_ids"]:
            with create_mock_http_client(case["subject_id"]) as client:
                service = BusinessService(BusinessAPIClient(client))
                result = service.query_task(expected["query_ids"][0])
            assert (result.data or {}).get("state") == expected["task_state"], case["case_id"]
            if expected["status"] == "error":
                assert result.error_code == expected["reason"], case["case_id"]


@DAY2
def test_day2_plan_branch_uses_intent_and_task_id():
    for task_id in (None, "", " "):
        request = Day1Request("查询当前状态并解释", True, True, task_id)
        assert route_after_plan({"request": request, "route": "tool_rag"}) == "finish"
    rule = Day1Request("以 12345 为例讲解状态枚举", False, True, "12345")
    assert route_after_plan({"request": rule, "route": "rag_only"}) == "prepare_search"
    query = Day1Request("查任务 12345", True, False, "12345")
    assert route_after_plan({"request": query, "route": "tool_only"}) == "query"


@DAY2
def test_day2_search_branch_stops_at_evidence_or_exhaustion(evidence):
    request = Day1Request("查状态并解释", True, True, "12345")
    state = {"request": request, "search_queries": ("RUNNING 的含义", "RUNNING"),
             "search_attempts": 1, "evidence": ()}
    assert route_after_search(state) == "search"
    assert route_after_search({**state, "search_attempts": 2}) == "finish"
    assert route_after_search({**state, "evidence": (evidence,)}) == "finish"


@DAY2
def test_day2_graph_executes_expected_paths_and_exits(dataset, evidence):
    for case in dataset["cases"]:
        case_id = case["case_id"]
        request = Day1Request(**case["input"])
        expected = case["expected"]
        # 按外部 Mock 配置造召回结果。参考答案只在运行结束后用于断言。
        hits = set(case["mock_search_hits"])
        search_knowledge = Mock(side_effect=lambda query: (evidence,) if query in hits else ())
        with create_mock_http_client(case["subject_id"]) as client:
            service = BusinessService(BusinessAPIClient(client))
            query_task = Mock(wraps=service.query_task)
            graph = build_day2_graph(query_task, search_knowledge)
            updates = list(graph.stream({"request": request}, stream_mode="updates",
                                        config={"recursion_limit": 12}))
        assert [name for update in updates for name in update] == expected["path"], case_id
        assert query_task.call_args_list == [call(task_id) for task_id in expected["query_ids"]], case_id
        assert search_knowledge.call_args_list == [call(query) for query in expected["search_queries"]], case_id
        result = updates[-1]["finish"]["result"]
        assert (result.route, result.status, result.reason, result.task_state) == (
            expected["route"], expected["status"], expected["reason"], expected["task_state"],
        ), case_id
        assert [item.doc_id for item in result.evidence] == expected["doc_ids"], case_id
