"""验收 Week11 Day3 的失败处理、有限重试和预算边界。"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from langgraph.errors import GraphRecursionError

from langgraph_learning.day3 import build_day3_graph, reserve_tool_call, should_retry_query
from week6 import BusinessAPIClient, BusinessResult, BusinessService, create_mock_http_client
from week10_tool_rag import Day1Request, KnowledgeEvidence


DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
KNOWLEDGE_ROOT = Path(os.environ.get(
    "WEEK9_RAG_DATASET", DATASET.parent / "week9_rag_quality",
))
DAY3 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY3") != "1",
                         reason="设置 RUN_WEEK11_DAY3=1 验收 Day3 失败处理")


@pytest.fixture(scope="module")
def dataset():
    return json.loads((DATASET / "day3_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence(dataset):
    item = dataset["knowledge"]
    line = item["line"]
    text = (KNOWLEDGE_ROOT / item["path"]).read_text(encoding="utf-8").splitlines()[line - 1]
    return KnowledgeEvidence(item["doc_id"], f"{item['path']}#L{line}", text)


def test_day3_mock_sources_and_cases(dataset, evidence):
    cases = dataset["cases"]
    assert len(cases) == len({case["case_id"] for case in cases}) == 9
    assert "RUNNING" in evidence.text
    with create_mock_http_client("alice") as client:
        service = BusinessService(BusinessAPIClient(client, max_attempts=1))
        assert service.query_task("22222").error_code == "TIMEOUT"
        assert service.query_task("33333").error_code == "FORBIDDEN"
        assert service.query_task("12345").data["state"] == "RUNNING"


@DAY3
def test_day3_retries_only_timeout_with_attempts_remaining():
    timeout = BusinessResult(status="error", error_code="TIMEOUT")
    assert should_retry_query(timeout, attempts=1, max_attempts=2) is True
    assert should_retry_query(timeout, attempts=2, max_attempts=2) is False
    for result in (
        BusinessResult(status="error", error_code="FORBIDDEN"),
        BusinessResult(status="error", error_code="NOT_FOUND"),
        BusinessResult(status="empty"),
        BusinessResult(status="success"),
    ):
        assert should_retry_query(result, attempts=1, max_attempts=2) is False


@DAY3
def test_day3_tool_budget_is_checked_before_each_call():
    assert reserve_tool_call(0, 1) == 1
    assert reserve_tool_call(1, 2) == 2
    assert reserve_tool_call(0, 0) is None
    assert reserve_tool_call(1, 1) is None
    assert reserve_tool_call(2, 1) is None


@DAY3
def test_day3_graph_retries_and_exits_with_expected_reason(dataset, evidence):
    for case in dataset["cases"]:
        case_id = case["case_id"]
        request = Day1Request(**case["input"])
        expected = case["expected"]
        mock_config = case["mock"]
        hits = set(mock_config["search_hits"])
        search_knowledge = Mock(side_effect=lambda query: (evidence,) if query in hits else ())
        with create_mock_http_client(case["subject_id"]) as client:
            service = BusinessService(BusinessAPIClient(client, max_attempts=1))
            timeouts = mock_config["leading_timeouts"]

            def query(task_id):
                nonlocal timeouts
                if timeouts > 0:
                    timeouts -= 1
                    return BusinessResult(status="error", error_code="TIMEOUT")
                return service.query_task(task_id)

            query_task = Mock(side_effect=query)
            graph = build_day3_graph(query_task, search_knowledge, **case["limits"])
            updates = list(graph.stream({"request": request}, stream_mode="updates"))

        assert [name for update in updates for name in update] == expected["path"], case_id
        assert query_task.call_args_list == [call(task_id) for task_id in expected["query_ids"]], case_id
        assert search_knowledge.call_args_list == [call(query) for query in expected["search_queries"]], case_id
        query_errors = []
        final = {}
        for update in updates:
            for node, fields in update.items():
                final.update(fields)
                if node == "query" and "query_result" in fields:
                    query_errors.append(fields["query_result"].error_code)
        assert query_errors == expected["query_errors"], case_id
        assert final["query_attempts"] == len(expected["query_ids"]), case_id
        assert final["tool_calls"] == expected["tool_calls"], case_id
        result = final["result"]
        assert (result.status, result.reason, result.task_state) == (
            expected["status"], expected["reason"], expected["task_state"],
        ), case_id
        assert [item.doc_id for item in result.evidence] == expected["doc_ids"], case_id


@DAY3
def test_day3_graph_step_limit_is_independent_of_tool_budget():
    request = Day1Request("查任务 12345 的状态并解释", True, True, "12345")
    query_task = Mock(return_value=BusinessResult(status="success", data={"state": "RUNNING"}))
    search_knowledge = Mock(return_value=())
    graph = build_day3_graph(query_task, search_knowledge, max_tool_calls=4, max_steps=2)

    with pytest.raises(GraphRecursionError):
        graph.invoke({"request": request})
    query_task.assert_called_once_with("12345")
    search_knowledge.assert_not_called()
