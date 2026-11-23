"""用 Week6 Mock API 验收 Week11 Day1 的 LangGraph 状态图。"""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from langgraph_learning.day1 import answer_task, build_day1_graph, parse_task
from week6 import BusinessAPIClient, BusinessService, create_mock_http_client


DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
DAY1 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY1") != "1",
                         reason="设置 RUN_WEEK11_DAY1=1 验收 Day1 状态图")


@pytest.fixture(scope="module")
def cases():
    return json.loads((DATASET / "day1_cases.json").read_text(encoding="utf-8"))["cases"]


def test_day1_samples_match_existing_mock_api(cases):
    """验证题目、Week6 Mock API 和固定答复是否一致。"""
    assert len(cases) == len({case["case_id"] for case in cases}) == 4
    for case in cases:
        expected = case["expected"]
        with create_mock_http_client(case["subject_id"]) as client:
            service = BusinessService(BusinessAPIClient(client))
            parsed = parse_task(case["input"])
            assert parsed == {"task_id": expected["task_id"]}, case["case_id"]
            result = service.query_task(parsed["task_id"])
        assert result.status == expected["status"], case["case_id"]
        assert result.error_code == expected["error_code"], case["case_id"]
        assert (result.data or {}).get("state") == expected["task_state"], case["case_id"]
        assert answer_task({**case["input"], **parsed, "query_result": result}) == {
            "answer": expected["answer"],
        }, case["case_id"]


@DAY1
def test_day1_invoke_returns_merged_state(cases):
    for case in cases:
        inputs = case["input"].copy()
        expected = case["expected"]
        with create_mock_http_client(case["subject_id"]) as client:
            service = BusinessService(BusinessAPIClient(client))
            query_task = Mock(wraps=service.query_task)
            graph = build_day1_graph(query_task)
            final = graph.invoke(inputs)
            query_task.assert_called_once_with(expected["task_id"])
        assert inputs == case["input"], case["case_id"]
        assert final["question"] == inputs["question"], case["case_id"]
        assert final["task_id"] == expected["task_id"], case["case_id"]
        assert final["query_result"].status == expected["status"], case["case_id"]
        assert final["query_result"].error_code == expected["error_code"], case["case_id"]
        assert (final["query_result"].data or {}).get("state") == expected["task_state"], case["case_id"]
        assert final["answer"] == expected["answer"], case["case_id"]


@DAY1
def test_day1_stream_shows_partial_updates_in_node_order(cases):
    case = cases[0]
    with create_mock_http_client(case["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        graph = build_day1_graph(service.query_task)
        updates = list(graph.stream(case["input"], stream_mode="updates"))
    assert [tuple(update) for update in updates] == [("parse",), ("query",), ("answer",)]
    parse, query, answer = updates
    assert parse == {"parse": {"task_id": case["expected"]["task_id"]}}
    assert set(query["query"]) == {"query_result"}
    assert query["query"]["query_result"].status == "success"
    assert answer == {"answer": {"answer": case["expected"]["answer"]}}


@DAY1
def test_day1_each_invocation_uses_its_own_input(cases):
    first, second = cases[:2]
    assert first["subject_id"] == second["subject_id"]
    with create_mock_http_client(first["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        graph = build_day1_graph(service.query_task)
        first_result = graph.invoke(first["input"])
        second_result = graph.invoke(second["input"])
    assert first_result["answer"] == first["expected"]["answer"]
    assert second_result["task_id"] == second["expected"]["task_id"]
    assert second_result["answer"] == second["expected"]["answer"]
