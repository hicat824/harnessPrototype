"""验收 Week11 Day5 的缺参暂停、人工补答与检查点恢复。"""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from langgraph.types import Command

from langgraph_learning.day4 import checkpoint_config
from langgraph_learning.day5 import open_day5_graph
from week6 import BusinessAPIClient, BusinessService, create_mock_http_client


DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
DAY5 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY5") != "1",
                         reason="设置 RUN_WEEK11_DAY5=1 验收 Day5 暂停和恢复")


@pytest.fixture(scope="module")
def dataset():
    day5 = json.loads((DATASET / "day5_cases.json").read_text(encoding="utf-8"))
    day1 = json.loads((DATASET / "day1_cases.json").read_text(encoding="utf-8"))
    return day5, {case["case_id"]: case for case in day1["cases"]}


def test_day5_dataset_reuses_day1_queries(dataset):
    data, cases = dataset
    direct, follow_up = data["direct"], data["follow_up"]
    assert len({direct["thread_id"], follow_up["thread_id"], data["unused_thread_id"]}) == 3
    assert direct["day1_case_id"] != follow_up["day1_case_id"]
    assert cases[direct["day1_case_id"]]["expected"]["status"] == "success"
    assert follow_up["valid_reply"].strip() == cases[follow_up["day1_case_id"]]["expected"]["task_id"]
    assert follow_up["invalid_replies"]


def _assert_paused(snapshot, case, message):
    assert snapshot.next == ("ask_task_id",)
    assert len(snapshot.interrupts) == 1
    assert snapshot.interrupts[0].value == {"kind": "task_id", "message": message}
    assert snapshot.values["question"] == case["input"]["question"]
    assert snapshot.values["task_id"] is None
    assert snapshot.values["query_result"] is None
    assert snapshot.values["answer"] is None


def _assert_answer(values, reference):
    expected = reference["expected"]
    assert values["task_id"] == expected["task_id"]
    assert values["query_result"].data["state"] == expected["task_state"]
    assert values["answer"] == expected["answer"]


@DAY5
def test_day5_question_with_task_id_skips_interrupt(tmp_path, dataset):
    data, cases = dataset
    case = data["direct"]
    reference = cases[case["day1_case_id"]]
    config = checkpoint_config(case["thread_id"])
    with create_mock_http_client(reference["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        query_task = Mock(wraps=service.query_task)
        with open_day5_graph(tmp_path / "tasks.sqlite", query_task) as graph:
            result = graph.invoke(reference["input"], config)
            assert "__interrupt__" not in result
            assert graph.get_state(config).next == ()
            _assert_answer(result, reference)
        query_task.assert_called_once_with(reference["expected"]["task_id"])


@DAY5
def test_day5_resume_runs_from_waiting_node(tmp_path, dataset):
    data, cases = dataset
    case = data["follow_up"]
    reference = cases[case["day1_case_id"]]
    config = checkpoint_config(case["thread_id"])
    with create_mock_http_client(reference["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        query_task = Mock(wraps=service.query_task)
        with open_day5_graph(tmp_path / "tasks.sqlite", query_task) as graph:
            result = graph.invoke(case["input"], config)
            paused = graph.get_state(config)
            assert result["__interrupt__"] == list(paused.interrupts)
            _assert_paused(paused, case, case["initial_message"])
            query_task.assert_not_called()

            updates = list(graph.stream(Command(resume=case["valid_reply"]), config, stream_mode="updates"))
            assert [name for update in updates for name in update] == ["ask_task_id", "query", "answer"]
            final = graph.get_state(config)
            assert final.next == ()
            assert final.values["question"] == case["input"]["question"]
            _assert_answer(final.values, reference)
        query_task.assert_called_once_with(reference["expected"]["task_id"])


@DAY5
def test_day5_invalid_replies_keep_workflow_waiting(tmp_path, dataset):
    data, cases = dataset
    case = data["follow_up"]
    reference = cases[case["day1_case_id"]]
    config = checkpoint_config(case["thread_id"])
    with create_mock_http_client(reference["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        query_task = Mock(wraps=service.query_task)
        with open_day5_graph(tmp_path / "tasks.sqlite", query_task) as graph:
            # 先跑完一个任务，验证同一线程开始新问题时清除了旧编号和结果。
            _assert_answer(graph.invoke(reference["input"], config), reference)
            query_task.reset_mock()
            graph.invoke(case["input"], config)
            _assert_paused(graph.get_state(config), case, case["initial_message"])
            for reply in case["invalid_replies"]:
                graph.invoke(Command(resume=reply), config)
                _assert_paused(graph.get_state(config), case, case["invalid_message"])
                query_task.assert_not_called()
            _assert_answer(graph.invoke(Command(resume=case["valid_reply"]), config), reference)
        query_task.assert_called_once_with(reference["expected"]["task_id"])


@DAY5
def test_day5_resumes_after_reopening_sqlite(tmp_path, dataset):
    data, cases = dataset
    case = data["follow_up"]
    reference = cases[case["day1_case_id"]]
    config = checkpoint_config(case["thread_id"])
    db_path = tmp_path / "tasks.sqlite"
    blocked_query = Mock(side_effect=AssertionError("暂停前不能查任务"))
    with open_day5_graph(db_path, blocked_query) as graph:
        graph.invoke(case["input"], config)
        _assert_paused(graph.get_state(config), case, case["initial_message"])
    blocked_query.assert_not_called()

    with create_mock_http_client(reference["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        query_task = Mock(wraps=service.query_task)
        with open_day5_graph(db_path, query_task) as graph:
            assert graph.get_state(checkpoint_config(data["unused_thread_id"])).values == {}
            _assert_paused(graph.get_state(config), case, case["initial_message"])
            resumed = graph.invoke(Command(resume=case["valid_reply"]), config)
            assert resumed["question"] == case["input"]["question"]
            assert graph.get_state(config).next == ()
            _assert_answer(resumed, reference)
        query_task.assert_called_once_with(reference["expected"]["task_id"])
