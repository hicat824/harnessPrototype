"""验收 Week11 Day4 的快照读取、线程隔离和进程间持久化。"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from langgraph_learning.day4 import checkpoint_config, open_day4_graph
from week6 import BusinessAPIClient, BusinessResult, BusinessService, create_mock_http_client


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
DAY4 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY4") != "1",
                         reason="设置 RUN_WEEK11_DAY4=1 验收 Day4 快照")


@pytest.fixture(scope="module")
def dataset():
    day4 = json.loads((DATASET / "day4_cases.json").read_text(encoding="utf-8"))
    day1 = json.loads((DATASET / "day1_cases.json").read_text(encoding="utf-8"))
    return day4, {case["case_id"]: case for case in day1["cases"]}


def test_day4_dataset_references_day1_cases(dataset):
    data, cases = dataset
    first, second = data["threads"]
    assert len({first["thread_id"], second["thread_id"], data["unused_thread_id"]}) == 3
    assert [len(thread["day1_case_ids"]) for thread in data["threads"]] == [2, 1]
    for thread in data["threads"]:
        requests = [cases[case_id] for case_id in thread["day1_case_ids"]]
        assert len({case["subject_id"] for case in requests}) == 1
        assert all(case["expected"]["status"] == "success" for case in requests)
        assert len({case["expected"]["task_id"] for case in requests}) == len(requests)


@DAY4
def test_day4_config_identifies_thread_and_optional_checkpoint():
    assert checkpoint_config("task-a") == {"configurable": {"thread_id": "task-a"}}
    assert checkpoint_config("task-a", "checkpoint-1") == {
        "configurable": {"thread_id": "task-a", "checkpoint_id": "checkpoint-1"},
    }
    with pytest.raises(ValueError, match="thread_id"):
        checkpoint_config(" ")
    with pytest.raises(ValueError, match="checkpoint_id"):
        checkpoint_config("task-a", "")


def _assert_completed_snapshot(snapshot, case):
    expected = case["expected"]
    assert snapshot.next == ()
    assert snapshot.values["question"] == case["input"]["question"]
    assert snapshot.values["task_id"] == expected["task_id"]
    assert snapshot.values["answer"] == expected["answer"]
    query_result = snapshot.values["query_result"]
    assert isinstance(query_result, BusinessResult)
    assert query_result.status == expected["status"]
    assert query_result.data["state"] == expected["task_state"]


@DAY4
def test_day4_latest_history_and_specific_checkpoint(tmp_path, dataset):
    data, cases = dataset
    thread = data["threads"][0]
    first, second = [cases[case_id] for case_id in thread["day1_case_ids"]]
    config = checkpoint_config(thread["thread_id"])
    with create_mock_http_client(first["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        query_task = Mock(wraps=service.query_task)
        with open_day4_graph(tmp_path / "tasks.sqlite", query_task) as graph:
            graph.invoke(first["input"], config)
            first_snapshot = graph.get_state(config)
            old_id = first_snapshot.config["configurable"]["checkpoint_id"]
            graph.invoke(second["input"], config)

            _assert_completed_snapshot(graph.get_state(config), second)
            history = list(graph.get_state_history(config))
            completed = [snapshot for snapshot in history if not snapshot.next]
            assert [s.values["task_id"] for s in completed] == [
                second["expected"]["task_id"], first["expected"]["task_id"],
            ]
            assert any(snapshot.next == ("query",) for snapshot in history)
            old = graph.get_state(checkpoint_config(thread["thread_id"], old_id))
            _assert_completed_snapshot(old, first)
            _assert_completed_snapshot(graph.get_state(config), second)

        assert query_task.call_args_list == [
            call(first["expected"]["task_id"]), call(second["expected"]["task_id"]),
        ]


@DAY4
def test_day4_separates_threads_in_one_database(tmp_path, dataset):
    data, cases = dataset
    db_path = tmp_path / "tasks.sqlite"
    for thread in data["threads"]:
        case = cases[thread["day1_case_ids"][0]]
        with create_mock_http_client(case["subject_id"]) as client:
            service = BusinessService(BusinessAPIClient(client))
            with open_day4_graph(db_path, service.query_task) as graph:
                graph.invoke(case["input"], checkpoint_config(thread["thread_id"]))

    query_task = Mock(side_effect=AssertionError("读取快照不应查询业务接口"))
    with open_day4_graph(db_path, query_task) as graph:
        for thread in data["threads"]:
            snapshot = graph.get_state(checkpoint_config(thread["thread_id"]))
            _assert_completed_snapshot(snapshot, cases[thread["day1_case_ids"][0]])
        unused = graph.get_state(checkpoint_config(data["unused_thread_id"]))
        assert unused.values == {}
    query_task.assert_not_called()


@DAY4
def test_day4_reads_checkpoint_after_writer_process_exits(tmp_path, dataset):
    data, cases = dataset
    thread = data["threads"][0]
    case = cases[thread["day1_case_ids"][0]]
    config = checkpoint_config(thread["thread_id"])
    db_path = tmp_path / "tasks.sqlite"
    writer = """
import json
import sys
from langgraph_learning.day4 import open_day4_graph
from week6 import BusinessAPIClient, BusinessService, create_mock_http_client

db_path, subject_id, inputs, config = sys.argv[1:]
with create_mock_http_client(subject_id) as client:
    service = BusinessService(BusinessAPIClient(client))
    with open_day4_graph(db_path, service.query_task) as graph:
        graph.invoke(json.loads(inputs), json.loads(config))
"""
    subprocess.run(
        [sys.executable, "-B", "-c", writer, str(db_path), case["subject_id"],
         json.dumps(case["input"]), json.dumps(config)],
        cwd=REPO_ROOT, check=True, timeout=20,
    )

    assert db_path.is_file()
    query_task = Mock(side_effect=AssertionError("重新打开数据库仅用于回读"))
    with open_day4_graph(db_path, query_task) as graph:
        _assert_completed_snapshot(graph.get_state(config), case)
    query_task.assert_not_called()
