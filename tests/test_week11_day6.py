"""验收 Week11 Day6 的恢复决策、权限复核和工具调用次数。"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from langgraph.types import Command

from langgraph_learning.day4 import checkpoint_config
from langgraph_learning.day6 import decide_resume_action, open_day6_graph
from week6 import BusinessAPIClient, BusinessResult, BusinessService, create_mock_http_client


DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
DAY6 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY6") != "1",
                        reason="设置 RUN_WEEK11_DAY6=1 验收 Day6 恢复决策")


@pytest.fixture(scope="module")
def dataset():
    data = json.loads((DATASET / "day6_cases.json").read_text(encoding="utf-8"))
    day1 = json.loads((DATASET / "day1_cases.json").read_text(encoding="utf-8"))
    references = {case["case_id"]: case for case in day1["cases"]}
    return data, references[data["day1_case_id"]]


def test_day6_dataset_reuses_day1_success(dataset):
    data, reference = dataset
    assert reference["expected"]["status"] == "success"
    assert data["cache_ttl_seconds"] > 0
    cases = data["cases"]
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["expected"]["action"] for case in cases} == {"reject", "forbidden", "reuse", "refresh"}
    assert any(case["elapsed_seconds"] == data["cache_ttl_seconds"] for case in cases)


@DAY6
def test_day6_resume_policy_boundaries():
    # 先核对拒绝和无权的优先级，再独立覆盖缓存有效期的边界。
    cases = (
        (False, False, 75, "reject"),
        (True, False, 0, "forbidden"),
        (True, True, 0, "reuse"),
        (True, True, 59.99, "reuse"),
        (True, True, 60, "refresh"),
        (True, True, -1, "refresh"),
    )
    for approved, allowed, age, expected in cases:
        assert decide_resume_action(
            approved=approved, allowed=allowed, cache_age_seconds=age, cache_ttl_seconds=60,
        ) == expected


def _assert_paused(snapshot, task_id, initial_state):
    assert snapshot.next == ("review",)
    assert len(snapshot.interrupts) == 1
    assert snapshot.interrupts[0].value == {"kind": "confirm_task_result", "task_id": task_id}
    assert snapshot.values["status"] == "pending_review"
    assert snapshot.values["query_result"].data["state"] == initial_state
    assert snapshot.values["answer"] is None


@DAY6
def test_day6_resume_trajectory_and_query_counts(tmp_path, dataset):
    data, reference = dataset
    task_id = reference["expected"]["task_id"]
    initial_state = reference["expected"]["task_state"]
    with create_mock_http_client(reference["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        for case in data["cases"]:
            case_id = case["case_id"]
            expected = case["expected"]
            config = checkpoint_config(case_id)
            clock = Mock(return_value=1000.0)
            timeline = Mock()
            can_read_task = timeline.authorize
            can_read_task.return_value = True
            query_task = timeline.query

            def fetch(task_id):
                # 首查复用 Day1 的 Mock API。恢复后可模拟状态变化或上游超时。
                if query_task.call_count == 2 and case.get("refresh_error"):
                    return BusinessResult(status="error", error_code=case["refresh_error"])
                result = service.query_task(task_id)
                if query_task.call_count == 2 and case.get("refreshed_task_state"):
                    return result.model_copy(update={
                        "data": {**result.data, "state": case["refreshed_task_state"]},
                    })
                return result

            query_task.side_effect = fetch
            with open_day6_graph(
                tmp_path / f"{case_id}.sqlite", query_task, can_read_task,
                cache_ttl_seconds=data["cache_ttl_seconds"], clock=clock,
            ) as graph:
                paused = graph.invoke({"task_id": task_id}, config)
                assert "__interrupt__" in paused, case_id
                _assert_paused(graph.get_state(config), task_id, initial_state)
                assert timeline.mock_calls == [call.authorize(task_id), call.query(task_id)], case_id

                can_read_task.return_value = case["allowed_after_resume"]
                clock.return_value += case["elapsed_seconds"]
                updates = list(graph.stream(Command(resume=case["approved"]), config, stream_mode="updates"))
                assert [name for update in updates for name in update] == expected["resume_nodes"], case_id
                final = graph.get_state(config)

            assert final.next == (), case_id
            values = final.values
            assert values["resume_action"] == expected["action"], case_id
            assert (values["status"], values["reason"]) == (expected["status"], expected["reason"]), case_id
            assert can_read_task.call_count == expected["permission_checks"], case_id
            assert query_task.call_count == expected["query_calls"], case_id
            # 顺序同样重要，特别是补查一定发生在恢复时的权限校验之后。
            if expected["query_calls"] == 2:
                assert timeline.mock_calls[-2:] == [call.authorize(task_id), call.query(task_id)], case_id
            if expected["task_state"] is None:
                assert values["answer"] is values["query_result"] is values["cached_at"] is None, case_id
            else:
                task_state = expected["task_state"]
                assert values["query_result"].data["state"] == task_state, case_id
                assert values["answer"] == f"任务 {task_id} 当前状态为 {task_state}。", case_id
                expected_cached_at = clock.return_value if expected["action"] == "refresh" else 1000.0
                assert values["cached_at"] == expected_cached_at, case_id


@DAY6
def test_day6_initial_denial_never_queries_or_pauses(tmp_path, dataset):
    _, reference = dataset
    task_id = reference["expected"]["task_id"]
    query_task, can_read_task = Mock(), Mock(return_value=False)
    config = checkpoint_config("initial-denial")
    with open_day6_graph(tmp_path / "tasks.sqlite", query_task, can_read_task) as graph:
        result = graph.invoke({"task_id": task_id}, config)
        assert "__interrupt__" not in result
        assert graph.get_state(config).next == ()
        assert (result["status"], result["reason"]) == ("error", "FORBIDDEN")
        assert result["query_result"] is result["cached_at"] is result["answer"] is None
    can_read_task.assert_called_once_with(task_id)
    query_task.assert_not_called()


@DAY6
def test_day6_reopens_sqlite_without_repeating_fresh_query(tmp_path, dataset):
    data, reference = dataset
    expected = reference["expected"]
    task_id = expected["task_id"]
    config = checkpoint_config("reopen")
    db_path = tmp_path / "tasks.sqlite"
    clock = Mock(return_value=1000.0)
    can_read_task = Mock(return_value=True)
    with create_mock_http_client(reference["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        query_task = Mock(wraps=service.query_task)
        with open_day6_graph(db_path, query_task, can_read_task,
                             cache_ttl_seconds=data["cache_ttl_seconds"], clock=clock) as graph:
            graph.invoke({"task_id": task_id}, config)
            _assert_paused(graph.get_state(config), task_id, expected["task_state"])

    # 模拟调用端退出后重新连接。第二次运行从 SQLite 读取暂停状态。
    clock.return_value += data["cache_ttl_seconds"] / 2
    resume_query = Mock(side_effect=AssertionError("仍在有效期内，不应重新查询"))
    with open_day6_graph(db_path, resume_query, can_read_task,
                         cache_ttl_seconds=data["cache_ttl_seconds"], clock=clock) as graph:
        _assert_paused(graph.get_state(config), task_id, expected["task_state"])
        result = graph.invoke(Command(resume=True), config)
        assert result["resume_action"] == "reuse"
        assert result["answer"] == expected["answer"]
        assert graph.get_state(config).next == ()
    query_task.assert_called_once_with(task_id)
    resume_query.assert_not_called()
    assert can_read_task.call_args_list == [call(task_id), call(task_id)]
