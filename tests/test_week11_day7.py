"""用相同 Mock 查询验收 Week11 的两条执行路径，并输出对照产物。"""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

from langgraph_learning.day5 import open_day5_graph
from langgraph_learning.day7 import QueryOutcome, compare_outcomes, run_agents_sdk_case, run_langgraph_case
from week6 import BusinessAPIClient, BusinessService, create_mock_http_client


DATASET = Path(os.environ.get(
    "WEEK11_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week11_langgraph",
))
DAY7 = pytest.mark.skipif(os.environ.get("RUN_WEEK11_DAY7") != "1",
                        reason="设置 RUN_WEEK11_DAY7=1 验收两条框架路径")


@pytest.fixture(scope="module")
def dataset():
    day1 = json.loads((DATASET / "day1_cases.json").read_text(encoding="utf-8"))
    day5 = json.loads((DATASET / "day5_cases.json").read_text(encoding="utf-8"))
    day7 = json.loads((DATASET / "day7_cases.json").read_text(encoding="utf-8"))
    return day7["cases"], {case["case_id"]: case for case in day1["cases"]}, day5["follow_up"]


def test_day7_references_same_business_cases(dataset):
    cases, references, follow_up = dataset
    assert len({case["case_id"] for case in cases}) == len(cases) == 3
    assert {case["kind"] for case in cases} == {"direct", "follow_up", "error"}
    for case in cases:
        reference = references[case["reference_case_id"]]
        assert reference["expected"]["status"] == ("error" if case["kind"] == "error" else "success")
        if case["kind"] == "follow_up":
            assert follow_up["day1_case_id"] == reference["case_id"]
            assert follow_up["valid_reply"].strip() == reference["expected"]["task_id"]


@DAY7
@pytest.mark.parametrize(
    ("graph_correct", "sdk_correct", "agrees"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, True)],
)
def test_day7_outcome_agreement_and_correctness_are_independent(graph_correct, sdk_correct, agrees):
    expected = QueryOutcome("success", "12345", "RUNNING", None)
    wrong = replace(expected, task_state="SUCCESS")
    assert compare_outcomes(expected, expected if graph_correct else wrong, expected if sdk_correct else wrong) == {
        "langgraph_matches_expected": graph_correct,
        "agents_sdk_matches_expected": sdk_correct,
        "frameworks_agree": agrees,
    }


def _write_report(directory, rows):
    data = {"mode": "mock_http_and_scripted_model", "cases": rows}
    (directory / "comparison_report.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    lines = [
        "# Week11 Day7 · LangGraph 与 Agents SDK 对照",
        "",
        "两条路径使用同一组 Mock API。Agents SDK 的模型动作为离线脚本。",
        "",
        "| 用例 | LangGraph 结果 / 逐轮查询次数 | Agents SDK 结果 / 逐轮查询次数 | 两侧都符合预期 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        graph_counts = ", ".join(str(len(turn["query_ids"])) for turn in row["langgraph"]["turns"])
        sdk_counts = ", ".join(str(len(turn["query_ids"])) for turn in row["agents_sdk"]["turns"])
        graph_result = row["langgraph"]["outcome"]
        sdk_result = row["agents_sdk"]["outcome"]
        graph_state = graph_result["task_state"] or graph_result["error_code"]
        sdk_state = sdk_result["task_state"] or sdk_result["error_code"]
        checks = row["checks"]
        passed = checks["langgraph_matches_expected"] and checks["agents_sdk_matches_expected"]
        lines.append(
            f"| {row['case_id']} | {graph_state} / {graph_counts} | {sdk_state} / {sdk_counts} "
            f"| {'是' if passed else '否'} |"
        )
    lines.extend([
        "",
        "| 对比维度 | LangGraph | Agents SDK |",
        "|---|---|---|",
        "| 编排 | 节点和条件边决定追问与查询顺序 | Runner 执行模型发出的工具调用，脚本模型决定何时追问或查任务 |",
        "| 持久化 | Checkpointer 保存业务 State 和待执行节点 | 本用例的 Session 保存消息历史，下一轮 Runner 读取历史 |",
        "| 人工补参 | 从同一线程的中断点恢复 | 追问结束第一轮，补答启动同一会话的下一轮 |",
        "| 测试 | 检查节点更新、暂停点和最终 State | 检查 Run 项、工具参数及模型能否取得上一轮历史 |",
        "",
        "如果业务需要显式定义并恢复执行步骤，本实验中的 LangGraph 流程可直接表达这些约束。"
        "如果业务以对话和模型选择工具为主，本实验中的 Runner 与 Session 已覆盖查询和补参。",
        "",
        "完整轨迹见 [JSON](comparison_report.json)，状态图见 [Mermaid](state_graph.mmd)。",
    ])
    (directory / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@DAY7
@pytest.mark.asyncio
async def test_day7_both_frameworks_use_same_mock_queries(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    cases, references, follow_up = dataset
    graph_db, sdk_db = tmp_path / "langgraph.sqlite", tmp_path / "agents_sdk.sqlite"
    rows = []
    for case in cases:
        reference = references[case["reference_case_id"]]
        needs_reply = case["kind"] == "follow_up"
        question = (follow_up if needs_reply else reference)["input"]["question"]
        reply = follow_up["valid_reply"] if needs_reply else None
        expected = QueryOutcome(**{key: reference["expected"][key] for key in QueryOutcome.__annotations__})
        with create_mock_http_client(reference["subject_id"]) as client:
            query_task = BusinessService(BusinessAPIClient(client)).query_task
            graph = run_langgraph_case(question, reply, case_id=case["case_id"], db_path=graph_db,
                                       query_task=query_task)
            sdk = await run_agents_sdk_case(question, reply, case_id=case["case_id"], db_path=sdk_db,
                                            query_task=query_task)
            if not rows:
                with open_day5_graph(graph_db, query_task) as compiled:
                    diagram = compiled.get_graph().draw_mermaid()
                (tmp_path / "state_graph.mmd").write_text(diagram + "\n", encoding="utf-8")
        rows.append({
            "case_id": case["case_id"], "kind": case["kind"], "case_source": reference["case_id"],
            "input": {"question": question, "reply": reply}, "expected": asdict(expected),
            "langgraph": asdict(graph), "agents_sdk": asdict(sdk),
            "checks": compare_outcomes(expected, graph.outcome, sdk.outcome),
        })

    _write_report(tmp_path, rows)
    print(f"Week11 Day7 框架对照产物：{tmp_path}")

    for row in rows:
        case_id = row["case_id"]
        assert all(row["checks"].values()), case_id
        graph, sdk = row["langgraph"], row["agents_sdk"]
        queries = (row["expected"]["task_id"],)
        assert graph["turns"][-1]["query_ids"] == sdk["turns"][-1]["query_ids"] == queries, case_id
        assert sdk["turns"][-1]["steps"] == ("call:query_task", "function_call_output", "message"), case_id
        if row["kind"] == "follow_up":
            assert len(graph["turns"]) == len(sdk["turns"]) == 2, case_id
            assert graph["turns"][0]["query_ids"] == sdk["turns"][0]["query_ids"] == (), case_id
            assert graph["turns"][0]["steps"] == ("parse", "__interrupt__"), case_id
            assert graph["turns"][1]["steps"] == ("ask_task_id", "query", "answer"), case_id
            assert sdk["turns"][0]["steps"] == ("message",), case_id
            assert graph["clarification"] == sdk["clarification"] == follow_up["initial_message"], case_id
            assert graph["restored_question"] == sdk["restored_question"] == row["input"]["question"], case_id
        else:
            assert len(graph["turns"]) == len(sdk["turns"]) == 1, case_id
            assert graph["turns"][0]["steps"] == ("parse", "query", "answer"), case_id
