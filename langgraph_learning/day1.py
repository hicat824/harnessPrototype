"""Week11 Day1：用 LangGraph 编排只读任务状态查询。"""

import re
from collections.abc import Callable
from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired

from week6 import BusinessResult


class TaskQueryState(TypedDict):
    """一次图执行中的业务状态，供节点逐步读取和更新。

    调用方只传 question。parse 写入 task_id，query 写入 query_result，
    answer 写入 answer。后面三个字段在对应节点执行前尚未生成。
    """

    question: str
    task_id: NotRequired[str]
    query_result: NotRequired[BusinessResult]
    answer: NotRequired[str]


def parse_task(state: TaskQueryState) -> dict[str, str]:
    """从 question 读取一个任务编号，返回本节点写入的 task_id。"""
    task_ids = re.findall(r"任务(?:编号)?\s*[:：]?\s*([0-9]+)(?![0-9A-Za-z_-])", state["question"])
    if len(task_ids) != 1:
        raise ValueError("Day1 的问题需要在「任务」或「任务编号」后提供一个数字编号")
    return {"task_id": task_ids[0]}


def answer_task(state: TaskQueryState) -> dict[str, str]:
    """根据 query 写入的 BusinessResult 生成可核对的固定答复。"""
    task_id = state["task_id"]
    result = state["query_result"]
    task_state = (result.data or {}).get("state")
    # 外层 status 表示 API 查询结果。查询成功的任务自身可以处于 FAILED。
    if result.status == "success" and isinstance(task_state, str) and task_state.strip():
        return {"answer": f"任务 {task_id} 当前状态为 {task_state}。"}
    reason = result.error_code or "缺少状态数据"
    return {"answer": f"无法确认任务 {task_id} 的当前状态（{reason}）。"}


def build_day1_graph(
    query_task: Callable[[str], BusinessResult],
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建并返回可用 invoke 和 stream 运行的 Day1 状态图。

    query_task 由调用方提供，接收 parse 提取的编号并返回 Week6 BusinessResult。
    测试会注入 Mock 业务查询。Day4 可以传入 checkpointer 保存线程状态快照。

    图按 START → parse → query → answer → END 执行。
    每个节点返回自己新增或更新的字段，LangGraph 会把这些字段合并进该次运行的 State。
    """
    def query(state: TaskQueryState) -> dict[str, BusinessResult]:
        return {"query_result": query_task(state["task_id"])}

    builder = StateGraph(TaskQueryState)
    builder.add_node("parse", parse_task)
    builder.add_node("query", query)
    builder.add_node("answer", answer_task)

    builder.add_edge(START, "parse")
    builder.add_edge("parse", "query")
    builder.add_edge("query", "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=checkpointer)
    return graph
