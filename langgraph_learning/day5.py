"""Week11 Day5：缺少任务编号时暂停，收到人工补答后继续只读查询。"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from typing_extensions import NotRequired

from langgraph_learning.day1 import answer_task, parse_task
from langgraph_learning.day4 import open_task_checkpointer
from week6 import BusinessResult


class Day5State(TypedDict):
    """初始输入是 question，State 还记录任务编号、追问文案和查询结果。"""

    question: str
    task_id: NotRequired[str | None]
    pending_question: NotRequired[str | None]
    query_result: NotRequired[BusinessResult | None]
    answer: NotRequired[str | None]


def _apply_task_id_reply(reply: object) -> dict[str, str | None]:
    """把人工补答转成状态更新。不合格式时保留待补参状态。"""
    task_id = reply.strip() if isinstance(reply, str) else ""
    if task_id.isascii() and task_id.isdecimal():
        return {"task_id": task_id, "pending_question": None}
    return {"task_id": None, "pending_question": "任务编号格式有误，请输入只含数字的编号。"}


def ask_task_id(state: Day5State) -> dict[str, str | None]:
    """W11D5-1 暂停图，恢复时用人工补答更新 State。

    把准备好的 payload 传给 interrupt。首次执行会暂停，调用端收到追问。
    调用端以同一 thread_id 传入 Command(resume=补答) 后，本节点会从头执行，
    此时 interrupt 返回补答。把该返回值传给 _apply_task_id_reply，再返回它的更新。
    每次进入节点调用 interrupt 一次。校验失败会通过条件边再次进入本节点。
    """
    payload = {"kind": "task_id", "message": state.get("pending_question") or "请提供任务编号。"}
    reply = interrupt(payload)
    return _apply_task_id_reply(reply)


def _parse_request(state: Day5State) -> dict:
    try:
        task_id = parse_task({"question": state["question"]})["task_id"]
    except ValueError:
        task_id = None
    # 同一线程可以发起新查询。进入 parse 后先清除上一轮派生字段。
    return {"task_id": task_id, "pending_question": None, "query_result": None, "answer": None}


def _route_after_task_id(state: Day5State) -> Literal["ask_task_id", "query"]:
    return "query" if state.get("task_id") else "ask_task_id"


def _answer(state: Day5State) -> dict[str, str]:
    task_id = state["task_id"]
    result = state["query_result"]
    assert task_id is not None and result is not None
    return answer_task({"question": state["question"], "task_id": task_id, "query_result": result})


def build_day5_graph(
    query_task: Callable[[str], BusinessResult], *, checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """编译支持人工补参的查询图。checkpointer 保存暂停的位置和本轮输入。"""

    def query(state: Day5State) -> dict[str, BusinessResult]:
        task_id = state["task_id"]
        assert task_id is not None, "进入 query 前应已取得任务编号"
        return {"query_result": query_task(task_id)}

    builder = StateGraph(Day5State)
    builder.add_node("parse", _parse_request)
    builder.add_node("ask_task_id", ask_task_id)
    builder.add_node("query", query)
    builder.add_node("answer", _answer)
    builder.add_edge(START, "parse")
    builder.add_conditional_edges("parse", _route_after_task_id)
    builder.add_conditional_edges("ask_task_id", _route_after_task_id)
    builder.add_edge("query", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_day5_graph(
    db_path: str | Path, query_task: Callable[[str], BusinessResult],
) -> Iterator[CompiledStateGraph]:
    """用 Day4 的 SQLite Checkpointer 打开可以暂停和恢复的图。"""
    with open_task_checkpointer(db_path) as checkpointer:
        yield build_day5_graph(query_task, checkpointer=checkpointer)
