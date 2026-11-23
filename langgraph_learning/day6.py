"""Week11 Day6：恢复后重新核验权限，并按有效期复用或补查结果。"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from typing_extensions import NotRequired

from langgraph_learning.day1 import answer_task
from langgraph_learning.day4 import open_task_checkpointer
from week6 import BusinessResult


ResumeAction = Literal["reject", "forbidden", "reuse", "refresh"]


class Day6State(TypedDict):
    """输入只有 task_id，查询结果、确认信息和退出原因由图写入。"""

    task_id: str
    query_result: NotRequired[BusinessResult | None]
    cached_at: NotRequired[float | None]
    approved: NotRequired[bool]
    resume_action: NotRequired[ResumeAction | None]
    status: NotRequired[Literal["ready", "pending_review", "answered", "cancelled", "error"]]
    reason: NotRequired[str | None]
    answer: NotRequired[str | None]


def decide_resume_action(
    *, approved: bool, allowed: bool, cache_age_seconds: float, cache_ttl_seconds: float,
) -> ResumeAction:
    """按人工决定、当前权限和缓存年龄依次选择恢复行为。

    approved 为 False 时返回 "reject"。同意后若 allowed 为 False，返回 "forbidden"。
    两项都通过后，年龄在 [0, cache_ttl_seconds) 内时返回 "reuse"，否则返回 "refresh"。
    恰好到期时需要补查。负年龄也补查，避免时钟回拨后误用旧数据。
    返回动作字符串即可，查询、清除缓存和设置停止原因由后续节点负责。
    """
    if not approved:
        return "reject"
    if not allowed:
        return "forbidden"
    if 0 <= cache_age_seconds < cache_ttl_seconds:
        return "reuse"
    return "refresh"


def _review(state: Day6State) -> dict[str, bool]:
    # 恢复时本节点从头执行。这里只生成追问，不在 interrupt 前调用工具。
    reply = interrupt({"kind": "confirm_task_result", "task_id": state["task_id"]})
    return {"approved": reply is True}


def build_day6_graph(
    query_task: Callable[[str], BusinessResult],
    can_read_task: Callable[[str], bool],
    *,
    checkpointer: BaseCheckpointSaver,
    cache_ttl_seconds: float = 60,
    clock: Callable[[], float] = time,
) -> CompiledStateGraph:
    """构建只读复核图。权限和时间由调用方注入，测试可在暂停期间修改它们。"""
    if cache_ttl_seconds <= 0:
        raise ValueError("cache_ttl_seconds 必须大于 0")

    def fetch_task(task_id: str) -> dict:
        result = query_task(task_id)
        task_state = (result.data or {}).get("state")
        if result.status == "success" and isinstance(task_state, str) and task_state.strip():
            return {"query_result": result, "cached_at": clock(), "status": "ready", "reason": None}
        return {"query_result": None, "cached_at": None, "status": "error",
                "reason": result.error_code or "TASK_STATE_UNAVAILABLE"}

    def prefetch(state: Day6State) -> dict:
        task_id = state["task_id"]
        if can_read_task(task_id):
            update = fetch_task(task_id)
            if update["status"] == "ready":
                update["status"] = "pending_review"
        else:
            update = {"query_result": None, "cached_at": None, "status": "error", "reason": "FORBIDDEN"}
        return {**update, "approved": False, "resume_action": None, "answer": None}

    def decide(state: Day6State) -> dict:
        approved = state["approved"]
        allowed = can_read_task(state["task_id"]) if approved else False
        cached_at = state["cached_at"]
        assert cached_at is not None, "只有成功获取状态后才进入人工复核"
        action = decide_resume_action(
            approved=approved, allowed=allowed, cache_age_seconds=clock() - cached_at,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        if action == "reject":
            return {"resume_action": action, "status": "cancelled", "reason": "USER_REJECTED",
                    "query_result": None, "cached_at": None}
        if action == "forbidden":
            return {"resume_action": action, "status": "error", "reason": "FORBIDDEN",
                    "query_result": None, "cached_at": None}
        return {"resume_action": action, "status": "ready"}

    def refresh(state: Day6State) -> dict:
        return fetch_task(state["task_id"])

    def finish(state: Day6State) -> dict:
        if state["status"] != "ready":
            return {"answer": None}
        result = state["query_result"]
        assert result is not None, "生成答复前应已取得任务状态"
        answer = answer_task({"question": "", "task_id": state["task_id"], "query_result": result})
        return {"status": "answered", **answer}

    def after_prefetch(state: Day6State) -> Literal["review", "finish"]:
        return "review" if state["status"] == "pending_review" else "finish"

    def after_decide(state: Day6State) -> Literal["refresh", "finish"]:
        return "refresh" if state["resume_action"] == "refresh" else "finish"

    builder = StateGraph(Day6State)
    builder.add_node("prefetch", prefetch)
    builder.add_node("review", _review)
    builder.add_node("decide", decide)
    builder.add_node("refresh", refresh)
    builder.add_node("finish", finish)
    builder.add_edge(START, "prefetch")
    builder.add_conditional_edges("prefetch", after_prefetch)
    builder.add_edge("review", "decide")
    builder.add_conditional_edges("decide", after_decide)
    builder.add_edge("refresh", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_day6_graph(
    db_path: str | Path,
    query_task: Callable[[str], BusinessResult],
    can_read_task: Callable[[str], bool],
    *,
    cache_ttl_seconds: float = 60,
    clock: Callable[[], float] = time,
) -> Iterator[CompiledStateGraph]:
    """复用 Day4 的 SQLite Checkpointer。重新打开同一文件后可继续等待中的线程。"""
    with open_task_checkpointer(db_path) as checkpointer:
        yield build_day6_graph(query_task, can_read_task, checkpointer=checkpointer,
                              cache_ttl_seconds=cache_ttl_seconds, clock=clock)
