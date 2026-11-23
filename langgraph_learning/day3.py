"""Week11 Day3：按业务错误码重试，并限制整张图的工具调用。"""

from collections.abc import Callable, Sequence
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired

from langgraph_learning.day2 import (
    Day2State,
    _finish,
    _prepare_search,
    _route_after_query,
    route_after_plan,
    route_after_search,
)
from week6 import BusinessResult
from week10_tool_rag import Day1Result, KnowledgeEvidence, select_route


class Day3State(Day2State):
    """在 Day2 的业务状态上记录实际发起的调用次数和预算退出原因。

    query_attempts 只记录任务 API 调用。tool_calls 还包括知识检索。
    两者都计入失败的调用。stop_reason 由工具节点在调用前拦截时写入。
    """

    query_attempts: NotRequired[int]
    tool_calls: NotRequired[int]
    stop_reason: NotRequired[Literal["TOOL_BUDGET_EXHAUSTED"]]


def should_retry_query(result: BusinessResult, attempts: int, max_attempts: int) -> bool:
    """W11D3-1 判断上一次业务 API 查询是否允许按原编号再试。

    result 是 Week6 Service 已归一化的返回值，attempts 包含刚刚完成的调用。
    只有 status 为 error、error_code 为 TIMEOUT，且 attempts 小于 max_attempts，
    才返回 True。成功、空数据、FORBIDDEN 等情况都返回 False。
    此函数只判断查询次数和错误类型。全图工具预算由工具节点在发起调用前检查。
    """
    return result.status == "error" and result.error_code == "TIMEOUT" and attempts < max_attempts


def reserve_tool_call(used: int, limit: int) -> int | None:
    """W11D3-2 计算下一次工具调用应写回的累计次数。

    used 是本次图运行中已经发起的查询与检索总次数，limit 是最多允许的次数。
    used 小于 limit 时返回 used + 1，表示允许一次调用，并占用一个名额。
    used 达到或超过 limit 时返回 None。工具节点会写入预算停止原因并跳过调用。
    """
    return used + 1 if used < limit else None


def build_day3_graph(
    query_task: Callable[[str], BusinessResult],
    search_knowledge: Callable[[str], Sequence[KnowledgeEvidence]],
    *,
    max_query_attempts: int = 2,
    max_tool_calls: int = 4,
    max_steps: int = 12,
) -> CompiledStateGraph:
    """编译 Day3 图。输入格式与 Day2 一致，调用方传入含 request 的 State。

    max_query_attempts 包含首次查询。max_tool_calls 同时限制查询、重试和知识检索。
    max_steps 是 LangGraph 的超步上限，超限会抛出 GraphRecursionError。
    query_task 应由调用方配置为单次请求，图负责决定是否重试。
    """
    if type(max_query_attempts) is not int or max_query_attempts < 1:
        raise ValueError("max_query_attempts 必须是正整数")
    if type(max_tool_calls) is not int or max_tool_calls < 0:
        raise ValueError("max_tool_calls 必须是非负整数")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps 必须是正整数")

    def plan(state: Day3State) -> dict:
        return {"route": select_route(state["request"]), "query_attempts": 0, "tool_calls": 0}

    def query(state: Day3State) -> dict:
        used = reserve_tool_call(state["tool_calls"], max_tool_calls)
        if used is None:
            return {"stop_reason": "TOOL_BUDGET_EXHAUSTED"}
        task_id = state["request"].task_id
        assert task_id is not None, "plan 的条件边只允许已提供编号的请求进入 query"
        result = query_task(task_id)
        value = (result.data or {}).get("state")
        task_state = value if result.status == "success" and isinstance(value, str) and value.strip() else None
        return {
            "query_result": result,
            "task_state": task_state,
            "query_attempts": state["query_attempts"] + 1,
            "tool_calls": used,
        }

    def after_query(state: Day3State) -> Literal["query", "prepare_search", "finish"]:
        if state.get("stop_reason"):
            return "finish"
        if should_retry_query(state["query_result"], state["query_attempts"], max_query_attempts):
            return "query"
        return _route_after_query(state)

    def search(state: Day3State) -> dict:
        used = reserve_tool_call(state["tool_calls"], max_tool_calls)
        if used is None:
            return {"stop_reason": "TOOL_BUDGET_EXHAUSTED"}
        index = state["search_attempts"]
        evidence = tuple(search_knowledge(state["search_queries"][index]))
        return {"evidence": evidence, "search_attempts": index + 1, "tool_calls": used}

    def after_search(state: Day3State) -> Literal["search", "finish"]:
        if state.get("stop_reason"):
            return "finish"
        return route_after_search(state)

    def finish(state: Day3State) -> dict[str, Day1Result]:
        stop_reason = state.get("stop_reason")
        if stop_reason is None:
            return _finish(state)
        query_result = state.get("query_result")
        task_state = state.get("task_state")
        return {"result": Day1Result(
            state["route"], "error", reason=stop_reason,
            task_state=task_state,
            task_source=query_result.source if query_result is not None and task_state is not None else None,
            evidence=state.get("evidence", ()),
        )}

    builder = StateGraph(Day3State)
    builder.add_node("plan", plan)
    builder.add_node("query", query)
    builder.add_node("prepare_search", _prepare_search)
    builder.add_node("search", search)
    builder.add_node("finish", finish)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("query", after_query)
    builder.add_edge("prepare_search", "search")
    builder.add_conditional_edges("search", after_search)
    builder.add_edge("finish", END)
    return builder.compile().with_config({"recursion_limit": max_steps})
