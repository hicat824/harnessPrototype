"""Week11 Day2：显式分支、有限检索回环与退出条件。"""

from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired

from week6 import BusinessResult
from week10_tool_rag import (
    Day1Request,
    Day1Result,
    KnowledgeEvidence,
    Route,
    prepare_knowledge_query,
    select_route,
)


class Day2State(TypedDict):
    """request 是输入，其他字段在图执行后按需生成。

    route 由 plan 写入。query 写入业务查询结果和已确认的任务状态。
    prepare_search 写入有序的候选检索词，并将 search_attempts 置零。
    search 每执行一次消耗一个候选词，并把该次的证据写入 evidence。
    finish 对已有结果归类并写入 result。
    """

    request: Day1Request
    route: NotRequired[Route]
    query_result: NotRequired[BusinessResult]
    task_state: NotRequired[str | None]
    search_queries: NotRequired[tuple[str, ...]]
    search_attempts: NotRequired[int]
    evidence: NotRequired[tuple[KnowledgeEvidence, ...]]
    result: NotRequired[Day1Result]


def route_after_plan(state: Day2State) -> Literal["query", "prepare_search", "finish"]:
    """W11D2-1 根据已写入 State 的路线和任务编号选择下一节点。

    rag_only 直接返回 "prepare_search"，即使问题用某个编号举例也走这条边。
    tool_only 和 tool_rag 需要任务编号。若为 None、空串或只有空白，返回 "finish"，
    由 finish 产出 needs_input。其余情况返回 "query"。函数只返回一个节点名。
    """
    if state["route"] == 'rag_only':
        return "prepare_search"
    elif state["route"] == 'tool_only' or state["route"] == 'tool_rag':
        if state["request"].task_id is None or not state["request"].task_id.strip():
            return "finish"
        else:
            return "query"
    else:
        raise ValueError("unsupported route")


def route_after_search(state: Day2State) -> Literal["search", "finish"]:
    """W11D2-2 检索命中就结束，空结果且还有备用词时回环。

    这里的 search_attempts 已包含刚刚结束的这次检索。evidence 非空时返回 "finish"。
    evidence 为空且 search_attempts 小于候选词数量时返回 "search"，下一次会取下一候选词。
    两次都为空或没有备用词可用时返回 "finish"，由 finish 给出证据不足的结果。
    """
    if len(state["evidence"]) > 0:
        return "finish"
    elif state["search_attempts"] < len(state["search_queries"]):
        return "search"
    else:
        return "finish"


def _route_after_query(state: Day2State) -> Literal["prepare_search", "finish"]:
    if state["route"] == "tool_rag" and state["task_state"] is not None:
        return "prepare_search"
    return "finish"


def _prepare_search(state: Day2State) -> dict:
    current_state = state.get("task_state")
    first_query = prepare_knowledge_query(state["request"], state["route"], current_state)
    if first_query is None:
        raise ValueError("当前路径缺少构建检索词所需的证据")
    # 组合问题首查完整状态释义，空结果时可用已确认的状态枚举补查。
    # 仅 RAG 路径使用原问题，因此只有一个候选词。
    queries = (first_query, current_state) if current_state is not None else (first_query,)
    return {"search_queries": queries, "search_attempts": 0}


def _finish(state: Day2State) -> dict[str, Day1Result]:
    route = state["route"]
    query_result = state.get("query_result")
    task_state = state.get("task_state")
    evidence = state.get("evidence", ())

    if route != "rag_only" and query_result is None:
        return {"result": Day1Result(route, "needs_input", reason="MISSING_PARAMETER")}
    if query_result is not None and task_state is None:
        if query_result.status == "error":
            reason = query_result.error_code or "TASK_QUERY_FAILED"
            return {"result": Day1Result(route, "error", reason=reason)}
        if query_result.status == "needs_input":
            reason = query_result.error_code or "MISSING_PARAMETER"
            return {"result": Day1Result(route, "needs_input", reason=reason)}
        return {"result": Day1Result(route, "insufficient", reason="TASK_STATE_UNAVAILABLE")}

    task_source = query_result.source if query_result is not None else None
    if state["request"].needs_rule and not evidence:
        # 即使查不到规则，也保留 API 已确认的事实。
        return {"result": Day1Result(route, "insufficient", reason="NO_KNOWLEDGE_EVIDENCE",
                                     task_state=task_state, task_source=task_source)}
    return {"result": Day1Result(route, "answered", task_state=task_state,
                                 task_source=task_source, evidence=evidence)}


def build_day2_graph(
    query_task: Callable[[str], BusinessResult],
    search_knowledge: Callable[[str], Sequence[KnowledgeEvidence]],
) -> CompiledStateGraph:
    """用两个可注入的只读工具编译图。调用时传入含 request 的 State。"""

    def plan(state: Day2State) -> dict[str, Route]:
        return {"route": select_route(state["request"])}

    def query(state: Day2State) -> dict:
        task_id = state["request"].task_id
        assert task_id is not None, "plan 的条件边只允许已提供编号的请求进入 query"
        result = query_task(task_id)
        value = (result.data or {}).get("state")
        task_state = value if result.status == "success" and isinstance(value, str) and value.strip() else None
        return {"query_result": result, "task_state": task_state}

    def search(state: Day2State) -> dict:
        index = state["search_attempts"]
        evidence = tuple(search_knowledge(state["search_queries"][index]))
        return {"evidence": evidence, "search_attempts": index + 1}

    builder = StateGraph(Day2State)
    builder.add_node("plan", plan)
    builder.add_node("query", query)
    builder.add_node("prepare_search", _prepare_search)
    builder.add_node("search", search)
    builder.add_node("finish", _finish)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_conditional_edges("query", _route_after_query)
    builder.add_edge("prepare_search", "search")
    builder.add_conditional_edges("search", route_after_search)
    builder.add_edge("finish", END)
    return builder.compile()
