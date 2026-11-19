"""Week10 的离线路由与多来源 Context 练习，复用 Week6 查询结果契约。"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from week6 import BusinessResult


Route = Literal["tool_only", "rag_only", "tool_rag"]
Outcome = Literal["answered", "needs_input", "insufficient", "error"]


@dataclass(frozen=True)
class Day1Request:
    """本轮问题与已识别的意图。

    Day1 由测试把 eval_cases.json 中的 input 转成此对象。Day3 再让模型识别意图与编号。

    字段：
        question: 用户原问题，供静态规则路由构造知识检索参数。
        needs_current_state: 是否需要从 Mock 任务系统读取任务此刻的状态。
        needs_rule: 是否需要知识库解释业务规则或状态含义。
        task_id: 用户明确给出的任务编号，按字符串传给 query_task。缺失时为 None。
    """

    question: str
    needs_current_state: bool
    needs_rule: bool
    task_id: str | None = None


@dataclass(frozen=True)
class KnowledgeEvidence:
    """知识工具返回的一条证据，用于支持业务规则的解释。

    doc_id 是评分时用于核对命中文档的标识。source 定位原文，例如
    knowledge/KB006.md#L6。text 是该位置的文档原文。
    """

    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Day1Result:
    """一次路由执行的结果，包含已取得的证据和停止原因。

    route 记录实际执行的路径。status 表示本轮执行结果：answered 表示所需证据齐全，
    needs_input 表示缺参，insufficient 表示查询事实或知识证据不足，error 表示业务
    查询报错。reason 提供相应的错误码或停止原因，正常完成时为 None。

    task_state 和 task_source 来自成功的业务查询。例如 task_state 为 FAILED 时，
    只说明任务执行失败，本轮仍可能得到 answered。evidence 存放知识检索返回的文档片段。
    """

    route: Route
    status: Outcome
    reason: str | None = None
    task_state: str | None = None
    task_source: str | None = None
    evidence: tuple[KnowledgeEvidence, ...] = ()


def select_route(request: Day1Request) -> Route:
    """根据已识别的意图选择工具执行路径（练习 W10D1-1）。

    输入是 Day1Request，测试已填好 needs_current_state（是否查当前状态）和
    needs_rule（是否解释业务规则或状态含义）。任务编号由 run_day1 在执行时检查。

    返回规则：
        仅 needs_current_state 为 True：返回 tool_only，只查业务任务。
        仅 needs_rule 为 True：返回 rag_only，只查知识文档。
        两项都为 True：返回 tool_rag，先查任务，再按实际状态检索文档。
        两项都为 False：抛出 ValueError，表示问题超出本练习的三条路径。
    """
    if request.needs_rule and request.needs_current_state:
        return 'tool_rag'
    elif request.needs_rule:
        return 'rag_only'
    elif request.needs_current_state:
        return 'tool_only'
    else:
        raise ValueError("unknown route")


def prepare_knowledge_query(
    request: Day1Request, route: Route, current_state: str | None,
) -> str | None:
    """为 run_day1 的下一步准备知识检索参数（练习 W10D1-2）。

    run_day1 先选路，按需调用 query_task，再调用本函数。返回字符串时，
    run_day1 将它传入 search_knowledge。返回 None 时跳过知识检索。

    参数：
        request: 本轮请求，request.question 是用户原问题。
        route: select_route 选出的 tool_only、rag_only 或 tool_rag。
        current_state: 从成功的 query_task 响应 data["state"] 提取的非空字符串。
            没有查询、查询失败或响应缺少有效状态时为 None。

    返回规则：
        tool_only: 返回 None，只需查询当前状态。
        rag_only: 返回 request.question，按原问题查业务规则。
        tool_rag: 用实际查到的状态生成询问其含义的检索问题。无状态时返回 None。

    例如用户问「任务现在还是 FAILED 吗？这个状态是什么意思？」，路由为
    tool_rag，实际查到 RUNNING，则返回「RUNNING 状态的业务含义是什么？」。
    """
    if route == "rag_only":
        return request.question
    if route == "tool_only":
        return None

    # 组合路由只解释查询确认的状态，避免把原问题里猜测的状态带入检索。
    if current_state is None:
        return None
    return f"{current_state} 状态的业务含义是什么？"


def run_day1(
    request: Day1Request,
    *,
    query_task: Callable[[str], BusinessResult],
    search_knowledge: Callable[[str], Sequence[KnowledgeEvidence]],
) -> Day1Result:
    """执行选路、Mock 查询和知识检索，汇总本轮已取得的证据。

    参数：
        request: 含用户问题、已识别意图和可选任务编号的 Day1Request。
        query_task: 只读业务查询。输入为任务编号字符串，输出为 Week6 BusinessResult，
            其中 status 表示查询是否成功，data["state"] 才是任务状态。
        search_knowledge: 知识检索。输入为检索问题字符串，输出为 KnowledgeEvidence
            序列，找不到可用文档时返回空序列。

    执行顺序：
        1. 用 select_route 选择路径。路径需要业务查询且缺任务编号时，直接返回 needs_input。
        2. 按需查询任务。从成功响应中取出非空的任务状态，供组合路由检索时使用。
        3. 用 prepare_knowledge_query 确定检索参数。返回 None 时跳过知识工具。

    返回 Day1Result，便于验收对照路由、证据和停止原因。查询出错时保留错误码。
    组合路由已查到状态但没有知识证据时，返回 insufficient 并保留已确认的任务状态。
    """
    route = select_route(request)
    task = None
    current_state = None
    if route != "rag_only":
        if request.task_id is None or request.task_id == "":
            return Day1Result(route, "needs_input", reason="MISSING_PARAMETER")
        task = query_task(request.task_id)
        if task.status == "success":
            state = (task.data or {}).get("state")
            if isinstance(state, str) and state.strip():
                current_state = state

    query = prepare_knowledge_query(request, route, current_state)
    evidence = tuple(search_knowledge(query)) if query is not None else ()

    if task is not None and current_state is None:
        if task.status == "needs_input":
            return Day1Result(route, "needs_input", reason=task.error_code or "MISSING_PARAMETER")
        if task.status == "error":
            return Day1Result(route, "error", reason=task.error_code or "TASK_QUERY_FAILED")
        return Day1Result(route, "insufficient", reason="TASK_STATE_UNAVAILABLE")

    # 组合路由缺少知识证据时，结果仍保留已经确认的任务状态及来源。
    task_source = task.source if task is not None else None
    if request.needs_rule and not evidence:
        return Day1Result(route, "insufficient", reason="NO_KNOWLEDGE_EVIDENCE",
                          task_state=current_state, task_source=task_source)
    return Day1Result(route, "answered", task_state=current_state,
                      task_source=task_source, evidence=evidence)


@dataclass(frozen=True)
class Day2Context:
    """本轮的证据 Context 及预算诊断。

    text 是完整、可解析的 JSON，后续可作为证据交给回答阶段。max_chars 是 text 的
    字符上限。omitted_ids 记录因预算跳过的证据编号，仅供程序和测试报告使用，
    避免把大量被跳过的编号也塞进 Context。
    """

    text: str
    max_chars: int
    omitted_ids: tuple[str, ...]


def _extract_task_fact(result: BusinessResult) -> dict[str, str] | None:
    """从已经完成的 Mock 任务详情查询中提取可引用的事实（练习 W10D2-1）。

    build_day2_context 传入 Week6 BusinessService.query_task 返回的 BusinessResult。
    result.status 表示查询结果，只有 success 才能提取事实。result.data 是响应的业务
    字典，其中 task_id 是任务标识，state 是实际查到的任务状态。它们都必须是非空白字符串。

    返回字典固定包含 evidence_id、task_id、state、source、fetched_at 五个字段。
    本轮最多查询一个任务，evidence_id 固定为 "T1"。task_id 和 state 保留响应中的原值，
    source 和 fetched_at 取自 BusinessResult 的同名字段。Context 会把此字典放在
    task_query.fact 下，回答阶段可以用 T1 引用这条事实。

    查询失败、无 data 或缺少有效任务标识与状态时返回 None。额外的 params、tags 等
    字段留在原响应中，输入对象保持不变。state 为 "FAILED" 可以是一条有效事实，
    只要查询本身成功。

    例如 status="success"、data={"task_id": "12345", "state": "RUNNING", "params": {}}
    时，返回 T1，保留 "12345"、"RUNNING"、来源和查询时间。
    """
    if result.status != "success" or result.data is None:
        return None

    task_id = result.data.get("task_id")
    state = result.data.get("state")
    # data 是通用字典，字段可能为 null 或其他类型。先确认是字符串，再检查是否只有空白。
    # strip() 只用于校验，事实中保留响应原值，便于回溯来源。
    if not isinstance(task_id, str) or not task_id.strip():
        return None
    if not isinstance(state, str) or not state.strip():
        return None

    return {
        "evidence_id": "T1",
        "task_id": task_id,
        "state": state,
        "source": result.source,
        "fetched_at": result.fetched_at,
    }


def build_day2_context(
    task_result: BusinessResult | None,
    evidence: Sequence[KnowledgeEvidence],
    *,
    max_chars: int = 6000,
) -> Day2Context:
    """把查询摘要和文档证据组装成有硬字符预算的 JSON（练习 W10D2-2）。

    参数：
        task_result: 本轮 query_task 的结果。静态规则路由没有业务查询时为 None。
        evidence: 知识工具返回的片段，调用方已按相关性排序。每条包含 doc_id、source 和原文 text。
        max_chars: 对最终 Context JSON 文本执行 len() 得到的上限，包含字段名、编号、引号以及
            转义字符。预算单位是字符，计算范围是此 Context，系统提示和问题有各自的预算。

    返回：
        Day2Context.text 包含 task_query、knowledge、knowledge_truncated。
        task_query 保存查询结果、错误码、上游截断标记以及可用的任务事实，整个摘要优先保留。
        knowledge 保存放得下的片段。编号 K1、K2……根据原始输入位置分配，跳过后保留其他编号。
        带有遗漏时 knowledge_truncated 为 true。被跳过的编号另存入 Day2Context.omitted_ids。
        若最小 Context 都超限，抛出 ValueError，让调用方调整预算。

    预算规则：从排序首项向后尝试，把当前片段暂时追加到已保留列表后，测量渲染后的 JSON 长度。
    长度小于或等于 max_chars 时加入该片段。超限时记录其 evidence_id，继续尝试下一个片段。
    例如 K1 已保留，K2 太长、K3 能放下时，JSON 应保留 K1、K3 和完整任务摘要，
    knowledge_truncated 为 true，omitted_ids 为 ("K2",)。每条保留的正文和引用位置保持原样。
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars 必须是正整数")

    task_query = None
    if task_result is not None:
        task_query = {
            "status": task_result.status,
            "error_code": task_result.error_code,
            "source_truncated": task_result.truncated,
            "fact": _extract_task_fact(task_result),
        }

    candidates = [
        {"evidence_id": f"K{index}", "doc_id": item.doc_id, "source": item.source, "text": item.text}
        for index, item in enumerate(evidence, start=1)
    ]

    def render(selected: list[dict[str, str]]) -> str:
        """序列化本次试放的片段集合，计数与最终输出采用相同格式。"""
        return json.dumps({
            "task_query": task_query,
            "knowledge": selected,
            "knowledge_truncated": len(selected) < len(candidates),
        }, ensure_ascii=False, separators=(",", ":"))

    kept = []
    omitted_ids = []
    # 空文档列表时也保留任务摘要和截断标记，作为 Context 的最小开销。
    if len(render(kept)) > max_chars:
        raise ValueError("Context 预算不足以容纳必要字段")

    for candidate in candidates:
        # 用与最终输出相同的 JSON 格式试放，计数覆盖字段名、引用和转义后的正文。
        trial_text = render([*kept, candidate])
        if len(trial_text) <= max_chars:
            kept.append(candidate)
        else:
            # 跳过的只是当前片段，后面较短的片段仍有机会放入。同一文档的片段用 evidence_id 区分。
            omitted_ids.append(candidate["evidence_id"])

    return Day2Context(render(kept), max_chars, tuple(omitted_ids))
