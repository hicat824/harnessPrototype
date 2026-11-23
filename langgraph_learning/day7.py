"""Week11 Day7：在相同只读业务用例上对照 LangGraph 和 Agents SDK。"""

import json
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from agents import SQLiteSession
from agents.testing.model import ModelCall, ScriptedModel, assistant_message, function_call
from langgraph.types import Command

from langgraph_learning.day1 import parse_task
from langgraph_learning.day4 import checkpoint_config
from langgraph_learning.day5 import open_day5_graph
from week10_tool_rag import build_day3_agent, run_day3_agent
from week6 import BusinessResult


@dataclass(frozen=True)
class QueryOutcome:
    """统一业务结果。status 描述查询是否成功，task_state 才是任务运行状态。"""

    status: str
    task_id: str
    task_state: str | None
    error_code: str | None


@dataclass(frozen=True)
class TurnObservation:
    """一次调用的实际轨迹。steps 是节点或 SDK 事件，query_ids 来自业务查询回调。"""

    steps: tuple[str, ...]
    query_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunObservation:
    """一次用例的业务结果、各轮轨迹，以及重新打开数据库后读回的原问题。"""

    outcome: QueryOutcome
    turns: tuple[TurnObservation, ...]
    clarification: str | None = None
    restored_question: str | None = None


def compare_outcomes(
    expected: QueryOutcome, langgraph: QueryOutcome, agents_sdk: QueryOutcome,
) -> dict[str, bool]:
    """比较两条路径的业务结果，并保留它们是否都符合预期。

    expected 来自 Day1 固定用例的预期字段，langgraph 和 agents_sdk 来自本次实际运行。
    QueryOutcome 是 dataclass，可用 == 对比 status、task_id、task_state 和 error_code。

    返回三个键。langgraph_matches_expected 表示 graph 结果是否等于 expected。
    agents_sdk_matches_expected 表示 SDK 结果是否等于 expected。
    frameworks_agree 表示两条路径是否给出相同结果。
    例如期望 RUNNING，两侧却都得出 SUCCESS，结果应依次为 False、False、True。
    这里仅评估业务结果。节点轨迹、查询次数和缺参续接由独立断言验收。
    """
    langgraph_matches_expected = expected == langgraph
    agents_sdk_matches_expected = agents_sdk == expected
    frameworks_agree = agents_sdk == langgraph
    return {
        "frameworks_agree": frameworks_agree,
        "agents_sdk_matches_expected": agents_sdk_matches_expected,
        "langgraph_matches_expected": langgraph_matches_expected,
    }


def _record_query(query_task: Callable[[str], BusinessResult], query_ids: list[str]):
    def query(task_id: str) -> BusinessResult:
        query_ids.append(task_id)
        return query_task(task_id)

    return query


def run_langgraph_case(
    question: str, reply: str | None, *, case_id: str, db_path: Path,
    query_task: Callable[[str], BusinessResult],
) -> RunObservation:
    """运行 Day5 状态图。缺参时关闭连接，再凭相同 thread_id 恢复执行。"""
    config = checkpoint_config(case_id)
    query_ids: list[str] = []
    query = _record_query(query_task, query_ids)
    with open_day5_graph(db_path, query) as graph:
        updates = graph.stream({"question": question}, config, stream_mode="updates")
        steps = tuple(name for update in updates for name in update)
        snapshot = graph.get_state(config)
    turns = [TurnObservation(steps, tuple(query_ids))]
    clarification = restored_question = None

    if reply is not None:
        if snapshot.next != ("ask_task_id",) or len(snapshot.interrupts) != 1:
            raise RuntimeError("缺参用例应暂停在 ask_task_id")
        clarification = snapshot.interrupts[0].value["message"]
        query_ids.clear()
        with open_day5_graph(db_path, query) as graph:
            restored_question = graph.get_state(config).values["question"]
            updates = graph.stream(Command(resume=reply), config, stream_mode="updates")
            steps = tuple(name for update in updates for name in update)
            snapshot = graph.get_state(config)
        turns.append(TurnObservation(steps, tuple(query_ids)))

    if snapshot.next or snapshot.interrupts:
        raise RuntimeError("用例结束后仍有待执行的图节点")
    result = snapshot.values["query_result"]
    outcome = QueryOutcome(
        result.status, snapshot.values["task_id"], (result.data or {}).get("state"), result.error_code,
    )
    return RunObservation(outcome, tuple(turns), clarification, restored_question)


def _sdk_choose_next_action(call: ModelCall):
    """用模型实际收到的用户消息选择追问或查询。补答中的编号须结合历史才生效。"""
    messages = [item["content"] for item in call.input if item.get("role") == "user"]
    latest = messages[-1].strip()
    if len(messages) == 2 and latest.isascii() and latest.isdecimal():
        task_id = latest
    else:
        try:
            task_id = parse_task({"question": latest})["task_id"]
        except ValueError:
            return [assistant_message("请提供任务编号。")]
    return [function_call("query_task", {"task_id": task_id}, call_id="day7-query")]


def _sdk_answer_from_tool(call: ModelCall):
    """根据 SDK 回传的工具结果生成结构化业务结论。错误时从对应调用取得请求编号。"""
    output = next(item for item in reversed(call.input) if item.get("type") == "function_call_output")
    tool_call = next(item for item in call.input if item.get("type") == "function_call"
                     and item["call_id"] == output["call_id"])
    query = json.loads(output["output"])["task_query"]
    fact = query["fact"]
    requested_id = json.loads(tool_call["arguments"])["task_id"]
    outcome = QueryOutcome(
        query["status"], fact["task_id"] if fact else requested_id,
        fact["state"] if fact else None, query["error_code"],
    )
    return [assistant_message(json.dumps(asdict(outcome), ensure_ascii=False))]


async def run_agents_sdk_case(
    question: str, reply: str | None, *, case_id: str, db_path: Path,
    query_task: Callable[[str], BusinessResult],
) -> RunObservation:
    """用真实 Runner 和 SQLiteSession 执行离线脚本模型。缺参追问完成第一轮 Run。"""
    async def run_turn(message: str, *, restore: bool = False):
        model = ScriptedModel([{"responder": _sdk_choose_next_action}, {"responder": _sdk_answer_from_tool}])
        query_ids: list[str] = []
        restored_question = None
        with closing(SQLiteSession(case_id, db_path)) as session:
            if restore:
                history = await session.get_items()
                restored_question = next(item["content"] for item in history if item.get("role") == "user")
            result = await run_day3_agent(
                build_day3_agent(model), message, query_task=_record_query(query_task, query_ids),
                search_knowledge=lambda _: (), session=session,
            )
        steps = tuple(
            f"call:{item['name']}" if item["type"] == "function_call" else item["type"]
            for item in (event.to_input_item() for event in result.new_items)
        )
        return result.final_output, TurnObservation(steps, tuple(query_ids)), restored_question

    first_result, first_turn, _ = await run_turn(question)
    if reply is None:
        return RunObservation(QueryOutcome(**json.loads(first_result)), (first_turn,))

    # 第二次打开同一个 session_id，只提交编号。原问题由 SDK 从历史加入模型输入。
    final_result, second_turn, restored_question = await run_turn(reply, restore=True)
    return RunObservation(
        QueryOutcome(**json.loads(final_result)), (first_turn, second_turn), first_result, restored_question,
    )
