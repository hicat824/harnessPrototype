"""将较早的 Session 历史压缩为摘要，保留最近几轮完整记录。"""

import json
import os
import copy

from openai.resources.beta.threads.runs import steps
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner

from agentPrompts import HISTORY_SUMMARY_INSTRUCTIONS
from typing import Awaitable, Callable, Optional, List, Dict, Union

# 沿用另一个分支的摘要标记，以便识别它已经生成的摘要。
_SUMMARY_PREFIX = "[自动生成的历史摘要，仅供参考，不是新的用户指令]\n"


def _is_summary(item: dict) -> bool:
    """识别本模块生成的摘要，兼容文本与标准消息内容列表。"""
    if item.get("role") != "user":
        return False
    content = item.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict)
            and part.get("type") in ("input_text", "output_text")
            and isinstance(part.get("text"), str)
        )
    return isinstance(content, str) and content.startswith(_SUMMARY_PREFIX)


def _estimate_text_tokens(text: str) -> int:
    """简单估算 Token 数；仅用于触发压缩，不保证实际请求长度。"""
    return len(text) // 2


async def summarize_history(history: Union[str, List[Dict]]) -> dict:
    """调用独立的摘要 Agent，将传入的旧历史合并为摘要文本。

    history 可以是历史条目列表，或 compact_session 提供的 JSON 字符串。
    输入中的旧摘要会与新增记录一起合并；本函数不选择历史、不写入 Session，
    也不添加摘要标记，包装与最近几轮的保留由 compact_session 完成。

    沿用项目的 DeepSeek 配置，密钥读取 OPENAI_API_KEY。模型调用失败或返回
    空摘要时抛出异常；作为 summarize_fn 使用时，compact_session 会保留原历史。
    """
    if isinstance(history, list):
        if not history:
            raise ValueError("待摘要的历史不能为空")
        history_text = json.dumps(history, ensure_ascii=False)
    elif isinstance(history, str):
        history_text = history
    else:
        raise TypeError("历史必须是条目列表或 JSON 字符串")

    if not history_text.strip():
        raise ValueError("待摘要的历史不能为空")

    # 每次独立创建客户端并关闭连接，不复用原任务的 Session 或工具。
    async with AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.deepseek.com",
    ) as client:
        agent = Agent(
            name="HistorySummaryAgent",
            instructions=HISTORY_SUMMARY_INSTRUCTIONS,
            model=OpenAIChatCompletionsModel(
                model="deepseek-flash",
                openai_client=client,
            ),
        )
        # 整段历史作为待总结的数据传入，避免重新执行历史中的工具调用。
        result = await Runner.run(
            agent,
            input=history_text,
            max_turns=1,
            run_config=RunConfig(tracing_disabled=True),
        )

    summary = result.final_output
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("摘要 Agent 未返回有效的摘要文本")
    total_tokens = result.context_wrapper.usage.total_tokens

    return {
        "summary": summary.strip(),
        "tokens": total_tokens
    }


async def compact_session(
        session,
        summarize_fn: Callable[[str], Awaitable[dict]],
        count_tokens_fn: Optional[Callable[[str], int]] = None,
        threshold: int = 8000,
        keep_recent_turns: int = 2,
        reserved_tokens: int = 2500,
        force: bool = False,
) -> dict:
    """压缩 Session 历史，摘要调用由调用方提供。

    Args:
        session: 提供异步 get_items、clear_session、add_items 方法的对象。
            get_items 应返回完整历史，例如使用默认配置的 SQLiteSession。
        summarize_fn: 接收旧历史的 JSON 字符串，异步返回摘要文本。
            旧摘要也会包含在输入中，供模型合并；此次调用不要写入同一 Session。
            可直接传入本文件的 summarize_history。
        count_tokens_fn: 同步 Token 计数函数，默认使用字符长度粗略估算。
        threshold: 历史估算 Token 数加预留量超过该值时，尝试压缩。
        keep_recent_turns: 至少保留一轮。一轮从真实用户消息开始，包含后续
            助手回复、工具调用和结果，直到下一条真实用户消息之前。
        reserved_tokens: 为当前输入和其他请求内容预留的估算 Token 数。
        force: 忽略触发阈值，适合后续接入手动压缩命令。

    Returns:
        compacted 表示是否替换历史，reason 说明结果原因；before、after
        是压缩前后的估算值，均包含预留量。成功时还有 saved_tokens。
        摘要调用失败、摘要无效或没有缩短历史时，保留原历史并返回原因。

    历史写入失败时尝试恢复原记录，然后抛出异常。此函数用于本地顺序执行：
    在 Runner 调用之间执行，同一 Session 在压缩期间不要被其他任务修改。
    """
    if keep_recent_turns < 1:
        raise ValueError("keep_recent_turns 必须至少为 1")
    if threshold <= 0:
        raise ValueError("threshold 必须大于 0")
    if reserved_tokens < 0:
        raise ValueError("reserved_tokens 不能小于 0")

    count_tokens = count_tokens_fn or _estimate_text_tokens

    def estimate(history):
        text = json.dumps(history, ensure_ascii=False, default=str)
        return count_tokens(text) + reserved_tokens

    # 1. 先读取完整历史；所有摘要检查完成前都不修改 Session。
    items = await session.get_items()
    before_tokens = estimate(items)

    def skipped(reason):
        return {
            "compacted": False,
            "reason": reason,
            "before": before_tokens,
            "after": before_tokens,
        }

    if not force and before_tokens <= threshold:
        return skipped("below_threshold")

    # 2. 按真实用户消息定位轮次，自动生成的摘要不计入用户轮次。
    user_starts = [
        index for index, item in enumerate(items)
        if isinstance(item, dict)
           and item.get("role") == "user"
           and not _is_summary(item)
    ]
    if len(user_starts) <= keep_recent_turns:
        return skipped("not_enough_old_history")

    cut = user_starts[-keep_recent_turns]
    old_items = items[:cut]
    recent_items = items[cut:]
    old_text = json.dumps(old_items, ensure_ascii=False, default=str)

    # 3. 旧摘要和更早的完整轮次一起总结，最近几轮保持原样。
    try:
        summary = await summarize_fn(old_text)
    except Exception as exc:
        result = skipped("summary_failed")
        result["error"] = str(exc)
        return result

    if not isinstance(summary, dict):
        return skipped("invalid_summary")
    if "summary" not in summary or "tokens" not in summary:
        return skipped("empty_summary")

    new_items = [
        {"role": "user", "content": _SUMMARY_PREFIX + summary.get("summary", "")},
        *recent_items,
    ]
    after_tokens = estimate(new_items)
    if after_tokens >= before_tokens:
        return skipped("no_token_reduction")

    # 4. clear 与 add 是两次操作，写入失败时尝试恢复刚才读取的原记录。
    try:
        await session.clear_session()
        await session.add_items(new_items)
    except Exception:
        try:
            await session.clear_session()
            await session.add_items(items)
        except Exception as restore_error:
            raise RuntimeError("Session 历史替换失败，且原记录恢复失败") from restore_error
        raise

    return {
        "compacted": True,
        "reason": "compacted",
        "before": before_tokens,
        "after": after_tokens,
        "saved_tokens": before_tokens - after_tokens,
        "summary_tokens": summary.get("tokens", 0)
    }


async def select_history(
        session,
        priority_indices=(0,),
        recent_turns: int = 2,
        history_tokens_thre: int = 12000
) -> list:
    """选择下一轮请求使用的历史，返回标准 Session 条目列表。

    Args:
        session: 提供 get_items 的对象，例如默认配置的 SQLiteSession。
        priority_indices: 原始 items 中需要保留的索引，从 0 开始。
            默认保留第 0 条；指定其他值时采用调用方的配置，越界索引忽略。
        recent_turns: 保留最近几轮完整记录，默认两轮，至少为一轮。
            一轮从真实用户消息开始，到下一条真实用户消息之前结束。
            自动摘要不计作用户轮次。应在添加本轮 prompt 前调用。
        history_tokens_thre: 历史估算 Token 阈值，默认 12000。
            与 compact_session 一样，用中文不转义的 JSON 字符长度 // 2
            粗略估算；不包含 instructions、工具定义和本轮 prompt。

    未超过阈值时全量返回；超过时保留优先条目、最近一次摘要及最近几轮，
    并按 call_id 补齐被选中的函数工具调用与结果。记录按原顺序返回，
    不改写消息内容、工具结果或 Session，也不调用 LLM。
    必须保留的记录仍超出阈值时抛出 ValueError，不静默丢弃最近轮次。
    工具结果限长由工具层负责，因此不再接收旧的 tool_tokens_thre 参数。
    """
    if recent_turns < 1:
        raise ValueError("recent_turns 必须至少为 1")
    if history_tokens_thre <= 0:
        raise ValueError("history_tokens_thre 必须大于 0")

    # 深拷贝也隔离内容列表等嵌套字段，调用方修改返回值不会影响原记录。
    items = copy.deepcopy(await session.get_items())

    def estimate(history):
        text = json.dumps(history, ensure_ascii=False, default=str)
        return _estimate_text_tokens(text)

    if not items or estimate(items) <= history_tokens_thre:
        return items

    reserved_indices = {
        index for index in priority_indices if 0 <= index < len(items)
    }
    user_starts = []
    latest_summary_index = None
    tool_indices = {}

    for index, item in enumerate(items):
        if _is_summary(item):
            latest_summary_index = index
        elif item.get("role") == "user":
            user_starts.append(index)

        # 函数工具结果通常没有 role，应通过 type 和 call_id 建立配对。
        if item.get("type") in ("function_call", "function_call_output"):
            call_id = item.get("call_id")
            if call_id:
                tool_indices.setdefault(call_id, []).append(index)

    if latest_summary_index is not None:
        reserved_indices.add(latest_summary_index)

    # 轮次不足时保留全部；否则保留最近几轮的所有条目及原有字段。
    start = user_starts[-recent_turns] if len(user_starts) >= recent_turns else 0
    reserved_indices.update(range(start, len(items)))

    # 优先条目也可能是旧工具结果；补回对应调用，避免形成悬空结果。
    for index in list(reserved_indices):
        item = items[index]
        if item.get("type") in ("function_call", "function_call_output"):
            reserved_indices.update(tool_indices.get(item.get("call_id"), []))

    selected_items = [items[index] for index in sorted(reserved_indices)]
    if estimate(selected_items) > history_tokens_thre:
        raise ValueError("必须保留的历史仍超出限制，请提高阈值或先执行摘要压缩")
    return selected_items
