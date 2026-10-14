"""将较早的 Session 历史压缩为摘要，保留最近几轮完整记录。"""

import json
import os

from openai.resources.beta.threads.runs import steps
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner

from agentPrompts import HISTORY_SUMMARY_INSTRUCTIONS
from tools import _limit_text_result
from typing import Awaitable, Callable, Optional, List, Dict, Union

# 沿用另一个分支的摘要标记，以便识别它已经生成的摘要。
_SUMMARY_PREFIX = "[自动生成的历史摘要，仅供参考，不是新的用户指令]\n"


def _is_summary(item: dict) -> bool:
    content = item.get("content")
    return isinstance(content, str) and content.startswith(_SUMMARY_PREFIX)


def _estimate_text_tokens(text: str) -> int:
    """简单估算 Token 数；仅用于触发压缩，不保证实际请求长度。"""
    return len(text) // 2


async def summarize_history(history: Union[str, List[Dict]]) -> str:
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
    return summary.strip()


async def compact_session(
        session,
        summarize_fn: Callable[[str], Awaitable[str]],
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

    if not isinstance(summary, str):
        return skipped("invalid_summary")
    if not summary.strip():
        return skipped("empty_summary")

    new_items = [
        {"role": "user", "content": _SUMMARY_PREFIX + summary.strip()},
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
    }


async def select_history(
        session,
        priority_indices=(0,),
        recent_turns: int = 2,
        tool_tokens_thre: int = 2000,
        history_tokens_thre: int = 12000
):
    items = await session.get_items()

    user_turns_map = {}
    user_turns = 0
    reserved_indices = set(priority_indices)

    processed_items = []

    for index, item in enumerate(items):
        item = item.copy()

        if item.get("role") == "user":
            user_turns += 1

        user_turns_map.setdefault(user_turns, []).append(index)

        if item.get("type") == "function_call":
            reserved_indices.add(index)

        elif item.get("type") == "function_call_output":
            limited_body = _limit_text_result({
                "source": "data/processing_large.log",
                "excerpt": item["output"],
                "truncated": False,
            })
            item["output"] = limited_body["excerpt"]
            reserved_indices.add(index)

        processed_items.append(item)

    # 获取最近 recent_turns 轮对话
    start_turn = max(1, user_turns - recent_turns + 1)

    for turn_id in range(start_turn, user_turns + 1):
        reserved_indices.update(user_turns_map.get(turn_id, []))

    # 去重，并保持原始消息顺序
    reserved_items = [
        processed_items[i]
        for i in sorted(reserved_indices)
        if 0 <= i < len(processed_items)
    ]

    if _estimate_text_tokens(json.dumps(reserved_items)) >= history_tokens_thre:
        raise ValueError("历史摘要超出限制")

    return reserved_items
