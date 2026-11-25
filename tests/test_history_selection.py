"""历史选择测试：覆盖标准条目、摘要识别及真实 SQLiteSession。"""

import asyncio
import copy
import json

import pytest
from agents import SQLiteSession

from contextCompact import _SUMMARY_PREFIX, select_history


class MemorySession:
    def __init__(self, items):
        self.items = copy.deepcopy(items)

    async def get_items(self):
        return self.items


def token_estimate(items):
    return len(json.dumps(items, ensure_ascii=False)) // 2


def turn(number):
    """模拟 SDK 生成的记录，保留消息内容列表与额外字段。"""
    return [
        {"role": "user", "content": [
            {"type": "input_text", "text": f"第{number}轮任务"},
        ]},
        {"type": "reasoning", "id": f"rs_{number}", "summary": [
            {"type": "summary_text", "text": "先检查代码"},
        ]},
        {"type": "function_call", "id": f"fc_{number}",
         "call_id": f"call_{number}", "name": "readFile", "status": "completed",
         "arguments": '{"path":"src/pipeline.py"}'},
        {"type": "function_call_output", "call_id": f"call_{number}",
         "output": json.dumps({
             "source": "src/pipeline.py", "excerpt": "代码预览", "truncated": True,
         }, ensure_ascii=False)},
        {"type": "message", "id": f"msg_{number}", "role": "assistant",
         "status": "completed", "content": [
             {"type": "output_text", "text": "已检查，尚未执行测试", "annotations": []},
         ]},
    ]


def long_history():
    return [
        {"role": "user", "content": "持续约束：只允许修改 src/pipeline.py"},
        {"role": "assistant", "content": "过期闲聊" * 10000},
        *turn(1), *turn(2), *turn(3), *turn(4),
    ]


@pytest.mark.parametrize("items", [[], turn(1) + turn(2) + turn(3)])
def test_small_history_returns_every_item_without_changing_source(items):
    session = MemorySession(items)
    selected = asyncio.run(select_history(session))
    assert selected == items
    assert session.items == items
    if selected:
        # 修改返回列表的嵌套字段，不会污染 Session 原记录。
        selected[0]["content"][0]["text"] = "已改写返回值"
        assert session.items == items


def test_history_at_threshold_is_returned_in_full():
    items = turn(1) + turn(2) + turn(3)
    selected = asyncio.run(select_history(
        MemorySession(items), history_tokens_thre=token_estimate(items),
    ))
    assert selected == items


def test_large_history_keeps_first_item_and_two_complete_recent_turns():
    original = long_history()
    expected = [original[0], *turn(3), *turn(4)]
    session = MemorySession(original)
    selected = asyncio.run(select_history(
        session, history_tokens_thre=token_estimate(expected),
    ))
    assert selected == expected
    assert session.items == original
    assert [item["call_id"] for item in selected if item.get("type") == "function_call"] == [
        "call_3", "call_4",
    ]


@pytest.mark.parametrize("content_as_list", [False, True])
def test_latest_summary_is_kept_but_does_not_count_as_a_user_turn(content_as_list):
    first = {"role": "user", "content": "只修改 src/pipeline.py"}
    old_summary = {"role": "user", "content": _SUMMARY_PREFIX + "过期摘要"}
    text = _SUMMARY_PREFIX + "最新摘要：筛选问题已定位，测试尚未执行"
    latest = {"role": "user", "content": (
        [{"type": "input_text", "text": text}] if content_as_list else text
    )}
    original = [
        first, old_summary, *turn(1),
        {"role": "assistant", "content": "无关记录" * 10000},
        *turn(2), latest, *turn(3),
    ]
    expected = [first, *turn(2), latest, *turn(3)]
    selected = asyncio.run(select_history(
        MemorySession(original), history_tokens_thre=token_estimate(expected),
    ))
    assert selected == expected
    assert old_summary not in selected


@pytest.mark.parametrize("priority_type", ["function_call", "function_call_output"])
def test_priority_tool_record_keeps_both_sides_of_the_call(priority_type):
    original = long_history()
    priority = next(index for index, item in enumerate(original)
                    if item.get("type") == priority_type and item.get("call_id") == "call_1")
    old_pair = [item for item in original if item.get("call_id") == "call_1"]
    expected = [original[0], *old_pair, *turn(3), *turn(4)]
    selected = asyncio.run(select_history(
        MemorySession(original), priority_indices=(0, priority, -1, 999),
        history_tokens_thre=token_estimate(expected),
    ))
    assert selected == expected


def test_recent_turn_count_can_be_configured():
    original = long_history()
    expected = [original[0], *turn(4)]
    selected = asyncio.run(select_history(
        MemorySession(original), recent_turns=1,
        history_tokens_thre=token_estimate(expected),
    ))
    assert selected == expected


def test_large_required_history_raises_without_modifying_session():
    original = long_history()
    session = MemorySession(original)
    with pytest.raises(ValueError, match="必须保留的历史仍超出限制"):
        asyncio.run(select_history(session, history_tokens_thre=1))
    assert session.items == original


@pytest.mark.parametrize("items", [turn(1), [{"role": "assistant", "content": "旧记录"}]])
def test_not_enough_user_turns_keeps_all_records(items):
    session = MemorySession(items)
    # 超过预算时无法通过丢弃完整轮次解决，应明确报错。
    with pytest.raises(ValueError, match="必须保留"):
        asyncio.run(select_history(session, history_tokens_thre=1))
    assert session.items == items


@pytest.mark.parametrize("settings", [{"recent_turns": 0}, {"history_tokens_thre": 0}])
def test_invalid_settings_raise(settings):
    with pytest.raises(ValueError):
        asyncio.run(select_history(MemorySession([]), **settings))


def test_real_sqlite_session_returns_selected_items_without_writing_history(tmp_path):
    original = long_history()
    expected = [original[0], *turn(3), *turn(4)]

    async def run():
        session = SQLiteSession("selection-test", str(tmp_path / "history.db"))
        try:
            await session.add_items(original)
            selected = await select_history(
                session, history_tokens_thre=token_estimate(expected),
            )
            assert selected == expected
            assert await session.get_items() == original
        finally:
            session.close()

    asyncio.run(run())
