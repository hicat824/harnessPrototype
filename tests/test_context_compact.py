import asyncio
import copy
import json

import pytest

from contextCompact import _SUMMARY_PREFIX, compact_session


class MemorySession:
    """用内存模拟 Session，验证历史内容及写入时机。"""

    def __init__(self, items):
        self.items = copy.deepcopy(items)
        self.writes = []

    async def get_items(self):
        return copy.deepcopy(self.items)

    async def clear_session(self):
        self.writes.append("clear")
        self.items.clear()

    async def add_items(self, items):
        self.writes.append("add")
        self.items.extend(copy.deepcopy(items))


def turn(index):
    return [
        {"role": "user", "content": f"第 {index} 轮请求"},
        {"type": "function_call", "name": "readFile", "call_id": f"call_{index}", "arguments": "{}"},
        {"type": "function_call_output", "call_id": f"call_{index}", "output": "代码内容" * 200},
        {"role": "assistant", "content": f"第 {index} 轮结果"},
    ]


def history(count):
    return [item for index in range(1, count + 1) for item in turn(index)]


async def short_summary(text):
    return "任务目标、已完成工作和待办事项。"


async def unexpected_summary(text):
    raise AssertionError("此分支不应调用摘要函数")


def test_keeps_complete_recent_turns_and_summarizes_only_old_history():
    original = history(4)
    session = MemorySession(original)
    summary_inputs = []

    async def summarize(text):
        summary_inputs.append(json.loads(text))
        return "旧记录摘要"

    result = asyncio.run(compact_session(session, summarize, threshold=1))
    assert result["compacted"] is True
    assert summary_inputs == [turn(1) + turn(2)]
    assert session.items[0] == {"role": "user", "content": _SUMMARY_PREFIX + "旧记录摘要"}
    assert session.items[1:] == turn(3) + turn(4)
    assert result["saved_tokens"] == result["before"] - result["after"] > 0
    assert session.writes == ["clear", "add"]


def test_below_threshold_leaves_history_untouched():
    original = history(4)
    session = MemorySession(original)
    result = asyncio.run(compact_session(session, unexpected_summary, threshold=100000))
    assert result["reason"] == "below_threshold"
    assert result["before"] == result["after"]
    assert session.items == original
    assert session.writes == []


def test_force_ignores_threshold():
    session = MemorySession(history(4))
    result = asyncio.run(compact_session(session, short_summary, threshold=100000, force=True))
    assert result["compacted"] is True


def test_old_summary_is_not_counted_as_a_user_turn():
    original = [{"role": "user", "content": _SUMMARY_PREFIX + "旧摘要" * 1000}] + history(2)
    session = MemorySession(original)
    result = asyncio.run(compact_session(session, unexpected_summary, threshold=1))
    assert result["reason"] == "not_enough_old_history"
    assert session.items == original
    assert session.writes == []


def test_repeated_compaction_merges_existing_summary_with_old_turns():
    session = MemorySession(history(4))
    summary_inputs = []

    async def summarize(text):
        summary_inputs.append(json.loads(text))
        return f"第 {len(summary_inputs)} 次摘要"

    async def run():
        assert (await compact_session(session, summarize, threshold=1))["compacted"] is True
        previous_summary = copy.deepcopy(session.items[0])
        await session.add_items(turn(5))
        assert (await compact_session(session, summarize, threshold=1))["compacted"] is True
        assert summary_inputs[1] == [previous_summary] + turn(3)
        assert session.items[1:] == turn(4) + turn(5)
        assert session.items[0]["content"] == _SUMMARY_PREFIX + "第 2 次摘要"

    asyncio.run(run())


@pytest.mark.parametrize("summary,reason", [
    (None, "invalid_summary"), (123, "invalid_summary"),
    ("", "empty_summary"), (" \n ", "empty_summary"),
    ("很长的摘要" * 10000, "no_token_reduction"),
])
def test_invalid_or_larger_summary_keeps_original_history(summary, reason):
    original = history(4)
    session = MemorySession(original)

    async def summarize(text):
        return summary

    result = asyncio.run(compact_session(session, summarize, threshold=1))
    assert result["compacted"] is False
    assert result["reason"] == reason
    assert result["after"] == result["before"]
    assert session.items == original
    assert session.writes == []


def test_summary_failure_keeps_original_history_and_reports_error():
    original = history(4)
    session = MemorySession(original)

    async def summarize(text):
        raise TimeoutError("摘要请求超时")

    result = asyncio.run(compact_session(session, summarize, threshold=1))
    assert result["reason"] == "summary_failed"
    assert result["error"] == "摘要请求超时"
    assert session.items == original
    assert session.writes == []


class FailingSession(MemorySession):
    def __init__(self, items, fail_restore=False):
        super().__init__(items)
        self.add_count = 0
        self.fail_restore = fail_restore

    async def add_items(self, items):
        self.add_count += 1
        if self.add_count == 1 or self.fail_restore:
            # 模拟部分写入后失败，检查恢复时是否会清除残留记录。
            self.items.extend(copy.deepcopy(items[:1]))
            raise OSError("历史写入失败")
        await super().add_items(items)


def test_write_failure_restores_original_history_and_reraises():
    original = history(4)
    session = FailingSession(original)
    with pytest.raises(OSError, match="历史写入失败"):
        asyncio.run(compact_session(session, short_summary, threshold=1))
    assert session.items == original
    assert session.add_count == 2


def test_restore_failure_is_reported_explicitly():
    session = FailingSession(history(4), fail_restore=True)
    with pytest.raises(RuntimeError, match="原记录恢复失败") as caught:
        asyncio.run(compact_session(session, short_summary, threshold=1))
    assert isinstance(caught.value.__cause__, OSError)
    assert session.add_count == 2


@pytest.mark.parametrize("arguments", [
    {"keep_recent_turns": 0}, {"threshold": 0}, {"reserved_tokens": -1},
])
def test_invalid_settings_raise_before_modifying_history(arguments):
    session = MemorySession(history(4))
    with pytest.raises(ValueError):
        asyncio.run(compact_session(session, short_summary, **arguments))
    assert session.writes == []


def test_custom_counter_and_reserve_are_used():
    original = history(4)
    session = MemorySession(original)
    result = asyncio.run(compact_session(
        session, short_summary, count_tokens_fn=len,
        threshold=1, reserved_tokens=123, keep_recent_turns=1,
    ))
    assert result["compacted"] is True
    assert result["before"] == len(json.dumps(original, ensure_ascii=False, default=str)) + 123
    assert result["after"] == len(json.dumps(session.items, ensure_ascii=False, default=str)) + 123
    assert session.items[1:] == turn(4)


def test_real_sqlite_session_persists_compacted_history(tmp_path):
    from agents import SQLiteSession

    async def run():
        path = str(tmp_path / "session.db")
        session = SQLiteSession("compaction-test", path)
        await session.add_items(history(4))
        result = await compact_session(session, short_summary, threshold=1)
        saved = await session.get_items()
        assert result["compacted"] is True
        assert saved[1:] == turn(3) + turn(4)
        reopened = SQLiteSession("compaction-test", path)
        assert await reopened.get_items() == saved
        session.close()
        reopened.close()

    asyncio.run(run())
