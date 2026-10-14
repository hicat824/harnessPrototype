import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
from contextCompact import compact_session, select_history
from tools import _read_text_range

_WORKSPACE = "/media/xumaozhou/data/AI/local_agent_test/week5_day3_context_pruning"

class FakeSession:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = Path(input_file_path)
        self.output_file_path = Path(output_file_path)

    async def get_items(self, limit=None):
        # 每次调用都从本地文件加载数据
        content = json.loads(
            self.input_file_path.read_text(encoding="utf-8")
        )
        items = content["items"]

        if not isinstance(items, list):
            raise ValueError("Session 数据必须是列表")

        if limit is None:
            return items

        if limit <= 0:
            return []

        return items[-limit:]


    async def add_items(self, item):
        if self.output_file_path.exists():
            content = json.loads(
                self.output_file_path.read_text(encoding="utf-8")
            )
        else:
            content = {"items": []}

        items = content.get("items", [])

        if not isinstance(items, list):
            raise ValueError("Session 数据必须是列表")

        if items is None or len(items) == 0:
            items = []
        if not isinstance(item, list):
            raise ValueError("item 必须是列表")

        items.extend(item)
        content["items"] = items

        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))


    async def clear_session(self):
        content = {"items": []}
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))


@pytest.mark.asyncio
async def test_compact_session():
    fake_summarize_fn = AsyncMock()
    fake_summarize_fn.return_value = "[摘要信息]"

    input_session_json = f'{_WORKSPACE}/session_history.json'
    output_session_json = f'{_WORKSPACE}/selected_context.json'
    session = FakeSession(input_session_json, output_session_json)

    result = await compact_session(session, fake_summarize_fn)
    assert result.get("compacted", False) == True



@pytest.mark.asyncio
async def test_select_history():
    input_session_json = f'{_WORKSPACE}/session_history.json'
    output_session_json = f'{_WORKSPACE}/selected_context.json'
    session = FakeSession(input_session_json, output_session_json)

    select_items = await select_history(session=session, priority_indices=(0,))
    assert len(select_items) > 0

    await session.clear_session()
    await session.add_items(select_items)
    assert Path(output_session_json).exists()



@pytest.mark.parametrize(
    "path, limit",
    [
        (f'{_WORKSPACE}/workspace/data/processing_large.log', 2000),
        (f'{_WORKSPACE}/workspace/data/tasks_large.json', 2000),
    ]
)
def test_read_text_range(path, limit):
    encoded_result = _read_text_range(target=Path(path))
    assert len(encoded_result) <= limit