"""摘要函数的离线测试：使用真实 Runner，替换模型以避免网络请求。"""

import asyncio
import copy
import json

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

import contextCompact
from agentPrompts import HISTORY_SUMMARY_INSTRUCTIONS


class SummaryModel(Model):
    """记录实际进入模型的输入，只返回预设摘要或抛出指定异常。"""

    def __init__(self):
        self.text = "目标：修复筛选逻辑。约束：只修改 pipeline.py。下一步：运行测试。"
        self.error = None
        self.requests = []

    async def get_response(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["input"]))
        assert kwargs["system_instructions"] == HISTORY_SUMMARY_INSTRUCTIONS
        assert kwargs["tools"] == []
        if self.error is not None:
            raise self.error
        return ModelResponse(
            output=[ResponseOutputMessage(
                id="summary_message", role="assistant", status="completed", type="message",
                content=[ResponseOutputText(
                    text=self.text, type="output_text", annotations=[],
                )],
            )],
            usage=Usage(),
            response_id="summary_response",
        )

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def summary_model(monkeypatch):
    # 使用没有远程 exporter 的 tracing provider，隔离本机代理配置。
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    model = SummaryModel()
    clients = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-placeholder")

    class OfflineClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "offline-test-placeholder"
            assert kwargs["base_url"] == "https://api.deepseek.com"
            self.closed = False
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

    def make_model(**kwargs):
        assert kwargs["model"] == "deepseek-flash"
        assert kwargs["openai_client"] is clients[-1]
        return model

    monkeypatch.setattr(contextCompact, "AsyncOpenAI", OfflineClient)
    monkeypatch.setattr(contextCompact, "OpenAIChatCompletionsModel", make_model)
    return model, clients


@pytest.mark.parametrize("as_json", [False, True])
def test_history_is_data_in_one_request_and_returns_plain_text(summary_model, as_json):
    model, clients = summary_model
    model.text = " \n摘要文本\n "
    items = [
        {"role": "user", "content": "只修改 pipeline.py"},
        {"type": "function_call", "name": "readFile", "call_id": "old_call",
         "arguments": '{"path":"pipeline.py"}'},
        {"type": "function_call_output", "call_id": "old_call", "output": "代码内容"},
    ]
    original = copy.deepcopy(items)
    history = json.dumps(items, ensure_ascii=False) if as_json else items

    res = asyncio.run(contextCompact.summarize_history(history))
    result = res.get("summary", "")

    assert result == "摘要文本"
    assert items == original
    assert len(model.requests) == 1
    # 原工具调用保留在 JSON 文本内，没有成为当前 Runner 的可执行调用。
    assert len(model.requests[0]) == 1
    message = model.requests[0][0]
    assert message["role"] == "user"
    assert json.loads(message["content"]) == original
    assert clients[0].closed


class MemorySession:
    """只在本地内存中验证压缩结果，不写练习目录或数据库。"""

    def __init__(self, items):
        self.items = copy.deepcopy(items)
        self.writes = 0

    async def get_items(self):
        return copy.deepcopy(self.items)

    async def clear_session(self):
        self.writes += 1
        self.items.clear()

    async def add_items(self, items):
        self.writes += 1
        self.items.extend(copy.deepcopy(items))


def old_history():
    return [
        {"role": "user", "content": contextCompact._SUMMARY_PREFIX + "旧摘要" * 300},
        {"role": "user", "content": "第1轮任务"},
        {"role": "assistant", "content": "已检查代码" * 300},
        {"role": "user", "content": "第2轮任务"},
        {"role": "assistant", "content": "等待修改"},
        {"role": "user", "content": "第3轮任务"},
        {"role": "assistant", "content": "等待测试"},
    ]


def test_can_be_callback_and_merges_old_summary_without_recent_turns(summary_model):
    model, clients = summary_model
    original = old_history()
    session = MemorySession(original)

    result = asyncio.run(contextCompact.compact_session(
        session, contextCompact.summarize_history, force=True,
    ))

    assert result["compacted"] is True
    assert json.loads(model.requests[0][0]["content"]) == original[:3]
    assert session.items[0] == {
        "role": "user", "content": contextCompact._SUMMARY_PREFIX + model.text,
    }
    assert session.items[1:] == original[3:]
    assert clients[0].closed


@pytest.mark.parametrize("failure", ["timeout", "empty"])
def test_summary_failure_keeps_history_and_closes_client(summary_model, failure):
    model, clients = summary_model
    if failure == "timeout":
        model.error = TimeoutError("摘要请求超时")
    else:
        model.text = " \n "
    original = old_history()
    session = MemorySession(original)

    result = asyncio.run(contextCompact.compact_session(
        session, contextCompact.summarize_history, force=True,
    ))

    assert result["reason"] == "summary_failed"
    assert session.items == original
    assert session.writes == 0
    assert clients[0].closed


@pytest.mark.parametrize("history,error", [
    ([], ValueError), (" \n ", ValueError), ({"items": []}, TypeError),
])
def test_invalid_input_is_rejected_before_creating_client(summary_model, history, error):
    model, clients = summary_model
    with pytest.raises(error):
        asyncio.run(contextCompact.summarize_history(history))
    assert clients == []
    assert model.requests == []


def test_missing_key_preserves_history(summary_model, monkeypatch):
    model, clients = summary_model
    monkeypatch.delenv("OPENAI_API_KEY")
    original = old_history()
    session = MemorySession(original)
    result = asyncio.run(contextCompact.compact_session(
        session, contextCompact.summarize_history, force=True,
    ))
    assert result["reason"] == "summary_failed"
    assert session.items == original
    assert session.writes == 0
    assert clients == []
    assert model.requests == []
