"""Week8 Day1：模型/消息学习的自检、离线验收和显式真实调用。

使用 langchain_learning/.venv/bin/python；主环境没有 LangChain 时整组跳过。
两个关键步骤用 RUN_WEEK8_DAY1=1 启用，真实模型另需 RUN_WEEK8_DAY1_LLM=1。
测试数据均在本文件定义，无需准备业务知识库或另写测试代码。
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("langchain_core", reason="请使用 Week8 独立环境，见 langchain_learning/week8.md")
pytest.importorskip("langchain_openai", reason="请安装 Week8 requirements.txt 中的模型适配器")

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import APIStatusError

from langchain_learning.day1 import build_messages, build_model, run_query, to_sdk_request


REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("WEEK8_DAY1_OUTPUT", str(REPO / "langchain_learning/artifacts/day1")))
DAY1 = pytest.mark.skipif(os.environ.get("RUN_WEEK8_DAY1") != "1",
                          reason="填写两个关键步骤后设置 RUN_WEEK8_DAY1=1")
REAL = pytest.mark.skipif(
    os.environ.get("RUN_WEEK8_DAY1") != "1" or os.environ.get("RUN_WEEK8_DAY1_LLM") != "1",
    reason="真实调用需额外设置 RUN_WEEK8_DAY1_LLM=1",
)
INSTRUCTIONS = "根据用户提供的信息简洁回答，不编造额外事实。"
HISTORY = [{"role": "user", "content": "任务标识为字符串 00123。"},
           {"role": "assistant", "content": "已知任务标识为字符串 00123。"}]
QUESTION = "刚才的任务标识是什么？请保留前导零。"
ANSWER = "任务标识是字符串 00123，应保留前导零。"


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    """本组测试不上传 LangSmith trace，离线用例只允许本地替身和 Mock HTTP。"""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class RecordingModel:
    """记录消息后交给 LangChain 自带 Fake ChatModel，不请求网络。"""

    def __init__(self, response=None, error=None):
        response = AIMessage(content=ANSWER) if response is None else response
        self.fake = FakeMessagesListChatModel(responses=[response])
        self.calls, self.error = [], error

    async def ainvoke(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        return await self.fake.ainvoke(messages)


def _transport(records, status=200):
    def handle(request):
        records.append({"url": str(request.url), "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "模拟接口失败", "type": "mock_error"}})
        return httpx.Response(200, json={
            "id": "chatcmpl-offline", "object": "chat.completion", "created": 1,
            "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": ANSWER}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        })
    return httpx.MockTransport(handle)


def test_provided_sdk_comparison_preserves_roles_and_text():
    messages = [SystemMessage(content=INSTRUCTIONS), HumanMessage(content=HISTORY[0]["content"]),
                AIMessage(content=HISTORY[1]["content"]), HumanMessage(content=QUESTION)]
    before = copy.deepcopy(messages)
    result = to_sdk_request(messages)
    assert result == {"instructions": INSTRUCTIONS, "input": [*HISTORY, {"role": "user", "content": QUESTION}]}
    result["input"][0]["content"] = "修改对照结果"
    assert messages == before


@pytest.mark.parametrize("messages", [
    [], [HumanMessage(content="缺少规则"), HumanMessage(content="问题")],
    [SystemMessage(content="规则"), SystemMessage(content="另一条规则")],
    [SystemMessage(content="规则"), ToolMessage(content="工具结果", tool_call_id="call-1")],
    [SystemMessage(content="规则"), AIMessage(content=[{"type": "text", "text": "内容块"}])],
])
def test_provided_sdk_comparison_rejects_unsupported_messages(messages):
    with pytest.raises(ValueError):
        to_sdk_request(messages)


@pytest.mark.parametrize("options", [
    {"instructions": " "}, {"question": ""}, {"history": "字符串不是列表"},
    {"history": [{"role": "system", "content": "不能混入历史规则"}]},
    {"history": [{"role": "user", "content": " "}]},
    {"history": [{"role": "user", "content": "内容", "unknown": True}]},
])
def test_provided_input_validation_runs_before_todo(options):
    values = {"instructions": INSTRUCTIONS, "question": QUESTION, **options}
    with pytest.raises(ValueError):
        build_messages(**values)


def test_provided_model_requires_existing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DeepSeek"):
        build_model()
    with pytest.raises(ValueError, match="model"):
        build_model(api_key="offline-key", model=" ")


@pytest.mark.asyncio
async def test_provided_chat_adapter_sends_roles_to_mock_api():
    records = []
    transport = _transport(records)
    with httpx.Client(transport=transport) as client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = build_model(api_key="offline-key", http_client=client, http_async_client=async_client)
            response = await model.ainvoke([SystemMessage(content=INSTRUCTIONS), HumanMessage(content=QUESTION)])
    assert isinstance(response, AIMessage) and response.content == ANSWER
    assert response.usage_metadata["input_tokens"] == 12
    assert response.usage_metadata["output_tokens"] == 8 and response.usage_metadata["total_tokens"] == 20
    assert len(records) == 1
    assert records[0]["url"] == "https://api.deepseek.com/chat/completions"
    body = records[0]["body"]
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == QUESTION
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_provided_http_failure_is_not_retried(status):
    records = []
    transport = _transport(records, status)
    with httpx.Client(transport=transport) as client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = build_model(api_key="offline-key", http_client=client, http_async_client=async_client)
            with pytest.raises(APIStatusError) as error:
                await model.ainvoke([HumanMessage(content=QUESTION)])
    assert error.value.status_code == status and len(records) == 1


def test_provided_cli_requires_explicit_real_flag():
    result = subprocess.run([sys.executable, "-B", "-m", "langchain_learning.day1", "--question", QUESTION],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--real-llm" in result.stderr


@DAY1
@pytest.mark.parametrize("history", [None, [], HISTORY])
def test_message_order_types_and_original_text(history):
    before = copy.deepcopy(history)
    instructions, question = " 规则\n保留原文。 ", " 任务 00123\n是什么？ "
    messages = build_messages(instructions, question, history)
    assert isinstance(messages[0], SystemMessage) and messages[0].content == instructions
    assert isinstance(messages[-1], HumanMessage) and messages[-1].content == question
    assert len(messages) == len(history or []) + 2
    if history:
        assert isinstance(messages[1], HumanMessage) and isinstance(messages[2], AIMessage)
        assert [message.content for message in messages[1:-1]] == [item["content"] for item in history]
    assert history == before


@DAY1
@pytest.mark.asyncio
async def test_one_async_call_records_aimessage_usage_and_sdk_comparison():
    response = AIMessage(content=ANSWER, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                         response_metadata={"finish_reason": "stop", "model_name": "fake-model", "private_field": "不应导出"})
    model = RecordingModel(response)
    output = OUTPUT / "offline/report.json"
    report = await run_query(model, INSTRUCTIONS, QUESTION, history=HISTORY, output_path=output)
    assert len(model.calls) == 1
    assert report["messages"] == [{"type": m.type, "content": m.content} for m in model.calls[0]]
    assert [m["type"] for m in report["messages"]] == ["system", "human", "ai", "human"]
    assert report["sdk_request"] == {"instructions": INSTRUCTIONS, "input": [*HISTORY, {"role": "user", "content": QUESTION}]}
    assert report["response"] == {"type": "ai", "content": ANSWER}
    assert report["usage"] == response.usage_metadata and report["elapsed_seconds"] >= 0
    assert report["response_metadata"] == {"finish_reason": "stop", "model_name": "fake-model"}
    assert json.loads(output.read_text(encoding="utf-8")) == report
    print(f"Week8 Day1 离线产物：{output}")


@DAY1
@pytest.mark.asyncio
async def test_model_instance_does_not_implicitly_keep_history():
    model = RecordingModel()
    await run_query(model, INSTRUCTIONS, "第一轮：任务标识是 00123。")
    second = await run_query(model, INSTRUCTIONS, "第二轮：刚才的标识是什么？")
    assert len(model.calls) == 2 and len(model.calls[1]) == 2
    assert "00123" not in json.dumps(second["messages"], ensure_ascii=False)
    assert second["usage"] is None  # 替身没有 usage，不能填成 0 或估计数。


@DAY1
@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("模拟故障"), httpx.ReadTimeout("模拟超时")])
async def test_model_failure_propagates_and_preserves_old_report(tmp_path, error):
    path = tmp_path / "report.json"
    path.write_text('{"previous": true}', encoding="utf-8")
    model = RecordingModel(error=error)
    with pytest.raises(type(error)):
        await run_query(model, INSTRUCTIONS, QUESTION, output_path=path)
    assert len(model.calls) == 1 and path.read_text(encoding="utf-8") == '{"previous": true}'


@DAY1
@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    HumanMessage(content="不是模型回答"), AIMessage(content=" "),
    AIMessage(content=[{"type": "text", "text": "本日暂不处理内容块"}]),
    AIMessage(content="", tool_calls=[{"name": "query_task", "args": {"task_id": "00123"}, "id": "call-1"}]),
])
async def test_invalid_or_tool_response_is_not_a_completed_text_answer(response):
    with pytest.raises(ValueError):
        await run_query(RecordingModel(response), INSTRUCTIONS, QUESTION)


@REAL
@pytest.mark.asyncio
async def test_real_model_messages_and_leading_zero():
    assert os.environ.get("OPENAI_API_KEY"), "需要项目已有 DeepSeek OPENAI_API_KEY"
    output = OUTPUT / "real_model/report.json"
    with httpx.Client(trust_env=False) as client:
        async with httpx.AsyncClient(trust_env=False) as async_client:
            model = build_model(http_client=client, http_async_client=async_client)
            report = await run_query(model, INSTRUCTIONS, QUESTION, history=HISTORY, output_path=output)
    assert report["response"]["type"] == "ai" and "00123" in report["response"]["content"]
    assert [m["type"] for m in report["messages"]] == ["system", "human", "ai", "human"]
    assert report["sdk_request"]["input"][-1]["role"] == "user"
    print(f"Week8 Day1 真实产物：{output}；请核对消息与 SDK 对照，Token 用量以实际返回为准")
