"""Week8 Day2：Prompt/Runnable/Parser 自检与显式真实调用。

数据均定义在本文件；普通测试不访问模型。使用 Week8 独立环境。
完整离线流程需设置 RUN_WEEK8_DAY2=1，真实调用另需 RUN_WEEK8_DAY2_LLM=1。
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("langchain_core", reason="请使用 langchain_learning/.venv/bin/python")
pytest.importorskip("langchain_openai", reason="请安装 Week8 requirements.txt")

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from langchain_learning.day1 import build_model
from langchain_learning.day2 import (
    Day2Answer, build_chain, build_prompt, prepare_input, run_chain, validate_sources,
)


REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("WEEK8_DAY2_OUTPUT", str(REPO / "langchain_learning/artifacts/day2")))
DAY2 = pytest.mark.skipif(os.environ.get("RUN_WEEK8_DAY2") != "1", reason="完整流程需设置 RUN_WEEK8_DAY2=1")
REAL = pytest.mark.skipif(
    os.environ.get("RUN_WEEK8_DAY2") != "1" or os.environ.get("RUN_WEEK8_DAY2_LLM") != "1",
    reason="真实调用需额外设置 RUN_WEEK8_DAY2_LLM=1",
)
QUESTION = "NOT_FOUND 是不是表示任务执行失败？"
EVIDENCE = [{"source_id": "S1", "text": "NOT_FOUND 表示本次查询没有找到指定任务，不等于任务执行失败。"}]
ANSWER = {"status": "answered", "answer": "NOT_FOUND 表示查询未找到，不代表任务执行失败。", "source_ids": ["S1"]}
INSUFFICIENT = {"status": "insufficient", "answer": "缺少该任务当前的实时状态证据，需要实时查询。", "source_ids": []}


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class RecordingModel:
    """实际使用 LangChain Fake ChatModel 返回 AIMessage，不请求网络。"""

    def __init__(self, raw=None, error=None):
        self.raw = json.dumps(ANSWER, ensure_ascii=False) if raw is None else raw
        self.fake = FakeMessagesListChatModel(responses=[AIMessage(
            content=self.raw, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )])
        self.calls, self.error = [], error

    async def ainvoke(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        return await self.fake.ainvoke(messages)


@pytest.mark.parametrize("changes", [
    {"status": "unknown"}, {"answer": " "}, {"source_ids": []},
    {"status": "insufficient"}, {"extra": "不能添加字段"}, {"source_ids": [1]},
    {"source_ids": ["S1", "S1"]}, {"source_ids": [" "]},
])
def test_provided_answer_schema_rejects_invalid_data(changes):
    with pytest.raises(ValidationError):
        Day2Answer.model_validate({**ANSWER, **changes})


@pytest.mark.parametrize("changes", [
    {"question": " "}, {"evidence": "不是列表"}, {"evidence": ["不是字典"]},
    {"evidence": [{"source_id": "S1"}]}, {"evidence": [{"source_id": "S1", "text": " "}]},
    {"evidence": EVIDENCE * 2}, {"evidence": [{"source_id": " ", "text": "正文"}]},
])
def test_provided_input_validation(changes):
    values = {"question": QUESTION, "evidence": copy.deepcopy(EVIDENCE), **changes}
    with pytest.raises(ValueError):
        prepare_input(**values)


def test_provided_input_preserves_literal_braces_and_original_data():
    evidence = [{"source_id": "S1", "text": '正文含 {question} 和 {"flag": true}，不是模板变量。'}]
    before = copy.deepcopy(evidence)
    result = prepare_input(" 保留问题\n ", evidence)
    assert result["question"] == " 保留问题\n "
    assert json.loads(result["context"]) == before and evidence == before
    assert prepare_input(QUESTION, [])["context"] == "[]"


@pytest.mark.parametrize("fenced", [False, True])
def test_provided_parser_returns_pydantic_object(fenced):
    raw = json.dumps(ANSWER, ensure_ascii=False)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    parser = PydanticOutputParser(pydantic_object=Day2Answer)
    answer = parser.invoke(AIMessage(content=raw))
    assert isinstance(answer, Day2Answer) and answer.model_dump() == ANSWER


@pytest.mark.parametrize("raw", ["not JSON", json.dumps({**ANSWER, "source_ids": []})])
def test_provided_parser_reports_unparseable_or_invalid_output(raw):
    parser = PydanticOutputParser(pydantic_object=Day2Answer)
    with pytest.raises(OutputParserException):
        parser.invoke(AIMessage(content=raw))


def test_provided_parser_validity_does_not_prove_source_exists():
    answer = Day2Answer.model_validate({**ANSWER, "source_ids": ["S99"]})
    with pytest.raises(ValueError, match="不存在"):
        validate_sources(answer, EVIDENCE)
    assert validate_sources(Day2Answer.model_validate(ANSWER), EVIDENCE) is None


def test_provided_cli_requires_explicit_real_flag():
    result = subprocess.run([sys.executable, "-B", "-m", "langchain_learning.day2", "--question", QUESTION],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--real-llm" in result.stderr


@DAY2
def test_prompt_binds_schema_and_only_exposes_runtime_variables():
    parser = PydanticOutputParser(pydantic_object=Day2Answer)
    prompt = build_prompt(parser)
    assert set(prompt.input_variables) == {"question", "context"}
    evidence = [{"source_id": "S1", "text": '保留 {question} 和 {"key": 1}'}]
    payload = prepare_input(QUESTION, evidence)
    value = prompt.invoke(payload)
    assert isinstance(value, ChatPromptValue)
    messages = value.to_messages()
    assert [m.type for m in messages] == ["system", "human"]
    assert parser.get_format_instructions() in messages[0].content
    assert QUESTION in messages[1].content and evidence[0]["text"] in messages[1].content.replace('\\"', '"')
    assert payload["context"] in messages[1].content


@DAY2
def test_runnable_pipeline_transfers_prompt_to_model_and_parsed_object():
    seen = []
    def model_step(prompt_value):
        seen.append(prompt_value)
        return AIMessage(content=json.dumps(ANSWER, ensure_ascii=False))
    chain = build_chain(RunnableLambda(model_step))
    result = chain.invoke(prepare_input(QUESTION, EVIDENCE))
    assert len(seen) == 1 and isinstance(seen[0], ChatPromptValue)
    assert isinstance(result, Day2Answer) and result.model_dump() == ANSWER


@DAY2
def test_missing_prompt_variable_stops_before_model():
    seen = []
    def model_step(value):
        seen.append(value)
        return AIMessage(content=json.dumps(ANSWER, ensure_ascii=False))
    chain = build_chain(RunnableLambda(model_step))
    with pytest.raises(KeyError):
        chain.invoke({"question": QUESTION})
    assert seen == []


@DAY2
@pytest.mark.asyncio
async def test_async_chain_records_actual_messages_raw_and_parsed_answer():
    model = RecordingModel()
    before = copy.deepcopy(EVIDENCE)
    output = OUTPUT / "offline/answered_report.json"
    result = await run_chain(model, QUESTION, EVIDENCE, output_path=output)
    assert result["execution_status"] == "succeeded" and result["error"] is None
    assert result["answer"] == ANSWER and result["raw_response"] == model.raw
    assert result["model_calls"] == len(model.calls) == 1
    assert result["messages"] == [{"type": m.type, "content": m.content} for m in model.calls[0]]
    assert result["usage"]["total_tokens"] == 15 and result["elapsed_seconds"] >= 0
    assert json.loads(output.read_text(encoding="utf-8")) == result and EVIDENCE == before
    print(f"Week8 Day2 离线产物：{output}")


@DAY2
@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", [EVIDENCE, []])
async def test_insufficient_is_successful_execution_not_exception(evidence, tmp_path):
    model = RecordingModel(raw=json.dumps(INSUFFICIENT, ensure_ascii=False))
    result = await run_chain(model, "任务 12345 当前完成了吗？", evidence,
                             output_path=tmp_path / "insufficient_report.json")
    assert result["execution_status"] == "succeeded" and result["answer"] == INSUFFICIENT
    assert result["error"] is None and len(model.calls) == 1


@DAY2
@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("private-error-detail"), httpx.ReadTimeout("private-error-detail")])
async def test_model_failure_preserves_stage_without_parsing_or_retry(tmp_path, error):
    model = RecordingModel(error=error)
    output = tmp_path / "failure_report.json"
    with pytest.raises(type(error)):
        await run_chain(model, QUESTION, EVIDENCE, output_path=output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["execution_status"] == "failed" and saved["error"]["stage"] == "model"
    assert saved["raw_response"] is None and saved["answer"] is None
    assert saved["model_calls"] == len(model.calls) == 1
    assert "private-error-detail" not in output.read_text(encoding="utf-8")


@DAY2
@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["not JSON", json.dumps({**ANSWER, "status": "unknown"})])
async def test_parser_failure_keeps_raw_response_and_propagates(tmp_path, raw):
    model = RecordingModel(raw=raw)
    output = tmp_path / "parser_failure.json"
    with pytest.raises(OutputParserException):
        await run_chain(model, QUESTION, EVIDENCE, output_path=output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["error"] == {"stage": "parser", "type": "OutputParserException"}
    assert saved["raw_response"] == raw and saved["answer"] is None
    assert saved["model_calls"] == len(model.calls) == 1


@DAY2
@pytest.mark.asyncio
async def test_unknown_source_is_detected_after_successful_parsing(tmp_path):
    raw = json.dumps({**ANSWER, "source_ids": ["S99"]}, ensure_ascii=False)
    output = tmp_path / "source_failure.json"
    with pytest.raises(ValueError, match="不存在"):
        await run_chain(RecordingModel(raw=raw), QUESTION, EVIDENCE, output_path=output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["error"] == {"stage": "source_validation", "type": "ValueError"}
    assert saved["raw_response"] == raw and saved["answer"] is None


@REAL
@pytest.mark.asyncio
@pytest.mark.parametrize("question,status", [
    (QUESTION, "answered"), ("任务 12345 现在已经完成了吗？", "insufficient"),
])
async def test_real_prompt_model_parser(question, status):
    assert os.environ.get("OPENAI_API_KEY"), "需要项目既有 DeepSeek OPENAI_API_KEY"
    output = OUTPUT / f"real_model/{status}_report.json"
    with httpx.Client(trust_env=False) as client:
        async with httpx.AsyncClient(trust_env=False) as async_client:
            model = build_model(http_client=client, http_async_client=async_client)
            result = await run_chain(model, question, EVIDENCE, output_path=output)
    assert result["execution_status"] == "succeeded" and result["model_calls"] == 1
    assert result["answer"]["status"] == status
    assert result["answer"]["source_ids"] == (["S1"] if status == "answered" else [])
    print(f"Week8 Day2 真实产物：{output}；请核对回答是否受证据支持")
