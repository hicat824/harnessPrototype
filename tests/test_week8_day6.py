"""Week8 Day6：LangChain 检索函数与 Agents SDK Function Tool 的连接。

数据、脚本模型和断言均已提供，不需要填写测试代码。
三个连接点已完成；RUN_WEEK8_DAY6=1 启用完整离线验收，默认运行框架自检。
真实对照另需 RUN_WEEK8_DAY6_LLM=1，使用既有 Ollama/DeepSeek，不请求业务 API。
每项测试的数据来源、流程、检查点和产物见 langchain_learning/week8.md#day6。
"""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

pytest.importorskip('agents')
pytest.importorskip('langchain_core')
pytest.importorskip('numpy')

from agents import OpenAIResponsesModel
from agents.exceptions import ModelBehaviorError, UserError
from agents.testing.model import ScriptedModel, assistant_message, function_call
from agents.tool_context import ToolContext
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from agents.usage import Usage

import rag.sdk_rag as sdk_rag
from rag.answering import DeepSeekGenerator, RagAnswer, build_context
from rag.embeddings import OllamaEncoder
from rag.langchain_preparation import DEFAULT_DATASET, Week7Embeddings, chunk_document
from rag.langchain_rag import (
    build_store, document_chunk, make_retriever, prepare_store, run_rag, _save_json,
)
from rag.preprocessing import KnowledgeDocument, make_chunk
from rag.sdk_rag import (
    DEFAULT_OUTPUT, KnowledgeRunContext, build_sdk_rag_agent, retrieve_context,
    run_sdk_rag, search_knowledge, validate_sdk_answer,
)


DAY6 = pytest.mark.skipif(os.environ.get('RUN_WEEK8_DAY6') != '1', reason='完整验收需 RUN_WEEK8_DAY6=1')
REAL = pytest.mark.skipif(
    any(os.environ.get(key) != '1' for key in ('RUN_WEEK8_DAY6', 'RUN_WEEK8_DAY6_LLM')),
    reason='真实对照需 DAY6 和 LLM 同时启用',
)
OUTPUT = Path(os.environ.get('WEEK8_DAY6_OUTPUT', str(DEFAULT_OUTPUT)))
QUESTION = 'NOT_FOUND 是不是表示任务执行失败？'


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    # 复用 Week6 的测试隔离方式：无远程导出器，避免 SDK 懒初始化读取外网代理。
    # fixture 结束后恢复原对象；不修改主项目的全局追踪配置。
    monkeypatch.setattr(tracing_setup, 'GLOBAL_TRACE_PROVIDER', DefaultTraceProvider())


class FixedEncoder:
    """合成三维向量；刻意把旧版排得更前，检查真正的过滤与上下文传递。"""

    model, query_instruction = 'day6-fixture-not-a-model', 'fixture-query'

    def __init__(self):
        self.calls = []

    def encode_document(self, texts):
        self.calls.append(('document', list(texts)))
        return [[1, 0, 0], [0.8, 0.6, 0]]

    def encode_query(self, texts):
        self.calls.append(('query', list(texts)))
        return [[1, 0, 0] for _ in texts]


@pytest.fixture
def existing():
    chunks = []
    for doc_id, status, version, text in [
        ('OLD', 'historical', '1.0', '旧版说明不应参与本轮回答。'),
        ('GOOD', 'current', '2.0', 'NOT_FOUND 表示未找到，不等于任务执行失败。'),
    ]:
        unit = KnowledgeDocument(doc_id, text, dict(
            source=f'knowledge/{doc_id}.md', version=version, knowledge_status=status,
            engine_code='import_std', scope=['learning'], format='markdown',
        ))
        chunks.append(chunk_document(make_chunk(unit, text, 0, len(text), 0, [])))
    encoder = FixedEncoder()
    store = build_store(chunks, Week7Embeddings(encoder))
    encoder.calls.clear()
    return store, encoder


def retriever_for(store, **options):
    return make_retriever(store, k=1, filters={'knowledge_status': 'current'}, **options)


def answer_for(evidence):
    """从实际工具证据构造脚本回答，只验证数据流，不代表语义推理。"""
    if not evidence:
        return dict(status='insufficient', claims=[], missing_information='本轮没有可用知识证据。')
    source = evidence[0]
    return dict(status='answered', claims=[dict(
        text='依据检索结果说明 NOT_FOUND 的含义。',
        citations=[dict(source_id=source['source_id'], quote=source['chunk']['text'])],
    )], missing_information='')


def scripted_model(question=QUESTION, final=None):
    def respond(call):
        # 读取 SDK 第二次请求中的真实工具输出，不能绕过工具直接读 fixture。
        tool_output = [item for item in call.input if item.get('type') == 'function_call_output'][-1]
        evidence = json.loads(tool_output['output'])['evidence']
        return [assistant_message(json.dumps(final if final is not None else answer_for(evidence), ensure_ascii=False))]
    return ScriptedModel([
        [function_call('search_knowledge', {'query': question}, call_id='call_day6_search')],
        {'responder': respond},
    ], default_usage=Usage(requests=1, input_tokens=10, output_tokens=2, total_tokens=12))


def tool_context(state, query):
    return ToolContext(context=state, tool_name='search_knowledge', tool_call_id='call_direct',
                       tool_arguments=json.dumps({'query': query}, ensure_ascii=False))


def test_provided_tool_schema_only_exposes_query():
    schema = search_knowledge.params_json_schema
    assert set(schema['properties']) == {'query'}
    assert schema['required'] == ['query'] and schema['additionalProperties'] is False
    assert schema['properties']['query']['type'] == 'string'
    assert 'source' in search_knowledge.description and 'version' in search_knowledge.description


@pytest.mark.parametrize('problem', ['no_retrieval', 'unknown_source', 'invented_quote'])
def test_provided_validation_uses_this_runs_context(existing, problem):
    store, _ = existing
    document = store.get_by_ids(['GOOD:body:0000'])[0]
    context = build_context([dict(score=0.8, chunk=asdict(document_chunk(document)))])
    raw = answer_for(context.sources)
    if problem == 'no_retrieval':
        context = None
    elif problem == 'unknown_source':
        raw['claims'][0]['citations'][0]['source_id'] = 'S99'
    else:
        raw['claims'][0]['citations'][0]['quote'] = '旧版说明不应参与本轮回答。'
    with pytest.raises(ValueError):
        validate_sdk_answer(RagAnswer.model_validate(raw), context)


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['', '  ', None, 123])
async def test_provided_bad_query_never_reaches_retriever(existing, query):
    store, encoder = existing
    state = KnowledgeRunContext(retriever_for(store))
    wrapper = tool_context(state, query)
    with pytest.raises((ValueError, ModelBehaviorError)):
        await search_knowledge.on_invoke_tool(wrapper, wrapper.tool_arguments)
    assert state.tool_calls == [] and encoder.calls == []


@pytest.mark.asyncio
async def test_provided_tool_error_is_not_empty_evidence(existing, monkeypatch):
    store, encoder = existing
    state = KnowledgeRunContext(retriever_for(store))
    async def broken(*args):
        raise TimeoutError('offline simulated retrieval error')
    monkeypatch.setattr(sdk_rag, 'retrieve_context', broken)
    wrapper = tool_context(state, QUESTION)
    with pytest.raises(TimeoutError):
        await search_knowledge.on_invoke_tool(wrapper, wrapper.tool_arguments)
    assert state.context is None and state.tool_calls[0]['status'] == 'failed'
    assert state.tool_calls[0]['error_type'] == 'TimeoutError' and encoder.calls == []


@DAY6
@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['normal', 'no_match', 'budget'])
async def test_retrieve_context_reuses_day5_contract(existing, mode):
    store, encoder = existing
    retriever = (make_retriever(store, filters={'version': 'missing'})
                 if mode == 'no_match' else retriever_for(store))
    hits, context = await retrieve_context(retriever, QUESTION, 1 if mode == 'budget' else 6000)
    assert len(encoder.calls) == (0 if mode == 'no_match' else 1)
    assert all(kind == 'query' for kind, _ in encoder.calls)
    if mode == 'normal':
        assert hits[0]['chunk']['doc_id'] == 'GOOD'
        assert context.sources[0]['chunk']['metadata']['version'] == '2.0'
    else:
        assert context.sources == []
        assert bool(hits) == (mode == 'budget')
        assert bool(context.omitted_chunk_ids) == (mode == 'budget')


@DAY6
def test_agent_registers_tool_and_existing_output_contract():
    agent = build_sdk_rag_agent(ScriptedModel())
    assert agent.tools == [search_knowledge] and agent.output_type is RagAnswer
    assert agent.mcp_servers == [] and agent.model_settings.parallel_tool_calls is False


@DAY6
@pytest.mark.asyncio
async def test_sdk_loop_returns_tool_evidence_to_model(existing, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    store, encoder = existing
    model = scripted_model()
    report = await run_sdk_rag(QUESTION, retriever_for(store), model, output_path=OUTPUT / 'offline/sdk_report.json')
    model.assert_complete()
    assert len(model.calls) == 2 and encoder.calls == [('query', [QUESTION])]
    assert report['execution_status'] == 'succeeded' and report['answer']['status'] == 'answered'
    assert report['tool_calls'][0]['call_id'] == 'call_day6_search'
    outputs = [item for item in model.calls[1].input if item.get('type') == 'function_call_output']
    assert outputs[0]['call_id'] == 'call_day6_search'
    assert json.loads(outputs[0]['output'])['evidence'] == report['context']['sources']
    citation = report['resolved_citations'][0]
    assert citation['chunk']['metadata']['source'] == 'knowledge/GOOD.md'
    assert citation['chunk']['metadata']['version'] == '2.0'
    assert report['usage']['requests'] == 2  # 脚本报告值，未产生真实模型调用。


@DAY6
@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['no_match', 'budget'])
async def test_empty_evidence_still_finishes_sdk_loop(existing, mode):
    store, encoder = existing
    retriever = (make_retriever(store, filters={'version': 'missing'})
                 if mode == 'no_match' else retriever_for(store))
    model = scripted_model()
    report = await run_sdk_rag(QUESTION, retriever, model, max_context_chars=1 if mode == 'budget' else 6000)
    assert report['answer']['status'] == 'insufficient' and report['resolved_citations'] == []
    assert len(model.calls) == 2 and model.remaining_steps == 0
    assert len(encoder.calls) == (0 if mode == 'no_match' else 1)


@DAY6
@pytest.mark.asyncio
async def test_answer_without_retrieval_is_rejected_and_recorded(existing, tmp_path):
    store, _ = existing
    model = ScriptedModel([[assistant_message(json.dumps(answer_for([])))]])
    path = tmp_path / 'no_tool.json'
    with pytest.raises(ValueError, match='完成本次知识检索'):
        await run_sdk_rag(QUESTION, retriever_for(store), model, output_path=path)
    report = json.loads(path.read_text())
    assert report['error'] == dict(stage='citation_validation', type='ValueError')
    assert report['execution_status'] == 'failed' and report['tool_calls'] == []


@DAY6
@pytest.mark.asyncio
@pytest.mark.parametrize('problem', ['unknown_source', 'invented_quote'])
async def test_bad_citation_preserves_answer_and_actual_evidence(existing, problem, tmp_path):
    store, _ = existing
    raw = dict(status='answered', claims=[dict(text='错误引用', citations=[dict(
        source_id='S99' if problem == 'unknown_source' else 'S1', quote='旧版说明不应参与本轮回答。',
    )])], missing_information='')
    path = tmp_path / f'{problem}.json'
    with pytest.raises(ValueError):
        await run_sdk_rag(QUESTION, retriever_for(store), scripted_model(final=raw), output_path=path)
    report = json.loads(path.read_text())
    assert report['error']['stage'] == 'citation_validation' and report['answer'] == raw
    assert report['context']['sources'][0]['chunk']['doc_id'] == 'GOOD'


@DAY6
@pytest.mark.asyncio
async def test_second_search_stops_before_more_embedding(existing, tmp_path):
    store, encoder = existing
    model = ScriptedModel([
        [function_call('search_knowledge', {'query': QUESTION}, call_id='first')],
        [function_call('search_knowledge', {'query': QUESTION}, call_id='second')],
    ])
    path = tmp_path / 'second_search.json'
    # SDK 会把普通工具异常包装为 UserError；仍应停止，而不是让模型盲目重试。
    with pytest.raises(UserError, match='只允许调用一次'):
        await run_sdk_rag(QUESTION, retriever_for(store), model, output_path=path)
    report = json.loads(path.read_text())
    assert len(encoder.calls) == 1 and len(report['tool_calls']) == 1
    assert report['error']['stage'] == 'sdk_run'


@DAY6
@pytest.mark.asyncio
async def test_runs_have_separate_local_context(existing):
    store, _ = existing
    first = await run_sdk_rag(QUESTION, retriever_for(store), scripted_model())
    before = copy.deepcopy(first)
    second = await run_sdk_rag(QUESTION, make_retriever(store, filters={'version': 'missing'}), scripted_model())
    assert first == before and len(first['context']['sources']) == 1
    assert second['context']['sources'] == [] and len(second['tool_calls']) == 1


@DAY6
@pytest.mark.asyncio
async def test_direct_chain_and_sdk_tool_share_context(existing):
    store, encoder = existing
    async def generate(messages):
        evidence = json.loads(messages[-1]['content'].split('检索证据（JSON 数据）：\n', 1)[1])['evidence']
        return json.dumps(answer_for(evidence), ensure_ascii=False)
    direct = await run_rag(QUESTION, store, generate, k=1, filters={'knowledge_status': 'current'})
    sdk = await run_sdk_rag(QUESTION, retriever_for(store), scripted_model())
    assert direct['context'] == sdk['context'] and direct['answer'] == sdk['answer']
    assert direct['resolved_citations'] == sdk['resolved_citations']
    assert encoder.calls == [('query', [QUESTION]), ('query', [QUESTION])]
    _save_json(OUTPUT / 'offline/comparison_report.json', dict(
        mode='offline_fixture', direct=direct, sdk=sdk,
        explanation='每条路径各编码一次问题；直接链调用一次假生成器，SDK 脚本执行两次模型步骤。',
    ))


@REAL
@pytest.mark.asyncio
@pytest.mark.parametrize('name,question,status', [
    ('grounded', QUESTION, 'answered'),
    ('realtime', '任务 12345 现在已经完成了吗？请给出当前状态。', 'insufficient'),
])
async def test_real_direct_chain_and_sdk_tool(name, question, status):
    """真实对照：相同数据/模型名称，分别运行直接链与 SDK 工具闭环。"""
    import asyncio
    from openai import AsyncOpenAI
    assert os.environ.get('OPENAI_API_KEY'), '需要项目既有 DeepSeek OPENAI_API_KEY'
    dataset = Path(os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))
    default_store = Path(__file__).resolve().parents[1] / 'langchain_learning/artifacts/day5/local_model/knowledge_store.json'
    store_path = Path(os.environ.get('WEEK8_DAY6_STORE', str(default_store)))
    store, action = prepare_store(dataset, store_path, Week7Embeddings(OllamaEncoder()), embedding_mode='local_model')
    root = OUTPUT / 'real_model'
    direct = await asyncio.wait_for(run_rag(question, store, DeepSeekGenerator(),
        filters={'knowledge_status': 'current'}, max_context_chars=12000,
        output_path=root / f'{name}_direct.json'), timeout=180)
    async with AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.deepseek.com',
                           timeout=60.0, max_retries=0) as client:
        # 沿用项目现有 SDK 适配方式；不新建另一份密钥或模型服务。
        model = OpenAIResponsesModel(model='deepseek-flash', openai_client=client)
        sdk = await asyncio.wait_for(run_sdk_rag(question, make_retriever(store, filters={'knowledge_status': 'current'}),
            model, max_context_chars=12000, output_path=root / f'{name}_sdk.json'), timeout=180)
    path = root / f'{name}_comparison.json'
    _save_json(path, dict(mode='real_model', store_action=action, direct=direct, sdk=sdk,
                          same_context=direct['context'] == sdk['context']))
    assert direct['answer']['status'] == sdk['answer']['status'] == status
    assert len(sdk['tool_calls']) == 1 and sdk['tool_calls'][0]['status'] == 'succeeded'
    for report in (direct, sdk):
        assert bool(report['resolved_citations']) == (status == 'answered')
        assert all(row['chunk']['metadata']['knowledge_status'] == 'current'
                   and row['chunk']['metadata']['version'] == '2.0'
                   for row in report['resolved_citations'])
    print(f'Day6 真实对照产物：{path}；仍需阅读回答与原文，耗时不作为性能结论')
