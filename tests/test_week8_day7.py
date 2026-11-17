"""Week8 Day7：同一知识快照上的手写 RAG / LangChain RAG 对照。

三个关键步骤已完成；RUN_WEEK8_DAY7=1 启用完整离线流程，默认检查辅助代码。
真实检索另需 MODEL=1；三题真实生成对照再需 LLM=1。均不在普通 pytest 中联网。
测试代码与数据已提供，无需填写；逐项说明见 langchain_learning/week8.md#day7。
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
from agents.testing.model import ScriptedModel, assistant_message
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from agents.usage import Usage

from rag.comparison import (
    DEFAULT_OUTPUT, OneShotSDKGenerator, code_profile, native_index_from_store,
    render_report, run_comparison, snapshot_id, token_delta,
)
from rag.embeddings import OllamaEncoder
from rag.evaluation import check_versions, document_group_recall, load_cases
from rag.langchain_preparation import DEFAULT_DATASET, Week7Embeddings, chunk_document
from rag.langchain_rag import _save_json, build_store, documents_to_hits, make_retriever, prepare_store
from rag.preprocessing import KnowledgeDocument, make_chunk
from rag.retrieval import search_knowledge


DAY7 = pytest.mark.skipif(os.environ.get('RUN_WEEK8_DAY7') != '1', reason='完整离线验收需 RUN_WEEK8_DAY7=1')
MODEL = pytest.mark.skipif(any(os.environ.get(key) != '1' for key in ('RUN_WEEK8_DAY7', 'RUN_WEEK8_DAY7_MODEL')),
                          reason='真实检索需同时启用 DAY7 和 MODEL')
LLM = pytest.mark.skipif(any(os.environ.get(key) != '1' for key in
                            ('RUN_WEEK8_DAY7', 'RUN_WEEK8_DAY7_MODEL', 'RUN_WEEK8_DAY7_LLM')),
                        reason='真实生成需同时启用 DAY7、MODEL 和 LLM')
OUTPUT = Path(os.environ.get('WEEK8_DAY7_OUTPUT', str(DEFAULT_OUTPUT)))
CASES = load_cases()
REAL_CASES = [case for case in CASES if case['case_id'] in ('W7D6-01', 'W7D6-07', 'W7D6-15')]


@pytest.fixture(autouse=True)
def local_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    monkeypatch.setattr(tracing_setup, 'GLOBAL_TRACE_PROVIDER', DefaultTraceProvider())


class FixedEncoder:
    model, query_instruction = 'day7-fixture-not-a-model', 'fixed-query'

    def __init__(self):
        self.calls, self.last_query = [], None
        self.fail = False

    def encode_document(self, texts):
        self.calls.append(('document', list(texts)))
        return [[index + 1, 1, 2] for index in range(len(texts))]

    def encode_query(self, texts):
        self.calls.append(('query', list(texts)))
        if self.fail:
            raise TimeoutError('offline embedding failure')
        self.last_query = [[1, 0, 0] for _ in texts]
        return copy.deepcopy(self.last_query)


class FixtureGenerator:
    """根据实际已发送的证据构造假回答，不具备语义推理能力。"""
    model = 'fixture-not-a-real-generation-model'

    def __init__(self, usage=True, error=None, raw=None):
        self.usage_enabled, self.error, self.raw = usage, error, raw
        self.last_usage, self.calls = None, []

    async def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        self.last_usage = None
        if self.error is not None:
            raise self.error
        if self.usage_enabled:
            self.last_usage = dict(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        if self.raw is not None:
            return self.raw
        evidence = json.loads(messages[-1]['content'].split('检索证据（JSON 数据）：\n', 1)[1])['evidence']
        first = evidence[0]
        return json.dumps(dict(status='answered', claims=[dict(text='依据本次证据回答。', citations=[dict(
            source_id=first['source_id'], quote=first['chunk']['text'],
        )])], missing_information=''), ensure_ascii=False)


def make_cases():
    result = []
    for case_id, version, status, groups, allowed in [
        ('D7-current', '2.0', 'answered', [['GOOD']], ['2.0']),
        ('D7-history', '1.0', 'answered', [['OLD']], ['1.0']),
        ('D7-empty', 'missing', 'insufficient', [], []),
    ]:
        result.append(dict(case_id=case_id, question='请根据指定版本的知识说明规则。', k=1,
                           filters={'version': version}, expected_status=status, expected_doc_groups=groups,
                           allowed_versions=allowed, allowed_knowledge_statuses=[],
                           reference_points=['EVALUATOR_ONLY_MUST_NOT_ENTER_MESSAGES']))
    return result


@pytest.fixture
def existing():
    chunks = []
    for doc_id, version, status, text in [('OLD', '1.0', 'historical', '旧版规则默认 30。'),
                                         ('GOOD', '2.0', 'current', '当前规则默认 3。')]:
        unit = KnowledgeDocument(doc_id, text, dict(source=f'knowledge/{doc_id}.md', version=version,
            knowledge_status=status, engine_code='import_std', format='markdown', scope=['learning']))
        chunks.append(chunk_document(make_chunk(unit, text, 0, len(text), 0, [])))
    encoder = FixedEncoder()
    store = build_store(chunks, Week7Embeddings(encoder))
    encoder.calls.clear()
    return store, encoder


def test_provided_index_uses_existing_vectors_without_embedding(existing):
    store, encoder = existing
    before = copy.deepcopy(store.store)
    index = native_index_from_store(store)
    assert encoder.calls == [] and store.store == before
    assert [chunk.chunk_id for chunk in index.chunks] == list(store.store)
    assert index.vectors == [row['vector'] for row in store.store.values()]
    index.chunks[0].metadata['scope'].append('changed')
    assert store.store == before


def test_provided_snapshot_tracks_content_and_configuration(existing):
    store, _ = existing
    index = native_index_from_store(store)
    original = snapshot_id(index)
    assert snapshot_id(copy.deepcopy(index)) == original
    index.query_instruction += ' changed'
    assert snapshot_id(index) != original


@pytest.mark.asyncio
async def test_provided_single_generation_adapter_preserves_text_and_usage(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    raw = '{"status":"insufficient","claims":[],"missing_information":"缺少证据"}'
    model = ScriptedModel([[assistant_message(raw)]], default_usage=Usage(
        requests=1, input_tokens=8, output_tokens=3, total_tokens=11))
    generate = OneShotSDKGenerator(model, model_name='scripted-fixture')
    messages = [dict(role='system', content='规则'), dict(role='user', content='问题与证据')]
    assert await generate(messages) == raw
    assert len(model.calls) == 1 and model.calls[0].tools == []
    assert model.calls[0].system_instructions == '规则' and model.calls[0].input == messages[1:]
    assert generate.last_usage['total_tokens'] == 11


@pytest.mark.parametrize('left,right', [({'total_tokens': True}, None), ({'total_tokens': -1}, None),
                                       (None, {'total_tokens': '20'}), ([], None)])
def test_provided_invalid_usage_is_rejected(left, right):
    with pytest.raises(ValueError):
        token_delta(left, right)


def test_provided_report_is_human_readable_and_marks_unknown_values():
    row = dict(execution_status='failed', checks=dict(automatic_passed=False),
               metrics=dict(generation_calls=0, usage=None, elapsed_seconds=0.01),
               error=dict(stage='retrieval_or_context', type='TimeoutError'))
    report = dict(mode='offline_fixture', snapshot_id='fixture-id',
                  cases=[dict(case_id='D7-case', native=row, langchain=copy.deepcopy(row))], code_profile=code_profile())
    text = render_report(report)
    assert '未提供' in text and 'retrieval_or_context' in text and '待复核' in text
    assert '| 文件 | 行数 | 函数数 |' in text and 'rag/answering.py' in text


@DAY7
@pytest.mark.parametrize('left,right,expected', [(None, None, None), ({}, {'total_tokens': 5}, None),
    ({'total_tokens': 4}, None, None), ({'total_tokens': 0}, {'total_tokens': 0}, 0),
    ({'total_tokens': 0}, {'total_tokens': 3}, 3), ({'total_tokens': 7}, {'total_tokens': 2}, -5)])
def test_token_delta_distinguishes_unknown_and_zero(left, right, expected):
    assert token_delta(left, right) == expected


@DAY7
@pytest.mark.asyncio
async def test_both_paths_run_same_cases_and_preserve_sources(existing, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    store, encoder = existing
    before = copy.deepcopy(store.store)
    report = await run_comparison(make_cases(), store, FixtureGenerator, output_dir=OUTPUT / 'offline')
    assert report['mode'] == 'offline_fixture' and report['automatic_passed']
    assert len(report['cases']) == 3 and store.store == before
    assert len(encoder.calls) == 4 and all(kind == 'query' for kind, _ in encoder.calls)
    for pair in report['cases']:
        assert pair['same_ranked_ids'] and pair['same_generation_model']
        assert pair['semantic_review'] == 'fixture_only'
        for name in ('native', 'langchain'):
            row = pair[name]
            assert row['execution_status'] == 'succeeded'
            assert 'EVALUATOR_ONLY' not in json.dumps(row['actual_messages'])
            assert row['metrics']['elapsed_seconds'] >= row['metrics']['generation_seconds'] >= 0
            if pair['case_id'] == 'D7-empty':
                assert row['metrics']['generation_calls'] == 0 and row['metrics']['usage'] is None
            else:
                assert row['details']['resolved_citations'][0]['chunk']['metadata']['version'] in ('1.0', '2.0')
    assert (OUTPUT / 'offline/comparison_report.md').is_file()


@DAY7
@pytest.mark.asyncio
async def test_unreported_usage_is_not_fabricated(existing, tmp_path):
    store, _ = existing
    report = await run_comparison(make_cases()[:1], store, lambda: FixtureGenerator(usage=False), output_dir=tmp_path)
    pair = report['cases'][0]
    assert pair['automatic_passed'] and pair['token_delta'] is None
    assert pair['native']['metrics']['usage'] is None
    assert pair['native']['metrics']['generation_calls'] == 1


@DAY7
@pytest.mark.asyncio
@pytest.mark.parametrize('problem,stage', [('embedding', 'retrieval_or_context'), ('generation', 'generation'),
                                         ('json', 'answer_validation'), ('citation', 'answer_validation')])
async def test_failures_keep_stage_raw_answer_and_continue(existing, problem, stage, tmp_path):
    store, encoder = existing
    if problem == 'embedding':
        encoder.fail = True
    bad_quote = json.dumps(dict(status='answered', claims=[dict(text='错误引用', citations=[dict(
        source_id='S99', quote='不存在')])], missing_information=''))
    def factory():
        return FixtureGenerator(error=TimeoutError('offline generation failure') if problem == 'generation' else None,
                                raw='not JSON' if problem == 'json' else bad_quote if problem == 'citation' else None)
    report = await run_comparison(make_cases()[:1], store, factory, output_dir=tmp_path)
    assert not report['automatic_passed']
    pair = report['cases'][0]
    for name in ('native', 'langchain'):
        assert pair[name]['error']['stage'] == stage and pair[name]['details'] is None
        if problem in ('json', 'citation'):
            assert pair[name]['raw_answer'] and pair[name]['actual_messages']
    assert len(list((tmp_path / 'cases').glob('*.json'))) == 2


@DAY7
@pytest.mark.asyncio
async def test_one_failed_path_does_not_skip_other_paths_or_cases(existing, tmp_path):
    store, _ = existing
    created = []
    def factory():
        generate = FixtureGenerator(error=TimeoutError('first path fails') if not created else None)
        created.append(generate)
        return generate
    report = await run_comparison(make_cases(), store, factory, output_dir=tmp_path)
    assert len(report['cases']) == 3 and not report['automatic_passed']
    assert report['cases'][0]['native']['execution_status'] == 'failed'
    assert report['cases'][0]['langchain']['execution_status'] == 'succeeded'
    assert all(pair['automatic_passed'] for pair in report['cases'][1:])


@DAY7
@pytest.mark.asyncio
async def test_different_generation_models_are_visible(existing, tmp_path):
    store, _ = existing
    created = []
    def factory():
        generate = FixtureGenerator()
        generate.model = 'model-a' if not created else 'model-b'
        created.append(generate)
        return generate
    report = await run_comparison(make_cases()[:1], store, factory, output_dir=tmp_path)
    assert report['cases'][0]['same_generation_model'] is False and not report['automatic_passed']


def retrieval_rows(store, encoder, *, real: bool):
    """提供好的 15 题检索对照；原生路径重放本题向量，排除两次编码的数值差异。"""
    index = native_index_from_store(store)
    originals = {chunk.chunk_id: asdict(chunk) for chunk in index.chunks}
    rows = []
    for case in CASES:
        options = dict(k=case.get('k', 5), filters=case['filters'])
        current = documents_to_hits(make_retriever(store, **options).invoke(case['question']))
        class ReplayEncoder:
            model, query_instruction = encoder.model, encoder.query_instruction
            def encode_query(self, texts):
                return copy.deepcopy(encoder.last_query)
        native = search_knowledge(case['question'], index, ReplayEncoder(), **options)
        assert len(current) == len(native)
        assert all(hit['chunk'] == originals[hit['chunk']['chunk_id']] for hit in current + native)
        assert [hit['score'] for hit in current] == pytest.approx([hit['score'] for hit in native], abs=1e-10)
        current_ids = [hit['chunk']['chunk_id'] for hit in current]
        native_ids = [hit['chunk']['chunk_id'] for hit in native]
        # 同分项可能换序；需要的原文、版本与文档组仍分别检查。
        if current_ids != native_ids:
            assert sorted(current_ids) == sorted(native_ids)
        versions_ok = all(check_versions([hit['chunk'] for hit in hits], case) for hits in (current, native))
        recall = document_group_recall(case['expected_doc_groups'], [hit['chunk'] for hit in current]) if real else None
        rows.append(dict(case_id=case['case_id'], k=options['k'], native=native, langchain=current,
                         same_ranked_ids=current_ids == native_ids, versions_ok=versions_ok,
                         recall=recall, recall_evaluated=real))
        assert versions_ok and recall in (None, 1.0)
    return rows


@DAY7
def test_fifteen_questions_compare_existing_retrievers_without_generation():
    dataset = Path(os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))
    encoder = FixedEncoder()
    store, _ = prepare_store(dataset, OUTPUT / 'offline/corpus/knowledge_store.json', Week7Embeddings(encoder),
                             embedding_mode='offline_fixture', rebuild=True)
    encoder.calls.clear()
    rows = retrieval_rows(store, encoder, real=False)
    assert len(rows) == 15 and all(kind == 'query' for kind, _ in encoder.calls)
    _save_json(OUTPUT / 'offline/retrieval_report.json', dict(mode='offline_fixture', rows=rows))


class RecordingOllama(OllamaEncoder):
    last_query = None
    def encode_query(self, texts):
        self.last_query = super().encode_query(texts)
        return copy.deepcopy(self.last_query)


def real_store():
    dataset = Path(os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))
    default_path = Path(__file__).resolve().parents[1] / 'langchain_learning/artifacts/day5/local_model/knowledge_store.json'
    path = Path(os.environ.get('WEEK8_DAY7_STORE', str(default_path)))
    encoder = RecordingOllama()
    store, _ = prepare_store(dataset, path, Week7Embeddings(encoder), embedding_mode='local_model')
    return store, encoder


@MODEL
def test_real_fifteen_retrieval_comparisons():
    store, encoder = real_store()
    rows = retrieval_rows(store, encoder, real=True)
    path = OUTPUT / 'local_model/retrieval_report.json'
    _save_json(path, dict(mode='local_model', snapshot_id=snapshot_id(native_index_from_store(store)), rows=rows))
    print(f'Day7 真实检索对照：{path}；无生成模型调用')


@LLM
@pytest.mark.asyncio
async def test_real_three_case_framework_comparison():
    import asyncio
    from openai import AsyncOpenAI
    assert os.environ.get('OPENAI_API_KEY'), '需要项目既有 DeepSeek OPENAI_API_KEY'
    store, _ = real_store()
    async with AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.deepseek.com',
                           timeout=60.0, max_retries=0) as client:
        model = OpenAIResponsesModel(model='deepseek-flash', openai_client=client)
        factory = lambda: OneShotSDKGenerator(model, model_name='deepseek-flash')
        report = await asyncio.wait_for(run_comparison(REAL_CASES, store, factory, mode='real_model',
            output_dir=OUTPUT / 'real_model'), timeout=600)
    assert len(report['cases']) == 3 and report['automatic_passed']
    assert all(pair['semantic_review'] == 'needs_review' for pair in report['cases'])
    print(f'Day7 真实周验收：{OUTPUT / "real_model/comparison_report.md"}；需补充人工结论和语义复核')
