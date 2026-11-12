"""Week8 Day5：向量库、Retriever、检索链与 Week7 回答层的连接测试。

三个关键步骤已完成；RUN_WEEK8_DAY5=1 启用完整离线测试。
真实 Ollama 检索另需 RUN_WEEK8_DAY5_MODEL=1，真实 DeepSeek 回答再需 LLM=1。
数据、输入输出和每项测试说明见 langchain_learning/week8.md#day5。
"""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

pytest.importorskip('numpy', reason='请在主项目 .venv 安装 langchain_learning/requirements.txt 中的依赖')
pytest.importorskip('langchain_core')

from langchain_core.vectorstores import InMemoryVectorStore

from rag.answering import DeepSeekGenerator
from rag.embeddings import OllamaEncoder
from rag.evaluation import check_versions, document_group_recall, load_cases
from rag.langchain_preparation import (
    DEFAULT_DATASET, ManifestLoader, Week7Embeddings, chunk_document, split_documents,
)
from rag.langchain_rag import (
    DEFAULT_OUTPUT, build_rag_chain, build_store, document_chunk, documents_to_hits,
    load_store, make_retriever, prepare_store, run_rag, save_store,
)
from rag.preprocessing import KnowledgeDocument, make_chunk
from rag.retrieval import KnowledgeIndex, search_knowledge


DAY5 = pytest.mark.skipif(os.environ.get('RUN_WEEK8_DAY5') != '1', reason='完整离线测试需设置 RUN_WEEK8_DAY5=1')
MODEL = pytest.mark.skipif(
    os.environ.get('RUN_WEEK8_DAY5') != '1' or os.environ.get('RUN_WEEK8_DAY5_MODEL') != '1',
    reason='真实本地检索需另行设置 RUN_WEEK8_DAY5_MODEL=1',
)
LLM = pytest.mark.skipif(
    any(os.environ.get(name) != '1' for name in ('RUN_WEEK8_DAY5', 'RUN_WEEK8_DAY5_MODEL', 'RUN_WEEK8_DAY5_LLM')),
    reason='真实回答另需 RUN_WEEK8_DAY5_LLM=1，且 DAY5/MODEL 同时启用',
)
OUTPUT = Path(os.environ.get('WEEK8_DAY5_OUTPUT', str(DEFAULT_OUTPUT)))
DATASET = Path(os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))
CASES = load_cases()  # 参考项只在测试侧评分，不进入索引或模型消息。


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


class FixedEncoder:
    """人工三维向量，只检查接口/排序；没有语义推理，不代表召回质量。"""
    model = 'fixed-3d-not-a-model'
    query_instruction = 'fixed-fixture'

    def __init__(self, rows=None):
        self.rows, self.calls = rows, []

    def encode_document(self, texts):
        self.calls.append(('document', list(texts)))
        return copy.deepcopy(self.rows) if self.rows is not None else [[number + 1, 1, 2] for number in range(len(texts))]

    def encode_query(self, texts):
        self.calls.append(('query', list(texts)))
        return [[1, 0, 0] for text in texts]


class EvidenceGenerator:
    """从已发送证据构造固定合法回答，仅验证链路和引用，不能评估真实模型质量。"""
    model = 'fixture-not-a-generation-model'

    def __init__(self, raw=None):
        self.raw, self.calls = raw, []

    async def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if self.raw is not None:
            return self.raw
        evidence = json.loads(messages[1]['content'].split('检索证据（JSON 数据）：\n', 1)[1])['evidence']
        source = evidence[0]
        quote = source['chunk']['text']
        return json.dumps(dict(status='answered', claims=[dict(text=quote, citations=[
            dict(source_id=source['source_id'], quote=quote)])], missing_information=''), ensure_ascii=False)


@pytest.fixture
def chunks():
    result = []
    for doc_id, engine, status, version, text in [
        ('OLD', 'import_std', 'historical', '1.0', '历史版本 limit 默认 30。'),
        ('SIM', 'import_standard', 'current', '2.0', '相似流程编码不能混用。'),
        ('GOOD', 'import_std', 'current', '2.0', 'NOT_FOUND 表示未找到，不等于任务执行失败。'),
        ('FAQ', None, 'current', '2.0', '静态文档不能证明任务当前状态。'),
    ]:
        unit = KnowledgeDocument(doc_id, text, dict(source=f'knowledge/{doc_id}.md', format='markdown',
            title=doc_id, version=version, scope=['learning'], updated_at='2026-11-04',
            engine_code=engine, knowledge_status=status))
        result.append(chunk_document(make_chunk(unit, text, 0, len(text), 0, [])))
    return result


@pytest.fixture
def existing(chunks):
    """已提供部分使用 LangChain 直接建人工库，避免依赖学生的 build_store 是否已填写。"""
    encoder = FixedEncoder([[1, 0, 0], [0.99, 0.1, 0], [0.8, 0.6, 0], [0.6, 0.8, 0]])
    store = InMemoryVectorStore(Week7Embeddings(encoder))
    store.add_documents(copy.deepcopy(chunks))
    encoder.calls.clear()
    return store, encoder


def baseline_index(store, encoder):
    rows = list(store.store.values())
    return KnowledgeIndex(encoder.model, encoder.query_instruction,
                          [document_chunk(item) for item in store.get_by_ids(list(store.store))],
                          [copy.deepcopy(row['vector']) for row in rows])


def assert_equivalent(actual, expected, index):
    """NumPy 与标准库浮点容差对照；同分项可换序，不要求框架采用相同 tie-break。"""
    assert len(actual) == len(expected)
    assert [hit['score'] for hit in actual] == pytest.approx([hit['score'] for hit in expected], abs=1e-10)
    by_id = {chunk.chunk_id: asdict(chunk) for chunk in index.chunks}
    for hit in actual:
        assert hit['chunk'] == by_id[hit['chunk']['chunk_id']]
    for new, old in zip(actual, expected):
        if new['chunk']['chunk_id'] != old['chunk']['chunk_id']:
            assert new['score'] == pytest.approx(old['score'], abs=1e-10)


@pytest.mark.parametrize('problem', ['duplicate', 'source', 'score'])
def test_provided_invalid_chunks_are_rejected_before_embedding(chunks, problem):
    if problem == 'duplicate':
        chunks.append(copy.deepcopy(chunks[0]))
    elif problem == 'source':
        chunks[0].metadata.pop('source')
    else:
        chunks[0].metadata['retrieval_score'] = 1.0
    encoder = FixedEncoder()
    with pytest.raises(ValueError):
        build_store(chunks, Week7Embeddings(encoder))
    assert encoder.calls == []


def test_provided_persistence_roundtrip_never_embeds(existing, tmp_path):
    store, encoder = existing
    path = tmp_path / 'knowledge_store.json'
    save_store(store, path, embedding_mode='offline_fixture')
    loaded = load_store(path, Week7Embeddings(encoder), embedding_mode='offline_fixture')
    assert loaded.store == store.store and encoder.calls == []
    assert 'NOT_FOUND' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('change', ['model', 'prefix', 'mode', 'dimension', 'vector', 'version'])
def test_provided_incompatible_or_corrupt_cache_is_rejected(existing, tmp_path, change):
    store, encoder = existing
    path = tmp_path / 'store.json'
    save_store(store, path, embedding_mode='offline_fixture')
    config_path = tmp_path / 'store.config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if change == 'model':
        config['model_name'] = 'another-model'
    elif change == 'prefix':
        config['query_instruction'] = 'another-prefix'
    elif change == 'mode':
        config['embedding_mode'] = 'local_model'
    elif change == 'dimension':
        config['dimensions'] = 999
    elif change == 'version':
        config['schema_version'] = 99
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
        next(iter(data.values()))['vector'] = [0, 0, 0]
        path.write_text(json.dumps(data), encoding='utf-8')
    config_path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError):
        load_store(path, Week7Embeddings(encoder), embedding_mode='offline_fixture')
    assert encoder.calls == []


@pytest.mark.parametrize('options', [{'k': 0}, {'k': True}, {'filters': {'unknown': 'x'}},
                                    {'filters': {'version': ''}}, {'min_score': 2}])
def test_provided_retrieval_config_validation_has_no_model_calls(existing, options):
    store, encoder = existing
    with pytest.raises(ValueError):
        make_retriever(store, **options)
    assert encoder.calls == []


@DAY5
def test_build_store_encodes_once_and_owns_metadata(chunks):
    encoder = FixedEncoder()
    before = copy.deepcopy(chunks)
    store = build_store(chunks, Week7Embeddings(encoder))
    assert encoder.calls == [('document', [item.page_content for item in chunks])]
    assert list(store.store) == [item.id for item in chunks] and chunks == before
    chunks[0].metadata['scope'].append('changed')
    assert store.get_by_ids([before[0].id])[0].metadata['scope'] == ['learning']


@DAY5
def test_retriever_prefilters_exact_metadata_and_only_encodes_query(existing):
    store, encoder = existing
    before = copy.deepcopy(store.store)
    retriever = make_retriever(store, k=1, filters={'engine_code': 'import_std', 'knowledge_status': 'current'})
    docs = retriever.invoke('问题')
    assert [item.metadata['doc_id'] for item in docs] == ['GOOD']
    assert docs[0].metadata['retrieval_score'] == pytest.approx(0.8)
    assert encoder.calls == [('query', ['问题'])] and store.store == before
    hit = documents_to_hits(docs)[0]
    assert hit['chunk']['metadata']['source'] == 'knowledge/GOOD.md'
    assert hit['chunk']['metadata']['version'] == '2.0'
    assert 'retrieval_score' not in hit['chunk']['metadata']


@DAY5
@pytest.mark.asyncio
async def test_rag_chain_keeps_question_context_and_resolved_citations(existing):
    store, encoder = existing
    generator = EvidenceGenerator()
    retriever = make_retriever(store, k=1, filters={'engine_code': 'import_std', 'version': '2.0'})
    chain = build_rag_chain(retriever, generator)
    report = await chain.ainvoke('NOT_FOUND 是任务失败吗？')
    assert report['question'] == 'NOT_FOUND 是任务失败吗？' and report['answer']['status'] == 'answered'
    assert len(generator.calls) == 1 and encoder.calls == [('query', ['NOT_FOUND 是任务失败吗？'])]
    assert [row['role'] for row in report['messages']] == ['system', 'user']
    citation = report['resolved_citations'][0]
    assert citation['chunk']['doc_id'] == 'GOOD' and citation['chunk']['metadata']['version'] == '2.0'
    assert citation['quote'] in citation['chunk']['text']


@DAY5
@pytest.mark.asyncio
@pytest.mark.parametrize('empty_reason', ['filter', 'budget', 'threshold'])
async def test_no_usable_evidence_skips_generation(existing, empty_reason, tmp_path):
    store, encoder = existing
    generator = EvidenceGenerator()
    options = dict(filters={'knowledge_status': 'current', 'engine_code': 'import_std'})
    if empty_reason == 'filter':
        options['filters']['version'] = '3.0'
    elif empty_reason == 'budget':
        options['max_context_chars'] = 1
    else:
        options['min_score'] = 0.9
    report = await run_rag('问题', store, generator, output_path=tmp_path / 'empty.json', **options)
    assert report['answer']['status'] == 'insufficient' and report['generation_called'] is False
    assert report['resolved_citations'] == [] and generator.calls == []
    assert len(encoder.calls) == (0 if empty_reason == 'filter' else 1)


@DAY5
@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['not JSON', json.dumps(dict(status='answered', claims=[
    dict(text='虚构引用', citations=[dict(source_id='S99', quote='不存在的摘录')])], missing_information=''))])
async def test_invalid_answer_propagates_and_keeps_raw_output(existing, raw, tmp_path):
    store, _ = existing
    output = tmp_path / 'failure.json'
    with pytest.raises(ValueError):
        await run_rag('问题', store, EvidenceGenerator(raw), output_path=output)
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['execution_status'] == 'failed' and report['raw_answer'] == raw
    assert report['error']['type'] in ('ValueError', 'ValidationError')


@DAY5
def test_prepare_store_reuses_cache_without_reading_sources_or_embedding(tmp_path, monkeypatch):
    encoder = FixedEncoder()
    path = tmp_path / 'knowledge_store.json'
    store, action = prepare_store(DATASET, path, Week7Embeddings(encoder), embedding_mode='offline_fixture')
    assert action == 'built' and len(store.store) == 64 and len(encoder.calls) == 1
    encoder.calls.clear()
    def unexpected_load(*args, **kwargs):
        raise AssertionError('缓存命中时不能重新加载源文档')
    monkeypatch.setattr('rag.langchain_rag.ManifestLoader', unexpected_load)
    loaded, action = prepare_store(tmp_path / 'missing', path, Week7Embeddings(encoder), embedding_mode='offline_fixture')
    assert action == 'loaded' and loaded.store == store.store and encoder.calls == []


@pytest.fixture(scope='module')
def corpus():
    encoder = FixedEncoder()
    chunks = split_documents(ManifestLoader(DATASET).load())
    store = build_store(chunks, Week7Embeddings(encoder))
    rows = []
    yield store, encoder, baseline_index(store, encoder), rows
    path = OUTPUT / 'offline/retrieval_comparison.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(embedding_mode='offline_fixture', expected_cases=len(CASES),
                                   completed_cases=len(rows), rows=rows),
                               ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


@DAY5
@pytest.mark.parametrize('case', CASES, ids=[case['case_id'] for case in CASES])
def test_fifteen_week7_questions_match_baseline_with_same_fixture_vectors(corpus, case):
    store, encoder, index, rows = corpus
    encoder.calls.clear()
    k = case.get('k', 5)
    actual = documents_to_hits(make_retriever(store, k=k, filters=case['filters']).invoke(case['question']))
    expected = search_knowledge(case['question'], index, encoder, k=k, filters=case['filters'])
    assert_equivalent(actual, expected, index)
    assert check_versions([hit['chunk'] for hit in actual], case)
    assert all(kind == 'query' for kind, texts in encoder.calls)
    rows.append(dict(case_id=case['case_id'], k=k, question=case['question'], filters=case['filters'],
                     chunk_ids=[hit['chunk']['chunk_id'] for hit in actual],
                     sources=[dict(source=hit['chunk']['metadata']['source'],
                                   version=hit['chunk']['metadata']['version']) for hit in actual],
                     same_ranked_ids=[h['chunk']['chunk_id'] for h in actual] == [h['chunk']['chunk_id'] for h in expected]))


@MODEL
def test_local_ollama_retrieval_on_fifteen_questions():
    class RecordingOllama(OllamaEncoder):
        last_query = None
        def encode_query(self, texts):
            rows = super().encode_query(texts)
            self.last_query = rows
            return rows
    encoder = RecordingOllama()
    store, action = prepare_store(DATASET, OUTPUT / 'local_model/knowledge_store.json', Week7Embeddings(encoder),
                                  embedding_mode='local_model')
    index = baseline_index(store, encoder)
    rows = []
    path = OUTPUT / 'local_model/retrieval_report.json'
    for case in CASES:
        print(f'正在检索 {case["case_id"]}')
        k = case.get('k', 5)
        hits = documents_to_hits(make_retriever(store, k=k, filters=case['filters']).invoke(case['question']))
        # 重放刚才的问题向量，避免为了 Week7 对照再次请求真实模型。
        class ReplayEncoder:
            model, query_instruction = encoder.model, encoder.query_instruction
            def encode_query(self, texts):
                return copy.deepcopy(encoder.last_query)
        old = search_knowledge(case['question'], index, ReplayEncoder(), k=k, filters=case['filters'])
        assert_equivalent(hits, old, index)
        recall = document_group_recall(case['expected_doc_groups'], [hit['chunk'] for hit in hits])
        versions_ok = check_versions([hit['chunk'] for hit in hits], case)
        rows.append(dict(case_id=case['case_id'], k=k, hits=hits, recall=recall, versions_ok=versions_ok,
                         same_ranked_ids=[h['chunk']['chunk_id'] for h in hits] == [h['chunk']['chunk_id'] for h in old]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(store_action=action, rows=rows), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failed = [row['case_id'] for row in rows if not row['versions_ok'] or row['recall'] not in (None, 1.0)]
    assert not failed, failed
    print(f'Day5 本地真实检索产物：{path}；没有调用回答型 LLM')


@LLM
@pytest.mark.asyncio
@pytest.mark.parametrize('name,question,status', [
    ('grounded', 'NOT_FOUND 是不是表示任务执行失败？', 'answered'),
    ('realtime', '任务 12345 现在已经完成了吗？请给出当前状态。', 'insufficient'),
])
async def test_real_rag_answers_use_existing_generation_layer(name, question, status):
    assert os.environ.get('OPENAI_API_KEY'), '需要项目既有 DeepSeek OPENAI_API_KEY'
    store, action = prepare_store(DATASET, OUTPUT / 'local_model/knowledge_store.json', Week7Embeddings(OllamaEncoder()),
                                  embedding_mode='local_model')
    path = OUTPUT / f'real_model/{name}_report.json'
    report = await run_rag(question, store, DeepSeekGenerator(), filters={'knowledge_status': 'current'},
                            max_context_chars=12000, output_path=path)
    assert report['execution_status'] == 'succeeded' and report['answer']['status'] == status
    if status == 'answered':
        assert report['resolved_citations']
        assert all(row['chunk']['metadata']['source'] and row['chunk']['metadata']['version'] == '2.0'
                   for row in report['resolved_citations'])
    else:
        assert report['resolved_citations'] == []
    print(f'Day5 真实回答产物：{path}；索引 {action}，回答支持性需阅读复核')
