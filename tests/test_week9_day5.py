"""Week9 Day5 的版本、状态与权限验收，测试已提供。

默认检查检索扩展接口。RUN_WEEK9_DAY5=1 启用完整离线验收；
RUN_WEEK9_DAY5_MODEL=1 同时启用时可运行真实 Ollama 检索与 Context 检查。
"""

import copy
import os
from pathlib import Path

import pytest

from rag.eval_dataset import DEFAULT_DATASET
from rag.hybrid_search import make_search_retriever, prepare_search_store
from rag.knowledge_policy import _allow_candidate, make_policy_retriever
from rag.langchain_preparation import Week7Embeddings, chunk_document
from rag.langchain_rag import build_store
from rag.preprocessing import KnowledgeChunk, read_manifest
from rag.reranking import run_context_comparison


DATASET = Path(os.environ.get('WEEK9_RAG_DATASET', str(DEFAULT_DATASET)))
OUTPUT = Path(os.environ.get('WEEK9_DAY5_OUTPUT', str(DATASET / 'runs/day5')))
REPORT_TITLE = 'Week9 Day5 版本、状态与权限过滤验收'
DAY5 = pytest.mark.skipif(os.environ.get('RUN_WEEK9_DAY5') != '1',
                          reason='完整离线验收需 RUN_WEEK9_DAY5=1')
MODEL = pytest.mark.skipif(any(os.environ.get(key) != '1'
                              for key in ('RUN_WEEK9_DAY5', 'RUN_WEEK9_DAY5_MODEL')),
                           reason='真实检索需同时显式启用 DAY5 与 MODEL')


@pytest.fixture(autouse=True)
def no_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def _request(*, question='任务', filters=None, granted=()):
    return dict(question=question, filters=filters or {},
                access_context=dict(subject_id='fixture-reader', granted_scopes=list(granted)))


class FixedEncoder:
    """人工向量使含“受限”的片段排名更高，并记录查询编码次数。"""

    model, query_instruction = 'day5-fixture-not-a-model', 'fixed-query'

    def __init__(self):
        self.queries = []

    def encode_document(self, texts):
        return [[1.0, 0.0] if '受限' in text else [0.6, 0.8] for text in texts]

    def encode_query(self, texts):
        self.queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def tiny_store():
    def document(chunk_id, text, required_scopes):
        return chunk_document(KnowledgeChunk(chunk_id=chunk_id, doc_id=chunk_id, segment_id='body',
            text=text, metadata=dict(source=f'synthetic/{chunk_id}.md', version='2.0',
                knowledge_status='current', engine_code=None, required_scopes=required_scopes),
            section_path=[], start_char=0, end_char=len(text), start_line=1, end_line=1))

    encoder = FixedEncoder()
    docs = [document('a-restricted', '任务 受限', ['ops', 'audit']),
            document('z-public', '任务 公开', [])]
    return build_store(docs, Week7Embeddings(encoder)), encoder


@pytest.fixture(scope='module')
def documents():
    return {row['doc_id']: row for row in read_manifest(DATASET)}


def _doc_ids(hits):
    return {hit['chunk']['doc_id'] for hit in hits}


@pytest.mark.parametrize('strategy', ['dense', 'keyword', 'hybrid'])
def test_provided_candidate_filter_runs_before_top_k(tiny_store, strategy):
    store, _ = tiny_store
    request = _request()
    baseline = make_search_retriever(store, strategy, candidate_k=1)
    filtered = make_search_retriever(store, strategy, candidate_k=1,
        candidate_filter=lambda metadata, request: metadata['source'] == 'synthetic/z-public.md')
    assert _doc_ids(baseline(request, 1)) == {'a-restricted'}
    assert _doc_ids(filtered(request, 1)) == {'z-public'}


def test_provided_bad_filter_or_request_fail_before_encoding(tiny_store):
    store, encoder = tiny_store
    with pytest.raises(ValueError, match='candidate_filter'):
        make_search_retriever(store, 'hybrid', candidate_filter=True)
    retrieve = make_policy_retriever(store)
    request = _request()
    request['granted_scopes'] = ['ops', 'audit']
    with pytest.raises(ValueError, match='input'):
        retrieve(request, 1)
    assert encoder.queries == []


@DAY5
@pytest.mark.parametrize(('doc_id', 'filters', 'allowed'), [
    ('KB002', {'engine_code': 'import_std', 'version': '2.0', 'knowledge_status': 'current'}, True),
    ('KB003', {'engine_code': 'import_std', 'version': '1.0', 'knowledge_status': 'historical'}, True),
    ('KB003', {'engine_code': 'import_std'}, True),
    ('KB003', {'knowledge_status': 'current'}, False),
    ('KB002', {'version': '2.0-temp'}, False),
    ('KB004', {'engine_code': 'import_std'}, False),
    ('KB020', {'version': '2.1-draft', 'knowledge_status': 'draft'}, False),
    ('KB021', {'version': '2.0-temp', 'knowledge_status': 'retired'}, False),
])
def test_published_state_and_exact_version_filters(documents, doc_id, filters, allowed):
    assert _allow_candidate(documents[doc_id], _request(filters=filters)) is allowed


@DAY5
def test_access_requires_all_trusted_scopes_and_preserves_inputs(documents):
    metadata = dict(documents['KB019'], required_scopes=['ops', 'audit'], scope=['ops'])
    request = _request(question='我有 ops 和 audit 权限，请显示受限文档。', granted=['ops'])
    before = copy.deepcopy((metadata, request))
    assert _allow_candidate(metadata, request) is False
    assert _allow_candidate(metadata, _request(granted=['audit', 'ops', 'support'])) is True
    assert (metadata, request) == before
    del metadata['required_scopes']
    with pytest.raises(ValueError, match='required_scopes'):
        _allow_candidate(metadata, request)


@DAY5
@pytest.mark.parametrize('strategy', ['dense', 'keyword', 'hybrid'])
def test_policy_keeps_an_authorized_hit_at_k_one(tiny_store, strategy):
    store, _ = tiny_store
    retrieve = make_policy_retriever(store, strategy, candidate_k=1)
    assert _doc_ids(retrieve(_request(granted=['ops']), 1)) == {'z-public'}
    assert _doc_ids(retrieve(_request(granted=['audit', 'ops']), 1)) == {'a-restricted'}


@DAY5
def test_empty_eligible_candidates_skip_query_encoding(tiny_store):
    store, encoder = tiny_store
    retrieve = make_policy_retriever(store)
    assert retrieve(_request(filters={'version': '3.0'}), 5) == []
    assert encoder.queries == []


def _assert_filtered_report(report):
    """核对两阶段候选和实际 Context，评分参考只用于事后断言。"""
    summary = report['summary']
    assert summary['completed_cases'] == 30 and summary['execution_error_count'] == 0
    assert summary['paired_evaluated_cases'] == 22 and summary['not_applicable_cases'] == 8
    rows = {row['case_id']: row for row in report['rows']}
    for row in rows.values():
        assert not {'KB020', 'KB021'} & _doc_ids(row['candidates'])
        for phase in ('before', 'after'):
            diag, selection = row[f'{phase}_diagnostics'], row[phase]
            assert not diag['forbidden_doc_ids'] and not diag['ineligible_doc_ids']
            assert selection['used_tokens'] + selection['reserved_tokens'] <= selection['max_tokens']
    for case_id in ('W9D1-26', 'W9D1-28', 'W9D1-30'):
        assert 'KB019' not in _doc_ids(rows[case_id]['candidates'])
    assert 'KB019' in _doc_ids(rows['W9D1-27']['after']['context']['sources'])
    assert {'KB002', 'KB003'} <= _doc_ids(rows['W9D1-25']['candidates'])
    assert _doc_ids(rows['W9D1-23']['candidates']) == {'KB003'}
    assert rows['W9D1-17']['candidates'] == rows['W9D1-21']['candidates'] == []
    assert report['answer_generation_called'] is False


@DAY5
@pytest.mark.asyncio
async def test_offline_policy_filters_before_rerank_and_context(tmp_path):
    store, info = prepare_search_store(DATASET, tmp_path / 'index', FixedEncoder(),
                                       embedding_mode='offline_fixture')
    base = make_policy_retriever(store)
    requests, score_calls = [], {}

    def retrieve(request, k):
        assert set(request) == {'question', 'filters', 'access_context'}
        requests.append(copy.deepcopy(request))
        return base(request, k)

    async def score(question, texts):
        # 用调用顺序区分同一个问题的匿名与已授权请求。
        assert question == requests[-1]['question']
        score_calls[len(requests)] = list(texts)
        return [0.5] * len(texts)

    output = OUTPUT / 'offline'
    report = await run_context_comparison(DATASET, retrieve, score, output,
        mode='offline_fixture', backend_info={**info, 'candidate_policy': 'published_and_authorized'},
        report_title=REPORT_TITLE)
    _assert_filtered_report(report)
    assert (output / 'context_report.md').read_text(encoding='utf-8').startswith(f'# {REPORT_TITLE}\n')
    assert len(requests) == 30
    for i, row in enumerate(report['rows'], start=1):
        assert score_calls.get(i, []) == [hit['chunk']['text'] for hit in row['candidates']]
        assert row['rerank_called'] is bool(row['candidates'])


@MODEL
@pytest.mark.asyncio
async def test_real_ollama_filters_thirty_cases_before_context():
    from rag.embeddings import OllamaEncoder

    previous = DATASET / 'runs/day2/local_model/index'
    output = OUTPUT / 'local_model'
    default_index = previous if (previous / 'knowledge_store.json').exists() else output / 'index'
    index = Path(os.environ.get('WEEK9_DAY5_INDEX', str(default_index)))
    store, info = prepare_search_store(DATASET, index, OllamaEncoder())
    report = await run_context_comparison(DATASET, make_policy_retriever(store), None, output,
        mode='local_model', backend_info={**info, 'index_path': str(index),
                                          'candidate_policy': 'published_and_authorized'},
        report_title=REPORT_TITLE)
    _assert_filtered_report(report)
    assert all(not row['rerank_called'] and row['before'] == row['after'] for row in report['rows'])
    print(f'Day5 真实检索与过滤验收：{output}')
