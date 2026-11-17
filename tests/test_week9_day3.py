"""Week9 Day3：关键词、精确标识符、RRF 与三策略对照；无需填写测试代码。

两个学习步骤已完成；默认检查基础部分，RUN_WEEK9_DAY3=1 启用完整离线验收。
真实 Ollama 另需 MODEL=1；不请求回答型 LLM。逐项说明见 rag/week9.md#day3。
"""

import copy
import json
import os
from pathlib import Path
import shutil

import pytest

from rag.eval_dataset import DEFAULT_DATASET, validate_benchmark
from rag.hybrid_search import (
    STRATEGIES, SharedQueryEncoder, identifier_constraints, keyword_search, make_search_retriever,
    matches_identifiers, prepare_search_store, reciprocal_rank_fusion, run_search_comparison, tokenize,
)


DATASET = Path(os.environ.get('WEEK9_RAG_DATASET', str(DEFAULT_DATASET)))
OUTPUT = Path(os.environ.get('WEEK9_DAY3_OUTPUT', str(DATASET / 'runs/day3')))
DAY3 = pytest.mark.skipif(os.environ.get('RUN_WEEK9_DAY3') != '1', reason='完整离线验收需 RUN_WEEK9_DAY3=1')
MODEL = pytest.mark.skipif(any(os.environ.get(key) != '1' for key in ('RUN_WEEK9_DAY3', 'RUN_WEEK9_DAY3_MODEL')),
                          reason='真实检索需同时显式开启 DAY3 与 MODEL')


@pytest.fixture(autouse=True)
def no_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


@pytest.fixture
def examples():
    path = DATASET / 'identifier_examples_day3.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['schema_version'] == 1 and data['mode'] == 'offline_fixture'
    assert len(data['chunks']) == 8 and len(data['examples']) == 9
    return data


class FixedEncoder:
    """人工二维向量：故意让相似流程排在目标前面，不具备语义推理能力。"""
    model, query_instruction = 'day3-fixture-not-a-model', 'fixed-query'

    def __init__(self):
        self.calls = []

    def encode_document(self, texts):
        self.calls.append(('document', list(texts)))
        return [[1.0, 0.0] if 'import_standard' in text else
                [0.95, 0.1] if 'import_std_extra' in text else
                [0.8, 0.6] if 'import_std' in text else [0.1, 1.0] for text in texts]

    def encode_query(self, texts):
        self.calls.append(('query', list(texts)))
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def tiny_store(examples):
    pytest.importorskip('langchain_core')
    from rag.langchain_preparation import Week7Embeddings, chunk_document
    from rag.langchain_rag import build_store
    from rag.preprocessing import KnowledgeChunk
    encoder = FixedEncoder()
    documents = [chunk_document(KnowledgeChunk(**chunk)) for chunk in examples['chunks']]
    store = build_store(documents, Week7Embeddings(encoder))
    encoder.calls.clear()
    return store, encoder


def _hit(chunk, score=0.5):
    return dict(score=score, chunk=copy.deepcopy(chunk))


def test_provided_tokenizer_preserves_complete_identifier_and_chinese_bigrams():
    tokens = tokenize('NOT_FOUND_EXTRA import_std 中文问题 2.0')
    assert {'NOT_FOUND_EXTRA', 'import_std', '中文', '文问', '问题', '2.0'} <= tokens
    assert 'NOT_FOUND' not in tokens and 'import' not in tokens
    assert 'NOT_FOUND' not in tokenize('not_found _NOT_FOUND xNOT_FOUND')
    assert tokenize('  !!! ') == set()


def test_provided_nine_identifier_examples_use_values_not_substrings(examples):
    before = copy.deepcopy(examples)
    for case in examples['examples']:
        constraints = identifier_constraints(case['input'])
        actual = {chunk['doc_id'] for chunk in examples['chunks'] if matches_identifiers(chunk, constraints)}
        assert actual == set(case['expected_doc_ids']), case['case_id']
    assert examples == before
    explicit = identifier_constraints(examples['examples'][1]['input'])
    assert explicit['engine_codes'] == ['import_standard'] and explicit['engine_source'] == 'filters'


@pytest.mark.parametrize('bad_k', [0, -1, True, '3'])
def test_provided_keyword_rejects_invalid_k_before_scoring(bad_k):
    with pytest.raises(ValueError):
        keyword_search('NOT_FOUND', [], k=bad_k)


def test_provided_empty_keyword_and_duplicate_chunks_are_handled(examples):
    assert keyword_search('!!!', examples['chunks']) == []
    assert keyword_search('NOT_FOUND', []) == []
    with pytest.raises(ValueError, match='唯一 ID'):
        keyword_search('NOT_FOUND', [examples['chunks'][0], examples['chunks'][0]])


@pytest.mark.parametrize('options', [{'k': 0}, {'k': True}, {'rank_constant': 0}, {'rank_constant': -1}, {'rank_constant': 1.5}])
def test_provided_rrf_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({'dense': []}, **options)


def test_provided_empty_rrf_and_bad_raw_score(examples):
    assert reciprocal_rank_fusion({'dense': [], 'keyword': []}) == []
    with pytest.raises(ValueError, match='有限数值'):
        reciprocal_rank_fusion({'dense': [_hit(examples['chunks'][0], float('nan'))]})


def test_provided_shared_query_encoder_reuses_vectors_without_aliasing():
    base = FixedEncoder()
    encoder = SharedQueryEncoder(base)
    first = encoder.encode_query(['q1', 'q1'])
    first[0][0] = 999
    second = encoder.encode_query(['q1', 'q2'])
    assert second == [[1.0, 0.0], [1.0, 0.0]]
    assert base.calls == [('query', ['q1']), ('query', ['q2'])]
    assert encoder.encode_query([]) == []


def test_provided_unknown_identifiers_short_circuit_both_branches(tiny_store, examples):
    store, encoder = tiny_store
    request = examples['examples'][2]['input']  # import_stdx 不存在。
    for strategy in ('keyword', 'hybrid'):
        assert make_search_retriever(store, strategy)(request, 3) == []
    assert encoder.calls == []


@pytest.mark.parametrize('options', [{'strategy': 'unknown'}, {'candidate_k': 0}, {'rank_constant': 0}])
def test_provided_invalid_search_configuration_never_encodes(tiny_store, examples, options):
    store, encoder = tiny_store
    with pytest.raises(ValueError):
        make_search_retriever(store, options.get('strategy', 'dense'),
            **{key: value for key, value in options.items() if key != 'strategy'})
    retrieve = make_search_retriever(store, 'dense', candidate_k=3)
    with pytest.raises(ValueError, match='不能超过'):
        retrieve(examples['examples'][0]['input'], 4)
    assert encoder.calls == []


def test_provided_store_loading_keeps_snapshot_and_document_vectors(tmp_path):
    pytest.importorskip('langchain_core')
    base = FixedEncoder()
    encoder = SharedQueryEncoder(base)
    _, info = prepare_search_store(DATASET, tmp_path, encoder, embedding_mode='offline_fixture')
    assert info['store_action'] == 'built'
    base.calls.clear()
    _, info = prepare_search_store(DATASET, tmp_path, encoder, embedding_mode='offline_fixture')
    assert info['store_action'] == 'loaded' and base.calls == []
    assert validate_benchmark(DATASET)['fingerprint'] == '31c20e959a1b4f0cf11af6a0c3ea5776640774f5446d2970994737c45a542143'


@DAY3
def test_keyword_coverage_scores_exact_terms_and_owns_returned_chunks(examples):
    chunks = examples['chunks']
    before = copy.deepcopy(chunks)
    results = keyword_search('NOT_FOUND 含义', chunks)
    assert [hit['chunk']['doc_id'] for hit in results] == ['NOT_FOUND']
    assert results[0]['score'] == 0.5
    assert results[0]['retrieval']['matched_tokens'] == ['NOT_FOUND']
    results[0]['chunk']['metadata']['version'] = 'changed'
    assert chunks == before


@DAY3
def test_rrf_accumulates_ranks_and_keeps_original_scores(examples):
    a, b, c = examples['chunks'][:3]
    rankings = {'dense': [_hit(a, 0.99), _hit(b, 0.8)], 'keyword': [_hit(b, 0.2), _hit(c, 0.1)]}
    before = copy.deepcopy(rankings)
    results = reciprocal_rank_fusion(rankings, k=3, rank_constant=60)
    assert [row['chunk']['doc_id'] for row in results] == ['SIM', 'STD', 'EXTRA']
    assert results[0]['score'] == pytest.approx(1 / 62 + 1 / 61)
    assert results[0]['retrieval']['ranks'] == {'dense': 2, 'keyword': 1}
    assert results[0]['retrieval']['raw_scores'] == {'dense': 0.8, 'keyword': 0.2}
    assert rankings == before
    tied = reciprocal_rank_fusion({'dense': [_hit(a)], 'keyword': [_hit(b)]})
    assert [row['chunk']['chunk_id'] for row in tied] == sorted([a['chunk_id'], b['chunk_id']])


@DAY3
def test_rrf_deduplicates_each_branch_but_preserves_rank_positions(examples):
    a, b = examples['chunks'][:2]
    results = reciprocal_rank_fusion({'dense': [_hit(a), _hit(a), _hit(b)]}, k=3)
    assert len(results) == 2
    assert results[0]['score'] == pytest.approx(1 / 61)
    assert results[1]['score'] == pytest.approx(1 / 63)
    other = copy.deepcopy(a)
    other['chunk_id'] += '-second'
    assert len(reciprocal_rank_fusion({'dense': [_hit(a), _hit(other)]})) == 2
    conflict = copy.deepcopy(a)
    conflict['text'] = 'conflicting source'
    with pytest.raises(ValueError, match='不一致'):
        reciprocal_rank_fusion({'dense': [_hit(a)], 'keyword': [_hit(conflict)]})


@DAY3
def test_guarded_hybrid_never_reintroduces_similar_engine_from_dense(tiny_store, examples):
    store, encoder = tiny_store
    request = examples['examples'][0]['input']
    dense = make_search_retriever(store, 'dense')(request, 3)
    assert dense[0]['chunk']['doc_id'] == 'SIM'  # 人工向量故意把相似流程排在前面。
    encoder.calls.clear()
    keyword = make_search_retriever(store, 'keyword')(request, 3)
    assert encoder.calls == []
    hybrid = make_search_retriever(store, 'hybrid')(request, 3)
    assert [hit['chunk']['doc_id'] for hit in keyword] == ['STD']
    assert [hit['chunk']['doc_id'] for hit in hybrid] == ['STD']
    assert hybrid[0]['retrieval']['score_kind'] == 'rrf'


@DAY3
def test_offline_thirty_case_three_strategy_reports_exclude_gold(tmp_path, monkeypatch):
    from rag import hybrid_search as module
    store, info = prepare_search_store(DATASET, tmp_path / 'index', SharedQueryEncoder(FixedEncoder()),
                                        embedding_mode='offline_fixture')
    original, calls = module.make_search_retriever, []

    def recording_factory(store, strategy, **options):
        retrieve = original(store, strategy, **options)
        def recorded(request, k):
            calls.append((strategy, copy.deepcopy(request), k))
            return retrieve(request, k)
        return recorded

    monkeypatch.setattr(module, 'make_search_retriever', recording_factory)
    report = run_search_comparison(DATASET, store, tmp_path / 'report', mode='offline_fixture', backend_info=info)
    assert len(calls) == 90 and all(set(request) == {'question', 'filters', 'access_context'} for _, request, _ in calls)
    assert set(report['strategies']) == set(STRATEGIES) and report['generation_called'] is False
    for summary in report['strategies'].values():
        assert summary['completed_cases'] == 30 and summary['execution_error_count'] == 0
        assert summary['not_applicable_cases'] == 8
        assert all(item['evaluated_cases'] == 22 for item in summary['by_k'].values())
    output = OUTPUT / 'offline'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path / 'report', output, dirs_exist_ok=True)
    print(f'Day3 人工向量流程验收产物：{output}；不代表语义质量')


@MODEL
def test_real_ollama_three_strategy_comparison():
    from rag.embeddings import OllamaEncoder
    output = OUTPUT / 'local_model'
    rebuild = os.environ.get('WEEK9_DAY3_REBUILD') == '1'
    day2_index = DATASET / 'runs/day2/local_model/index'
    default_index = day2_index if (day2_index / 'knowledge_store.json').exists() and not rebuild else output / 'index'
    index = Path(os.environ.get('WEEK9_DAY3_INDEX', str(default_index)))
    store, info = prepare_search_store(DATASET, index, SharedQueryEncoder(OllamaEncoder()), rebuild=rebuild)
    info['index_path'] = str(index)
    report = run_search_comparison(DATASET, store, output, mode='local_model', backend_info=info)
    assert set(report['strategies']) == set(STRATEGIES) and report['generation_called'] is False
    assert all(summary['completed_cases'] == 30 and summary['execution_error_count'] == 0
               for summary in report['strategies'].values())
    print(f'Day3 真实三策略对照：{output}；请阅读 Recall/MRR 和不适用来源，不预设混合策略胜出')
