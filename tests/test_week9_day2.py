"""Week9 Day2：指标算例、排名诊断、30 题离线报告和显式真实 Ollama 基线。

两个指标学习步骤已完成；设置 RUN_WEEK9_DAY2=1 启用完整离线验收。
普通 pytest 不请求模型。来源、公式和每项测试说明见 rag/week9.md#day2。
"""

from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from rag.eval_dataset import DEFAULT_DATASET, load_benchmark, validate_benchmark
from rag.preprocessing import load_documents, read_manifest, split_document
from rag.retrieval_metrics import (
    DEFAULT_KS, _diagnostics, _validate_hits, load_rankings_fixture, prepare_langchain_baseline,
    recall_at_k, reciprocal_rank_at_k, render_metrics_report, run_retrieval_evaluation, summarize_rows,
)


DATASET = Path(os.environ.get('WEEK9_RAG_DATASET', str(DEFAULT_DATASET)))
OUTPUT = Path(os.environ.get('WEEK9_DAY2_OUTPUT', str(DATASET / 'runs/day2')))
DAY2 = pytest.mark.skipif(os.environ.get('RUN_WEEK9_DAY2') != '1', reason='完整离线验收需 RUN_WEEK9_DAY2=1')
MODEL = pytest.mark.skipif(any(os.environ.get(name) != '1' for name in ('RUN_WEEK9_DAY2', 'RUN_WEEK9_DAY2_MODEL')),
                          reason='真实检索需同时显式启用 DAY2 与 MODEL')


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


@pytest.fixture
def fixture_retrieval():
    """把固定的 doc_id 排名展开成实际知识 Chunk；人工排名没有语义检索能力。"""
    cases, rankings = load_benchmark(DATASET)['cases'], load_rankings_fixture(DATASET)
    chunks = defaultdict(list)
    for unit in load_documents(DATASET):
        for chunk in split_document(unit):
            chunks[chunk.doc_id].append(asdict(chunk))
    mapping = {json.dumps(case['input'], sort_keys=True): rankings[case['case_id']] for case in cases}
    calls = []

    def retrieve(request, k):
        calls.append(copy.deepcopy((request, k)))
        doc_ids = mapping[json.dumps(request, sort_keys=True)][:k]
        used, hits = defaultdict(int), []
        for rank, doc_id in enumerate(doc_ids, 1):
            chunk = copy.deepcopy(chunks[doc_id][used[doc_id]])
            used[doc_id] += 1
            hits.append(dict(score=1.0 / rank, chunk=chunk))
        return hits

    return cases, retrieve, calls


def test_provided_rankings_cover_thirty_frozen_cases_and_include_failure_examples():
    rankings = load_rankings_fixture(DATASET)
    assert len(rankings) == 30
    assert 'KB005' not in rankings['W9D1-07']
    assert rankings['W9D1-25'] == ['KB001', 'KB002', 'KB002', 'KB005', 'KB003']
    assert rankings['W9D1-17'] == []
    assert 'KB019' in rankings['W9D1-26']
    assert validate_benchmark(DATASET)['fingerprint'] == '31c20e959a1b4f0cf11af6a0c3ea5776640774f5446d2970994737c45a542143'


@pytest.mark.parametrize('groups,ranked,k', [
    ([['A']], ['A'], 0), ([['A']], ['A'], -1), ([['A']], ['A'], True),
    ([['A']], ['A'], '3'), ([['A']], ['A'], 1.5), ('A', ['A'], 1),
    ([[]], ['A'], 1), ([['']], ['A'], 1), ([['A', 'A']], ['A'], 1),
    ([['A'], ['A']], ['A'], 1), ([['A']], 'A', 1), ([['A']], [None], 1),
])
def test_provided_metric_inputs_reject_invalid_k_groups_and_rankings(groups, ranked, k):
    for metric in (recall_at_k, reciprocal_rank_at_k):
        with pytest.raises(ValueError):
            metric(groups, ranked, k)


def test_provided_no_gold_is_not_applicable_even_with_returned_chunks():
    assert recall_at_k([], ['BACKGROUND'], 3) is None
    assert reciprocal_rank_at_k([], [], 3) is None


def test_provided_summary_keeps_denominators_errors_and_none_separate():
    rows = [dict(execution_error=None, by_k={'3': dict(recall_at_k=value, rr_at_k=value)})
            for value in (0.5, 1.0, None)]
    rows.append(dict(execution_error={'type': 'TimeoutError'}, by_k={}))
    result = summarize_rows(rows, [3], planned_cases=4, planned_applicable=3)
    assert result['by_k']['3'] == dict(mean_recall_at_k=0.75, mrr_at_k=0.75,
                                      evaluated_cases=2, full_recall_cases=1)
    assert result['planned_applicable_cases'] == 3 and result['not_applicable_cases'] == 1
    assert result['execution_error_count'] == 1
    assert summarize_rows([], [3], planned_cases=30, planned_applicable=22)['by_k']['3']['mrr_at_k'] is None


def test_provided_diagnostics_expose_misses_and_forbidden_sources(fixture_retrieval):
    cases, retrieve, _ = fixture_retrieval
    missed = next(case for case in cases if case['case_id'] == 'W9D1-25')
    partial = _diagnostics(missed, retrieve(missed['input'], 3))
    assert partial['missed_doc_groups'] == [['KB003']]
    denied = next(case for case in cases if case['case_id'] == 'W9D1-26')
    result = _diagnostics(denied, retrieve(denied['input'], 3))
    assert result['forbidden_doc_ids'] == result['ineligible_doc_ids'] == ['KB019']
    assert result['non_target_doc_ids'] is None  # 拒答题不能把所有背景来源都判为错召回。


@pytest.mark.parametrize('problem', ['unknown_doc', 'duplicate_chunk', 'changed_version', 'nan_score'])
def test_provided_bad_retrieval_results_are_rejected(fixture_retrieval, problem):
    cases, retrieve, _ = fixture_retrieval
    hits = retrieve(cases[0]['input'], 3)
    if problem == 'unknown_doc':
        hits[0]['chunk']['doc_id'] = 'MISSING'
    elif problem == 'duplicate_chunk':
        hits.append(copy.deepcopy(hits[0]))
    elif problem == 'changed_version':
        hits[0]['chunk']['metadata']['version'] = 'invented'
    else:
        hits[0]['score'] = float('nan')
    with pytest.raises(ValueError):
        _validate_hits(hits, {row['doc_id']: row for row in read_manifest(DATASET)})


def test_provided_retrieval_errors_are_saved_without_faking_zero_metrics(tmp_path):
    def unavailable(request, k):
        assert set(request) == {'question', 'filters', 'access_context'}
        raise TimeoutError('potentially-sensitive-body-not-to-be-recorded')

    result = run_retrieval_evaluation(DATASET, unavailable, tmp_path, mode='offline_fixture')
    assert result['summary']['completed_cases'] == result['summary']['execution_error_count'] == 30
    assert result['summary']['planned_applicable_cases'] == 22
    assert all(item['evaluated_cases'] == 0 and item['mrr_at_k'] is None
               for item in result['summary']['by_k'].values())
    saved = (tmp_path / 'retrieval_report.json').read_text(encoding='utf-8')
    assert 'potentially-sensitive-body' not in saved
    assert 'TimeoutError' in saved and '未完成' in (tmp_path / 'retrieval_report.md').read_text(encoding='utf-8')


def test_provided_report_shows_mrr_denominator_and_diagnostic_columns():
    report = dict(mode='offline_fixture', dataset_fingerprint='fixture', rows=[],
                  summary=summarize_rows([], [1, 3], planned_cases=30, planned_applicable=22))
    text = render_metrics_report(report)
    assert 'MRR@K' in text and '0/22' in text and '不适用 8 题' in text
    assert '遗漏文档组' in text and '禁止来源' in text and '不适用来源' in text


def test_provided_langchain_cache_is_reused_and_stale_snapshot_is_rejected(tmp_path):
    pytest.importorskip('langchain_core')

    class FixedEncoder:
        model, query_instruction = 'day2-fixture-not-a-model', 'fixed-query'

        def __init__(self):
            self.calls = []

        def encode_document(self, texts):
            self.calls.append('document')
            return [[index + 1, 1.0] for index in range(len(texts))]

        def encode_query(self, texts):
            self.calls.append('query')
            return [[1.0, 0.0] for _ in texts]

    encoder = FixedEncoder()
    _, info = prepare_langchain_baseline(DATASET, tmp_path, encoder, embedding_mode='offline_fixture')
    assert info['store_action'] == 'built' and encoder.calls == ['document']
    encoder.calls.clear()
    retrieve, info = prepare_langchain_baseline(DATASET, tmp_path, encoder, embedding_mode='offline_fixture')
    assert info['store_action'] == 'loaded' and encoder.calls == []
    retrieve(load_benchmark(DATASET)['cases'][0]['input'], 3)
    assert encoder.calls == ['query']
    (tmp_path / 'dataset_fingerprint.json').write_text('{"fingerprint":"changed"}', encoding='utf-8')
    with pytest.raises(ValueError, match='快照不匹配'):
        prepare_langchain_baseline(DATASET, tmp_path, encoder, embedding_mode='offline_fixture')


@DAY2
@pytest.mark.parametrize('groups,ranked,k,recall,rr', [
    ([['A']], ['A', 'X'], 1, 1.0, 1.0),
    ([['A']], ['X', 'A'], 1, 0.0, 0.0),
    ([['A']], ['X', 'A'], 3, 1.0, 0.5),
    ([['A'], ['C']], ['X', 'A', 'A', 'C'], 3, 0.5, 0.5),
    ([['A'], ['C']], ['X', 'A', 'A', 'C'], 4, 1.0, 0.5),
    ([['A', 'B'], ['C']], ['B', 'X', 'C'], 3, 1.0, 1.0),
    ([['A']], [], 5, 0.0, 0.0),
    ([['A']], ['X', 'Y'], 5, 0.0, 0.0),
    ([], ['A'], 3, None, None),
])
def test_metrics_match_hand_calculated_examples_without_mutating_inputs(groups, ranked, k, recall, rr):
    original = copy.deepcopy((groups, ranked))
    assert recall_at_k(groups, ranked, k) == recall
    assert reciprocal_rank_at_k(groups, ranked, k) == rr
    assert (groups, ranked) == original


@DAY2
def test_offline_thirty_case_baseline_uses_one_ranking_and_excludes_gold(fixture_retrieval, tmp_path):
    cases, retrieve, calls = fixture_retrieval
    before = copy.deepcopy(cases)
    result = run_retrieval_evaluation(DATASET, retrieve, tmp_path, mode='offline_fixture', backend_info=dict(
        fixture_sha256=hashlib.sha256((DATASET / 'rankings_day2.json').read_bytes()).hexdigest()))
    assert len(calls) == 30 and all(k == max(DEFAULT_KS) for _, k in calls)
    assert all(set(request) == {'question', 'filters', 'access_context'} for request, _ in calls)
    assert cases == before and result['generation_called'] is False
    assert result['summary']['execution_error_count'] == 0
    assert result['summary']['not_applicable_cases'] == 8
    assert all(item['evaluated_cases'] == 22 for item in result['summary']['by_k'].values())
    by_id = {row['case_id']: row for row in result['rows']}
    assert by_id['W9D1-25']['by_k']['3']['recall_at_k'] == 0.5
    assert by_id['W9D1-25']['by_k']['3']['rr_at_k'] == 0.5
    assert by_id['W9D1-25']['by_k']['5']['recall_at_k'] == 1.0
    assert all(item['recall_at_k'] == 0.0 for item in by_id['W9D1-07']['by_k'].values())
    assert by_id['W9D1-26']['by_k']['3']['recall_at_k'] is None
    assert by_id['W9D1-26']['by_k']['3']['forbidden_doc_ids'] == ['KB019']
    assert json.loads((tmp_path / 'retrieval_report.json').read_text(encoding='utf-8')) == result
    target = OUTPUT / 'offline'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('retrieval_report.json', 'retrieval_report.md'):
        shutil.copyfile(tmp_path / name, target / name)
    print(f'Day2 人工排名验收产物：{target}；不代表真实检索质量')


@MODEL
def test_real_ollama_records_thirty_case_retrieval_baseline():
    """实际语义质量先记录基线，不要求召回满分；不请求回答模型，不下载模型。"""
    from rag.embeddings import OllamaEncoder

    output = OUTPUT / 'local_model'
    retrieve, info = prepare_langchain_baseline(DATASET, output / 'index', OllamaEncoder(),
        rebuild=os.environ.get('WEEK9_DAY2_REBUILD') == '1')
    result = run_retrieval_evaluation(DATASET, retrieve, output, mode='local_model', backend_info=info)
    assert result['summary']['completed_cases'] == 30 and result['generation_called'] is False
    assert result['summary']['execution_error_count'] == 0
    assert all(item['evaluated_cases'] == 22 for item in result['summary']['by_k'].values())
    for row in result['rows']:
        recalls = [row['by_k'][str(k)]['recall_at_k'] for k in DEFAULT_KS]
        if recalls[0] is None:
            assert all(value is None for value in recalls)
        else:
            assert 0 <= recalls[0] <= recalls[1] <= recalls[2] <= recalls[3] <= 1
    print(f'Day2 真实检索基线：{output}；pytest 通过不表示召回全通过，请阅读报告定位遗漏与不适用来源')
