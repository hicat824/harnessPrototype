"""Week9 Day6 的阶段归因与人工复核验收，测试代码已提供。"""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from rag.eval_dataset import DEFAULT_DATASET, build_case_input, load_benchmark
from rag.knowledge_policy import make_policy_retriever
from rag.hybrid_search import prepare_search_store
from rag.output_evaluation import apply_reviews, collect_trace, grade_trace, run_output_evaluation
from rag.preprocessing import load_documents, read_manifest, split_document
from rag.reranking import build_budgeted_context


DATASET = Path(os.environ.get('WEEK9_RAG_DATASET', str(DEFAULT_DATASET)))
OUTPUT = Path(os.environ.get('WEEK9_DAY6_OUTPUT', str(DATASET / 'runs/day6')))
DAY6 = pytest.mark.skipif(os.environ.get('RUN_WEEK9_DAY6') != '1',
                          reason='完整离线验收需 RUN_WEEK9_DAY6=1')
REAL = pytest.mark.skipif(any(os.environ.get(key) != '1'
                             for key in ('RUN_WEEK9_DAY6', 'RUN_WEEK9_DAY6_REAL')),
                         reason='真实 Ollama + DeepSeek 需同时启用 DAY6 与 REAL')
REAL_CASES = ['W9D1-09', 'W9D1-17', 'W9D1-18', 'W9D1-25', 'W9D1-27', 'W9D1-28']


@pytest.fixture(autouse=True)
def no_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


@pytest.fixture(scope='module')
def benchmark():
    return {case['case_id']: case for case in load_benchmark(DATASET)['cases']}


@pytest.fixture(scope='module')
def manifest():
    return {doc['doc_id']: doc for doc in read_manifest(DATASET)}


@pytest.fixture(scope='module')
def hits():
    result = {}
    for document in load_documents(DATASET):
        for chunk in split_document(document):
            result.setdefault(chunk.doc_id, []).append(dict(score=0.5, chunk=asdict(chunk)))
    return result


def _hit_with_quote(hits, reference):
    return next(hit for hit in hits[reference['doc_id']]
                if hit['chunk']['segment_id'] == reference['segment_id']
                and reference['quote'] in hit['chunk']['text'])


def _answered(quote, *, claim='查询没有找到任务，也不能据此判断任务执行失败或权限拒绝。', source_id='S1'):
    return json.dumps(dict(status='answered', missing_information='', claims=[
        dict(text=claim, citations=[dict(source_id=source_id, quote=quote)])
    ]), ensure_ascii=False)


async def _insufficient(messages):
    return json.dumps(dict(status='insufficient', claims=[],
                           missing_information='还缺少确认当前事实的可用证据。'), ensure_ascii=False)


def _review(**decisions):
    return dict(claims_supported=None, answer_complete=None, refusal_reason_ok=None, notes='') | decisions


@pytest.mark.parametrize(('problem', 'stage'), [
    ('valid', None), ('invalid_json', 'answer_format'),
    ('unknown_source', 'citation'), ('invented_quote', 'citation'), ('timeout', 'generation'),
])
@pytest.mark.asyncio
async def test_provided_trace_separates_format_reference_and_execution(benchmark, manifest, hits, problem, stage):
    case = benchmark['W9D1-09']
    reference = case['expected']['evidence'][0]
    allowed = _hit_with_quote(hits, reference)
    request = build_case_input(case)
    seen = []

    def retrieve(actual_input, k):
        assert actual_input == request and set(actual_input) == {'question', 'filters', 'access_context'}
        assert actual_input is not request
        return [allowed]

    async def generate(messages):
        text = messages[1]['content']
        assert request['question'] in text and reference['quote'] in text
        assert all(key not in text for key in ('acceptable_points', 'doc_groups', 'granted_scopes', 'case_id'))
        seen.extend(copy.deepcopy(messages))
        if problem == 'timeout':
            raise TimeoutError('不应写入报告的异常详情')
        if problem == 'invalid_json':
            return 'not-json'
        quote = '原文中不存在的摘录' if problem == 'invented_quote' else reference['quote']
        return _answered(quote, source_id='S2' if problem == 'unknown_source' else 'S1')

    trace = await collect_trace(request, retrieve, generate, manifest)
    assert trace['context']['sources'][0]['chunk'] == allowed['chunk']
    assert trace['messages'] == seen and trace['generation_called'] is True
    assert (trace['execution_error'] or {}).get('stage') == stage
    assert trace['answer_format_ok'] is (False if stage == 'answer_format' else None if stage == 'generation' else True)
    assert trace['citations_ok'] is (True if stage is None else False if stage == 'citation' else None)
    if problem == 'invalid_json':
        assert trace['raw_answer'] == 'not-json'
    if problem == 'timeout':
        assert trace['execution_error'] == dict(stage='generation', error_type='TimeoutError')


@pytest.mark.parametrize(('case_id', 'doc_id'), [
    ('W9D1-17', None), ('W9D1-28', 'KB019'), ('W9D1-25', 'KB020'), ('W9D1-25', 'KB021'),
])
@pytest.mark.asyncio
async def test_provided_empty_or_disallowed_candidates_never_reach_generator(benchmark, manifest, hits, case_id, doc_id):
    async def forbidden(messages):
        pytest.fail('不应调用生成器')

    blocked = [] if doc_id is None else hits[doc_id][:1]
    trace = await collect_trace(build_case_input(benchmark[case_id]), lambda request, k: blocked, forbidden, manifest)
    assert trace['generation_called'] is False and trace['messages'] == []
    if doc_id is None:
        assert trace['answer']['status'] == 'insufficient' and trace['context']['sources'] == []
        assert trace['execution_error'] is None
    else:
        assert trace['rejected_doc_ids'] == [doc_id] and trace['execution_error']['stage'] == 'policy'
        assert trace['retrieved_hits'] is trace['context'] is trace['answer'] is None


@DAY6
@pytest.mark.parametrize('cause', ['retrieval', 'context'])
@pytest.mark.asyncio
async def test_one_missing_version_is_attributed_to_the_right_stage(benchmark, manifest, hits, cause):
    case = benchmark['W9D1-25']
    first, second = [_hit_with_quote(hits, ref) for ref in case['expected']['evidence']]
    capacity = build_budgeted_context([first], reserved_tokens=0).used_tokens
    candidates = [first] if cause == 'retrieval' else [first, second]
    trace = await collect_trace(build_case_input(case), lambda request, k: candidates,
                                _insufficient, manifest, max_tokens=capacity, reserved_tokens=0)
    grade = grade_trace(trace, case['expected'])
    assert grade['metrics']['retrieval_recall'] == (0.5 if cause == 'retrieval' else 1.0)
    assert grade['metrics']['context_recall'] == 0.5
    assert grade['checks']['context_documents_ok'] is (None if cause == 'retrieval' else False)
    assert grade['error_tags'] == [f'{cause}_miss', 'answer_status_error']


@DAY6
@pytest.mark.parametrize(('budget', 'context_recall', 'context_ok', 'error_tags'), [
    (0, None, None, ['context_build_error']),
    (1, 0.0, False, ['context_miss', 'answer_status_error']),
], ids=['build-error', 'built-empty'])
@pytest.mark.asyncio
async def test_context_build_error_and_empty_context_have_different_tags(
        benchmark, manifest, hits, budget, context_recall, context_ok, error_tags):
    case = benchmark['W9D1-09']
    hit = _hit_with_quote(hits, case['expected']['evidence'][0])
    trace = await collect_trace(build_case_input(case), lambda request, k: [hit],
                                _insufficient, manifest, max_tokens=budget, reserved_tokens=0)
    grade = grade_trace(trace, case['expected'])
    assert grade['metrics'] == dict(retrieval_recall=1.0, context_recall=context_recall)
    assert grade['checks']['context_documents_ok'] is context_ok
    assert grade['error_tags'] == error_tags
    assert trace['generation_called'] is False


@DAY6
@pytest.mark.asyncio
async def test_valid_quote_can_still_support_an_incomplete_or_incorrect_answer(benchmark, manifest, hits):
    case = benchmark['W9D1-09']
    ref = case['expected']['evidence'][0]
    async def generate(messages):
        return _answered(ref['quote'], claim='查询没有找到任务。')

    trace = await collect_trace(build_case_input(case), lambda req, k: [_hit_with_quote(hits, ref)], generate, manifest)
    pending = grade_trace(trace, case['expected'])
    assert pending['checks']['citations_ok'] is True and pending['automated_passed'] is True
    assert pending['final_status'] == 'needs_review'
    partial = grade_trace(trace, case['expected'], _review(claims_supported=True))
    assert partial['final_status'] == 'needs_review'
    incomplete = grade_trace(trace, case['expected'],
        _review(claims_supported=True, answer_complete=False, notes='漏答是否代表失败或权限拒绝。'))
    assert incomplete['error_tags'] == ['incomplete_answer'] and incomplete['final_status'] == 'failed'

    async def contradict(messages):
        return _answered(ref['quote'], claim='查询没有找到任务，也不能据此判断任务执行失败或权限拒绝。记录保留 30 天。')

    trace = await collect_trace(build_case_input(case), lambda req, k: [_hit_with_quote(hits, ref)], contradict, manifest)
    unsupported = grade_trace(trace, case['expected'],
        _review(claims_supported=False, answer_complete=True, notes='引用原文不支持 30 天保留期限。'))
    assert unsupported['checks']['citations_ok'] is True
    assert unsupported['error_tags'] == ['unsupported_claim'] and unsupported['final_status'] == 'failed'


@DAY6
@pytest.mark.asyncio
async def test_refusal_checks_distinguish_status_and_reason(benchmark, manifest, hits):
    case = benchmark['W9D1-18']
    trace = await collect_trace(build_case_input(case), lambda req, k: hits['KB018'][:1], _insufficient, manifest)
    pending = grade_trace(trace, case['expected'])
    assert pending['metrics'] == dict(retrieval_recall=None, context_recall=None)
    assert pending['checks']['answer_status_ok'] is True and pending['checks']['citations_ok'] is None
    assert pending['review_applicable'] == ['refusal_reason_ok'] and pending['final_status'] == 'needs_review'
    bad = grade_trace(trace, case['expected'],
        _review(refusal_reason_ok=False, notes='说资料不足，但未指出需要实时 API。'))
    assert bad['error_tags'] == ['incorrect_refusal_reason']
    assert bad['final_status'] == 'failed'
    assert grade_trace(trace, case['expected'], _review(refusal_reason_ok=True))['final_status'] == 'passed'
    with pytest.raises(ValueError, match='不适用'):
        grade_trace(trace, case['expected'], _review(answer_complete=True))
    with pytest.raises(ValueError, match='notes'):
        grade_trace(trace, case['expected'], _review(refusal_reason_ok=False))


@DAY6
@pytest.mark.asyncio
async def test_model_failure_never_rewrites_as_refusal(benchmark, manifest, hits):
    case = benchmark['W9D1-18']

    async def fail(messages):
        raise RuntimeError('服务不可用')

    trace = await collect_trace(build_case_input(case), lambda req, k: hits['KB018'][:1], fail, manifest)
    grade = grade_trace(trace, case['expected'])
    assert trace['answer'] is None and grade['checks']['answer_status_ok'] is None
    assert grade['final_status'] == 'failed' and grade['error_tags'] == ['generation_error']


@DAY6
@pytest.mark.asyncio
async def test_partial_manual_review_is_bound_to_the_original_execution(tmp_path, benchmark, hits):
    case = benchmark['W9D1-09']
    ref = case['expected']['evidence'][0]
    allowed = _hit_with_quote(hits, ref)

    def retrieve(request, k):
        return [allowed] if request['question'] == case['input']['question'] else []

    async def generate(messages):
        return _answered(ref['quote'])

    report = await run_output_evaluation(DATASET, retrieve, generate, tmp_path,
        mode='offline_fixture', case_ids=['W9D1-09', 'W9D1-17'])
    assert report['summary']['final_statuses'] == {'needs_review': 2}
    path = tmp_path / 'evaluation_report.json'
    original = path.read_bytes()
    review_path = tmp_path / 'reviews_template.json'
    reviews = json.loads(review_path.read_text(encoding='utf-8'))
    decisions = dict(case_id='W9D1-09', **_review(claims_supported=True, answer_complete=True))
    reviews['reviews'] = [decisions]
    review_path.write_text(json.dumps(reviews), encoding='utf-8')
    reviewed = apply_reviews(path, review_path)
    assert reviewed['summary']['final_statuses'] == {'passed': 1, 'needs_review': 1}
    assert path.read_bytes() == original and (tmp_path / 'reviewed_report.md').exists()
    path.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='不匹配'):
        apply_reviews(path, review_path)


class FixedEncoder:
    """固定向量只验证完整离线链路。"""

    model, query_instruction = 'day6-fixture-not-a-model', 'fixed-query'

    def encode_document(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, texts):
        return [[1.0, 0.0] for _ in texts]


@DAY6
@pytest.mark.asyncio
async def test_offline_thirty_cases_record_separate_dimensional_counts(tmp_path):
    store, _ = prepare_search_store(DATASET, tmp_path / 'index', FixedEncoder(),
                                    embedding_mode='offline_fixture')
    report = await run_output_evaluation(DATASET, make_policy_retriever(store), _insufficient,
                                          OUTPUT / 'offline', mode='offline_fixture')
    summary = report['summary']
    assert summary['case_count'] == 30 and summary['execution_errors'] == 0
    assert summary['automatic']['policy_ok']['failed'] == 0
    assert summary['automatic']['retrieval_documents_ok']['unchecked'] == 8
    assert summary['automatic']['answer_status_ok']['failed'] == 22
    assert summary['final_statuses'] == {'failed': 22, 'needs_review': 8}
    assert summary['manual']['refusal_reason_ok']['unchecked'] == 8


@REAL
@pytest.mark.asyncio
async def test_real_ollama_and_deepseek_record_six_cases_for_review():
    from rag.answering import DeepSeekGenerator
    from rag.embeddings import OllamaEncoder

    output = OUTPUT / 'real_model'
    day2_index = DATASET / 'runs/day2/local_model/index'
    default = day2_index if (day2_index / 'knowledge_store.json').exists() else output / 'index'
    index = Path(os.environ.get('WEEK9_DAY6_INDEX', str(default)))
    store, _ = prepare_search_store(DATASET, index, OllamaEncoder())
    report = await run_output_evaluation(DATASET, make_policy_retriever(store), DeepSeekGenerator(),
                                          output, mode='real_llm', case_ids=REAL_CASES)
    assert [row['case_id'] for row in report['rows']] == REAL_CASES
    for row in report['rows']:
        trace = row['trace']
        assert trace['policy_ok'] is True and not trace['rejected_doc_ids']
        assert trace['execution_error'] is None or trace['execution_error']['stage'] in ('answer_format', 'citation')
        print(f"{row['case_id']}: {row['assessment']['final_status']} {row['assessment']['error_tags']}")
    print(f'Day6 真实输出报告与人工复核模板：{output}')
