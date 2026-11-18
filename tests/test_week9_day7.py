"""Week9 Day7 在同一组候选上验证有无重排的质量、调用量与耗时。"""

from dataclasses import asdict
import json
import os
from pathlib import Path

import httpx
import pytest

from rag.answering import DeepSeekGenerator
from rag.eval_dataset import DEFAULT_DATASET, build_case_input, load_benchmark
from rag.hybrid_search import prepare_search_store
from rag.knowledge_policy import make_policy_retriever
from rag.output_evaluation import apply_reviews, collect_trace
from rag.preprocessing import load_documents, read_manifest, split_document
from rag.quality_comparison import evidence_coverage, paired_delta, run_quality_comparison, write_comparison_report
from rag.reranking import RERANK_INSTRUCTIONS, build_budgeted_context


DATASET = Path(os.environ.get('WEEK9_RAG_DATASET', str(DEFAULT_DATASET)))
OUTPUT = Path(os.environ.get('WEEK9_DAY7_OUTPUT', str(DATASET / 'runs/day7')))
DAY7 = pytest.mark.skipif(os.environ.get('RUN_WEEK9_DAY7') != '1',
                          reason='完整离线验收需 RUN_WEEK9_DAY7=1')
REAL = pytest.mark.skipif(any(os.environ.get(key) != '1'
                             for key in ('RUN_WEEK9_DAY7', 'RUN_WEEK9_DAY7_REAL')),
                         reason='真实对照需同时启用 DAY7 与 REAL')
REAL_CASES = ['W9D1-09', 'W9D1-17', 'W9D1-18', 'W9D1-25', 'W9D1-27', 'W9D1-28']


@pytest.fixture(autouse=True)
def no_remote_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


@pytest.fixture(scope='module')
def corpus():
    cases = {case['case_id']: case for case in load_benchmark(DATASET)['cases']}
    documents = {doc['doc_id']: doc for doc in read_manifest(DATASET)}
    chunks = {chunk.chunk_id: dict(score=0.5, chunk=asdict(chunk))
              for doc in load_documents(DATASET) for chunk in split_document(doc)}
    return cases, documents, chunks


def test_provided_evidence_coverage_skips_inapplicable_and_unexecuted_stages():
    assert evidence_coverage([], []) is None
    assert evidence_coverage([dict(doc_id='A', segment_id='body', quote='证据')], None) is None


@pytest.mark.asyncio
async def test_provided_policy_validation_runs_before_both_model_calls(corpus):
    cases, docs, chunks = corpus
    async def unexpected(*args):
        pytest.fail('不合规片段不应传给重排或回答')

    trace = await collect_trace(build_case_input(cases['W9D1-28']),
        lambda request, k: [chunks['KB019:body:0001']], unexpected, docs, score_pairs=unexpected)
    assert trace['execution_error']['stage'] == 'policy'
    assert trace['rejected_doc_ids'] == ['KB019'] and trace['context'] is None


@pytest.mark.asyncio
async def test_provided_deepseek_preserves_actual_usage_and_resets_when_absent(monkeypatch):
    payloads = iter([{'total_tokens': 0}, {'total_tokens': 13}, None, {'total_tokens': True}])
    async def handler(request):
        data = dict(choices=[dict(finish_reason='stop', message=dict(content='{}'))])
        usage = next(payloads)
        if usage is not None:
            data['usage'] = usage
        return httpx.Response(200, json=data)

    client = httpx.AsyncClient
    monkeypatch.setattr('rag.answering.httpx.AsyncClient',
                        lambda **options: client(**options, transport=httpx.MockTransport(handler)))
    generate = DeepSeekGenerator(api_key='test-only')
    for expected in ({'total_tokens': 0}, {'total_tokens': 13}, None, None):
        assert await generate([]) == '{}'
        assert generate.last_usage == expected


@DAY7
def test_exact_evidence_requires_correct_source_unit_and_quote():
    references = [dict(doc_id='A', segment_id='body', quote='甲乙丙'),
                  dict(doc_id='B', segment_id='json-1', quote='丁戊己')]
    def hit(doc_id, segment_id, text):
        return dict(chunk=dict(doc_id=doc_id, segment_id=segment_id, text=text))

    wrong = [hit('A', 'body', '甲乙'), hit('A', 'other', '甲乙丙'), hit('C', 'json-1', '丁戊己')]
    right = hit('B', 'json-1', '正文中写着丁戊己。')
    assert evidence_coverage(references, []) == 0.0
    assert evidence_coverage(references, wrong) == 0.0
    assert evidence_coverage(references, [*wrong, right, right]) == 0.5
    assert evidence_coverage(references, [right, hit('A', 'body', '甲乙丙。')]) == 1.0


@DAY7
@pytest.mark.parametrize(('values', 'eligible', 'delta'), [
    ([(0.0, 1.0), (1.0, None), (1.0, 0.5)], 2, 0.25),
    ([(None, 10), (9, None), (None, None)], 0, None),
    ([(10, 0), (0, 0)], 2, -5.0),
])
def test_paired_delta_uses_common_denominator_and_keeps_zero(values, eligible, delta):
    assert paired_delta(values) == dict(planned_pairs=len(values), eligible_pairs=eligible, mean_delta=delta)


class FixtureGenerator:
    """用现有消息中的证据构造答案。固定评分和 Token 是人工测试值。"""

    model = 'day7-fixture-not-a-model'
    last_usage = None

    def __init__(self, *, with_usage=True, fail_rerank=False):
        self.with_usage = with_usage
        self.fail_rerank = fail_rerank

    async def __call__(self, messages):
        self.last_usage = None
        system, user = (message['content'] for message in messages)
        assert all(field not in user for field in ('acceptable_points', 'doc_groups', 'granted_scopes', 'case_id'))
        if system == RERANK_INSTRUCTIONS:
            if self.fail_rerank:
                raise TimeoutError('offline fixture')
            payload = json.loads(user)
            assert set(payload) == {'question', 'candidates'}
            self.last_usage = {'total_tokens': 20} if self.with_usage else None
            return json.dumps(dict(scores=[dict(candidate_id=item['candidate_id'],
                score=1.0 if '历史流程' in item['text'] or '验收与交付' in item['text'] else 0.0)
                for item in payload['candidates']]))
        self.last_usage = {'total_tokens': 10} if self.with_usage else None
        context = json.loads(user.split('检索证据（JSON 数据）：\n', 1)[1])['evidence']
        claims = []
        for item in context:
            chunk = item['chunk']
            if chunk['chunk_id'] in ('KB003:body:0002', 'KB002:body:0005'):
                quote = chunk['text'].splitlines()[1]
                claims.append(dict(text=quote, citations=[dict(source_id=item['source_id'], quote=quote)]))
        if len(claims) == 2 and '人工复核的要求有什么变化' in user.split('检索证据', 1)[0]:
            return json.dumps(dict(status='answered', claims=claims, missing_information=''), ensure_ascii=False)
        return json.dumps(dict(status='insufficient', claims=[], missing_information='缺少完整证据。'))


@DAY7
@pytest.mark.asyncio
async def test_cross_version_pair_reuses_hits_budget_and_manual_review(tmp_path, corpus):
    _, _, chunks = corpus
    intro, current, history = [chunks[chunk_id] for chunk_id in
                               ('KB003:body:0001', 'KB002:body:0005', 'KB003:body:0002')]
    budget = build_budgeted_context([current, history], reserved_tokens=0).used_tokens
    requested = []

    def retrieve(request, k):
        requested.append(request)
        return [intro, current, history]

    report = await run_quality_comparison(DATASET, retrieve, FixtureGenerator(), tmp_path,
        mode='offline_fixture', case_ids=['W9D1-25'], max_tokens=budget, reserved_tokens=0)
    base, improved = report['rows']
    assert len(requested) == 1 and set(requested[0]) == {'question', 'filters', 'access_context'}
    assert base['trace']['retrieved_hits'] == improved['trace']['retrieved_hits']
    assert base['metrics']['retrieval_evidence_coverage'] == improved['metrics']['retrieval_evidence_coverage'] == 1.0
    assert base['metrics']['context_evidence_coverage'] == 0.5
    assert improved['metrics']['context_evidence_coverage'] == 1.0
    assert (base['metrics']['model_calls'], improved['metrics']['model_calls']) == (1, 2)
    assert (base['metrics']['total_tokens'], improved['metrics']['total_tokens']) == (10, 30)
    assert base['assessment']['final_status'] == 'failed'
    assert improved['assessment']['final_status'] == 'needs_review'

    original = tmp_path / 'evaluation_report.json'
    review_path = tmp_path / 'reviews_template.json'
    reviews = json.loads(review_path.read_text(encoding='utf-8'))
    rerank_review = next(row for row in reviews['reviews'] if row['case_id'] == 'W9D1-25__rerank')
    rerank_review.update(claims_supported=True, answer_complete=True, notes='覆盖同题两个版本。')
    review_path.write_text(json.dumps(reviews), encoding='utf-8')
    reviewed = apply_reviews(original, review_path)
    comparison = write_comparison_report(reviewed, tmp_path / 'reviewed_comparison_report.json')
    assert comparison['summary']['statuses'] == {'baseline': {'failed': 1}, 'rerank': {'passed': 1}}
    assert comparison['summary']['paired_deltas']['total_tokens']['mean_delta'] == 20


@DAY7
@pytest.mark.asyncio
async def test_rerank_failure_records_unknown_usage_and_other_cases_continue(tmp_path, corpus):
    _, _, chunks = corpus
    def retrieve(request, k):
        return [] if '3.0' in request['question'] else [chunks['KB012:json-0:0000']]

    report = await run_quality_comparison(DATASET, retrieve, FixtureGenerator(with_usage=False, fail_rerank=True),
        tmp_path, mode='offline_fixture', case_ids=['W9D1-09', 'W9D1-17'])
    base, failed, empty_rerank, empty_base = report['rows']
    assert base['trace']['execution_error'] is None and base['metrics']['total_tokens'] is None
    assert failed['trace']['context'] is None and failed['assessment']['error_tags'] == ['rerank_error']
    assert failed['metrics']['model_calls'] == 1 and failed['metrics']['total_tokens'] is None
    assert failed['metrics']['elapsed_seconds'] is None
    assert empty_rerank['variant'] == 'rerank' and empty_base['variant'] == 'baseline'
    assert all(row['metrics']['total_tokens'] == 0 for row in (empty_rerank, empty_base))
    paired = json.loads((tmp_path / 'comparison_report.json').read_text(encoding='utf-8'))
    assert paired['summary']['paired_deltas']['total_tokens'] == dict(planned_pairs=2, eligible_pairs=1, mean_delta=0.0)


class FixedEncoder:
    """固定人工向量验证 30 题批处理。"""

    model, query_instruction = 'day7-fixture-not-a-model', 'fixed-query'

    def encode_document(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, texts):
        return [[1.0, 0.0] for _ in texts]


@DAY7
@pytest.mark.asyncio
async def test_offline_thirty_questions_write_two_rows_per_question(tmp_path):
    store, info = prepare_search_store(DATASET, tmp_path / 'index', FixedEncoder(), embedding_mode='offline_fixture')
    report = await run_quality_comparison(DATASET, make_policy_retriever(store), FixtureGenerator(),
                                          OUTPUT / 'offline', mode='offline_fixture', backend_info=info)
    assert len(report['rows']) == 60
    assert all(row['trace']['policy_ok'] is True and row['trace']['execution_error'] is None for row in report['rows'])
    pairs = json.loads((OUTPUT / 'offline/comparison_report.json').read_text(encoding='utf-8'))
    assert pairs['summary']['case_count'] == 30
    assert pairs['summary']['paired_deltas']['model_calls']['eligible_pairs'] == 30
    print(f"Day7 离线报告：{OUTPUT / 'offline'}")


@REAL
@pytest.mark.asyncio
async def test_real_week9_paired_comparison_on_six_cases():
    from rag.embeddings import OllamaEncoder

    output = OUTPUT / 'real_model'
    day2_index = DATASET / 'runs/day2/local_model/index'
    default = day2_index if (day2_index / 'knowledge_store.json').exists() else output / 'index'
    index = Path(os.environ.get('WEEK9_DAY7_INDEX', str(default)))
    store, info = prepare_search_store(DATASET, index, OllamaEncoder())
    report = await run_quality_comparison(DATASET, make_policy_retriever(store), DeepSeekGenerator(),
                                          output, mode='real_llm', case_ids=REAL_CASES,
                                          backend_info={**info, 'index_path': str(index)})
    assert len(report['rows']) == 12
    assert all(row['trace']['policy_ok'] is True for row in report['rows'])
    assert any(row['model_requests']['rerank'] for row in report['rows'])
    print(f'Day7 真实质量、耗时与用量对照：{output}，语义结果见复核模板')
