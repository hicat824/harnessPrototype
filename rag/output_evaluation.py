"""Week9 Day6：评估过滤后的 RAG，分别记录检索、引用、语义和拒答。"""

import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from rag.answering import ContextBundle, RagAnswer, build_messages, validate_answer
from rag.eval_dataset import build_case_input, load_benchmark, reference_is_eligible, validate_benchmark
from rag.preprocessing import read_manifest
from rag.reranking import ScorePairs, build_budgeted_context, deduplicate_hits, rerank_hits
from rag.retrieval_metrics import _validate_hits, recall_at_k


REVIEW_FIELDS = ('claims_supported', 'answer_complete', 'refusal_reason_ok')
CHECK_TAGS = {
    'retrieval_documents_ok': 'retrieval_miss',
    'context_documents_ok': 'context_miss',
    'policy_ok': 'ineligible_source',
    'answer_format_ok': 'answer_format_error',
    'citations_ok': 'citation_error',
    'answer_status_ok': 'answer_status_error',
    'claims_supported': 'unsupported_claim',
    'answer_complete': 'incomplete_answer',
    'refusal_reason_ok': 'incorrect_refusal_reason',
}


async def collect_trace(request: dict, retrieve, generate, documents: dict, *, candidate_k: int = 20,
                        max_tokens: int = 3000, reserved_tokens: int = 256,
                        score_pairs: ScorePairs | None = None) -> dict:
    """已提供：用 case.input 运行检索和回答，记录阶段与实参。

    retrieve(request, k) 返回标准命中，generate(messages) 异步返回 JSON 正文。
    documents 是可信 manifest 按 doc_id 建立的映射。送生成前复核候选资格，
    违规只记文档 ID 并结束。空 Context 直接生成结构化 insufficient。
    score_pairs 可选，合规候选去重后、组装 Context 前评分并重新排序。
    结果中的 None 表示阶段未执行或检查不适用，异常只记录阶段和类型。
    """
    trace = dict(retrieved_hits=None, policy_ok=None, rejected_doc_ids=[], context=None,
                 messages=[], raw_answer=None, answer=None, answer_format_ok=None,
                 citations_ok=None, generation_called=False, execution_error=None)
    started, stage = time.perf_counter(), 'retrieval'
    try:
        hits = retrieve(copy.deepcopy(request), candidate_k)
        if not isinstance(hits, list) or len(hits) > candidate_k:
            raise ValueError('检索结果必须为列表且不超过 candidate_k')
        stage = 'policy'
        _validate_hits(hits, documents)
        violations = {hit['chunk']['doc_id'] for hit in hits
                      if not reference_is_eligible(documents[hit['chunk']['doc_id']], request)}
        trace['policy_ok'] = not violations
        trace['rejected_doc_ids'] = sorted(violations)
        if violations:
            raise ValueError('检索返回了当前请求不可使用的文档')
        trace['retrieved_hits'] = copy.deepcopy(hits)

        stage = 'context'
        unique, _ = deduplicate_hits(hits)
        if score_pairs is not None and unique:
            stage = 'rerank'
            unique = await rerank_hits(request['question'], unique, score_pairs)
            trace['reranked_hits'] = copy.deepcopy(unique)
        stage = 'context'
        context = build_budgeted_context(unique, max_tokens=max_tokens, reserved_tokens=reserved_tokens).context
        trace['context'] = asdict(context)
        if context.sources:
            trace['messages'] = build_messages(request['question'], context)
            stage = 'generation'
            trace['generation_called'] = True
            raw = await generate(copy.deepcopy(trace['messages']))
            stage = 'answer_format'
            if not isinstance(raw, str):
                raise TypeError('生成器需要返回 JSON 字符串')
            trace['raw_answer'] = raw
            answer = RagAnswer.model_validate_json(raw)
        else:
            answer = RagAnswer(status='insufficient', claims=[],
                               missing_information='未找到可用知识证据，无法依据知识库回答。')
        trace['answer'] = answer.model_dump()
        trace['answer_format_ok'] = True
        stage = 'citation'
        validate_answer(answer, context)
        if answer.status == 'answered':
            trace['citations_ok'] = True
    except Exception as error:
        check = {'policy': 'policy_ok', 'answer_format': 'answer_format_ok', 'citation': 'citations_ok'}.get(stage)
        if check is not None:
            trace[check] = False
        trace['execution_error'] = dict(stage=stage, error_type=type(error).__name__)
    trace['elapsed_seconds'] = time.perf_counter() - started
    return trace


def _review_checks(review: dict | None, applies: tuple[str, ...]) -> dict:
    """校验人工标注。空值待复核，False 必须说明原因。"""
    if review is None:
        return {**dict.fromkeys(REVIEW_FIELDS), 'notes': ''}
    if (not isinstance(review, dict) or set(review) != {*REVIEW_FIELDS, 'notes'}
            or not isinstance(review['notes'], str)):
        raise ValueError('复核需要三项检查值与字符串 notes')
    for name in REVIEW_FIELDS:
        value = review[name]
        if value is not None and (type(value) is not bool or name not in applies):
            raise ValueError(f'不适用的复核项必须留空，适用项为 true、false 或 null：{name}')
    if any(review[name] is False for name in applies) and not review['notes'].strip():
        raise ValueError('人工检查失败时需要填写 notes')
    return dict(review)


def grade_trace(trace: dict, expected: dict, review: dict | None = None) -> dict:
    """用评分参考后验计算检查值与归因，保留上游证据和下游结果。

    输入 trace 为 collect_trace 的结果，expected 为原题库的评分参考。
    review 包含三项可空布尔和 notes。真正的语义判断只来自人工复核。
    输出分阶段指标、检查值、失败标签和最终状态。自动检查未通过或任何适用
    人工项失败返回 failed，自动检查通过但仍有适用人工项未填返回 needs_review，
    其余完整通过的记录返回 passed。None 不单独产生失败标签。
    """
    groups = expected['doc_groups']
    hits, context, answer = trace['retrieved_hits'], trace['context'], trace['answer']
    retrieval_recall = None if hits is None else recall_at_k(
        groups, [hit['chunk']['doc_id'] for hit in hits], max(1, len(hits)))
    sources = [] if context is None else context['sources']
    context_recall = None if context is None else recall_at_k(
        groups, [source['chunk']['doc_id'] for source in sources], max(1, len(sources)))
    retrieval_ok = None if retrieval_recall is None else retrieval_recall == 1.0
    # W9D6-1：检索找齐证据后，才判断 Context 是否漏组。
    # None 表示没有可评估结果，0.0 表示组装成功但未覆盖任何目标组。
    context_ok = None
    if retrieval_ok is True and context_recall is not None:
        context_ok = context_recall == 1.0

    status = answer['status'] if answer is not None else None
    status_ok = None if status is None else status == expected['status']
    automatic = dict(retrieval_documents_ok=retrieval_ok, context_documents_ok=context_ok,
                     policy_ok=trace['policy_ok'], answer_format_ok=trace['answer_format_ok'],
                     citations_ok=trace['citations_ok'], answer_status_ok=status_ok)
    # 此时 errors 只保存自动检查与执行阶段的失败。None 表示未检查或不适用。
    errors = [CHECK_TAGS[name] for name, value in automatic.items() if value is False]
    if trace['execution_error']:
        stage = trace['execution_error']['stage']
        stage_tags = {'retrieval': 'retrieval_execution_error', 'policy': 'ineligible_source',
                      'rerank': 'rerank_error',
                      'context': 'context_build_error', 'generation': 'generation_error',
                      'answer_format': 'answer_format_error', 'citation': 'citation_error'}
        errors.append(stage_tags[stage])
    elif hits is None or context is None or answer is None:
        errors.append('incomplete_trace')
    automatic_passed = not errors  # 在加入人工结论前记录自动检查是否通过。

    # applies 是本题需要人工填写的字段名。回答题检查支持性和完整性，拒答题检查理由。
    # 执行失败或回答状态与预期不符时，applies 保持为空。
    applies = ()
    if status_ok and trace['execution_error'] is None:
        applies = REVIEW_FIELDS[:2] if status == 'answered' else REVIEW_FIELDS[2:]
    # 复核值为 True、False 或 None。适用项为 None 表示待复核，False 才追加错误。
    review = _review_checks(review, applies)
    errors.extend(CHECK_TAGS[name] for name in applies if review[name] is False)
    # W9D6-2：已知失败优先，其次是待复核。只有适用的检查都通过，才标为通过。
    if errors:
        final_status = 'failed'
    elif any(review[name] is None for name in applies):
        final_status = 'needs_review'
    else:
        final_status = 'passed'
    return dict(metrics=dict(retrieval_recall=retrieval_recall, context_recall=context_recall),
                checks=automatic, review=review, review_applicable=list(applies),
                automated_passed=automatic_passed, final_status=final_status,
                error_tags=list(dict.fromkeys(errors)))


def _summarize(rows: list[dict]) -> dict:
    """各维度分别统计通过、失败与未检查。"""
    def counts(values):
        counts_by_value = Counter(values)
        return dict(passed=counts_by_value[True], failed=counts_by_value[False], unchecked=counts_by_value[None])

    return dict(case_count=len(rows), execution_errors=sum(bool(row['trace']['execution_error']) for row in rows),
                final_statuses=dict(Counter(row['assessment']['final_status'] for row in rows)),
                automatic={name: counts(row['assessment']['checks'][name] for row in rows)
                           for name in CHECK_TAGS if name not in REVIEW_FIELDS},
                manual={name: {**counts(row['assessment']['review'][name] for row in rows
                                       if name in row['assessment']['review_applicable']),
                               'not_applicable': sum(name not in row['assessment']['review_applicable'] for row in rows)}
                        for name in REVIEW_FIELDS},
                errors=dict(Counter(tag for row in rows for tag in row['assessment']['error_tags'])))


def _write_report(path: Path, report: dict) -> bytes:
    report['summary'] = _summarize(report['rows'])
    raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(raw)
    lines = [f"# {report.get('title', 'Week9 Day6 RAG 输出评估')}", '', f"运行模式为 `{report['mode']}`。", '',
             '| 检查项 | 通过 | 失败 | 未检查 | 不适用 |', '|---|---:|---:|---:|---:|']
    summary = report['summary']
    for name, check in {**summary['automatic'], **summary['manual']}.items():
        lines.append(f"| {name} | {check['passed']} | {check['failed']} | "
                     f"{check['unchecked']} | {check.get('not_applicable', '-')} |")
    lines.extend(['', '自动检查中的“未检查”包含不适用和阶段未执行。', '',
                  '| 用例 | 检索召回 | Context 召回 | 状态 | 错误标签 |', '|---|---:|---:|---|---|'])
    for row in report['rows']:
        grade = row['assessment']
        lines.append(f"| {row['case_id']} | {grade['metrics']['retrieval_recall']} | "
                     f"{grade['metrics']['context_recall']} | {grade['final_status']} | "
                     f"{', '.join(grade['error_tags']) or '-'} |")
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return raw


def write_evaluation_report(output: Path, report: dict) -> None:
    """保存原始执行报告及与本轮原始字节绑定的人工复核模板。"""
    output.mkdir(parents=True, exist_ok=True)
    raw = _write_report(output / 'evaluation_report.json', report)
    templates = [dict(case_id=row['case_id'], **row['assessment']['review']) for row in report['rows']]
    template = dict(report_sha256=hashlib.sha256(raw).hexdigest(), reviews=templates)
    (output / 'reviews_template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')


async def run_output_evaluation(dataset_dir, retrieve, generate, output_dir, *,
                                mode: str, case_ids: list[str] | None = None) -> dict:
    """已提供：从冻结题库选题并记录。评分参考用于运行结束后的评分。

    输出 evaluation_report.json/md 和绑定该次报告的 reviews_template.json。
    mode 区分 offline_fixture 与 real_llm。两种模式使用相同验收契约。
    """
    if mode not in ('offline_fixture', 'real_llm'):
        raise ValueError('mode 需要是 offline_fixture 或 real_llm')
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output == root or output == root / 'knowledge' or root / 'knowledge' in output.parents:
        raise ValueError('不能将报告写入数据集输入目录')
    snapshot = validate_benchmark(root)
    cases = {case['case_id']: case for case in load_benchmark(root)['cases']}
    if case_ids is not None and (not isinstance(case_ids, list) or not case_ids
                                 or any(not isinstance(case_id, str) or case_id not in cases for case_id in case_ids)
                                 or len(set(case_ids)) != len(case_ids)):
        raise ValueError('case_ids 需要是已知且不重复的用例')
    documents = {doc['doc_id']: doc for doc in read_manifest(root)}
    rows = []
    for case_id in cases if case_ids is None else case_ids:
        case = cases[case_id]
        trace = await collect_trace(build_case_input(case), retrieve, generate, documents)
        expected = copy.deepcopy(case['expected'])
        grade = grade_trace(trace, expected)
        rows.append(dict(case_id=case_id, category=case['category'], input=copy.deepcopy(case['input']),
                         expected=expected, trace=trace, assessment=grade))
    report = dict(mode=mode, dataset_fingerprint=snapshot['fingerprint'], rows=rows)
    write_evaluation_report(output, report)
    return report


def apply_reviews(report_path, reviews_path) -> dict:
    """已提供：对指定执行报告离线合并复核，另存 reviewed_report.json/md。

    复核文件的 report_sha256 必须和执行报告原始字节一致。允许只填写部分题，
    未覆盖的题仍待复核。发生格式错误时保留执行报告，且不会写入部分结果。
    """
    path = Path(report_path)
    if path.name != 'evaluation_report.json':
        raise ValueError('请传入原始 evaluation_report.json')
    raw = path.read_bytes()
    report = json.loads(raw)
    reviews = json.loads(Path(reviews_path).read_text(encoding='utf-8'))
    if (not isinstance(reviews, dict) or reviews.get('report_sha256') != hashlib.sha256(raw).hexdigest()
            or not isinstance(reviews.get('reviews'), list)):
        raise ValueError('复核文件与本次原始报告不匹配')
    known, by_case = {row['case_id'] for row in report['rows']}, {}
    for item in reviews['reviews']:
        if not isinstance(item, dict) or not isinstance(item.get('case_id'), str):
            raise ValueError('复核行需要 case_id')
        case_id = item['case_id']
        if case_id in by_case or case_id not in known:
            raise ValueError('复核 case_id 未知或重复')
        by_case[case_id] = {name: value for name, value in item.items() if name != 'case_id'}
    for row in report['rows']:
        row['assessment'] = grade_trace(row['trace'], row['expected'], by_case.get(row['case_id']))
    report['reviewed_source_sha256'] = reviews['report_sha256']
    _write_report(path.with_name('reviewed_report.json'), report)
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Week9 Day6 离线合并人工复核')
    parser.add_argument('report', help='本次执行的 evaluation_report.json')
    parser.add_argument('reviews', help='填写后的 reviews JSON')
    args = parser.parse_args()
    result = apply_reviews(args.report, args.reviews)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
