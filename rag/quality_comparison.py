"""Week9 Day7：在同一批合规候选上对照关闭和开启重排的完整回答。"""

import copy
from collections import Counter
import json
from pathlib import Path
from statistics import fmean
import time

from rag.eval_dataset import build_case_input, load_benchmark, validate_benchmark
from rag.output_evaluation import collect_trace, grade_trace, write_evaluation_report
from rag.preprocessing import read_manifest
from rag.reranking import JsonLLMReranker


PATHS = ('baseline', 'rerank')
METRICS = ('context_evidence_coverage', 'elapsed_seconds', 'model_calls', 'total_tokens')


def evidence_coverage(references: list[dict], hits: list[dict] | None) -> float | None:
    """计算参考证据在检索结果或 Context 中的命中比例。

    references 每项为 {doc_id, segment_id, quote}。hits 每项的 chunk
    含 doc_id、segment_id 与 text。hits=None 表示阶段无结果，返回 None；
    references=[] 表示本题没有参考摘录，也返回 None。hits=[] 才是实际零命中。

    例：两条参考中，A 仅选入标题，B 选入正确单元和原文，得 1/2。
    即使在另一文档或单元找到同一句，仍需核对参考来源。
    """
    if not references or hits is None:
        return None
    chunks = [hit['chunk'] for hit in hits]
    matched = 0
    for reference in references:
        # 评分参考仅在执行后使用。来源、正文单元和完整摘录同时匹配，每条参考最多记一次。
        found = False
        for chunk in chunks:
            if (reference['doc_id'] == chunk['doc_id']
                    and reference['segment_id'] == chunk['segment_id']
                    and reference['quote'] in chunk['text']):
                found = True
                break
        matched += int(found)
    return matched / len(references)


def paired_delta(values: list[tuple[float | int | None, float | int | None]]) -> dict:
    """只在同题两路都存在观测值时计入平均差。

    values 的每对来自同一题，顺序固定为 (baseline, rerank)。
    None 表示缺测或不适用，0 是实测值。以 [(0, 1), (1, None), (1, 0.5)]
    为例，只计第一和第三对，差值分别为 +1 和 -0.5，均值 +0.25。
    返回 planned_pairs、eligible_pairs、mean_delta。无有效配对时均值为 None。
    """
    # 同题双侧都有值才计入分母。零是有效观测，只有 None 表示缺测。
    differences = [
        rerank - baseline
        for baseline, rerank in values
        if baseline is not None and rerank is not None
    ]
    return dict(planned_pairs=len(values), eligible_pairs=len(differences),
                mean_delta=fmean(differences) if differences else None)


def _track_generation(generate, requests: list[dict]):
    async def tracked(messages):
        started = time.perf_counter()
        try:
            return await generate(messages)
        finally:
            usage = getattr(generate, 'last_usage', None)
            total = usage.get('total_tokens') if isinstance(usage, dict) else None
            requests.append(dict(elapsed_seconds=time.perf_counter() - started,
                                 total_tokens=total if type(total) is int and total >= 0 else None))
    return tracked


def _score_row(trace: dict, expected: dict, retrieval_seconds: float, calls: dict) -> dict:
    context = trace['context']
    refs = expected['evidence']
    requests = calls['answer'] + calls['rerank']
    counts = [call['total_tokens'] for call in requests]
    # 失败分支的原始耗时仍在 trace 中。比较完整链路时，把提前中止视为缺测。
    elapsed = None if trace['execution_error'] else retrieval_seconds + trace['elapsed_seconds']
    return dict(
        retrieval_evidence_coverage=evidence_coverage(refs, trace['retrieved_hits']),
        context_evidence_coverage=evidence_coverage(refs, None if context is None else context['sources']),
        elapsed_seconds=elapsed, model_calls=len(requests),
        total_tokens=sum(counts) if all(count is not None for count in counts) else None,
    )


async def run_quality_comparison(dataset_dir, retrieve, generate, output_dir, *, mode: str,
                                 case_ids: list[str] | None = None, candidate_k: int = 20,
                                 max_tokens: int = 3000, reserved_tokens: int = 256,
                                 backend_info: dict | None = None) -> dict:
    """逐题只检索一次。复用候选快照、预算和回答模型，轮换两路的运行顺序。

    retrieve(request,k) 接收可信 case.input，generate(messages) 异步返回 JSON
    文本，可通过 last_usage 报告本次 total_tokens。开启重排的分支先执行策略
    复核，再调用同一生成器打分。输出含两路执行轨迹、成对对照和复核模板。
    """
    if mode not in ('offline_fixture', 'real_llm'):
        raise ValueError('mode 需要是 offline_fixture 或 real_llm')
    if type(candidate_k) is not int or not 1 <= candidate_k <= 20:
        raise ValueError('candidate_k 需要在 1～20 之间')
    if (type(max_tokens) is not int or type(reserved_tokens) is not int
            or max_tokens <= 0 or not 0 <= reserved_tokens <= max_tokens):
        raise ValueError('预算需要是正整数，预留额必须在预算内')
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output == root or output == root / 'knowledge' or root / 'knowledge' in output.parents:
        raise ValueError('不能将报告写入数据集输入目录')
    snapshot = validate_benchmark(root)
    cases = {case['case_id']: case for case in load_benchmark(root)['cases']}
    if case_ids is not None and (not isinstance(case_ids, list) or not case_ids
                                 or any(not isinstance(cid, str) or cid not in cases for cid in case_ids)
                                 or len(set(case_ids)) != len(case_ids)):
        raise ValueError('case_ids 需要是已知且不重复的用例')
    docs = {doc['doc_id']: doc for doc in read_manifest(root)}
    report = dict(title='Week9 Day7 方案验收明细', mode=mode, dataset_fingerprint=snapshot['fingerprint'],
                  comparison='week9_day7_rerank', model=getattr(generate, 'model', None),
                  backend_info=copy.deepcopy(backend_info or {}), candidate_k=candidate_k,
                  max_tokens=max_tokens, reserved_tokens=reserved_tokens, rows=[])
    for index, case_id in enumerate(cases if case_ids is None else case_ids):
        case = cases[case_id]
        request = build_case_input(case)
        started, error = time.perf_counter(), None
        try:
            candidates = copy.deepcopy(retrieve(copy.deepcopy(request), candidate_k))
        except Exception as failure:
            candidates, error = None, failure
        retrieval_seconds = time.perf_counter() - started

        def snapshot_retrieve(request, k):
            if error is not None:
                raise error
            return copy.deepcopy(candidates)

        order = PATHS if index % 2 == 0 else PATHS[::-1]
        for variant in order:
            calls = dict(answer=[], rerank=[])
            answer = _track_generation(generate, calls['answer'])
            scorer = JsonLLMReranker(_track_generation(generate, calls['rerank'])) if variant == 'rerank' else None
            trace = await collect_trace(request, snapshot_retrieve, answer, docs, candidate_k=candidate_k,
                                        max_tokens=max_tokens, reserved_tokens=reserved_tokens, score_pairs=scorer)
            expected = copy.deepcopy(case['expected'])
            report['rows'].append(dict(case_id=f'{case_id}__{variant}', question_id=case_id,
                variant=variant, run_order=list(order), category=case['category'], input=copy.deepcopy(request),
                expected=expected, trace=trace, assessment=grade_trace(trace, expected), model_requests=calls,
                shared_retrieval_seconds=retrieval_seconds, metrics=_score_row(trace, expected, retrieval_seconds, calls)))
    write_evaluation_report(output, report)
    write_comparison_report(report, output / 'comparison_report.json')
    return report


def write_comparison_report(report: dict, path: Path) -> dict:
    """汇总原始或人工复核后的明细。配对质量、延迟和用量分别给分母。"""
    if report.get('comparison') != 'week9_day7_rerank':
        raise ValueError('需要 Week9 Day7 方案明细')
    by_question = {}
    for row in report['rows']:
        question = by_question.setdefault(row['question_id'], {})
        if row['variant'] not in PATHS or row['variant'] in question:
            raise ValueError('每题每条方案应恰有一条记录')
        question[row['variant']] = row
    if any(set(paths) != set(PATHS) for paths in by_question.values()):
        raise ValueError('对照记录缺少成对方案')
    pairs = []
    for case_id, pair in by_question.items():
        sides = {}
        for variant, row in pair.items():
            grade = row['assessment']
            sides[variant] = {**row['metrics'], 'status': grade['final_status'], 'errors': grade['error_tags'],
                              'review_notes': grade['review']['notes']}
        pairs.append(dict(case_id=case_id, **sides))
    summary = dict(case_count=len(pairs),
                   statuses={variant: dict(Counter(pair[variant]['status'] for pair in pairs)) for variant in PATHS},
                   paired_deltas={name: paired_delta([(pair['baseline'][name], pair['rerank'][name])
                                                     for pair in pairs]) for name in METRICS})
    result = dict(mode=report['mode'], dataset_fingerprint=report['dataset_fingerprint'],
                  model=report['model'], backend_info=report['backend_info'],
                  candidate_k=report['candidate_k'], max_tokens=report['max_tokens'],
                  reserved_tokens=report['reserved_tokens'], summary=summary, rows=pairs)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Week9 Day7 基线与重排对照', '', f"模式为 `{report['mode']}`，共 {len(pairs)} 道配对题。", '',
             '| 方案 | 通过 | 失败 | 待复核 |', '|---|---:|---:|---:|']
    for variant, counts in summary['statuses'].items():
        lines.append(f"| {variant} | {counts.get('passed', 0)} | {counts.get('failed', 0)} | "
                     f"{counts.get('needs_review', 0)} |")
    lines.extend(['', '| 指标 | 有效配对 / 总数 | 重排减基线的均值 |', '|---|---:|---:|'])
    for name, delta in summary['paired_deltas'].items():
        value = delta['mean_delta']
        lines.append(f"| {name} | {delta['eligible_pairs']} / {delta['planned_pairs']} | "
                     f"{'未知' if value is None else f'{value:+.4f}'} |")
    lines.extend(['', '| 用例 | 关键摘录覆盖 基线 → 重排 | 最终状态 基线 → 重排 | 失败标签 基线 → 重排 |',
                  '|---|---|---|---|'])
    for pair in pairs:
        left, right = pair['baseline'], pair['rerank']
        lines.append(f"| {pair['case_id']} | {left['context_evidence_coverage']} → {right['context_evidence_coverage']} | "
                     f"{left['status']} → {right['status']} | "
                     f"{', '.join(left['errors']) or '-'} → {', '.join(right['errors']) or '-'} |")
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='离线刷新 Week9 Day7 的成对汇总')
    parser.add_argument('report', type=Path, help='evaluation_report.json 或 reviewed_report.json')
    source = parser.parse_args().report
    if source.name not in ('evaluation_report.json', 'reviewed_report.json'):
        parser.error('需要原始或人工复核后的方案验收明细')
    name = 'reviewed_comparison_report.json' if source.name == 'reviewed_report.json' else 'comparison_report.json'
    write_comparison_report(json.loads(source.read_text(encoding='utf-8')), source.with_name(name))
