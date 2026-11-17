"""Week9 Day2：Recall@K、RR@K、MRR@K 与召回诊断。

两个关键步骤分别练习命中文档组计数和首个相关项的排名倒数。
其余数据加载、阶段错误、分母和报告已提供。
默认不调用模型；真实检索复用 Week8 LangChain 组件，不调用回答型 LLM。
"""

import copy
import json
import math
from pathlib import Path
from statistics import fmean
import time
from typing import Callable

from rag.eval_dataset import (
    DEFAULT_DATASET, build_case_input, load_benchmark, reference_is_eligible, validate_benchmark,
)
from rag.preprocessing import read_manifest


DEFAULT_KS = (1, 3, 5, 10)


def _metric_inputs(groups: list[list[str]], ranked_doc_ids: list[str], k: int) -> None:
    """已提供：K 是正整数；排名允许同文档的不同 Chunk 重复占位。"""
    if type(k) is not int or k <= 0:
        raise ValueError('k 必须是正整数')
    if (not isinstance(ranked_doc_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in ranked_doc_ids)):
        raise ValueError('ranked_doc_ids 必须是非空白字符串列表，允许空列表和重复文档')
    if not isinstance(groups, list):
        raise ValueError('expected_doc_groups 必须是列表')
    seen = set()
    for group in groups:
        if (not isinstance(group, list) or not group
                or any(not isinstance(value, str) or not value.strip() for value in group)):
            raise ValueError('每个预期文档组必须是非空字符串列表')
        if len(set(group)) != len(group) or seen.intersection(group):
            raise ValueError('预期文档不能在组内或组间重复')
        seen.update(group)


def recall_at_k(expected_doc_groups: list[list[str]], ranked_doc_ids: list[str], k: int) -> float | None:
    """练习 1：计算前 K 个 Chunk 覆盖的目标文档组比例，返回 [0,1] 或 None。

    输入：expected_doc_groups 来自 case.expected.doc_groups，组内可替代、组间都需要；
    ranked_doc_ids 是按检索排名排列的 Chunk.doc_id，允许重复，不修改输入。
    先截取前 K 项再去重计分；同一文档的两个 Chunk 占两个位置，但不能重复加分。
    无目标组返回 None（不适用）；有目标但没命中返回 0.0。
    示例：目标 [[A,B],[C]]，排名 [X,A,A,C]，K=3 时命中第一组，Recall@3=1/2。
    """
    _metric_inputs(expected_doc_groups, ranked_doc_ids, k)
    if not expected_doc_groups:
        return None
    # 先取前 K 个 Chunk，再按文档去重；重复文档仍然占用原排名位置。
    present = set(ranked_doc_ids[:k])
    # 学习点 W9D2-1：组内任一文档命中即算覆盖该组，每组最多贡献 1 分。
    # any() 返回布尔值，sum() 将每组的 True/False 分别按 1/0 累加。
    matched_groups = sum(
        any(doc_id in present for doc_id in group)
        for group in expected_doc_groups
    )
    return matched_groups / len(expected_doc_groups)


def reciprocal_rank_at_k(expected_doc_groups: list[list[str]], ranked_doc_ids: list[str], k: int) -> float | None:
    """练习 2：单题 RR@K，首个相关 Chunk 的排名倒数；不是求所有相关项的平均。

    输入与 recall_at_k 相同。目标组中任一文档都算相关；从 1 开始计排名。
    输出：首次命中第 r 位时为 1/r，前 K 项没有命中为 0.0，无目标组为 None。
    示例：目标 [[A],[C]]，排名 [X,A,A,C]，K=3 时 RR@3=1/2，尽管 Recall@3 只有 1/2。
    框架负责把多题 RR 的均值汇总为 MRR；这里不写批量平均。
    """
    _metric_inputs(expected_doc_groups, ranked_doc_ids, k)
    if not expected_doc_groups:
        return None
    relevant = {doc_id for group in expected_doc_groups for doc_id in group}
    head = ranked_doc_ids[:k]
    # 学习点 W9D2-2：实际排名从 1 开始；空结果或前 K 项未命中时保留 0.0。
    score = 0.0
    for rank, doc_id in enumerate(head, start=1):
        if doc_id in relevant:
            score = 1.0 / rank
            # RR 只取首个相关项；及时退出，避免后续相关项覆盖它的倒数排名。
            break

    return score


def load_rankings_fixture(dataset_dir: str | Path = DEFAULT_DATASET) -> dict[str, list[str]]:
    """已提供：读取人工排名。只验证计算/报告，不把它当真实语义检索成绩。"""
    root = Path(dataset_dir)
    data = json.loads((root / 'rankings_day2.json').read_text(encoding='utf-8'))
    benchmark = load_benchmark(root)
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data['schema_version'] != 1 or data.get('mode') != 'offline_fixture'
            or data.get('dataset_id') != benchmark['dataset_id'] or not isinstance(data.get('rankings'), dict)):
        raise ValueError('需要同一数据集的版本 1 人工排名')
    rows = data['rankings']
    if set(rows) != {case['case_id'] for case in benchmark['cases']}:
        raise ValueError('人工排名必须恰好覆盖全部用例 ID')
    known = {document['doc_id'] for document in read_manifest(root)}
    for doc_ids in rows.values():
        _metric_inputs([], doc_ids, 1)
        if not set(doc_ids) <= known:
            raise ValueError('人工排名引用了未知文档')
    return rows


def _diagnostics(case: dict, hits: list[dict]) -> dict:
    """已提供：评分参考只在检索后参与诊断，不参与候选准备或排名。"""
    doc_ids = list(dict.fromkeys(hit['chunk']['doc_id'] for hit in hits))
    groups = case['expected']['doc_groups']
    relevant = {doc_id for group in groups for doc_id in group}
    return dict(
        missed_doc_groups=[list(group) for group in groups if not set(group).intersection(doc_ids)],
        # 非目标来源也可能是有用背景，不能仅凭不在 gold 中就断言语义错误。
        non_target_doc_ids=[doc_id for doc_id in doc_ids if doc_id not in relevant] if groups else None,
        forbidden_doc_ids=[doc_id for doc_id in doc_ids if doc_id in case['expected']['forbidden_doc_ids']],
        ineligible_doc_ids=list(dict.fromkeys(hit['chunk']['doc_id'] for hit in hits
            if not reference_is_eligible(hit['chunk']['metadata'], case['input']))),
    )


def summarize_rows(rows: list[dict], ks: list[int], *, planned_cases: int, planned_applicable: int) -> dict:
    """已提供：MRR 是有效 RR 的均值；未知题不入分母，执行错误不伪装成零分。

    同时保留计划适用题数与实际算分题数，避免请求失败后悄悄缩小分母。
    """
    by_k = {}
    for k in ks:
        scored = [row['by_k'][str(k)] for row in rows if row['execution_error'] is None]
        recalls = [item['recall_at_k'] for item in scored if item['recall_at_k'] is not None]
        rrs = [item['rr_at_k'] for item in scored if item['rr_at_k'] is not None]
        by_k[str(k)] = dict(mean_recall_at_k=fmean(recalls) if recalls else None,
                            mrr_at_k=fmean(rrs) if rrs else None, evaluated_cases=len(recalls),
                            full_recall_cases=sum(value == 1.0 for value in recalls))
    return dict(planned_cases=planned_cases, completed_cases=len(rows), planned_applicable_cases=planned_applicable,
                not_applicable_cases=planned_cases - planned_applicable,
                execution_error_count=sum(row['execution_error'] is not None for row in rows), by_k=by_k)


def _validate_hits(hits: list[dict], documents: dict) -> None:
    """已提供：保留原检索顺序，拒绝未知来源、重复 Chunk 或被改写的来源元数据。"""
    if not isinstance(hits, list):
        raise ValueError('检索结果必须为列表')
    ids = set()
    for hit in hits:
        if not isinstance(hit, dict) or not isinstance(hit.get('chunk'), dict):
            raise ValueError('每个结果需要 chunk 字典')
        chunk, score = hit['chunk'], hit.get('score')
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score)):
            raise ValueError('score 必须为有限数值')
        chunk_id, doc_id = chunk.get('chunk_id'), chunk.get('doc_id')
        if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in ids:
            raise ValueError('Chunk ID 必须非空且不能重复')
        if not isinstance(doc_id, str) or doc_id not in documents:
            raise ValueError('检索结果引用未知文档')
        metadata = chunk.get('metadata')
        if not isinstance(metadata, dict) or any(metadata.get(key) != documents[doc_id].get(key)
                for key in ('source', 'version', 'knowledge_status', 'engine_code', 'required_scopes')):
            raise ValueError('来源元数据与冻结清单不一致')
        ids.add(chunk_id)


def render_metrics_report(report: dict) -> str:
    """已提供：人可读的指标、分母与遗漏定位；不把 pytest 成功当质量满分。"""
    def number(value):
        return '不适用/未完成' if value is None else f'{value:.4f}'
    summary = report['summary']
    lines = ['# Week9 Day2 检索基线', '', f"模式：`{report['mode']}`；知识快照：`{report['dataset_fingerprint']}`。", '',
             '人工排名只验证算法和报告；真实基线也不代表回答语义、权限隔离或召回全部通过。', '',
             f"已记录 {summary['completed_cases']}/{summary['planned_cases']} 题；计划适用 {summary['planned_applicable_cases']} 题；"
             f"不适用 {summary['not_applicable_cases']} 题；执行错误 {summary['execution_error_count']} 题。", '',
             '| K | 平均 Recall@K | MRR@K | 实际算分/计划适用 | 全召回题数 |', '|---|---|---|---|---|']
    for k, item in summary['by_k'].items():
        lines.append(f"| {k} | {number(item['mean_recall_at_k'])} | {number(item['mrr_at_k'])} | "
                     f"{item['evaluated_cases']}/{summary['planned_applicable_cases']} | {item['full_recall_cases']} |")
    lines += ['', '## 逐题诊断', '', '| 用例 | K | Recall | RR | 遗漏文档组 | 禁止来源 | 不适用来源 |',
              '|---|---|---|---|---|---|---|']
    for row in report['rows']:
        if row['execution_error'] is not None:
            lines.append(f"| {row['case_id']} | - | 未完成 | 未完成 | {row['execution_error']['type']} | - | - |")
            continue
        for k, item in row['by_k'].items():
            lines.append(f"| {row['case_id']} | {k} | {number(item['recall_at_k'])} | {number(item['rr_at_k'])} | "
                         f"{json.dumps(item['missed_doc_groups'], ensure_ascii=False)} | "
                         f"{', '.join(item['forbidden_doc_ids']) or '-'} | {', '.join(item['ineligible_doc_ids']) or '-'} |")
    return '\n'.join(lines) + '\n'


def run_retrieval_evaluation(dataset_dir: str | Path, retrieve: Callable[[dict, int], list[dict]], output_dir: str | Path, *,
                             mode: str, ks=DEFAULT_KS, backend_info: dict | None = None) -> dict:
    """已提供：每题只检索一次 max(K)，按同一排名的前缀评分并逐题落盘。

    retrieve(request, k) -> 标准 [{score, chunk}]；request 是独立复制的 case.input，
    不含 expected、case_id 或参考答案。mode 明确区分人工排名和真实本地模型。
    返回完整 JSON 报告；异常保存阶段/类型并继续后续题，不记录可能带凭据的异常正文。
    未实现的检索器直接抛出 NotImplementedError，不把框架未完成当成模型故障。
    """
    if mode not in ('offline_fixture', 'local_model'):
        raise ValueError('mode 必须为 offline_fixture 或 local_model')
    if (not isinstance(ks, (list, tuple)) or not ks
            or any(type(k) is not int or k <= 0 for k in ks) or len(set(ks)) != len(ks)):
        raise ValueError('ks 必须为不重复的正整数列表或元组')
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output == root or output == root / 'knowledge' or root / 'knowledge' in output.parents:
        raise ValueError('输出目录不能覆盖输入或知识目录')
    snapshot = validate_benchmark(root)
    cases = load_benchmark(root)['cases']
    documents = {row['doc_id']: row for row in read_manifest(root)}
    cutoffs = sorted(ks)
    policy = ('人工排名故意包含错误版本和越权来源，仅验证指标与诊断，不代表执行了 filters'
              if mode == 'offline_fixture' else '复用 Week8 精确 filters，尚未执行访问权限过滤')
    report = dict(mode=mode, dataset_fingerprint=snapshot['fingerprint'], ks=cutoffs,
                  backend_info=copy.deepcopy(backend_info or {}), baseline_policy=policy,
                  generation_called=False, rows=[])
    output.mkdir(parents=True, exist_ok=True)

    def save():
        report['summary'] = summarize_rows(report['rows'], cutoffs, planned_cases=len(cases),
            planned_applicable=sum(bool(case['expected']['doc_groups']) for case in cases))
        (output / 'retrieval_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (output / 'retrieval_report.md').write_text(render_metrics_report(report), encoding='utf-8')

    save()
    for case in cases:
        request = build_case_input(case)
        row = dict(case_id=case['case_id'], category=case['category'], input=copy.deepcopy(request),
                   hits=None, by_k={}, execution_error=None)
        started, stage = time.perf_counter(), 'retrieval'
        try:
            hits = retrieve(request, max(cutoffs))
            stage = 'retrieval_validation'
            _validate_hits(hits, documents)
            row['hits'] = copy.deepcopy(hits[:max(cutoffs)])
        except NotImplementedError:
            raise
        except Exception as error:
            row['execution_error'] = dict(stage=stage, type=type(error).__name__)
        row['retrieval_seconds'] = time.perf_counter() - started
        if row['execution_error'] is None:
            ids = [hit['chunk']['doc_id'] for hit in row['hits']]
            for k in cutoffs:
                row['by_k'][str(k)] = dict(
                    recall_at_k=recall_at_k(case['expected']['doc_groups'], ids, k),
                    rr_at_k=reciprocal_rank_at_k(case['expected']['doc_groups'], ids, k),
                    **_diagnostics(case, row['hits'][:k]),
                )
        report['rows'].append(row)
        save()
        print(f"{case['case_id']}: {'recorded' if row['execution_error'] is None else row['execution_error']['type']}")
    return report


def prepare_langchain_baseline(dataset_dir: str | Path, index_dir: str | Path, encoder, *,
                                embedding_mode: str = 'local_model', rebuild: bool = False) -> tuple[Callable, dict]:
    """已提供：复用 Week8 的建库/加载/检索，使用 Week9 独立缓存，不自动修复旧缓存。

    返回 (retrieve, info)。需保持知识及题库 fingerprint、模型和查询前缀一致。
    本日测量未经权限优化的基线，不把 Day1 的参考校验悄悄变成检索过滤。
    """
    from rag.langchain_preparation import Week7Embeddings
    from rag.langchain_rag import documents_to_hits, make_retriever, prepare_store

    snapshot = validate_benchmark(dataset_dir)
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path, stamp = directory / 'knowledge_store.json', directory / 'dataset_fingerprint.json'
    if path.exists() and not rebuild:
        if not stamp.exists() or json.loads(stamp.read_text(encoding='utf-8')).get('fingerprint') != snapshot['fingerprint']:
            raise ValueError('缓存知识快照不匹配，请显式 rebuild=True 重建')
    store, action = prepare_store(dataset_dir, path, Week7Embeddings(encoder),
                                  embedding_mode=embedding_mode, rebuild=rebuild)
    if action == 'built':
        stamp.write_text(json.dumps({'fingerprint': snapshot['fingerprint']}) + '\n', encoding='utf-8')

    def retrieve(request, k):
        retriever = make_retriever(store, k=k, filters=request['filters'])
        return documents_to_hits(retriever.invoke(request['question']))

    return retrieve, dict(store_action=action, embedding_model=encoder.model, query_instruction=encoder.query_instruction,
                          embedding_mode=embedding_mode, chunk_count=len(store.store))
