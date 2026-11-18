"""Week9 Day3：精确标识符约束、轻量关键词检索与 RRF 融合。

两个关键学习步骤已完成：词项覆盖分数与倒数排名贡献；分词、候选准备、去重、报告已提供。
复用既有 LangChain 向量库和 Day2 指标，不增加数据库、分词模型或回答型 LLM。
"""

import copy
import json
import math
from pathlib import Path
import re
from typing import Callable

from rag.eval_dataset import DEFAULT_DATASET, _check_input, validate_benchmark
from rag.retrieval_metrics import DEFAULT_KS, run_retrieval_evaluation


STRATEGIES = ('dense', 'keyword', 'hybrid')
ASCII_WORD = re.compile(r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])')


def _positive(value, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} 必须为正整数')


def tokenize(text: str) -> set[str]:
    """已提供：英文标识符保持完整及大小写，连续中文取二字片段，版本数字另保留。

    NOT_FOUND、NOT_FOUND_EXTRA、import_std 都是不同词项，不用子串代替词项。
    这只是教学用确定性切词，不代表中文语义分词，也不计算词频或 BM25。
    """
    if not isinstance(text, str):
        raise ValueError('text 必须为字符串')
    tokens = set(ASCII_WORD.findall(text))
    tokens.update(re.findall(r'\d+(?:\.\d+)*', text))
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        tokens.update(run[index:index + 2] for index in range(len(run) - 1))
        if len(run) == 1:
            tokens.add(run)
    return tokens


def identifier_constraints(request: dict) -> dict:
    """已提供：从明确字段或原文提取标识符，不猜测同义编码、大小写或身份权限。

    现有 filters.engine_code 优先；省略时识别教学平台的 import_*/export_* 原值。
    多流程比较允许这些编码的并集，不擅自选一个。其余含下划线的词及 FORBIDDEN
    作为正文标识符；比较多个标识符时允许分别命中，避免要求每个 Chunk 同时含全部错误码。
    这是本教学语料的确定性规则，不是通用自然语言实体解析器。
    """
    _check_input(request)
    words = set(ASCII_WORD.findall(request['question']))
    engines = {word for word in words if word.startswith(('import_', 'export_'))}
    explicit = request['filters'].get('engine_code')
    textual = {word for word in words if ('_' in word or word == 'FORBIDDEN') and word not in engines}
    return dict(engine_codes=[explicit] if explicit is not None else sorted(engines),
                text_identifiers=sorted(textual), engine_source='filters' if explicit is not None else 'question')


def matches_identifiers(chunk: dict, constraints: dict) -> bool:
    """已提供：流程看元数据精确相等，正文标识符看完整词项；不把正文提及当流程归属。"""
    engines = constraints['engine_codes']
    if engines and chunk['metadata'].get('engine_code') not in engines:
        return False
    identifiers = constraints['text_identifiers']
    return not identifiers or bool(set(identifiers).intersection(ASCII_WORD.findall(chunk['text'])))


def _check_chunks(chunks: list[dict]) -> None:
    if not isinstance(chunks, list):
        raise ValueError('chunks 必须为列表')
    ids = set()
    for chunk in chunks:
        if (not isinstance(chunk, dict) or not isinstance(chunk.get('chunk_id'), str)
                or not chunk['chunk_id'].strip() or chunk['chunk_id'] in ids
                or not isinstance(chunk.get('text'), str) or not chunk['text'].strip()
                or not isinstance(chunk.get('metadata'), dict)):
            raise ValueError('Chunk 需要唯一 ID、非空正文和 metadata')
        ids.add(chunk['chunk_id'])


def keyword_search(query: str, chunks: list[dict], *, k: int = 20) -> list[dict]:
    """练习 1：计算轻量词项覆盖分数，返回按分数降序的标准 hits，不修改输入。

    输入：问题原文、已按 filters/标识符筛好的候选 Chunk、正整数 K。
    输出：[{score, chunk, retrieval}]；零命中不返回，同分按 chunk_id 升序，最多 K 项。
    分数 = 查询词项中被当前 Chunk 命中的数量 / 查询词项总数，范围 [0,1]。
    query_tokens/document_tokens 已提供，例如 {NOT_FOUND, 含义} 与 {NOT_FOUND} 的分数为 1/2。
    不将列表排名当分数，也不调用 Embedding；无词项或无候选直接返回 []。
    """
    _positive(k, 'k')
    _check_chunks(chunks)
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    results = []
    for chunk in chunks:
        document_tokens = tokenize(chunk['text'])
        # 学习点 W9D3-1：命中词项数除以查询词项总数，得到 [0,1] 内的覆盖比例。
        score = len(query_tokens & document_tokens) / len(query_tokens)
        if score > 0:
            results.append(dict(score=score, chunk=copy.deepcopy(chunk), retrieval=dict(
                method='keyword', score_kind='query_token_coverage',
                matched_tokens=sorted(query_tokens.intersection(document_tokens)))))
    results.sort(key=lambda hit: (-hit['score'], hit['chunk']['chunk_id']))
    return results[:k]


def reciprocal_rank_fusion(rankings: dict[str, list[dict]], *, k: int = 10,
                            rank_constant: int = 60) -> list[dict]:
    """练习 2：融合多路 Chunk 排名。RRF 与 Day2 的 RR/MRR 评估指标用途不同。

    输入：例如 {'dense': hits1, 'keyword': hits2}，每路已从高到低排序。
    输出：标准 hits，score 为各路 1/(rank_constant + rank) 的和，不是余弦相似度。
    rank 从 1 开始；同一路同一 Chunk 只贡献一次，跨路相同 Chunk 累加。
    同文档的不同 Chunk 保持独立；同分按 chunk_id 升序；retrieval 保留各路排名与原分数。
    示例：c=60，同一 Chunk 两路都排第 1，则融合分数为 1/61 + 1/61。
    """
    _positive(k, 'k')
    _positive(rank_constant, 'rank_constant')
    if not isinstance(rankings, dict):
        raise ValueError('rankings 必须是分支名到 hits 的字典')
    scores, chunks, ranks, raw_scores = {}, {}, {}, {}
    for branch, hits in rankings.items():
        if not isinstance(branch, str) or not branch.strip() or not isinstance(hits, list):
            raise ValueError('每个分支需要非空名称和 hits 列表')
        seen = set()
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict) or 'chunk' not in hit:
                raise ValueError('hit 需要 chunk')
            chunk = hit['chunk']
            _check_chunks([chunk])
            score = hit.get('score')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('原始分数必须为有限数值')
            chunk_id = chunk['chunk_id']
            if chunk_id in chunks and chunks[chunk_id] != chunk:
                raise ValueError('相同 Chunk ID 对应的原文或元数据不一致')
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks[chunk_id] = copy.deepcopy(chunk)
            # 学习点 W9D3-2：每路按从 1 开始的 rank 贡献分数，再按 Chunk ID 跨路累加。
            contribution = 1 / (rank_constant + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            ranks.setdefault(chunk_id, {})[branch] = rank
            raw_scores.setdefault(chunk_id, {})[branch] = score
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:k]
    return [dict(score=scores[chunk_id], chunk=chunks[chunk_id], retrieval=dict(
        method='rrf', score_kind='rrf', rank_constant=rank_constant,
        ranks=ranks[chunk_id], raw_scores=raw_scores[chunk_id])) for chunk_id in ordered]


class SharedQueryEncoder:
    """已提供：同次三策略对照复用相同问题向量，避免重复编码影响公平性。

    只缓存本次进程内的 query，不缓存正文或模型故障；返回副本防止向量被调用方修改。
    因此本次对照的耗时包含缓存影响，不能据此下模型/框架性能结论。
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.model, self.query_instruction = encoder.model, encoder.query_instruction
        self.cache = {}

    def encode_document(self, texts):
        return self.encoder.encode_document(texts)

    def encode_query(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            vectors = self.encoder.encode_query(missing)
            if len(vectors) != len(missing):
                raise ValueError('问题向量条数不匹配')
            self.cache.update(zip(missing, copy.deepcopy(vectors)))
        return [copy.deepcopy(self.cache[text]) for text in texts]


def prepare_search_store(dataset_dir: str | Path, index_dir: str | Path, encoder, *,
                           embedding_mode: str = 'local_model', rebuild: bool = False):
    """已提供：复用 Day2 缓存检查与 Week8 load_store；第二次加载不会重新编码正文。"""
    from rag.langchain_preparation import Week7Embeddings
    from rag.langchain_rag import load_store
    from rag.retrieval_metrics import prepare_langchain_baseline

    _, info = prepare_langchain_baseline(dataset_dir, index_dir, encoder,
        embedding_mode=embedding_mode, rebuild=rebuild)
    store = load_store(Path(index_dir) / 'knowledge_store.json', Week7Embeddings(encoder),
                       embedding_mode=embedding_mode)
    return store, info


def make_search_retriever(store, strategy: str, *, candidate_k: int = 20, rank_constant: int = 60,
                          candidate_filter: Callable[[dict, dict], bool] | None = None):
    """已提供：准备 dense/keyword/hybrid 回调，兼容 Day2 的 retrieve(request, k)。

    dense 保留 Day2 基线；keyword/hybrid 在两路共同候选中执行精确标识符约束。
    每路最多 candidate_k 项，融合后再取 K。candidate_filter(metadata, request)
    可在两路共同的候选阶段增加条件，省略时沿用 Day3 基线。
    """
    from dataclasses import asdict
    from rag.langchain_rag import _store_info, document_chunk
    from rag.retrieval import _check_search_options, _matches_filters

    if strategy not in STRATEGIES:
        raise ValueError('strategy 必须为 dense/keyword/hybrid')
    _positive(candidate_k, 'candidate_k')
    _positive(rank_constant, 'rank_constant')
    if candidate_filter is not None and not callable(candidate_filter):
        raise ValueError('candidate_filter 必须可调用')
    _store_info(store)
    chunks = [asdict(document_chunk(document)) for document in store.get_by_ids(list(store.store))]

    def retrieve(request, k):
        _check_input(request)
        _check_search_options(k, request['filters'], None)
        if k > candidate_k:
            raise ValueError('最终 K 不能超过固定 candidate_k')
        constraints = identifier_constraints(request)
        candidates = [chunk for chunk in chunks
                      if (candidate_filter is None or candidate_filter(chunk['metadata'], request))
                      and _matches_filters(chunk['metadata'], request['filters'])
                      and (strategy == 'dense' or matches_identifiers(chunk, constraints))]
        if not candidates:
            return []
        if strategy == 'keyword':
            return keyword_search(request['question'], candidates, k=candidate_k)[:k]
        ids = {chunk['chunk_id'] for chunk in candidates}
        pairs = store.similarity_search_with_score(request['question'], k=candidate_k,
                                                   filter=lambda document: document.id in ids)
        dense = [dict(score=float(score), chunk=asdict(document_chunk(document)), retrieval=dict(
            method='dense', score_kind='cosine')) for document, score in pairs]
        if strategy == 'dense':
            return dense[:k]
        lexical = keyword_search(request['question'], candidates, k=candidate_k)
        return reciprocal_rank_fusion({'dense': dense, 'keyword': lexical}, k=k, rank_constant=rank_constant)

    return retrieve


def run_search_comparison(dataset_dir: str | Path, store, output_dir: str | Path, *,
                            mode: str, candidate_k: int = 20, rank_constant: int = 60,
                            backend_info: dict | None = None) -> dict:
    """已提供：三种策略使用同一题库/向量快照，复用 Day2 指标与逐题异常报告。

    保存各策略 retrieval_report.json/md 及 comparison_report.json/md。
    指标变化只记录实测，不要求混合检索必然优于向量基线，不根据参考答案改排名。
    """
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output == root or output == root / 'knowledge' or root / 'knowledge' in output.parents:
        raise ValueError('输出目录不能覆盖输入或知识目录')
    _positive(candidate_k, 'candidate_k')
    if candidate_k < max(DEFAULT_KS):
        raise ValueError('candidate_k 至少覆盖全部评估 K')
    snapshot = validate_benchmark(root)
    report = dict(mode=mode, dataset_fingerprint=snapshot['fingerprint'],
        candidate_k=candidate_k, rank_constant=rank_constant, generation_called=False,
        backend_info=copy.deepcopy(backend_info or {}), planned_strategies=list(STRATEGIES), strategies={})
    for strategy in STRATEGIES:
        print(f'正在记录策略：{strategy}')
        retrieve = make_search_retriever(store, strategy, candidate_k=candidate_k, rank_constant=rank_constant)
        info = dict(backend_info or {}, strategy=strategy, candidate_k=candidate_k,
                    rank_constant=rank_constant, identifier_guard=strategy != 'dense')
        result = run_retrieval_evaluation(root, retrieve, output / strategy, mode=mode, backend_info=info)
        report['strategies'][strategy] = copy.deepcopy(result['summary'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'comparison_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        lines = ['# Week9 Day3 检索对照', '', f'模式：`{mode}`；同一知识/题库快照。', '',
                 '人工向量不能证明语义质量；真实结果不预设混合策略胜出，权限与失效文档仍需后续过滤。', '',
                 '| 策略 | K | 平均 Recall | MRR | 实际算分/计划适用 | 执行错误 |', '|---|---|---|---|---|---|']
        for name, summary in report['strategies'].items():
            for k, values in summary['by_k'].items():
                recall = values['mean_recall_at_k']
                mrr = values['mrr_at_k']
                lines.append(f"| {name} | {k} | {recall if recall is not None else '未完成'} | "
                    f"{mrr if mrr is not None else '未完成'} | {values['evaluated_cases']}/{summary['planned_applicable_cases']} | "
                    f"{summary['execution_error_count']} |")
        (output / 'comparison_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report
