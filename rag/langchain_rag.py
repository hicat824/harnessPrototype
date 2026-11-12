"""Week8 Day5：Vector Store → Retriever → Context → 既有回答层。

已完成三个关键连接点：批量入库、带过滤的检索，以及 Runnable 回答链。
使用 LangChain 内存向量库和 JSON，不启动数据库，不修改主项目 Agent。
学习说明与逐项测试见 langchain_learning/week8.md#day5。
"""

import copy
from dataclasses import asdict
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langsmith import tracing_context
from pydantic import Field

from rag.answering import Generate, answer_from_context, build_context
from rag.embeddings import _check_texts, _vectors
from rag.langchain_preparation import ManifestLoader, Week7Embeddings, split_documents
from rag.preprocessing import KnowledgeChunk
from rag.retrieval import _check_search_options, _encoder_config, _matches_filters


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'langchain_learning/artifacts/day5'
LOCATION_FIELDS = ('doc_id', 'segment_id', 'section_path', 'start_char', 'end_char', 'start_line', 'end_line')


def document_chunk(document: Document) -> KnowledgeChunk:
    """已提供：恢复 Week7 Chunk 契约，位置字段回到顶层，来源/版本保留在 metadata。

    不把 Document 对象直接交给回答层，也不重复把所有位置字段塞入 Context 元数据。
    retrieval_score 是本次查询的临时字段，移除后不影响原来源元数据。
    """
    metadata = copy.deepcopy(document.metadata)
    metadata.pop('retrieval_score', None)
    try:
        location = {key: metadata.pop(key) for key in LOCATION_FIELDS}
    except KeyError as error:
        raise ValueError('Chunk Document 缺少来源身份或定位字段') from error
    return KnowledgeChunk(chunk_id=document.id, text=document.page_content, metadata=metadata, **location)


def _check_chunks(chunks: list[Document]) -> None:
    if not isinstance(chunks, list) or any(not isinstance(item, Document) for item in chunks):
        raise ValueError('chunks 必须是 Document 列表')
    ids = set()
    for document in chunks:
        if not isinstance(document.id, str) or not document.id.strip() or document.id in ids:
            raise ValueError('Chunk ID 必须非空且不能重复')
        ids.add(document.id)
        _check_texts([document.page_content])
        chunk = document_chunk(document)
        if (not isinstance(chunk.doc_id, str) or not chunk.doc_id
                or any(not isinstance(chunk.metadata.get(key), str) or not chunk.metadata[key].strip()
                       for key in ('source', 'version'))):
            raise ValueError('Chunk 必须保留 doc_id/source/version')
        if 'retrieval_score' in document.metadata:
            raise ValueError('本次检索分数不能作为固定知识元数据入库')


def _store_info(store: InMemoryVectorStore) -> dict:
    """已提供：检查当前轻量向量库，返回条数/维数；不请求模型。

    store.store 是该内存实现保存的 id→记录字典，本日只用它核对 JSON 完整性和候选。
    实际入库、余弦计算和检索仍调用 LangChain，不重新实现向量库。
    """
    rows = list(store.store.values())
    try:
        documents = [Document(id=row['id'], page_content=row['text'], metadata=row['metadata']) for row in rows]
        vectors = _vectors([row['vector'] for row in rows])
        if any(key != row['id'] for key, row in store.store.items()):
            raise ValueError('向量库键与记录 ID 不一致')
    except (KeyError, TypeError) as error:
        raise ValueError('向量库记录缺少正文、元数据或向量') from error
    _check_chunks(documents)
    return dict(chunk_count=len(rows), dimensions=len(vectors[0]) if vectors else 0)


def build_store(chunks: list[Document], embeddings: Week7Embeddings) -> InMemoryVectorStore:
    """建库：一次批量编码正文，返回独立的内存向量库，不执行问题查询。

    输入为 Day4 Chunk Document 列表和同一 Week7Embeddings；ID/正文/元数据均保留。
    空列表不编码；重复 ID 或缺失来源在模型调用前拒绝。
    学习点 W8D5-1：把 owned 交给 store.add_documents 的 documents 参数。
    owned 已深拷贝，避免内存库保留调用方 metadata 引用。
    """
    _check_chunks(chunks)
    _encoder_config(embeddings.encoder)
    store = InMemoryVectorStore(embeddings)
    if chunks:
        owned = copy.deepcopy(chunks)
        # 批量入库时编码正文；查询阶段只编码问题，后续加载缓存不重复编码正文。
        store.add_documents(owned)
    _store_info(store)
    return store


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _config_path(path: Path) -> Path:
    return path.with_name(path.stem + '.config.json')


def save_store(store: InMemoryVectorStore, path: str | Path, *, embedding_mode: str) -> None:
    """已提供：用 LangChain dump 保存向量，再旁存模型配置；不调用模型。

    embedding_mode 只能为 offline_fixture/local_model，不能把人工向量当真实索引。
    两个 JSON 文件应一起保存。仅适用于本地学习，不实现事务或自动重建策略。
    """
    if embedding_mode not in ('offline_fixture', 'local_model'):
        raise ValueError('embedding_mode 必须明确区分人工向量和本地模型')
    info = _store_info(store)
    model, instruction = _encoder_config(store.embedding.encoder)
    path = Path(path)
    store.dump(str(path))
    # dump 默认转义中文；重排成中文可读 JSON，不改变文件的数据结构。
    _save_json(path, json.loads(path.read_text(encoding='utf-8')))
    _save_json(_config_path(path), dict(schema_version=1, model_name=model,
                                       query_instruction=instruction, embedding_mode=embedding_mode, **info))


def load_store(path: str | Path, embeddings: Week7Embeddings, *, embedding_mode: str) -> InMemoryVectorStore:
    """已提供：恢复已保存的向量，不编码正文；模型/查询前缀/向量模式必须一致。

    配置和实际维数/条数不一致时抛 ValueError，缺文件或非法 JSON 原样报错。
    不仅凭维数相同就混用模型；JSON 来自本地练习产物，不加载 pickle。
    """
    path = Path(path)
    config = json.loads(_config_path(path).read_text(encoding='utf-8'))
    model, instruction = _encoder_config(embeddings.encoder)
    if (not isinstance(config, dict)
            or type(config.get('schema_version')) is not int or config['schema_version'] != 1
            or config.get('model_name') != model or config.get('query_instruction') != instruction
            or config.get('embedding_mode') != embedding_mode):
        raise ValueError('索引模型、查询前缀、向量模式或版本不匹配，请使用原配置或显式重建')
    store = InMemoryVectorStore.load(str(path), embeddings)
    info = _store_info(store)
    if any(type(config.get(key)) is not int or config[key] != value for key, value in info.items()):
        raise ValueError('索引声明的条数/维数与实际内容不一致')
    return store


def prepare_store(dataset_dir: str | Path, path: str | Path, embeddings: Week7Embeddings, *,
                  embedding_mode: str, rebuild: bool = False) -> tuple[InMemoryVectorStore, str]:
    """已提供：首次由 Day4 加载/分块后建库，后续只加载缓存，返回 (store, built/loaded)。

    源文档变更后由调用方显式 rebuild=True；本练习不自动检测知识更新。
    不自动删除不兼容缓存或在加载失败时悄悄重编码；异常交给调用方处理。
    """
    path = Path(path)
    if path.exists() and not rebuild:
        return load_store(path, embeddings, embedding_mode=embedding_mode), 'loaded'
    chunks = split_documents(ManifestLoader(dataset_dir).load())
    store = build_store(chunks, embeddings)
    save_store(store, path, embedding_mode=embedding_mode)
    return store, 'built'


class KnowledgeRetriever(BaseRetriever):
    """返回带来源和真实余弦分数的 Document，接回 Week7 Context 契约。

    普通 as_retriever 默认只返回 Document、不附分数。本日用一个小型 BaseRetriever
    适配器调用带分数检索，避免把排名或距离伪造成旧接口要求的余弦分数。
    候选为空时先短路，不调用问题编码器；filter 是程序配置，不由模型猜测。
    """

    store: InMemoryVectorStore
    k: int = 5
    filters: dict = Field(default_factory=dict)
    min_score: float | None = None

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        """学习点 W8D5-2：调用带分数检索，传 query、k=self.k 和 filter=predicate。

        输入为问题字符串；输出 [(Document, score), ...] 中的 Document 副本，
        在 metadata.retrieval_score 保存本次分数，不改库内元数据。
        当前后端分数是余弦相似度，越高越相似；不能推广为所有向量库的分数语义。
        """
        _check_texts([query])
        _check_search_options(self.k, self.filters, self.min_score)
        if not any(_matches_filters(row['metadata'], self.filters) for row in self.store.store.values()):
            return []
        predicate = lambda document: _matches_filters(document.metadata, self.filters)
        # 后端使用单数 filter：先过滤候选，再在候选中计算相似度并取 Top-K。
        pairs = self.store.similarity_search_with_score(query, k=self.k, filter=predicate)
        results = []
        for document, score in pairs:
            # 浮点误差可能略越过端点；不把分数改成距离或排名。
            score = max(-1.0, min(1.0, float(score)))
            if self.min_score is None or score >= self.min_score:
                item = copy.deepcopy(document)
                item.metadata['retrieval_score'] = score
                results.append(item)
        return results


def make_retriever(store: InMemoryVectorStore, *, k: int = 5, filters: dict | None = None,
                   min_score: float | None = None) -> KnowledgeRetriever:
    """已提供：校验检索配置，不请求模型；三个精确过滤条件间为 AND。"""
    filters = _check_search_options(k, filters, min_score)
    _store_info(store)
    return KnowledgeRetriever(store=store, k=k, filters=copy.deepcopy(filters), min_score=min_score)


def documents_to_hits(documents: list[Document]) -> list[dict]:
    """已提供：把 Retriever 结果适配为 Week7 的 [{score, chunk}]，不改正文和来源。"""
    return [dict(score=document.metadata['retrieval_score'], chunk=asdict(document_chunk(document)))
            for document in documents]


def build_rag_chain(retriever: KnowledgeRetriever, generate: Generate, *, max_context_chars: int = 6000):
    """组装最小检索链，构造本身不执行模型。

    retriever 来自 make_retriever；generate 接收 list[dict] 消息并异步返回 JSON 文本，
    沿用 Week7 Generate 契约。离线替身和真实 DeepSeekGenerator 都已提供，无需再实现。
    max_context_chars 是证据字符预算，不含问题和 system 消息。
    链输入为一个问题字符串；并行分支分别保留 question、执行 retriever 并转换 hits。
    输出字典由既有 answer_from_context 提供：context/messages/answer/resolved_citations 等。
    学习点 W8D5-3：用 RunnableParallel(question=RunnablePassthrough(),
    hits=retriever | RunnableLambda(documents_to_hits))，再接 RunnableLambda(answer_step)。
    answer_step 已提供；仍由 Week7 负责预算、无证据拒答和引用校验，不重写这些规则。
    """
    build_context([], max_chars=max_context_chars)  # 只验证预算配置，不调用模型。

    async def answer_step(payload):
        context = build_context(payload['hits'], max_chars=max_context_chars)
        report = await answer_from_context(payload['question'], context, generate)
        report['retrieved_hits'] = payload['hits']
        report['retrieved_chunk_ids'] = [item['chunk']['chunk_id'] for item in payload['hits']]
        return report

    # 同一个问题进入两条分支；合并为 question/hits 字典后，再交给回答步骤。
    chain = RunnableParallel(
        question=RunnablePassthrough(),
        hits=retriever | RunnableLambda(documents_to_hits),
    ) | RunnableLambda(answer_step)
    return chain


async def run_rag(question: str, store: InMemoryVectorStore, generate: Generate, *, k: int = 5,
                  filters: dict | None = None, min_score: float | None = None,
                  max_context_chars: int = 6000, output_path: str | Path | None = None) -> dict:
    """已提供：运行链并保存成功/失败报告；不自动重试或把执行错误当成证据不足。

    generate 接收 Week7 两条消息并异步返回 JSON 字符串；可以注入离线假响应。
    失败时保留实际模型原文及异常类型，不记录任意异常正文。语义支持性仍需复核。
    """
    _check_texts([question])
    retriever = make_retriever(store, k=k, filters=filters, min_score=min_score)
    report = dict(execution_status='running', raw_answer=None, messages=[], error=None)

    async def recording_generate(messages):
        report['messages'] = copy.deepcopy(messages)
        report['raw_answer'] = await generate(messages)
        return report['raw_answer']

    try:
        chain = build_rag_chain(retriever, recording_generate, max_context_chars=max_context_chars)
        with tracing_context(enabled=False):
            result = await chain.ainvoke(question)
        report.update(result, execution_status='succeeded')
        return report
    except Exception as error:
        report.update(execution_status='failed', error=dict(type=type(error).__name__))
        raise
    finally:
        if output_path is not None:
            _save_json(Path(output_path), report)


if __name__ == '__main__':
    import argparse
    import asyncio
    import os
    from rag.answering import DeepSeekGenerator
    from rag.embeddings import OllamaEncoder
    from rag.langchain_preparation import DEFAULT_DATASET

    parser = argparse.ArgumentParser(description='Week8 Day5：Ollama 检索 + 既有 DeepSeek 回答层')
    parser.add_argument('--real-llm', action='store_true')
    parser.add_argument('--question', required=True)
    parser.add_argument('--dataset', default=os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))
    parser.add_argument('--store', default=str(DEFAULT_OUTPUT / 'local_model/knowledge_store.json'))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT / 'real_model/cli_report.json'))
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--engine-code')
    parser.add_argument('--version')
    args = parser.parse_args()
    if not args.real_llm:
        parser.error('调用真实回答模型必须显式添加 --real-llm；先通过离线测试')
    filters = {'knowledge_status': 'current'}
    if args.engine_code:
        filters['engine_code'] = args.engine_code
    if args.version:
        filters['version'] = args.version
    embeddings = Week7Embeddings(OllamaEncoder())
    store, action = prepare_store(args.dataset, args.store, embeddings,
                                  embedding_mode='local_model', rebuild=args.rebuild)
    result = asyncio.run(run_rag(args.question, store, DeepSeekGenerator(), k=args.k,
                                 filters=filters, output_path=args.output))
    print(f'索引：{action}；回答状态：{result["answer"]["status"]}；报告：{args.output}')
