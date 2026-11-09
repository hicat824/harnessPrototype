"""Week7 Day7：原生 Python RAG 完整流程演示入口。

首次：清单 → 正文 → 清洗/分块 → 正文向量 → 索引 → 检索 → Context → 回答。
再次：加载已有索引 → 只编码问题 → 检索 → Context → 回答。
默认不访问模型；CLI 必须显式指定 --real-models。学习说明见 rag/week7.md#day7。
"""

import copy
from pathlib import Path
import time

from rag.answering import DeepSeekGenerator, ask_knowledge
from rag.embeddings import OllamaEncoder, _check_texts, _save_json
from rag.preprocessing import load_documents, split_document
from rag.retrieval import (
    KnowledgeIndex, _check_search_options, build_index, load_index, save_index,
)


DEFAULT_DATASET = "/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics"
DEFAULT_OUTPUT = "rag/artifacts/day7/real_model"


def prepare_index(dataset_dir: str | Path, index_path: str | Path, encoder, *,
                  rebuild: bool = False) -> tuple[KnowledgeIndex, str]:
    """取得可用索引，返回 (KnowledgeIndex, "built" 或 "loaded")。

    输入：dataset_dir 包含既有 manifest.json；index_path 为索引 JSON 路径；
    encoder 提供 model、query_instruction、encode_document/encode_query。
    rebuild=False 且文件存在时直接加载，不读取源文档、不编码正文。
    否则从原始资料重建：复用 Day2 默认分块参数 500/50，Day4 批量建库并保存。

    建库分支复用已有 build_index 批量编码并保存索引，查询分支直接加载索引。

    源文件只读。读取、解析、模型和索引校验错误原样抛出；损坏索引不会
    被悄悄重建掩盖。空知识集允许生成空索引，后续自然返回信息不足。
    已有索引不会自动检测原文变化；需要调用方显式 rebuild=True。
    """
    path = Path(index_path)
    if path.exists() and not rebuild:
        return load_index(path), "loaded"

    documents = load_documents(dataset_dir)
    chunks = []
    for document in documents:
        # split_document 内部先做保守清洗，保留原文行号/JSONPath 的含义。
        chunks.extend(split_document(document, max_chars=500, overlap=50))

    # 首次建库时批量编码正文；后续查询通过上面的加载分支复用向量。
    index = build_index(chunks, encoder)
    save_index(index, path)
    return index, "built"


async def run_demo(dataset_dir: str | Path, output_dir: str | Path, question: str,
                   encoder, generate, *, rebuild: bool = False, k: int = 5,
                   filters: dict | None = None, max_context_chars: int = 9000) -> dict:
    """运行一次完整演示，成功后保存 demo_report.json 并返回相同的字典。

    输入：dataset_dir 为原始知识目录；output_dir 保存本日索引和报告；
    question 为非空问题；encoder 是 Day3 编码器或离线替身；generate 是
    异步函数，接收 Day5 的 messages 列表并返回 RagAnswer 格式的 JSON 文本。
    filters 只支持 Day4 的明确元数据过滤；None 表示不过滤，不自动猜版本。
    k 是正整数，max_context_chars 为正整数字符预算，不是 Token。

    输出：Day5 问答报告，包括 context、messages、answer、resolved_citations、
    retrieved_chunk_ids、generation_called；另附 index_mode/index_path、已入库
    文档数、Chunk 数、维数、检索参数和本次耗时。信息不足是合法结果。

    通过 ask_knowledge 串联检索、Context 组装和有引用回答，透传检索参数。
    不要使用评估参考答案生成回答，也不要在查询阶段重新编码全部正文。

    输入错误在建库前拒绝；下游异常继续抛出，不伪装成信息不足或通过。
    报告仅在成功返回后覆盖；失败时旧报告可能仍在，不能当成本次成功产物。
    本函数不创建真实客户端，由调用方显式注入模型或离线替身。
    """
    _check_texts([question])
    _check_search_options(k, filters, None)
    if type(max_context_chars) is not int or max_context_chars <= 0:
        raise ValueError("max_context_chars 必须为正整数")
    started = time.perf_counter()
    output = Path(output_dir)
    index_path = output / "knowledge_index.json"
    index, mode = prepare_index(dataset_dir, index_path, encoder, rebuild=rebuild)

    # 查询复用已有正文向量，并将检索数量、过滤条件和 Context 预算交给问答层。
    result = await ask_knowledge(question, index, encoder, generate, k=k, filters=filters, max_context_chars=max_context_chars)

    report = {
        **result,
        "index_mode": mode,
        "index_path": str(index_path.resolve()),
        "indexed_document_count": len({chunk.doc_id for chunk in index.chunks}),
        "chunk_count": len(index.chunks),
        "vector_dimensions": index.dimensions,
        "k": k,
        "filters": copy.deepcopy(filters if filters is not None else {}),
        "max_context_chars": max_context_chars,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "demo_report.json", report)
    return report


if __name__ == "__main__":
    import argparse
    import asyncio
    import os

    parser = argparse.ArgumentParser(description="Day7：从原始文档到有依据回答的最小 RAG")
    parser.add_argument("--real-models", action="store_true", help="显式允许调用 Ollama 和 DeepSeek")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--question", required=True)
    parser.add_argument("--rebuild", action="store_true", help="原文更新后主动重建，否则复用已有索引")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--context-chars", type=int, default=9000)
    parser.add_argument("--engine-code")
    parser.add_argument("--version")
    parser.add_argument("--knowledge-status", choices=("current", "historical", "all"), default="current")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b", help="仅用于新建/重建索引")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--model", default=os.environ.get("RAG_GENERATION_MODEL", "deepseek-flash"))
    args = parser.parse_args()
    if not args.real_models:
        parser.error("真实演示需要显式添加 --real-models；先完成两个步骤并运行离线验收")

    filters = {}
    if args.knowledge_status != "all":
        filters["knowledge_status"] = args.knowledge_status
    if args.engine_code:
        filters["engine_code"] = args.engine_code
    if args.version:
        filters["version"] = args.version

    index_path = Path(args.output) / "knowledge_index.json"
    if index_path.exists() and not args.rebuild:
        # 加载时沿用索引的编码配置，防止同维度但不同模型混用。
        stored = load_index(index_path)
        encoder = OllamaEncoder(stored.model_name, args.ollama_url,
                                query_instruction=stored.query_instruction)
    else:
        encoder = OllamaEncoder(args.embedding_model, args.ollama_url)
    generate = DeepSeekGenerator(args.model)
    report = asyncio.run(run_demo(
        args.dataset, args.output, args.question, encoder, generate,
        rebuild=args.rebuild, k=args.k, filters=filters, max_context_chars=args.context_chars,
    ))
    print(f"Day7 产物：{Path(args.output).resolve()}；索引：{report['index_mode']}；回答：{report['answer']['status']}")
    print("回答是否受引用支持仍需核对；信息不足也是合法结果。")
