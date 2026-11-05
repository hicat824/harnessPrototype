"""Week7 Day4：保存轻量索引，重启后检索知识片段。

小数据量使用 JSON + 逐个计算余弦相似度，不增加数据库依赖。
三个学习重点：批量建库、候选过滤、候选下标映射回原始 Chunk。
学习说明、测试来源和运行命令见 rag/README_day4.md。
"""

import copy
from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path

from rag.embeddings import OllamaEncoder, _check_texts, _vectors, embed_texts, top_k
from rag.preprocessing import KnowledgeChunk


@dataclass
class KnowledgeIndex:
    """同一编码配置下的 Chunk 与向量；两个列表按下标一一对应。

    model_name 和 query_instruction 是检索需要的编码配置，不保存 HTTP 客户端。
    dimensions 从向量推导；空索引维数为 0，保存后仍能正常加载。
    """

    model_name: str
    query_instruction: str
    chunks: list[KnowledgeChunk]
    vectors: list[list[float]]

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


def _encoder_config(encoder) -> tuple[str, str]:
    """已提供：读取编码器声明的模型名和查询说明，防止仅因维数相同就混用模型。"""
    model = getattr(encoder, "model", None)
    instruction = getattr(encoder, "query_instruction", None)
    if not isinstance(model, str) or not model.strip() or not isinstance(instruction, str):
        raise ValueError("encoder 必须提供非空 model 和字符串 query_instruction")
    return model, instruction


def _validate_index(index: KnowledgeIndex) -> None:
    """已提供：复用 Day3 数值校验，并检查 Chunk 与向量的数量和来源字段。"""
    if not isinstance(index, KnowledgeIndex):
        raise ValueError("index 必须为 KnowledgeIndex")
    if not isinstance(index.model_name, str) or not index.model_name.strip():
        raise ValueError("索引必须记录非空模型名称")
    if not isinstance(index.query_instruction, str) or not isinstance(index.chunks, list):
        raise ValueError("索引查询说明或 chunks 类型不正确")
    vectors = _vectors(index.vectors)
    if len(index.chunks) != len(vectors):
        raise ValueError("Chunk 和向量必须一一对应")
    ids = set()
    for chunk in index.chunks:
        if not isinstance(chunk, KnowledgeChunk):
            raise ValueError("索引中的每个 Chunk 必须为 KnowledgeChunk")
        if (not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip()
                or chunk.chunk_id in ids):
            raise ValueError("chunk_id 必须非空且不能重复")
        ids.add(chunk.chunk_id)
        if not isinstance(chunk.doc_id, str) or not chunk.doc_id.strip():
            raise ValueError("Chunk 必须保留 doc_id")
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            raise ValueError("Chunk 正文不能为空")
        if (not isinstance(chunk.metadata, dict)
                or any(not isinstance(chunk.metadata.get(key), str) or not chunk.metadata[key].strip()
                       for key in ("source", "version"))):
            raise ValueError("Chunk 必须保留非空 source 和 version")


def build_index(chunks: list[KnowledgeChunk], encoder) -> KnowledgeIndex:
    """建库阶段：批量编码一次正文，返回独立复制的索引。

    输入：Chunk 列表及提供 model/query_instruction 和编码入口的 encoder。
    输出：KnowledgeIndex，向量顺序与 Chunk 相同，不修改调用方对象。
    空列表也合法，且不请求模型；非法来源、重复 ID 或非法向量抛 ValueError。

    复用 Day3 的 embed_texts 批量编码正文，默认使用 document 模式。
    本函数只在知识首次入库或重建时调用，不应在每次查询时重跑。
    """
    model, instruction = _encoder_config(encoder)
    if not isinstance(chunks, list) or any(not isinstance(chunk, KnowledgeChunk) for chunk in chunks):
        raise ValueError("chunks 必须为 KnowledgeChunk 列表")
    texts = [chunk.text for chunk in chunks]
    # 建库时一次批量编码正文；后续查询直接复用这些向量。
    vectors = embed_texts(texts, encoder)
    index = KnowledgeIndex(model, instruction, copy.deepcopy(chunks), vectors)
    _validate_index(index)
    return index


def save_index(index: KnowledgeIndex, path: str | Path) -> Path:
    """已提供：把完整索引保存为 JSON，返回文件路径；不调用模型。

    每条 item 同时保存 Chunk 和 vector，避免分开保存时误配顺序。
    文件包含 schema_version=1、模型名、查询说明和维数。父目录自动创建。
    验证/序列化完成后才写文件；同名文件会被覆盖，不修改输入索引。
    """
    _validate_index(index)
    payload = {
        "schema_version": 1,
        "embedding": {"model_name": index.model_name, "query_instruction": index.query_instruction,
                      "dimensions": index.dimensions},
        "items": [{"chunk": asdict(chunk), "vector": vector}
                  for chunk, vector in zip(index.chunks, index.vectors)],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_index(path: str | Path) -> KnowledgeIndex:
    """已提供：只读 JSON 并恢复索引，不访问源文档或 Embedding 服务。

    文件缺失和 JSON 语法错误保留原异常；未知版本、字段缺失、条数/维数错误、
    非法向量或丢失来源抛 ValueError。不能静默跳过损坏条目。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1):
        raise ValueError("不支持的索引 schema_version")
    try:
        config = data["embedding"]
        items = data["items"]
        if not isinstance(config, dict) or not isinstance(items, list):
            raise ValueError("embedding 必须是对象，items 必须是列表")
        index = KnowledgeIndex(
            config["model_name"], config["query_instruction"],
            [KnowledgeChunk(**item["chunk"]) for item in items],
            [item["vector"] for item in items],
        )
        dimensions = config["dimensions"]
    except (KeyError, TypeError) as error:
        raise ValueError("索引缺少必要字段或 Chunk 格式不正确") from error
    _validate_index(index)
    if type(dimensions) is not int or dimensions != index.dimensions:
        raise ValueError("声明的维数与实际向量不一致")
    return index


def import_day3_artifacts(directory: str | Path) -> KnowledgeIndex:
    """已提供：复用 Day3 真实实验向量，不重复编码已有的 64 个 Chunk。

    输入：含 day3_report.json 与 embedded_chunks.json 的目录。
    输出：通过校验的 KnowledgeIndex；必须来自 local_model 模式，且模型配置、
    条数和维数记录齐全。人工向量演示不能被当成真实模型索引。
    """
    directory = Path(directory)
    report = json.loads((directory / "day3_report.json").read_text(encoding="utf-8"))
    items = json.loads((directory / "embedded_chunks.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("mode") != "local_model":
        raise ValueError("导入目录必须是 Day3 local_model 真实实验产物")
    try:
        if not isinstance(items, list):
            raise ValueError("embedded_chunks 必须是列表")
        index = KnowledgeIndex(
            report["model_name"], report["query_instruction"],
            [KnowledgeChunk(**item["chunk"]) for item in items], [item["vector"] for item in items],
        )
        count, dimensions = report["chunk_count"], report["dimensions"]
    except (KeyError, TypeError) as error:
        raise ValueError("Day3 产物缺少编码配置或 Chunk/向量字段") from error
    _validate_index(index)
    if type(count) is not int or count != len(index.chunks):
        raise ValueError("Day3 报告条数与向量文件不一致")
    if type(dimensions) is not int or dimensions != index.dimensions:
        raise ValueError("Day3 报告维数与实际向量不一致")
    return index


def _check_search_options(k: int, filters: dict | None, min_score: float | None) -> dict:
    """已提供：本日只支持三个简单元数据字段的精确相等过滤，条件之间为 AND。"""
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须是正整数")
    if filters is None:
        filters = {}
    if not isinstance(filters, dict) or not set(filters) <= {"engine_code", "knowledge_status", "version"}:
        raise ValueError("filters 只支持 engine_code/knowledge_status/version")
    if any(not isinstance(value, str) or not value.strip() for value in filters.values()):
        raise ValueError("过滤值必须为非空字符串")
    if "knowledge_status" in filters and filters["knowledge_status"] not in ("current", "historical"):
        raise ValueError("knowledge_status 只能是 current/historical")
    if min_score is not None and (
        isinstance(min_score, bool) or not isinstance(min_score, Real)
        or not math.isfinite(min_score) or not -1 <= min_score <= 1
    ):
        raise ValueError("min_score 必须是 [-1,1] 内的有限数，或 None")
    return filters


def _matches_filters(metadata: dict, filters: dict) -> bool:
    """已提供：按字段精确比较，不用子串匹配，不从正文猜测适用流程。"""
    return all(metadata.get(key) == value for key, value in filters.items())


def search_knowledge(query: str, index: KnowledgeIndex, encoder, *, k: int = 3,
                     filters: dict | None = None, min_score: float | None = None) -> list[dict]:
    """查询阶段：过滤候选 → 只编码问题 → Top-K → 返回带来源的片段。

    输入：非空白问题、索引、与索引模型名/查询说明一致的 encoder。
    filters 支持 engine_code/knowledge_status/version 精确匹配，默认不筛选。
    min_score 默认 None；设置后只保留 score>=min_score 的结果，不凑满 K 项。
    输出：[{"score": float, "chunk": KnowledgeChunk 的字典}, ...]。
    chunk 中保留 text/doc_id/chunk_id、metadata.source/version/scope 和定位信息。
    无候选时返回 [] 且不请求模型；有候选时只编码一次 query，绝不重编码正文。
    非法参数、损坏索引、编码配置不匹配或问题向量维数错误均抛 ValueError。

    candidate_indices 保存匹配条件的 Chunk 在原索引中的下标。
    Top-K 给出的候选内下标需再映射回原索引，才能返回正确的原文与来源。
    """
    _check_texts([query])
    filters = _check_search_options(k, filters, min_score)
    _validate_index(index)
    if _encoder_config(encoder) != (index.model_name, index.query_instruction):
        raise ValueError("查询模型或查询说明与索引不一致，请使用原配置或重建索引")
    # 先筛选候选并保留原始下标；循环变量不能覆盖传入的 index 对象。
    candidate_indices = []
    for chunk_index, chunk in enumerate(index.chunks):
        if _matches_filters(chunk.metadata, filters):
            candidate_indices.append(chunk_index)
    if not candidate_indices:
        return []

    # 与建库阶段不同：查询时只生成问题向量，直接复用已保存的正文向量。
    query_vector = embed_texts([query], encoder, kind="query")[0]
    if len(query_vector) != index.dimensions:
        raise ValueError("问题向量维数与索引不一致")
    candidate_vectors = [index.vectors[index_] for index_ in candidate_indices]
    ranked = top_k(query_vector, candidate_vectors, k=k)
    results = []
    for local_index, score in ranked:
        if min_score is not None and score < min_score:
            continue
        # Top-K 的下标属于过滤后的列表，不能直接拿来索引原始 chunks。
        original_index = candidate_indices[local_index]
        results.append({"score": score, "chunk": asdict(index.chunks[original_index])})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Day4 JSON 索引；查询只编码新问题")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="把 Day3 已有真实向量导入索引，不调用模型")
    build.add_argument("--day3-dir", default="rag/artifacts/day3/local_model")
    build.add_argument("--index", default="rag/artifacts/day4/local_model/knowledge_index.json")
    search = commands.add_parser("search", help="加载索引并调用本地 Ollama 编码一个问题")
    search.add_argument("--index", default="rag/artifacts/day4/local_model/knowledge_index.json")
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=3)
    search.add_argument("--engine-code")
    search.add_argument("--knowledge-status", choices=["current", "historical", "all"], default="current")
    search.add_argument("--min-score", type=float)
    search.add_argument("--ollama-url", default="http://localhost:11434")
    search.add_argument("--timeout", type=float, default=120)
    search.add_argument("--output", default="rag/artifacts/day4/local_model/search_results.json")
    args = parser.parse_args()
    if args.command == "build":
        index = import_day3_artifacts(args.day3_dir)
        save_index(index, args.index)
        print(f"索引已保存：{Path(args.index).resolve()}；{len(index.chunks)} 个 Chunk，{index.dimensions} 维")
    else:
        index = load_index(args.index)
        encoder = OllamaEncoder(index.model_name, args.ollama_url, args.timeout, index.query_instruction)
        filters = {}
        if args.engine_code:
            filters["engine_code"] = args.engine_code
        if args.knowledge_status != "all":
            filters["knowledge_status"] = args.knowledge_status
        results = search_knowledge(args.query, index, encoder, k=args.k,
                                   filters=filters, min_score=args.min_score)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": args.query, "model_name": index.model_name,
                                    "filters": filters, "results": results},
                                   ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"检索结果已保存：{path.resolve()}；命中 {len(results)} 项")
