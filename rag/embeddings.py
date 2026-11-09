"""Week7 Day3：Embedding → 余弦相似度 → Top-K → 热力图。

三个学习重点：批量编码、单位向量点积、按相似度取 Top-K。
数值运算使用标准库；画图需要 matplotlib，通过 httpx 调用本地 Ollama 模型。
示例、运行命令和逐测试说明见 rag/week7.md#day3。
"""

from dataclasses import asdict
import json
import math
from numbers import Real
from pathlib import Path

import httpx

from rag.preprocessing import KnowledgeChunk, load_documents, split_document


def _check_texts(texts: list[str]) -> None:
    """已提供：空批次合法，但批次内每项必须是非空白字符串。"""
    if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("texts 必须是非空白字符串列表")


def _vector(values) -> list[float]:
    """已提供：复制一维非零向量，统一拒绝空值、bool、NaN 和无穷值。"""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("向量必须是非空 list/tuple")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("向量分量必须是有限实数，不能是 bool")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("向量不能包含 NaN/Infinity")
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("向量模长必须有限且大于 0，零向量不能计算余弦相似度")
    return vector


def _vectors(rows) -> list[list[float]]:
    """已提供：验证二维向量列表，各行维数必须相同；空列表保持为空。"""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("向量批次必须是 list/tuple")
    result = [_vector(row) for row in rows]
    if result and any(len(row) != len(result[0]) for row in result):
        raise ValueError("同一批向量的维数必须一致")
    return result


def embed_texts(texts: list[str], encoder, kind: str = "document") -> list[list[float]]:
    """把文本批量交给同一个编码器，返回顺序对应的二维向量列表。

    输入：非空白文本列表；encoder 提供 encode_query / encode_document；
    kind 为 query 或 document，区分短问题与知识片段。[] 返回 []，不调用模型。
    输出：每条文本对应一个等维、有限、非零向量；不要求已归一化。
    非法输入或模型输出抛 ValueError，模型自身异常原样抛出，不伪造成功向量。

    一次将整个 texts 批次交给已选好的 encode 方法，保持输入输出顺序。
    只传 texts，不传 Sentence Transformers 专用选项；编码器负责适配后端。
    """
    if kind not in ("query", "document"):
        raise ValueError("kind 只能为 query/document")
    _check_texts(texts)
    if not texts:
        return []
    # 同一个模型编码两类输入；保留 query/document 入口，兼容模型的不同提示前缀。
    encode = encoder.encode_query if kind == "query" else encoder.encode_document
    # 一次批量调用，避免为每条文本单独请求模型。
    raw = encode(texts)
    # 后端统一返回普通列表，后面的数学计算和 JSON 保存不依赖具体模型 SDK。
    vectors = _vectors(raw)
    if len(vectors) != len(texts):
        raise ValueError("编码器返回的向量条数必须与输入文本条数一致")
    return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个同维非零向量的余弦相似度，返回 [-1, 1] 内的 float。

    原理：cos(a,b) = dot(a,b) / (norm(a) * norm(b))。
    先将两个向量单位化，再求对应分量乘积之和，无需重复计算模长。
    例如同向为 1、垂直为 0、反向为 -1；负分不等于业务含义相反。
    """
    left, right = _vector(left), _vector(right)
    if len(left) != len(right):
        raise ValueError("两个向量的维数必须相同")
    # 先各自单位化，避免把向量模长误当成语义相似度。
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    left_unit = [value / left_norm for value in left]
    right_unit = [value / right_norm for value in right]
    # 单位向量的模长均为 1，因此点积就是余弦相似度。
    score = sum(a * b for a, b in zip(left_unit, right_unit))
    return max(-1.0, min(1.0, score))  # 消除浮点误差产生的轻微越界。


def top_k(query_vector: list[float], vectors: list[list[float]], k: int = 3) -> list[tuple[int, float]]:
    """返回 [(原始行号, 相似度), ...]，按分数降序取前 k 项。

    输入：问题向量、候选向量列表、正整数 k（拒绝 bool）。候选为空返回 []；
    k 超过候选数时返回全部。所有向量必须同维，非法输入抛 ValueError。
    分数相同按原始行号升序，确保结果稳定；不修改候选列表。
    先生成分数列表，再排序并切片；返回下标可关联原始 Chunk。
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须是正整数")
    query_vector, vectors = _vector(query_vector), _vectors(vectors)
    scores = [(index, cosine_similarity(query_vector, row)) for index, row in enumerate(vectors)]
    # 负分数使高分排在前面；行号作为第二个键，保证同分时顺序稳定。
    def custom_key(item):
        return (-item[1], item[0])

    # sorted 返回新列表；list.sort() 只原地排序并返回 None，不能拿它做切片。
    ranked = sorted(scores, key=custom_key)
    return ranked[:k]


def pairwise_similarity(vectors: list[list[float]]) -> list[list[float]]:
    """已提供：计算 N×N 的 Chunk 相似度矩阵；行列都沿用输入顺序。"""
    vectors = _vectors(vectors)
    return [[cosine_similarity(left, right) for right in vectors] for left in vectors]


def save_heatmap(matrix: list[list[float]], labels: list[str], output: str | Path,
                 title: str = "Cosine similarity") -> Path:
    """已提供：将已算好的方阵保存为 PNG，无需图形桌面，不调用模型。

    matrix 必须是非空 N×N，值在 [-1,1]；labels 为 N 个不重复的非空字符串。
    图中使用 C01 等短标签，中文原文在相邻 JSON 中查找，避免依赖中文字体。
    N<=12 时标出分数；更大矩阵只显示颜色。固定色标，便于对比不同实验。
    """
    n = len(labels)
    if (not n or len(matrix) != n
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != n):
        raise ValueError("标签必须非空、唯一，并与矩阵行数一致")
    for row in matrix:
        if len(row) != n or any(
            isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not -1 <= value <= 1 for value in row
        ):
            raise ValueError("matrix 必须为 [-1,1] 范围内的有限数值方阵")
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("热力图输出文件应使用 .png 后缀")

    # 延迟导入：仅数值计算的测试不依赖绘图库；使用 Agg 在无桌面环境输出图片。
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    size = max(6, min(22, n * 0.35))
    figure = Figure(figsize=(size, size), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    plot = axes.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes.set_xticks(range(n), labels=labels, rotation=90, fontsize=7)
    axes.set_yticks(range(n), labels=labels, fontsize=7)
    axes.set_title(title)
    if n <= 12:
        for row in range(n):
            for column in range(n):
                axes.text(column, row, f"{matrix[row][column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(plot, ax=axes, label="Cosine similarity")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    figure.clear()
    return output


class OllamaEncoder:
    """已提供：将 Qwen3 Embedding 的本地 HTTP 接口适配成两个编码入口。

    输入：已拉取的模型名称、Ollama 服务地址、每次请求的超时秒数。
    encode_document 原样编码正文；encode_query 按 Qwen 约定添加检索说明。
    两个方法均返回与 texts 顺序对应的 list[list[float]]；空批次不请求服务。
    HTTP 错误、连接失败和超时原样抛出；JSON 或向量格式错误抛 ValueError。
    不下载模型、不重试，也不在失败时返回人工向量。服务未启动时会明确失败。
    """

    def __init__(self, model: str = "qwen3-embedding:0.6b",
                 base_url: str = "http://localhost:11434", timeout: float = 120,
                 query_instruction: str = "Given a user question, retrieve relevant passages that answer the question"):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model 必须为非空模型名称")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url 必须为 HTTP/HTTPS 服务地址")
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("base_url 必须为 HTTP/HTTPS 服务地址")
        if (isinstance(timeout, bool) or not isinstance(timeout, Real)
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout 必须是大于 0 的有限秒数")
        if not isinstance(query_instruction, str):
            raise ValueError("query_instruction 必须为字符串；空串表示不加说明")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.query_instruction = query_instruction

    def encode_document(self, texts: list[str]) -> list[list[float]]:
        """正文直接编码；不把查询提示词附加到知识片段中。"""
        _check_texts(texts)
        return self._embed(texts)

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        """先校验原始问题，再生成带任务说明的新列表，保留调用方原文。"""
        _check_texts(texts)
        if self.query_instruction:
            texts = [f"Instruct: {self.query_instruction}\nQuery:{text}" for text in texts]
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """一次请求编码整个批次；请求和响应都不依赖向量维数常量。"""
        if not texts:
            return []
        # Client 在每次批量请求后关闭；忽略代理环境变量，直接连接所配服务。
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, trust_env=False) as client:
            response = client.post("/api/embed", json={
                "model": self.model, "input": texts,
                "truncate": False,  # 超过模型窗口时显式报错，避免正文不完整却不知情。
            })
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or "error" in data:
            raise ValueError("Ollama 响应必须是包含 embeddings 的成功 JSON 对象")
        vectors = _vectors(data.get("embeddings"))
        if len(vectors) != len(texts):
            raise ValueError("Ollama 返回的向量条数与输入文本条数不一致")
        return vectors


def collect_chunks(dataset_dir: str | Path) -> list[KnowledgeChunk]:
    """已提供：复用 Day2 原流程生成 Chunk，不依赖随时可能清理的 pytest 临时文件。"""
    return [chunk for doc in load_documents(dataset_dir)
            for chunk in split_document(doc, max_chars=500, overlap=50)]


def _save_json(path: Path, data) -> None:
    """已提供：写中文可读 JSON，禁止把 NaN/Infinity 写成无效数值。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(chunks: list[KnowledgeChunk], encoder, query: str, output_dir: str | Path,
                   *, model_name: str, mode: str, k: int = 3) -> dict:
    """已提供：编排本日完整实验，返回报告，并在固定目录输出 JSON 与 PNG。

    输入：非空且 ID 唯一的 Chunk 列表、编码器、问题、输出目录、模型标识。
    mode 明确写 offline_fixture 或 local_model，区分人工向量与真实推理。
    输出：报告包含维数、来源、Top-K；图片是 Chunk×Chunk，不包含问题向量。
    相同目录会覆盖同名文件；不创建向量索引，不调用生成模型。
    """
    if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("chunks 必须非空且 chunk_id 唯一")
    if mode not in ("offline_fixture", "local_model"):
        raise ValueError("mode 只能为 offline_fixture/local_model")
    vectors = embed_texts([chunk.text for chunk in chunks], encoder, kind="document")
    query_vector = embed_texts([query], encoder, kind="query")[0]
    hits = top_k(query_vector, vectors, k=k)
    matrix = pairwise_similarity(vectors)
    output_dir = Path(output_dir)
    labels = [f"C{index + 1:02d}" for index in range(len(chunks))]
    save_heatmap(matrix, labels, output_dir / "similarity_heatmap.png",
                 title=f"Chunk cosine similarity ({mode})")
    # 每个向量都绑定来源，而不是只保存一堆无法对应原文的浮点数组。
    _save_json(output_dir / "embedded_chunks.json", [
        {"label": label, "chunk": asdict(chunk), "vector": vector}
        for label, chunk, vector in zip(labels, chunks, vectors)
    ])
    _save_json(output_dir / "similarity_matrix.json", {"labels": labels, "matrix": matrix})
    report = {
        "mode": mode, "model_name": model_name, "chunk_count": len(chunks),
        "dimensions": len(vectors[0]), "query": query, "query_vector": query_vector,
        "top_k": [{"label": labels[index], "score": score, "chunk": asdict(chunks[index])}
                  for index, score in hits],
        "scope": "向量计算与来源回溯；语义排序质量需要结合原文人工判断",
    }
    if isinstance(encoder, OllamaEncoder):
        # 记录实际编码说明和服务地址，便于解释同一个问题为何得到不同排序。
        report["ollama_url"] = encoder.base_url
        report["query_instruction"] = encoder.query_instruction
    _save_json(output_dir / "day3_report.json", report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Day3 Ollama 本地模型实验；不会自动下载模型")
    parser.add_argument("--model", default="qwen3-embedding:0.6b", help="Ollama 已拉取的模型名称")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama 服务根地址")
    parser.add_argument("--timeout", type=float, default=120, help="每次批量编码请求的超时秒数")
    parser.add_argument("--dataset", default="/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics")
    parser.add_argument("--output", default="rag/artifacts/day3/local_model")
    parser.add_argument("--query", default="任务查不到时应该先检查什么？")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    encoder = OllamaEncoder(model=args.model, base_url=args.ollama_url, timeout=args.timeout)
    report = run_experiment(collect_chunks(args.dataset), encoder, args.query, args.output,
                            model_name=encoder.model, mode="local_model", k=args.k)
    print(f"Day3 产物：{Path(args.output).resolve()}；{report['chunk_count']} 个 Chunk，{report['dimensions']} 维")
