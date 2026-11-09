"""Week7 Day3：固定向量回归、本地模型实验与相似度热力图。

默认执行框架和 HTTP Mock 自检，并生成一张人工向量热力图；
设置 RUN_WEEK7_DAY3=1 启用算法测试。真实模型还需 RUN_WEEK7_DAY3_MODEL=1。
数据来源、运行方式和每项验收边界见 rag/week7.md#day3。
"""

import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import httpx
import pytest

from rag.embeddings import (
    OllamaEncoder, collect_chunks, cosine_similarity, embed_texts,
    pairwise_similarity, run_experiment, save_heatmap, top_k,
)
from rag.preprocessing import KnowledgeDocument, make_chunk


DAY3 = pytest.mark.skipif(os.environ.get("RUN_WEEK7_DAY3") != "1",
                          reason="完整离线验收需显式设置 RUN_WEEK7_DAY3=1")
REAL_MODEL = pytest.mark.skipif(
    os.environ.get("RUN_WEEK7_DAY3") != "1" or os.environ.get("RUN_WEEK7_DAY3_MODEL") != "1",
    reason="真实本地模型需显式启用 RUN_WEEK7_DAY3=1 和 RUN_WEEK7_DAY3_MODEL=1",
)
OUTPUT_ROOT = Path(os.environ.get(
    "RAG_DAY3_OUTPUT", str(Path(__file__).resolve().parents[1] / "rag/artifacts/day3"),
))
SAMPLE_ROOT = Path(os.environ.get(
    "WEEK7_RAG_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics",
))

# 这是人工定义的二维向量及手算答案，不来自 Embedding 模型。
VECTORS = [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]]
EXPECTED_MATRIX = [[1, 0.8, 0, -1], [0.8, 1, 0.6, -0.8],
                   [0, 0.6, 1, 0], [-1, -0.8, 0, 1]]


class FixedEncoder:
    """测试替身：记录模型调用，返回固定向量；不分析文本语义。"""

    def __init__(self, documents=None, query=None):
        self.documents = copy.deepcopy(VECTORS if documents is None else documents)
        self.query = copy.deepcopy([[1, 0]] if query is None else query)
        self.calls = []

    def encode_document(self, texts):
        self.calls.append(("document", list(texts)))
        return copy.deepcopy(self.documents)

    def encode_query(self, texts):
        self.calls.append(("query", list(texts)))
        return copy.deepcopy(self.query)


def _save(path, data):
    """测试侧导出人工答案与标签说明，不要求学习者再实现文件保存。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def test_provided_heatmap_preview():
    """独立查看绘图效果，不依赖余弦计算；数据明确标为人工向量。"""
    output = OUTPUT_ROOT / "preview"
    labels = ["unit_x", "near_x", "unit_y", "opposite_x"]
    image = save_heatmap(EXPECTED_MATRIX, labels, output / "similarity_heatmap.png",
                         title="Hand-written vectors (not model embeddings)")
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.stat().st_size > 1000
    _save(output / "preview_data.json", {
        "mode": "hand_written_vectors", "labels": labels,
        "vectors": VECTORS, "matrix": EXPECTED_MATRIX,
        "notice": "仅验证绘图；未调用模型，也未调用余弦计算函数。",
    })
    print(f"Day3 人工向量预览：{image}")


@pytest.mark.parametrize("matrix,labels,filename", [
    ([], [], "x.png"), ([[1]], ["A", "B"], "x.png"),
    ([[1, 0], [0, 1]], ["A", "A"], "x.png"),
    ([[math.nan]], ["A"], "x.png"), ([[1.01]], ["A"], "x.png"),
    ([[1]], ["A"], "x.txt"),
])
def test_provided_heatmap_rejects_bad_input(tmp_path, matrix, labels, filename):
    """不依赖模型：矩阵、标签和输出文件格式不合法时直接报错。"""
    with pytest.raises(ValueError):
        save_heatmap(matrix, labels, tmp_path / filename)


@pytest.mark.parametrize("options", [
    {"model": ""}, {"base_url": ""}, {"base_url": "file:///tmp/model"},
    {"timeout": 0}, {"timeout": True}, {"timeout": math.nan},
    {"query_instruction": None},
])
def test_provided_ollama_rejects_bad_config(options):
    """初始化只检查配置，不访问 Ollama；无效参数立即报错。"""
    with pytest.raises(ValueError):
        OllamaEncoder(**options)


@DAY3
@pytest.mark.parametrize("kind", ["document", "query"])
def test_embed_texts_calls_correct_encoder(kind):
    """检查问题/正文编码入口、批量调用参数及输入输出的顺序。"""
    encoder = FixedEncoder(documents=[[1, 2], [2, 1]], query=[[1, 2], [2, 1]])
    texts = ["第一条中文", "第二条中文"]
    assert embed_texts(texts, encoder, kind=kind) == [[1.0, 2.0], [2.0, 1.0]]
    assert encoder.calls == [(kind, texts)]
    assert texts == ["第一条中文", "第二条中文"]


@DAY3
def test_embed_texts_empty_and_bad_inputs():
    """输入校验不应造成任何模型调用；空批次可以直接返回。"""
    encoder = FixedEncoder()
    assert embed_texts([], encoder) == []
    for texts, kind in [("字符串不是列表", "document"), ([" "], "document"),
                        ([12], "document"), (["正文"], "wrong")]:
        with pytest.raises(ValueError):
            embed_texts(texts, encoder, kind=kind)
    assert encoder.calls == []


@DAY3
@pytest.mark.parametrize("rows", [
    [], [[1, 0], [0, 1]], [[0, 0]], [[math.nan, 1]],
    [[math.inf, 1]], [[True, 1]], [["1", 0]],
])
def test_embed_texts_rejects_bad_model_output(rows):
    """模型返回条数错误或不可计算的向量时，不能继续生成相似度与热力图。"""
    with pytest.raises(ValueError):
        embed_texts(["只有一条正文"], FixedEncoder(documents=rows))


@DAY3
def test_embed_texts_rejects_mixed_dimensions_and_propagates_error():
    """向量维数不齐以及模型本身报错需要保留为明确失败。"""
    with pytest.raises(ValueError):
        embed_texts(["A", "B"], FixedEncoder(documents=[[1, 0], [1, 0, 0]]))

    class BrokenEncoder:
        def encode_document(self, texts):
            raise RuntimeError("模拟本地推理失败")

    with pytest.raises(RuntimeError, match="模拟本地推理失败"):
        embed_texts(["A"], BrokenEncoder())


@DAY3
@pytest.mark.parametrize("left,right,expected", [
    ([1, 0], [1, 0], 1), ([1, 0], [0, 1], 0),
    ([1, 0], [-1, 0], -1), ([3, 4], [6, 8], 1),
    ([1, 0], [4, 3], 0.8),
])
def test_cosine_similarity_matches_hand_calculation(left, right, expected):
    """手算答案检查方向、模长归一化和负分，不使用模型分数作自证。"""
    before = copy.deepcopy((left, right))
    assert cosine_similarity(left, right) == pytest.approx(expected, abs=1e-12)
    assert (left, right) == before


@DAY3
@pytest.mark.parametrize("left,right", [
    ([], [1]), ([0, 0], [1, 0]), ([1], [1, 0]),
    ([math.nan], [1]), ([math.inf], [1]), ([True], [1]),
])
def test_cosine_similarity_rejects_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


@DAY3
def test_pairwise_matrix_is_symmetric_and_matches_expected():
    matrix = pairwise_similarity(VECTORS)
    for row in range(4):
        assert matrix[row] == pytest.approx(EXPECTED_MATRIX[row], abs=1e-12)
        assert matrix[row][row] == pytest.approx(1)
        for column in range(4):
            assert matrix[row][column] == pytest.approx(matrix[column][row])
    assert pairwise_similarity([]) == []


@DAY3
def test_top_k_orders_scores_and_breaks_ties_by_index():
    """相同方向但长度不同应同分；同分按原始下标稳定排序。"""
    vectors = [[1, 0], [-1, 0], [2, 0], [0, 1]]
    before = copy.deepcopy(vectors)
    assert top_k([1, 0], vectors, k=3) == [(0, 1.0), (2, 1.0), (3, 0.0)]
    assert [index for index, _ in top_k([1, 0], vectors, k=20)] == [0, 2, 3, 1]
    assert top_k([1, 0], [], k=2) == []
    assert vectors == before


@DAY3
@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_top_k_rejects_invalid_budget(k):
    with pytest.raises(ValueError):
        top_k([1, 0], VECTORS, k=k)


@DAY3
def test_offline_pipeline_exports_sources_scores_and_heatmap():
    """固定向量走完整流程；只证明计算与来源关联正确，不证明模型理解中文。"""
    texts = ["任务不存在，请核对编号。", "未找到任务，需要检查编号。",
             "没有访问权限，请申请权限。", "engineCode 需要精确匹配。"]
    chunks = []
    for index, text in enumerate(texts):
        doc = KnowledgeDocument(f"DEMO{index + 1}", text, {
            "format": "text", "source": f"synthetic/demo_{index + 1}.txt", "version": "1.0",
            "scope": ["learning"], "updated_at": "2026-11-05",
        })
        chunks.append(make_chunk(doc, text, 0, len(text), 0, []))
    before = copy.deepcopy(chunks)
    encoder = FixedEncoder()
    output = OUTPUT_ROOT / "offline_fixture"
    report = run_experiment(chunks, encoder, "查不到任务怎么办？", output,
                            model_name="hand_written_vectors", mode="offline_fixture", k=2)
    assert report["mode"] == "offline_fixture" and report["dimensions"] == 2
    assert [row["label"] for row in report["top_k"]] == ["C01", "C02"]
    assert [row["score"] for row in report["top_k"]] == pytest.approx([1, 0.8])
    assert report["top_k"][0]["chunk"] == asdict(chunks[0])
    saved = json.loads((output / "embedded_chunks.json").read_text(encoding="utf-8"))
    assert len(saved) == 4 and saved[0]["vector"] == [1, 0]
    assert saved[0]["chunk"]["metadata"]["source"] == "synthetic/demo_1.txt"
    assert (output / "similarity_heatmap.png").read_bytes().startswith(b"\x89PNG")
    assert [call[0] for call in encoder.calls] == ["document", "query"]
    assert chunks == before
    print(f"Day3 固定向量流程产物：{output}")


@REAL_MODEL
def test_local_model_experiment():
    """真实 Ollama 模型编码 Day2 Chunk；不硬编码语义分数或排名。"""
    # 使用用户已拉取的模型；本测试不会执行 pull，只调用配置的服务地址。
    encoder = OllamaEncoder(
        model=os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        timeout=float(os.environ.get("OLLAMA_EMBED_TIMEOUT", "120")),
    )
    chunks = collect_chunks(SAMPLE_ROOT)
    before = copy.deepcopy(chunks)
    output = OUTPUT_ROOT / "local_model"
    report = run_experiment(chunks, encoder, "任务查不到时应该先检查什么？", output,
                            model_name=encoder.model, mode="local_model", k=3)
    assert report["mode"] == "local_model" and report["chunk_count"] == len(chunks)
    assert report["model_name"] == encoder.model and report["ollama_url"] == encoder.base_url
    assert report["query_instruction"] == encoder.query_instruction
    assert report["dimensions"] > 1 and len(report["top_k"]) == 3
    assert len({row["chunk"]["chunk_id"] for row in report["top_k"]}) == 3
    assert all(-1 <= row["score"] <= 1 for row in report["top_k"])
    matrix = json.loads((output / "similarity_matrix.json").read_text(encoding="utf-8"))["matrix"]
    for index, row in enumerate(matrix):
        assert len(row) == len(chunks) and row[index] == pytest.approx(1, abs=1e-6)
        for column, value in enumerate(row):
            assert value == pytest.approx(matrix[column][index], abs=1e-6)
    assert chunks == before
    print(f"Day3 真实本地模型产物：{output}；请结合原文人工检查 Top-K 的语义相关性")


@pytest.fixture
def mock_ollama(monkeypatch):
    """已提供：仅替换 HTTP 传输；序列化、状态码检查和响应解析走实际代码。"""
    original_client = httpx.Client

    def install(handler):
        def create_client(**options):
            return original_client(**options, transport=httpx.MockTransport(handler))

        monkeypatch.setattr("rag.embeddings.httpx.Client", create_client)

    return install


@pytest.mark.parametrize("kind", ["document", "query"])
def test_provided_ollama_sends_batch_and_formats_query(mock_ollama, kind):
    """批次顺序不变，只有问题加说明；请求显式禁止静默截断。"""
    texts = ["任务查不到怎么办？", "权限不足怎么办？"]
    before = list(texts)
    instruction = "Retrieve relevant knowledge for the question"
    expected = texts if kind == "document" else [
        f"Instruct: {instruction}\nQuery:{text}" for text in texts
    ]
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "POST" and str(request.url) == "http://localhost:11434/api/embed"
        assert json.loads(request.content) == {
            "model": "qwen3-embedding:0.6b", "input": expected, "truncate": False,
        }
        assert request.extensions["timeout"]["read"] == 17
        return httpx.Response(200, json={"embeddings": [[1, 2], [3, 4]]})

    mock_ollama(handler)
    encoder = OllamaEncoder(timeout=17, query_instruction=instruction)
    encode = encoder.encode_query if kind == "query" else encoder.encode_document
    assert encode(texts) == [[1.0, 2.0], [3.0, 4.0]]
    assert len(calls) == 1 and texts == before


def test_provided_ollama_empty_and_invalid_texts_never_request(mock_ollama):
    """先检查原始问题，不能因为加了说明而把空白问题误认为有效文本。"""
    def handler(request):
        pytest.fail("无效文本或空批次不应请求服务")

    mock_ollama(handler)
    encoder = OllamaEncoder()
    for encode in (encoder.encode_query, encoder.encode_document):
        assert encode([]) == []
        for texts in ([" "], [123], "不是列表"):
            with pytest.raises(ValueError):
                encode(texts)


@pytest.mark.parametrize("status", [404, 500])
def test_provided_ollama_propagates_http_error(mock_ollama, status):
    """模型不存在或服务失败时保留 HTTP 错误，不下载、不重试、不返回假向量。"""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "模拟模型不存在或服务失败"})

    mock_ollama(handler)
    with pytest.raises(httpx.HTTPStatusError) as error:
        OllamaEncoder().encode_document(["正文"])
    assert error.value.response.status_code == status and len(calls) == 1


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_provided_ollama_propagates_transport_error(mock_ollama, error_type):
    """服务未启动或推理超时时，调用方能区分原始异常类型。"""
    calls = []

    def handler(request):
        calls.append(request)
        raise error_type("模拟连接失败或超时", request=request)

    mock_ollama(handler)
    with pytest.raises(error_type):
        OllamaEncoder().encode_document(["正文"])
    assert len(calls) == 1


@pytest.mark.parametrize("body", [
    [], {}, {"error": "失败"}, {"embeddings": []},
    {"embeddings": [[1, 0], [0, 1]]}, {"embeddings": [[0, 0]]},
    {"embeddings": [[True, 1]]}, {"embeddings": [["NaN", 1]]},
])
def test_provided_ollama_rejects_bad_response(mock_ollama, body):
    """HTTP 200 不等于有效向量：仍检查对象结构、条数、数值与非零模长。"""
    mock_ollama(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ValueError):
        OllamaEncoder().encode_document(["一条正文"])


def test_provided_ollama_rejects_invalid_json(mock_ollama):
    """服务返回非 JSON 正文时保留解析失败，不当成成功响应。"""
    mock_ollama(lambda request: httpx.Response(200, text="not json"))
    with pytest.raises(json.JSONDecodeError):
        OllamaEncoder().encode_document(["正文"])
