"""Week7 Day4：JSON 索引、过滤检索和新进程重启验收。

默认运行保存/加载等自检；RUN_WEEK7_DAY4=1 启用完整离线流程测试。
真实 Ollama 查询另外需要 RUN_WEEK7_DAY4_MODEL=1，不会在普通测试中运行。
每个测试的数据、检查点和输出见 rag/week7.md#day4。
"""

import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rag.preprocessing import KnowledgeDocument, make_chunk
from rag.retrieval import KnowledgeIndex, build_index, import_day3_artifacts, load_index, save_index, search_knowledge


DAY4 = pytest.mark.skipif(os.environ.get("RUN_WEEK7_DAY4") != "1",
                          reason="完整离线验收需显式设置 RUN_WEEK7_DAY4=1")
REAL_MODEL = pytest.mark.skipif(
    os.environ.get("RUN_WEEK7_DAY4") != "1" or os.environ.get("RUN_WEEK7_DAY4_MODEL") != "1",
    reason="真实查询需显式启用 RUN_WEEK7_DAY4=1 和 RUN_WEEK7_DAY4_MODEL=1",
)
REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("RAG_DAY4_OUTPUT", str(REPO / "rag/artifacts/day4")))


class FixedEncoder:
    """测试替身：预设分数关系，记录正文/问题调用；不具备语义理解能力。"""

    model = "fixed/day4"
    query_instruction = "offline query instruction"

    def __init__(self):
        self.calls = []
        self.query_vectors = [[1, 0]]

    def encode_document(self, texts):
        self.calls.append(("document", list(texts)))
        return [[1, 0], [1, 0], [0.8, 0.6], [0, 1]]

    def encode_query(self, texts):
        self.calls.append(("query", list(texts)))
        return copy.deepcopy(self.query_vectors)


@pytest.fixture
def index():
    """前两条分数更高但流程/版本不适用，用来检查是否先过滤再 Top-K。"""
    rows = [
        ("export_std", "current", "1.0", "导出流程约定。"),
        ("import_std", "historical", "1.0", "旧版本的任务查询规则。"),
        ("import_std", "current", "2.0", "任务查不到时请核对编号。"),
        ("import_std", "current", "1.0", "其他导入流程参数。"),
    ]
    chunks = []
    for number, (engine, status, version, text) in enumerate(rows):
        metadata = {"format": "json" if number == 2 else "text",
                    "source": f"knowledge/D{number}.json" if number == 2 else f"knowledge/D{number}.txt",
                    "version": version, "scope": ["learning"], "updated_at": "2026-11-05",
                    "engine_code": engine, "knowledge_status": status}
        if number == 2:
            metadata["json_path"] = "$.sections[0].text"
        doc = KnowledgeDocument(f"D{number}", text, metadata,
                                segment_id="json-0" if number == 2 else "body")
        chunks.append(make_chunk(doc, text, 0, len(text), 0, ["任务查询"]))
    return KnowledgeIndex(FixedEncoder.model, FixedEncoder.query_instruction, chunks,
                          [[1, 0], [1, 0], [0.8, 0.6], [0, 1]])


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _day3_files(index, directory):
    """构造 Day3 格式的合成文件；local_model 字段仅为测试格式，不是真实推理。"""
    directory.mkdir(parents=True)
    _write_json(directory / "day3_report.json", {
        "mode": "local_model", "model_name": index.model_name,
        "query_instruction": index.query_instruction,
        "chunk_count": len(index.chunks), "dimensions": index.dimensions,
    })
    _write_json(directory / "embedded_chunks.json", [
        {"chunk": asdict(chunk), "vector": vector}
        for chunk, vector in zip(index.chunks, index.vectors)
    ])


def test_provided_index_round_trip_preserves_sources(index, tmp_path):
    before = copy.deepcopy(index)
    path = save_index(index, tmp_path / "nested/knowledge_index.json")
    restored = load_index(path)
    assert restored == before and index == before
    assert "任务查不到" in path.read_text(encoding="utf-8")
    assert restored.chunks[2].metadata["json_path"] == "$.sections[0].text"
    assert restored.chunks[2].start_line is None
    restored.chunks[0].metadata["scope"].append("changed")
    assert index.chunks[0].metadata["scope"] == ["learning"]


@pytest.mark.parametrize("problem", [
    "version", "bool_version", "dimension", "bool_dimension", "missing_vector",
    "zero_vector", "mixed_dimensions", "duplicate_id", "missing_source", "missing_chunk_field",
])
def test_provided_load_rejects_corrupt_index(index, tmp_path, problem):
    path = save_index(index, tmp_path / "index.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if problem == "version":
        data["schema_version"] = 2
    elif problem == "bool_version":
        data["schema_version"] = True
    elif problem == "dimension":
        data["embedding"]["dimensions"] = 3
    elif problem == "bool_dimension":
        data["embedding"]["dimensions"] = True
    elif problem == "missing_vector":
        del data["items"][0]["vector"]
    elif problem == "zero_vector":
        data["items"][0]["vector"] = [0, 0]
    elif problem == "mixed_dimensions":
        data["items"][0]["vector"] = [1, 0, 0]
    elif problem == "duplicate_id":
        data["items"][1]["chunk"]["chunk_id"] = data["items"][0]["chunk"]["chunk_id"]
    elif problem == "missing_source":
        del data["items"][0]["chunk"]["metadata"]["source"]
    else:
        del data["items"][0]["chunk"]["text"]
    _write_json(path, data)
    with pytest.raises(ValueError):
        load_index(path)


def test_provided_load_reports_missing_file_and_invalid_json(tmp_path):
    path = tmp_path / "index.json"
    with pytest.raises(FileNotFoundError):
        load_index(path)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_index(path)


def test_provided_save_validates_before_overwriting(index, tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("保留已有文件", encoding="utf-8")
    index.vectors.pop()
    with pytest.raises(ValueError):
        save_index(index, path)
    assert path.read_text(encoding="utf-8") == "保留已有文件"


def test_provided_import_day3_vectors_without_reembedding(index, tmp_path):
    directory = tmp_path / "day3"
    _day3_files(index, directory)
    assert import_day3_artifacts(directory) == index


@pytest.mark.parametrize("problem", ["mode", "count", "dimensions", "instruction"])
def test_provided_import_rejects_inconsistent_day3_files(index, tmp_path, problem):
    directory = tmp_path / "day3"
    _day3_files(index, directory)
    path = directory / "day3_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if problem == "mode":
        report["mode"] = "offline_fixture"
    elif problem == "count":
        report["chunk_count"] = 5
    elif problem == "dimensions":
        report["dimensions"] = 3
    else:
        del report["query_instruction"]
    _write_json(path, report)
    with pytest.raises(ValueError):
        import_day3_artifacts(directory)


def test_provided_build_command_imports_day3_files(index, tmp_path):
    directory = tmp_path / "day3"
    _day3_files(index, directory)
    target = tmp_path / "knowledge_index.json"
    result = subprocess.run([sys.executable, "-B", "-m", "rag.retrieval", "build",
                             "--day3-dir", str(directory), "--index", str(target)],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert load_index(target) == index


@DAY4
def test_build_encodes_documents_once_and_copies_chunks(index):
    encoder = FixedEncoder()
    built = build_index(index.chunks, encoder)
    assert built == index
    assert encoder.calls == [("document", [chunk.text for chunk in index.chunks])]
    built.chunks[0].metadata["scope"].append("changed")
    assert index.chunks[0].metadata["scope"] == ["learning"]


@DAY4
def test_build_empty_index_never_requests_model(tmp_path):
    encoder = FixedEncoder()
    empty = build_index([], encoder)
    assert empty.chunks == empty.vectors == [] and empty.dimensions == 0
    assert load_index(save_index(empty, tmp_path / "empty.json")) == empty
    assert encoder.calls == []


@DAY4
def test_search_returns_scores_original_chunks_and_sources(index):
    encoder = FixedEncoder()
    results = search_knowledge("查询规则？", index, encoder, k=3)
    assert [row["chunk"]["doc_id"] for row in results] == ["D0", "D1", "D2"]
    assert [row["score"] for row in results] == pytest.approx([1, 1, 0.8])
    assert results[2]["chunk"] == asdict(index.chunks[2])
    assert encoder.calls == [("query", ["查询规则？"])]
    results[0]["chunk"]["metadata"]["scope"].append("changed")
    assert index.chunks[0].metadata["scope"] == ["learning"]


@DAY4
@pytest.mark.parametrize("filters,expected", [
    ({"engine_code": "import_std", "knowledge_status": "current"}, ["D2", "D3"]),
    ({"engine_code": "import_std"}, ["D1", "D2", "D3"]),
    ({"knowledge_status": "historical"}, ["D1"]),
    ({"version": "2.0"}, ["D2"]),
])
def test_search_filters_before_ranking_and_maps_original_indices(index, filters, expected):
    encoder = FixedEncoder()
    results = search_knowledge("查询规则？", index, encoder, filters=filters, k=20)
    assert [row["chunk"]["doc_id"] for row in results] == expected
    if filters == {"engine_code": "import_std", "knowledge_status": "current"}:
        # 前两条高分记录被排除；候选下标 0 应映射回原下标 2，而不是误取 D0。
        best = search_knowledge("查询规则？", index, encoder, filters=filters, k=1)
        assert best[0]["chunk"]["doc_id"] == "D2" and best[0]["score"] == pytest.approx(0.8)


@DAY4
def test_no_candidates_and_empty_index_never_request_model(index):
    encoder = FixedEncoder()
    assert search_knowledge("问题", index, encoder, filters={"engine_code": "import"}) == []
    index.chunks, index.vectors = [], []
    assert search_knowledge("问题", index, encoder) == []
    assert encoder.calls == []


@DAY4
@pytest.mark.parametrize("threshold,count", [(-1, 2), (0.8, 1), (0.9, 0), (1, 0)])
def test_score_threshold_does_not_pad_results(index, threshold, count):
    results = search_knowledge("问题", index, FixedEncoder(), k=3, min_score=threshold,
                               filters={"engine_code": "import_std", "knowledge_status": "current"})
    assert len(results) == count
    assert all(row["score"] >= threshold for row in results)


@DAY4
@pytest.mark.parametrize("options", [
    {"k": 0}, {"k": True}, {"filters": {"scope": "learning"}},
    {"filters": {"engine_code": " "}}, {"filters": {"knowledge_status": "wrong"}},
    {"min_score": math.nan}, {"min_score": 1.1}, {"min_score": True},
])
def test_invalid_search_options_never_request_model(index, options):
    encoder = FixedEncoder()
    with pytest.raises(ValueError):
        search_knowledge("问题", index, encoder, **options)
    assert encoder.calls == []


@DAY4
def test_model_configuration_and_query_dimensions_must_match(index):
    for field in ("model", "query_instruction"):
        encoder = FixedEncoder()
        setattr(encoder, field, "different")
        with pytest.raises(ValueError, match="不一致"):
            search_knowledge("问题", index, encoder)
        assert encoder.calls == []
    encoder = FixedEncoder()
    encoder.query_vectors = [[1, 0, 0]]
    with pytest.raises(ValueError, match="维数"):
        search_knowledge("问题", index, encoder)


@DAY4
def test_offline_build_save_reload_and_search(index):
    encoder = FixedEncoder()
    built = build_index(index.chunks, encoder)
    output = OUTPUT_ROOT / "offline_fixture"
    path = save_index(built, output / "knowledge_index.json")
    restored = load_index(path)
    results = search_knowledge("任务查不到怎么办？", restored, encoder, k=1,
                               filters={"engine_code": "import_std", "knowledge_status": "current"})
    assert results[0]["chunk"]["doc_id"] == "D2"
    assert [kind for kind, _ in encoder.calls] == ["document", "query"]
    _write_json(output / "search_results.json", results)
    _write_json(output / "day4_report.json", {
        "mode": "offline_fixture", "passed": True, "chunk_count": len(restored.chunks),
        "dimensions": restored.dimensions, "document_batches": 1, "query_batches": 1,
    })
    print(f"Day4 离线流程产物：{output}")


@DAY4
def test_fresh_process_loads_index_and_searches_without_reembedding(index, tmp_path):
    path = save_index(index, tmp_path / "knowledge_index.json")
    output = tmp_path / "fresh_process_results.json"
    # 新解释器只读取索引文件；原有 Python 对象、缓存和模型调用记录均不存在。
    script = '''
import json, sys
from pathlib import Path
from rag.retrieval import load_index, search_knowledge
index = load_index(sys.argv[1])
class QueryOnlyEncoder:
    model = index.model_name
    query_instruction = index.query_instruction
    def encode_query(self, texts):
        return [[1, 0]]
    def encode_document(self, texts):
        raise AssertionError("查询不得重新编码正文")
results = search_knowledge("问题", index, QueryOnlyEncoder(), k=1,
    filters={"engine_code": "import_std", "knowledge_status": "current"})
Path(sys.argv[2]).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(path), str(output)],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows[0]["chunk"] == asdict(index.chunks[2])
    assert rows[0]["score"] == pytest.approx(0.8)
    print(f"Day4 新进程验证产物：{tmp_path}")


@REAL_MODEL
def test_real_ollama_query_uses_saved_day3_vectors():
    directory = Path(os.environ.get("RAG_DAY3_MODEL_OUTPUT", str(REPO / "rag/artifacts/day3/local_model")))
    assert (directory / "day3_report.json").is_file(), "请先保留 Day3 真实模型实验产物"
    output = OUTPUT_ROOT / "local_model"
    path = save_index(import_day3_artifacts(directory), output / "knowledge_index.json")
    timeout = float(os.environ.get("OLLAMA_EMBED_TIMEOUT", "120"))
    target = output / "search_results.json"
    # CLI 在新进程中加载索引，再让 Ollama 只编码一个问题；无正文重新编码。
    result = subprocess.run([
        sys.executable, "-B", "-m", "rag.retrieval", "search", "--index", str(path),
        "--query", "任务查不到时应该先检查什么？", "--knowledge-status", "current",
        "--ollama-url", os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        "--timeout", str(timeout), "--output", str(target),
    ], cwd=REPO, capture_output=True, text=True, timeout=timeout + 30)
    assert result.returncode == 0, result.stderr
    report = json.loads(target.read_text(encoding="utf-8"))
    rows = report["results"]
    assert len(rows) == 3
    assert all(row["chunk"]["metadata"]["knowledge_status"] == "current" for row in rows)
    assert all(-1 <= row["score"] <= 1 for row in rows)
    assert all(row["chunk"]["text"] and row["chunk"]["metadata"]["source"] for row in rows)
    assert [row["score"] for row in rows] == sorted((row["score"] for row in rows), reverse=True)
    print(f"Day4 真实查询产物：{output}；语义相关性请结合原文人工检查")
