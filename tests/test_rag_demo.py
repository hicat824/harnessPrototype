"""Week7 Day7：原始知识建库、重复查询和证据边界的周验收。

默认检查已提供的加载分支、参数和命令行保护。
完成两个 TODO 后用 RUN_WEEK7_DAY7=1 启用完整离线测试。
真实重建和回答额外要求 RUN_WEEK7_DAY7_LLM=1，本文件不会下载模型。
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from rag.answering import DeepSeekGenerator
from rag.demo import DEFAULT_DATASET, prepare_index, run_demo
from rag.embeddings import OllamaEncoder, _save_json
from rag.preprocessing import clean_text, load_documents, split_document
from rag.retrieval import build_index, load_index, save_index


REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("RAG_DAY7_OUTPUT", str(REPO / "rag/artifacts/day7")))
TEXT = "NOT_FOUND 表示本次查询没有找到指定任务，不等于任务执行失败，也不等于无访问权限。"
DAY7 = pytest.mark.skipif(os.environ.get("RUN_WEEK7_DAY7") != "1",
                          reason="完成 rag/demo.py 两个步骤后设置 RUN_WEEK7_DAY7=1")
REAL = pytest.mark.skipif(
    os.environ.get("RUN_WEEK7_DAY7") != "1" or os.environ.get("RUN_WEEK7_DAY7_LLM") != "1",
    reason="真实周验收需额外设置 RUN_WEEK7_DAY7_LLM=1",
)


class FixedEncoder:
    """只验证调用顺序和数据对应关系，不代表模型真实的语义检索能力。"""

    model = "fixed/week7-day7"
    query_instruction = "offline retrieval"

    def __init__(self, query_only=False):
        self.documents, self.queries = [], []
        self.query_only = query_only

    def encode_document(self, texts):
        if self.query_only:
            raise AssertionError("复用索引时不能重新编码正文")
        self.documents.append(list(texts))
        return [[1, 0] if TEXT in text else [0, 1] for text in texts]

    def encode_query(self, texts):
        self.queries.append(list(texts))
        return [[1, 0] for _ in texts]


def _answer():
    return {"status": "answered", "claims": [{"text": "未找到任务不等于任务执行失败。",
            "citations": [{"source_id": "S1", "quote": TEXT}]}], "missing_information": ""}


class FixedGenerator:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    async def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        return json.dumps(_answer(), ensure_ascii=False)


@pytest.fixture
def dataset(tmp_path):
    """生成两个独立合成文件：带 BOM/CRLF 的当前 Markdown 和历史 JSON。"""
    root = tmp_path / "source"
    (root / "knowledge").mkdir(parents=True)
    markdown = "\ufeff# 查询说明\r\n\r\n## 错误解释\r\n" + TEXT + "\r\n"
    (root / "knowledge/current.md").write_bytes(markdown.encode("utf-8"))
    payload = {"notice": "不属于正文的说明", "sections": [
        {"heading": "历史规则", "text": "这是 1.0 历史规则。"},
        {"heading": "使用边界", "text": "不得用历史规则解释当前任务。"},
    ]}
    (root / "knowledge/history.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    common = {"scope": ["learning"], "updated_at": "2026-11-06", "engine_code": "import_std"}
    records = [
        {**common, "doc_id": "D1", "title": "查询说明", "source": "knowledge/current.md",
         "version": "2.0", "knowledge_status": "current", "format": "markdown"},
        {**common, "doc_id": "D2", "title": "历史边界", "source": "knowledge/history.json",
         "version": "1.0", "knowledge_status": "historical", "format": "json"},
    ]
    (root / "manifest.json").write_text(json.dumps({"document_count": 2, "documents": records},
                                                 ensure_ascii=False), encoding="utf-8")
    return root


def _seed_index(dataset, output):
    """通过已完成的 Day2/Day4 接口准备索引，只供测试加载和查询分支。"""
    chunks = [chunk for document in load_documents(dataset) for chunk in split_document(document)]
    index = build_index(chunks, FixedEncoder())
    save_index(index, output / "knowledge_index.json")
    return index


def test_provided_reuses_saved_index_without_source_files(dataset, tmp_path):
    output = tmp_path / "demo"
    expected = _seed_index(dataset, output)
    actual, mode = prepare_index(tmp_path / "absent_source", output / "knowledge_index.json",
                                 FixedEncoder(query_only=True))
    assert mode == "loaded" and actual == expected


def test_provided_corrupt_cache_is_not_silently_rebuilt(dataset, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    encoder = FixedEncoder()
    with pytest.raises(json.JSONDecodeError):
        prepare_index(dataset, path, encoder)
    assert path.read_text(encoding="utf-8") == "{not json"
    assert encoder.documents == encoder.queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"question": " "}, {"k": 0}, {"k": True}, {"filters": {"owner": "someone"}},
    {"max_context_chars": 0}, {"max_context_chars": True},
])
async def test_provided_invalid_input_fails_before_building(tmp_path, options):
    encoder, generator = FixedEncoder(), FixedGenerator()
    values = {"question": "NOT_FOUND 是失败吗？", **options}
    with pytest.raises(ValueError):
        await run_demo(tmp_path / "absent", tmp_path / "output", encoder=encoder,
                       generate=generator, **values)
    assert not (tmp_path / "output").exists()
    assert encoder.documents == encoder.queries == generator.calls == []


def test_provided_cli_requires_explicit_real_flag():
    result = subprocess.run([sys.executable, "-B", "-m", "rag.demo", "--question", "NOT_FOUND 是失败吗？"],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--real-models" in result.stderr


@DAY7
def test_build_from_raw_documents_preserves_cleaning_and_locations(dataset, tmp_path):
    before = {str(p): p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
    encoder = FixedEncoder()
    index, mode = prepare_index(dataset, tmp_path / "index.json", encoder)
    assert mode == "built" and len(index.chunks) == 4
    assert encoder.documents == [[chunk.text for chunk in index.chunks]]
    assert encoder.queries == [] and index.dimensions == 2
    sources = {(d.doc_id, d.segment_id): d for d in load_documents(dataset)}
    for chunk in index.chunks:
        source = sources[chunk.doc_id, chunk.segment_id]
        assert chunk.text == clean_text(source.text)[chunk.start_char:chunk.end_char]
        assert "\r" not in chunk.text and not chunk.text.startswith("\ufeff")
        assert chunk.metadata["source"] == source.metadata["source"]
        if chunk.doc_id == "D2":
            assert chunk.start_line is None and chunk.metadata["json_path"].endswith(".text")
    assert {str(p): p.read_bytes() for p in dataset.rglob("*") if p.is_file()} == before
    assert load_index(tmp_path / "index.json") == index


@DAY7
@pytest.mark.asyncio
async def test_full_demo_builds_retrieves_and_answers(dataset, tmp_path):
    encoder, generator = FixedEncoder(), FixedGenerator()
    filters = {"knowledge_status": "current"}
    report = await run_demo(dataset, tmp_path / "demo", "NOT_FOUND 是失败吗？", encoder, generator,
                            filters=filters)
    assert report["index_mode"] == "built" and report["answer"]["status"] == "answered"
    assert report["indexed_document_count"] == 2 and report["chunk_count"] == 4
    assert len(encoder.documents) == len(encoder.queries) == len(generator.calls) == 1
    assert report["messages"] == generator.calls[0]
    citation = report["resolved_citations"][0]
    assert citation["quote"] == TEXT and citation["chunk"]["metadata"]["source"] == "knowledge/current.md"
    assert citation["chunk"]["start_line"] == 3
    assert all(s["chunk"]["metadata"]["knowledge_status"] == "current" for s in report["context"]["sources"])
    saved = json.loads((tmp_path / "demo/demo_report.json").read_text(encoding="utf-8"))
    assert saved == report and report["elapsed_seconds"] >= 0
    report["filters"]["knowledge_status"] = "historical"
    assert filters == {"knowledge_status": "current"}


@DAY7
@pytest.mark.asyncio
async def test_repeat_query_loads_index_without_reembedding(dataset, tmp_path):
    output = tmp_path / "demo"
    first = await run_demo(dataset, output, "NOT_FOUND 是失败吗？", FixedEncoder(), FixedGenerator(),
                           filters={"knowledge_status": "current"})
    before = (output / "knowledge_index.json").read_bytes()
    encoder = FixedEncoder(query_only=True)
    second = await run_demo(tmp_path / "absent_source", output, "NOT_FOUND 表示什么？", encoder,
                            FixedGenerator(), filters={"knowledge_status": "current"})
    assert first["index_mode"] == "built" and second["index_mode"] == "loaded"
    assert not encoder.documents and len(encoder.queries) == 1
    assert (output / "knowledge_index.json").read_bytes() == before


@DAY7
def test_rebuild_explicitly_refreshes_changed_documents(dataset, tmp_path):
    path = tmp_path / "index.json"
    encoder = FixedEncoder()
    first, _ = prepare_index(dataset, path, encoder)
    source = dataset / "knowledge/current.md"
    source.write_bytes(source.read_bytes() + "新增的当前规则。\n".encode("utf-8"))
    cached, mode = prepare_index(dataset, path, encoder)
    assert mode == "loaded" and cached == first and len(encoder.documents) == 1
    rebuilt, mode = prepare_index(dataset, path, encoder, rebuild=True)
    assert mode == "built" and len(encoder.documents) == 2
    assert any("新增的当前规则" in chunk.text for chunk in rebuilt.chunks)


@DAY7
@pytest.mark.asyncio
async def test_no_evidence_skips_query_embedding_and_generation(dataset, tmp_path):
    output = tmp_path / "demo"
    _seed_index(dataset, output)
    encoder, generator = FixedEncoder(query_only=True), FixedGenerator()
    report = await run_demo(dataset, output, "3.0 新增哪些规则？", encoder, generator,
                            filters={"version": "3.0"})
    assert report["answer"]["status"] == "insufficient" and report["answer"]["claims"] == []
    assert report["retrieved_chunk_ids"] == [] and not report["generation_called"]
    assert encoder.documents == encoder.queries == generator.calls == []


@DAY7
@pytest.mark.asyncio
async def test_generation_error_does_not_overwrite_previous_report(dataset, tmp_path):
    output = tmp_path / "demo"
    _seed_index(dataset, output)
    path = output / "demo_report.json"
    path.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="模拟模型故障"):
        await run_demo(dataset, output, "NOT_FOUND 是失败吗？", FixedEncoder(),
                       FixedGenerator(error=RuntimeError("模拟模型故障")),
                       filters={"knowledge_status": "current"})
    assert path.read_text(encoding="utf-8") == '{"previous": true}'


@DAY7
def test_fresh_process_loads_index_and_answers(dataset, tmp_path):
    output = tmp_path / "demo"
    _seed_index(dataset, output)
    # 子进程不继承内存索引；它必须只靠文件重启，再编码问题和生成回答。
    code = textwrap.dedent(f'''
        import asyncio, json, sys
        from rag.demo import run_demo
        class QueryOnlyEncoder:
            model = "fixed/week7-day7"
            query_instruction = "offline retrieval"
            def encode_document(self, texts):
                raise AssertionError("新进程不能重编码正文")
            def encode_query(self, texts):
                return [[1, 0] for _ in texts]
        async def generate(messages):
            return json.dumps({repr(_answer())}, ensure_ascii=False)
        report = asyncio.run(run_demo(sys.argv[1], sys.argv[2], "NOT_FOUND 是失败吗？",
            QueryOnlyEncoder(), generate, filters={{"knowledge_status": "current"}}))
        assert report["index_mode"] == "loaded"
        assert report["answer"]["status"] == "answered"
    ''')
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path / "absent_source"), str(output)],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    saved = json.loads((output / "demo_report.json").read_text(encoding="utf-8"))
    assert saved["resolved_citations"][0]["chunk"]["doc_id"] == "D1"


@DAY7
@pytest.mark.asyncio
async def test_existing_corpus_full_offline_pipeline():
    """18 份原始资料使用固定向量和手写回答，证明接线，不证明模型语义质量。"""
    source = Path(os.environ.get("WEEK7_RAG_DATASET", DEFAULT_DATASET))
    encoder = FixedEncoder()
    report = await run_demo(source, OUTPUT / "offline_fixture", "NOT_FOUND 是失败吗？", encoder,
                            FixedGenerator(), rebuild=True, filters={"knowledge_status": "current"})
    assert report["indexed_document_count"] == 18 and report["chunk_count"] == 64
    assert report["vector_dimensions"] == 2 and len(encoder.documents) == 1
    assert report["answer"]["status"] == "answered"
    citation = report["resolved_citations"][0]["chunk"]
    assert citation["doc_id"] == "KB012" and citation["metadata"]["json_path"] == "$.sections[0].text"
    print(f"Day7 离线完整链路产物：{OUTPUT / 'offline_fixture'}")


@REAL
@pytest.mark.asyncio
async def test_real_full_build_then_reuse_for_realtime_question():
    """真实完整建库，再用新编码器复用索引；语义结论仍需阅读报告核对。"""
    assert os.environ.get("OPENAI_API_KEY"), "需要项目既有 DeepSeek OPENAI_API_KEY"
    source = Path(os.environ.get("WEEK7_RAG_DATASET", DEFAULT_DATASET))
    output = OUTPUT / "real_model"
    encoder = OllamaEncoder(base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    generate = DeepSeekGenerator(os.environ.get("RAG_GENERATION_MODEL", "deepseek-flash"))
    first = await run_demo(source, output, "NOT_FOUND 是不是表示任务执行失败？", encoder, generate,
                           rebuild=True, filters={"knowledge_status": "current"})
    _save_json(output / "grounded_report.json", first)
    assert first["index_mode"] == "built" and first["indexed_document_count"] == 18
    assert first["chunk_count"] == 64 and first["answer"]["status"] == "answered"
    assert first["resolved_citations"]
    assert any(s["chunk"]["doc_id"] == "KB012" for s in first["context"]["sources"])

    class QueryOnlyOllama(OllamaEncoder):
        def encode_document(self, texts):
            raise AssertionError("第二次运行必须复用已有正文向量")

    stored = load_index(output / "knowledge_index.json")
    query_encoder = QueryOnlyOllama(stored.model_name, encoder.base_url,
                                   query_instruction=stored.query_instruction)
    second = await run_demo(source, output, "任务 12345 现在已经完成了吗？请给出当前状态。",
                            query_encoder, generate, filters={"knowledge_status": "current"})
    _save_json(output / "realtime_report.json", second)
    assert second["index_mode"] == "loaded" and second["answer"]["status"] == "insufficient"
    assert second["answer"]["claims"] == [] and second["answer"]["missing_information"].strip()
    assert second["generation_called"], "本例需观察模型如何依据静态证据拒答，不能以空 Context 代替"
    print(f"Day7 真实周验收产物：{output}；请核对回答是否受证据支持及实时拒答理由")
