"""Week7 Day2：三个练习函数的离线验收。

数据来源：tmp_path 中的合成小文档，以及 Week7 已准备的 18 份知识文件。
JSON/定位辅助代码默认自检；完整处理流程需 RUN_WEEK7_DAY2=1 才执行。
不请求 LLM、不下载模型；每个测试的来源、流程与产物见 rag/week7.md#day2。
"""

import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from rag.preprocessing import (
    KnowledgeDocument, clean_text, load_documents, load_json_sections,
    make_chunk, read_manifest, split_document,
)


DAY2 = pytest.mark.skipif(
    os.environ.get("RUN_WEEK7_DAY2") != "1",
    reason="Week7 Day2 完整验收需设置 RUN_WEEK7_DAY2=1；说明见 rag/week7.md#day2",
)
SAMPLE_ROOT = Path(os.environ.get(
    "WEEK7_RAG_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics",
))


@pytest.fixture
def small_dataset(tmp_path):
    """三份源文件→四个正文单元；清单顺序故意不同于 doc_id 排序。"""
    root = tmp_path / "dataset"
    (root / "knowledge").mkdir(parents=True)
    md = "\ufeff# 导入说明\r\n\r\n## 参数\r\n流程 import_std，版本 2.0。\r\n"
    txt = "查询返回 3 条，但 total=12。\r\n来源不能丢失。\r\n"
    business_json = {
        "title": "查询 FAQ", "notice": "NOTICE_NOT_KNOWLEDGE",
        "sections": [
            {"heading": "错误解释", "text": "NOT_FOUND 不等于任务执行失败。"},
            {"heading": "处理方式", "text": "请核对 task_id，不能编造标识符。"},
        ],
    }
    contents = {"knowledge/z.md": md, "knowledge/a.txt": txt,
                "knowledge/m.json": json.dumps(business_json, ensure_ascii=False, indent=2)}
    records = []
    for doc_id, source, fmt, engine in [
        ("D-Z", "knowledge/z.md", "markdown", "import_std"),
        ("D-A", "knowledge/a.txt", "text", "export_std"),
        ("D-M", "knowledge/m.json", "json", None),
    ]:
        (root / source).write_bytes(contents[source].encode("utf-8"))
        records.append({
            "doc_id": doc_id, "title": "查询 FAQ" if fmt == "json" else doc_id,
            "source": source, "format": fmt, "version": "2.0", "scope": ["learning"],
            "updated_at": "2026-11-04", "engine_code": engine,
            "logical_id": doc_id.lower(), "knowledge_status": "current",
            "custom": {"keep_this_field": True},
        })
    (root / "manifest.json").write_text(json.dumps({
        "document_count": len(records), "documents": records,
    }, ensure_ascii=False), encoding="utf-8")
    (root / "README.md").write_text("AUXILIARY_NOT_KNOWLEDGE", encoding="utf-8")
    return root, records, contents, business_json


def _text_document(text, fmt="text"):
    """直接构造分块输入，避免分块测试依赖加载函数是否写对。"""
    return KnowledgeDocument("TEST", text, {
        "source": "knowledge/example.md" if fmt == "markdown" else "knowledge/example.txt",
        "format": fmt, "title": "样例", "version": "2.0", "scope": ["learning"],
        "updated_at": "2026-11-04", "engine_code": "import_std",
    })


def _canonical(text):
    """验收定位时使用约定的文本坐标系，不依赖待实现的 clean_text。"""
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _assert_chunks(document, chunks, max_chars):
    """检查来源、区间和内容覆盖，不复写标题识别或分块算法。"""
    text = _canonical(document.text)
    if not text.strip():
        assert chunks == []
        return
    assert chunks, "非空正文不能被整体丢弃"
    covered = set()
    starts = []
    for number, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{document.doc_id}:{document.segment_id}:{number:04d}"
        assert chunk.doc_id == document.doc_id and chunk.segment_id == document.segment_id
        assert chunk.metadata == document.metadata
        assert chunk.metadata is not document.metadata
        assert type(chunk.start_char) is int and type(chunk.end_char) is int
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)
        assert chunk.text == text[chunk.start_char:chunk.end_char]
        assert chunk.text.strip() and len(chunk.text) <= max_chars
        if document.metadata["format"] == "json":
            assert chunk.start_line is None and chunk.end_line is None
            assert "json_path" in chunk.metadata
        else:
            assert chunk.start_line == text.count("\n", 0, chunk.start_char) + 1
            assert chunk.end_line == text.count("\n", 0, chunk.end_char - 1) + 1
        starts.append(chunk.start_char)
        covered.update(range(chunk.start_char, chunk.end_char))
    assert starts == sorted(starts)
    assert all(index in covered for index, char in enumerate(text) if not char.isspace()), \
        "除允许跳过的空白外，不能丢失正文"


def test_provided_json_adapter_keeps_source_paths(small_dataset):
    """独立自检 JSON 适配器，不依赖加载、清洗和分块函数。"""
    root, records, contents, data = small_dataset
    assert read_manifest(root) == records
    docs = load_json_sections(records[2], contents["knowledge/m.json"])
    assert len(docs) == 2
    for index, doc in enumerate(docs):
        assert doc.doc_id == "D-M" and doc.segment_id == f"json-{index}"
        assert doc.text == data["sections"][index]["text"]
        assert doc.section_path == ["查询 FAQ", data["sections"][index]["heading"]]
        assert doc.metadata["json_path"] == f"$.sections[{index}].text"
        assert "NOTICE_NOT_KNOWLEDGE" not in doc.text


def test_provided_chunk_locator_and_metadata_isolation():
    """自检已提供的定位辅助代码：原行号与清洗字符偏移不是同一种坐标。"""
    doc = _text_document("\ufeff# 标题\r\n  中文\r\n", "markdown")
    text = _canonical(doc.text)
    start = text.index("  中文")
    first = make_chunk(doc, text, start, len(text), 0, ["标题"])
    second = make_chunk(doc, text, start, len(text), 1, ["标题"])
    assert (first.start_line, first.end_line) == (2, 2)
    assert first.text == "  中文\n"
    first.metadata["scope"].append("changed")
    first.section_path.append("changed")
    assert doc.metadata["scope"] == second.metadata["scope"] == ["learning"]
    assert second.section_path == ["标题"] and doc.section_path == []
    json_doc = KnowledgeDocument("J", "正文", {
        "format": "json", "source": "knowledge/j.json", "json_path": "$.sections[0].text",
    }, segment_id="json-0")
    chunk = make_chunk(json_doc, "正文", 0, 2, 0, [])
    assert chunk.start_line is None and chunk.end_line is None


@DAY2
@pytest.mark.parametrize("extensionless", [False, True])
def test_load_documents_uses_manifest_and_preserves_raw_text(small_dataset, extensionless):
    root, records, contents, data = small_dataset
    if extensionless:
        # 正文格式由清单决定；换成无后缀文件后，也必须完整加载这一条记录。
        source = "knowledge/plain_document"
        (root / "knowledge/z.md").rename(root / source)
        contents[source] = contents.pop("knowledge/z.md")
        records[0]["source"] = source
        _save_json(root / "manifest.json", {"document_count": len(records), "documents": records})
    before = {source: (root / source).read_bytes() for source in contents}
    docs = load_documents(root)
    assert [(doc.doc_id, doc.segment_id) for doc in docs] == [
        ("D-Z", "body"), ("D-A", "body"), ("D-M", "json-0"), ("D-M", "json-1"),
    ]
    assert docs[0].text == contents[records[0]["source"]]
    assert docs[1].text == contents["knowledge/a.txt"]
    for doc in docs:
        record = next(record for record in records if record["doc_id"] == doc.doc_id)
        assert all(doc.metadata[key] == value for key, value in record.items() if key != "doc_id")
        assert "AUXILIARY_NOT_KNOWLEDGE" not in doc.text
        assert "NOTICE_NOT_KNOWLEDGE" not in doc.text
    assert docs[2].text == data["sections"][0]["text"]
    assert docs[3].metadata["json_path"] == "$.sections[1].text"
    assert before == {source: (root / source).read_bytes() for source in contents}


@DAY2
@pytest.mark.parametrize("problem,error_type", [
    ("missing", FileNotFoundError), ("invalid_utf8", UnicodeDecodeError),
    ("bad_json", json.JSONDecodeError),
])
def test_load_documents_does_not_silently_skip_errors(small_dataset, problem, error_type):
    root, _, _, _ = small_dataset
    if problem == "missing":
        (root / "knowledge/z.md").unlink()
    elif problem == "invalid_utf8":
        (root / "knowledge/z.md").write_bytes(b"\xff\xfe")
    else:
        (root / "knowledge/m.json").write_text("{broken json", encoding="utf-8")
    with pytest.raises(error_type):
        load_documents(root)


@DAY2
@pytest.mark.parametrize("raw,expected", [
    ("", ""), ("\ufeff# 标题\r\n\r\n正文\r尾行", "# 标题\n\n正文\n尾行"),
    ("\n  保留缩进  \n\n\n", "\n  保留缩进  \n\n\n"),
    ("# H  \r\n```python\r\n  x = '中文'\r\n```\r\n", "# H  \n```python\n  x = '中文'\n```\n"),
    ("import_std / v2.0 / 标点。\ufeff保留中间字符", "import_std / v2.0 / 标点。\ufeff保留中间字符"),
])
def test_clean_text_is_conservative_and_idempotent(raw, expected):
    assert clean_text(raw) == expected
    assert clean_text(expected) == expected


@DAY2
@pytest.mark.parametrize("max_chars,overlap", [
    (0, 0), (-1, 0), (True, 0), ("10", 0), (5, -1),
    (5, 5), (5, 6), (5, True), (5, 0.5),
])
def test_split_rejects_invalid_budget(max_chars, overlap):
    with pytest.raises(ValueError):
        split_document(_text_document(""), max_chars=max_chars, overlap=overlap)


@DAY2
@pytest.mark.parametrize("overlap,expected_ranges", [
    (0, [(0, 4), (4, 8), (8, 10)]),
    (1, [(0, 4), (3, 7), (6, 10)]),
    (3, [(index, index + 4) for index in range(7)]),
])
def test_split_plain_text_has_bounded_windows(overlap, expected_ranges):
    doc = _text_document("甲乙丙丁戊己庚辛壬癸")
    chunks = split_document(doc, max_chars=4, overlap=overlap)
    _assert_chunks(doc, chunks, 4)
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == expected_ranges
    assert all(chunk.section_path == [] for chunk in chunks)


@DAY2
def test_split_markdown_respects_heading_boundaries():
    text = "导言\n# 总览\n这是导入说明。\n## 参数\n流程编码必须保留。\n### 细节\n版本也必须保留。\n# 导出\n这里是其他流程。\n"
    doc = _text_document(text, "markdown")
    before = copy.deepcopy(doc)
    markers = [0, text.index("# 总览"), text.index("## 参数"), text.index("### 细节"), text.index("# 导出"), len(text)]
    paths = [[], ["总览"], ["总览", "参数"], ["总览", "参数", "细节"], ["导出"]]
    chunks = split_document(doc, max_chars=18, overlap=3)
    _assert_chunks(doc, chunks, 18)
    for chunk in chunks:
        section = next(index for index in range(len(paths)) if markers[index] <= chunk.start_char < markers[index + 1])
        assert chunk.end_char <= markers[section + 1], "不能跨标题分块或重叠"
        assert chunk.section_path == paths[section]
    assert doc == before
    assert split_document(doc, max_chars=18, overlap=3) == chunks, "相同输入必须得到稳定 ID 和区间"


@DAY2
@pytest.mark.parametrize("fence,inner_fence", [
    ("```", ""), ("~~~", ""), ("````", "```\n"), ("```", "~~~\n"),
])
def test_markdown_headings_inside_code_fences_are_not_sections(fence, inner_fence):
    # 较短或不同字符的围栏不能结束代码块，否则内部的 ## 会被误当标题。
    text = f"# Guide\n{fence}python\n{inner_fence}## not a heading\n  x = 1\n{fence}\n## Real\n说明。\n"
    doc = _text_document(text, "markdown")
    chunks = split_document(doc, max_chars=1000, overlap=0)
    _assert_chunks(doc, chunks, 1000)
    assert len(chunks) == 2
    assert chunks[0].section_path == ["Guide"]
    assert "## not a heading\n  x = 1" in chunks[0].text
    assert chunks[1].section_path == ["Guide", "Real"]


@DAY2
@pytest.mark.parametrize("text", ["", " \t\n\n", "\ufeff\r\n \r\n"])
def test_split_empty_documents_returns_no_chunks(text):
    assert split_document(_text_document(text), max_chars=10, overlap=0) == []


@DAY2
def test_text_and_json_use_their_own_structure(small_dataset):
    _, records, contents, _ = small_dataset
    txt = _text_document("# 这不是 Markdown 标题\n保留为一个普通文本节。")
    text_chunks = split_document(txt, max_chars=1000, overlap=0)
    assert len(text_chunks) == 1 and text_chunks[0].section_path == []
    docs = load_json_sections(records[2], contents["knowledge/m.json"])
    all_chunks = []
    for doc in docs:
        chunks = split_document(doc, max_chars=8, overlap=2)
        _assert_chunks(doc, chunks, 8)
        assert all(chunk.section_path == doc.section_path for chunk in chunks)
        all_chunks.extend(chunks)
    assert len({chunk.chunk_id for chunk in all_chunks}) == len(all_chunks)
    all_chunks[0].metadata["scope"].append("changed")
    assert docs[0].metadata["scope"] == all_chunks[1].metadata["scope"] == ["learning"]


def _save_json(path, value):
    """测试侧保存可读产物，不要求你再实现文件导出函数。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@DAY2
def test_week7_samples_end_to_end(tmp_path):
    """18 份实际学习文件 → 20 个正文单元 → 清洗与分块，保留回溯证据。"""
    assert (SAMPLE_ROOT / "manifest.json").is_file(), "请准备 Week7 样例或设置 WEEK7_RAG_DATASET"
    records = read_manifest(SAMPLE_ROOT)
    before = {record["source"]: hashlib.sha256((SAMPLE_ROOT / record["source"]).read_bytes()).hexdigest()
              for record in records}
    docs = load_documents(SAMPLE_ROOT)
    assert len(records) == 18 and len(docs) == 20
    assert {doc.doc_id for doc in docs} == {record["doc_id"] for record in records}
    assert len({(doc.doc_id, doc.segment_id) for doc in docs}) == 20
    assert {doc.doc_id for doc in docs if doc.metadata["knowledge_status"] == "historical"} == {"KB003", "KB009"}
    cleaned_docs, all_chunks = [], []
    for doc in docs:
        before_doc = copy.deepcopy(doc)
        cleaned_docs.append(replace(doc, text=clean_text(doc.text), metadata=copy.deepcopy(doc.metadata)))
        chunks = split_document(doc, max_chars=500, overlap=50)
        _assert_chunks(doc, chunks, 500)
        assert doc == before_doc
        all_chunks.extend(chunks)
    assert len({chunk.chunk_id for chunk in all_chunks}) == len(all_chunks)
    assert before == {record["source"]: hashlib.sha256((SAMPLE_ROOT / record["source"]).read_bytes()).hexdigest()
                      for record in records}
    _save_json(tmp_path / "loaded_documents.json", [asdict(doc) for doc in docs])
    _save_json(tmp_path / "cleaned_documents.json", [asdict(doc) for doc in cleaned_docs])
    _save_json(tmp_path / "chunks.json", [asdict(chunk) for chunk in all_chunks])
    _save_json(tmp_path / "day2_report.json", {
        "passed": True, "source_file_count": len(records), "document_unit_count": len(docs),
        "chunk_count": len(all_chunks), "max_chars": 500, "overlap": 50,
        "scope": "文档加载、清洗、分块和回溯；不包含向量检索或生成回答",
    })
    print(f"Week7 Day2 离线产物：{tmp_path}")
