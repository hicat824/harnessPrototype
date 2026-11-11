"""Week8 Day4：Document/Loader/Splitter/Embeddings 分层验收。

默认检查辅助代码；完整离线流程需设置 RUN_WEEK8_DAY4=1。
真实本地推理另需 RUN_WEEK8_DAY4_MODEL=1；本日不使用 DeepSeek 或回答型 LLM。
合成小数据在本文件生成；完整语料复用 WEEK7_RAG_DATASET 指向的 Week7 清单。
每项测试的数据、流程和产物见 langchain_learning/week8.md#day4。
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

pytest.importorskip('langchain_text_splitters', reason='请使用 Week8 独立环境并安装 requirements.txt')

import httpx
from langchain_core.documents import Document

from rag.embeddings import OllamaEncoder, embed_texts
from rag.langchain_preparation import (
    DEFAULT_DATASET, DEFAULT_OUTPUT, QUERY, ManifestLoader, Week7Embeddings,
    chunk_document, document_record, make_splitter, run_preparation, split_documents, to_document,
)
from rag.preprocessing import KnowledgeDocument, clean_text, load_documents, make_chunk, split_document


DAY4 = pytest.mark.skipif(os.environ.get('RUN_WEEK8_DAY4') != '1', reason='完整流程需设置 RUN_WEEK8_DAY4=1')
REAL = pytest.mark.skipif(
    os.environ.get('RUN_WEEK8_DAY4') != '1' or os.environ.get('RUN_WEEK8_DAY4_MODEL') != '1',
    reason='本地真实推理还需 RUN_WEEK8_DAY4_MODEL=1',
)
OUTPUT = Path(os.environ.get('WEEK8_DAY4_OUTPUT', str(DEFAULT_OUTPUT)))
DATASET = Path(os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))


@pytest.fixture
def small_dataset(tmp_path):
    root = tmp_path / 'dataset'
    (root / 'knowledge').mkdir(parents=True)
    contents = {
        'knowledge/z.md': '\ufeff# 导入\r\n正文保留 import_std 与缩进。\r\n## 参数\r\n' +
                          '参数解释不能跨流程。' * 8 + '\r\n```python\r\n# 不是标题\r\n```\r\n## 状态\r\n只解释状态。\r\n',
        'knowledge/a.txt': 'total=12，但本次仅返回 3 条。\r\n来源与版本应保留。\r\n',
        'knowledge/m.json': json.dumps({'notice': 'NOTICE_NOT_KNOWLEDGE', 'sections': [
            {'heading': '错误解释', 'text': 'NOT_FOUND 不等于任务失败。' * 6},
            {'heading': '处理方式', 'text': '核对 task_id；不编造实时状态。'},
        ]}, ensure_ascii=False),
    }
    records = []
    for doc_id, source, fmt in [('D-Z', 'knowledge/z.md', 'markdown'),
                                ('D-A', 'knowledge/a.txt', 'text'), ('D-M', 'knowledge/m.json', 'json')]:
        (root / source).write_bytes(contents[source].encode('utf-8'))
        records.append(dict(doc_id=doc_id, title='查询 FAQ' if fmt == 'json' else doc_id,
                            source=source, format=fmt, version='2.0', scope=['learning'],
                            updated_at='2026-11-04', engine_code='import_std',
                            custom={'keep': ['original']}))
    (root / 'manifest.json').write_text(json.dumps(dict(document_count=3, documents=records),
                                                  ensure_ascii=False), encoding='utf-8')
    (root / 'README.md').write_text('AUXILIARY_NOT_KNOWLEDGE', encoding='utf-8')
    return root


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def unit(text='\ufeff  中文\r\n', fmt='text'):
    return KnowledgeDocument('D1', text, dict(source='knowledge/d.txt', format=fmt,
                             title='样例', version='2.0', scope=['learning'], updated_at='2026-11-04'),
                             segment_id='body', section_path=['已有标题'])


def assert_traceable(document, chunks, max_chars):
    """按可回溯和非空白内容覆盖验收，不重写分块算法。"""
    text = clean_text(document.text)
    covered = set()
    for chunk in chunks:
        meta = chunk.metadata
        assert len(chunk.page_content) <= max_chars and chunk.page_content.strip()
        assert text[meta['start_char']:meta['end_char']] == chunk.page_content
        assert meta['source'] == document.metadata['source']
        assert meta['version'] == document.metadata['version']
        assert meta['scope'] == document.metadata['scope']
        covered.update(range(meta['start_char'], meta['end_char']))
        if meta['format'] == 'json':
            assert meta['start_line'] is None and meta['end_line'] is None
            assert meta['json_path'] == document.metadata['json_path']
        else:
            assert meta['start_line'] == text.count('\n', 0, meta['start_char']) + 1
            assert meta['end_line'] == text.count('\n', 0, meta['end_char'] - 1) + 1
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())


class FixedEncoder:
    """按文本长度生成非零三维假向量，只验证调用和绑定，不表示语义质量。"""

    def __init__(self, rows=None):
        self.rows, self.calls = rows, []

    def encode_document(self, texts):
        self.calls.append(('document', list(texts)))
        return copy.deepcopy(self.rows) if self.rows is not None else [[len(text), 1, 2] for text in texts]

    def encode_query(self, texts):
        self.calls.append(('query', list(texts)))
        return [[1, 2, 3] for text in texts]


@pytest.mark.parametrize('fmt', ['markdown', 'json'])
def test_provided_chunk_conversion_keeps_locations_and_isolates_metadata(fmt):
    document = unit('中文\n第二行', fmt)
    if fmt == 'json':
        document.metadata['json_path'] = '$.sections[0].text'
    original = make_chunk(document, document.text, 0, len(document.text), 0, ['标题'])
    result = chunk_document(original)
    assert isinstance(result, Document) and result.id == 'D1:body:0000'
    assert result.page_content == original.text
    assert result.metadata['start_char'] == 0 and result.metadata['end_char'] == len(original.text)
    result.metadata['scope'].append('changed')
    assert original.metadata['scope'] == document.metadata['scope'] == ['learning']
    assert document_record(result)['id'] == result.id


@pytest.mark.parametrize('options', [
    {'max_chars': 0}, {'max_chars': True}, {'max_chars': 1.5},
    {'overlap': -1}, {'max_chars': 10, 'overlap': 10}, {'overlap': True},
    {'mode': 'unknown'},
])
def test_provided_split_options_fail_before_exercise(options):
    with pytest.raises(ValueError):
        make_splitter(**options)


def test_provided_empty_or_invalid_document_lists():
    assert split_documents([]) == []
    with pytest.raises(ValueError):
        split_documents(['不是 Document'])


@pytest.mark.parametrize('options', [
    {'embedding_mode': 'local_model'},
    {'embeddings': object()},
    {'embeddings': object(), 'embedding_mode': 'offline_fixture'},
    {'query': ' '},
])
def test_provided_invalid_run_config_stops_before_loading(tmp_path, options):
    output = tmp_path / 'output'
    with pytest.raises(ValueError):
        run_preparation(tmp_path / 'missing_dataset', output, **options)
    assert not output.exists()


@DAY4
def test_document_conversion_preserves_raw_text_identity_and_metadata():
    original = unit()
    before = copy.deepcopy(original)
    result = to_document(original)
    assert isinstance(result, Document) and result.id == 'D1:body'
    assert result.page_content == '\ufeff  中文\r\n'  # Loader 阶段不提前清洗。
    assert all(result.metadata[key] == value for key, value in before.metadata.items())
    assert result.metadata['doc_id'] == 'D1' and result.metadata['segment_id'] == 'body'
    result.metadata['scope'].append('changed')
    result.metadata['section_path'].append('changed')
    assert original == before


@DAY4
def test_manifest_loader_keeps_order_and_json_body_boundaries(small_dataset):
    before = hashes(small_dataset)
    docs = ManifestLoader(small_dataset).load()
    assert [doc.id for doc in docs] == ['D-Z:body', 'D-A:body', 'D-M:json-0', 'D-M:json-1']
    assert docs[0].page_content.startswith('\ufeff# 导入\r\n')
    assert docs[2].metadata['json_path'] == '$.sections[0].text'
    assert docs[3].metadata['section_path'] == ['查询 FAQ', '处理方式']
    assert all('NOTICE_NOT_KNOWLEDGE' not in doc.page_content and
               'AUXILIARY_NOT_KNOWLEDGE' not in doc.page_content for doc in docs)
    assert [doc.id for doc in ManifestLoader(small_dataset).lazy_load()] == [doc.id for doc in docs]
    assert hashes(small_dataset) == before


@DAY4
@pytest.mark.parametrize('problem,error', [('missing', FileNotFoundError),
                                         ('utf8', UnicodeDecodeError), ('json', json.JSONDecodeError)])
def test_loader_propagates_bad_source_errors(small_dataset, problem, error):
    if problem == 'missing':
        (small_dataset / 'knowledge/z.md').unlink()
    elif problem == 'utf8':
        (small_dataset / 'knowledge/z.md').write_bytes(b'\xff')
    else:
        (small_dataset / 'knowledge/m.json').write_text('bad JSON', encoding='utf-8')
    with pytest.raises(error):
        ManifestLoader(small_dataset).load()


@DAY4
def test_aligned_splitter_uses_character_windows_and_exact_overlap():
    text = '甲乙丙丁戊己庚辛壬癸子丑寅卯辰'
    parts = make_splitter(max_chars=6, overlap=2).create_documents([text])
    assert [part.page_content for part in parts] == [text[0:6], text[4:10], text[8:14], text[12:15]]
    assert [part.metadata['start_index'] for part in parts] == [0, 4, 8, 12]
    assert all(left.page_content[-2:] == right.page_content[:2] for left, right in zip(parts, parts[1:]))


@DAY4
def test_chunks_keep_titles_code_fences_and_exact_source_locations(small_dataset):
    units = load_documents(small_dataset)
    documents = ManifestLoader(small_dataset).load()
    before = copy.deepcopy(documents)
    chunks = split_documents(documents, max_chars=32, overlap=6)
    baseline = [chunk_document(chunk) for source in units for chunk in split_document(source, 32, 6)]
    assert [document_record(item) for item in chunks] == [document_record(item) for item in baseline]
    assert documents == before
    for source in units:
        selected = [item for item in chunks if item.metadata['doc_id'] == source.doc_id and
                    item.metadata['segment_id'] == source.segment_id]
        assert_traceable(source, selected, 32)
    assert all('不是标题' not in item.metadata['section_path'] for item in chunks)
    chunks[0].metadata['scope'].append('changed')
    assert documents[0].metadata['scope'] == chunks[1].metadata['scope'] == ['learning']


@DAY4
def test_recursive_separator_mode_changes_cuts_and_keeps_coverage():
    source = unit('甲' * 8 + '\n\n' + '乙' * 8 + '\n\n' + '丙' * 8)
    documents = [to_document(source)]
    aligned = split_documents(documents, 14, 4, mode='aligned')
    recursive = split_documents(documents, 14, 4, mode='recursive')
    assert [item.page_content for item in aligned] != [item.page_content for item in recursive]
    assert_traceable(source, recursive, 14)
    for left, right in zip(recursive, recursive[1:]):
        actual_overlap = max(0, left.metadata['end_char'] - right.metadata['start_char'])
        assert actual_overlap <= 4  # 递归模式不要求每次重叠满 4 个字符。


@DAY4
@pytest.mark.parametrize('kind', ['document', 'query'])
def test_embeddings_adapter_uses_correct_entry_and_array_shape(kind):
    encoder = FixedEncoder()
    adapter = Week7Embeddings(encoder)
    if kind == 'document':
        texts = [' 第一段中文 ', '第二段']
        result = adapter.embed_documents(texts)
        assert result == [[len(texts[0]), 1, 2], [len(texts[1]), 1, 2]]
        assert encoder.calls == [('document', texts)]
    else:
        assert adapter.embed_query(' 原始问题 ') == [1, 2, 3]
        assert encoder.calls == [('query', [' 原始问题 '])]
    encoder.calls.clear()
    assert adapter.embed_documents([]) == [] and encoder.calls == []


@DAY4
def test_adapter_preserves_qwen_query_prefix_on_mock_http(monkeypatch):
    requests = []
    original_client = httpx.Client
    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == '/api/embed'
        return httpx.Response(200, json={'embeddings': [[1, 2, 3]]})
    def mock_client(**kwargs):
        return original_client(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr('rag.embeddings.httpx.Client', mock_client)
    adapter = Week7Embeddings(OllamaEncoder())
    adapter.embed_documents(['正文'])
    adapter.embed_query('问题')
    assert requests[0] == {'model': 'qwen3-embedding:0.6b', 'input': ['正文'], 'truncate': False}
    assert requests[1]['input'] == [
        'Instruct: Given a user question, retrieve relevant passages that answer the question\nQuery:问题']
    assert requests[1]['truncate'] is False


@DAY4
@pytest.mark.parametrize('rows', [[], [[1, 0]], [[1, 0], [0, 0]], [[True, 1], [1, 0]],
                                  [[math.nan, 1], [1, 0]], [[1], [1, 2]]])
def test_embeddings_reject_invalid_model_vectors(rows):
    with pytest.raises(ValueError):
        Week7Embeddings(FixedEncoder(rows=rows)).embed_documents(['A', 'B'])


@DAY4
def test_embeddings_preserve_model_errors_and_reject_blank_query():
    class BrokenEncoder:
        def encode_document(self, texts):
            raise RuntimeError('模拟本地模型异常')
    with pytest.raises(RuntimeError, match='模拟本地模型异常'):
        Week7Embeddings(BrokenEncoder()).embed_documents(['正文'])
    encoder = FixedEncoder()
    with pytest.raises(ValueError):
        Week7Embeddings(encoder).embed_query(' ')
    assert encoder.calls == []


@DAY4
def test_small_preparation_saves_comparison_and_bound_vectors(small_dataset, tmp_path):
    before = hashes(small_dataset)
    encoder = FixedEncoder()
    output = tmp_path / 'artifacts'
    report = run_preparation(small_dataset, output, max_chars=32, overlap=6,
                             embeddings=Week7Embeddings(encoder), embedding_mode='offline_fixture',
                             model_name='fixed_vectors_not_a_model')
    assert report['same_chunks'] and report['document_units'] == 4
    assert report['dimensions'] == 3 and report['embedding_mode'] == 'offline_fixture'
    assert [kind for kind, texts in encoder.calls] == ['document', 'query']
    chunks = json.loads((output / 'chunks.json').read_text(encoding='utf-8'))
    vectors = json.loads((output / 'embedded_chunks.json').read_text(encoding='utf-8'))
    assert [item['chunk_id'] for item in vectors] == [item['id'] for item in chunks]
    assert encoder.calls[0][1] == [item['page_content'] for item in chunks]
    assert json.loads((output / 'comparison_report.json').read_text(encoding='utf-8')) == report
    assert (output / 'query_embedding.json').is_file() and hashes(small_dataset) == before


@DAY4
@pytest.mark.parametrize('mode', ['aligned', 'recursive'])
def test_week7_corpus_comparison_preserves_same_sources(mode):
    assert (DATASET / 'manifest.json').is_file(), '请准备原 Week7 语料或设置 WEEK7_RAG_DATASET'
    before = hashes(DATASET)
    output = OUTPUT / mode
    report = run_preparation(DATASET, output, mode=mode)
    assert report['source_documents'] == 18 and report['document_units'] == 20
    assert report['week7_chunk_count'] == 64 and len(report['unit_comparison']) == 20
    if mode == 'aligned':
        assert report['same_chunks'] and report['chunk_count'] == 64
    documents = ManifestLoader(DATASET).load()
    chunks = split_documents(documents, mode=mode)
    for source in load_documents(DATASET):
        selected = [item for item in chunks if item.metadata['doc_id'] == source.doc_id and
                    item.metadata['segment_id'] == source.segment_id]
        assert_traceable(source, selected, 500)
    assert hashes(DATASET) == before
    print(f'Week8 Day4 语料对照产物：{output}；same_chunks={report["same_chunks"]}')


@REAL
def test_local_ollama_embeddings_match_week7_backend():
    output = OUTPUT / 'local_model'
    report = run_preparation(DATASET, output, embeddings=Week7Embeddings(OllamaEncoder()),
                             embedding_mode='local_model', model_name='qwen3-embedding:0.6b')
    chunks = json.loads((output / 'chunks.json').read_text(encoding='utf-8'))
    vectors = json.loads((output / 'embedded_chunks.json').read_text(encoding='utf-8'))
    assert report['same_chunks'] and report['chunk_count'] == 64 and report['dimensions'] > 0
    # 仅额外编码第一块进行跨接口对照，不重复推理整批 64 块。
    baseline = embed_texts([chunks[0]['page_content']], OllamaEncoder(), kind='document')[0]
    assert vectors[0]['vector'] == pytest.approx(baseline, abs=1e-5)
    query = json.loads((output / 'query_embedding.json').read_text(encoding='utf-8'))
    assert query['query'] == QUERY and len(query['vector']) == report['dimensions']
    print(f'Week8 Day4 真实本地模型产物：{output}；仅验证编码/来源，不代表检索质量')
