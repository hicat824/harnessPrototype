"""Week8 Day4：Manifest Loader → Document → Text Splitter → Embeddings。

沿用 Week7 的同一清单、JSON 适配器、标题边界和 Ollama 后端。
已实现 Document 转换、Splitter 构造和两个编码入口。
不建立向量索引、不调用回答模型。说明见 langchain_learning/week8.md#day4。
"""

import copy
import json
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import OllamaEncoder, _vectors, embed_texts
from rag.preprocessing import (
    KnowledgeChunk, KnowledgeDocument, clean_text, load_documents, load_json_sections,
    make_chunk, read_manifest, resolve_source_path, split_document,
)


DEFAULT_DATASET = Path('/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics')
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'langchain_learning/artifacts/day4'
QUERY = 'NOT_FOUND 是否表示任务执行失败？'


def to_document(unit: KnowledgeDocument) -> Document:
    """将一个 Week7 正文单元转换为 LangChain Document，保留原文和独立元数据。

    输入：unit.text 为未清洗原文；metadata 包含 source/version/scope/format 等。
    输出：id='doc_id:segment_id'，page_content=unit.text；metadata 在原字段上增加
    doc_id、segment_id、section_path。此函数不清洗正文，分块时才统一清洗。
    元数据使用深拷贝，避免后续处理影响源正文单元或其他 Document。
    """
    metadata = copy.deepcopy(unit.metadata)
    metadata.update(doc_id=unit.doc_id, segment_id=unit.segment_id,
                    section_path=list(unit.section_path))
    # Document 的正文、身份和来源信息分开保存，不把 metadata 混入正文。
    return Document(id=f'{unit.doc_id}:{unit.segment_id}', page_content=unit.text, metadata=metadata)


class ManifestLoader(BaseLoader):
    """已提供：实现 LangChain BaseLoader 协议，只加载清单声明的正文。

    load() 是继承的方法，收集 lazy_load() 的结果得到 list[Document]。
    MD/TXT 每文件一个正文；JSON 复用 sections[].text 适配，忽略 notice 等说明。
    不扫描 README、问题集或参考答案。路径/文件/编码/JSON 错误原样向外抛出。
    """

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)

    def lazy_load(self) -> Iterator[Document]:
        for record in read_manifest(self.dataset_dir):
            path = resolve_source_path(self.dataset_dir, record['source'])
            with path.open('r', encoding='utf-8', newline='') as stream:
                raw = stream.read()
            if record['format'] == 'json':
                units = load_json_sections(record, raw)
            else:
                units = [KnowledgeDocument(
                    doc_id=record['doc_id'], text=raw,
                    metadata=copy.deepcopy({key: value for key, value in record.items() if key != 'doc_id'}),
                )]
            for unit in units:
                yield to_document(unit)


def _check_split_options(max_chars: int, overlap: int, mode: str) -> None:
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError('max_chars 必须为正整数')
    if type(overlap) is not int or not 0 <= overlap < max_chars:
        raise ValueError('overlap 必须满足 0 <= overlap < max_chars')
    if mode not in ('aligned', 'recursive'):
        raise ValueError('mode 只能为 aligned/recursive')


def make_splitter(max_chars: int = 500, overlap: int = 50,
                  mode: str = 'aligned') -> RecursiveCharacterTextSplitter:
    """创建节内分块器；只构造对象，不加载文件或执行模型。

    输入：max_chars/overlap 按 Python len(str) 计字符，不是 Token。
    aligned 仅用空字符串分隔，保持 Week7 固定字符窗口；recursive 优先段落/换行/
    中文句号等边界，overlap 是目标上限，不能要求相邻块一定重叠满额。
    输出：RecursiveCharacterTextSplitter，add_start_index=True 用于回溯节内位置；
    strip_whitespace=False 保留原文，分隔符也保留。参数校验和 options 已提供。
    options 显式固定字符计数、分隔符、原文保留及位置跟踪规则。
    """
    _check_split_options(max_chars, overlap, mode)
    separators = [''] if mode == 'aligned' else ['\n\n', '\n', '。', '；', ' ', '']
    options = dict(chunk_size=max_chars, chunk_overlap=overlap, length_function=len,
                   separators=separators, keep_separator=True,
                   add_start_index=True, strip_whitespace=False)
    # ** 将配置字典展开成关键字参数；* 只会将键名展开为位置参数。
    splitter = RecursiveCharacterTextSplitter(**options)
    return splitter


def chunk_document(chunk: KnowledgeChunk) -> Document:
    """已提供：把已定位的 Chunk 表示为 Document，字符/行号放 metadata。"""
    metadata = copy.deepcopy(chunk.metadata)
    metadata.update(doc_id=chunk.doc_id, segment_id=chunk.segment_id,
                    section_path=list(chunk.section_path), start_char=chunk.start_char,
                    end_char=chunk.end_char, start_line=chunk.start_line, end_line=chunk.end_line)
    return Document(id=chunk.chunk_id, page_content=chunk.text, metadata=metadata)


def split_documents(documents: list[Document], max_chars: int = 500, overlap: int = 50,
                    mode: str = 'aligned') -> list[Document]:
    """已提供：按 Week7 标题边界分节，再调用 LangChain 拆分每一节。

    输入为 ManifestLoader 输出的 Document 列表，不改输入对象；[] 返回 []。
    标题解析/代码围栏规则复用 Week7 split_document，给足整个单元长度使每节只产生
    一个区间；节内窗口才交给 LangChain，不是用 Week7 完成全部分块后再换类型。
    输出 Chunk 仍为 Document；id 与 Week7 一样按单元内序号编号。
    start_char/end_char 相对清洗后的整个正文单元，start_index 只作为节内中间值。
    TXT/MD 保留原文件行号；JSON 保留 json_path，行号为 None，不伪造 JSON 行号。
    """
    _check_split_options(max_chars, overlap, mode)
    if not isinstance(documents, list) or any(not isinstance(item, Document) for item in documents):
        raise ValueError('documents 必须为 Document 列表')
    if not documents:
        return []
    splitter = make_splitter(max_chars, overlap, mode)
    chunks = []
    for document in documents:
        metadata = copy.deepcopy(document.metadata)
        if not {'doc_id', 'segment_id', 'section_path', 'source', 'format'} <= metadata.keys():
            raise ValueError('Document 缺少来源或正文单元元数据')
        unit = KnowledgeDocument(
            doc_id=metadata.pop('doc_id'), segment_id=metadata.pop('segment_id'),
            section_path=metadata.pop('section_path'), text=document.page_content, metadata=metadata,
        )
        cleaned = clean_text(unit.text)
        sections = split_document(unit, max_chars=max(1, len(cleaned)), overlap=0)
        number = 0
        for section in sections:
            parts = splitter.split_documents([Document(page_content=section.text)])
            for part in parts:
                if not part.page_content.strip():
                    continue
                start = section.start_char + part.metadata['start_index']
                end = start + len(part.page_content)
                if not section.start_char <= start < end <= section.end_char:
                    raise ValueError('Splitter 返回的区间超出本节边界')
                if cleaned[start:end] != part.page_content:
                    raise ValueError('Splitter 位置无法回溯原文')
                chunk = make_chunk(unit, cleaned, start, end, number, section.section_path)
                chunks.append(chunk_document(chunk))
                number += 1
    return chunks


class Week7Embeddings(Embeddings):
    """将既有编码器适配为 LangChain Embeddings 接口，不替换模型或修改前缀。

    encoder 提供 encode_document(list[str]) 和 encode_query(list[str])。
    embed_texts 已处理空批次、输入/向量校验和模型异常；不重复实现 HTTP。
    两个入口不同：文档接收列表、返回二维数组；问题接收字符串、返回一维向量。
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码正文；顺序对应 texts，空列表不请求模型。"""
        # embed_texts 默认使用 document 入口，一次批量编码并保留顺序。
        return embed_texts(texts, self.encoder)

    def embed_query(self, text: str) -> list[float]:
        """编码一个问题；底层接收单元素列表，返回时取对应的一维向量。"""
        # 底层统一接收列表；query 入口保留既有检索前缀，结果中只有一个向量。
        vector_list = embed_texts([text], self.encoder, kind="query")
        return vector_list[-1]


def document_record(document: Document) -> dict:
    """已提供：导出普通 JSON 字典，明确保存正文、ID 与来源元数据。"""
    return dict(id=document.id, page_content=document.page_content, metadata=copy.deepcopy(document.metadata))


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_preparation(dataset_dir: str | Path, output_dir: str | Path, *,
                    max_chars: int = 500, overlap: int = 50, mode: str = 'aligned',
                    embeddings: Embeddings | None = None, query: str = QUERY,
                    embedding_mode: str = 'not_run', model_name: str | None = None) -> dict:
    """已提供：完成加载/分块/对照，并可选编码；成功后保存报告。

    dataset_dir 为 Week7 清单目录，output_dir 存本次 JSON；mode 决定节内切分规则。
    embeddings=None 时只做离线预处理；否则显式标明 offline_fixture/local_model 及
    model_name。query 为非空文本；不做检索、不调用回答模型，不建立向量索引。
    输出报告含两版数量、块级一致性、逐正文单元对比和实际向量维数。
    空语料不编码问题或文档；执行失败不写成功文件，旧产物不会自动删除。
    """
    _check_split_options(max_chars, overlap, mode)
    if (embeddings is None and embedding_mode != 'not_run') or (
        embeddings is not None and embedding_mode not in ('offline_fixture', 'local_model')
    ):
        raise ValueError('embeddings 与 embedding_mode 不匹配')
    if embeddings is not None and (not isinstance(model_name, str) or not model_name.strip()):
        raise ValueError('编码时必须提供 model_name')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query 必须为非空白字符串')

    documents = ManifestLoader(dataset_dir).load()
    chunks = split_documents(documents, max_chars, overlap, mode)
    # 另一条加载/分块路径只用于对照，不作为 LangChain 结果的输入。
    baseline_units = load_documents(dataset_dir)
    baseline = [chunk_document(chunk) for unit in baseline_units
                for chunk in split_document(unit, max_chars, overlap)]
    rows = [document_record(item) for item in chunks]
    baseline_rows = [document_record(item) for item in baseline]
    report = dict(splitter_mode=mode, max_chars=max_chars, overlap=overlap,
                  document_units=len(documents), source_documents=len(read_manifest(dataset_dir)),
                  chunk_count=len(chunks), week7_chunk_count=len(baseline), same_chunks=rows == baseline_rows,
                  embedding_mode=embedding_mode, model_name=model_name, dimensions=None,
                  query=query, unit_comparison=[])
    for document in documents:
        key = (document.metadata['doc_id'], document.metadata['segment_id'])
        selected = [item for item in rows if (item['metadata']['doc_id'], item['metadata']['segment_id']) == key]
        old = [item for item in baseline_rows if (item['metadata']['doc_id'], item['metadata']['segment_id']) == key]
        report['unit_comparison'].append(dict(doc_id=key[0], segment_id=key[1],
                                             chunk_count=len(selected), week7_chunk_count=len(old),
                                             same_chunks=selected == old))
    vectors = query_vector = None
    if embeddings is not None and chunks:
        vectors = _vectors(embeddings.embed_documents([item.page_content for item in chunks]))
        query_vector = _vectors([embeddings.embed_query(query)])[0]
        if len(vectors) != len(chunks) or len(query_vector) != len(vectors[0]):
            raise ValueError('向量数量或问题/正文向量维数不匹配')
        report['dimensions'] = len(query_vector)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / 'documents.json', [document_record(item) for item in documents])
    _save_json(output / 'chunks.json', rows)
    _save_json(output / 'week7_chunks.json', baseline_rows)
    _save_json(output / 'comparison_report.json', report)
    if vectors is not None:
        _save_json(output / 'embedded_chunks.json', [dict(chunk_id=item.id, vector=vector)
                                                   for item, vector in zip(chunks, vectors)])
        _save_json(output / 'query_embedding.json', dict(query=query, vector=query_vector))
    return report


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Week8 Day4：LangChain 文档预处理与 Week7 对照')
    parser.add_argument('--dataset', default=os.environ.get('WEEK7_RAG_DATASET', str(DEFAULT_DATASET)))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT / 'preprocess'))
    parser.add_argument('--mode', choices=['aligned', 'recursive'], default='aligned')
    parser.add_argument('--local-model', action='store_true', help='显式调用已安装的本地 Ollama Embedding')
    args = parser.parse_args()
    embeddings = Week7Embeddings(OllamaEncoder()) if args.local_model else None
    report = run_preparation(args.dataset, args.output, mode=args.mode, embeddings=embeddings,
                             embedding_mode='local_model' if args.local_model else 'not_run',
                             model_name='qwen3-embedding:0.6b' if args.local_model else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Week8 Day4 产物：{Path(args.output).resolve()}')
