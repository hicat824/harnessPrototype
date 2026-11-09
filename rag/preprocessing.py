"""Week7 Day2：知识加载、保守清洗和分块练习。

使用 Python 标准库，不调用模型或建立向量索引。
处理顺序：按清单加载 → 保守清洗 → 按标题分节 → 节内重叠分块。
输入输出契约与逐测试说明见 rag/week7.md#day2。
"""

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re


@dataclass
class KnowledgeDocument:
    """待处理的正文单元。

    doc_id 标识来源文档；segment_id 区分同一来源中的正文单元。
    MD/TXT 每文件一个 body；JSON 每个 sections[i].text 一个 json-i。
    text 是未清洗正文，metadata 保存清单字段（不含 doc_id）。
    JSON 的标题路径放 section_path，原 JSON 定位放 metadata['json_path']。
    """

    doc_id: str
    text: str
    metadata: dict
    segment_id: str = "body"
    section_path: list[str] = field(default_factory=list)


@dataclass
class KnowledgeChunk:
    """正文片段及来源信息，元数据和标题路径均独立复制。

    start_char/end_char 是清洗后 Document.text 的左闭右开字符区间，
    不是原文件的字节偏移。MD/TXT 的 start_line/end_line 为原文件行号，
    从 1 开始且包含两端；JSON 两者为 None，使用 metadata.json_path 回溯。
    """

    chunk_id: str
    doc_id: str
    segment_id: str
    text: str
    metadata: dict
    section_path: list[str]
    start_char: int
    end_char: int
    start_line: int | None
    end_line: int | None


def resolve_source_path(dataset_dir: str | Path, source: str) -> Path:
    """已提供：将清单中的相对路径解析到数据集内；越界抛 ValueError。"""
    root = Path(dataset_dir).resolve()
    if not isinstance(source, str) or not source or Path(source).is_absolute():
        raise ValueError("source 必须是数据集内的相对路径")
    path = (root / source).resolve()
    if root not in path.parents:
        raise ValueError("source 超出数据集目录")
    return path


def read_manifest(dataset_dir: str | Path) -> list[dict]:
    """已提供：读取并检查清单，按清单顺序返回记录，不递归扫描目录。

    支持 markdown/text/json；必填字段缺失、ID/来源重复或格式未知时抛
    ValueError。文件不存在、JSON 无法解析等错误保留原异常。
    本函数不加载正文；路径解析可继续使用 resolve_source_path。
    """
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("documents")
    if not isinstance(records, list) or manifest.get("document_count") != len(records):
        raise ValueError("documents 必须是与 document_count 一致的列表")
    required = {"doc_id", "title", "source", "version", "scope", "updated_at", "format"}
    ids, sources = set(), set()
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("知识清单缺少必填字段")
        if not isinstance(record["doc_id"], str) or not record["doc_id"]:
            raise ValueError("doc_id 必须是非空字符串")
        if record["format"] not in ("markdown", "text", "json"):
            raise ValueError("仅支持 markdown/text/json")
        resolve_source_path(root, record["source"])
        if record["doc_id"] in ids or record["source"] in sources:
            raise ValueError("清单 doc_id/source 不能重复")
        ids.add(record["doc_id"])
        sources.add(record["source"])
    return records


def load_json_sections(record: dict, raw_text: str) -> list[KnowledgeDocument]:
    """已提供：把样例 JSON 的 sections[].text 适配成正文单元。

    输入：一条清单记录、JSON 文件原文。输出：按 section 顺序的 Document
    列表，doc_id 不变，segment_id 为 json-0/json-1/...。
    title/heading 保存在标题路径，notice 等其他字段不作为正文。
    json_path 精确指向原 JSON 字段，不用拼接文本的行号冒充原文件行号。
    JSON 语法错误保留原异常；sections 或 heading/text 类型不符抛 ValueError。
    """
    data = json.loads(raw_text.removeprefix("\ufeff"))
    sections = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(sections, list):
        raise ValueError("JSON 正文必须包含 sections 列表")
    documents = []
    for index, section in enumerate(sections):
        if (not isinstance(section, dict)
                or not isinstance(section.get("heading"), str)
                or not isinstance(section.get("text"), str)):
            raise ValueError("每个 section 必须包含字符串 heading/text")
        metadata = copy.deepcopy({key: value for key, value in record.items() if key != "doc_id"})
        metadata["json_path"] = f"$.sections[{index}].text"
        documents.append(KnowledgeDocument(
            doc_id=record["doc_id"], segment_id=f"json-{index}", text=section["text"],
            metadata=metadata, section_path=[record["title"], section["heading"]],
        ))
    return documents


def make_chunk(
    document: KnowledgeDocument, cleaned_text: str, start: int, end: int,
    number: int, section_path: list[str],
) -> KnowledgeChunk:
    """已提供：按已确定的区间构造 Chunk，不替你决定分块边界。

    start/end 是清洗正文内的 [start, end)，必须非空；number 从 0 开始。
    清洗不删行，因此文本文件可用 LF 数计算原文件行号；JSON 用 JSONPath。
    深拷贝元数据，避免修改一个 Chunk 时影响原 Document 或其他 Chunk。
    """
    if not 0 <= start < end <= len(cleaned_text):
        raise ValueError("Chunk 区间必须是正文内的非空左闭右开区间")
    if document.metadata["format"] == "json":
        start_line = end_line = None
    else:
        start_line = cleaned_text.count("\n", 0, start) + 1
        end_line = cleaned_text.count("\n", 0, end - 1) + 1
    return KnowledgeChunk(
        chunk_id=f"{document.doc_id}:{document.segment_id}:{number:04d}",
        doc_id=document.doc_id, segment_id=document.segment_id,
        text=cleaned_text[start:end], metadata=copy.deepcopy(document.metadata),
        section_path=list(section_path), start_char=start, end_char=end,
        start_line=start_line, end_line=end_line,
    )


def load_documents(dataset_dir: str | Path) -> list[KnowledgeDocument]:
    """按 manifest 加载正文并保留元数据，返回 Document 列表。

    输入：包含 manifest.json 的数据集目录。输出：按清单顺序排列的正文单元。
    使用 read_manifest / resolve_source_path，不扫描清单外的 README、问题或示例。
    - UTF-8 读取，使用 open(..., encoding='utf-8', newline='') 保留原换行。
      本函数不调用 clean_text；BOM、CRLF 等保留给下一阶段处理。
    - markdown/text：每文件一个 KnowledgeDocument，segment_id='body'，
      text 为完整原文，section_path=[]；metadata 为清单中除 doc_id 外的全部字段。
    - json：直接复用提供的 load_json_sections(record, raw_text)。
    - 当前 18 份样例文件会产生 20 个正文单元：17 份文本 + JSON 的 3 个 section。
      doc_id 标识来源文件，segment_id 区分同文件正文，不随意重命名 doc_id。
    文件缺失、UTF-8 解码失败等异常原样抛出，不静默跳过，也不改写源文件。
    """
    manifest_list = read_manifest(dataset_dir)
    doc_data_list = []
    for manifest in manifest_list:
        # source 相对数据集目录解析，不能依赖运行命令时的当前目录。
        doc_path = resolve_source_path(dataset_dir, manifest["source"])
        with doc_path.open("r", encoding="utf-8", newline="") as f:
            content = f.read()

        # 清单 format 决定正文结构；JSON 复用适配器，避免把说明字段混入正文。
        if manifest["format"] == "json":
            doc_data_list.extend(load_json_sections(manifest, content))
        else:
            knowledge = KnowledgeDocument(
                doc_id=manifest["doc_id"],
                text=content,
                # 深拷贝也隔离 scope 等嵌套字段，方便后续安全地处理元数据。
                metadata=copy.deepcopy({k: v for k, v in manifest.items() if k != "doc_id"}),
                section_path=[],
                segment_id="body",
            )
            doc_data_list.append(knowledge)

        # 不捕获后原样重抛：读取、解码、解析失败自然传给调用方。

    return doc_data_list


def clean_text(text: str) -> str:
    """对一个正文字符串做保守清洗，返回新字符串。

    仅做两种确定的转换：
    1. 移除文本最开头的一个 BOM（U+FEFF），正文中间的字符保留。
    2. CRLF / 单独 CR 统一成 LF（先处理 CRLF，避免变成两个换行）。
    其余字符原样保留：不 strip 全文、不合并空行、不压缩缩进、不改标识符、
    标点和中文，不删除标题/代码围栏。空串返回空串，重复清洗应得到同样结果。
    保持逻辑行数，是后续直接回溯原文件行号的前提；需要删行时应另做映射。
    """
    # 只处理开头的编码标记，避免误删正文中间的 U+FEFF。
    text = text.removeprefix("\ufeff")
    # 先替换双字符换行，再替换单独 CR，防止一行变成两行。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def split_document(
    document: KnowledgeDocument, max_chars: int = 500, overlap: int = 50,
) -> list[KnowledgeChunk]:
    """先按标题分节，再在节内按字符长度分块。

    输入：一个 Document，max_chars 为真正 int 且 >0，overlap 为真正 int，
    满足 0<=overlap<max_chars；否则在处理正文前抛 ValueError（拒绝 bool）。
    输出：按原文顺序排列的 Chunk 列表；空/全空白正文返回 []，不改 Document。

    清洗：先调用 clean_text，后续字符区间都相对此次清洗后的正文。
    分节：
    - markdown 识别行首 1～6 个 # 后接空格的 ATX 标题，标题行属于新节。
      更新层级路径，例如 '# 总览'/'## 参数' 对应 ['总览', '参数']。
      标题前的导言路径为 []。不跨节合并，不往正文添加重复标题。
    - 行首三个反引号或三个波浪号的代码围栏内，不把 '# ...' 识别为标题。
      本日只实现上述简单 Markdown 子集，不要求完整解析 Markdown。
    - text/json 不解析 Markdown，整个正文单元为一个节，沿用 document.section_path。

    长度：max_chars 按 Python len(str) 计字符，不是 Token；每个节独立拆分。
    节内连续窗口重叠 overlap 个字符，最后一块到节末结束，不能跨节重叠。
    不输出纯空白 Chunk；允许跳过纯空白片段，其他字符必须被至少一块覆盖。
    定位与复制：使用 make_chunk，number 为本单元返回顺序的连续编号，从 0 开始。
    必须满足 chunk.text == cleaned_text[start_char:end_char]，元数据完整保留。
    JSON 行号为 None，保留 json_path；MD/TXT 保留 1-based 的原文件行号。
    """
    # bool 是 int 的子类，所以这里用 type(...) is int，拒绝 True/False。
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars 必须为大于 0 的整数")
    if type(overlap) is not int or not 0 <= overlap < max_chars:
        raise ValueError("overlap 必须为整数，且满足 0 <= overlap < max_chars")

    # 1. 只生成清洗后的副本；后续所有字符位置统一在这份正文上计算。
    cleaned_content = clean_text(document.text)
    if not cleaned_content.strip():
        return []

    # 每个节保存 (起点, 终点, 标题路径)，直接引用正文区间，不重新拼接正文。
    sections = []
    if document.metadata["format"] == "markdown":
        section_start = 0
        section_path = []
        heading_stack = []  # (标题级别, 标题文字)，允许 # 后直接出现 ###。
        offset = 0
        fence_char = None
        fence_length = 0

        # 2. 一次扫描确定标题边界；保留换行才能准确累加原正文字符位置。
        for line in cleaned_content.splitlines(keepends=True):
            line_text = line.rstrip("\n")
            fence = re.match(r"^(`{3,}|~{3,})(.*)$", line_text)
            if fence_char is not None:
                # 只有相同字符、长度足够且无其他文字的围栏才能结束代码块。
                if (fence and fence.group(1)[0] == fence_char
                        and len(fence.group(1)) >= fence_length
                        and not fence.group(2).strip()):
                    fence_char = None
            elif fence:
                fence_char = fence.group(1)[0]
                fence_length = len(fence.group(1))
            else:
                heading = re.match(r"^(#{1,6}) (.*)$", line_text)
                if heading:
                    # 当前标题行属于新节，旧节在该行之前结束。
                    if offset > section_start:
                        sections.append((section_start, offset, section_path))

                    level = len(heading.group(1))
                    # 遇到同级或更高层标题，弹出旧分支；保留仍有效的父级。
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, heading.group(2).strip()))
                    section_path = [title for _, title in heading_stack]
                    section_start = offset
            offset += len(line)

        sections.append((section_start, len(cleaned_content), section_path))
    else:
        # TXT 不识别 #；JSON 每个正文单元已有独立标题路径，不跨单元合并。
        sections.append((0, len(cleaned_content), document.section_path))

    # 3. 只在各节内部移动窗口，步长为 max_chars - overlap。
    chunks = []
    for section_start, section_end, section_path in sections:
        start = section_start
        while start < section_end:
            end = min(start + max_chars, section_end)
            if cleaned_content[start:end].strip():
                # 复用辅助函数保留字符/行号定位，并独立复制嵌套元数据。
                chunks.append(make_chunk(
                    document, cleaned_content, start, end, len(chunks), section_path,
                ))
            if end == section_end:
                break  # 已覆盖节末，不再为尾部重叠区域重复生成一块。
            start = end - overlap
    return chunks
