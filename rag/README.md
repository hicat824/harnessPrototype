# Week7 Day2：文档加载、清洗与分块

本日目标：用简单 Python 代码把知识文件转换为可追溯的 Chunk，保留正文、标题层级和元数据。暂不涉及 Embedding、向量索引、检索或调用 LLM。

## 实现与复习入口

`rag/preprocessing.py` 中的三个处理函数均已实现，关键流程旁已添加注释。
复习时建议按下表顺序阅读；后续练习只在关键算法和流程步骤处留空，不要求从零编写整个函数。

| 函数 | 输入 | 输出 | 练习重点 |
|---|---|---|---|
| `load_documents` | 含 `manifest.json` 的目录 | `list[KnowledgeDocument]` | 按清单加载、保留原文和元数据，复用 JSON 适配器 |
| `clean_text` | 单个正文字符串 | 清洗后的字符串 | 去除开头 BOM、统一换行，保留正文结构 |
| `split_document` | 单个 Document、`max_chars`、`overlap` | `list[KnowledgeChunk]` | Markdown 标题分节、节内长度分块、保留来源 |

详细参数、异常与边界要求写在函数 docstring 中。

已提供的 `read_manifest`、`resolve_source_path`、`load_json_sections` 和 `make_chunk` 分别负责清单检查、路径解析、JSON 正文提取和 Chunk 构造，无须重写。

## 数据来源与处理架构

两类测试数据：

- **临时小样例**：测试文件自行在 pytest 的 `tmp_path` 下创建，包括中文、BOM、CRLF、标题、代码围栏及错误文件。不会修改已有业务文件。
- **Week7 知识样例**：此前准备的合成业务知识，默认目录为 `/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics`。只加载 `manifest.json` 列出的 18 份文件：15 份 Markdown、2 份 TXT、1 份 JSON。

JSON 适配器按 `sections[i].text` 提取正文，每节一个 Document。因此 18 份文件产生 **20 个正文单元**。JSON 的 `notice`、数据集 README、练习问题和格式示例不进入知识正文。

```text
manifest.json
    ↓ load_documents（保留原文）
KnowledgeDocument[]
    ↓ clean_text
清洗后的正文
    ↓ split_document（内部也调用 clean_text）
KnowledgeChunk[]
    ↓ 测试检查、导出
JSON 调试产物
```

清洗与分块测试直接构造输入，不要求加载函数先通过。完整样例测试再串起整个流程。

### 本日约定

1. 清洗只移除开头一个 BOM，并把 CRLF/CR 转为 LF。保留空行、缩进、尾部空格、代码和标识符；不要对全文调用 `strip()`。
2. `max_chars` 按 Python 字符数计算，**不是 Token 数**。重叠仅发生在同一标题节内，不能跨标题或文档。
3. Markdown 只要求识别行首 1～6 个 `#` 后接空格的标题；三个反引号或波浪号围栏内的标题样式文本不分节。不要求实现完整 Markdown 解析器。
4. TXT 不解析 Markdown；JSON 保留适配器给出的标题路径。
5. 不改输入对象、源文件或元数据；当前版本和历史版本都加载，版本选择留给后续检索处理。
6. 只使用已有合成样例。基础清洗不承担敏感信息识别或脱敏；这组测试也不证明任意外部文档已完成脱敏。

### 来源字段如何理解

| 字段 | 含义 |
|---|---|
| `doc_id` | 来源文件的知识标识，同一个 JSON 文件的多个正文单元共用它 |
| `segment_id` | 文件内正文单元：文本为 `body`，JSON 为 `json-0` 等 |
| `metadata` | 清单中除 `doc_id` 外的全部字段，包括来源、版本、scope、日期和业务编码 |
| `section_path` | Markdown 标题层级；JSON 为文档标题和 section 标题 |
| `start_char` / `end_char` | 清洗后正文内的 `[start_char, end_char)` 字符区间，不是原文件字节偏移 |
| `start_line` / `end_line` | MD/TXT 原文件的行号，从 1 开始，包含两端；JSON 为 `None` |
| `metadata.json_path` | JSON 原字段位置，例如 `$.sections[0].text`，与 `source` 配合回溯 |
| `chunk_id` | `doc_id:segment_id:编号`，编号从 `0000` 开始，同样输入应得到同样结果 |

## 运行方式

在仓库根目录执行。无需安装新依赖或下载模型。

**框架自检**：两个辅助函数的测试执行，32 个流程验收用例默认跳过。

```bash
.venv/bin/python -B -m pytest tests/test_rag_preprocessing.py -q -s -p no:cacheprovider
```

**执行全部验收**：

```bash
RUN_WEEK7_DAY2=1 .venv/bin/python -B -m pytest tests/test_rag_preprocessing.py -q -s -p no:cacheprovider
```

只检查某个函数时可以加 `-k`，例如：

```bash
RUN_WEEK7_DAY2=1 .venv/bin/python -B -m pytest tests/test_rag_preprocessing.py -q -k clean_text
```

如果样例目录不同，设置 `WEEK7_RAG_DATASET`。该目录仍需是同一组 Week7 样例，完整验收会检查固定文档数量和历史版本标识。

下面每节对应一个测试函数；参数化测试会执行多组输入。

## 测试一：JSON 正文与字段定位自检

函数：`test_provided_json_adapter_keeps_source_paths`，默认执行。

- **数据来源**：临时清单及含两个 section 的 JSON。
- **执行流程**：调用已提供的清单读取、JSON 适配函数。
- **检查点**：正文顺序、`doc_id`、`segment_id`、标题路径和 JSONPath 正确；`notice` 不进入正文。
- **输出**：pytest 断言结果，无导出文件。

## 测试二：原文定位与元数据复制自检

函数：`test_provided_chunk_locator_and_metadata_isolation`，默认执行。

- **数据来源**：直接构造带 BOM/CRLF 的 Markdown 和一个 JSON 正文单元。
- **执行流程**：调用已提供的 `make_chunk`，修改其中一个 Chunk 的元数据。
- **检查点**：Markdown 的原行号正确；JSON 不伪造行号；修改 Chunk 不影响输入和其他 Chunk。
- **输出**：pytest 断言结果，无导出文件。

## 测试三：按清单加载文件

函数：`test_load_documents_uses_manifest_and_preserves_raw_text`。

- **数据来源**：临时目录的 MD/TXT/JSON，以及清单外 README；另覆盖无后缀的 Markdown 文件。
- **执行流程**：调用你实现的 `load_documents`，与原文件字节和清单逐项对照。
- **检查点**：按清单顺序返回四个正文单元；按 `format` 判断格式；保留 BOM/CRLF 和全部元数据；不加载辅助 README；源文件不变。
- **输出**：pytest 断言结果，无导出文件。

## 测试四：加载错误不能静默跳过

函数：`test_load_documents_does_not_silently_skip_errors`，三组输入。

- **数据来源**：临时样例，分别制造文件缺失、非法 UTF-8 和损坏 JSON。
- **执行流程**：调用 `load_documents`，捕获并检查异常类型。
- **检查点**：分别抛出 `FileNotFoundError`、`UnicodeDecodeError`、`JSONDecodeError`，不能返回缺少文档的成功结果。
- **输出**：pytest 断言结果；仅修改临时测试副本。

## 测试五：保守清洗

函数：`test_clean_text_is_conservative_and_idempotent`，五组输入。

- **数据来源**：测试内定义的中文、BOM、混合换行、空行、缩进和代码文本。
- **执行流程**：调用 `clean_text`，与明确的预期字符串比较，并再次清洗。
- **检查点**：仅转换约定字符，正文不丢失；空串正常；对这些输入重复清洗结果不变。
- **输出**：pytest 断言结果，无导出文件。

## 测试六：分块参数边界

函数：`test_split_rejects_invalid_budget`，九组输入。

- **数据来源**：空正文和非法 `max_chars` / `overlap`。
- **执行流程**：调用 `split_document`。
- **检查点**：长度必须为正整数，重叠必须为非负整数且小于长度；拒绝 bool、字符串、小数。空正文也先校验参数。
- **输出**：pytest 断言结果，预期抛出 `ValueError`。

## 测试七：长度与重叠窗口

函数：`test_split_plain_text_has_bounded_windows`，三组输入。

- **数据来源**：十个汉字，`max_chars=4`，重叠分别为 0、1、3。
- **执行流程**：调用 `split_document`，核对每块字符区间。
- **检查点**：窗口长度和步长准确、最后一块到正文末尾即停止；正文完整覆盖、内容与区间一致、ID 稳定、元数据保留。
- **输出**：pytest 断言结果，无导出文件。

## 测试八：Markdown 标题边界

函数：`test_split_markdown_respects_heading_boundaries`。

- **数据来源**：导言、三级导入标题和独立导出标题组成的短文档。
- **执行流程**：以较小长度分块，独立核对已知的标题位置和路径，再重复调用。
- **检查点**：标题层级正确；正文不丢失；分块和重叠不跨节；输入不变；重复执行得到相同 Chunk。
- **输出**：pytest 断言结果，无导出文件。

## 测试九：代码围栏中的伪标题

函数：`test_markdown_headings_inside_code_fences_are_not_sections`，四组输入。

- **数据来源**：分别使用反引号和波浪号围栏的 Markdown，代码内包含 `## not a heading`；另覆盖较短或不同字符的内部围栏。
- **执行流程**：设置足够大长度，使每个真正标题节各生成一块。
- **检查点**：代码内的 `#` 不产生新节，较短或不同字符的围栏不能提前结束代码块；缩进与代码原文保留，围栏外真实标题仍正常分节。
- **输出**：pytest 断言结果，无导出文件。

## 测试十：空正文处理

函数：`test_split_empty_documents_returns_no_chunks`，三组输入。

- **数据来源**：空串、纯空白、带 BOM 和换行的空正文。
- **执行流程**：用合法参数调用 `split_document`。
- **检查点**：全部返回空列表，不生成无意义的空白 Chunk。
- **输出**：pytest 断言结果，无导出文件。

## 测试十一：TXT 与 JSON 不套用 Markdown 规则

函数：`test_text_and_json_use_their_own_structure`。

- **数据来源**：以 `#` 开头的普通 TXT，以及临时 JSON 的两个 section。
- **执行流程**：分别分块，再修改一个结果的嵌套元数据。
- **检查点**：TXT 的 `#` 当普通文本；JSON 保留原标题路径和 JSONPath；跨 section 的 Chunk ID 唯一；嵌套元数据互不影响。
- **输出**：pytest 断言结果，无导出文件。

## 测试十二：完整 Week7 样例验收

函数：`test_week7_samples_end_to_end`。

- **数据来源**：已有的 18 份 Week7 合成知识文件与清单。
- **执行流程**：加载 → 清洗 → 分块（500 字符、重叠 50）→ 核对区间、元数据及原文件哈希 → 导出。
- **检查点**：18 份文件得到 20 个正文单元；历史版本 KB003/KB009 保留；Chunk 唯一、长度合规、可回溯且覆盖正文；处理不修改输入及源文件。
- **输出**：通过本测试后在 pytest 临时目录生成以下文件，使用 `-s` 可看到目录路径。

| 文件 | 内容 |
|---|---|
| `loaded_documents.json` | 加载后的正文单元及元数据 |
| `cleaned_documents.json` | 清洗后的正文单元 |
| `chunks.json` | 全部 Chunk、标题路径与定位字段 |
| `day2_report.json` | 本次完整样例检查通过标记、文档数、Chunk 数及参数 |

`day2_report.json` 只代表测试十二的断言通过；全部学习验收以整组 pytest 结果为准。

完成后还需人工抽查 `chunks.json` 中一个 Markdown Chunk 和一个 JSON Chunk，依据 `source` 与行号/JSONPath 找回原文，并解释为什么没有跨流程混块。本日不会验证检索效果或回答质量。
