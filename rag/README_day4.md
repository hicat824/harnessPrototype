# Week7 Day4：最小向量索引与 Retriever

本日实现文件：`rag/retrieval.py`。测试文件：`tests/test_rag_retrieval.py`。
采用 **JSON 保存索引 + Python 逐个计算相似度**，复用 Day2/Day3，不安装新数据库或检索框架。

> 学习状态（2026-11-05）：**Week7 Day4 已完成，当前进入 Day5。**
> 离线验收：42 passed、1 skipped；真实 Ollama 查询验收：1 passed。
> 已验证保存、加载、过滤、来源回溯，以及新进程查询时只编码问题、不重新编码正文。

## 学习内容与验收标准

| 学习内容 | 验收标准 |
|---|---|
| Index 保存什么 | 能说明正文、向量、来源及模型配置的对应关系 |
| 建库与查询的区别 | 建库批量编码正文；查询只编码新问题，复用已有正文向量 |
| Retriever 如何返回结果 | 命中包含原文、分数、doc_id、source、version 和原文定位 |
| 候选过滤与排名 | 先按明确元数据条件过滤，再取 Top-K；过滤后仍能映射回正确 Chunk |
| 持久化与重启 | 索引保存后，新 Python 进程可以加载并检索，无需重编码正文 |

本日不生成回答、不组装 LLM Prompt、不评估完整问答质量，也不实现 ANN、Hybrid Search 或 Rerank。

## 三个关键步骤回顾

三个关键步骤均已实现，可按下表复习建库与查询的不同处理流程。

| 位置 | 输入 | 当前实现 | 理解重点 |
|---|---|---|---|
| `build_index` | 正文列表 `texts`、编码器 | 生成 `vectors` | 建库时一次批量编码正文 |
| `search_knowledge` 前半段 | 索引、`filters`、`_matches_filters` | 生成 `candidate_indices` | 先过滤，并保存候选在原索引中的下标 |
| `search_knowledge` 结果循环 | `local_index`、候选下标列表 | 得到 `original_index` | Top-K 返回的是候选内位置，需要映射回原 Chunk |

函数 docstring 已写明输入、输出、异常与边界。保存/加载、Day3 导入、参数检查、元数据匹配、问题编码、Top-K 调用和命令行均已提供。

## 数据来源与处理架构

**离线数据**由测试内生成，包含四个合成 Chunk 和固定二维向量：

| 原下标 | doc_id | 流程 | 状态 | 与固定问题的预设相似度 |
|---|---|---|---|---|
| 0 | D0 | export_std | current | 1.0 |
| 1 | D1 | import_std | historical | 1.0 |
| 2 | D2 | import_std | current | 0.8 |
| 3 | D3 | import_std | current | 0.0 |

D2 是带 JSONPath 的 JSON 正文单元，其他记录是普通文本。向量关系是人工设定的，不代表真实模型的中文理解。
筛选 `import_std + current` 后，候选下标 0 对应原下标 2，应该返回 D2。直接拿候选下标查原列表会误返回 D0，本日专门测试这一点。

**真实数据**复用 Day3 的两份已有文件：

- `rag/artifacts/day3/local_model/embedded_chunks.json`：Chunk 与向量。
- `rag/artifacts/day3/local_model/day3_report.json`：模型名、查询说明、条数和维数。

```text
新知识建库：Chunk → build_index（批量正文编码）→ KnowledgeIndex
已有向量导入：Day3 两份产物 → import_day3_artifacts → KnowledgeIndex
                                                        ↓
                                                save_index → JSON 文件
                                                        ↓
新进程：load_index → 条件过滤 → 编码新问题 → Top-K → 映射原 Chunk → 返回结果
```

CLI 的 `build` 使用第二条导入路径，不调用 `build_index`，也不请求模型。“新知识首次入库”的 `build_index` 流程由离线测试覆盖。
每次查询都重新加载/编码所有正文，会失去持久化索引的意义。

## 索引与检索接口

`KnowledgeIndex` 保存模型名、查询说明、Chunk 列表和对应向量列表；维数从实际向量推导。
JSON 文件包括 `schema_version=1`、`embedding` 配置和 `items`。每个 item 同时保存一个 Chunk 及其 vector。

| 函数 | 输入 | 输出与作用 |
|---|---|---|
| `build_index` | Chunk 列表、编码器 | 新建索引；不修改输入 |
| `save_index` | 索引、文件路径 | 保存完整 JSON，返回路径；不请求模型 |
| `load_index` | 文件路径 | 校验并恢复索引；不读取源文件、不请求模型 |
| `import_day3_artifacts` | Day3 真实产物目录 | 复用已有向量，拒绝错误模式或不一致的条数/维数 |
| `search_knowledge` | 问题、索引、编码器、K、可选过滤和分数门槛 | 返回带分数与完整 Chunk 的列表 |

检索结果的单项结构：

```text
{
  "score": 相似度,
  "chunk": {
    "chunk_id": 原 Chunk ID,
    "doc_id": 来源文档 ID,
    "text": 原片段正文,
    "metadata": {"source": 来源, "version": 版本, ...},
    ...原有标题路径、字符区间、行号或 JSONPath...
  }
}
```

过滤参数支持 `engine_code`、`knowledge_status`、`version`，按字段精确相等，多个条件之间为 AND。
函数默认不过滤；CLI 默认只查 `current`，通过 `--knowledge-status all` 可以包含历史资料。
这些条件是检索适用范围筛选，不替代业务授权检查。

`min_score=None` 默认关闭分数门槛；设置后只返回达到门槛的项，不保证凑满 K。
分数门槛不是通用的“事实可信度”，需要结合模型和任务观察。
零候选返回 `[]`，且不请求模型。

查询编码器的模型名称和查询说明必须与索引一致，问题向量也必须同维。
同名模型的实际权重若被替换，仍应主动重建索引；本练习不管理模型摘要或复杂版本迁移。
来源路径保持 Day2 的原值，回溯相对路径时仍需配合同一知识数据集根目录。

## 运行命令

均在仓库根目录执行，无新依赖。

### 1. 已提供代码的离线自检

```bash
.venv/bin/python -B -m pytest tests/test_rag_retrieval.py -q -s -p no:cacheprovider
```

默认执行保存、加载、错误校验及 Day3 导入测试；完整检索流程和真实模型测试需通过后面的环境变量显式启用。

### 2. 运行完整离线验收

```bash
RUN_WEEK7_DAY4=1 RUN_WEEK7_DAY4_MODEL=0 \
.venv/bin/python -B -m pytest tests/test_rag_retrieval.py -q -s -p no:cacheprovider
```

只使用固定向量，不请求 Ollama。新进程测试也使用测试侧的固定问题向量。
可加 `-k filters` 查看候选筛选测试，或 `-k fresh_process` 单独验证重启。

### 3. 复用 Day3 真实向量建立本地索引

```bash
.venv/bin/python -B -m rag.retrieval build
```

不调用模型，将 Day3 已有真实向量导入并保存为：
`rag/artifacts/day4/local_model/knowledge_index.json`。
如果 Day3 产物位于其他目录，使用 `--day3-dir /实际目录`。

### 4. 完成检索步骤后，真实模型查询

确认 Ollama 服务已启动且仍有索引记录的模型，再执行：

```bash
.venv/bin/python -B -m rag.retrieval search \
  --query '任务查不到时应该先检查什么？' \
  --knowledge-status current \
  --k 3
```

该命令每次启动一个新进程，加载索引，并只为新问题调用一次 Ollama。
结果写入 `rag/artifacts/day4/local_model/search_results.json`。
需要明确流程时添加 `--engine-code import_std`；服务地址和请求超时可通过 `--ollama-url`、`--timeout` 设置。

真实 pytest 验收命令：

```bash
RUN_WEEK7_DAY4=1 RUN_WEEK7_DAY4_MODEL=1 \
.venv/bin/python -B -m pytest tests/test_rag_retrieval.py -q -s \
  -k real_ollama_query -p no:cacheprovider
```

该测试导入 Day3 真实向量、保存索引，再通过新进程 CLI 查询。可选环境变量：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `RAG_DAY4_OUTPUT` | 仓库下的 `rag/artifacts/day4` | pytest 的固定产物根目录 |
| `RAG_DAY3_MODEL_OUTPUT` | 仓库下的 `rag/artifacts/day3/local_model` | 真实测试读取的 Day3 产物目录 |
| `OLLAMA_URL` | `http://localhost:11434` | 真实测试使用的服务地址 |
| `OLLAMA_EMBED_TIMEOUT` | `120` | 真实查询请求超时秒数 |

模型名称与查询说明直接来自索引，不另外设置一个可能冲突的模型名。
以上环境变量作用于 pytest；CLI 使用对应的命令行参数。
固定产物目录重新运行会覆盖同名文件；`rag/artifacts/` 已被 Git 忽略。

## 测试一：索引保存与加载

函数：`test_provided_index_round_trip_preserves_sources`，默认执行。

- **数据来源**：测试侧构造的四条合成知识及二维向量。
- **流程**：保存 JSON → 加载 → 对照原始索引，再修改加载结果。
- **检查点**：正文、向量、模型配置、元数据及 JSONPath 完整；中文可读；修改副本不影响原对象。
- **输出**：pytest 临时目录的 `knowledge_index.json`。

## 测试二：损坏索引不能静默加载

函数：`test_provided_load_rejects_corrupt_index`，默认执行。

- **数据来源**：合法临时索引，逐项破坏版本、维数、向量、ID 或来源字段。
- **流程**：改写临时副本后加载。
- **检查点**：未知格式版本、bool 维数、缺字段、零向量、维数不齐和重复 ID 均被拒绝。
- **输出**：pytest 结果；仅修改临时文件。

## 测试三：文件与 JSON 错误

函数：`test_provided_load_reports_missing_file_and_invalid_json`，默认执行。

- **数据来源**：不存在的路径、损坏的 JSON 字符串。
- **流程**：调用加载函数。
- **检查点**：保留 `FileNotFoundError` 和 `JSONDecodeError`，不返回半份索引。
- **输出**：pytest 结果。

## 测试四：保存前先检查数据

函数：`test_provided_save_validates_before_overwriting`，默认执行。

- **数据来源**：已有临时文件及 Chunk/向量数量不一致的索引。
- **流程**：尝试保存非法索引。
- **检查点**：报错后原文件内容保持不变。
- **输出**：pytest 临时文件与断言结果。

## 测试五：导入 Day3 格式产物

函数：`test_provided_import_day3_vectors_without_reembedding`，默认执行。

- **数据来源**：测试侧模拟的 Day3 报告及向量文件，不是真实模型输出。
- **流程**：读取两份文件并恢复索引。
- **检查点**：模型配置、正文和向量一一对应；导入过程不需要编码器。
- **输出**：pytest 结果。

## 测试六：Day3 文件一致性

函数：`test_provided_import_rejects_inconsistent_day3_files`，默认执行。

- **数据来源**：临时 Day3 格式产物，修改模式、条数、维数或删除查询说明。
- **流程**：执行导入。
- **检查点**：拒绝人工演示模式及不完整、不一致的报告。
- **输出**：pytest 结果。

## 测试七：索引导入命令

函数：`test_provided_build_command_imports_day3_files`，默认执行。

- **数据来源**：临时 Day3 格式合成文件。
- **流程**：在新 Python 进程执行 `rag.retrieval build`，再加载输出索引。
- **检查点**：命令成功，内容与输入一致，不连接模型服务。
- **输出**：pytest 临时目录的 `knowledge_index.json`。

## 测试八：新知识批量建库

函数：`test_build_encodes_documents_once_and_copies_chunks`。

- **数据来源**：四个合成 Chunk 和 `FixedEncoder`。
- **流程**：调用你补全的 `build_index`。
- **检查点**：正文按顺序一次编码；索引配置正确；嵌套元数据独立复制。
- **输出**：pytest 结果。

## 测试九：空索引

函数：`test_build_empty_index_never_requests_model`。

- **数据来源**：空 Chunk 列表和固定编码器。
- **流程**：建库 → 保存 → 加载。
- **检查点**：空索引可往返，维数为 0，不请求模型。
- **输出**：临时 `empty.json`。

## 测试十：检索输出结构与来源

函数：`test_search_returns_scores_original_chunks_and_sources`。

- **数据来源**：已知四条索引及固定问题向量。
- **流程**：不加过滤查询 Top-3，并修改返回结果。
- **检查点**：排名和分数符合手算；返回完整原 Chunk；只编码问题一次；结果修改不影响索引。
- **输出**：pytest 结果。

## 测试十一：先过滤再排名、映射原始下标

函数：`test_search_filters_before_ranking_and_maps_original_indices`。

- **数据来源**：高分但流程或版本不适用的 D0/D1，以及真正匹配条件的 D2/D3。
- **流程**：分别按流程、状态、版本或组合条件查询；关键场景只取 Top-1。
- **检查点**：先筛候选；候选下标 0 映射到原下标 2，正确返回 D2；K 超过候选数时不补入其他记录。
- **输出**：pytest 结果。

## 测试十二：没有候选时不调用模型

函数：`test_no_candidates_and_empty_index_never_request_model`。

- **数据来源**：无精确流程匹配的过滤条件、空索引。
- **流程**：执行检索并查看模型替身调用记录。
- **检查点**：返回 `[]`，调用记录为空；`import` 不能子串匹配 `import_std`。
- **输出**：pytest 结果。

## 测试十三：分数门槛

函数：`test_score_threshold_does_not_pad_results`。

- **数据来源**：适用候选得分为 0.8 和 0.0，配合四组门槛。
- **流程**：过滤后检索并应用 `min_score`。
- **检查点**：门槛包含等值边界；结果不足 K 时直接少返回，全部低于门槛则返回空列表。
- **输出**：pytest 结果。

## 测试十四：非法查询配置

函数：`test_invalid_search_options_never_request_model`。

- **数据来源**：非法 K、未知过滤字段、空过滤值、非法状态、NaN 和越界门槛。
- **流程**：尝试检索。
- **检查点**：抛 `ValueError`，不请求模型。
- **输出**：pytest 结果。

## 测试十五：模型配置与维数一致性

函数：`test_model_configuration_and_query_dimensions_must_match`。

- **数据来源**：修改了模型名或查询说明的编码器，以及三维问题向量。
- **流程**：在二维索引上发起查询。
- **检查点**：模型配置不同在请求前拒绝；问题向量维数不同在排名前拒绝。
- **输出**：pytest 结果。

## 测试十六：离线完整流程

函数：`test_offline_build_save_reload_and_search`。

- **数据来源**：四条合成 Chunk、固定正文与问题向量。
- **流程**：建库 → 保存 → 加载 → 过滤检索 → 导出。
- **检查点**：命中 D2；正文编码一次、问题编码一次；来源与模型配置完整。
- **输出**：固定目录 `offline_fixture/` 下的 `knowledge_index.json`、`search_results.json`、`day4_report.json`。

## 测试十七：真正的新进程重启

函数：`test_fresh_process_loads_index_and_searches_without_reembedding`。

- **数据来源**：保存后的临时索引与子进程内固定问题向量。
- **流程**：启动新 Python 解释器，只从文件加载索引后检索。
- **检查点**：仍返回 D2 和 0.8 分；子进程若尝试编码正文会主动报错。
- **输出**：pytest 临时目录的 `knowledge_index.json` 和 `fresh_process_results.json`；`-s` 可查看目录。

## 测试十八：复用 Day3 向量的真实查询

函数：`test_real_ollama_query_uses_saved_day3_vectors`，需另行显式启用。

- **数据来源**：Day3 真实本地模型产物、固定中文问题、本地 Ollama。
- **流程**：导入并保存索引，再由新进程 CLI 编码一个问题并查询。
- **检查点**：返回三个当前版本片段，分数降序且合法，保留原文及来源；不强制不同模型得到同一排名。
- **输出**：固定目录 `local_model/` 下的 `knowledge_index.json`、`search_results.json`。

完成自动测试后，人工从一个命中结果找到原文，并说明它为何适用于当前问题。Day4 的结构与持久化验收通过，不等于完成 Day6 的召回质量评估。
