# Week9：RAG 质量优化与评估

> 基准计划：`/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md`。
>
> 当前进度（2026-11-17）：**Day1、Day2 已完成，下一阶段为 Day3：Hybrid Search**。Week7、Week8 已完成。
>
> 默认使用主项目 `.venv/bin/python`；实现放在 `rag/`，测试放在 `tests/`。

## 本周安排

| 日期 | 内容 | 主要产物 |
|---|---|---|
| **Day1（已完成）** | 固定评估集，明确答案依据与拒答边界 | 30 道题、知识快照、运行输入与评分参考 |
| **Day2（已完成）** | Recall@K、MRR 与召回错误分析 | 有分母的检索基线 |
| Day3 | 关键词与向量结合的 Hybrid Search | 精确标识符与语义检索对照 |
| Day4 | 重排、去重和 Context 预算 | 可追溯且集中的证据 |
| Day5 | 生效版本、审核状态和访问权限过滤 | 版本与越权专项测试 |
| Day6 | 检索、引用、回答、拒答分别评估 | 错误归因与人工复核 |
| Day7 | 基础方案与优化方案对照 | 质量、成本与延迟报告 |

<a id="day1"></a>

## Day1：学习内容与验收目标（已完成）

### 完成记录（2026-11-17）

- `reference_is_eligible` 已实现：已发布状态、每个精确过滤条件与全部所需权限同时满足。
- `build_case_input` 已实现：只提取运行 input，并深拷贝嵌套对象，评分参考不会混入请求。
- 主项目 `.venv` 完整离线回归为 **49 passed**；没有调用 Embedding 或回答模型。
- 固定评估集为 **30 题、21 份知识、23 个正文单元、30 条原文证据**，覆盖五类问题；22 题预期回答，8 题预期信息不足。
- 数据 fingerprint：`31c20e959a1b4f0cf11af6a0c3ea5776640774f5446d2970994737c45a542143`。
- 数据与验收产物位于 `/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality/`，其中 `runs/day1/` 保存数据报告和运行请求；外部数据与运行产物不纳入主项目提交。

通过范围为数据契约与参考适用性。实际检索质量、回答语义和 Retriever 权限执行仍在后续阶段验收。

今天建立后续六天共用的评估依据。重点是把“问题、可用证据、答案要求、拒答条件”写清楚，
而不是直接增加 Top-K 或根据模型回答反过来修改参考答案。

| 学习点 | 今日验收 |
|---|---|
| 固定问题覆盖 | 30 道题，覆盖常见、易混淆、未知、历史与权限五类 |
| 可追溯参考 | 有依据回答题包含文档组、实际正文摘录、可接受答案要点 |
| 明确拒答 | 未知、实时、未发布和权限不足分别记录原因及拒答条件 |
| 版本与访问适用性 | 参考证据满足本题版本、状态与可信访问要求 |
| 评估参考隔离 | 运行输入不带标准答案、参考摘录或评分要求，嵌套对象也独立复制 |
| 基线可复现 | 为题库与知识文件记录 SHA256，整体 fingerprint 随输入变化 |

**本日全部离线，不使用 Ollama 或回答模型，不需要新增三方库。**
Day1 通过表示题库结构与参考契约可用；检索指标、真实回答和 Retriever 权限执行分别在后续阶段验证。

### 文件与数据来源

```text
主项目/
├── rag/eval_dataset.py           # 两个关键学习步骤已完成
├── rag/week9.md                  # 本文档
└── tests/test_week9_day1.py       # 测试已全部提供，无需填写

/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality/
├── README.md                    # 题目目录、来源和数据约定
├── manifest.json                # 21 份知识的元数据与文件哈希
├── knowledge/                   # 仅这个目录的清单内文件作为知识正文
│   ├── KB001～KB018              # Week7 原文件副本，字节不变
│   ├── KB019.md                 # ops 才能访问的合成内部演练顺序
│   ├── KB020.md                 # 未发布草案，draft
│   └── KB021.md                 # 已撤销说明，retired
├── eval_cases.json              # 固定 30 题及 expected；禁止入库
└── runs/day1/                   # 完整验收通过后才生成
    ├── dataset_report.json      # 数据校验结果、分类数、哈希和 fingerprint
    └── runtime_inputs.json      # 仅 case_id 与运行 input
```

原始来源：`/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics/`。
原数据不修改；新增三份文档与全部问题均为教学合成内容，不是业务事实或真实权限配置。
18 份旧知识加 3 份新知识，共 21 份文件；KB012 的 JSON 含三个正文单元，因此共加载 23 个正文单元。
现有 Week7/Week8 索引不修改，Week9 数据后续需要单独建索引，不能直接复用缺少新增文档的旧缓存。

### 30 道题的分布

| 分类 | 数量 | 用例 ID | 重点 |
|---|---:|---|---|
| common | 8 | W9D1-01～08 | 当前 SOP、字段、limit、重试和导出要求 |
| ambiguous | 8 | W9D1-09～16 | NOT_FOUND/FORBIDDEN/FAILED、相似编码、部分结果与子串误判 |
| unknown | 5 | W9D1-17～21 | 缺少版本、实时状态/列表、未规定 SLA、未发布规范 |
| historical | 4 | W9D1-22～25 | 回溯 1.0 版，并分别引用新旧版做对照 |
| permission | 5 | W9D1-26～30 | 未授权、已授权、正文伪造身份、公开规则与错误权限组 |

22 题预期 `answered`，8 题预期 `insufficient`。
W9D1-26、27、30 的问题完全相同，可信权限分别为空、ops、support；这使权限差异可以单独检验。
W9D1-28 则在正文自称拥有 ops，但可信上下文没有授予，仍应拒答。

### 输入与评分参考的格式

每条用例只有四个顶层字段：`case_id`、`category`、`input`、`expected`。

| 位置 | 含义 | 能否送入运行流程 |
|---|---|---|
| input.question | 实际问题 | 用于检索和回答 |
| input.filters | 明确的流程、版本或知识状态条件，允许 `{}` | 程序用于匹配，不是标准答案 |
| input.access_context | subject_id 与 granted_scopes | 模拟可信程序授权，由程序使用 |
| expected.status | answered 或 insufficient | 仅评分 |
| expected.doc_groups | 预期答案文档组；组间均需命中，组内可替代 | 仅评分 |
| expected.evidence | doc_id、segment_id、逐字 quote | 仅评分；不是预先喂给模型的证据 |
| expected.acceptable_points | 应涵盖的语义要点，不要求逐字复述 | 仅评分与人工复核 |
| expected.refusal_condition / refusal_reason | 何时拒答及拒答原因，回答题为 null | 仅评分 |
| expected.forbidden_doc_ids | 本题不能用于回答的来源 | 仅评分；不能当作运行过滤的答案提示 |

拒答题没有目标答案文档时，`doc_groups=[]`。可以保存公开边界说明作为参考 evidence，
例如“静态知识不能确认当前任务状态”；它不是该任务当前状态的目标答案文档。
后续这种题的 Recall/MRR 应标为不适用，不能计成召回满分。

`scope` 是沿用的主题/适用范围标签；`required_scopes` 是本练习新增的文档访问要求。
访问要求采用“全部满足”：要求 `['ops', 'audit']`，仅有 `['ops']` 不够；要求为空表示公开。
问题中的自我授权不改变 `access_context`。当前函数只检查参考数据的适用性，
尚未让既有检索器执行访问控制；Day5 会验证实际检索与上下文中的权限过滤。

### 已完成的两个关键步骤

实现文件：[eval_dataset.py](eval_dataset.py)。测试由助手提供，不需要重写。

| 标记 | 函数 | 输入 | 输出与关键原理 |
|---|---|---|---|
| W9D1-1 | reference_is_eligible | manifest 文档记录、case.input | bool；已发布状态、所有精确过滤条件、全部访问权限必须同时满足 |
| W9D1-2 | build_case_input | 完整 case | 独立深拷贝的 case.input；使用明确字段提取，隔离参考答案与嵌套可变对象 |

两个函数的 docstring 已写明字段、空值含义、边界与示例。
其余格式检查、清单加载、JSON 正文提取、哈希、证据校验和落盘均已实现。
默认测试保留基础自检；完整契约验收仍需显式开启环境变量，避免把默认跳过误认为完整通过。

### 测试代码的大体架构

`load_benchmark` 先校验题目结构；`read_manifest` / `load_documents` 复用现有知识加载器。
`validate_benchmark` 再核对覆盖、哈希、文档组、原文摘录与参考适用性。
最后 `write_dataset_report` 调用 `build_case_input` 导出运行输入和数据报告。

默认执行框架自检。设置 `RUN_WEEK9_DAY1=1` 才启用参考适用性、输入隔离和完整报告测试。
异常用例复制输入到 pytest 临时目录后再篡改，不修改正式数据。

### 测试一：30 题覆盖与答案契约

- **函数**：`test_provided_thirty_cases_cover_all_five_categories`。
- **数据来源**：正式数据集的 `eval_cases.json`。
- **目标与检查**：30 个唯一 ID、五类及其固定数量、22/8 的回答/拒答分布；每题有答案要点，回答题有证据，拒答题有条件与原因。
- **输出**：pytest 断言结果；不落盘。默认执行。

### 测试二：清单加载与原文证据

- **函数**：`test_provided_manifest_only_loads_knowledge_and_every_quote_exists`。
- **数据来源**：`manifest.json`、21 份 `knowledge/` 文件和用例 evidence。
- **目标与检查**：加载 23 个正文单元；逐文件 SHA256 正确；逐条摘录存在于指定 doc_id/segment_id；清单没有把题库和运行报告当成知识。
- **输出**：pytest 断言结果；不落盘。默认执行。

### 测试三：权限对照与草稿/失效反例

- **函数**：`test_provided_permission_pairs_and_unpublished_negative_documents`。
- **数据来源**：W9D1-02、21、26～30。
- **目标与检查**：同题不同权限有不同预期；未授权题的评分参考不包含受限正文；当前规则明确排除草稿与失效说明。
- **输出**：pytest 断言结果；不落盘。仅验证题目设计，默认执行。

### 测试四：非法输入结构

- **函数**：`test_provided_loader_rejects_invalid_contracts`，18 个参数用例。
- **数据来源**：复制题库到 `tmp_path`，单独构造一个无效字段。
- **目标与检查**：布尔 schema、重复 ID、空问题、错误分类、非法 filter、无效权限、缺失字段、评分字段进入 input、缺少文档组/拒答条件等均抛 `ValueError`。
- **输出**：pytest 临时目录中的 `eval_cases.json`；默认执行。

### 测试五：知识文件发生变化

- **函数**：`test_provided_validation_detects_changed_source_before_reference_checks`。
- **数据来源**：知识副本，向 KB001 追加文本但保留原哈希。
- **目标与检查**：在参考适用性检查前检测文件变化，不继续使用旧快照。
- **输出**：临时数据副本；默认执行。

### 测试六：报告目录边界

- **函数**：`test_provided_output_cannot_overwrite_input_directories`。
- **数据来源**：pytest 临时路径。
- **目标与检查**：拒绝将报告写到数据根目录或知识目录中，避免与输入混放或覆盖。
- **输出**：无文件；默认执行。

### 测试七：参考文档适用性

- **函数**：`test_reference_eligibility_uses_exact_filters_and_all_required_scopes`，15 个参数用例。
- **数据来源**：测试内的小字典，不需要模型或知识全文。
- **目标与检查**：精确过滤、current/historical、draft/retired、公开文档、错误权限组、多权限必须全满足、正文声明不能授权，以及输入不被修改。
- **输出**：pytest 断言结果；学习点 W9D1-1 已完成，启用 DAY1 后执行。

### 测试八：运行输入与评分参考隔离

- **函数**：`test_runtime_input_excludes_gold_and_has_independent_nested_values`。
- **数据来源**：W9D1-27 副本，额外放入仅供评分的哨兵文本。
- **目标与检查**：输出等于 input，字段严格受限；评分哨兵不泄露；修改 filters 或权限列表不影响原用例，再次调用也不受影响。
- **输出**：pytest 断言结果；学习点 W9D1-2 已完成，启用 DAY1 后执行。

### 测试九：错误参考答案的定位

- **函数**：`test_full_validator_rejects_inconsistent_gold`，9 个参数用例。
- **数据来源**：正式数据集的临时副本。
- **目标与检查**：虚构摘录、错误正文段、未知文档、允许/禁止冲突、版本错误、未授权引用、文档组缺少摘录、题数不足和漏分类均被拒绝。
- **输出**：临时数据副本；参考适用性已实现，启用 DAY1 后执行。

### 测试十：整集验收与产物保存

- **函数**：`test_complete_dataset_report_and_runtime_export`。
- **数据来源**：完整正式数据集；先写临时报告并断言成功，再保存固定产物。
- **目标与检查**：30 题、21 文档、分类/状态计数、来源与适用性校验、稳定 fingerprint；所有导出请求都不带 expected，原始题库不被修改，model_called 为 false。
- **输出**：`runs/day1/dataset_report.json`、`runs/day1/runtime_inputs.json`。
- **启用条件**：`RUN_WEEK9_DAY1=1`。

### 运行命令

从主项目根目录执行。默认先检查已提供的部分：

```bash
.venv/bin/python -B -m pytest tests/test_week9_day1.py -q -p no:cacheprovider
```

执行完整离线验收：

```bash
RUN_WEEK9_DAY1=1 .venv/bin/python -B -m pytest tests/test_week9_day1.py -v -s -p no:cacheprovider
```

只检查某一练习，例如参考文档适用性：

```bash
RUN_WEEK9_DAY1=1 .venv/bin/python -B -m pytest \
tests/test_week9_day1.py::test_reference_eligibility_uses_exact_filters_and_all_required_scopes \
-v -p no:cacheprovider
```

可选环境变量：`WEEK9_RAG_DATASET` 指定数据根目录，`WEEK9_DAY1_OUTPUT` 指定产物目录。
它们不改变题目内容或评分标准。本日没有真实模型测试开关。

### 验收边界与后续使用

本日校验能证明：题库数量和分类达标，评分参考可追溯，参考版本/权限条件一致，输入与参考分离。
不能证明：实际检索召回正确、答案语义正确、现有 Agent 已阻止越权。
答案要点与原文是否充分支持仍需阅读核对；程序只验证摘录存在，不会自动证明所有自然语言主张。

准备阶段的默认自检为 **23 passed、26 skipped**；完成两个步骤后的完整离线验收为 **49 passed**。
默认模式仍保留 26 个显式开启项，复验全部检查应使用上面的完整命令。
18 份旧知识副本已与 Week7 原文件逐字节核对；两次检查均未调用真实模型。

Day2 从这份冻结数据统计召回；Day3～Day5 调整检索与过滤；Day6、Day7 对比回答和质量。
如果发现题库本身标注有误，应先记录更正原因并重新冻结 fingerprint，再让各方案使用相同新版本，
不要为了某一次模型通过而悄悄修改评分参考。

<a id="day2"></a>

## Day2：Recall@K、MRR 与召回质量分析（已完成）

### 完成记录（2026-11-17）

- 已实现目标文档组 Recall@K、首个相关 Chunk 的 RR@K，以及批量 MRR 汇总和逐题诊断。
- Day2 完整离线验收为 **33 passed、1 skipped**；跳过项为需另行开启的真实模型测试。
- 收尾连同 Day1 回归为 **82 passed、1 skipped**，真实推理未在本次回归中重复执行。
- 用户已运行真实 Ollama 检索测试通过；助手从留存命中重新计算了每题指标与批次汇总，结果一致。
- 真实模型为 `qwen3-embedding:0.6b`，索引含 **70 个 Chunk**；30 题全部记录，执行错误 0 题，22 题适用、8 题不适用。
- 基线沿用 Day1 fingerprint：`31c20e959a1b4f0cf11af6a0c3ea5776640774f5446d2970994737c45a542143`。
- 本次收尾执行离线回归；真实结果来自用户运行后保存在外部数据目录的报告，没有调用回答型 LLM。

| K | 平均 Recall@K | MRR@K | 实际算分/计划适用 | 全部目标组命中题数 |
|---|---:|---:|---:|---:|
| 1 | 0.8182 | 0.8182 | 22/22 | 18 |
| 3 | 0.9773 | 0.9015 | 22/22 | 21 |
| 5 | 1.0000 | 0.9015 | 22/22 | 22 |
| 10 | 1.0000 | 0.9015 | 22/22 | 22 |

W9D1-25 在 K=3 时遗漏 KB003，K=5 才覆盖两个目标组。首次相关项的位置保持不变，因此 MRR 没有随之上升。
该题仍召回草稿 KB020 与失效说明 KB021；W9D1-26、28、30 仍召回无权访问的 KB019。
这些已定位问题留给后续状态和权限过滤优化，Day2 完成不代表权限隔离或回答语义验收完成。
K=10 在本批次没有增加目标文档组覆盖，不据此推断其他语料的最优 K。

真实报告位于 `/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality/runs/day2/local_model/`；
本提交保留上述摘要，向量、原始运行报告与外部固定样例不纳入主项目 Git 提交。

本日沿用 Day1 冻结的 30 道题，先建立检索基线，再定位遗漏和不适用来源。
不修改答案标准，不提高某一道题的 K 来隐藏失败，也不调用回答型 LLM。
两个关键计算已在 [retrieval_metrics.py](retrieval_metrics.py) 中实现，测试代码已完整提供。

### 学习目标与验收标准

| 学习内容 | 能展示的验收结果 |
|---|---|
| Recall@K 的含义 | 根据前 K 个 Chunk 判断目标文档组覆盖率；重复文档不能重复加分 |
| RR 与 MRR 的区别 | 单题计算首个相关项排名倒数；MRR 是多题 RR 的平均 |
| 排名单位和截断 | K 按 Chunk 排名计数，先取前 K 项，再按文档组计覆盖 |
| 有分母的统计 | 22 题有目标来源，8 题不适用；执行错误单独统计，不伪装成零分 |
| 召回诊断 | 找出遗漏文档组、非目标来源、禁止来源以及版本/权限不适用来源 |
| 固定基线对照 | 每题只查询一次 max(K)，使用同一排名计算 K=1、3、5、10 |

**本日没有召回率必须达到 100% 的门槛。**
离线验收要求指标算对、报告可追溯；真实检索先记录当前效果和失败样例，后续再优化。
只命中文档 ID 不能保证具体 Chunk 足以支持答案，回答语义支持在后续验收。

### 两个指标怎么理解

今天沿用 Week7 的“预期文档组”口径：`[[A, B], [C]]` 表示需要两类依据，
第一类命中 A 或 B 任意一个即可，第二类需要 C。

```text
Recall@K = 前 K 个 Chunk 覆盖的目标组数 / 目标组总数
RR@K     = 首个相关 Chunk 在前 K 名中的排名倒数；没有命中为 0
MRR@K    = 多道适用问题的 RR@K 平均值
```

例如目标是 `[[A], [C]]`，检索顺序是 `[X, A, A, C]`：

| 截断位置 | 目标覆盖情况 | Recall | RR |
|---|---|---:|---:|
| 前 1 个 Chunk | 没有命中 | 0 | 0 |
| 前 3 个 Chunk | 只覆盖 A 这一组 | 1/2 | 1/2 |
| 前 4 个 Chunk | 覆盖 A、C 两组 | 1 | 1/2 |

两个 A 表示同一文档的不同 Chunk，会占用两个排名位置；不能先去重成 `[X,A,C]` 再取前 3 项。
首次相关项在第 2 位，所以 RR 始终为 1/2；它不表示已覆盖全部需要的依据。
若两道题的 RR 分别是 1 和 1/2，则 MRR 为 0.75。

单题 `doc_groups=[]` 时，两个指标都返回 `None`，含义是不适用。
8 道拒答题仍会记录检索结果和禁止来源，但不计入 Recall/MRR 分母。
平均 Recall 是适用题各自 Recall 的算术平均，不是把所有题的目标组混成一个大分母。
请求失败也不能当成“已成功检索但没命中”：报告同时列出计划适用 22 题与实际算分题数。

### 代码、数据与产物

```text
主项目/
├── rag/retrieval_metrics.py        # 两个指标与评估流程已完成
├── rag/week9.md                   # 逐项说明
└── tests/test_week9_day2.py        # 无需填写测试

/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality/
├── manifest.json                 # Day1 冻结的清单，保持原样
├── eval_cases.json               # Day1 冻结的 30 题，保持原样
├── knowledge/                    # Day1 冻结的 21 份合成知识，保持原样
├── rankings_day2.json            # 新增：人工排名，只用于离线算例和诊断
├── README_day2.md                # 数据说明
└── runs/day2/
    ├── offline/
    │   ├── retrieval_report.json  # 人工排名下的指标、原请求、命中片段、错误
    │   └── retrieval_report.md    # 指标分母与逐题诊断表
    └── local_model/
        ├── retrieval_report.json # 真实 Ollama 检索基线
        ├── retrieval_report.md
        └── index/                # Week9 独立缓存，不覆盖 Week7/Week8 索引
            ├── knowledge_store.json
            ├── knowledge_store.config.json
            └── dataset_fingerprint.json
```

`rankings_day2.json` 全部手写，不是模型输出，也不会参与知识分块入库。
它故意包含错误与遗漏：W9D1-07 没有 KB005；W9D1-25 前三项只覆盖一组；
W9D1-02 包含旧版、草稿和失效文档；W9D1-26/28/30 包含无权访问的 KB019。
测试按这些文档 ID 从现有知识中取不同 Chunk 填充元数据，分数只是人工递减值，不代表语义相似度。

人工排名不改变 Day1 fingerprint；人工样例的 SHA256 会另行写入离线报告的 backend_info。
真实基线复用 Week8 的 ManifestLoader、分块、Week7Embeddings、prepare_store 和 make_retriever。
首次编码 Week9 正文并保存缓存，后续加载缓存只编码问题；快照不匹配会报错，不会静默重建。

### 已完成的关键代码

| 学习点 | 函数 | 已提供部分 | 已实现的关键步骤 |
|---|---|---|---|
| W9D2-1 | recall_at_k | 输入检查、前 K 项、命中文档集合、分母、无目标处理 | 逐组判断是否命中，计算 matched_groups |
| W9D2-2 | reciprocal_rank_at_k | 输入检查、相关文档集合、前 K 项、无目标处理 | 找到首个相关项，按从 1 开始的排名返回倒数；没找到为 0 |

两个函数都有完整的输入输出类型、示例和空列表语义；不需要实现文件读取、HTTP 或整个评估循环。
`summarize_rows` 已实现均值与分母统计，`run_retrieval_evaluation` 已实现逐题记录与阶段异常保存。

### 测试架构与报告边界

数据加载 → 每题提取纯运行 input → retrieve(request, max_k) → 保留原排名 →
计算各 K 的 Recall/RR → 参考适用性诊断 → 汇总 MRR 与分母 → 保存 JSON/Markdown。

检索回调只接收 question、filters、access_context，不接收标准答案、摘录、禁止文档列表或 case_id。
参考项只在检索完成后评分。默认模式保留框架自检，设置 `RUN_WEEK9_DAY2=1` 才执行完整离线验收。

本日真实基线仍是 Week8 的精确 filters，没有新增权限过滤；`access_context` 暂用于评分诊断。
Day1 的 `reference_is_eligible` 在这里检查已召回来源，不改变排名。
报告中的 `ineligible_doc_ids` 和 `forbidden_doc_ids` 用于暴露问题，不能被解释成问题已经被阻止。
`non_target_doc_ids` 只表示不在目标组，可能是有用背景，不能直接当作语义错召回。

### 测试一：固定排名样例与 Day1 快照

- **函数**：`test_provided_rankings_cover_thirty_frozen_cases_and_include_failure_examples`。
- **数据来源**：`rankings_day2.json`、Day1 原知识与题库。
- **目标与检查**：30 道题 ID 全覆盖、文档存在、重复/缺失/越权样例存在，Day1 fingerprint 保持不变。
- **输出**：pytest 断言结果，不落盘；默认执行。

### 测试二：指标参数边界

- **函数**：`test_provided_metric_inputs_reject_invalid_k_groups_and_rankings`，12 个参数用例。
- **数据来源**：测试内的小列表与非法参数。
- **目标与检查**：拒绝零/负数/布尔/字符串 K、空文档组、重复目标组、错误排名类型。
- **输出**：pytest 断言结果；默认执行。

### 测试三：不适用与零分的区别

- **函数**：`test_provided_no_gold_is_not_applicable_even_with_returned_chunks`。
- **数据来源**：无目标组的小列表，包括已有背景返回项的情况。
- **目标与检查**：有无返回片段都不改变无目标组为 None，不能强制计成 0 或 1。
- **输出**：pytest 断言结果；默认执行。

### 测试四：批量均值与统计分母

- **函数**：`test_provided_summary_keeps_denominators_errors_and_none_separate`。
- **数据来源**：人工评分行，含两条有效分数、一条不适用和一条超时。
- **目标与检查**：均值为 0.75，实际算分 2/计划适用 3，错误独立统计；没有有效分数时均值保留 None。
- **输出**：pytest 断言结果；默认执行。

### 测试五：遗漏与禁止来源诊断

- **函数**：`test_provided_diagnostics_expose_misses_and_forbidden_sources`。
- **数据来源**：W9D1-25、26 的人工排名和真实知识元数据。
- **目标与检查**：定位缺少 KB003；未授权题明确记录 KB019 为禁止且不适用来源；拒答题不把所有背景都判为错召回。
- **输出**：pytest 断言结果；默认执行。

### 测试六：损坏的检索结果

- **函数**：`test_provided_bad_retrieval_results_are_rejected`，4 个参数用例。
- **数据来源**：人工命中副本，分别注入未知 doc_id、重复 Chunk ID、篡改版本和 NaN 分数。
- **目标与检查**：拒绝无法追溯或与清单不一致的数据，保留合法的重复文档不同 Chunk。
- **输出**：pytest 断言结果；默认执行。

### 测试七：请求失败的阶段记录

- **函数**：`test_provided_retrieval_errors_are_saved_without_faking_zero_metrics`。
- **数据来源**：30 道题与始终抛 TimeoutError 的回调，不访问网络。
- **目标与检查**：失败后继续记录，30 条错误、0 条实际算分，均值为 None；保存异常类型和阶段，不保存异常正文。
- **输出**：pytest 临时目录的 `retrieval_report.json`、`retrieval_report.md`；默认执行。

### 测试八：人可读报告

- **函数**：`test_provided_report_shows_mrr_denominator_and_diagnostic_columns`。
- **数据来源**：未完成批次的最小报告字典。
- **目标与检查**：显示 MRR、0/22 分母、不适用 8 题、遗漏/禁止/不适用来源等列，不伪装成完整成绩。
- **输出**：内存中的 Markdown；默认执行。

### 测试九：复用既有 LangChain 组件与缓存

- **函数**：`test_provided_langchain_cache_is_reused_and_stale_snapshot_is_rejected`。
- **数据来源**：Day1 知识与人工二维编码器，在临时目录建库。
- **目标与检查**：首次只批量编码正文，加载不重编码，查询只编码问题，快照不匹配显式报错。
- **输出**：pytest 临时目录的索引和快照文件；默认执行，使用现有依赖，无真实模型请求。

### 测试十：手算 Recall/RR 对照

- **函数**：`test_metrics_match_hand_calculated_examples_without_mutating_inputs`，9 个参数用例。
- **数据来源**：测试内的 A/B/C/X 列表。
- **目标与检查**：头部命中、K 外命中、多组覆盖、替代文档、重复文档占位、空结果和无目标；输入不修改。
- **输出**：pytest 断言结果；两个指标已实现，开启 DAY2 后执行。

### 测试十一：30 题完整离线报告

- **函数**：`test_offline_thirty_case_baseline_uses_one_ranking_and_excludes_gold`。
- **数据来源**：Day1 题库与 `rankings_day2.json`，不调用模型。
- **目标与检查**：每题仅一次 max(K) 检索，运行输入无评分参考，22/8 分母正确；检查 W9D1-25 的 0.5→1 覆盖变化和 W9D1-07 的真实遗漏记录。
- **输出**：`runs/day2/offline/retrieval_report.json`、`retrieval_report.md`；开启 DAY2 后执行。
- **限制**：人工排名只证明计算和报告流程，不代表真实检索质量。

### 测试十二：真实 Ollama 检索基线

- **函数**：`test_real_ollama_records_thirty_case_retrieval_baseline`。
- **数据来源**：同一知识和 30 道题，使用已安装的 `qwen3-embedding:0.6b`。
- **目标与检查**：实际建库/加载，逐题查询并记录四个 K 的成绩与诊断，执行错误不能通过；不要求召回满分。
- **输出**：`runs/day2/local_model/retrieval_report.json`、`retrieval_report.md` 及独立 `index/`。
- **启用条件**：DAY2=1 且 MODEL=1；不会在普通测试中联网，不下载模型，不需要 API Key。

### 运行命令

主项目根目录下，先检查已提供的部分：

```bash
.venv/bin/python -B -m pytest tests/test_week9_day2.py -q -p no:cacheprovider
```

运行全部离线测试：

```bash
RUN_WEEK9_DAY2=1 RUN_WEEK9_DAY2_MODEL=0 \
.venv/bin/python -B -m pytest tests/test_week9_day2.py -v -s -p no:cacheprovider
```

真实检索（先启动 `ollama serve`，使用已有模型）：

```bash
RUN_WEEK9_DAY2=1 RUN_WEEK9_DAY2_MODEL=1 \
.venv/bin/python -B -m pytest \
tests/test_week9_day2.py::test_real_ollama_records_thirty_case_retrieval_baseline \
-v -s -p no:cacheprovider
```

可选环境变量：`WEEK9_RAG_DATASET` 指定数据目录，`WEEK9_DAY2_OUTPUT` 指定产物根目录。
确认需要重建时显式设置 `WEEK9_DAY2_REBUILD=1`；默认缓存不匹配会报错，避免混用旧数据。
真实调用可能随缓存状态、候选数量变化；无候选时既有 Retriever 会跳过问题编码。

### 验证记录与边界

准备阶段默认自检为 **23 passed、11 skipped**；完成两个指标后的完整离线验收为 **33 passed、1 skipped**。
默认模式仍主动跳过扩展验收，复验时应使用上面的完整离线命令。
真实 Ollama 的 1 项基线测试由用户单独运行通过，已核对本地报告；本次收尾不重复执行真实推理。
学习完成范围为指标、基线和问题定位，后续继续验证检索优化、证据支持和访问隔离。
