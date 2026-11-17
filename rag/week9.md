# Week9：RAG 质量优化与评估

> 基准计划：`/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md`。
>
> 当前进度（2026-11-17）：**Day1～Day3 已完成，下一阶段为 Day4：重排与 Context 组装**。Week7、Week8 已完成。
>
> 默认使用主项目 `.venv/bin/python`；实现放在 `rag/`，测试放在 `tests/`。

## 本周安排

| 日期 | 内容 | 主要产物 |
|---|---|---|
| **Day1（已完成）** | 固定评估集，明确答案依据与拒答边界 | 30 道题、知识快照、运行输入与评分参考 |
| **Day2（已完成）** | Recall@K、MRR 与召回错误分析 | 有分母的检索基线 |
| **Day3（已完成）** | 关键词与向量结合的 Hybrid Search | 精确标识符与语义检索对照 |
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

<a id="day3"></a>

## Day3：关键词与向量结合的 Hybrid Search（已完成）

### 完成记录（2026-11-17）

- 已完成查询词项覆盖计分、RRF 排名贡献、同路去重与跨路累加，并保留各路排名和原始分数。
- 精确标识符约束同时作用于关键词和混合策略的向量候选，离线反例验证相似编码不会在融合时重新带回。
- 最近一次 Day1～Day3 联合离线验收为 **106 passed、2 skipped**；跳过项是 Day2、Day3 的真实 Ollama 测试。
- 已核对 Day3 的 dense、keyword、hybrid 三份人工向量报告：各记录 30 题，22 题适用、8 题不适用，执行错误均为 0。
- 原 Day1 知识与题库 fingerprint 保持为 `31c20e959a1b4f0cf11af6a0c3ea5776640774f5446d2970994737c45a542143`，Day2 基线报告保持原样。
- 验收产物统一保存到外部数据目录 `runs/day3/offline/`，原始报告、人工反例和向量不纳入主项目 Git 提交。

本次按离线验收标记学习完成。尚未提供真实 Ollama 三策略结果，默认 `runs/day3/local_model/` 下未找到对应报告，
因此不记录真实语义检索提升、性能优势或权限隔离已通过的结论。后续仍可使用保留的真实测试命令补充对照。

今天把关键词检索与既有向量检索组合起来，重点观察错误码和流程编码是否精确匹配。
沿用 Day1 的 30 题、同一知识快照和 Day2 指标；不修改原题答案，也不覆盖 Day2 基线报告。
两个计算片段已在 [hybrid_search.py](hybrid_search.py) 中完成，测试代码已完整提供。

### 学习内容与验收标准

| 学习点 | 可以展示的验收结果 |
|---|---|
| 关键词与语义检索各自的作用 | 对照 dense、keyword、hybrid 三种策略，解释词项匹配与向量相似度的不同 |
| 标识符的精确匹配 | import_std、import_standard、import_std_extra 分开；NOT_FOUND 与 NOT_FOUND_EXTRA、not_found 分开 |
| 词项覆盖分数 | 用查询词项与正文词项的交集计算比例，零命中不返回 |
| 排名融合 | RRF 使用两路排名计算融合分数，保留各路位置与原分数 |
| 候选约束的执行位置 | 标识符规则同时约束关键词和向量候选，避免融合时带回不匹配来源 |
| 固定集效果对照 | 30 题按相同 K=1、3、5、10 记录 Recall/MRR、分母与诊断，不预设混合策略必胜 |

本日使用轻量词项覆盖，中文连续文本取二字片段，英文标识符保留完整原值及大小写。
它不是 BM25，不额外安装分词器、数据库或新本地模型。
RRF 公式参考 [Elastic 官方说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)，当前融合实现使用 Python 标准库。

### RRF 与前一天的 RR 有什么不同

| 名称 | 用途 | 计算依据 | 是否读取评分参考 |
|---|---|---|---|
| RR / MRR | 评估检索结果是否把相关项排在前面 | 首个相关项的倒数排名及多题均值 | 是，检索完成后评分 |
| RRF（Reciprocal Rank Fusion） | 将多路候选排成一份结果 | 各路排名贡献之和 | 否 |

```text
关键词分数 = 命中的查询词项数 / 查询词项总数
RRF 分数   = 各路 1 / (rank_constant + 本路排名) 的和
```

本路排名从 1 开始，默认 `rank_constant=60`；同一 Chunk 两路都排第一时，分数为 `1/61 + 1/61`。
关键词分数、余弦相似度与 RRF 分数含义不同，不能直接相加，也不能跨策略按分数大小判优。
RRF 最终按融合分数降序，同分按 Chunk ID 升序，保留原始排名和分数供排查。

### 混合策略流程

```mermaid
flowchart TD
    I[运行 input：问题、filters、可信上下文] --> F[精确 filters 与标识符候选约束]
    F --> V[向量检索：候选 20 项]
    F --> W[词项检索：候选 20 项]
    V --> R[RRF：按 Chunk ID 累加排名贡献]
    W --> R
    R --> K[截取前 K 个 Chunk]
    K --> E[复用 Day2 指标与来源诊断]
```

同一路重复出现同一 Chunk 只贡献一次，跨路同一 Chunk 累加；同文档的不同 Chunk 保持独立。
去掉一路中的重复项后，不重新编号后面原有的排名位置。

`filters.engine_code` 有明确值时优先使用；省略时从问题原文提取本语料的 `import_*`、`export_*` 编码。
比较多个流程时允许其并集，不自行选一个。错误码、字段名等正文标识符使用完整 ASCII 词项匹配，
多标识符问题允许不同片段分别命中。这里是有限的确定性规则，不是通用的业务意图识别器。

纯 dense 保留 Day2 的过滤基线；keyword/hybrid 额外执行标识符约束。
因此三组效果差异可能来自候选约束、关键词或融合，不能把全部提升归因于 RRF。
访问权限与草稿/失效文档过滤仍留待 Day5；本日不新增 Agent 或修改通用 Instructions。

### 文件与数据来源

```text
主项目/
├── rag/hybrid_search.py              # 两个计算与候选准备、对照均已完成
├── rag/week9.md                      # 本说明
└── tests/test_week9_day3.py           # 测试完整，无需填写

/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality/
├── manifest.json、knowledge/、eval_cases.json  # 原样沿用 Day1
├── identifier_examples_day3.json     # 新增：8 个合成内存 Chunk、9 个标识符反例
├── README_day3.md
└── runs/day3/
    ├── offline/
    │   ├── comparison_report.json、comparison_report.md
    │   ├── dense/retrieval_report.json、retrieval_report.md
    │   ├── keyword/retrieval_report.json、retrieval_report.md
    │   └── hybrid/retrieval_report.json、retrieval_report.md
    └── local_model/                  # 真实三策略对照采用相同布局
```

反例的 `synthetic/` 路径是模拟来源，只用于测试内存对象，未加入清单，也不混入 30 题指标分母。
其中相似流程文档正文故意提到 import_std，用来证明正文提及不能替代 metadata.engine_code。
纯向量替身故意把 import_standard 排在 import_std 前面，用于检查标识符约束是否真的执行。

真实测试优先加载 `runs/day2/local_model/index/` 的同一向量快照；不存在时在 Day3 独立目录建库。
复用 Day2 的 fingerprint/模型配置检查，加载不会重编码正文。
`SharedQueryEncoder` 让同次对照复用同一问题向量；记录中的耗时受缓存影响，不能作为严格性能结论。
普通测试只使用人工向量，真实 Ollama 需另行显式开启；没有回答型 LLM 测试，不需要 API Key。

### 已完成的两个关键步骤

| 学习点 | 函数 | 已提供 | 已完成的计算 |
|---|---|---|---|
| W9D3-1 | keyword_search | query_tokens、document_tokens、零分处理与排序 | 词项集合交集占查询词项总数的比例 |
| W9D3-2 | reciprocal_rank_fusion | 分支遍历、从 1 开始的 rank、按 ID 累加、去重和排序 | 当前分支的倒数排名 contribution |

两个函数均已写明输入输出、边界和数值示例；不需要自行编写文件读取、模型客户端或整个检索器。

### 测试一：完整词项与中文切片

- **函数**：`test_provided_tokenizer_preserves_complete_identifier_and_chinese_bigrams`。
- **数据来源**：测试内字符串，含 NOT_FOUND_EXTRA、下划线、前缀、中文与版本号。
- **目标与输出**：验证标识符不被拆成子串、大小写不归一化、中文二字片段和版本词项保留；仅 pytest 断言，默认执行。

### 测试二：9 个标识符反例

- **函数**：`test_provided_nine_identifier_examples_use_values_not_substrings`。
- **数据来源**：`identifier_examples_day3.json` 的全部 9 个请求与 8 个合成候选。
- **目标与输出**：精确流程、明确 filter 优先、未知流程、错误码前后缀与大小写、多错误码比较、字段名；输入不修改，仅 pytest 断言，默认执行。

### 测试三：关键词参数校验

- **函数**：`test_provided_keyword_rejects_invalid_k_before_scoring`，4 个参数用例。
- **数据来源**：测试内非法 K：0、负数、布尔和字符串。
- **目标与输出**：在关键词计分前抛 ValueError；默认执行，不调用模型。

### 测试四：空关键词与重复 Chunk

- **函数**：`test_provided_empty_keyword_and_duplicate_chunks_are_handled`。
- **数据来源**：标点问题、空候选，以及合成候选的重复 ID。
- **目标与输出**：空查询词项/空候选返回 []，重复 Chunk ID 被拒绝；仅 pytest 断言，默认执行。

### 测试五：RRF 参数校验

- **函数**：`test_provided_rrf_rejects_invalid_options`，5 个参数用例。
- **数据来源**：非法 k 和 rank_constant。
- **目标与输出**：要求正整数，拒绝布尔、非整数和非正值；仅 pytest 断言，默认执行。

### 测试六：空融合与无效原分数

- **函数**：`test_provided_empty_rrf_and_bad_raw_score`。
- **数据来源**：两路空排名、原始分数 NaN 的合成命中。
- **目标与输出**：空排名返回 []，非有限原分数被拒绝；仅 pytest 断言，默认执行。

### 测试七：共享问题向量

- **函数**：`test_provided_shared_query_encoder_reuses_vectors_without_aliasing`。
- **数据来源**：固定二维编码器，重复 q1 和新问题 q2。
- **目标与输出**：重复问题只编码一次，批量顺序保持，修改返回值不破坏缓存；记录内存调用列表，默认执行。

### 测试八：未知标识符短路

- **函数**：`test_provided_unknown_identifiers_short_circuit_both_branches`。
- **数据来源**：合成小向量库与不存在的 import_stdx 请求。
- **目标与输出**：keyword/hybrid 均返回 []，不请求编码器；仅 pytest 断言，默认执行。

### 测试九：非法策略配置

- **函数**：`test_provided_invalid_search_configuration_never_encodes`，3 个参数用例。
- **数据来源**：未知策略、非法候选数量/常数，以及最终 K 超出固定候选窗口。
- **目标与输出**：请求编码前失败，不因配置错误产生模型调用；仅 pytest 断言，默认执行。

### 测试十：索引加载与快照

- **函数**：`test_provided_store_loading_keeps_snapshot_and_document_vectors`。
- **数据来源**：Day1 原知识，人工向量及 pytest 临时索引。
- **目标与输出**：首次建库、后续加载无重复正文编码，Day1 fingerprint 不变；临时索引，不覆盖正式索引，默认执行。

### 测试十一：关键词分数和原文隔离

- **函数**：`test_keyword_coverage_scores_exact_terms_and_owns_returned_chunks`。
- **数据来源**：合成候选与查询“NOT_FOUND 含义”。
- **目标与输出**：手算命中比例为 0.5，NOT_FOUND_EXTRA 不因子串命中，返回元数据与原输入隔离；开启 DAY3 后执行。

### 测试十二：RRF 累加与同分顺序

- **函数**：`test_rrf_accumulates_ranks_and_keeps_original_scores`。
- **数据来源**：两路手写的三个候选排名与不同原始分数。
- **目标与输出**：双路支持项的分数等于排名贡献之和，保留各路 rank/raw_scores，同分按 ID 排序，不修改输入；开启 DAY3 后执行。

### 测试十三：RRF 去重与来源冲突

- **函数**：`test_rrf_deduplicates_each_branch_but_preserves_rank_positions`。
- **数据来源**：同路重复项、同文档不同 Chunk，以及相同 ID 不同原文的反例。
- **目标与输出**：同路不重复加分、后续排名不重新编号、不同 Chunk 保持独立，来源冲突被拒绝；开启 DAY3 后执行。

### 测试十四：向量结果不能绕过标识符约束

- **函数**：`test_guarded_hybrid_never_reintroduces_similar_engine_from_dense`。
- **数据来源**：人工向量故意把 import_standard 排在 import_std 前面的小库。
- **目标与输出**：dense 基线能暴露错召回，keyword 不编码问题，hybrid 最终只返回正确流程；保留融合分数类型，开启 DAY3 后执行。

### 测试十五：30 题三策略离线对照

- **函数**：`test_offline_thirty_case_three_strategy_reports_exclude_gold`。
- **数据来源**：Day1 知识与 30 道题，使用人工向量，不读取补充反例作为业务知识。
- **目标与输出**：90 次策略调用只接收纯 input，三策略分别记录 30 题、22/8 分母和错误数，保存完整报告树；开启 DAY3 后执行。
- **产物**：`runs/day3/offline/` 的三策略指标与 `comparison_report.json/md`；不能当真实语义检索成绩。

### 测试十六：真实 Ollama 三策略对照

- **函数**：`test_real_ollama_three_strategy_comparison`。
- **数据来源**：同一知识快照、题库与 `qwen3-embedding:0.6b`，优先复用 Day2 索引。
- **目标与输出**：三策略完整记录、无执行错误，保存各 K 的质量与来源诊断；不预设提升门槛，不调用回答模型。
- **产物**：`runs/day3/local_model/`；需同时显式开启 DAY3 和 MODEL。

### 运行命令

在主项目根目录检查已提供的框架：

```bash
.venv/bin/python -B -m pytest tests/test_week9_day3.py -q -p no:cacheprovider
```

运行全部离线验收：

```bash
RUN_WEEK9_DAY3=1 RUN_WEEK9_DAY3_MODEL=0 \
.venv/bin/python -B -m pytest tests/test_week9_day3.py -v -s -p no:cacheprovider
```

真实 Ollama 对照（先启动本地服务，使用已有模型）：

```bash
RUN_WEEK9_DAY3=1 RUN_WEEK9_DAY3_MODEL=1 \
.venv/bin/python -B -m pytest \
tests/test_week9_day3.py::test_real_ollama_three_strategy_comparison \
-v -s -p no:cacheprovider
```

可选配置：`WEEK9_RAG_DATASET`、`WEEK9_DAY3_OUTPUT`、`WEEK9_DAY3_INDEX`。
显式设置 `WEEK9_DAY3_REBUILD=1` 可重建索引，默认改用 Day3 自己的目录；自行指定 INDEX 时以该路径为准。
准备阶段框架自检为 **19 passed、6 skipped**；完成两处计算后的 Day1～Day3 联合离线验收为 **106 passed、2 skipped**。
默认模式仍跳过扩展验收，复验全部离线检查应开启对应 DAY 开关，并关闭各 MODEL 开关。
真实 Ollama 对照不计入当前完成证据，后续可另行运行；主项目仍不调用回答型 LLM 完成本日检索。
