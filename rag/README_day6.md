# Week7 Day6：最小 RAG Evals 与错误分析

实现：`rag/evaluation.py`。用例：`rag/eval_cases_day6.json`。测试：`tests/test_rag_evaluation.py`。
沿用 Day4 索引、Ollama Embedding 和 Day5 DeepSeek 回答，不修改检索或生成主流程。

## 学习状态：已完成

2026-11-06 用户确认 Week7 Day6 学习完成，下一项为 Day7 周验收。

- Day6 完整离线验收：43 passed、1 skipped；跳过项为真实模型测试。
- Week7 Day2～Day6 联合离线回归：225 passed、5 skipped；真实模型开关均关闭。
- 已保存的真实评估：15 条流程运行完成，自动检查全部通过，无执行异常。
- 本次助手逐条语义复核：15 条 supported；离线合并后最终通过 15 条、失败 0 条、待复核 0 条。复核来源已标明为助手，没有冒充独立人工验收。
- 错误分析实例：W7D6-08 的交付证据原排第 6，默认 K=5 未召回；单独调整为 K=10 后进入 Context，回答正确引用该证据。

本次流程中 14 条调用生成模型，W7D6-02 无候选时由程序直接返回信息不足。
文档组召回均值为 1.0，分母是 13 条有目标文档组的用例；不代表任意问题都能召回关键段落。
过滤条件由用例明确提供，未验证模型自动解析过滤条件或真实业务 API 状态。

本地证据见 `artifacts/day6/real_model/` 下的 `evaluation_report.json`、`manual_reviews.json` 和 `semantic_review.md`；运行产物沿用 `.gitignore`，不纳入代码提交。

## 学习内容与验收标准

| 学习内容 | 验收标准 |
|---|---|
| 固定问题与预期来源 | 理解 15 条问题的适用版本、目标文档和参考事实 |
| 检索与 Context 分阶段观察 | 能区分没召回目标文档，还是召回后未装入 Context |
| 元数据适用性 | 能发现当前问题混入历史版本或错误流程的片段 |
| 引用检查与语义判断的边界 | 知道原文摘录存在不等于生成结论成立，能人工识别曲解或虚构 |
| 评估结果与错误分析 | 保存 15 条真实运行记录，逐条复核并说明失败原因；不要求第一次质量就全部通过 |

离线注入错误的测试用于练习归因，不证明真实模型能力。真实结果没有某类错误时，也不需要编造错误；可以结合注入测试解释如何定位。

## 只需填写三个关键步骤

搜索 `TODO(W7D6-1/2/3)`，补相应表达式并删除紧邻的 `raise NotImplementedError(...)`。

| 函数 | 已提供的变量/辅助函数 | 需要补的内容 |
|---|---|---|
| `document_group_recall` | `expected_groups`、去重后的 `present_docs` | 统计至少命中一篇允许文档的预期组数 |
| `check_versions` | `chunks`、`case`、`_version_matches` | 判断所有片段是否适用，空列表也要正确处理 |
| `classify_failures` | `value`，取值 True/False/None | 只把明确的 False 判为失败，不把 None 当失败 |

函数输入输出和边界都写在 docstring 中。文件读写、批量运行、阶段轨迹、异常记录、报告汇总及人工复核合并不需要再手写。

## 15 条用例的数据来源

原始知识仍来自 `/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics/knowledge/`，均为既有合成资料。
本日根据原文编写独立评估文件，没有把参考答案加入知识索引。

| 用例 | 问题重点 | 预期来源或结果 |
|---|---|---|
| W7D6-01 | NOT_FOUND 是否等于任务失败 | KB012 |
| W7D6-02 | 不存在的 import_std 3.0 规则 | 信息不足；版本过滤后无候选 |
| W7D6-03 | FORBIDDEN 是否能推断任务状态 | KB010 或 KB011 |
| W7D6-04 | 查询成功与 state=FAILED | KB006 |
| W7D6-05 | task_id 的前导零 | KB007 |
| W7D6-06 | 2.0 版 limit | KB008 |
| W7D6-07 | 历史 1.0 版 limit | KB009 |
| W7D6-08 | import_std 2.0 人工复核 | KB002 |
| W7D6-09 | import_std 1.0 历史交付规则 | KB003 |
| W7D6-10 | import_standard 的职责 | KB004 |
| W7D6-11 | items 与 total | KB008 或 KB014 |
| W7D6-12 | 截断预览与全量统计 | KB014 或 KB017 |
| W7D6-13 | 只读请求重试次数与例外 | KB013 |
| W7D6-14 | INFO 正文包含 ERROR 子串 | KB017 |
| W7D6-15 | 任务此刻是否完成 | 信息不足，需要实时 API |

用例中的 `question` 和 `filters` 是请求输入；`expected_*`、允许版本和 `reference_points` 是评估参考。
用例可选字段 `k` 覆盖批量运行的默认检索数量（默认 5），必须是正整数。
`W7D6-08` 单独设置 `k=10`，用于扩大召回范围；其余用例沿用默认值。
实际使用值保存在逐条报告的 `retrieval_k` 中。Context 字符预算仍然生效，增大 k 不保证所有片段都能进入 Context。
本日过滤条件由用例显式提供，不测试 Agent 是否会自动从自然语言解析过滤条件。
生成模型只接收 Day5 组装的“问题 + 最终 Context”，不接收用例 ID、预期文档、参考事实或评估结果。

## 指标和判断边界

### 文档组召回率

`expected_doc_groups` 使用“组内可替代、组间都需要”的形式：

```text
[["KB008", "KB014"], ["KB007"]]
```

这表示需要两类来源：第一类命中 KB008/KB014 任意一个即可，第二类需要 KB007。
只命中 KB014 时召回率为 1/2；同一文档有十个 Chunk 也不能重复加分。
当前 15 条用例中，13 条有预期文档组；两条信息不足问题不强行指定目标文档。

没有预期组时返回 `None`，汇总时不进入召回均值分母。有预期组但没有命中则是 `0`。
检索接口失败、根本没得到检索结果时也是未知，不能猜成 0 或满分。

此指标只看 doc_id，不检查具体 Chunk 是否包含全部所需事实。命中了文档标题却没命中关键正文，可能召回指标仍通过，但人工会发现回答遗漏。

### 自动检查

| 检查/标签 | 含义 |
|---|---|
| `retrieval_error` | 未召回预期文档组 |
| `context_error` | 目标已召回，但最终 Context 缺少它；上游本就未召回时不重复归因 |
| `version_error` | 检索或 Context 中存在不适用版本/知识状态 |
| `scope_error` | 结果不符合明确请求的 engine_code |
| `answer_status_error` | 应回答却拒答，或应说明不足却回答 |
| `citation_error` | 引用来源或原文摘录校验失败 |
| `unnecessary_generation` | 没有 Context 却调用了生成器 |
| `incomplete_trace` | 没有异常记录，但执行轨迹缺少必要阶段结果 |

接口和输出格式异常另记 `retrieval_execution_error`、`context_build_error`、`generation_error`、`answer_format_error` 等标签。
标签可以同时出现，定位时先看上游阶段，不能仅凭标签就断定是模型还是程序的责任。

检查值 `True` 为通过、`False` 为失败、`None` 为未检查或不适用。
自动检查通过记为 `automated_passed=true`，不等于最终语义通过。

### 人工语义复核

| verdict | 含义 |
|---|---|
| `pending` | 尚未核对原文 |
| `supported` | 结论/拒答理由受证据支持，并覆盖问题要点 |
| `hallucination` | 虚构事实、曲解摘录或把静态知识当实时事实 |
| `incomplete` | 有依据，但遗漏问题中的必要要点 |

判为 hallucination/incomplete 时，`notes` 必须说明理由。
自动通过且人工 supported 才是 `final_status=passed`；未复核是 `needs_review`，已有失败则为 `failed`。
没有使用额外 LLM 当裁判，也没有用关键词出现次数代替语义判断。

## 处理架构与产物

```text
固定 15 条用例
    ↓ 每条只传 question/filters 进入 RAG
Day4 检索 → Day5 Context → 生成 → 结构与引用校验
    ↓ 逐阶段保存，包括出错前已得到的数据
自动指标和错误标签 → 待人工复核
    ↓ 离线合并人工 verdict/notes
最终通过/失败与错误分布
```

单条模型失败会被记录，后续用例继续；评估器自身未完成的 TODO 或实现错误不会被当成模型错误吞掉。
报告会保留原始生成文本，便于定位格式、编号或摘录错误；异常只记录阶段、类型和可用 HTTP 状态码，不保存任意异常正文或请求认证头。

| 文件 | 内容 |
|---|---|
| `cases/W7D6-xx.json` | 单条问题、参考项、检索片段、Context、实际消息、原始输出、异常和检查结果 |
| `evaluation_report.json` | 全部条目与汇总，便于离线复查 |
| `summary.json` | 召回均值与分母、自动通过数、人工待复核数、最终通过/失败数、错误分布 |
| `manual_reviews_template.json` | 所有用例的 pending 模板，不包含已确认的人工结论 |
| `manual_reviews.json` | 由你复制模板并填写的人工复核文件 |

每条完成后就保存，不等全部结束才写文件。目录重复运行会覆盖同名结果；重新生成回答后应重新人工复核，不能直接套用旧回答的复核结论。
评估文件不属于知识语料，不要写入原知识清单。

## 运行命令

在仓库根目录执行，无新依赖。

### 1. 框架离线自检

```bash
.venv/bin/python -B -m pytest tests/test_rag_evaluation.py -q -s -p no:cacheprovider
```

默认只运行已提供代码。只读知识清单，使用固定回答和固定向量，不调用模型。

### 2. 完成三个步骤后的完整离线验收

```bash
RUN_WEEK7_DAY6=1 RUN_WEEK7_DAY6_LLM=0 \
.venv/bin/python -B -m pytest tests/test_rag_evaluation.py -q -s -p no:cacheprovider
```

离线完整批量流程使用两条独立合成用例，验证报告与人工复核机制，不冒充 15 条真实语义评估。
固定产物目录：`rag/artifacts/day6/offline_fixture/`。

### 3. 真实 15 条问题评估（另行显式启用）

需要保留 Day4 索引、启动 Ollama、在当前终端配置既有 DeepSeek `OPENAI_API_KEY`。
本轮准备代码时不运行下面的命令。

```bash
RUN_WEEK7_DAY6=1 RUN_WEEK7_DAY6_LLM=1 \
.venv/bin/python -B -m pytest tests/test_rag_evaluation.py -q -s \
  -k real_rag_evaluation -p no:cacheprovider
```

默认 Top-K=5、Context 预算 9000 字符。每条最多一次问题编码和一次生成，没有候选时两个模型都跳过。
会调用本地 Ollama 与 DeepSeek，产生真实生成费用；正文向量仍复用索引。

这里 **pytest 通过只说明 15 条跑完且没有执行、格式或引用校验异常**。召回、版本、回答状态等自动指标仍可能失败，语义仍需人工复核，不能把这一行 pytest 当作 RAG 质量全通过。

也可直接运行批量评估，便于调整 K 或 Context 预算做对照实验：

```bash
.venv/bin/python -B -m rag.evaluation run --real-models \
  --k 5 --context-chars 9000 \
  --output rag/artifacts/day6/real_model
```

不加 `--real-models` 时命令会直接拒绝执行。不要为追求满分而把参考答案塞进 Prompt。

### 4. 人工复核并离线汇总

真实批量完成后：

```bash
cp rag/artifacts/day6/real_model/manual_reviews_template.json \
   rag/artifacts/day6/real_model/manual_reviews.json
```

逐条打开 `cases/W7D6-xx.json`，对照回答、引用、原文和 `reference_points`，修改复核文件，例如：

```json
{
  "reviews": [
    {"case_id": "W7D6-01", "verdict": "supported", "notes": "结论与 KB012 原文一致"},
    {"case_id": "W7D6-02", "verdict": "hallucination", "notes": "示意：若模型编造了不存在的 3.0 步骤，应记录具体虚构内容"}
  ]
}
```

示例第二项只说明格式，不是预先判定真实结果。允许先复核一部分条目，其余保持 pending。

```bash
.venv/bin/python -B -m rag.evaluation review \
  --report rag/artifacts/day6/real_model/evaluation_report.json \
  --reviews rag/artifacts/day6/real_model/manual_reviews.json
```

此命令不调用模型，会重算自动指标、合并人工判断，并更新逐条报告和汇总。

| 测试环境变量 | 默认值/用途 |
|---|---|
| `RAG_DAY6_OUTPUT` | 仓库下的 `rag/artifacts/day6`，pytest 产物根目录 |
| `RAG_DAY6_CASES` | 本模块的固定 15 条用例文件 |
| `RAG_DAY6_INDEX` | `rag/artifacts/day4/local_model/knowledge_index.json` |
| `WEEK7_RAG_DATASET` | 框架测试核对原始知识清单的位置 |
| `OLLAMA_URL` | `http://localhost:11434` |
| `RAG_GENERATION_MODEL` | 既有 `deepseek-flash` 生成模型 |
| `OPENAI_API_KEY` | 既有 DeepSeek 认证，不写入报告或命令行 |

CLI 使用自己的 `--cases`、`--index`、`--output` 参数；`rag/artifacts/` 已被 Git 忽略。

下面每节对应一个测试函数。

## 测试一：15 条用例与知识清单

函数：`test_provided_fifteen_cases_match_knowledge_manifest`，默认执行。

- **数据来源**：固定评估 JSON 与已有 Week7 知识清单。
- **流程**：读取用例，核对目标文档、版本、知识状态及文件存在性。
- **检查点**：15 个唯一 ID；目标来源真实存在且版本一致；两条信息不足问题。
- **输出**：pytest 结果，不调用模型。

## 测试二：用例文件错误

函数：`test_provided_case_loader_rejects_bad_data`，默认执行。

- **数据来源**：临时修改的用例副本。
- **流程**：分别破坏版本、ID、问题、文档组或过滤条件后加载。
- **检查点**：非法用例报错，不带着错误参考项运行真实评估。
- **输出**：pytest 临时文件与结果。

## 测试三：执行轨迹与参考答案隔离

函数：`test_provided_trace_records_stages_without_sending_gold`，默认执行。

- **数据来源**：合成索引、固定回答和带唯一参考标记的用例。
- **流程**：真实执行 Day4/Day5 代码，由替身提供问题向量和生成文本。
- **检查点**：保存实际消息和原始输出；只编码问题；参考标记不进入模型消息。
- **输出**：pytest 结果。

## 测试四：阶段错误与部分数据保留

函数：`test_provided_trace_preserves_partial_results_and_errors`，默认执行。

- **数据来源**：模拟检索异常、生成超时、坏 JSON、未知引用编号。
- **流程**：运行执行轨迹收集。
- **检查点**：阶段与异常类型准确；出错前的检索结果、Context、原始回答仍被保留。
- **输出**：pytest 结果，不打印密钥或任意异常正文。

## 测试五：汇总中的未知值与待复核

函数：`test_provided_summary_excludes_unknown_recall_and_pending_passes`，默认执行。

- **数据来源**：人工构造的两条汇总输入，一条召回率为 None。
- **流程**：写汇总和批量报告。
- **检查点**：None 不进入均值分母，pending 不算最终通过，全部未知时均值仍为 None。
- **输出**：临时 `summary.json`、`evaluation_report.json`。

## 测试六：真实调用显式开关

函数：`test_provided_cli_requires_explicit_real_flag`，默认执行。

- **数据来源**：未带真实调用确认参数的 CLI 命令。
- **流程**：新进程启动 `rag.evaluation run`。
- **检查点**：直接拒绝，不调用 Ollama 或 DeepSeek。
- **输出**：pytest 结果。

## 测试七：文档组召回计算

函数：`test_document_group_recall_counts_groups_not_chunks`。

- **数据来源**：重复 Chunk、可替代文档组、部分命中、零命中和无预期组。
- **流程**：调用你填写的召回函数。
- **检查点**：按组计数，重复文档不加分；部分命中 0.5，无预期组为 None。
- **输出**：pytest 结果。

## 测试八：全部片段的版本适用性

函数：`test_version_check_requires_all_chunks_to_apply`。

- **数据来源**：正确版本、混入历史版本、错误知识状态和空片段列表。
- **流程**：调用版本检查。
- **检查点**：必须每条都适用，不能一条正确就判全组通过；空列表为 True。
- **输出**：pytest 结果；是否召回仍由独立指标判断。

## 测试九：False 与 None

函数：`test_failure_labels_distinguish_false_from_unknown`。

- **数据来源**：同时包含 True、False、None 的检查字典及非法数字值。
- **流程**：分类错误标签。
- **检查点**：只为 False 生成标签，不能把 None 当失败，拒绝非布尔检查值。
- **输出**：pytest 结果。

## 测试十：自动通过还需人工确认

函数：`test_automatic_pass_still_requires_manual_review`。

- **数据来源**：来源、引用与回答状态均正确的固定执行记录。
- **流程**：先自动评分，再给出人工 supported。
- **检查点**：最初 needs_review；人工支持后才 passed；不修改原轨迹。
- **输出**：pytest 结果。

## 测试十一：不完整轨迹不能通过

函数：`test_incomplete_trace_is_not_automatically_passed`。

- **数据来源**：没有异常记录，但移除了检索、Context 和回答的轨迹。
- **流程**：尝试评分。
- **检查点**：标为 incomplete_trace，不把没检查当作成功。
- **输出**：pytest 结果。

## 测试十二：引用存在但结论虚构

函数：`test_manual_review_detects_hallucination_even_with_valid_quote`。

- **数据来源**：正确摘录搭配“所有任务都完成”的不受支持结论。
- **流程**：先做自动检查，再输入人工 hallucination 及理由。
- **检查点**：自动引用检查可能通过；人工复核后最终失败，标记 answer_hallucination。
- **输出**：pytest 结果；展示自动引用检查的实际边界。

## 测试十三：错误阶段归因

函数：`test_error_tags_locate_retrieval_context_version_and_answer`。

- **数据来源**：分别注入错误召回、Context 放不下、历史版本和不必要拒答。
- **流程**：运行固定流程并评分。
- **检查点**：分别出现 retrieval/context/version/answer_status 标签；上游未召回时不额外怪罪 Context。
- **输出**：pytest 结果。

## 测试十四：信息不足没有伪造召回分数

函数：`test_abstention_case_has_no_fake_retrieval_score`。

- **数据来源**：没有目标文档组、过滤后无候选的信息不足问题。
- **流程**：执行并评分。
- **检查点**：召回率及对应检查为 None，合理拒答仍可自动通过其他检查。
- **输出**：pytest 结果。

## 测试十五：离线批量与人工合并

函数：`test_offline_batch_saves_reports_and_applies_reviews`。

- **数据来源**：两条合成问题、固定向量和固定回答，不使用 15 条真实题的预设答案来冒充模型。
- **流程**：批量运行 → 保存 → 写人工结论 → 离线合并。
- **检查点**：自动通过仍 pending；人工确认后才更新最终通过数。
- **输出**：`offline_fixture/` 下的完整报告、逐条记录和人工复核示例。

## 测试十六：单条模型失败后继续

函数：`test_batch_continues_after_generation_error`。

- **数据来源**：首次生成抛错、后续成功的替身。
- **流程**：顺序执行两条问题。
- **检查点**：两条都有记录，第一条保留失败阶段和 Context，第二条仍能正常检查。
- **输出**：pytest 临时目录中的批量报告及单条文件。

## 测试十七：人工复核输入检查

函数：`test_manual_review_rejects_inconsistent_input`。

- **数据来源**：未知 ID、重复 ID、判定幻觉却没有理由的人工文件。
- **流程**：尝试离线合并。
- **检查点**：明确报错，不静默套用到错误用例。
- **输出**：pytest 临时文件与结果。

## 测试十八：真实 15 条批量评估

函数：`test_real_rag_evaluation_records_fifteen_cases`，需另行显式启用。

- **数据来源**：固定 15 条问题、Day4 真实索引、Ollama 和既有 DeepSeek 配置。
- **流程**：逐条检索、生成、评分和保存。
- **检查点**：15 条都已运行、没有执行/格式/引用异常，人工状态初始为 pending；不要求自动召回等质量指标全通过。
- **输出**：`real_model/` 下的评估报告与人工复核模板。

## 测试十九：按用例覆盖检索 K 值

函数：`test_case_k_overrides_batch_default`，随 Day6 离线练习一起启用。

- **数据来源**：测试文件中的两条合成用例、两段正文、固定向量与固定回答。
- **流程**：以默认 `k=1` 批量执行，第二条用例单独配置 `k=2`，再读取落盘记录。
- **检查点**：两条分别召回 1 和 2 个片段；报告保存实际 `retrieval_k`；覆盖不影响其他用例。
- **输出**：pytest 临时目录下的 `evaluation_report.json`、`summary.json` 和 `cases/SYNTH-EXPANDED.json`。

## 测试二十：拒绝无效的用例 K 值

函数：`test_provided_case_rejects_invalid_k`，默认运行。

- **数据来源**：合成用例分别填入 0、负数、布尔值、小数、字符串和 null。
- **流程**：将用例写为 JSON，再调用 `load_cases` 校验。
- **检查点**：统一拒绝非正整数，不发送模型或网络请求。
- **输出**：pytest 临时目录中的 `cases.json` 及测试结果。

本日最终验收应能拿一条记录解释：问题希望得到什么、检索到了什么、模型看到了什么、回答依据是否成立，以及错误更应归到哪个阶段。
