# Week7 Day5：Context 组装与有依据的回答

实现：`rag/answering.py`。测试：`tests/test_rag_answering.py`。
沿用现有 **Ollama Embedding + DeepSeek 生成**：前者检索证据，后者阅读证据并回答。

> 学习状态（2026-11-05）：**Week7 Day5 已完成，当前进入 Day6。**
> 离线验收：47 passed、2 skipped；两组真实模型测试由用户运行，2 passed。
> 已对照报告核验：NOT_FOUND 回答的三条结论均有 KB012 原文支持；实时状态问题正确说明信息不足。

## 学习内容与验收标准

| 学习内容 | 验收标准 |
|---|---|
| Retrieval 与 Generation 的分工 | 检索返回原片段，生成结论不能冒充工具或文档原文 |
| Context 选择与预算 | 只发送选中的完整片段，保留来源，不超字符预算 |
| Instructions 与本轮输入 | 规则放 system，问题和检索证据放 user；文档内容作为数据处理 |
| 引用与来源解析 | 每条结论有证据编号和原文摘录，程序能映射回本次 Context 的原 Chunk |
| 信息不足 | 无证据时不调用生成模型；静态资料不能证明实时状态时明确说明不足 |

程序检查的是结构、引用范围和逐字摘录。即使摘录确实存在，也不自动证明模型结论被摘录充分支持，仍需人工对照原文。本日不实现 Day6 的成组问答质量评估。

## 三个关键步骤回顾

三个关键步骤均已实现，可按下表复习 Context 选择、消息角色与引用校验。

| 位置 | 输入变量 | 当前实现 | 原理 |
|---|---|---|---|
| `build_context` | `candidate_text`、`max_chars` | `fits` | 加入完整证据后仍在预算内，等于预算也允许 |
| `build_messages` | `system_text`、`user_text` | `messages` | 按 system、user 的顺序组装两条 role/content 消息 |
| `validate_answer` | `sources_by_id`、`citation.source_id` | `source` | 只在本次 Context 的编号映射中查找，未知编号得到 None |

保存文件、引用摘录检查、Pydantic 输出结构、API 请求和错误处理不需要你重新实现。

## 测试数据来源与整体流程

| 数据 | 来源 | 使用范围 |
|---|---|---|
| 四条合成知识与固定向量 | `tests/test_rag_answering.py` 内的 `TEXTS`、`hits`、`index` | 完整离线流程；第四条为历史规则，用来检查是否错误进入模型输入 |
| 固定 JSON 回答 | 测试内的 `FixedGenerator` | 只验证流程和引用，不代表真实模型理解能力 |
| API 请求与响应 | `httpx.MockTransport`、显式测试假 Key | 验证 DeepSeek 参数和异常，不访问公网 |
| 真实知识索引 | `rag/artifacts/day4/local_model/knowledge_index.json` | 显式真实验收；沿用此前的合成 Week7 知识 |
| 真实问题 | 一个 NOT_FOUND 解释问题、一个当前任务状态问题 | 检查有引用回答与信息不足两条路径 |

```text
问题 + Day4 索引
    ↓ search_knowledge（当前版本过滤，只编码问题）
检索结果 hits
    ↓ build_context（去重、完整片段预算、分配 S1/S2/...）
ContextBundle
    ├─ 无片段 → 程序直接返回 insufficient，不调用生成器
    └─ 有片段 → build_messages → DeepSeek / 固定回答替身
                                  ↓
                     RagAnswer 结构检查 → validate_answer
                                  ↓
                  结论 + 引用摘录 + 原始 Chunk + 调试产物
```

Embedding 模型和生成模型是两个不同模型。已有 `qwen3-embedding:0.6b` 用来生成向量，不能代替生成回答的 `deepseek-flash`。
本实现不创建新的 Agent/Session，也不改 `agentPrompts.py` 的通用提示词。RAG 专用规则放在 `rag/answering.py` 的 `RAG_ANSWER_INSTRUCTIONS`。

## 输入输出约定

### Context

`build_context(hits, max_chars=6000)` 返回 `ContextBundle`：

- `text`：完整序列化的证据 JSON；字符预算包含正文、编号、分数和元数据。
- `sources`：`[{source_id, score, chunk}, ...]`，与 `text` 一致的引用映射。
- `omitted_chunk_ids`：因预算放不下而跳过的原 Chunk ID。

按检索顺序逐个尝试，相同 `chunk_id` 去重；片段放不下就跳过，继续尝试后续较短片段。
正文不截断、不改写。S1/S2/... 在本次 Context 内连续编号，不是全局文档 ID。
所有片段放不下时返回空 Context。

`max_chars` 是 Python 字符数，不是 Token；不包含 system、问题或模型输出预算。

### 模型消息

`build_messages(question, context)` 返回两条消息：

```text
[
  {"role": "system", "content": 规则与输出结构},
  {"role": "user", "content": 原始问题与本次检索证据}
]
```

Context 中的角色声明或命令仍属于文档数据，不应升级成 system 指令。
离线测试只检查消息分层和内容去向，不能据此宣称模型绝不会受提示词注入影响。

### 模型回答

```json
{
  "status": "answered",
  "claims": [
    {
      "text": "NOT_FOUND 不代表任务执行失败。",
      "citations": [
        {
          "source_id": "S1",
          "quote": "NOT_FOUND 表示未找到任务，不等于任务执行失败。"
        }
      ]
    }
  ],
  "missing_information": ""
}
```

上述正文来自离线合成样例；真实模型必须引用当次实际证据。
证据不足时结构为：

```json
{
  "status": "insufficient",
  "claims": [],
  "missing_information": "缺少任务当前状态，需要调用实时 API。"
}
```

当前练习采用保守的二选一，不实现部分回答。`answered` 必须有带引用结论且缺失信息为空；`insufficient` 必须没有结论并说明缺失信息。
引用路径不由模型填写，程序根据 `source_id` 解析到原 Chunk 的 `doc_id`、`metadata.source/version` 和定位字段。
不允许引用已检索到但没有进入 Context 的片段；`quote` 必须是对应正文中的非空白逐字摘录。

### 生成器接口

`generate(messages)` 是异步可调用对象：输入两条消息，返回 JSON 字符串。
`FixedGenerator` 用于离线测试；`DeepSeekGenerator` 使用现有环境变量 `OPENAI_API_KEY` 中的 DeepSeek Key，不保存或打印密钥。
[DeepSeek JSON 模式](https://api-docs.deepseek.com/guides/json_mode/)只约束 JSON 格式，字段与引用仍由程序验证；适配器拒绝空正文和未完整结束的输出。
失败时保留 HTTP、超时、结构或引用异常，不无限重试，不把故障伪装成“信息不足”。

## 运行命令

在仓库根目录执行，无新增依赖。

### 1. 框架和生成接口 Mock 自检

```bash
.venv/bin/python -B -m pytest tests/test_rag_answering.py -q -s -p no:cacheprovider
```

默认只运行已提供代码的测试；不用真实 Key，不访问 Ollama 或 DeepSeek。

### 2. 完整离线验收

```bash
RUN_WEEK7_DAY5=1 RUN_WEEK7_DAY5_LLM=0 \
.venv/bin/python -B -m pytest tests/test_rag_answering.py -q -s -p no:cacheprovider
```

全部使用固定向量、固定回答或 HTTP Mock。产物保存在 `rag/artifacts/day5/offline_fixture/answer_report.json`。

### 3. 真实回答验收（另行显式启用）

需要保留 Day4 索引、启动 Ollama，并配置既有 DeepSeek Key。不要把 Key 写进命令行、代码或报告。

```bash
RUN_WEEK7_DAY5=1 RUN_WEEK7_DAY5_LLM=1 \
.venv/bin/python -B -m pytest tests/test_rag_answering.py -q -s \
  -k real_retrieval_and_grounded_answer -p no:cacheprovider
```

会执行两组问题，调用本地 Ollama 编码问题，并把选中的合成片段发送到 DeepSeek 生成回答，产生真实模型调用成本。
Embedding 正文向量仍复用 Day4 索引，不重新编码全部文档。

也可单独运行一个问题：

```bash
.venv/bin/python -B -m rag.answering \
  --question 'NOT_FOUND 是任务失败吗？查不到时应该先核对什么？' \
  --context-chars 6000 \
  --output rag/artifacts/day5/real_model/answer_report.json
```

CLI 默认只检索当前版本，支持 `--engine-code`、`--min-score`、`--k`、`--index` 和 `--ollama-url`。

| 测试环境变量 | 默认值 | 作用 |
|---|---|---|
| `RAG_DAY5_OUTPUT` | 仓库下的 `rag/artifacts/day5` | pytest 产物根目录 |
| `RAG_DAY5_INDEX` | 仓库下的 `rag/artifacts/day4/local_model/knowledge_index.json` | 真实测试索引 |
| `OLLAMA_URL` | `http://localhost:11434` | 真实测试的本地向量服务地址 |
| `RAG_GENERATION_MODEL` | `deepseek-flash` | 真实测试的生成模型 |
| `OPENAI_API_KEY` | 项目既有环境配置 | DeepSeek 认证，只在真实生成适配器中使用 |

CLI 使用自己的命令行参数；同一输出目录重新运行会覆盖同名文件。`rag/artifacts/` 已被 Git 忽略。

## 如何阅读产物

| 文件 | 来源 | 主要内容 |
|---|---|---|
| `offline_fixture/answer_report.json` | 离线完整流程 | 人工向量与固定回答，不代表真实语义能力 |
| `real_model/grounded_report.json` | 真实 NOT_FOUND 问题 | 检索结果、最终 Context、实际消息、结论与解析后的引用 |
| `real_model/realtime_report.json` | 真实当前状态问题 | 信息不足结论、缺失信息及模型输入 |
| `real_model/answer_report.json` | CLI 默认输出 | 单个问题的完整调试记录 |

重点查看 `retrieved_chunk_ids` 与 `context.sources` 的差异、`context.omitted_chunk_ids`、`messages` 和 `resolved_citations`。
`generation_called=false` 表示没有证据、由程序直接拒答；不是模型故障。

下面每节对应一个测试函数。

## 测试一：回答结构校验

函数：`test_provided_answer_schema_rejects_invalid_payload`，默认执行。

- **数据来源**：测试内固定 JSON，分别修改状态、结论、引用或增加未知字段。
- **流程**：直接执行 Pydantic 校验。
- **检查点**：无引用结论、状态不一致、空缺失说明、非法编号与未知字段均被拒绝。
- **输出**：pytest 结果。

## 测试二：空 Context 直接说明不足

函数：`test_provided_empty_context_refuses_without_generation`，默认执行。

- **数据来源**：空检索列表及禁止调用的生成器替身。
- **流程**：创建空 Context，执行回答流程。
- **检查点**：返回 insufficient，没有结论、引用或模型消息，不调用生成器。
- **输出**：pytest 结果。

## 测试三：零候选不调用两个模型

函数：`test_provided_no_candidates_skip_embedding_and_generation`，默认执行。

- **数据来源**：合成索引与无法匹配的流程条件。
- **流程**：执行检索到回答的完整空结果分支。
- **检查点**：Embedding 与生成器均无调用，返回信息不足。
- **输出**：pytest 结果。

## 测试四：生成请求契约

函数：`test_provided_deepseek_request_contract`，默认执行。

- **数据来源**：HTTP Mock、固定消息与显式测试假 Key。
- **流程**：执行真实适配器的序列化、HTTP 请求和响应解析。
- **检查点**：模型、消息、JSON 模式、超时、输出上限正确，一次请求返回原始 JSON 字符串。
- **输出**：pytest 结果，不访问公网。

## 测试五：生成器配置错误

函数：`test_provided_deepseek_configuration_guard`，默认执行。

- **数据来源**：临时移除 Key 的测试环境、空模型名、非法超时。
- **流程**：构造生成适配器。
- **检查点**：配置不完整立即报错，不发起请求、不泄露 Key。
- **输出**：pytest 结果；测试结束后环境变量自动恢复。

## 测试六：HTTP 错误传播

函数：`test_provided_deepseek_propagates_http_errors`，默认执行。

- **数据来源**：Mock 返回 401、429、500。
- **流程**：调用生成器并检查调用次数。
- **检查点**：保留 HTTPStatusError，不重试、不伪装成信息不足。
- **输出**：pytest 结果。

## 测试七：连接与超时错误

函数：`test_provided_deepseek_propagates_transport_errors`，默认执行。

- **数据来源**：Mock 抛出连接错误或读取超时。
- **流程**：调用生成适配器。
- **检查点**：保留原异常类型。
- **输出**：pytest 结果。

## 测试八：不完整生成响应

函数：`test_provided_deepseek_rejects_incomplete_responses`，默认执行。

- **数据来源**：缺失 choices/message、空正文、未正常结束的 Mock 响应。
- **流程**：走实际响应检查。
- **检查点**：HTTP 成功也不能把空内容或被截断回答当成有效输出。
- **输出**：pytest 结果。

## 测试九：Context 顺序与来源保留

函数：`test_context_preserves_order_sources_and_input`。

- **数据来源**：前三条合成检索结果。
- **流程**：执行你补全的 Context 构建，再修改返回对象。
- **检查点**：原文、全部来源、顺序、连续编号正确，输入不受修改影响。
- **输出**：pytest 结果。

## 测试十：整块预算与等值边界

函数：`test_context_budget_is_whole_chunk_and_inclusive`。

- **数据来源**：放不下的长片段和恰好放得下的短片段。
- **流程**：按预算依次尝试加入 Context。
- **检查点**：长片段跳过并记录；短片段完整保留、编号仍从 S1 开始；等于预算也接受。
- **输出**：pytest 结果。

## 测试十一：去除重复 Chunk

函数：`test_context_deduplicates_chunk_ids`。

- **数据来源**：重复出现的同一 Chunk，加一个不同 Chunk。
- **流程**：构建 Context。
- **检查点**：重复正文只进入一次，编号连续。
- **输出**：pytest 结果。

## 测试十二：非法预算

函数：`test_context_rejects_invalid_budget`。

- **数据来源**：零、负数、bool 和小数预算。
- **流程**：调用 Context 构建。
- **检查点**：抛 ValueError。
- **输出**：pytest 结果。

## 测试十三：规则与证据的角色分离

函数：`test_messages_separate_rules_question_and_evidence`。

- **数据来源**：正常片段中加入一条伪指令，以及独立问题。
- **流程**：执行消息组装。
- **检查点**：规则仅在 system，问题与完整 Context 在 user；伪指令没有进入 system；输入不变。
- **输出**：pytest 结果；不证明真实模型完全免疫提示词注入。

## 测试十四：引用解析到原文

函数：`test_citations_resolve_to_selected_original_sources`。

- **数据来源**：两个已选片段及带正确引用的固定回答。
- **流程**：根据编号解析引用。
- **检查点**：结论索引、摘录和原 Chunk 对应；返回副本不影响 Context。
- **输出**：pytest 结果。

## 测试十五：拒绝伪造引用

函数：`test_citations_reject_unknown_or_fabricated_evidence`。

- **数据来源**：未知编号、虚构摘录、引用错片段、空白摘录。
- **流程**：结构校验后执行引用校验。
- **检查点**：全部拒绝，不能因为来源编号格式正确就接受回答。
- **输出**：pytest 结果。

## 测试十六：不能引用预算外片段

函数：`test_cannot_cite_retrieved_but_budget_excluded_chunk`。

- **数据来源**：检索到两个片段，但预算只允许第一个进入 Context。
- **流程**：固定回答尝试引用第二个片段。
- **检查点**：报错；引用校验只看最终 Context，不偷用完整原检索列表。
- **输出**：pytest 结果。

## 测试十七：模型实际输入与调用次数

函数：`test_answer_call_uses_selected_context_once`。

- **数据来源**：已选 Context 与固定回答生成器。
- **流程**：调用一次生成器、解析回答、解析引用。
- **检查点**：生成器收到的消息与报告一致；只调用一次；引用全部可解析。
- **输出**：pytest 结果。

## 测试十八：离线完整 RAG 回答

函数：`test_offline_retrieval_context_and_answer_pipeline`。

- **数据来源**：包含历史规则的合成索引、固定向量与固定回答。
- **流程**：当前版本检索 → Context → 生成 → 引用解析 → 保存。
- **检查点**：只编码一个问题，不重编码正文；历史规则不进入模型输入；输入索引不变。
- **输出**：`offline_fixture/answer_report.json`。

## 测试十九：生成故障不能伪装成拒答

函数：`test_generation_failure_is_not_misreported_as_insufficient`。

- **数据来源**：非法 JSON、非法结构、抛异常的生成器。
- **流程**：执行回答流程。
- **检查点**：保留结构或模型异常，一次调用后报错，而不是返回看似正常的 insufficient。
- **输出**：pytest 结果。

## 测试二十：全部片段超预算

函数：`test_all_chunks_over_budget_skip_generation`。

- **数据来源**：两个正常片段、仅一个字符的合法预算。
- **流程**：构建空 Context 后执行回答。
- **检查点**：记录两个遗漏 ID，不调用生成器，明确说明证据不足。
- **输出**：pytest 结果。

## 测试二十一：真实生成与实时信息边界

函数：`test_real_retrieval_and_grounded_answer`，另行显式启用，两组问题。

- **数据来源**：Day4 真实索引、本地 Ollama、现有 DeepSeek 配置。
- **流程**：真实检索后，把最终 Context 发给生成模型并执行结构与引用校验。
- **检查点**：NOT_FOUND 解释问题应回答并引用原文；当前任务状态问题应说明静态证据不足，不能捏造完成状态。
- **输出**：`real_model/grounded_report.json`、`real_model/realtime_report.json`。

人工验收时，逐条对照 `claims.text` 与 `resolved_citations.quote`，说明模型做了哪些归纳、哪些事实直接来自文档，以及缺少哪些信息。不要只凭 JSON 校验通过就认定答案正确。
