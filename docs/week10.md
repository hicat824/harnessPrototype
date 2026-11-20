# Week10 · Tool + RAG 协同

**当前进度**　Day1 至 Day6 已完成（2026-11-20），下一阶段为 Day7 周验收。

本周用 Week6 的 Mock 任务查询提供当前状态，用 Week9 的知识库解释静态业务规则。学习重点是决定每个问题需要哪些数据、先取哪一类证据，以及证据不足时如何追问或停止。每阶段的目标以 [V4 学习计划](/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md#L359) 为准。

## 本周安排

| 日期 | 学习内容 | 验收重点 |
|---|---|---|
| **Day1（已完成）** | 路由与任务阶段 | 三条路由调用顺序正确，缺参与查询失败时按条件停止 |
| **Day2（已完成）** | 多来源 Context | 限长后保留查询事实、文档引用与截断标记 |
| **Day3（已完成）** | Agents SDK 集成 | 用 `Runner` 和 Session 接入 Mock 查询和知识工具，验证 Agent 选路与传参 |
| **Day4（已完成）** | 证据与不确定性 | 能区分查询事实、文档规则、推断与未知项 |
| **Day5（已完成）** | 多轮续接与异常退出 | 补参和压缩后继续执行，权限失败或预算耗尽时明确停止 |
| **Day6（已完成）** | 分层 Agent Evals | 用至少 30 条混合用例分别评估最终结果和工具调用过程 |
| Day7 | 周验收 | 同题比较仅工具、仅 RAG 和自动混合路由的质量、耗时与可记录的 Token 用量 |

## Day1 · 路由与任务阶段

Day1 根据问题所需的证据选择工具。测试预先给出两个意图字段：`needs_current_state` 表示要查任务的当前状态，`needs_rule` 表示要解释业务规则。两者共同决定路由。 [选路逻辑](../week10_tool_rag.py#L72)

| 用户想知道什么 | 路由 | 回答依据 |
|---|---|---|
| 某个任务现在是什么状态 | `tool_only` | Mock 任务查询返回的状态 |
| 某个状态或规则是什么意思 | `rag_only` | 知识库中适用的文档片段 |
| 任务当前是什么状态，以及该状态意味着什么 | `tool_rag` | 先查任务，再用实际返回的状态检索其含义 |

组合路由的两次调用有先后依赖。即使用户猜测任务已经失败，只要查询返回 `RUNNING`，后续就检索 `RUNNING` 的含义。需要查询却缺任务编号时，先追问；查询失败或没有有效状态时，停止依赖该状态的检索。如果已查到状态但缺少解释它的文档，保留任务事实并标记证据不足。两个意图均为假时，问题超出本练习范围，路由函数抛出 `ValueError`。 [执行入口](../week10_tool_rag.py#L127)

Day1 完整离线验收 **5 项通过**。 [验收报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day1/routing_report.json)

### 测试一 · 核对 Mock 与知识样例

- **数据来源**　`mock_scenarios.json` 指定 Mock 身份、任务编号和文档位置。业务响应来自 Week6 的 `MockTransport`，知识原文取自 Week9 的 `KB006.md` 第 6 行。
- **检查点**　`12345` 查询成功并返回 `RUNNING`，`99999` 返回 `NOT_FOUND`，知识原文覆盖状态枚举。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L53)

### 测试二 · 核对组合路由的检索参数

- **数据来源**　测试分别传入 `SUCCESS`、`FAILED` 和 `None`，问题中带有干扰词 `RUNNING`。
- **检查点**　检索参数跟随实际查到的状态。没有状态时返回 `None`。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L67)

### 测试三 · 核对路由与停止条件

- **数据来源**　`eval_cases.json` 提供三条正常路径和两个停止场景。测试把 `input` 传给执行框架，运行结束后读取 `expected` 评分。
- **检查点**　记录工具名、参数和调用顺序。缺任务 ID 时零工具调用。组合查询失败时保留错误码并停止检索。组合查询成功时保留 Mock 来源与知识引用。
- **输出**　默认写入测试目录下的 `runs/day1/routing_report.json`。 [测试入口](../tests/test_week10_tool_rag.py#L116)

### 运行验收

测试数据位于 `/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/`。在主项目根目录使用 `.venv` 运行完整验收。

```bash
RUN_WEEK10_DAY1=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day1 -v -s -p no:cacheprovider
```

去掉 `RUN_WEEK10_DAY1=1` 时只核对 Mock 与知识样例。Day1 全部离线，实际检索质量与模型的自主选路能力安排在后续阶段验证。 [测试数据 README](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/README.md)

## Day2 · 多来源 Context

Day1 解决“去哪找证据”，Day2 解决“哪些证据能交给回答阶段”。组装 Context 时先从业务查询中提取一条可核验的任务事实，再用剩余空间容纳知识片段。回答阶段由此可以分别引用当前状态和规则依据。 [组装入口](../week10_tool_rag.py#L235)

**提取任务事实。** 保留本次查询的结果和错误码。查询成功且任务编号、状态都是非空白字符串时，将这两个原值与来源、查询时间组成事实 `T1`。这保留了回答“哪个任务、何时查到、状态如何”所需的信息。查询成功可以得到任务状态 `FAILED`；查询本身失败或关键字段无效时，则没有可引用的任务事实。仅查规则的问题可以直接使用知识证据。 [事实提取](../week10_tool_rag.py#L195)

**在预算内选择知识片段。** 采用按相关性优先的贪心策略。先给候选片段分配稳定编号，再按原排序选择。每次判断的是加入整条片段后，最终序列化文本是否仍在字符预算内，计数包括任务摘要、引用信息和转义后的正文。恰好达到上限可以保留；片段过长就跳过并继续考察后面的候选。这样既保留了原文，也给靠后的短片段留下空间。例如跳过 `K2` 后仍可引用 `K3`，引用始终指向原来的片段。连任务摘要和基础结构都放不下时，抛出 `ValueError`。 [预算选择](../week10_tool_rag.py#L286)

**判断证据是否完整。** API 的截断标记说明上游数据是否完整，知识截断标记说明本轮是否因预算舍弃了片段。回答阶段获得选中的证据与这两种标记；测试报告还可查看具体遗漏的片段编号。

Day2 完整离线验收 **4 项通过**，与 Day1 合计 **9 项通过**。紧预算场景使用 507 / 550 字符，跳过 `K2`，保留任务事实和 `K1`、`K3` 的引用与原文。 [验收报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day2/context_report.json)

### 测试一 · 样例来源核对

- **数据来源**　`day2_context_cases.json` 指定 Mock 任务 `12345`、不存在的任务 `99999`，以及 Week9 `KB006.md` 第 6、14 行。中间的长片段由样例中的合成文本生成，用于触发预算边界。
- **检查点**　Mock 状态与错误码正确，文档原文可定位，长片段超过紧预算。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L166)

### 测试二 · 成功查询的摘要

- 用 Week6 Mock 的成功响应验证事实字段、来源和查询时间。给响应加上游截断标记，检查该标记能保留到 Context 中，同时验证输入对象保持原样。输出为 pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L177)

### 测试三 · 无有效任务事实

- 用仅查知识、Mock 返回 `NOT_FOUND` 和字段无效的响应检查引用边界。仅查知识时保留文档证据，查询失败时保留错误码，缺少有效任务事实时不生成 `T1`。输出为 pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L198)

### 测试四 · Context 的字符预算

- **数据来源**　复用同一份 Mock 查询结果和三个片段。样例提供宽预算与紧预算，测试另构造恰好容纳完整 Context 和连最小结构都放不下的预算。
- **检查点**　按实际序列化长度验收。宽预算保留全部，紧预算跳过 `K2` 并保留 `K1`、`K3`。任务事实、原文及引用保持完整，真实遗漏有标记。恰好等于预算也可以纳入。
- **输出**　通过后默认写入测试数据目录下的 `runs/day2/context_report.json`，记录预算、实际字符数、跳过的编号和各组 Context。 [测试入口](../tests/test_week10_tool_rag.py#L223)

### 运行验收

在主项目根目录运行以下命令，启用 Day2 完整离线验收。

```bash
RUN_WEEK10_DAY2=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day2 -v -s -p no:cacheprovider
```

默认数据目录为 `/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/`。可用 `WEEK10_DATASET` 指定数据目录，用 `WEEK10_DAY2_OUTPUT` 指定报告目录。 [Day2 样例](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/day2_context_cases.json)

## Day3 · Agents SDK 集成

Day1 的三条路由变成 Agent 的执行规则。Agent 读取原问题，自行决定调用哪个工具和传什么参数。Runner 执行工具，将结果送回模型，再由模型决定继续调用还是回答。Day2 的 Context 在工具返回前组装，模型收到的是有字符预算的任务摘要与知识证据。 [SDK 工具与执行入口](../week10_tool_rag.py#L330) · [Agents SDK 工具文档](https://openai.github.io/openai-agents-python/tools/)

- **只查状态。** Agent 从用户问题中取得任务编号，调用 `query_task`，根据查到的任务事实回答。缺编号时追问。
- **只查规则。** Agent 将原问题传给 `search_knowledge`，根据命中的文档解释规则。用户举出任务编号时，如果仍然只要求解释字段含义，也走这条路径。
- **组合查询。** 先查任务，再根据本轮返回的有效状态构造检索问题。业务查询失败或没有有效状态时，结束这条检索链。知识工具会把命中的片段与已经取得的任务事实组装进同一份 Context。

工具的 Context 是本轮的本地依赖和查询结果。每次运行都重新创建它。Session 则跨轮保存用户消息、工具调用和回答，同一会话的续问可以引用上一轮记录。需要判断任务**此刻**的状态时，重新查询。 [Agents SDK 会话文档](https://openai.github.io/openai-agents-python/sessions/)

Day3 离线验收 **3 项通过**，与 Day1、Day2 合计 **12 项通过**。DeepSeek 真实模型验收 **3 条路由全部通过**。只解释字段的用例只查知识库，组合用例按本轮查到的 `RUNNING` 检索状态含义。 [真实模型验收报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day3/real_model_report.json)

### 练习入口

1. **W10D3-1** 把任务查询和知识检索注册为 Agent 的两个工具。工具签名决定模型可以传入的参数。
2. **W10D3-2** 调用 Runner，传入本轮 Context 与已有 Session，返回 SDK 的运行结果。 [实现入口](../week10_tool_rag.py#L359)

### 测试一 · 工具参数

检查 SDK 生成的工具定义。任务查询接收 `task_id`，知识检索接收 `query`。本地 Context 由 Runner 注入。 [测试入口](../tests/test_week10_tool_rag.py#L260)

### 测试二 · 路由与工具结果

复用 Day1 的五个案例、Week6 Mock 查询和 Week9 知识原文。脚本模型驱动真实 Runner，校验三条正常路径、缺参追问和查询失败停止。组合案例会读取 SDK 返回的实际状态，再形成下一次调用，验收工具参数、顺序和证据是否沿链路传递。通过后保存 `runs/day3/sdk_report.json`。 [测试入口](../tests/test_week10_tool_rag.py#L384)

### 测试三 · 会话续问

连续两轮调用 Runner。第一轮查询任务，第二轮根据同一 Session 中的上轮工具结果回答“我们上一轮查的是哪个任务”。校验续问读到了原任务编号并且没有重复查询。 [测试入口](../tests/test_week10_tool_rag.py#L394)

### 测试四 · 真实模型自主选路

把三条正常路径的原问题交给 `deepseek-flash`。Runner 执行模型自主选择的工具，业务查询和知识检索继续使用本地样例。验收工具选择、参数、顺序和实际返回的证据。通过后保存 `runs/day3/real_model_report.json`。 [测试入口](../tests/test_week10_tool_rag.py#L427) · [DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/)

### 运行验收

默认运行工具参数检查。用以下命令启用离线验收。

```bash
RUN_WEEK10_DAY3=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day3 -v -s -p no:cacheprovider
```

在已配置项目的 DeepSeek `OPENAI_API_KEY` 的终端中运行真实模型验收。

```bash
RUN_WEEK10_DAY3_LLM=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py::test_day3_real_model_routes -v -s -p no:cacheprovider
```

测试复用 `WEEK10_DATASET` 和 `WEEK9_RAG_DATASET`。可设置 `WEEK10_DAY3_OUTPUT` 指定报告目录。

## Day4 · 证据链与不确定性

Day4 把回答整理为五部分。任务 API 提供当前状态，文档提供规则，推断说明从证据得出的判断，未知项说明证据缺口，建议给出下一步。校验器接收这份结构化回答和 Day2 组装的本轮 Context。调用方只在用户确实要查当前状态时传入所查任务的编号。 [校验入口](../week10_tool_rag.py#L445)

| 回答部分 | 本次练习的规则 |
|---|---|
| 当前状态 | 任务编号、状态和引用编号须与本轮 API 事实一致，任务编号还须与用户所问一致 |
| 文档规则 | 引用实际选入 Context 的知识片段 |
| 推断 | 可以联合引用 API 事实和知识片段 |
| 未知项 | 用户要查当前状态而回答无法确认时，明确说明缺少的证据 |
| 下一步建议 | 给出可执行的补充动作，例如重新查询当前任务状态 |

**W10D4-1 检查回答的 `current_task`。** 先确认本轮 Context 有 API 事实 `task_query.fact`，再比较回答与事实中的任务编号、状态和引用编号。回答的任务编号还必须等于调用方传入的 `requested_task_id`。例如用户问任务 `12345`，本轮 API 返回 `12345`、`RUNNING`、`T1`，回答也须给出这三项。把状态写成 `FAILED` 或把引用换成知识片段 `K1` 都应报错。 [练习实现](../week10_tool_rag.py#L445)

**W10D4-2 检查规则和推断的引用。** 编号从实际发送的 Context 中取得，因此被预算跳过的片段无法通过校验。程序检查来源资格、任务身份和状态一致性，规则和推断的文字是否充分受证据支持留给语义验收。

**历史案例命中时。** 即使文档写着任务 `12345` 曾经失败，也只能提供历史背景。如果本轮没有成功查到该任务，应在未知项说明缺少当前状态，并可建议重新查询。只问规则时，即使问题包含示例任务编号，也可以直接解释规则。

Day4 离线验收 **5 项通过**。覆盖结构化回答、任务身份与状态一致性、历史案例与未知项、引用范围，以及预算裁剪后的证据校验。

### 测试一 · 回答结构

用本地构造的组合回答验证五类输出可分别读取，并检查推断必须携带引用。结果显示在 pytest 输出中。 [测试入口](../tests/test_week10_tool_rag.py#L460)

### 测试二 · 接受有效证据

复用 Week6 Mock 返回的当前状态与 Week9 的 KB006 文档原文，验证组合回答和仅查规则两种情况。结果显示在 pytest 输出中。 [测试入口](../tests/test_week10_tool_rag.py#L470)

### 测试三 · 拒绝错误的当前状态

用同一份 Mock 查询结果分别改动回答的任务编号、状态或引用，并改动调用方请求的任务编号，验证只有一致的当前事实才能通过。结果显示在 pytest 输出中。 [测试入口](../tests/test_week10_tool_rag.py#L481)

### 测试四 · 历史案例与未知项

用包含相同任务编号的合成历史记录，覆盖无查询和查询失败。校验器应拒绝用历史状态作答，并接受明确说明证据缺口的回答。结果显示在 pytest 输出中。 [测试入口](../tests/test_week10_tool_rag.py#L497)

### 测试五 · 无效引用

复用 Day2 紧预算样例，验证规则引用任务事实、引用未出现的编号、推断引用被预算跳过的片段都会失败。结果显示在 pytest 输出中。 [测试入口](../tests/test_week10_tool_rag.py#L523)

### 运行验收

默认检查回答结构。下列命令在项目根目录启用完整 Day4 离线验收。

```bash
RUN_WEEK10_DAY4=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day4 -v -s -p no:cacheprovider
```

## Day5 · 多轮任务、缺参追问与异常退出

Day5 把一次查询扩展成可续接的任务。调用方为每个任务保留一份 `Day5TaskState`，记录原问题、两种意图、任务编号、已用工具次数和上轮结果。Session 保存对话，压缩时归纳旧消息。用户补参后，执行器根据仍在的待办状态继续处理原任务。 [任务状态与执行入口](../week10_tool_rag.py#L495)

- **缺参追问。** 组合任务缺任务编号时返回 `needs_input`，只等用户补答，工具调用次数保持为零。补入编号后，按原来的意图查任务并解释查到的状态。
- **工具预算。** 任务查询和知识检索各用一次预算。每次调用前检查余额，通过后立即占用一次。预算只有一次时，组合任务可得到任务事实，加查知识时会以 `TOOL_BUDGET_EXHAUSTED` 结束，保留已查到的事实。
- **不可恢复错误。** 遇到 `FORBIDDEN` 时保留错误码并结束。后续调用同一任务时复用停止结果。新的查询目标由调用方建立新的任务状态。
- **压缩后续接。** 摘要可以只概括「此前讨论了一项任务」。原问题和追问状态仍由待办对象保存。收到明确的任务编号后继续执行，预算从该任务已用的次数接着算。

Day5 离线验收 **6 项通过**，与 Day1～Day4 合计 **23 项通过**。覆盖带空格的编号补参、零次和一次工具预算、权限失败退出，以及 Session 历史压缩后的待办续接。

### 练习入口

1. **W10D5-1** 在待补参状态下接收用户提供的非空白编号，保留原问题和意图。 [补参实现](../week10_tool_rag.py#L517)
2. **W10D5-2** 检查下一次调用是否有预算，允许时计数，耗尽时在工具执行前抛出预算异常。 [预算实现](../week10_tool_rag.py#L533)

### 测试一 · 初始化任务预算

用一条本地状态查询验证任务具备独立计数和上轮结果字段。覆盖零预算和非法负预算。 [测试入口](../tests/test_week10_tool_rag.py#L551)

### 测试二 · 追问与补参续跑

复用 Week6 的 `12345` Mock 任务和 KB006 的状态说明。验证缺参和空白补答均保持等待，用户提供编号后按原意图运行两步，完成后再次请求同一任务时复用结果。 [测试入口](../tests/test_week10_tool_rag.py#L560)

### 测试三 · 预算耗尽前停下

对同一组合请求分别给零次和一次预算，核对实际工具次数。零次时在第一步停下，一次时保留任务事实并跳过知识检索。 [测试入口](../tests/test_week10_tool_rag.py#L586)

### 测试四 · 权限错误退出

用 Week6 Mock 的故障编号 `33333` 触发 403。验收保留 `FORBIDDEN`，停止知识检索，并且续轮不会重试已结束的任务。 [测试入口](../tests/test_week10_tool_rag.py#L598)

### 测试五 · 压缩后恢复待办

在本地 SQLite Session 中写入待补参的旧对话，用现有 `compact_session` 和固定摘要确实压缩历史。原始提问被压缩后，验收同一待办对象仍可接受任务编号并完成查询。 [测试入口](../tests/test_week10_tool_rag.py#L612)

### 运行验收

默认检查任务状态初始化。下列命令启用完整 Day5 离线测试，结果显示在 pytest 输出中。

```bash
RUN_WEEK10_DAY5=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day5 -v -s -p no:cacheprovider
```

## Day6 · 分层 Agent Evals

Day6 离线验收 **4 项通过**。30 题共 37 轮，Trajectory 和 Outcome 均为 30/30。[Mock 评估报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day6/mock_eval_report.json)

Day6 要写的是**评分器**。前五天实现的任务流程作为被测对象。每道题给它一份输入，测试记录它实际做了什么、最后答了什么，再把这两份记录和该题的预期逐轮对照。今天要学会区分两种情况：答案碰巧正确，但中途查错了；工具都用对了，但最后漏答了。

### 跟一题看两处练习接在哪里

以 `W10D6-17` 为例，用户问「我猜 12345 是 FAILED，确认并解释查到的状态」。数据集的 `input` 提供问题、已识别的查询与解释意图、任务编号和 Mock 身份。`expected_turns` 另存每轮的评分标准，执行结束后才交给评分器。[Day6 用例集](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/day6_eval_cases.json)

1. **执行并记录。** Day5 复用 Day1 的组合路由，先用 `query_task` 查任务 `12345`。Week6 Mock 返回当前状态 `RUNNING`，执行器再用这个状态询问其含义，知识工具命中 `KB006.md#L6`。测试把两次调用的工具名、实际参数、检索命中位置记入本轮的 `calls`。**这就是练习一的输入。**
2. **组装证据并回答。** Day2 把本轮 API 事实和预算内的文档片段组装成 `context`。测试从中生成 Day4 格式的固定答复 `answer`，其中当前任务应为 `12345 / RUNNING`，引用 API 事实 `T1`，状态释义引用文档片段 `K1`。执行器另给出 `result`，记录本轮是否答完、为何停止。**这三份数据和本轮要查询的任务编号一起进入练习二。** [执行与采集](../tests/test_week10_tool_rag.py#L685)
3. **逐轮打分。** 批量测试取出对应的 `expected_turns` 条目，与上述实际记录一起传给两个评分函数。两个函数各返回三项成绩。多轮题每轮都有自己的调用记录和预期，例如第一轮应追问，补齐编号的下一轮才应查工具。[评分调用处](../tests/test_week10_tool_rag.py#L804)

### 练习一 · Trajectory 怎样判断工具是否用对

在 `grade_day6_trajectory(calls, expected)` 中，每个 `call` 是一次工具调用，`call.name` 是工具名。函数给**整轮**返回一个字典，三个键固定为 `tool_order`、`tool_arguments`、`retrieval`。每个值是该项的成绩，`True` 表示通过，`False` 表示失败，`None` 表示本轮不适用。召回项直接读取本轮检索记录，三项一起返回。[W10D6-1](../week10_tool_rag.py#L635)

| 评分项 | 比较的数据与通过条件 |
|---|---|
| `tool_order` | 把 `[call.name for call in calls]` 与 `list(expected.tool_order)` 比较，将结果放在返回字典的 `tool_order` 键下。少一次、多一次或顺序颠倒都失败。预期和实际都为空列表时通过。 |
| `tool_arguments` | 顺序正确后，将实际查询的任务编号列表与预期比较。每次知识检索的 `query` 必须包含全部指定词，并避开全部排除词。顺序错误时本项失败，双方均零调用时记为不适用。 |
| `retrieval` | 用**检索刚返回的来源位置**核对预期，要求覆盖全部指定位置。预期空列表表示应搜索且零命中，预期为 `None` 表示本轮无需评召回。 |

例如预期先查任务再查知识，实际却先查知识再查任务，知识检索仍命中了预期片段。此轮应返回 `{"tool_order": False, "tool_arguments": False, "retrieval": True}`。批量测试把这个字典放进该轮的 `trajectory`，再将值为 `False` 的项目列入 `failed_checks`。测试二包含完整字典的断言，也单独断言了交换顺序后的 `["tool_order"] is False`。[返回值断言](../tests/test_week10_tool_rag.py#L759) · [报告写入位置](../tests/test_week10_tool_rag.py#L811)

### 练习二 · Outcome 怎样判断回答是否成立、是否答全

在 `grade_day6_outcome` 中给**本轮停止结果和实际答复**评分。引用与原文由证据校验函数核对。[W10D6-2](../week10_tool_rag.py#L713)

| 评分项 | 比较的数据与通过条件 |
|---|---|
| `status` | 将 `result.status`、`result.reason` 与预期逐项比较。例如正常答完应为 `answered` 且无停止原因，缺编号应为 `needs_input` 和 `MISSING_PARAMETER`。这里评的是执行结果，任务自身的 `RUNNING` 属于回答内容。 |
| `source_support` | 用实际传给回答阶段的 **Context** 核验回答。当前状态声明须与所问任务、本轮 API 事实及引用一致。对本练习的抽取式规则答复，引用必须指向保留下来的片段，正文必须与该片段原文一致。 |
| `final_answer` | 用**题目预期**核对回答内容。应报告当前状态时须答对任务编号和状态，应无状态声明时须留空。有效引用解析出的原文位置集合须与 `rule_sources` 一致。题目要求说明缺口时，`unknowns` 至少要有一项非空说明。 |

仍看第 17 题。如果回答只解释了 `RUNNING`，并把当前状态写成未知，它给出的规则可以有据可查，执行也可以正常结束。但 API 已查明任务状态，用户要求的事实漏答了，因此 `final_answer` 失败。另一个值得对照的题是 `W10D6-11`：检索应命中第 6、14 行，260 字符的 Context 预算只留下第 6 行。因此召回按两处来源评分，回答按一处有效引用加缺口说明评分。

### 怎样读验收报告

先看 `failed_checks` 定位题号、轮次和失败项，再看该题 `turns` 里的实际调用、答复和被裁剪的证据编号。例如 `turn1.trajectory.tool_arguments` 指向首轮调用参数，`turn1.outcome.final_answer` 指向首轮答复内容。

报告的通过率按题计算。同一指标在一题的所有适用轮次都通过，才算该题通过；整题都不适用就跳过该题。每项的 `rate` 等于通过题数 `passed` 除以适用题数 `applicable`，分母为零时记录 `null`。每层的 `overall` 则看一题所有轮次、该层所有指标是否出现失败。整批验收要求两层都通过。

### 测试一 · 数据集检查

- **数据来源**　`week10_tool_rag/day6_eval_cases.json` 提供 30 个案例，含仅工具、仅 RAG、组合路由、停止与续接五类。
- **测试逻辑**　检查题量、ID 唯一性和每题的参考轮次。可单独运行 `-k day6_dataset`。

### 测试二 · 执行轨迹评分

- **数据来源**　测试提供一条 `query_task` 和一条 `search_knowledge` 记录，知识返回引用 Week9 的 `KB006.md#L6`。
- **测试逻辑**　即使检索命中，带入干扰词也应判为参数错误。交换调用顺序或遗漏检索时，对应指标失败。正常零调用时参数和召回均不适用。

### 测试三 · 结果评分

- **数据来源**　测试构造 `RUNNING` 的查询结果，引用原有的 KB006 文档片段。
- **测试逻辑**　先验收正确答复，再分别删掉当前状态、把规则正文改成与引用相反的意思，核对能否定位到回答不完整或来源不支持。

### 测试四 · 30 题 Mock 批量评估

- **数据来源**　题目与参考答案来自 Day6 数据集。业务查询由 Week6 的 Mock API 执行，知识原文来自 Week9 的 KB006 第 6、11、14 行，按固定词表检索。
- **测试逻辑**　逐题新建任务，逐轮采集调用参数、召回位置和实际答复。两层指标全部通过才验收成功。覆盖 `FORBIDDEN`、超时、空结果、知识缺失、补参续跑、预算耗尽和已结束任务的重复请求。
- **输出**　默认写入测试数据目录下的 `runs/day6/mock_eval_report.json`。报告记录两层通过率、适用题数、失败项和实测耗时。Mock 路径的模型用量记为 `null`。本测试验收确定性流程、证据和评分逻辑。真实模型的自主选路、真实检索质量及语义支持需通过各自的模型验收。

### 运行验收

在主项目根目录运行 Day6 离线验收。

```bash
RUN_WEEK10_DAY6=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day6 -v -s -p no:cacheprovider
```

测试默认从 `/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/` 读数据。可用 `WEEK10_DATASET`、`WEEK9_RAG_DATASET` 覆盖输入目录，用 `WEEK10_DAY6_OUTPUT` 覆盖报告目录。
