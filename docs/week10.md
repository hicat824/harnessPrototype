# Week10 · Tool + RAG 协同

**当前进度**　Day1、Day2 已完成（2026-11-19）。下一阶段为 Day3 Agents SDK 集成。

本周用 Week6 的 Mock 任务查询提供当前状态，用 Week9 的知识库解释静态业务规则。学习重点是决定每个问题需要哪些数据、先取哪一类证据，以及证据不足时如何追问或停止。每阶段的目标以 [V4 学习计划](/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md#L359) 为准。

## 本周安排

| 日期 | 学习内容 | 验收重点 |
|---|---|---|
| **Day1（已完成）** | 路由与任务阶段 | 三条路由调用顺序正确，缺参与查询失败时按条件停止 |
| **Day2（已完成）** | 多来源 Context | 限长后保留查询事实、文档引用与截断标记 |
| Day3 | Agents SDK 集成 | 用 `Runner` 和 Session 接入 Mock 查询和知识工具，验证 Agent 选路与传参 |
| Day4 | 证据与不确定性 | 能区分查询事实、文档规则、推断与未知项 |
| Day5 | 多轮续接与异常退出 | 补参和压缩后继续执行，权限失败或预算耗尽时明确停止 |
| Day6 | 分层 Agent Evals | 用至少 30 条混合用例分别评估最终结果和工具调用过程 |
| Day7 | 周验收 | 同题比较仅工具、仅 RAG 和自动混合路由的质量、耗时与可记录的 Token 用量 |

## Day1 · 路由与任务阶段

Day1 根据问题所需的证据选择工具。测试预先给出两个意图字段：`needs_current_state` 表示要查任务的当前状态，`needs_rule` 表示要解释业务规则。两者共同决定路由。 [选路逻辑](../week10_tool_rag.py#L67)

| 用户想知道什么 | 路由 | 回答依据 |
|---|---|---|
| 某个任务现在是什么状态 | `tool_only` | Mock 任务查询返回的状态 |
| 某个状态或规则是什么意思 | `rag_only` | 知识库中适用的文档片段 |
| 任务当前是什么状态，以及该状态意味着什么 | `tool_rag` | 先查任务，再用实际返回的状态检索其含义 |

组合路由的两次调用有先后依赖。即使用户猜测任务已经失败，只要查询返回 `RUNNING`，后续就检索 `RUNNING` 的含义。需要查询却缺任务编号时，先追问；查询失败或没有有效状态时，停止依赖该状态的检索。如果已查到状态但缺少解释它的文档，保留任务事实并标记证据不足。两个意图均为假时，问题超出本练习范围，路由函数抛出 `ValueError`。 [执行入口](../week10_tool_rag.py#L122)

Day1 完整离线验收 **5 项通过**。 [验收报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day1/routing_report.json)

### 测试一 · 核对 Mock 与知识样例

- **数据来源**　`mock_scenarios.json` 指定 Mock 身份、任务编号和文档位置。业务响应来自 Week6 的 `MockTransport`，知识原文取自 Week9 的 `KB006.md` 第 6 行。
- **检查点**　`12345` 查询成功并返回 `RUNNING`，`99999` 返回 `NOT_FOUND`，知识原文覆盖状态枚举。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L43)

### 测试二 · 核对组合路由的检索参数

- **数据来源**　测试分别传入 `SUCCESS`、`FAILED` 和 `None`，问题中带有干扰词 `RUNNING`。
- **检查点**　检索参数跟随实际查到的状态。没有状态时返回 `None`。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L57)

### 测试三 · 核对路由与停止条件

- **数据来源**　`eval_cases.json` 提供三条正常路径和两个停止场景。测试把 `input` 传给执行框架，运行结束后读取 `expected` 评分。
- **检查点**　记录工具名、参数和调用顺序。缺任务 ID 时零工具调用。组合查询失败时保留错误码并停止检索。组合查询成功时保留 Mock 来源与知识引用。
- **输出**　默认写入测试目录下的 `runs/day1/routing_report.json`。 [测试入口](../tests/test_week10_tool_rag.py#L106)

### 运行验收

测试数据位于 `/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/`。在主项目根目录使用 `.venv` 运行完整验收。

```bash
RUN_WEEK10_DAY1=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day1 -v -s -p no:cacheprovider
```

去掉 `RUN_WEEK10_DAY1=1` 时只核对 Mock 与知识样例。Day1 全部离线，实际检索质量与模型的自主选路能力安排在后续阶段验证。 [测试数据 README](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/README.md)

## Day2 · 多来源 Context

Day1 解决“去哪找证据”，Day2 解决“哪些证据能交给回答阶段”。组装 Context 时先从业务查询中提取一条可核验的任务事实，再用剩余空间容纳知识片段。回答阶段由此可以分别引用当前状态和规则依据。 [组装入口](../week10_tool_rag.py#L230)

**提取任务事实。** 保留本次查询的结果和错误码。查询成功且任务编号、状态都是非空白字符串时，将这两个原值与来源、查询时间组成事实 `T1`。这保留了回答“哪个任务、何时查到、状态如何”所需的信息。查询成功可以得到任务状态 `FAILED`；查询本身失败或关键字段无效时，则没有可引用的任务事实。仅查规则的问题可以直接使用知识证据。 [事实提取](../week10_tool_rag.py#L190)

**在预算内选择知识片段。** 采用按相关性优先的贪心策略。先给候选片段分配稳定编号，再按原排序选择。每次判断的是加入整条片段后，最终序列化文本是否仍在字符预算内，计数包括任务摘要、引用信息和转义后的正文。恰好达到上限可以保留；片段过长就跳过并继续考察后面的候选。这样既保留了原文，也给靠后的短片段留下空间。例如跳过 `K2` 后仍可引用 `K3`，引用始终指向原来的片段。连任务摘要和基础结构都放不下时，抛出 `ValueError`。 [预算选择](../week10_tool_rag.py#L281)

**判断证据是否完整。** API 的截断标记说明上游数据是否完整，知识截断标记说明本轮是否因预算舍弃了片段。回答阶段获得选中的证据与这两种标记；测试报告还可查看具体遗漏的片段编号。

Day2 完整离线验收 **4 项通过**，与 Day1 合计 **9 项通过**。紧预算场景使用 507 / 550 字符，跳过 `K2`，保留任务事实和 `K1`、`K3` 的引用与原文。 [验收报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day2/context_report.json)

### 测试一 · 样例来源核对

- **数据来源**　`day2_context_cases.json` 指定 Mock 任务 `12345`、不存在的任务 `99999`，以及 Week9 `KB006.md` 第 6、14 行。中间的长片段由样例中的合成文本生成，用于触发预算边界。
- **检查点**　Mock 状态与错误码正确，文档原文可定位，长片段超过紧预算。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L156)

### 测试二 · 成功查询的摘要

- 用 Week6 Mock 的成功响应验证事实字段、来源和查询时间。给响应加上游截断标记，检查该标记能保留到 Context 中，同时验证输入对象保持原样。输出为 pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L167)

### 测试三 · 无有效任务事实

- 用仅查知识、Mock 返回 `NOT_FOUND` 和字段无效的响应检查引用边界。仅查知识时保留文档证据，查询失败时保留错误码，缺少有效任务事实时不生成 `T1`。输出为 pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L188)

### 测试四 · Context 的字符预算

- **数据来源**　复用同一份 Mock 查询结果和三个片段。样例提供宽预算与紧预算，测试另构造恰好容纳完整 Context 和连最小结构都放不下的预算。
- **检查点**　按实际序列化长度验收。宽预算保留全部，紧预算跳过 `K2` 并保留 `K1`、`K3`。任务事实、原文及引用保持完整，真实遗漏有标记。恰好等于预算也可以纳入。
- **输出**　通过后默认写入测试数据目录下的 `runs/day2/context_report.json`，记录预算、实际字符数、跳过的编号和各组 Context。 [测试入口](../tests/test_week10_tool_rag.py#L213)

### 运行验收

在主项目根目录运行以下命令，启用 Day2 完整离线验收。

```bash
RUN_WEEK10_DAY2=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -k day2 -v -s -p no:cacheprovider
```

默认数据目录为 `/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/`。可用 `WEEK10_DATASET` 指定数据目录，用 `WEEK10_DAY2_OUTPUT` 指定报告目录。 [Day2 样例](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/day2_context_cases.json)
