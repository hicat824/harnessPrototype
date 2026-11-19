# Week10 · Tool + RAG 协同

**当前进度**　Day1 已完成（2026-11-19）。下一阶段为 Day2 多来源 Context。

本周用 Week6 的 Mock 任务查询提供当前状态，用 Week9 的知识库解释静态业务规则。学习重点是决定每个问题需要哪些数据、先取哪一类证据，以及证据不足时如何追问或停止。每阶段的目标以 [V4 学习计划](/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md#L359) 为准。

## 本周安排

| 日期 | 学习内容 | 验收重点 |
|---|---|---|
| **Day1（已完成）** | 路由与任务阶段 | 三条路由调用顺序正确，缺参与查询失败时按条件停止 |
| Day2 | 多来源 Context | 限长后保留查询事实、文档引用与截断标记 |
| Day3 | Agents SDK 集成 | 用 `Runner` 和 Session 接入 Mock 查询和知识工具，验证 Agent 选路与传参 |
| Day4 | 证据与不确定性 | 能区分查询事实、文档规则、推断与未知项 |
| Day5 | 多轮续接与异常退出 | 补参和压缩后继续执行，权限失败或预算耗尽时明确停止 |
| Day6 | 分层 Agent Evals | 用至少 30 条混合用例分别评估最终结果和工具调用过程 |
| Day7 | 周验收 | 同题比较仅工具、仅 RAG 和自动混合路由的质量、耗时与可记录的 Token 用量 |

## Day1 · 路由与任务阶段

完整离线验收 **5 项通过**。报告中五个场景全部通过，组合路由用 Mock 实际返回的 `RUNNING` 检索，缺任务编号时零工具调用，查询失败时跳过知识检索。 [验收报告](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/runs/day1/routing_report.json)

Day1 的测试已经从问题中标出两种需求，分别放在 `Day1Request` 的两个布尔字段中：

- `needs_current_state` 为 `True` 表示要**查询任务此刻的状态**，由 Mock `query_task` 提供。
- `needs_rule` 为 `True` 表示要**解释业务规则或状态含义**，由 `search_knowledge` 提供文档依据。

| 示例问题 | `needs_current_state` | `needs_rule` | `select_route` 的结果 |
|---|---|---|---|
| 任务 `12345` 现在是什么状态？ | `True` | `False` | `tool_only`，只查任务 |
| `FAILED` 是什么意思？ | `False` | `True` | `rag_only`，只查文档 |
| 任务 `12345` 现在是什么状态？这个状态是什么意思？ | `True` | `True` | `tool_rag`，先查任务，再用查到的状态检索文档 |
| 你好 | `False` | `False` | 抛出 `ValueError` |

`select_route` 直接读取这两个字段。`task_id` 在执行阶段检查，缺少编号时仍可先选路，再返回 `needs_input`。从自然语言识别这两个字段安排在 Day3。

### 两处练习的代码入口

1. **`W10D1-1`** `select_route` 已按上表实现三种路由。 [select_route](../week10_tool_rag.py#L66)
2. **`W10D1-2`** `prepare_knowledge_query` 用 Mock 实际查询到的状态生成检索问题。没有可用状态时返回 `None`，框架跳过知识工具。 [prepare_knowledge_query](../week10_tool_rag.py#L88)

### 测试一 · 核对 Mock 与知识样例

- **数据来源**　`mock_scenarios.json` 指定 Mock 身份、任务编号和文档位置。业务响应来自 Week6 的 `MockTransport`，知识原文取自 Week9 的 `KB006.md` 第 6 行。
- **检查点**　`12345` 查询成功并返回 `RUNNING`，`99999` 返回 `NOT_FOUND`，知识原文覆盖状态枚举。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L38)

### 测试二 · 核对组合路由的检索参数

- **数据来源**　测试分别传入 `SUCCESS`、`FAILED` 和 `None`，问题中带有干扰词 `RUNNING`。
- **检查点**　检索参数跟随实际查到的状态。没有状态时返回 `None`。
- **输出**　pytest 结果。 [测试入口](../tests/test_week10_tool_rag.py#L52)

### 测试三 · 核对路由与停止条件

- **数据来源**　`eval_cases.json` 提供三条正常路径和两个停止场景。测试把 `input` 传给执行框架，运行结束后读取 `expected` 评分。
- **检查点**　记录工具名、参数和调用顺序。缺任务 ID 时零工具调用。组合查询失败时保留错误码并停止检索。组合查询成功时保留 Mock 来源与知识引用。
- **输出**　默认写入测试目录下的 `runs/day1/routing_report.json`。 [测试入口](../tests/test_week10_tool_rag.py#L101)

### 运行验收

测试数据位于 `/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/`。在主项目根目录使用 `.venv` 运行完整验收。

```bash
RUN_WEEK10_DAY1=1 .venv/bin/python -B -m pytest tests/test_week10_tool_rag.py -v -s -p no:cacheprovider
```

去掉 `RUN_WEEK10_DAY1=1` 时只核对 Mock 与知识样例。Day1 全部离线，实际检索质量与模型的自主选路能力安排在后续阶段验证。 [测试数据 README](/media/xumaozhou/data/AI/local_agent_test/week10_tool_rag/README.md)
