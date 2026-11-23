# Week11 · LangGraph 专项学习

**当前进度**　Day1 已完成（2026-11-23），下一阶段为 Day2 条件分支、循环和流程退出。本周用 Week6 的 Mock API 练习显式的只读查询工作流。[V4 学习计划](/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md#L375)

## Day1 · State、节点与边

今天用一个问题走完最小流程。例如输入「请查询任务 12345 的当前状态」，最终拿到任务编号、API 结果以及「任务 12345 当前状态为 RUNNING。」这句答复。

```mermaid
flowchart LR
    S([START]) --> P["parse · 提取 task_id"]
    P --> Q["query · 写入 query_result"]
    Q --> A["answer · 写入答复"]
    A --> E([END])
```

| 概念 | 在 Day1 中的作用 |
|---|---|
| **State** | 当前运行里供节点读取的数据。初始只有 `question`，后来逐步累计 `task_id`、`query_result` 和 `answer`。 |
| **Node** | 读取自己需要的字段并返回本节点的更新。例如 `parse` 只返回 `{"task_id": "12345"}`，LangGraph 将它合并到含原问题的 State。 |
| **Edge** | 规定当前节点结束后进入哪个节点。这里四条固定边定义了依赖顺序，使 `query` 读取到任务编号，`answer` 读取到查询结果。 |

Agents SDK 的 Session 用于保存和读取跨轮消息历史。Day1 的 LangGraph State 用于传递本次运行内的业务数据，每次调用以新问题开始。Week11 Day4 再用 Checkpointer 练习按 `thread_id` 保存和恢复图的状态。[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

### 状态图实现

`build_day1_graph(query_task)` 注册三个节点、连接四条边并返回编译后的图。调用方提供的 `query_task` 接收任务编号并返回 Week6 的 `BusinessResult`。使用 `graph.invoke({"question": ...})` 获取合并后的最终 State，使用 `graph.stream(..., stream_mode="updates")` 查看各节点分别写入的字段。[代码入口](../langgraph_learning/day1.py#L47) · [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

### 测试数据与验收

Day1 离线测试 **4 项通过**。

四道用例来自现有 Week6 Mock API。每道题提供 `input.question`、Mock 身份 `subject_id` 和参考结果 `expected`。前三题查询成功，任务状态分别为 `RUNNING`、`SUCCESS`、`FAILED`。第四题由 Mock 返回 `NOT_FOUND`，答复报告当前状态无法确认。[用例文件](/media/xumaozhou/data/AI/local_agent_test/week11_langgraph/day1_cases.json)

- **数据核对。** 验证任务编号提取、Mock API 返回和固定答复。
- **最终 State。** 验证四题都保留原问题、写入正确的结果，每题查询恰好一次。
- **节点更新。** 通过 `stream_mode="updates"` 验证节点执行顺序和各自更新的字段。
- **独立运行。** 对同一个编译好的图连续传入不同问题，验证两次查询分别使用各自的编号。

在主项目根目录运行。当前项目环境使用 LangGraph `1.2.20`。

```bash
RUN_WEEK11_DAY1=1 .venv/bin/python -B -m pytest tests/test_week11_day1.py -v -p no:cacheprovider
```

只检查数据时可省略 `RUN_WEEK11_DAY1=1`。用例目录可由 `WEEK11_DATASET` 指定。
