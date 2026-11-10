# Week8：LangChain 学习

> 基准计划：`/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md`。
> 当前进度（2026-11-10 用户确认）：**Day1：架构、Models 与 Messages 已完成**；下一项为 Day2：Prompt Template、Runnable 与 Parser。
> 本目录为独立练习，测试仍放在仓库 `tests/`。主项目的 Agents SDK 和 Week7 RAG 保持现有实现。

## Day1 完成记录

- 两个关键步骤已实现：展开历史消息，await 模型的 ainvoke 接口。
- 完整离线回归：29 passed、1 skipped；跳过项为显式真实模型测试。
- 用户运行真实测试通过；助手核对实际报告，消息顺序为 system/human/ai/human，回答正确保留任务标识 00123。
- 报告保留了实际 usage、AIMessage 正文及 SDK 输入对照；本次提交前未重复调用真实模型。
- 本地报告为 `artifacts/day1/real_model/report.json`，运行产物和独立虚拟环境均不提交。

## Day1 学什么、验收什么

今天先理解模型调用和消息格式，不开始 RAG 重构。需要掌握的是：

| 学习内容 | 可以展示的验收结果 |
|---|---|
| LangChain 的组成 | 说明 core、模型适配包和高层 Agent 各自负责什么 |
| Message 是输入输出的数据对象 | 能区分 SystemMessage、HumanMessage、AIMessage，解释顺序和正文来源 |
| Model 是模型调用接口 | 通过一次 ainvoke 得到 AIMessage，读取 content 和实际 usage |
| 多轮上下文由调用方组织 | 证明同一个模型实例不会自动携带上次问题，历史需明确传入 |
| 与 Agents SDK 对照 | 说明规则对应 Agent.instructions，对话列表对应 Runner 的 input |
| 模型调用与 Agent Run 的边界 | 明白 ainvoke 返回一条消息，不会替你执行本项目的工具循环和 Session 管理 |

今天不要求手写 HTTP、另写测试框架或接入业务工具。Prompt Template、Runnable 组合和 Parser 是 Day2 内容；Tools/Agent 是 Day3 内容。

### 包和组件如何理解

| 名称 | 本日作用 |
|---|---|
| `langchain-core` | 提供消息对象、聊天模型接口及离线 Fake ChatModel 等基础能力 |
| `langchain-openai` | 提供 ChatOpenAI，把统一模型接口适配到兼容的 Chat Completions API |
| `ChatOpenAI` | 本日通过明确的 base_url 调用已有 DeepSeek，不代表请求被发送给 OpenAI |
| `langchain` 高层包 | 后续 Agent 学习再按需要安装；Day1 无需先安装整个框架 |
| LangGraph | 本日仅知道它与高层 Agent 的关系，状态图专项留给后续计划 |

本日使用 DeepSeek 的文本兼容接口，不承诺 ChatOpenAI 能保留该服务的全部专有功能。

## 代码结构与两个关键步骤回顾

| 文件 | 职责 |
|---|---|
| [day1.py](day1.py) | 消息组装、SDK 对照、一次异步调用与报告保存 |
| [requirements.txt](requirements.txt) | 固定本次验证过的直接依赖版本 |
| [test_week8_day1.py](../tests/test_week8_day1.py) | 已提供的离线测试和显式真实测试 |

处理流程：

```text
规则 + 可选历史 + 本轮问题
          ↓ build_messages
SystemMessage + 历史 HumanMessage/AIMessage + HumanMessage
          ├─ to_sdk_request → 仅生成 SDK 输入对照，不执行 Runner
          ↓ model.ainvoke
AIMessage → content / usage_metadata → JSON 调试报告
```

| 练习标识 | 输入 | 已完成的输出 |
|---|---|---|
| `W8D1-1` | instructions、history_messages、question | 一个 messages 列表：规则在最前，本轮问题在最后 |
| `W8D1-2` | model 和 messages | await 一次异步模型接口，得到 response: AIMessage |

两处均已实现：使用 `*history_messages` 展开历史消息，使用 `await model.ainvoke(messages)` 调用模型。
每个函数的 docstring 已说明参数、输出、异常和支持范围，关键代码旁保留原理注释。

### 输入与输出的具体含义

```python
instructions = "根据用户提供的信息简洁回答。"
history = [
    {"role": "user", "content": "任务标识为字符串 00123。"},
    {"role": "assistant", "content": "已知任务标识为字符串 00123。"},
]
question = "刚才的任务标识是什么？请保留前导零。"
```

对应的 LangChain 消息类型顺序是 `system → human → ai → human`。
Message 的 `.type` 使用 human/ai；转换到 HTTP 或 SDK 输入时，对应 role=user/assistant。
不能把 `.type` 字符串直接当成所有接口的 role 值。

本日只支持非空纯文本消息。工具结果、多模态内容块、SDK 的完整 Session 条目不属于这个转换器的输入范围。
history 是调用方显式提供的记录；模型实例不会自动把之前的调用放进下一次输入。

报告包括：

| 字段 | 含义 |
|---|---|
| `messages` | 本次实际传入模型的 type/content，便于查看角色及顺序 |
| `sdk_request.instructions` | 对应 SDK 的 Agent instructions；本日只展示，不执行 SDK |
| `sdk_request.input` | 对应 Runner.run 的纯文本对话 input |
| `response` | AIMessage 的 type 和原始纯文本 content |
| `usage` | AIMessage.usage_metadata，缺失时保持 null，不猜测 Token 数量 |
| `response_metadata` | 只导出返回中存在的 model_name、finish_reason |
| `elapsed_seconds` | 本次模型接口调用及返回检查的耗时 |

### 和当前 Agents SDK 的区别

| 对比点 | 本日 LangChain 调用 | 当前项目 Agents SDK |
|---|---|---|
| 规则 | 列表中的 SystemMessage | Agent(instructions=...) |
| 用户输入与历史 | HumanMessage / AIMessage 列表 | Runner.run(agent, input=...) 的文本或消息列表 |
| 单次返回 | AIMessage，读取 content | RunResult，读取 final_output 等运行结果 |
| 工具执行 | 本日不执行，收到工具调用会明确报错 | Runner 结合工具定义管理 Agent Loop |
| 历史保存 | 本日由调用方明确传 history | 当前项目使用 Session，并在 Runner 中传入 |

这里是同一段文本的接口对照，不是把完整 SDK 运行结果、工具调用或 Session 无损转换为 LangChain。
现有 SDK 调用入口可参看 [agentLite.py](../agentLite.py) 的 `simpleCallLLM`。

## 环境与数据来源

### 独立 Python 环境

本次已创建并安装：`langchain_learning/.venv`，使用 Python 3.10。
主项目 `.venv` 未安装或升级 LangChain。以后重建独立环境时执行：

```bash
.venv/bin/python -m venv langchain_learning/.venv
langchain_learning/.venv/bin/python -m pip install -r langchain_learning/requirements.txt
```

已固定 `langchain-core==1.7.6`、`langchain-openai==1.7.4` 以及本次 pytest 版本。
其余为解析得到的传递依赖；requirements.txt 不是包含所有平台依赖的完整锁文件。
IDE 中运行本日测试时，应选择 `langchain_learning/.venv/bin/python`。
若误用未安装 LangChain 的主环境，本组测试会整体跳过，并提示正确环境。

### 测试数据来源

所有消息、假响应和错误状态都定义在 `tests/test_week8_day1.py`，无需另找数据集：

- **消息样例**：中文任务标识 `00123`，用于观察前导零及原文保留。
- **Fake ChatModel**：LangChain 自带 FakeMessagesListChatModel，返回测试指定的 AIMessage。
- **HTTP Mock**：httpx.MockTransport 返回合成 Chat Completions 响应、usage 和 401/429/500；不访问 DeepSeek。
- **真实测试**：使用同样的消息历史和问题，沿用项目现有 DeepSeek Key；不使用 Week7 知识库。

离线测试的固定回答和 usage 都是合成值，只验证消息及接口处理，不代表真实模型表现或真实成本。
代码关闭本次调用的 LangSmith 远程追踪；HTTP 客户端由已提供的代码管理，异常不自动重试。

## 运行命令与产物

均在仓库根目录执行。

### 1. 只运行已提供代码的自检

```bash
RUN_WEEK8_DAY1=0 RUN_WEEK8_DAY1_LLM=0 \
langchain_learning/.venv/bin/python -B -m pytest tests/test_week8_day1.py \
-q -s -p no:cacheprovider
```

本次框架自检已通过：18 passed、12 skipped。
只执行已提供的参数、SDK 对照、Mock HTTP 及 CLI 保护测试；设置 RUN_WEEK8_DAY1=1 可启用完整流程。

### 2. 完整离线验收

```bash
RUN_WEEK8_DAY1=1 RUN_WEEK8_DAY1_LLM=0 \
langchain_learning/.venv/bin/python -B -m pytest tests/test_week8_day1.py \
-q -s -p no:cacheprovider
```

两处关键步骤已完成，当前仓库实现的完整离线验收结果为 29 passed、1 skipped。
唯一跳过项为真实模型测试。沙箱内的异步线程唤醒受限时需在正常本地终端运行，本日没有为了绕过环境问题改写 LangChain。

### 3. 显式真实调用

完成离线验收后，确保当前终端已配置既有 DeepSeek `OPENAI_API_KEY`：

```bash
RUN_WEEK8_DAY1=1 RUN_WEEK8_DAY1_LLM=1 \
langchain_learning/.venv/bin/python -B -m pytest \
tests/test_week8_day1.py::test_real_model_messages_and_leading_zero \
-v -s -p no:cacheprovider
```

用户已运行真实测试并确认通过，助手已核对本地报告中的实际输入与返回内容。下方命令用于显式重跑。
真实测试会调用一次 DeepSeek，不下载本地模型。通过后仍需阅读 report.json，解释消息顺序、返回类型及 SDK 对照。

也可使用已提供的 CLI，手动观察一次不带历史的调用：

```bash
langchain_learning/.venv/bin/python -B -m langchain_learning.day1 --real-llm \
  --question '字符串任务标识 00123 为什么不应该转成整数？'
```

CLI 的规则使用 DEFAULT_INSTRUCTIONS；history 的传入由真实 pytest 展示。

### 输出位置

| 文件 | 来源 |
|---|---|
| `langchain_learning/artifacts/day1/offline/report.json` | 完整离线测试，假回答及假 usage |
| `langchain_learning/artifacts/day1/real_model/report.json` | 显式真实 pytest 或 CLI 的实际返回 |

pytest 可使用 `WEEK8_DAY1_OUTPUT` 修改产物根目录，CLI 使用 `--output`。
同名报告只在成功后覆盖；失败时旧文件可能仍在，不能据此认定本次成功。产物及独立环境均已 Git 忽略。

## 测试一：SDK 消息对照保留角色和原文

函数：`test_provided_sdk_comparison_preserves_roles_and_text`，默认执行。

- **数据来源**：测试内的规则、任务标识历史和本轮问题。
- **流程**：直接构造 LangChain 消息 → to_sdk_request → 修改返回字典。
- **检查点**：规则独立为 instructions，其他消息按 user/assistant 映射；正文及顺序不变；副本修改不影响原消息。
- **输出**：pytest 断言结果，不生成文件。

## 测试二：不支持的消息不能冒充已转换

函数：`test_provided_sdk_comparison_rejects_unsupported_messages`，默认执行，五组输入。

- **数据来源**：空列表、缺少规则、重复规则、工具消息和内容块。
- **流程**：尝试 SDK 输入对照转换。
- **检查点**：明确抛 ValueError，不把本日未支持的格式悄悄转成文本。
- **输出**：pytest 断言结果。

## 测试三：输入错误在消息组装前报错

函数：`test_provided_input_validation_runs_before_todo`，默认执行，六组输入。

- **数据来源**：空白规则、空问题、错误历史类型、非法 role、空正文和额外字段。
- **流程**：调用 build_messages 的已提供校验部分。
- **检查点**：先抛 ValueError，不依赖后续消息组装和模型调用。
- **输出**：pytest 断言结果，不请求模型。

## 测试四：真实模型配置需要既有凭据

函数：`test_provided_model_requires_existing_key`，默认执行。

- **数据来源**：删除本测试进程的 Key 环境变量，另注入假 Key 配合空模型名。
- **流程**：调用 build_model 的配置检查，不调用模型。
- **检查点**：缺 Key 明确提示；模型名必须非空，不读取 IDE 或磁盘中的其他凭据。
- **输出**：pytest 断言结果。

## 测试五：适配器确实发送正确 HTTP 消息

函数：`test_provided_chat_adapter_sends_roles_to_mock_api`，默认执行。

- **数据来源**：固定 HTTP 成功响应和假 usage，完全使用 MockTransport。
- **流程**：ChatOpenAI.ainvoke → Mock HTTP → AIMessage。
- **检查点**：请求为 DeepSeek 的 chat/completions 路径；消息角色为 system/user；正文保持；返回 content 和 usage 正确。
- **输出**：内存中的请求记录及 pytest 断言结果；不输出凭据。

## 测试六：HTTP 失败不自动重复请求

函数：`test_provided_http_failure_is_not_retried`，默认执行，三种状态。

- **数据来源**：HTTP Mock 的 401、429、500。
- **流程**：调用模型适配器并捕获 APIStatusError。
- **检查点**：保留实际状态码，每种情况只发送一次请求；不伪造文本回答。
- **输出**：pytest 断言结果。

## 测试七：CLI 显式真实开关

函数：`test_provided_cli_requires_explicit_real_flag`，默认执行。

- **数据来源**：提供问题但省略 --real-llm 的子进程命令。
- **流程**：执行独立模块 CLI。
- **检查点**：明确拒绝，不初始化真实模型请求。
- **输出**：子进程提示和 pytest 结果。

## 测试八：Message 类型、顺序与原文

函数：`test_message_order_types_and_original_text`，设置 RUN_WEEK8_DAY1=1 启用，三组历史输入。

- **数据来源**：None、空历史、两条文本历史，以及带空格和换行的规则/问题。
- **流程**：build_messages → 核对消息类和内容。
- **检查点**：规则在前，本轮问题在后；历史 human/ai 顺序准确；不 strip 原文，不修改输入列表。
- **输出**：pytest 断言结果。

## 测试九：一次异步调用和可读报告

函数：`test_one_async_call_records_aimessage_usage_and_sdk_comparison`，设置 RUN_WEEK8_DAY1=1 启用。

- **数据来源**：LangChain Fake ChatModel 返回固定 AIMessage、usage 和响应元数据。
- **流程**：build_messages → ainvoke 一次 → AIMessage → SDK 对照和 JSON 报告。
- **检查点**：实际消息一致，返回类型为 ai，usage 原值保留，只导出规定的元数据字段；不把响应当 SDK RunResult。
- **输出**：`artifacts/day1/offline/report.json`。

## 测试十：模型实例不会自动携带历史

函数：`test_model_instance_does_not_implicitly_keep_history`，设置 RUN_WEEK8_DAY1=1 启用。

- **数据来源**：同一个 Fake ChatModel 的两次调用；第二次没有传 history。
- **流程**：分别运行两个问题，检查第二次的实际输入。
- **检查点**：第二次只有规则和当前问题，不包含第一轮的任务标识；缺失 usage 保持 null。
- **输出**：pytest 断言结果。替身的固定回答不用于证明模型记忆能力，重点是检查实际输入。

## 测试十一：模型异常传播与旧报告保留

函数：`test_model_failure_propagates_and_preserves_old_report`，设置 RUN_WEEK8_DAY1=1 启用，两类异常。

- **数据来源**：RuntimeError、httpx.ReadTimeout 和临时旧报告。
- **流程**：执行 run_query，模型调用时抛错。
- **检查点**：异常原样传播，调用一次，旧文件不被写成假成功。
- **输出**：pytest 临时目录中的 `report.json` 保持原内容。

## 测试十二：错误类型、空响应和工具调用的边界

函数：`test_invalid_or_tool_response_is_not_a_completed_text_answer`，设置 RUN_WEEK8_DAY1=1 启用，四组响应。

- **数据来源**：HumanMessage、空白 AIMessage、内容块 AIMessage、含工具调用的 AIMessage。
- **流程**：替身返回上述消息，run_query 检查。
- **检查点**：本日只接受非空纯文本 AIMessage，不冒充已执行工具或已支持多模态。
- **输出**：pytest 断言结果。

## 测试十三：真实消息与前导零

函数：`test_real_model_messages_and_leading_zero`，需显式开启真实开关。

- **数据来源**：同样的中文历史及问题，既有 DeepSeek 配置。
- **流程**：一次真实 ainvoke → 保存消息、AIMessage 与 SDK 输入对照。
- **检查点**：历史确实发送，回答保留 00123，输出类型为 ai；usage 有则保留，无则不猜。
- **输出**：`artifacts/day1/real_model/report.json`，需阅读实际内容完成概念验收。

## 官方资料

- [LangChain Models](https://docs.langchain.com/oss/python/langchain/models)：模型调用接口与参数。
- [LangChain Messages](https://docs.langchain.com/oss/python/langchain/messages)：消息类型、content 与 usage_metadata。
- [LangChain Unit Testing](https://docs.langchain.com/oss/python/langchain/test/unit-testing)：Fake ChatModel 等离线测试方法。

今天验收时，能拿一份 report.json 解释 Model、Message、SDK Run 各做什么即可；不要求提前做 Runnable 链、工具循环或知识检索。
