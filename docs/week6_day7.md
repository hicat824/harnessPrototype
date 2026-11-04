# Week6 Day7：业务 MCP 查询 Agent V1 周项目

学习基准：`/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md`。

Day6 已完成真实模型的 10 条固定验收。本日将已有实现整理为一个可重复运行的周项目入口，补齐整合报告和资源清理验收，不增加新的业务 API。

验收记录（2026-11-04）：完整周离线回归为 171 passed、4 skipped（真实模型测试）。
用户已运行真实 DeepSeek 演示，报告为
`/tmp/pytest-of-xumaozhou/pytest-79/test_day7_real_llm_demo0/day7_real_llm_demo.json`；
已重新用测试验收器核对，四个场景全部通过。SDK 记录 9 次请求，输入 8273 Token、
输出 850 Token、总计 9123 Token；报告记录总耗时约 10.57 秒。
业务 API 仍是本地 HTTPX Mock，不代表公司线上接口联调。

## 学习目标

1. 能说明自然语言问题如何经过 Agents SDK、MCP Client、MCP Server、BusinessService、HTTPX Mock，再得到结构化回答。
2. 整合已有参数准备、来源校验、工具选择和错误停止，避免重复实现业务逻辑。
3. 明确 MCP 连接、Agent 和模型 HTTP 客户端分别由谁创建和关闭。
4. 保存包含实际查询记录、来源、错误、调用数、Token 和耗时的周项目报告。
5. 区分脚本模型、真实模型与 Mock API 的证据范围，结合已有失败复盘解释程序边界。

## 验收目标

| 项目 | 完成标准 |
|---|---|
| 单一业务执行层 | 继续调用已有 Service 和 Client，不重新写 HTTP 或注册重复 Function Tools |
| 两个只读 MCP 工具 | 实际发现且只发现 query_task、query_tasks_by_engine |
| 完整演示 | 同一入口运行详情成功、列表部分结果、缺参追问、403 权限失败四个场景 |
| 参数与过程可追溯 | 每条记录保留 Day6 的参数校验、实际工具参数、结果及阻止记录 |
| 生命周期 | 整批只建立一个 MCP 连接、构建一个业务 Agent；正常和异常路径均关闭连接 |
| 报告 | 返回值与 UTF-8 JSON 文件一致；用量按实际记录累计；证据类型明确 |
| 旧功能回归 | Day1～Day6 的离线测试全部通过，权限、非法参数和网络异常不退化 |
| 真实模型演示 | 显式启用真实 DeepSeek，业务仍使用本地 Mock；四个场景通过并保存报告 |

`completed` 仅表示查询流程正常返回，不等于回答质量通过。403 等业务错误可以正常完成；触发执行护栏时应保留 `completed=false`。测试对实际答复和调用记录另外验收。

## 本日文件与调用结构

沿用三个已有 Python 文件和一个新说明文件：

```text
week6.py                       Client / Service / MCP Server / Mock 数据，继续复用
week6_fake_agent.py             run_day7_demo 周项目整合入口
tests/test_week6_fake_agent.py  新增 Day7 整合测试
docs/week6_day7.md              本说明与待填写的学习复盘
```

调用流程：

```text
测试或调用方创建 Model，并提供自然语言问题
    ↓
run_day7_demo
    ├─ 校验批次配置
    ├─ fakeAgentLite.Init → 本地 stdio MCP Server
    ├─ showMcpTools → 检查两个工具
    ├─ build_day6_agent → 仅构建一次业务 Agent
    ├─ 逐问调用 run_day6_query
    │    ├─ 无工具的参数准备 + 原文来源校验
    │    └─ 业务 Agent → MCP → Service → HTTPX Mock
    ├─ finally：关闭 MCP
    └─ 汇总并保存 JSON，返回同一个报告
```

Day6 的参数准备模型调用仍逐问执行，业务 Agent 和 MCP 连接在整批中复用。没有共享 Session，也不加入跨问题记忆。

## 本日整合函数（已实现）

在 `week6_fake_agent.py` 中查看 `run_day7_demo`：

```python
async def run_day7_demo(
    questions: list[str],
    model: Model,
    output_path: str | Path,
    *,
    model_mode: Literal["scripted", "real_llm"],
) -> dict:
    ...
```

### 输入与所有权

| 参数 | 规则 |
|---|---|
| questions | 非空列表，元素是非空、非纯空白字符串；保持输入顺序与原文 |
| model | 调用方创建的 Model，函数内部直接复用，不读取 API Key |
| output_path | JSON 文件路径，支持字符串或 Path；父目录不存在时创建 |
| model_mode | 只能是 scripted / real_llm，标记本次模型证据类型 |

questions 或 model_mode 不合法时，在连接 MCP 前抛 `ValueError`。函数负责关闭自己创建的 MCP 连接；传入模型的 HTTP 客户端由调用方负责关闭。

### 处理规则

1. 用一个 `fakeAgentLite` 管理本次连接，`Init` 和全部查询放在 `try/finally` 中。
2. 校验发现的工具恰好是两个只读查询；不符合时抛 `RuntimeError`。
3. 调用一次 `build_day6_agent`，按顺序执行 `run_day6_query` 并保留原记录。
4. 未受控异常直接抛出，不能吞掉或自动重跑整批；`finally` 仍清理连接。
5. 全部查询正常返回并关闭连接后，组装报告、创建父目录、写盘并返回。

本函数不包含预期答案或验收器，不改写 Trace 来让测试通过。初始化、发现或查询发生未受控异常时，本函数不保证生成批次报告；可使用 pytest 堆栈或已有 Day6 定位测试继续排查。

### 输出报告

| 字段 | 类型与含义 |
|---|---|
| business_backend | 固定为 httpx_mock，不能写成公司线上 API |
| transport | 固定为 stdio |
| model_mode | 调用方提供的模式 |
| usage_kind | scripted 对应 synthetic；real_llm 对应 provider_reported |
| tools | 实际发现的工具名称列表 |
| query_count | 本批问题条数 |
| results | `run_day6_query` 返回的原记录列表，按问题顺序排列 |
| completed | 所有记录的 runner_completed 均为 True 时才为 True |
| usage_total | 累加 requests、input_tokens、output_tokens、total_tokens |
| elapsed_seconds | Init 前到 close 后的 perf_counter 差值，包含生命周期，不包括报告写盘 |

使用 `json.dumps(..., ensure_ascii=False, indent=2)` 和 UTF-8 保存。每条 Day6 usage 已包含参数准备，不要再次额外加一次请求或估算 Token。

## 四个演示场景的数据来源

业务数据继续来自 `week6.py` 的 TASKS、MockTransport 和权限规则；问题与预期由测试中的 `DAY7_CASES` 引用已有 Day6 用例。运行入口只接收问题和模型，不能读取测试预期。

| 场景 | 输入 | 预期 |
|---|---|---|
| 详情成功 | 查询任务 12345 的状态和流程编码 | success；RUNNING / import_std；有工具来源 |
| 部分列表 | 列出前 3 个 import_std 任务 | partial；ID 为 12345、20000、20001，总数 12 |
| 缺参追问 | 未提供流程编码，要求最多 3 条 | needs_input；MISSING_PARAMETER；无 MCP 查询 |
| 权限失败 | 查询任务 34567 | error / FORBIDDEN；执行层直接结束 |

脚本模型按“准备参数 → 业务调用 → 必要时回答”预设步骤，四问合计 9 次合成模型请求。真实模型报告使用实际 SDK usage，不硬编码为 9。业务权限由 Service/Mock API 执行，不由模型声明授权。

## 测试一：完整本地整合与生命周期

函数：`test_day7_full_mock_demo`。

- 数据来源：上述四个场景、ScriptedModel、本地真实 MCP 子进程、已有 HTTPX Mock。
- 流程：调用你实现的入口，复用真实查询链路；测试 spy 只观察，不替换业务结果。
- 检查：两工具、四问结果、原始 Trace、合成 usage、同一个连接/业务 Agent；实际 MCP session 关闭；嵌套输出目录创建；文件内容等于返回报告。
- 产物：pytest 临时目录中的 `nested/day7_mock_demo.json`，运行时打印路径。
- 证据边界：证明整合流程正确，不能证明真实模型理解自然语言。

## 测试二：非法批次配置在连接前拒绝

函数：`test_day7_invalid_input_before_connect`。

- 数据来源：空列表、字符串代替列表、None、空白问题、非字符串元素和未知模式。
- 流程：使用明确标注的 AsyncMock/Mock 依赖，调用整合入口。
- 检查：抛 ValueError；没有初始化连接、构建 Agent、请求查询或保存伪报告。
- 产物：不应生成 `invalid.json`。

## 测试三：未受控异常也清理连接

函数：`test_day7_unexpected_error_closes_mcp`。

- 数据来源：替身分别在初始化、工具发现、查询阶段抛出同一个合成 RuntimeError。
- 流程：运行入口，观察 finally 行为。
- 检查：异常对象原样抛出，close 被执行，未执行后续阶段，不生成假成功报告。
- 边界：这是生命周期分支测试，实际 MCP 关闭由测试一及既有 Day4 测试另行证明。

## 测试四：拒绝意外扩大的工具范围

函数：`test_day7_unexpected_tool_scope_is_rejected`。

- 数据来源：合成发现结果，额外包含 write_task；不会真的注册或执行写工具。
- 流程：工具发现后检查范围。
- 检查：抛 RuntimeError，关闭连接，不构建业务 Agent 或继续查询。

## 测试五：受控中止不冒充正常完成

函数：`test_day7_controlled_stop_is_not_claimed_completed`。

- 数据来源：人工构造的 Day6 护栏中止记录，不执行实际模型。
- 流程：替身返回该记录，入口保存周项目报告。
- 检查：completed=False，原始阻止记录和 usage 不被改写，仍关闭连接并保存报告。
- 产物：pytest 临时目录中的 `controlled_stop.json`。

## 测试六：真实模型的周项目演示

函数：`test_day7_real_llm_demo`，默认跳过。

- 数据来源：相同四个自然语言问题；真实 DeepSeek；业务仍为本地 Mock。
- 流程：调用方创建并关闭模型客户端，你的入口管理 MCP，复用同一周项目函数。
- 检查：两工具、四问回答、工具参数、来源、错误、调用轨迹和真实 usage；保存内容等于返回报告。
- 产物：pytest 临时目录中的 `day7_real_llm_demo.json`，运行前打印路径。
- 边界：未做公司线上接口联调。未知模型/网络异常会抛出，不能伪称演示完成。

## 如何运行

所有命令均在仓库根目录执行。Day7 测试默认跳过，需要显式开启；当前整合函数已经实现，不要把 skip 当成验收通过。

```bash
# 1. 查看本日测试清单，不执行。
RUN_WEEK6_DAY7=1 .venv/bin/python -B -m pytest \
  tests/test_week6_fake_agent.py --collect-only -q -k day7 -p no:cacheprovider

# 2. 执行 Day7 离线整合验收；不调用真实 LLM。
RUN_WEEK6_DAY7=1 RUN_WEEK6_DAY7_LLM=0 .venv/bin/python -B -m pytest \
  tests/test_week6_fake_agent.py -q -s -k day7 -p no:cacheprovider

# 3. 完整周回归，包含旧的权限、非法参数、网络异常、脱敏和调用边界测试。
RUN_WEEK6_DAY3=1 RUN_WEEK6_DAY5=1 RUN_WEEK6_DAY6=1 RUN_WEEK6_DAY7=1 \
RUN_WEEK6_DAY4_LLM=0 RUN_WEEK6_DAY6_LLM=0 RUN_WEEK6_DAY7_LLM=0 \
  .venv/bin/python -B -m pytest tests/test_week6.py tests/test_week6_fake_agent.py \
  -q -p no:cacheprovider

# 4. 最后运行真实演示；使用已有 DeepSeek OPENAI_API_KEY。
RUN_WEEK6_DAY7=1 RUN_WEEK6_DAY7_LLM=1 .venv/bin/python -B -m pytest \
  tests/test_week6_fake_agent.py::test_day7_real_llm_demo -q -s -p no:cacheprovider
```

本日用 pytest 作为演示调用入口，不要求新增 CLI、部署服务或拆分目录。pytest 临时产物可能被清理；需要长期保留时，将 JSON 复制到自己的归档目录，或在自己的调用代码中传入固定 output_path。

## 学习复盘：你填写

结合本日真实报告和 [Day6 失败复盘](week6_day6_review.md)，每项写 2～3 句话即可：

1. API Client、BusinessService、MCP Server、业务 Agent 各自负责什么？
2. 为什么“格式合法”不等于“参数来自用户”？来源校验还有哪些边界？
3. needs_input、业务 error、护栏中止和 SDK 异常分别如何表现？
4. 本次演示哪些部分是真实模型，哪些部分为 Mock？completed 能证明什么？

完成标准：整合函数实现、Day7 离线验收与完整周回归通过、真实演示通过并保留报告、完成以上简短复盘。只准备代码框架或得到脚本模型全绿，还不能标记 Day7 完成。
