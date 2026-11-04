# Week6 Day6：Agents SDK 自主调用 MCP 与本地 Evals

学习基准：`/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md`。

验收记录（2026-11-04）：用户完成真实模型验收，已读取
`/tmp/pytest-of-xumaozhou/pytest-74/test_day6_real_llm_evals0/day6_real_llm_report.json`，
10/10 用例通过；模型为既有 DeepSeek 配置，业务 API 为本地 Mock。
SDK 记录 21 次请求，输入 18253 Token、输出 1666 Token、总计 19919 Token。
这是一次完整真实模型验收记录，不表示统计稳定性或公司线上联调。

## 学习目标

1. 先提取并校验结构化参数，再由同一个业务 Agent 根据自然语言选择 MCP 工具。
2. 缺参时返回 `needs_input` 和缺少的字段，暂停本次查询，不猜测参数或数据。
3. 区分正常空结果、部分结果、API 业务错误与执行边界中止；业务错误由程序直接结束 Runner。
4. 在执行层限制重复调用和本次总调用数，不能只依赖提示词让模型自觉停止。
5. 使用固定问题和验收器记录工具参数、业务结果、错误代码、Token、耗时与通过情况。

继续使用 `week6.py`、`week6_fake_agent.py` 和现有测试文件。不新增业务 API、Supervisor、检索服务或状态图；Day1～Day5 已有代码继续复用。

## 本日代码结构

```text
固定自然语言问题
    ↓
不带工具的参数准备 Agent → Day6QueryParameters
    ↓
程序校验本轮原文依据 → 缺参/无依据直接 needs_input，不进入业务查询
    ↓ 参数通过
复用同一个业务 Agent / Runner.run（每条问题使用新的 Day6Context）
    ↓
SDK 输入护栏 → 限次/查重/失败停止，并复核实际调用参数的原文依据
    ↓
现有 MCP Server → Service → HTTPX Mock
    ↓
SDK 输出护栏按 call_id 记录实际业务结果
    ↓
tool_use_behavior：error 直接组织回答并结束；其他状态继续由模型总结
    ↓
Day6Answer → 单次查询报告 → 测试中的固定验收器
```

`Day6Answer` 是 Agent 的最终回答，`BusinessResult` 是工具结果，两者用途不同。最终回答字段已经提供：

| 字段 | 含义 |
|---|---|
| `status` | success / empty / partial / error / needs_input |
| `summary` | 简短结论；详情查询保留任务状态和流程编码原值 |
| `task_ids` | 实际返回的任务 ID，保持顺序；无数据时为空列表 |
| `total` | 列表的原匹配总数；详情或未查询时为 null |
| `source` | 工具返回的来源原值；没有查询时为 null |
| `truncated` | 是否只返回了部分数据 |
| `error_code` | API 错误、缺参或执行边界代码；无错误时为 null |
| `missing_fields` | 缺少的参数名；非缺参状态为空列表 |

这里的“暂停追问”是返回 `needs_input` 并结束本次 Runner，不是挂起线程等待输入，也不实现 Checkpoint。用户补参后的调用仍由调用方发起。

工具业务错误的结束动作由 `day6_tool_use_behavior` 保证：使用原始 `error_code/source/truncated` 生成 `Day6Answer`，正常结束 Runner，且不再请求模型。实体提取、首次业务工具选择和成功结果总结仍交给模型；缺参或无依据时由程序生成追问。现有护栏继续限制重复、超额和异常的失败后调用。

## 参数准备与来源校验

新增代码都在 `week6_fake_agent.py`，测试端不替 Agent 提供参数。

| 名称 | 输入 / 输出与作用 |
|---|---|
| `Day6QueryParameters` | tool_name、可空 task_id/engine_code、limit（默认 100，整数 1～100） |
| `DAY6_PARAMETER_INSTRUCTIONS` | 参数准备提示词；模型没有 MCP 或本地工具，只能返回结构化参数 |
| `validate_day6_query_parameters` | 输入参数对象和本轮问题，返回需补充/确认的字段列表；通过返回 [] |
| `day6_needs_input_answer` | 输入需确认字段，返回统一 Day6Answer，不编造任务或来源 |

先执行一次不带工具的参数准备请求。程序检查必填标识符是否非空、task_id 是否为 ASCII 数字串，以及标识符是否完整出现在用户原文中。不能将 `import_std_extra` 的一部分当作 `import_std`，也不能将 `12345` 的一部分当作 `123`。不使用 Mock 编码白名单，因此用户明确提供但没有记录的编码仍可查询并返回 empty。

准备未通过时直接返回 needs_input，不请求业务 Agent。通过后，将已核验参数连同原问题交给业务 Agent；实际调用还会基于原用户问题再次校验来源，不能用程序追加的文本作为来源。无依据的调用以 `UNCONFIRMED_PARAMETER` 阻止，并返回追问。此时仍保留 blocked_calls、runner_completed=False，原业务 Evals 继续判这次轨迹失败，不能用拦截成功掩盖模型偏差。

**适用范围**：当前只支持本轮显式给出的原值，不支持多轮指代或别名转换。原文匹配核验的是值的直接依据，不是完整语义理解，不能保证识别所有否定句、多个候选参数或数字在句子中的角色；limit 校验类型和范围，不在这里做全文语义核验。这些边界需要新增明确用例后再扩展，不能宣称来源校验解决了所有模型提取错误。

参数准备增加一次模型请求，但不增加 MCP 连接或业务 API。两阶段使用同一个模型对象和本次 wrapper，报告中的 usage/耗时包含两阶段。业务 Agent 仍复用；参数准备 Agent 每次在函数内构建，不增加独立进程、Supervisor 或文件。

## 三个核心函数（已实现）

都在 `week6_fake_agent.py`，以下保留各自的输入、输出和处理边界，方便复习。

### 1. `check_day6_tool_call(history, tool_name, arguments, max_calls=2)`

- 输入：本次已放行工具调用的列表、下一次工具名和参数、总调用上限。
- 输出：允许时返回 `None`，否则返回一个阻止原因字符串。
- 核心逻辑：**先检查已有终止错误，再检查相同调用，最后检查总次数**。
- 错误顺序：`TERMINAL_TOOL_ERROR` → `REPEATED_TOOL_CALL` → `TOOL_CALL_LIMIT`。
- 字典内容相等就是相同参数，不能受键顺序影响。不要修改输入历史。
- `max_calls` 只允许 1～3 的真正整数，非法值抛 `ValueError`。

每条 history 中包含 `name/arguments/result/call_id`。工具尚未完成时 `result=None`，完成后是实际 `BusinessResult` 字典。SDK 适配器负责登记，检查函数只判断。

### 2. `build_day6_agent(model, mcp_server)`

- 输入：已经配置的 SDK Model，以及已经连接的 MCP 客户端。
- 输出：一个 Agent，整批问题复用该对象。
- 设置 `DAY6_INSTRUCTIONS`、`output_type=Day6Answer`、`mcp_servers=[mcp_server]`。
- 设置 `ModelSettings(parallel_tool_calls=False)`，不重复注册 Function Tools。
- 给 MCP 的 `tool_input_guardrails`、`tool_output_guardrails` 接上提供的两个适配器。
- 保持 `use_structured_content=True`，使工具输出可以按业务 JSON 记录。
- 设置 `tool_use_behavior=day6_tool_use_behavior`；它只在业务 `error` 时直接结束。

模型由调用方注入：离线使用 `ScriptedModel`，真实评估使用既有 DeepSeek Responses 模型。函数不读取参考答案、不根据测试编号匹配问题，也不负责连接资源的开关。

### 3. `run_day6_query(agent, question, max_tool_calls=2)`

- 输入：同一个 Agent、一条自然语言问题、本次调用上限。
- 输出：普通字典，包含 answer、Trace、usage、耗时等字段，完整字段说明在 docstring 中。
- 核心逻辑：**创建独立上下文 → 准备参数 → 程序校验 → 追问或执行业务 Runner → 组装报告**。
- 每次新建 `Day6Context` 和 `RunContextWrapper`，不能复用上一条问题的计数和 usage。
- 参数准备 Runner 最多 1 轮，业务 Runner 最多 6 轮；两者均禁用 tracing、不传 Session。这个轮数限制分别作用于两阶段。
- 护栏触发时，无依据参数返回 needs_input，其他限制返回对应边界错误；最大轮数触发时返回 `MAX_TURNS_EXCEEDED`。
- 其他异常保留原类型抛出，不能吞掉程序错误或伪装成查询成功。
- 用 `time.perf_counter()` 记录本次执行耗时，读取 `wrapper.usage`，不要估算 Token。

调用限制、报告数据类及护栏适配均在程序侧，不作为 MCP 参数提供给模型。`RunContextWrapper` 可由当前 SDK 直接接收；其上下文不会自动发给模型或跨进程传给 Service。

## 两类测试证据

| 运行方式 | 模型 | 业务 API | 能证明什么 |
|---|---|---|---|
| 脚本模型验收 | 预设输出的 ScriptedModel | 真实本地 MCP + HTTPX Mock | SDK 接线、Trace、边界控制、报告与验收器可复现 |
| 真实模型 Evals | 真实 DeepSeek | 同一套本地 MCP + HTTPX Mock | 自然语言实体抽取、工具选择、追问和回答是否正确 |

脚本模型不理解自然语言，参数准备、工具调用及成功/空/部分答复由测试预设，缺参答复由实际校验逻辑生成，业务错误答复由回调生成；不能拿离线全绿证明真实 LLM 选择正确。脚本 usage 标记 `synthetic`；真实报告标记 `provider_reported`。真实模型仍需显式启用，不自动读取 PyCharm 凭据或调用公司系统。

## 固定 10 条业务用例

数据来源：`week6.py` 已有 TASKS 和故障 ID。自然语言问题及人工预期集中在 `tests/test_week6_fake_agent.py` 的 `DAY6_CASES`，不会注入真实 Agent 的 Instructions。

| 编号 | 场景 | 预期调用 / 结果 |
|---|---|---|
| D6-01 | 查询任务 12345 状态和流程编码 | query_task；success，RUNNING / import_std |
| D6-02 | 查询 import_std 前 3 条 | query_tasks_by_engine；partial，3 个 ID，总数 12 |
| D6-03 | 缺任务 ID | 不调用工具；needs_input，missing_fields=[task_id] |
| D6-04 | 缺流程编码 | 不调用工具；needs_input，missing_fields=[engine_code] |
| D6-05 | 查询无权访问的 34567 | 调用一次；error / FORBIDDEN |
| D6-06 | 查询不存在的 99999 | 调用一次；error / NOT_FOUND |
| D6-07 | 查询 engine_not_found | 调用一次；empty，总数 0 |
| D6-08 | 查询超时 ID 22222 | 调用一次 MCP；error / TIMEOUT |
| D6-09 | 查询连接失败 ID 44444 | 调用一次 MCP；error / CONNECTION_ERROR |
| D6-10 | 要求对不存在的任务无限重查 | 仍只查询一次；error / NOT_FOUND，不服从无限循环要求 |

“调用一次 MCP”不等于发送一次 HTTP：Day5 Client 的内部有限重试仍存在。Day6 只限制模型发起的 MCP 工具调用。

## 测试 1：验收器自检

函数：`test_day6_grader_self_check`，默认运行。

- 数据来源：人工构造的正确报告，以及删掉调用、改错来源/ID/状态等反例。
- 流程：直接调用已提供的本地验收器和保存函数，不执行 TODO 或模型。
- 检查：反例不会假通过，合成 usage 标记正确，未知 usage 保持 null。
- 产物：pytest 临时目录中的 `grader_self_check.json`、`missing_usage.json`、`invalid_usage.json`。

## 测试 2：SDK 护栏适配器接线

函数：`test_day6_provided_guardrail_wiring`，默认运行。

- 数据来源：ScriptedModel 的预设调用，加真实本地 MCP 查询 12345。
- 流程：测试直接构建最小 Agent，并将策略临时替换为放行，验证已提供的 SDK 适配代码。
- 检查：输入参数和实际输出按 call_id 登记；结构化回答可解析；usage 能从同一个 wrapper 读取。
- 边界：本测试不验收调用策略 TODO，也不能证明真实模型会自主选择工具。

Day6 的 Runner 测试使用临时的无远程导出器追踪器，避免 SDK 初始化追踪器时读取本机代理；测试结束后恢复原对象，不修改系统代理。

## 测试 3：调用限制规则

函数：`test_day6_call_policy`。

- 数据来源：合成调用历史，包括成功、进行中、失败和参数键顺序变体。
- 流程：直接调用 `check_day6_tool_call`。
- 检查：三类阻止原因及优先顺序正确，历史不被修改，不同参数的正常调用可以放行。

## 测试 4：非法调用上限

函数：`test_day6_invalid_call_limit`。

- 数据来源：`0/4/True/"2"`。
- 流程：直接调用策略函数。
- 检查：拒绝非法配置，抛 ValueError。

## 测试 5：Agent 配置

函数：`test_day6_agent_configuration`。

- 数据来源：本地 MCP 客户端及无预设步骤的 ScriptedModel。
- 流程：调用 `build_day6_agent`。
- 检查：正确模型、同一个 MCP、Day6Answer、两个护栏、非并行设置；没有重复 Function Tools。

## 测试 6：脚本模型跑 10 条固定用例

函数：`test_day6_scripted_ten_case_evals`。

- 数据来源：DAY6_CASES、预设模型调用及既有 Mock API。
- 流程：共用一个 Agent，依次执行新 Runner；每个用例后保存报告。
- 检查：Trace、参数、实际业务结果、结构化答复、来源、调用数、错误、usage 和耗时一致。含准备阶段：成功/空/部分结果请求模型 3 次，业务错误 2 次，缺参 1 次；每条问题独立计数。
- 错误用例在参数准备后只预设首次工具调用，不提供错误后的模型回答，确保程序直接结束；这证明执行层终止正确，不证明模型会自主停止。
- 产物：`day6_scripted_report.json`，路径由 pytest 打印。

## 测试 7：执行前阻止重复、失败后续调和超额调用

函数：`test_day6_execution_guard_blocks_before_mcp`。

- 数据来源：故意预设重复相同调用、失败后换参数继续调用、超过 2 次上限的脚本模型。此测试的问题明确给出所有使用的 ID，独立覆盖调用限制，避免因没有参数来源而提前拦截。
- 流程：真实 Runner 和 MCP；使用 spy 记录实际到达 MCP 的调用次数。
- “失败后继续调用”这一条专门关闭直接结束回调，使用 `run_llm_again` 强制覆盖备用护栏路径；正常 Evals 仍使用正式回调。
- 检查：护栏触发、Runner 受控中止、错误代码正确；被拒绝的请求没有到达 MCP；中止前的 usage 仍可读取。
- 注意：这证明执行边界有效；若真实模型在正常 Evals 中触发护栏，依然应判该业务用例失败，不能因为“拦住了”就算模型选择正确。

## 测试 8：同一 Agent 的计数隔离

函数：`test_day6_counts_reset_for_each_question`。

- 数据来源：连续两次查询同一个合法任务，仍是脚本模型。
- 流程：复用 Agent，调用两次 `run_day6_query`。
- 检查：每次只有本次 Trace、1 次工具调用和本次 usage，不被上一条的查重或预算干扰。

## 测试 9：最大轮数与未知错误

函数：`test_day6_max_turns_and_unexpected_errors`。

- 数据来源：AsyncMock 模拟 SDK 最大轮数异常和程序错误。
- 流程：运行单次查询框架。
- 检查：max_turns=6、禁用远程 tracing；轮数耗尽返回明确错误，其他错误原样抛出。

## 测试 10：非法运行配置不调用模型

函数：`test_day6_invalid_run_limit_never_calls_model`。

- 数据来源：非法上限 0、4、True、字符串 2。
- 流程：Mock Runner，调用查询框架。
- 检查：在调用模型前拒绝，而不是已经付费请求后才发现参数错误。

## 测试 11：真实 LLM 的 10 条固定 Evals

函数：`test_day6_real_llm_evals`，默认跳过。

- 数据来源：同一批 DAY6_CASES；业务接口仍然是本地 Mock。
- 流程：共用同一个真实 DeepSeek 模型与 Agent，逐条发送自然语言问题；有失败也继续记录剩余用例，最后汇总判断。
- 检查：参数准备、工具名与参数、真实工具结果、缺参追问、失败停止、来源支持、调用数及成本。
- 边界：业务错误和参数未确认时的结束由程序完成；参数准备模型仍可能提取错误，应结合 parameter_validation.proposed 审核，不能把系统返回正确理解为模型每一步都正确。
- 产物：`day6_real_llm_report.json`，测试启动时打印路径，每条用例结束后刷新文件。
- 异常记录：`execution_error` 为类型，`execution_error_message` 为错误说明，
  `execution_traceback` 保留堆栈及异常链。测试会在执行时观察 Runner 上下文，
  并在校验前保存模型响应，异常时记录 `tool_calls`、`blocked_calls`、已统计的
  `usage` 和 `model_outputs`；观察数据缺失时再尝试 SDK 的 `run_data`。
  模型输出项可用于检查最终 JSON 或工具参数，
  不等于完整 HTTP 响应；异常未返回的模型输出不能凭空补齐。
- 可用性：`diagnostics_source` 标记 test_observer / sdk_run_data；
  `run_data_available` 表示 SDK 是否附带运行数据，`tool_trace_available` 表示
  是否取得本次调用记录。两者不必相同：SDK 可能清理异常附带的数据，测试观察器
  仍能保留现场。无法取得的 Trace、调用数、usage 或模型输出为 null，不伪造零值。
- 控制台只打印脱敏后的简短错误；报告保存完整诊断。复用 Day5 递归字段脱敏，
  并替换文本中的当前 API Key 和已知 Mock Token，不保存请求头或 HTTP 客户端。
- 限制：关键字段的机器检查属于初筛，summary 是否清楚、追问是否合理仍需人工阅读。单次通过不代表统计稳定性或生产验收。

## 测试 12：普通异常记录与脱敏

函数：`test_day6_exception_record_without_run_data`，默认运行。

- 数据来源：人工构造的 ValueError → RuntimeError 异常链，以及合成密钥。
- 流程：调用测试侧异常记录函数，再保存 JSON，不请求模型或 MCP。
- 检查：错误说明和异常链保留；已知密钥被脱敏；未知 Trace 和 usage 为 null；仍判失败。
- 产物：pytest 临时目录中的 `day6_exception_unknown.json`。

## 测试 13：SDK 异常保留失败前的现场

函数：`test_day6_exception_record_with_sdk_run_data`，需开启 Day6 离线验收。

- 数据来源：真实本地 MCP 的 12345/RUNNING 成功结果，以及脚本模型故意生成的无效 JSON。错误结果已由程序直接结束，因此用成功查询覆盖模型回答解析异常。
- 流程：测试侧观察实际 Runner 的上下文和模型响应，在解析最终回答时触发 ModelBehaviorError，再保存异常现场。
- 检查：实际调用及 call_id、业务结果、失败前已统计的 usage、模型输出和堆栈均可查看；仍判失败，下一条问题的诊断不混入本条数据。
- 产物：pytest 临时目录中的 `day6_exception_sdk.json`，usage 标记 synthetic。

## 测试 14：标识符来源校验

函数：`test_day6_query_parameter_source`。

- 数据来源：合成参数与用户问题，覆盖缺失值、占位值、部分字符串、前导零和未知编码。
- 流程：直接调用 `validate_day6_query_parameters`，不执行模型或 MCP。
- 检查：完整原值才可放行；不修改参数对象；不以 Mock 数据白名单代替来源校验。

## 测试 15：准备阶段缺参或无依据时不查询

函数：`test_day6_preparation_missing_source_never_calls_mcp`。

- 数据来源：同一个缺编码问题，脚本模型分别提取 null、占位符和编造值。
- 流程：实际参数准备 Runner → 程序校验，spy 观察 MCP 调用。
- 检查：只请求模型一次，直接追问，业务查询和 blocked_calls 均为空，MCP 没有收到调用；报告保留原提议值及校验结果。

## 测试 16：实际工具参数再次校验

函数：`test_day6_actual_tool_parameters_are_checked_again`。

- 数据来源：合法的参数准备结果，加业务模型随后生成的无依据占位参数。
- 流程：实际两阶段 Runner 与输入护栏，spy 观察 MCP 调用。
- 检查：MCP 调用数为 0，返回 needs_input，保留 UNCONFIRMED_PARAMETER 阻止记录；原业务验收仍判轨迹失败。

## 如何运行

仓库根目录执行：

```bash
# 先检查提供的验收器和 SDK 接线；无 API Key、无外网请求。
.venv/bin/python -B -m pytest tests/test_week6_fake_agent.py -q -s \
  -k 'day6_grader_self_check or day6_provided_guardrail_wiring' -p no:cacheprovider

# 执行 Day6 离线验收。真实模型仍明确关闭。
RUN_WEEK6_DAY6=1 RUN_WEEK6_DAY6_LLM=0 .venv/bin/python -B -m pytest \
  tests/test_week6_fake_agent.py -q -s -k day6 -p no:cacheprovider

# Day1～Day6 离线回归。
RUN_WEEK6_DAY3=1 RUN_WEEK6_DAY5=1 RUN_WEEK6_DAY6=1 \
RUN_WEEK6_DAY4_LLM=0 RUN_WEEK6_DAY6_LLM=0 .venv/bin/python -B -m pytest \
  tests/test_week6.py tests/test_week6_fake_agent.py -q -p no:cacheprovider

# 最后显式启用真实模型评估，需要已有可用的 DeepSeek OPENAI_API_KEY。
RUN_WEEK6_DAY6=1 RUN_WEEK6_DAY6_LLM=1 .venv/bin/python -B -m pytest \
  tests/test_week6_fake_agent.py -q -s -k day6_real_llm_evals -p no:cacheprovider
```

`@DAY6` 是练习开关，默认跳过 Day6 功能验收；真实评估另有 `RUN_WEEK6_DAY6_LLM` 开关。当前三个核心函数已实现，启用开关后执行对应验收，不要把 skip 计作通过。

## 如何读报告与判定完成

- `tool_calls` 是输入护栏放行的调用，包含 name、arguments、call_id 和实际 result。
- `blocked_calls` 是护栏在执行前拒绝的请求，单独记录阻止原因，不算已经执行的 MCP 请求。
- `parameter_validation` 包含 proposed、passed、missing_fields。passed=False 表示尚不具备查询条件；proposed 中的 null 可以是正确识别缺参，不能自动视为模型错误。非空但无依据的提议也会保存，便于审核模型提取偏差。
- API 业务 error 由 `tool_use_behavior` 正常结束 Runner，因此 `runner_completed=True`，保留原业务错误码；护栏中止时为 False，记录执行边界代码。不能混淆两者。
- `usage` 是模型请求的 SDK usage。脚本值是合成值，真实值来自模型返回；异常时从本次上下文保留已统计值，不能理解为未返回用量的请求也被完整计入；未知值保留 null。
- `elapsed_seconds` 包含参数准备、业务 Runner 的工具发现和调用，不包含共享 MCP 服务启动。
- 正常用例的错误类别直接看 answer.error_code；未被业务映射的 SDK/模型异常看 execution_error、execution_error_message 和 execution_traceback，再结合工具 Trace、model_outputs 排查，不能伪装成业务成功。

完成标准：三个核心函数实现；Day6 离线功能验收及旧回归通过；真实模型 10 条固定用例通过并保存报告；人工检查缺参追问和失败结论有依据。所有证据都标记真实模型 / 脚本模型、Mock API 的边界，以及程序直接结束与模型回答的区别。
