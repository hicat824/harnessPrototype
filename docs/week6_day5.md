# Week6 Day5：MCP 工具结果、权限与可靠性

学习基准：`/home/xumaozhou/Music/agent_engineer_learning_plan_v4_mcp.md`。

## 今天学习什么

1. 区分正常空结果、部分结果、业务错误和 MCP 连接异常。
2. 在服务端输出边界精简大 JSON，保留业务标识、来源、总数和截断标记。
3. 对敏感字段做递归脱敏；身份和权限继续由执行层管理。
4. 对只读 HTTP 请求执行有限重试，明确总尝试次数和不可重试错误。
5. 确保 stdio 的 stdout 只传协议，日志写 stderr，日志中不出现凭据。

沿用两个业务工具及现有文件，不新增业务服务或 Agent。今天使用 `limit` 和前缀截断；不新增 cursor 分页接口。`next_cursor=None` 表示没有实现续页能力。

## 本日实现的代码

下面三个函数位于 `week6.py`，均已完成并接入 HTTP 查询或 MCP 出站路径。表格保留本日练习的输入输出和实现要求，便于复习。

| 函数 | 输入 | 输出 | 需要实现 |
|---|---|---|---|
| `redact_sensitive_data(value)` | JSON 字典、列表或标量 | 不共享嵌套容器的新值 | 忽略字典键大小写，把 `SENSITIVE_FIELDS` 内的字段值替换为 `[REDACTED]`，递归处理列表和字典 |
| `prepare_tool_result(result, max_chars=6000)` | `BusinessResult`、至少 512 的整数预算 | 新 `BusinessResult` | 字段白名单、调用脱敏、精简大字段、列表前缀裁剪、正确标记状态和来源 |
| `request_json_with_retry(client, path, params=None, max_attempts=1)` | 已注入的 HTTPX Client、路径、查询参数、1～3 次总尝试数 | 成功解析的原始响应字典；失败抛原异常 | 仅重试超时、连接失败、429 和 5xx；每次重试重新请求 |

详细校验顺序、状态和异常规则写在函数 docstring 中；注释分别说明递归遍历、分阶段裁剪和有限循环重试。

### 结果限长的具体约定

- 预算是整个业务响应的 JSON 字符数，按 `len(json.dumps(result.model_dump(), ensure_ascii=True))` 计算，与当前 SDK 的结构化工具输出一致；不是 Token 数。
- 任务顶层只保留 `task_id/state/engineCode/params/tags`。列表只保留 `items/total`。
- 短结果保留公开信息；超预算时先删除任务的 `params/tags`，再从列表末尾删除记录。不要截断 JSON 字符串、业务 ID 或来源。
- 内容删减后，成功结果改为 `partial`，并设置 `truncated=True`；保留原 `total`、顺序和来源。
- 一条记录都放不下时，列表仍是 `partial`，不能伪装成没有匹配结果的 `empty`。
- 单任务的最小字段也无法放下时返回 `error/RESULT_TOO_LARGE`。连基础元数据都放不下时抛出 `ValueError`。
- 脱敏只按明确字段名执行，不推测任意自由文本中的秘密；异常正文不直接返回模型。

### 重试与错误分层

```text
HTTP 临时失败 → Client 最多尝试 max_attempts 次 → 耗尽后抛原异常
    ↓
Service 将 HTTP / JSON 错误映射为 BusinessResult(status="error")
    ↓
MCP 返回统一业务结果
```

- Client 默认总尝试数为 1，保持旧测试行为；本地 MCP 启动入口配置为 2。
- 403/404/其他 4xx、JSON 解析错误不重试；空结果不重试。
- MCP 客户端显式设置 `max_retry_attempts=0`，不再叠加一轮重试。
- MCP 断连保留 SDK 异常向上抛出，不返回 `empty`，也不自动重连。
- 本日验证 HTTP/MCP 执行层行为；真实 LLM 是否反复委派属于 Day6 Evals，不用 Mock 结果冒充该项证明。

## 运行方法与完成标准

仓库根目录执行：

```bash
# 只验收 Day5；无需 API Key，无真实网络和 LLM 调用。
RUN_WEEK6_DAY5=1 .venv/bin/python -B -m pytest \
  tests/test_week6.py tests/test_week6_fake_agent.py -q -s -k day5 \
  -p no:cacheprovider --junitxml=/tmp/week6_day5.xml

# Day1～Day5 离线回归；Day4 真实 LLM 用例显式关闭。
RUN_WEEK6_DAY3=1 RUN_WEEK6_DAY5=1 RUN_WEEK6_DAY4_LLM=0 \
  .venv/bin/python -B -m pytest tests/test_week6.py tests/test_week6_fake_agent.py -q \
  -p no:cacheprovider
```

`@DAY5` 默认跳过，含义与此前 `@DAY3` 相同。显式启用后执行验收；跳过不表示通过。

完成标准：Day5 用例全部通过，Day1～Day4 离线回归仍通过。读取 `stdio_report.json`，确认 stdout 是 JSON-RPC，来源和脱敏符合预期。原有 Day4 真实 LLM 验收无需在每次改动后自动重跑。

验收记录：三个函数完成后，完整离线回归 **119 通过、2 跳过**，包含全部 **52 个 Day5 用例**。跳过的是两条 Day4 真实 LLM 用例，本次未调用真实 LLM。

练习起点记录：骨架生成时，原有离线用例 67 通过；Day5 用例 29 通过、23 失败。该历史记录用于对照练习进展，不是当前验收结果。

## 测试 1：递归脱敏及输入不变

函数：`test_day5_redact_sensitive_fields_without_changing_input`。

- 数据来源：测试现场构造的字典和嵌套列表，所有密钥、密码都是合成值。
- 流程：直接调用 `redact_sensitive_data`。
- 检查：大小写变体敏感键被遮蔽，普通字段保留，修改返回值不影响输入。

## 测试 2：短结果保真

函数：`test_day5_short_result_preserves_business_data`。

- 数据来源：现有 `TASKS["12345"]` 的公开字段。
- 流程：短结果经过 `prepare_tool_result`。
- 检查：无故丢字段、改变状态或共享嵌套容器均不通过。

## 测试 3：字段白名单及脱敏接入

函数：`test_day5_unknown_task_fields_are_removed_and_redacted`。

- 数据来源：公开任务中额外注入内部字段和合成 `api_key`。
- 流程：直接经过 MCP 出站准备函数。
- 检查：内部字段移除、敏感值遮蔽、删减标记和来源正确，输入不变。

## 测试 4：大详情限长

函数：`test_day5_large_detail_has_budget_and_original_reference`。

- 数据来源：`_day5_large_task` 合成的长中文参数和备注。
- 流程：以 1024 字符预算准备详情。
- 检查：整个响应符合预算；保留 ID、状态、流程编码、来源及截断标记。

## 测试 5：大列表前缀、总数与顺序

函数：`test_day5_large_list_keeps_prefix_total_and_truncation`。

- 数据来源：现场构造的 30 条大任务。
- 流程：以 1024 字符预算准备列表。
- 检查：保留非空前缀，原总数 30 不变，顺序不变，返回 `partial`。

## 测试 6：没有记录能放下

函数：`test_day5_no_record_fits_is_partial_instead_of_empty`。

- 数据来源：包含超长 ID 的合成单元素列表。
- 流程：以 512 字符预算准备列表。
- 检查：`items=[]/total=1/status=partial/truncated=True`，不能标为 `empty`。

## 测试 7：业务标识不可截断

函数：`test_day5_identifier_cannot_be_cut_to_fit`。

- 数据来源：ID 长度 3000 的合成任务。
- 流程：以 512 字符预算准备详情。
- 检查：明确返回 `RESULT_TOO_LARGE`，不生成一个截断后失真的任务 ID。

## 测试 8：基础元数据也超预算

函数：`test_day5_metadata_itself_cannot_fit`。

- 数据来源：人为放大的来源字符串。
- 流程：预算小于必须保留的元数据。
- 检查：抛出 `ValueError`，不能悄悄截断来源或返回超预算结果。

## 测试 9：非法字符预算

函数：`test_day5_invalid_budget_is_rejected`。

- 数据来源：`0/100/True/"1024"`。
- 流程：直接调用准备函数。
- 检查：拒绝过小、布尔值和字符串预算。

## 测试 10：已有状态及元数据保留

函数：`test_day5_existing_result_states_and_metadata_are_preserved`。

- 数据来源：合成 `empty/error/needs_input/partial` 结果。
- 流程：短结果经过准备函数。
- 检查：状态、总数、来源、错误代码和已有截断标记不被改变。

## 测试 11：瞬时失败后真的重新请求

函数：`test_day5_transient_failure_really_sends_a_second_request`。

- 数据来源：MockTransport 首次超时、连接失败、429 或 503，第二次成功。
- 流程：真实 Client → 重试函数 → MockTransport。
- 检查：发送两次 GET，第二次成功；不是只增加计数或重复读取旧响应。

## 测试 12：尝试次数与最终错误映射

函数：`test_day5_attempt_limit_and_final_service_error`。

- 数据来源：持续超时、连接失败、429 或 503；总尝试数分别为 1、2、3。
- 流程：Client → Service。
- 检查：准确达到上限后结束，并得到正确业务错误，不伪称成功。

## 测试 13：不可恢复 HTTP 错误不重试

函数：`test_day5_non_retryable_http_failure_only_once`。

- 数据来源：Mock HTTP 400、403、404、422。
- 流程：配置 3 次上限后调用 Service。
- 检查：仍只请求一次，403/404 分别映射为 `FORBIDDEN/NOT_FOUND`。

## 测试 14：非法 JSON 不重试、不泄漏正文

函数：`test_day5_invalid_json_does_not_retry_or_expose_body`。

- 数据来源：含合成秘密的非法 JSON 响应。
- 流程：Client 解析失败，Service 统一映射。
- 检查：只请求一次，返回 `INVALID_RESPONSE`，正文秘密不进入结果。

## 测试 15：非法尝试次数

函数：`test_day5_invalid_attempt_limit_sends_no_request`。

- 数据来源：`0/4/True/"2"`。
- 流程：直接调用重试函数，记录 MockTransport 请求。
- 检查：抛 `ValueError`，请求数为 0。

## 测试 16：空结果不是临时故障

函数：`test_day5_empty_result_is_not_retried`。

- 数据来源：详情及列表的正常空响应。
- 流程：Client → Service，配置总尝试上限为 3。
- 检查：只请求一次，保留 `empty` 和空数据。

## 测试 17：两个 MCP 工具都经过结果边界

函数：`test_day5_mcp_boundary_uses_prepared_result`。

- 数据来源：注入 Client 返回的大详情 / 大列表，含合成密钥。
- 流程：实际注册的 MCP 工具 → Service → 结果准备。
- 检查：两个工具都满足默认 6000 字符预算、正确截断且不泄漏秘密。

## 测试 18：服务端读取权限

函数：`test_day5_server_permissions_are_enforced`。

- 数据来源：程序侧 `can_read_tasks=False`，工具仍正常可见。
- 流程：通过 MCP 直接查询。
- 检查：服务端拒绝，返回 `FORBIDDEN`，API 调用数为 0。

## 测试 19：上游任务归属权限

函数：`test_day5_upstream_owner_permission_is_not_bypassed`。

- 数据来源：alice 的客户端查询 bob 的任务 `34567`。
- 流程：MCP → Service → Mock API 的 Token / owner 校验。
- 检查：没有任务数据返回，结果为 `FORBIDDEN`。

## 测试 20：其他协议异常不重试

函数：`test_day5_other_protocol_error_is_not_retried`。

- 数据来源：MockTransport 抛出 `httpx.RemoteProtocolError`。
- 流程：Client 直接查询。
- 检查：原异常继续抛出，只尝试一次，避免所有 RequestError 都被重试。

## 测试 21：耗尽后保留原异常

函数：`test_day5_exhausted_retry_preserves_original_exception`。

- 数据来源：两次抛出同一个合成连接异常。
- 流程：直接调用重试函数。
- 检查：恰好两次请求，最终抛原异常对象，不包装成成功或改换类型。

## 测试 22：MCP 断连语义

函数：`test_day5_disconnect_stays_an_exception`，位于 `tests/test_week6_fake_agent.py`。

- 数据来源：AsyncMock 模拟 SDK 抛出 `MCPError("Connection closed")`。
- 流程：`fakeAgentLite.callMcpTool` 包装层。
- 检查：保留原异常，只调用一次；不生成业务空结果、不重连。这里验证异常处理，不声称已经杀进程模拟真实断网。

## 测试 23：不叠加 MCP 自动重试

函数：`test_day5_mcp_does_not_add_another_retry_layer`。

- 数据来源：记录传入 SDK 构造函数的配置，连接由 AsyncMock 替代。
- 流程：`fakeAgentLite.Init`。
- 检查：MCP 的 `max_retry_attempts=0`，与 HTTP 的总尝试上限分开。

## 测试 24：stdio 协议与日志脱敏

函数：`test_day5_stdio_stdout_is_protocol_and_secrets_are_hidden`。

- 数据来源：测试临时生成的启动脚本，在任务参数中加入 `DAY5-STDIO-SECRET` 合成值。
- 流程：真实本地服务子进程 → initialize → tools/call → 关闭 stdin → 正常退出。
- 检查：stdout 每行都是 JSON-RPC；密钥被遮蔽；stdout/stderr 都不包含合成密钥或 Mock Token。
- 产物：pytest 临时目录下的 `stdio_report.json`，测试输出打印路径。协议收发辅助代码已经提供，不要求你手写。
