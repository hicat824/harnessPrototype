# 工具结果限长与补读

本地工具在返回给 SDK 前限制结果大小。默认上限为 2000 字符，配置位于
`tools.py` 的 `_MAX_TOOL_RESULT_CHARS`。计入完整 JSON，包括转义、来源与游标；
字符数不等于模型 Token 数，也不代表整个会话请求已经受控。

## 返回格式

`readFile` 现在返回 JSON 字符串，原文位于 `excerpt`，不再直接返回文件正文。
`listFiles` 返回 JSON 对象，文件列表位于 `files`，不再直接返回数组。
`applyPatch`、`runSafeTest` 也返回 JSON 字符串，保留原有状态与错误码字段。
Python 调用方可用 `json.loads(result)` 解析；模型直接读取这些字段。

## 文件读取流程

- `readFile(path)`：小文件全文、大文件头部预览，不修改原文件。
- `searchFile(path, query, offset=0, max_matches=20)`：区分大小写的字面量搜索，
  返回真实行号、命中附近片段。`total_matches` 是全文件命中行数，不是词的出现次数。
- `readFileRange(path, start_line, end_line, start_column=1)`：行列从 1 开始，
  行范围两端包含，每次最多 200 行。首行从指定字符列开始读取，后续行从第一列开始。

已知关键词先搜索，再按命中行补读。`truncated=true` 不能当作读完全文；
文件补读使用 `next_line`、`next_column`，因此极长单行也不会反复从头读取。
行列位置按 UTF-8 解码后的字符计数，CRLF 会归一化为换行符。
一页完整只表示本次请求范围完整，不意味着整个文件已读完。

`searchFile` 与 `listFiles(path, offset=0, max_entries=20)` 按条目分页：
`has_more=true` 时使用 `next_offset` 继续，页长仍受 2000 字符上限约束。
搜索总数需要扫描全文件；单行搜索使用 Python 按行读取，此版本没有扫描时间或内存预算。
若文件发生变化，应重新搜索或读取，不复用旧位置作为新版本证据。

## 测试日志与专家结论

`runSafeTest` 保留 `status`、`success`、`exit_code`。长输出保留有限头尾，
全文保存到当前 `EvalContext.workspace/.tool_results/`，返回 `source` 路径。
两个专家通过 `as_tool` 返回 Supervisor 的长结论同样限长并保存全文。
`.tool_results` 是工具生成的记录，可由文件工具补读；暂不自动清理，避免删除仍被引用的日志。
所有日志路径继续经过原有授权检查。

此次只处理本地工具和专家结论，MCP 查询返回值未改动。
历史裁剪与 `contextCompact.py` 的摘要接入仍独立处理。

## 离线验证

```bash
.venv/bin/python -B -m pytest tests -q -p no:cacheprovider
```

测试使用临时目录与脚本化模型，不访问模型服务。测试进程用模拟结果覆盖，
验证真实 SDK 工具结果进入下一次模型请求时的长度，不启动真实 Docker。
