"""使用真实 SDK 工具入口验证预算、分页和补读，不访问模型服务。"""

import asyncio
import json
import subprocess
import uuid

import pytest
from agents.tool_context import ToolContext
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

import tools
from baseModel import FileReadOutput, ToolOutputBase


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    # 参数错误也经过真实 SDK 入口，离线测试不创建远程 tracing exporter。
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    root = tmp_path / "authorized"
    root.mkdir()
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", root)
    return root


def invoke(tool, state, **arguments):
    payload = json.dumps(arguments)
    context = ToolContext(
        context=state, tool_name=tool.name,
        tool_call_id=uuid.uuid4().hex, tool_arguments=payload,
    )
    output = asyncio.run(tool.on_invoke_tool(context, payload))
    assert isinstance(output, str)
    assert len(output) <= 2000
    return json.loads(output)


@pytest.mark.parametrize("text", ["", "中文源码\nprint('hello')\n", " \n\t  缩进\t \n\n"])
def test_small_file_is_complete(workspace, text):
    (workspace / "source.py").write_text(text, encoding="utf-8")
    result = invoke(tools.readFile, tools.EvalContext(), path="source.py")
    assert result["excerpt"] == text
    assert result["complete"] is True
    assert result["truncated"] is False
    assert result["next_line"] is None
    assert result["source"] == str(workspace / "source.py")


@pytest.mark.parametrize("tool,arguments", [
    (tools.readFile, {}),
    (tools.readFileRange, {"start_line": 1, "end_line": 2}),
])
def test_read_outputs_include_common_fields_and_preserve_original_text(workspace, tool, arguments):
    text = "  第一行 \t\n\t第二行  \n"
    path = workspace / "带 空格.txt"
    path.write_text(text, encoding="utf-8")
    result = invoke(tool, tools.EvalContext(), path=path.name, **arguments)
    assert set(ToolOutputBase.model_fields) <= result.keys()
    assert result["success"] is True
    assert result["message"] == ""
    assert result["error_code"] is None
    assert result["retryable"] is False
    assert result["source"] == str(path)
    assert result["excerpt"] == text
    assert FileReadOutput.model_validate(result).model_dump() == result


@pytest.mark.parametrize("tool,arguments,error_code", [
    (tools.readFile, {"path": "missing.txt"}, "NOT_FOUND"),
    (tools.readFileRange, {"path": "missing.txt", "start_line": 1, "end_line": 1}, "NOT_FOUND"),
    (tools.readFileRange, {"path": "lines.txt", "start_line": 0, "end_line": 1}, "INVALID_ARGUMENT"),
    (tools.readFileRange, {"path": "lines.txt", "start_line": 1, "end_line": 1, "start_column": 100}, "INVALID_ARGUMENT"),
    (tools.readFile, {}, "INVALID_ARGUMENT"),
    (tools.readFileRange, {"path": "lines.txt"}, "INVALID_ARGUMENT"),
])
def test_read_errors_use_common_model_including_sdk_parameter_errors(workspace, tool, arguments, error_code):
    (workspace / "lines.txt").write_text("short\n", encoding="utf-8")
    result = invoke(tool, tools.EvalContext(), **arguments)
    assert set(result) == set(ToolOutputBase.model_fields)
    assert result["success"] is False
    assert result["error_code"] == error_code
    assert result["message"].strip()
    assert result["retryable"] is True
    assert result["source"] is None
    assert ToolOutputBase.model_validate(result).model_dump() == result


@pytest.mark.parametrize("text", [
    "长行中有中文和引号\"\\\t" * 600,
    "ordinary line\n" * 400,
    "\x01\t\"\\" * 1000 + "\nlast line",
    "windows line\r\n" * 300,
    " \n\t缩进和尾空格  \n" * 250,
], ids=["long_unicode_line", "many_lines", "escaped_characters", "crlf", "whitespace_lines"])
def test_preview_and_range_recover_every_character(workspace, text):
    """覆盖行内截断、JSON 转义和换行归一化，续读不得跳过或重复字符。"""
    (workspace / "large.txt").write_text(text, encoding="utf-8")
    expected = text.replace("\r\n", "\n")
    state = tools.EvalContext()
    result = invoke(tools.readFile, state, path="large.txt")
    assert result["truncated"] is True
    pieces = [result["excerpt"]]
    positions = set()
    end_line = None
    while True:
        if result["truncated"]:
            position = (result["next_line"], result["next_column"])
        else:
            # 行段完整不代表文件结束；继续下一段，直到读到空内容。
            if not result["excerpt"]:
                break
            position = (end_line + 1, 1)
        assert position not in positions
        positions.add(position)
        end_line = position[0] + 199
        result = invoke(
            tools.readFileRange, state, path="large.txt",
            start_line=position[0], end_line=end_line,
            start_column=position[1],
        )
        pieces.append(result["excerpt"])
    assert "".join(pieces) == expected


def test_common_metadata_is_included_before_limiting_and_updating_cursors(workspace, monkeypatch):
    """收紧预算，覆盖公共字段、JSON 转义和行内续读的共同影响。"""
    budget = 550
    monkeypatch.setattr(tools, "_MAX_TOOL_RESULT_CHARS", budget)
    text = '  中文\t"\\\x01  \n' * 120 + "尾部无换行  "
    (workspace / "budget.txt").write_text(text, encoding="utf-8")
    state = tools.EvalContext()
    result = invoke(tools.readFile, state, path="budget.txt")
    assert result["truncated"] is True
    pieces = []
    positions = set()
    while True:
        assert len(tools._serialize_result(result)) <= budget
        assert set(ToolOutputBase.model_fields) <= result.keys()
        FileReadOutput.model_validate(result)
        pieces.append(result["excerpt"])
        if not result["truncated"]:
            assert result["complete"] is True
            assert result["next_line"] is None
            assert result["next_column"] is None
            break
        position = (result["next_line"], result["next_column"])
        assert position not in positions
        positions.add(position)
        result = invoke(
            tools.readFileRange, state, path="budget.txt", start_line=position[0],
            end_line=position[0] + 199, start_column=position[1],
        )
    assert "".join(pieces) == text


def test_range_start_column_preserves_spaces_and_counts_unicode_characters(workspace):
    (workspace / "columns.txt").write_text("忽略首行\n甲乙  \t正文  \n后续行\n", encoding="utf-8")
    result = invoke(
        tools.readFileRange, tools.EvalContext(), path="columns.txt",
        start_line=2, end_line=2, start_column=3,
    )
    assert result["excerpt"] == "  \t正文  \n"
    assert result["start_line"] == 2 and result["start_column"] == 3
    # 完整返回第 2 行，不代表已经读取仍然存在的第 3 行。
    assert result["complete"] is True and result["truncated"] is False
    assert result["next_line"] is None and result["next_column"] is None


def test_line_range_is_inclusive_and_can_be_empty(workspace):
    (workspace / "lines.txt").write_text("first\nsecond\nthird\nlast", encoding="utf-8")
    state = tools.EvalContext()
    result = invoke(tools.readFileRange, state, path="lines.txt", start_line=2, end_line=3)
    assert result["excerpt"] == "second\nthird\n"
    assert result["complete"] is True
    result = invoke(tools.readFileRange, state, path="lines.txt", start_line=10, end_line=11)
    assert result["excerpt"] == ""
    assert result["complete"] is True


def test_metadata_is_counted_near_budget_boundary(workspace):
    """短正文恰好接近预算时，next_line=null 等元数据也必须计入长度。"""
    path = workspace / "boundary.txt"
    for size in range(1650, 2001):
        text = "x" * size
        path.write_text(text, encoding="utf-8")
        encoded = tools._read_text_range(path)
        assert len(encoded) <= 2000
        result = json.loads(encoded)
        if not result["truncated"]:
            assert result["excerpt"] == text


@pytest.mark.parametrize("arguments", [
    {"start_line": 0, "end_line": 1},
    {"start_line": 2, "end_line": 1},
    {"start_line": 1, "end_line": 201},
    {"start_line": 1, "end_line": 1, "start_column": 100},
])
def test_invalid_range_retries_and_success_clears_counter(workspace, arguments):
    (workspace / "lines.txt").write_text("short\n", encoding="utf-8")
    state = tools.EvalContext()
    result = invoke(tools.readFileRange, state, path="lines.txt", **arguments)
    assert result["error_code"] == "INVALID_ARGUMENT"
    assert result["retryable"] is True
    invoke(tools.readFileRange, state, path="lines.txt", start_line=1, end_line=1)
    assert state.failure_counts == {}


def test_search_is_literal_case_sensitive_and_counts_lines(workspace):
    (workspace / "search.txt").write_text("a.b a.b\naXb\nA.B\na.b\n", encoding="utf-8")
    result = invoke(tools.searchFile, tools.EvalContext(), path="search.txt", query="a.b")
    assert result["total_matches"] == 2
    assert [match["line"] for match in result["matches"]] == [1, 4]
    assert result["has_more"] is False


def test_search_budget_pagination_preserves_all_matches(workspace):
    lines = [f"target_{number}: " + '\\"\t' * 100 + "\n" for number in range(80)]
    (workspace / "search.txt").write_text("".join(lines), encoding="utf-8")
    state = tools.EvalContext()
    offset = 0
    matched_lines = []
    while True:
        result = invoke(tools.searchFile, state, path="search.txt", query="target_", offset=offset, max_matches=100)
        assert result["total_matches"] == 80
        matched_lines.extend(match["line"] for match in result["matches"])
        if not result["has_more"]:
            break
        assert result["next_offset"] > offset
        offset = result["next_offset"]
    assert matched_lines == list(range(1, 81))


def test_search_long_line_preview_has_real_column(workspace):
    line = "x" * 5000 + "target_keyword" + "y" * 5000
    (workspace / "long.txt").write_text(line, encoding="utf-8")
    result = invoke(tools.searchFile, tools.EvalContext(), path="long.txt", query="target_keyword")
    match = result["matches"][0]
    assert match["line"] == 1
    assert match["truncated"] is True
    begin = match["start_column"] - 1
    assert match["excerpt"] == line[begin:begin + len(match["excerpt"])]
    assert "target_keyword" in match["excerpt"]


def test_directory_budget_pagination_is_stable(workspace):
    expected = []
    for number in range(32):
        path = workspace / (f"{number:03d}_" + "中文文件名" * 8 + ".txt")
        path.touch()
        expected.append(str(path))
    offset = 0
    collected = []
    state = tools.EvalContext()
    while True:
        result = invoke(tools.listFiles, state, path=".", offset=offset, max_entries=100)
        assert result["total_count"] == 32
        collected.extend(result["files"])
        if not result["has_more"]:
            break
        assert result["next_offset"] > offset
        offset = result["next_offset"]
    assert collected == expected


@pytest.mark.parametrize("exit_code,status", [(0, "passed"), (1, "failed"), (5, "no_tests")])
def test_large_test_output_keeps_status_and_full_log(workspace, monkeypatch, exit_code, status):
    full_output = "开始测试\n" + "large output \"\\\t\n" * 500 + "最终统计\n"
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], exit_code, full_output, "stderr\n"))
    state = tools.EvalContext()
    result = invoke(tools.runSafeTest, state, project_path=".", test_path=".")
    assert result["status"] == status
    assert result["exit_code"] == exit_code
    assert result["success"] is (exit_code == 0)
    assert result["truncated"] is True
    assert "开始测试" in result["output"]
    assert "最终统计" in result["output"]
    source = tools.resolve_path(state, result["source"])
    assert source.read_text(encoding="utf-8") == full_output + "stderr\n"
    result = invoke(tools.searchFile, state, path=str(source), query="最终统计")
    assert result["total_matches"] == 1


def test_test_log_save_cannot_escape_workspace(workspace, monkeypatch):
    outside = workspace.parent / "outside"
    outside.mkdir()
    (workspace / ".tool_results").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "x" * 5000, ""))
    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        invoke(tools.runSafeTest, tools.EvalContext(), project_path=".", test_path=".")
    assert not list(outside.iterdir())


def test_patch_error_output_is_bounded_and_retryable(workspace, monkeypatch):
    (workspace / "sample.py").write_text("old\n", encoding="utf-8")
    patch = "diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n@@ -1 +1 @@\n-old\n+new\n"
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "\x01" * 5000))
    state = tools.EvalContext()
    result = invoke(tools.applyPatch, state, workspace=".", patch=patch)
    assert result["error_code"] == "PATCH_FAILED"
    assert result["retryable"] is True
    assert result["truncated"] is True
    with pytest.raises(tools.FatalToolError, match="PATCH_FAILED_LIMIT"):
        invoke(tools.applyPatch, state, workspace=".", patch=patch)


@pytest.mark.parametrize("tool,arguments", [
    (tools.searchFile, {"query": "text"}),
    (tools.readFileRange, {"start_line": 1, "end_line": 1}),
])
def test_new_file_tools_preserve_permission_and_decode_errors(workspace, tool, arguments):
    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        invoke(tool, tools.EvalContext(), path="../outside", **arguments)
    (workspace / "binary").write_bytes(b"\xff")
    with pytest.raises(tools.FatalToolError, match="ENVIRONMENT_ERROR"):
        invoke(tool, tools.EvalContext(), path="binary", **arguments)
