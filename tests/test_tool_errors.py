import asyncio
import json
import subprocess
import uuid
from pathlib import Path

import pytest
from agents import set_tracing_disabled
from agents.tool_context import ToolContext
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

import tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    # SDK 在处理可重试错误后会记录 tracing；离线测试不创建远程 exporter。
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    set_tracing_disabled(True)
    root = tmp_path / "authorized"
    root.mkdir()
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", root)
    return root


def invoke(tool, state, **arguments):
    return invoke_raw(tool, state, json.dumps(arguments))


def invoke_raw(tool, state, payload):
    """通过 SDK 入口调用，覆盖进入工具函数之前的参数校验。"""
    context = ToolContext(
        context=state, tool_name=tool.name,
        tool_call_id=uuid.uuid4().hex, tool_arguments=payload,
    )
    result = asyncio.run(tool.on_invoke_tool(context, payload))
    return json.loads(result) if isinstance(result, str) and result.startswith("{") else result


def patch_for(name="sample.py"):
    return (
        f"diff --git a/{name} b/{name}\n"
        f"--- a/{name}\n+++ b/{name}\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )


@pytest.mark.parametrize("tool,arguments", [
    (tools.listFiles, {"path": "missing"}),
    (tools.readFile, {"path": "missing"}),
    (tools.searchFile, {"path": "missing", "query": "text"}),
    (tools.readFileRange, {"path": "missing", "start_line": 1, "end_line": 2}),
    (tools.writeFile, {"path": "missing/file", "content": "text"}),
    (tools.copyFile, {"source": "missing", "target": "copy"}),
    (tools.applyPatch, {"workspace": "missing", "patch": "invalid"}),
    (tools.runSafeTest, {"project_path": "missing", "test_path": "."}),
])
def test_missing_path_retries_then_raises_and_isolates_tasks(workspace, tool, arguments):
    state = tools.EvalContext()
    first = invoke(tool, state, **arguments)
    assert first["success"] is False
    assert first["retryable"] is True
    assert first["error_code"] == "NOT_FOUND"
    with pytest.raises(tools.FatalToolError, match="NOT_FOUND_LIMIT"):
        invoke(tool, state, **arguments)
    assert invoke(tool, tools.EvalContext(), **arguments)["retryable"] is True


def test_success_clears_missing_file_counter(workspace):
    state = tools.EvalContext()
    invoke(tools.readFile, state, path="sample.py")
    (workspace / "sample.py").write_text("text")
    assert invoke(tools.readFile, state, path="sample.py")["excerpt"] == "text"
    assert state.failure_counts == {}
    assert invoke(tools.readFile, state, path="other.py")["retryable"] is True


def test_file_tools_keep_success_outputs(workspace):
    state = tools.EvalContext()
    assert invoke(tools.writeFile, state, path="sample.py", content="text") is None
    assert invoke(tools.copyFile, state, source="sample.py", target="copy.py") is True
    assert invoke(tools.readFile, state, path=str(workspace / "copy.py"))["excerpt"] == "text"
    files = invoke(tools.listFiles, state, path=".")["files"]
    assert set(files) == {str(workspace / "sample.py"), str(workspace / "copy.py")}


def test_paths_reject_parent_sibling_and_symlink_escape(workspace):
    sibling = workspace.parent / "sibling"
    sibling.mkdir()
    (workspace / "escape").symlink_to(sibling, target_is_directory=True)
    state = tools.EvalContext()
    for path in ("../sibling", str(sibling), "escape"):
        with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
            invoke(tools.listFiles, state, path=path)
    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        tools.resolve_path(tools.EvalContext(workspace=workspace.parent), ".")


def test_context_can_narrow_authorization(workspace):
    child = workspace / "child"
    child.mkdir()
    state = tools.EvalContext(workspace=child)
    assert tools.resolve_path(state, ".") == child
    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        tools.resolve_path(state, "..")


def test_missing_context_and_workspace_are_fatal(workspace):
    with pytest.raises(tools.FatalToolError, match="EvalContext"):
        invoke(tools.readFile, None, path="sample.py")
    with pytest.raises(tools.FatalToolError, match="ENVIRONMENT_ERROR"):
        tools.resolve_path(tools.EvalContext(workspace=workspace / "missing"), ".")


def test_read_permission_and_decode_errors_are_fatal(workspace, monkeypatch):
    def deny(*args, **kwargs):
        raise PermissionError("denied")

    with monkeypatch.context() as patch:
        patch.setattr(tools, "open", deny, raising=False)
        with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
            invoke(tools.readFile, tools.EvalContext(), path="sample.py")
    (workspace / "binary").write_bytes(b"\xff")
    with pytest.raises(tools.FatalToolError, match="ENVIRONMENT_ERROR"):
        invoke(tools.readFile, tools.EvalContext(), path="binary")


def test_invalid_patch_retries_then_raises(workspace):
    state = tools.EvalContext()
    result = invoke(tools.applyPatch, state, workspace=".", patch="invalid")
    assert result["error_code"] == "INVALID_PATCH"
    with pytest.raises(tools.FatalToolError, match="INVALID_PATCH_LIMIT"):
        invoke(tools.applyPatch, state, workspace=".", patch="invalid")


def test_patch_tool_exposes_format_and_path_contract_to_model():
    """检查 SDK 提取后的工具描述，避免文档只对读源码的人可见。"""
    description = tools.applyPatch.description
    for marker in ("diff --git a/", "--- a/", "+++ b/", "@@ -"):
        assert marker in description
    assert "变更前后各 3 行原文作为上下文" in description
    assert "区块末尾也应保留后续上下文" in description
    properties = tools.applyPatch.params_json_schema["properties"]
    assert "workspace" in properties["patch"]["description"]
    assert properties["workspace"]["description"]


@pytest.mark.parametrize("patch,expected_message", [
    (patch_for().split("\n", 1)[1], "缺少 diff --git 文件头"),
    (patch_for().replace("--- a/", "--- "), "第 2 行原文件头格式错误；应为 --- a/"),
    (patch_for().replace("+++ b/", "+++ "), "第 3 行新文件头格式错误；应为 +++ b/"),
    (patch_for().replace("diff --git a/sample.py b/sample.py", "diff --git sample.py sample.py"),
     "第 1 行 Git 文件头格式错误"),
    (patch_for().replace("b/sample.py", "b/other.py", 1), "新旧文件路径不同"),
    (patch_for().replace("--- a/sample.py", "--- a/other.py"), "文件头路径不一致"),
])
def test_invalid_patch_feedback_identifies_header_problem(workspace, patch, expected_message):
    """结构错误应给出修正方向，且不能改写目标文件。"""
    source = workspace / "sample.py"
    source.write_text("old\n", encoding="utf-8")
    result = invoke(tools.applyPatch, tools.EvalContext(), workspace=".", patch=patch)
    assert result["error_code"] == "INVALID_PATCH"
    assert result["retryable"] is True
    assert expected_message in result["message"]
    assert source.read_text(encoding="utf-8") == "old\n"
    assert len(json.dumps(result, ensure_ascii=False)) <= tools._MAX_TOOL_RESULT_CHARS


def test_wrong_hunk_count_reports_git_error_and_corrected_patch_can_apply(workspace):
    """真实 git 检查拒绝错误行数，修正后仍可重试成功。"""
    source = workspace / "sample.py"
    source.write_text("old\n", encoding="utf-8")
    state = tools.EvalContext()
    wrong_count = patch_for().replace("@@ -1 +1 @@", "@@ -1,2 +1 @@")
    result = invoke(tools.applyPatch, state, workspace=".", patch=wrong_count)
    assert result["error_code"] == "PATCH_FAILED"
    assert result["retryable"] is True
    assert "区块行数" in result["message"]
    assert "Git:" in result["message"]
    assert "corrupt patch" in result["message"]
    assert source.read_text(encoding="utf-8") == "old\n"
    assert invoke(tools.applyPatch, state, workspace=".", patch=patch_for())["success"] is True
    assert source.read_text(encoding="utf-8") == "new\n"
    assert state.failure_counts == {}


def test_patch_context_mismatch_suggests_rereading_current_file(workspace):
    source = workspace / "sample.py"
    source.write_text("changed\n", encoding="utf-8")
    result = invoke(tools.applyPatch, tools.EvalContext(), workspace=".", patch=patch_for())
    assert result["error_code"] == "PATCH_FAILED"
    assert "上下文不匹配" in result["message"]
    assert "重新读取目标文件" in result["message"]
    assert "Git:" in result["message"]
    assert source.read_text(encoding="utf-8") == "changed\n"


@pytest.mark.parametrize("hunk", [
    "@@ -4 +4 @@\n-old\n+new\n",  # 完全没有上下文。
    "@@ -1,4 +1,4 @@\n before1\n before2\n before3\n-old\n+new\n",  # 缺少后续上下文。
])
def test_middle_file_patch_requires_context_and_can_be_corrected(workspace, hunk):
    """使用真实 Git 复现中间区块缺上下文，修正后通过同一个 SDK 工具应用。"""
    source = workspace / "sample.py"
    original = "before1\nbefore2\nbefore3\nold\nafter1\nafter2\nafter3\n"
    source.write_text(original, encoding="utf-8")
    header = "diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n"
    state = tools.EvalContext()

    result = invoke(tools.applyPatch, state, workspace=".", patch=header + hunk)
    assert result["error_code"] == "PATCH_FAILED"
    assert result["retryable"] is True
    assert "缺少上下文行" in result["message"]
    assert "各保留 3 行原文" in result["message"]
    assert "Git:" in result["message"]
    assert len(tools._serialize_result(result)) <= tools._MAX_TOOL_RESULT_CHARS
    assert source.read_text(encoding="utf-8") == original

    # 变更前后各保留三行，不更改 Git 应用选项，也不自动改写模型生成的补丁。
    corrected = header + (
        "@@ -1,7 +1,7 @@\n before1\n before2\n before3\n-old\n+new\n"
        " after1\n after2\n after3\n"
    )
    assert invoke(tools.applyPatch, state, workspace=".", patch=corrected)["success"] is True
    assert source.read_text(encoding="utf-8") == original.replace("old\n", "new\n")
    assert state.failure_counts == {}


@pytest.mark.parametrize("patch", [
    "new file mode 100644\n",
    patch_for("../outside.py"),
    patch_for("linked.py"),
])
def test_patch_disallowed_operations_are_fatal(workspace, patch):
    (workspace / "sample.py").write_text("old\n")
    (workspace / "linked.py").symlink_to(workspace / "sample.py")
    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        invoke(tools.applyPatch, tools.EvalContext(), workspace=".", patch=patch)


def test_real_git_patch_clears_counters(workspace):
    state = tools.EvalContext()
    (workspace / "sample.py").write_text("mismatch\n")
    assert invoke(tools.applyPatch, state, workspace=".", patch=patch_for())["error_code"] == "PATCH_FAILED"
    assert invoke(tools.applyPatch, state, workspace=".", patch="invalid")["error_code"] == "INVALID_PATCH"
    assert invoke(tools.applyPatch, state, workspace=".", patch=patch_for("missing.py"))["error_code"] == "NOT_FOUND"
    (workspace / "sample.py").write_text("old\n")
    assert invoke(tools.applyPatch, state, workspace=".", patch=patch_for())["success"] is True
    assert (workspace / "sample.py").read_text() == "new\n"
    assert state.failure_counts == {}


@pytest.mark.parametrize("error,code", [
    (FileNotFoundError("git missing"), "ENVIRONMENT_ERROR"),
    (PermissionError("git denied"), "ENVIRONMENT_ERROR"),
    (subprocess.TimeoutExpired("git", 15), "TIMEOUT"),
])
def test_patch_process_errors_are_fatal(workspace, monkeypatch, error, code):
    (workspace / "sample.py").write_text("old\n")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tools.subprocess, "run", fail)
    with pytest.raises(tools.FatalToolError, match=code):
        invoke(tools.applyPatch, tools.EvalContext(), workspace=".", patch=patch_for())


def test_patch_execution_failure_after_check_is_fatal(workspace, monkeypatch):
    (workspace / "sample.py").write_text("old\n")
    results = iter([
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 1, "", "apply failed"),
    ])
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: next(results))
    with pytest.raises(tools.FatalToolError, match="PATCH_FAILED: apply failed"):
        invoke(tools.applyPatch, tools.EvalContext(), workspace=".", patch=patch_for())


@pytest.mark.parametrize("exit_code,status", [
    (0, "passed"), (1, "failed"), (2, "interrupted"),
    (3, "internal_error"), (4, "usage_error"), (5, "no_tests"),
])
def test_pytest_results_remain_recoverable(workspace, monkeypatch, capsys, exit_code, status):
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], exit_code, "output", ""))
    result = invoke(tools.runSafeTest, tools.EvalContext(), project_path=".", test_path=".")
    assert result["status"] == status
    assert result["success"] is (exit_code == 0)
    assert "测试通过" not in capsys.readouterr().out


@pytest.mark.parametrize("exit_code", [125, 126, 127, 137])
def test_docker_execution_errors_are_fatal(workspace, monkeypatch, exit_code):
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], exit_code, "", "docker error"))
    with pytest.raises(tools.FatalToolError, match="EXECUTION_ERROR"):
        invoke(tools.runSafeTest, tools.EvalContext(), project_path=".", test_path=".")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_test_timeout_cleans_container_and_raises(workspace, monkeypatch, cleanup_fails):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 60)
        if cleanup_fails:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.FatalToolError, match="TIMEOUT") as caught:
        invoke(tools.runSafeTest, tools.EvalContext(), project_path=".", test_path=".")
    assert calls[1][0] == ["docker", "rm", "-f", calls[0][0][4]]
    assert calls[1][1]["check"] is True
    assert ("清理失败" in str(caught.value)) is cleanup_fails


def test_docker_missing_is_fatal(workspace, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("docker missing")

    monkeypatch.setattr(tools.subprocess, "run", missing)
    with pytest.raises(tools.FatalToolError, match="ENVIRONMENT_ERROR"):
        invoke(tools.runSafeTest, tools.EvalContext(), project_path=".", test_path=".")


@pytest.mark.parametrize("tool,expected", [
    (tools.listFiles, {"path", "offset", "max_entries"}), (tools.readFile, {"path"}),
    (tools.searchFile, {"path", "query", "offset", "max_matches"}),
    (tools.readFileRange, {"path", "start_line", "end_line", "start_column"}),
    (tools.writeFile, {"path", "content"}), (tools.copyFile, {"source", "target"}),
    (tools.applyPatch, {"patch", "workspace"}),
    (tools.runSafeTest, {"project_path", "test_path"}),
])
def test_context_is_not_a_model_argument(tool, expected):
    assert set(tool.params_json_schema["properties"]) == expected


@pytest.mark.parametrize("tool,field", [
    (tools.listFiles, "path"), (tools.readFile, "path"),
    (tools.readFileRange, "start_line"), (tools.searchFile, "query"),
    (tools.writeFile, "content"), (tools.copyFile, "source"),
    (tools.applyPatch, "patch"), (tools.runSafeTest, "project_path"),
])
def test_missing_arguments_retry_then_stop_and_isolate_tasks(workspace, tool, field):
    state = tools.EvalContext()
    first = invoke(tool, state)
    assert first["error_code"] == "INVALID_ARGUMENT"
    assert first["retryable"] is True
    assert field in first["message"]
    with pytest.raises(tools.FatalToolError, match="INVALID_ARGUMENT_LIMIT"):
        invoke(tool, state)
    assert state.failure_counts == {f"{tool.name}_INVALID_ARGUMENT": 2}
    assert invoke(tool, tools.EvalContext())["retryable"] is True


@pytest.mark.parametrize("payload,expected", [
    ('{"path":', "合法 JSON"),
    ('["sample.txt"]', "JSON 对象"),
    ('null', "JSON 对象"),
    ('{"path":"sample.txt","offset":0,"max_matches":2}', "query"),
    ('{"path":"sample.txt","query":[],"offset":0,"max_matches":2}', "query:string"),
    ('{"path":"sample.txt","query":"x","offset":"bad","max_matches":2}', "offset:integer"),
])
def test_sdk_input_errors_provide_actionable_feedback(workspace, payload, expected):
    result = invoke_raw(tools.searchFile, tools.EvalContext(), payload)
    assert result["success"] is False
    assert result["error_code"] == "INVALID_ARGUMENT"
    assert result["retryable"] is True
    assert expected in result["message"]
    assert len(tools._serialize_result(result)) <= tools._MAX_TOOL_RESULT_CHARS


def test_schema_and_function_argument_errors_share_counter(workspace):
    (workspace / "sample.txt").write_text("text", encoding="utf-8")
    state = tools.EvalContext()
    invoke(tools.searchFile, state, path="sample.txt")  # SDK 检出缺少 query。
    with pytest.raises(tools.FatalToolError, match="INVALID_ARGUMENT_LIMIT"):
        invoke(tools.searchFile, state, path="sample.txt", query="")  # 函数体检出空 query。


@pytest.mark.parametrize("tool,arguments", [
    (tools.listFiles, {"path": "."}),
    (tools.readFile, {"path": "sample.py"}),
    (tools.readFileRange, {"path": "sample.py", "start_line": 1, "end_line": 1}),
    (tools.searchFile, {"path": "sample.py", "query": "old"}),
    (tools.writeFile, {"path": "sample.py", "content": "new\n"}),
    (tools.copyFile, {"source": "sample.py", "target": "copy.py"}),
    (tools.applyPatch, {"workspace": ".", "patch": patch_for()}),
    (tools.runSafeTest, {"project_path": ".", "test_path": "."}),
])
def test_successful_correction_resets_input_error_counter(workspace, monkeypatch, tool, arguments):
    (workspace / "sample.py").write_text("old\n", encoding="utf-8")
    # applyPatch 与 runSafeTest 的进程只返回模拟成功结果，不运行 Docker。
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""))
    state = tools.EvalContext()
    assert invoke(tool, state)["retryable"] is True
    invoke(tool, state, **arguments)
    assert state.failure_counts == {}
    assert invoke(tool, state)["retryable"] is True


def test_sdk_error_feedback_does_not_echo_large_argument_values(workspace):
    sensitive_text = "do-not-echo-this-value" * 1000
    result = invoke(
        tools.searchFile, tools.EvalContext(), path="sample.txt",
        query=[sensitive_text], offset=0, max_matches=2,
    )
    serialized = tools._serialize_result(result)
    assert "do-not-echo-this-value" not in serialized
    assert len(serialized) <= tools._MAX_TOOL_RESULT_CHARS


@pytest.mark.parametrize("error", [
    RuntimeError("unexpected failure"),
    tools.ModelBehaviorError("invalid tool output"),
    tools.FatalToolError("PERMISSION_DENIED: outside workspace"),
])
def test_input_error_handler_preserves_other_exceptions(workspace, monkeypatch, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(tools, "resolve_path", fail)
    state = tools.EvalContext()
    with pytest.raises(type(error)) as caught:
        invoke(tools.readFile, state, path="sample.txt")
    assert caught.value is error
    assert state.failure_counts == {}


def test_invalid_input_without_task_context_is_fatal(workspace):
    with pytest.raises(tools.FatalToolError, match="ENVIRONMENT_ERROR"):
        invoke(tools.searchFile, None, path="sample.txt")
