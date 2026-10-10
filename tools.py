import os
import shutil
import sys
import json
from pathlib import Path
import subprocess
import re
import uuid
import tempfile
from dataclasses import dataclass, field
from agents import RunContextWrapper
from agents.decorators import tool
from agents.exceptions import AgentsException

__all__ = [
    "readFile",
    "listFiles",
    "writeFile",
    "runSafeTest",
    "applyPatch",
    "FatalToolError",
    "EvalContext",
]

_SAFE_WORKSPACE = Path("/media/xumaozhou/data/AI/local_agent_test").resolve()
_MAX_FAILURE_COUNT = 2


# 1. 定义致命工具异常
class FatalToolError(AgentsException):
    """终止当前任务，继承 SDK 基类以避免 Runner 将其包装成 UserError。"""


# 2. 每次 Eval 创建独立的上下文
@dataclass
class EvalContext:
    workspace: Path = field(default_factory=lambda: _SAFE_WORKSPACE)
    failure_counts: dict = field(default_factory=dict)


# 3. 统一解析授权路径，任务上下文只能收窄原有授权范围。
def resolve_path(ctx: EvalContext, path: str) -> Path:
    if not isinstance(ctx, EvalContext):
        raise FatalToolError("ENVIRONMENT_ERROR: 工具调用需要 EvalContext")

    try:
        root = ctx.workspace.resolve()
        target = (root / Path(path)).resolve()
        if not root.is_relative_to(_SAFE_WORKSPACE):
            raise FatalToolError("PERMISSION_DENIED: 任务工作目录超出授权范围")
        if not root.is_dir():
            raise FatalToolError("ENVIRONMENT_ERROR: 授权工作目录不存在")
        if not target.is_relative_to(root):
            raise FatalToolError("PERMISSION_DENIED: 路径超出授权工作目录")
    except (OSError, RuntimeError, ValueError) as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: 无法解析路径：{exc}") from exc

    return target


def _retryable_error(state: EvalContext, tool_name: str, error_code: str,
                     message: str) -> dict:
    key = f"{tool_name}_{error_code}"
    count = state.failure_counts.get(key, 0) + 1
    state.failure_counts[key] = count
    if count >= _MAX_FAILURE_COUNT:
        raise FatalToolError(f"{error_code}_LIMIT: {message}")
    return {
        "success": False,
        "error_code": error_code,
        "retryable": True,
        "message": message,
    }


def _raise_walk_error(error: OSError):
    raise error


@tool(failure_error_function=None)
def listFiles(ctx: RunContextWrapper[EvalContext], path: str) -> str:
    """
    用于列举本地授权工作目录下的文件列表
    """
    state = ctx.context
    # 权限错误：立即抛出致命异常
    target = resolve_path(state, path)

    if not target.is_dir():
        # 第一次错误允许 Agent 修正路径
        return json.dumps(_retryable_error(
            state, "listFiles", "NOT_FOUND", f"找不到本地目录： {path}"
        ), ensure_ascii=False)

    try:
        res = []
        for root, dirs, files in os.walk(target, onerror=_raise_walk_error):
            for file in files:
                res.append(os.path.join(root, file))
    except PermissionError as e:
        raise FatalToolError(f"PERMISSION_DENIED: {e}") from e
    except OSError as e:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {e}") from e

    # 成功后清除对应错误计数
    state.failure_counts.pop(
        "listFiles_NOT_FOUND", None
    )

    print(f'在 {path} 下扫描发现 {len(res)} 个文件')
    return json.dumps(res, ensure_ascii=False)


@tool(failure_error_function=None)
def readFile(ctx: RunContextWrapper[EvalContext], path: str) -> str:
    """
    用于读取本地指定文件内容
    """
    state = ctx.context
    target = resolve_path(state, path)
    try:
        with open(target, "r") as fp:
            content = fp.read()
    except FileNotFoundError:
        return json.dumps(_retryable_error(
            state, "readFile", "NOT_FOUND", f"找不到本地文件： {path}"
        ), ensure_ascii=False)
    except PermissionError as e:
        raise FatalToolError(f"PERMISSION_DENIED: {e}") from e
    except (OSError, UnicodeError) as e:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {e}") from e
    state.failure_counts.pop("readFile_NOT_FOUND", None)
    print(f'成功读取本地文件 {path} 内容')

    return content


@tool(failure_error_function=None)
def writeFile(ctx: RunContextWrapper[EvalContext], path: str,
              content: str) -> dict | None:
    """
    用于输出指定内容到本地文件
    """
    state = ctx.context
    target = resolve_path(state, path)
    try:
        with open(target, "w", encoding='utf-8') as fp:
            fp.write(content)
    except FileNotFoundError:
        return _retryable_error(
            state, "writeFile", "NOT_FOUND", f"文件父目录不存在： {path}"
        )
    except PermissionError as e:
        raise FatalToolError(f"PERMISSION_DENIED: {e}") from e
    except (OSError, UnicodeError) as e:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {e}") from e
    state.failure_counts.pop("writeFile_NOT_FOUND", None)
    print(f'成功输出内容到 {path}')


@tool(failure_error_function=None)
def copyFile(ctx: RunContextWrapper[EvalContext], source: str,
             target: str) -> bool | dict:
    """
    用于复制文件到指定路径

    输入参数：
        source: 来源文件绝对路径，比如: /workspace/A/test.txt
        target: 目标文件绝对路径，比如: /workspace/B/test.txt
    """
    state = ctx.context
    source_path = resolve_path(state, source)
    target_path = resolve_path(state, target)
    try:
        shutil.copyfile(source_path, target_path)
    except FileNotFoundError as e:
        return _retryable_error(state, "copyFile", "NOT_FOUND", str(e))
    except PermissionError as e:
        raise FatalToolError(f"PERMISSION_DENIED: {e}") from e
    except OSError as e:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {e}") from e
    state.failure_counts.pop("copyFile_NOT_FOUND", None)
    print(f'成功复制文件到{target}')
    return True


@tool(failure_error_function=None)
def applyPatch(ctx: RunContextWrapper[EvalContext], patch: str,
               workspace: str) -> dict:
    """
    将 Unified Diff 补丁应用到指定工作目录。

    参数:
        patch: Agent 生成的补丁文本
        workspace: 允许修改的工作目录

    返回:
        success: 是否修改成功
        output: 执行结果或错误信息

    限制:
        仅支持修改已有的普通文件。
        不支持新增、删除、重命名文件及修改文件权限。
    """

    state = ctx.context
    try:
        # 1. 获取工作目录的绝对路径
        root = resolve_path(state, workspace)

        if not root.is_dir():
            return _retryable_error(
                state, "applyPatch", "NOT_FOUND", "workspace 不存在或不是目录"
            )

        # 限制补丁大小，避免意外传入超大内容
        if len(patch.encode("utf-8")) > 100_000:
            raise ValueError("补丁过大")

        # 2. 禁止新增、删除、重命名等操作
        forbidden = (
            "new file mode ",
            "deleted file mode ",
            "rename from ",
            "rename to ",
            "copy from ",
            "copy to ",
            "old mode ",
            "new mode ",
            "GIT binary patch",
            "Binary files ",
            "--- /dev/null",
            "+++ /dev/null",
        )

        diff_paths = []
        old_paths = []
        new_paths = []

        # 3. 提取补丁中涉及的文件路径
        for line in patch.splitlines():

            if line.startswith(forbidden):
                raise FatalToolError("PERMISSION_DENIED: 补丁包含不允许的文件操作")

            if line.startswith("diff --git "):
                match = re.fullmatch(
                    r"diff --git a/([^\s]+) b/([^\s]+)",
                    line,
                )

                if not match or match.group(1) != match.group(2):
                    raise ValueError("不支持的补丁文件路径")

                diff_paths.append(match.group(1))

            elif line.startswith("--- "):
                if not line.startswith("--- a/"):
                    raise ValueError("无效的原文件路径")
                old_paths.append(line[6:])

            elif line.startswith("+++ "):
                if not line.startswith("+++ b/"):
                    raise ValueError("无效的新文件路径")
                new_paths.append(line[6:])

        # 要求补丁中三种文件路径声明一致
        if (
            not diff_paths
            or sorted(diff_paths) != sorted(old_paths)
            or sorted(diff_paths) != sorted(new_paths)
        ):
            raise ValueError("补丁文件路径不一致")

        # 4. 检查文件是否属于允许修改的目录
        for name in diff_paths:

            # 只允许简单的相对路径
            if (
                "\\" in name
                or name.startswith("/")
                or any(p in ("", ".", "..") for p in name.split("/"))
            ):
                raise FatalToolError(f"PERMISSION_DENIED: 非法路径: {name}")

            relative = Path(name)
            target = root

            # 检查每一级路径，拒绝符号链接
            for part in relative.parts:
                target = target / part

                if target.is_symlink():
                    raise FatalToolError(f"PERMISSION_DENIED: 不允许通过符号链接修改: {name}")

            if not target.is_file():
                return _retryable_error(
                    state, "applyPatch", "NOT_FOUND", f"文件不存在或不是普通文件: {name}"
                )

        # 5. 检查补丁能否应用，不实际修改文件
        check = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=15,
        )

        if check.returncode != 0:
            error = _retryable_error(
                state, "applyPatch", "PATCH_FAILED", check.stderr or check.stdout
            )
            error["output"] = error["message"]
            return error

        # 6. 正式应用补丁
        result = subprocess.run(
            ["git", "apply", "-"],
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise FatalToolError(
                f"PATCH_FAILED: {result.stderr or result.stdout}"
            )
        state.failure_counts.pop("applyPatch_NOT_FOUND", None)
        state.failure_counts.pop("applyPatch_PATCH_FAILED", None)
        state.failure_counts.pop("applyPatch_INVALID_PATCH", None)
        print(f'成功应用补丁到 {patch}')

        return {
            "success": result.returncode == 0,
            "output": (
                result.stderr
                or result.stdout
                or "补丁应用成功"
            ),
        }
    except ValueError as exc:
        error = _retryable_error(state, "applyPatch", "INVALID_PATCH", str(exc))
        error["output"] = error["message"]
        return error
    except subprocess.TimeoutExpired as exc:
        raise FatalToolError("TIMEOUT: 补丁执行超时（15 秒）") from exc
    except OSError as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {exc}") from exc


@tool(failure_error_function=None)
def runSafeTest(ctx: RunContextWrapper[EvalContext], project_path: str,
                test_path: str) -> dict:
    """
    在临时 Docker 容器中执行 pytest 验证代码改动是否正确。
    每次调用创建一个新容器，执行完成后自动删除。

    使用场景：
    1. 修复 Bug 或修改代码后，调用本工具验证修改结果。
    2. 测试失败时，根据返回的错误信息分析问题。
    3. 测试通过后，才能报告相关测试已通过。
    4. 未发现测试用例时，不得认为测试通过。

    输入：
        project_path: 被测试源码目录
        test_path: pytest 测试目录

    返回：
        status: passed、failed 或 no_tests
        success: 是否测试通过
        exit_code: 返回码
        output: pytest 的测试输出及错误信息
    """
    state = ctx.context
    project = resolve_path(state, project_path)
    tests = resolve_path(state, test_path)

    if not project.is_dir() or not tests.is_dir():
        error = _retryable_error(
            state, "runSafeTest", "NOT_FOUND", "源码目录或测试目录不存在"
        )
        error.update(status="error", exit_code=None, output=error["message"])
        return error

    # 为每次测试生成唯一容器名称
    container_name = f"agent-test-{uuid.uuid4().hex[:12]}"

    command = [
        "docker", "run", "--rm",
        "--name", container_name,

        # 基础隔离
        "--network", "none",
        "--user", "1000:1000",
        "--cap-drop", "ALL", # 移除 Linux 特权能力
        "--security-opt", "no-new-privileges", # 禁止获取新的特权
        "--memory", "512m",
        "--cpus", "1",
        "--pids-limit", "128",
        "--read-only",

        # 为 pytest 提供临时写入空间
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",

        # 源码和固定测试均为只读
        "--mount", f"type=bind,src={project},dst=/workspace,readonly",
        "--mount", f"type=bind,src={tests},dst=/tests",

        # 工作目录及 Python 导入路径
        "-w", "/workspace",
        "-e", "PYTHONPATH=/workspace",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "HOME=/tmp",

        # Docker 镜像
        "agent-python:3.10",

        # 容器内执行的 pytest 命令
        "python", "-m", "pytest",
        "/tests",
        "-q",
        "-x",
        "--tb=short",
        "-p", "no:cacheprovider",
    ]
    print(f'执行测试, dokcer: {container_name}')

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        # 超时时强制删除容器，避免测试进程残留
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=15,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as cleanup_exc:
            raise FatalToolError(
                f"TIMEOUT: 测试执行超时（60 秒）；容器清理失败：{cleanup_exc}"
            ) from exc
        raise FatalToolError("TIMEOUT: 测试执行超时（60 秒）") from exc

    except OSError as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {exc}") from exc

    # 根据 pytest 退出码判断结果
    # Docker 自身启动失败也可能返回非零退出码
    status_map = {
        0: "passed",
        1: "failed",
        2: "interrupted",
        3: "internal_error",
        4: "usage_error",
        5: "no_tests",
    }

    if result.returncode not in status_map:
        raise FatalToolError(
            f"EXECUTION_ERROR: exit_code={result.returncode}, "
            f"{result.stdout}{result.stderr}"
        )

    status = status_map[result.returncode]
    state.failure_counts.pop("runSafeTest_NOT_FOUND", None)
    print(f'测试结束：{status}')

    return {
        "status": status,
        "success": status == "passed",
        "exit_code": result.returncode,
        "output": result.stdout + result.stderr,
    }
