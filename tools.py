import os
import shutil
import sys
import json
from pathlib import Path
import subprocess
import re
import uuid
import tempfile

from agents.decorators import tool

__all__ = [
    "readFile",
    "listFiles",
    "writeFile",
    "runSafeTest",
    "applyPatch",
]

_SAFE_WORKSPACE = Path("/media/xumaozhou/data/AI/local_agent_test").resolve()


def _check_path(path: str):
    local_path = Path(path).resolve()
    return local_path == _SAFE_WORKSPACE or local_path.is_relative_to(_SAFE_WORKSPACE)


@tool
def listFiles(path: str) -> str:
    """
    用于列举本地文件列表
    """
    if not _check_path(path):
        raise PermissionError(f"不支持当前目录 {path}")
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            res.append(os.path.join(root, file))

    print(f'在 {path} 下扫描发现 {len(res)} 个文件')
    return json.dumps(res, ensure_ascii=False)


@tool
def readFile(path: str) -> str:
    """
    用于读取本地指定文件内容
    """
    if not _check_path(path):
        raise PermissionError(f"禁止访问工作目录之外的路径: {path}")
    with open(path, "r") as fp:
        content = fp.read()
    print(f'成功读取本地文件 {path} 内容')

    return content


@tool
def writeFile(path: str, content: str) -> None:
    """
    用于输出指定内容到本地文件
    """
    if not _check_path(path):
        raise PermissionError(f"禁止访问工作目录之外的路径: {path}")
    with open(path, "w", encoding='utf-8') as fp:
        fp.write(content)
    print(f'成功输出内容到 {path}')


@tool
def copyFile(source: str, target: str) -> bool:
    """
    用于复制文件到指定路径

    输入参数：
        source: 来源文件绝对路径，比如: /workspace/A/test.txt
        target: 目标文件绝对路径，比如: /workspace/B/test.txt
    """
    try:
        shutil.copyfile(source, target)
        print(f'成功复制文件到{target}')
        return True
    except Exception as e:
        raise IOError(f"复制文件报错： {e}")


@tool
def applyPatch(patch: str, workspace: str) -> dict:
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

    if not _check_path(workspace):
        raise PermissionError(f"禁止访问工作目录之外的路径: {workspace}")

    # 1. 获取工作目录的绝对路径
    root = Path(workspace).resolve(strict=True)

    if not root.is_dir():
        raise ValueError("workspace 必须是目录")

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
            raise ValueError("补丁包含不允许的文件操作")

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
            raise ValueError(f"非法路径: {name}")

        relative = Path(name)
        target = root

        # 检查每一级路径，拒绝符号链接
        for part in relative.parts:
            target = target / part

            if target.is_symlink():
                raise ValueError(f"不允许通过符号链接修改: {name}")

        if not target.is_file():
            raise ValueError(f"文件不存在或不是普通文件: {name}")

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
        return {
            "success": False,
            "output": check.stderr or check.stdout,
        }

    # 6. 正式应用补丁
    result = subprocess.run(
        ["git", "apply", "-"],
        cwd=root,
        input=patch,
        text=True,
        capture_output=True,
        timeout=15,
    )
    print(f'成功应用补丁到 {patch}')

    return {
        "success": result.returncode == 0,
        "output": (
            result.stderr
            or result.stdout
            or "补丁应用成功"
        ),
    }


@tool
def runSafeTest(project_path: str, test_path: str) -> dict:
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
    if not _check_path(project_path):
        raise PermissionError(f"禁止访问工作目录之外的路径: {project_path}")

    project = Path(project_path).resolve()
    tests = Path(test_path).resolve()

    if not project.is_dir() or not tests.is_dir():
        return {
            "status": "error",
            "success": False,
            "exit_code": None,
            "output": "源码目录或测试目录不存在",
        }

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
        print('测试通过')

    except subprocess.TimeoutExpired:
        # 超时时强制删除容器，避免测试进程残留
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=15,
        )

        return {
            "status": "timeout",
            "success": False,
            "exit_code": None,
            "output": "测试执行超时（60 秒）",
        }

    except OSError as exc:
        return {
            "status": "error",
            "success": False,
            "exit_code": None,
            "output": str(exc),
        }

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

    if result.returncode in (125, 126, 127):
        status = "execution_error"
    else:
        status = status_map.get(
            result.returncode,
            "execution_error",
        )

    return {
        "status": status,
        "success": status == "passed",
        "exit_code": result.returncode,
        "output": result.stdout + result.stderr,
    }