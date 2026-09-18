import os
import sys
import json
from pathlib import Path
import subprocess
import re

from agents.decorators import tool

__all__ = [
    "readFile",
    "listFiles",
    "writeFile",
    "runTest",
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
def runTest(path: str) -> dict:
    """
    执行当前 Python 项目的 pytest 测试，验证代码是否正确。
    默认 pytest 文件与待测试代码处于同一个目录下。

    使用场景：
    1. 修复 Bug 或修改代码后，调用本工具验证修改结果。
    2. 测试失败时，根据返回的错误信息分析问题。
    3. 测试通过后，才能报告相关测试已通过。
    4. 未发现测试用例时，不得认为测试通过。

    输入：
        path: 待测试代码所在文件夹绝对路径

    返回：
        status: passed、failed 或 no_tests
        success: 是否测试通过
        output: pytest 的测试输出及错误信息

    注意：本工具只负责执行测试，不负责修改代码。
    """
    if not _check_path(path):
        raise PermissionError(f"禁止访问工作目录之外的路径: {path}")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # 根据 pytest 返回码判断测试状态
    if result.returncode == 0:
        status = "passed"

    elif result.returncode == 5:
        status = "no_tests"

    else:
        status = "failed"

    test_result = {
        "status": status,
        "success": status == "passed",
        "output": result.stdout + result.stderr,
    }
    print(f'测试结果： {test_result}')

    return test_result