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
from agents.exceptions import AgentsException, ModelBehaviorError

__all__ = [
    "readFile",
    "searchFile",
    "readFileRange",
    "listFiles",
    "writeFile",
    "runSafeTest",
    "applyPatch",
    "FatalToolError",
    "EvalContext",
]

_SAFE_WORKSPACE = Path("/media/xumaozhou/data/AI/local_agent_test").resolve()
_MAX_FAILURE_COUNT = 2
# 这是字符预算，不是模型 Token 上限；计入完整 JSON 响应。
_MAX_TOOL_RESULT_CHARS = 2000
_MAX_PAGE_ITEMS = 100
_MAX_RANGE_LINES = 200
_MAX_MATCH_EXCERPT_CHARS = 160
_MAX_ERROR_MESSAGE_CHARS = 120


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
    result = {
        "success": False,
        "error_code": error_code,
        "retryable": True,
        "message": message[:_MAX_ERROR_MESSAGE_CHARS],
    }
    if len(message) > _MAX_ERROR_MESSAGE_CHARS:
        result["truncated"] = True
    return result


def _serialize_result(result: dict) -> str:
    """统一返回 JSON 字符串，避免 SDK 再将字典转成 Python repr。"""
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _tool_input_error(ctx, error: Exception) -> str:
    """将 SDK 的入参校验错误反馈给模型；致命错误和其他异常继续抛出。"""
    prefix = f"Invalid JSON input for tool {ctx.tool_name}"
    if not isinstance(error, ModelBehaviorError) or not (
            str(error) == prefix or str(error).startswith(prefix + ":")):
        raise error
    if not isinstance(ctx.context, EvalContext):
        raise FatalToolError("ENVIRONMENT_ERROR: 工具调用需要 EvalContext") from error

    try:
        arguments = json.loads(ctx.tool_arguments)
    except json.JSONDecodeError:
        message = "参数不是合法 JSON，请按工具 schema 重新提供 JSON 对象"
    else:
        if not isinstance(arguments, dict):
            message = "参数必须是 JSON 对象，请按工具 schema 提供字段"
        else:
            # 本模块的工具名称与全局变量同名，直接复用 SDK 对外提供的 schema。
            schema = globals()[ctx.tool_name].params_json_schema
            missing = [name for name in schema.get("required", []) if name not in arguments]
            if missing:
                message = "缺少必填参数：" + ", ".join(missing) + "；请补齐后重试"
            else:
                fields = ", ".join(
                    f"{name}:{field.get('type', '见 schema')}"
                    for name, field in schema["properties"].items()
                )
                message = "参数类型或结构不符合工具 schema，请修正后重试：" + fields

    # 与函数体内的参数错误共用计数；同一任务、同一工具累计两次即终止。
    return _serialize_result(_retryable_error(
        ctx.context, ctx.tool_name, "INVALID_ARGUMENT", message,
    ))


def _limit_text_result(result: dict, text_key: str = "excerpt",
                       keep_tail: bool = False) -> dict:
    """缩短正文，保留状态和来源；用二分查找计入转义后的 JSON 长度。"""
    result = dict(result)
    if len(_serialize_result(result)) <= _MAX_TOOL_RESULT_CHARS:
        return result
    text = result[text_key]
    result["truncated"] = True
    if "complete" in result:
        result["complete"] = False
    result[text_key] = ""
    if len(_serialize_result(result)) > _MAX_TOOL_RESULT_CHARS:
        raise FatalToolError("RESULT_TOO_LARGE: 结果元数据超出预算，请缩短来源路径")

    low, high = 0, len(text)
    while low < high:
        size = (low + high + 1) // 2
        if keep_tail and size > 0:
            head_size = size // 3
            tail_size = size - head_size
            result[text_key] = text[:head_size] + "\n...[truncated]...\n" + text[-tail_size:]
        else:
            result[text_key] = text[:size]
        if len(_serialize_result(result)) <= _MAX_TOOL_RESULT_CHARS:
            low = size
        else:
            high = size - 1
    if keep_tail and low > 0:
        head_size = low // 3
        result[text_key] = text[:head_size] + "\n...[truncated]...\n" + text[-(low - head_size):]
    else:
        result[text_key] = text[:low]
    return result


def _limit_page_result(result: dict, entries_key: str, offset: int,
                       total: int) -> str:
    """整项裁剪列表并更新分页位置，不把文件路径或命中记录从中间切断。"""
    entries = result[entries_key]
    while True:
        has_more = offset + len(entries) < total
        result["has_more"] = has_more
        result["next_offset"] = offset + len(entries) if has_more else None
        result["truncated"] = has_more or any(
            isinstance(entry, dict) and entry.get("truncated", False)
            for entry in entries
        )
        encoded = _serialize_result(result)
        if len(encoded) <= _MAX_TOOL_RESULT_CHARS:
            # 有后续记录却一项也返回不了时，不能给出不前进的游标。
            if has_more and not entries:
                raise FatalToolError("RESULT_TOO_LARGE: 单个结果超出预算，请缩短来源路径")
            return encoded
        if not entries:
            raise FatalToolError("RESULT_TOO_LARGE: 结果元数据超出预算，请缩短来源路径")
        entries.pop()


def _read_text_range(target: Path, start_line: int = 1,
                     end_line: int | None = None, start_column: int = 1) -> str:
    """流式定位后读取有限字符；行、列从 1 开始，列按解码后的字符计数。"""
    with open(target, "r", encoding="utf-8") as fp:
        current_line = 1
        # 分块跳过前面的行，避免跳过极长单行时一次读入整个文件。
        while current_line < start_line:
            part = fp.readline(_MAX_TOOL_RESULT_CHARS + 1)
            if not part:
                break
            if part.endswith("\n"):
                current_line += 1

        if current_line == start_line:
            remaining = start_column - 1
            while remaining:
                part = fp.readline(min(remaining, _MAX_TOOL_RESULT_CHARS + 1))
                if not part or part.endswith("\n"):
                    raise ValueError("start_column 超出指定行的范围")
                remaining -= len(part)

        parts = []
        remaining = _MAX_TOOL_RESULT_CHARS + 1
        while current_line >= start_line and remaining:
            if end_line is not None and current_line > end_line:
                break
            part = fp.readline(remaining)
            if not part:
                break
            parts.append(part)
            remaining -= len(part)
            if part.endswith("\n"):
                current_line += 1

    text = "".join(parts)
    truncated = remaining == 0
    result = {
        "success": True,
        "source": str(target),
        "complete": not truncated,
        "truncated": truncated,
        "start_line": start_line,
        "start_column": start_column,
        "excerpt": text,
        # 先预留最大可能的位置数字宽度，正文限长后再计算真实续读位置。
        "next_line": start_line + text.count("\n"),
        "next_column": start_column + len(text),
    }
    # 续读位置更新后也检查预算，例如 JSON null 比单数字位置更长。
    while True:
        result = _limit_text_result(result)
        if result["truncated"]:
            excerpt = result["excerpt"]
            result["next_line"] = start_line + excerpt.count("\n")
            if "\n" in excerpt:
                result["next_column"] = len(excerpt.rsplit("\n", 1)[1]) + 1
            else:
                result["next_column"] = start_column + len(excerpt)
            if not excerpt:
                raise FatalToolError("RESULT_TOO_LARGE: 预算不足以返回文件正文，请缩短来源路径")
        else:
            result["next_line"] = None
            result["next_column"] = None
        encoded = _serialize_result(result)
        if len(encoded) <= _MAX_TOOL_RESULT_CHARS:
            return encoded


def _save_tool_output(state: EvalContext, output: str, prefix: str = "pytest-") -> str:
    """保存完整工具输出；路径仍经过授权校验，不覆盖已有日志。"""
    directory = resolve_path(state, ".tool_results")
    try:
        directory.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=prefix, suffix=".log", delete=False,
        ) as fp:
            fp.write(output)
            return fp.name
    except PermissionError as exc:
        raise FatalToolError(f"PERMISSION_DENIED: 无法保存工具输出：{exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: 无法保存工具输出：{exc}") from exc


def limit_expert_output(state: EvalContext, output: str, name: str) -> str:
    """专家也是 Supervisor 的工具，长结论同样限长并保存全文。"""
    text = output if isinstance(output, str) else str(output)
    result = {"source": name, "excerpt": text, "truncated": False}
    if len(_serialize_result(result)) > _MAX_TOOL_RESULT_CHARS:
        result["source"] = _save_tool_output(state, text, prefix="expert-")
    return _serialize_result(_limit_text_result(result))


def _raise_walk_error(error: OSError):
    raise error


@tool(failure_error_function=_tool_input_error)
def listFiles(ctx: RunContextWrapper[EvalContext], path: str,
              offset: int = 0, max_entries: int = 20) -> str:
    """
    递归列举文件，按 offset 分页；响应含 total_count、next_offset。
    max_entries 为 1 到 100，目录不变时分页顺序稳定。
    """
    state = ctx.context
    # 权限错误：立即抛出致命异常
    target = resolve_path(state, path)
    if offset < 0 or not 1 <= max_entries <= _MAX_PAGE_ITEMS:
        return _serialize_result(_retryable_error(
            state, "listFiles", "INVALID_ARGUMENT", "offset 必须非负，max_entries 必须为 1 到 100"
        ))

    if not target.is_dir():
        # 第一次错误允许 Agent 修正路径
        return _serialize_result(_retryable_error(
            state, "listFiles", "NOT_FOUND", f"找不到本地目录： {path}"
        ))

    try:
        res = []
        total = 0
        for root, dirs, files in os.walk(target, onerror=_raise_walk_error):
            dirs.sort()
            for file in sorted(files):
                if offset <= total < offset + max_entries:
                    res.append(os.path.join(root, file))
                total += 1
    except PermissionError as e:
        raise FatalToolError(f"PERMISSION_DENIED: {e}") from e
    except OSError as e:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {e}") from e

    # 成功后清除对应错误计数
    state.failure_counts.pop(
        "listFiles_NOT_FOUND", None
    )
    state.failure_counts.pop("listFiles_INVALID_ARGUMENT", None)

    print(f'在 {path} 下扫描发现 {total} 个文件')
    return _limit_page_result({
        "success": True, "source": str(target), "files": res,
        "total_count": total, "offset": offset,
    }, "files", offset, total)


@tool(failure_error_function=_tool_input_error)
def readFile(ctx: RunContextWrapper[EvalContext], path: str) -> str:
    """
    读取 UTF-8 文件；小文件返回全文，大文件返回有限头部预览。
    返回 JSON 的 excerpt 为原文，truncated 为 true 时按 next_line、
    next_column 调用 readFileRange 补读，或使用 searchFile 定位。
    """
    state = ctx.context
    target = resolve_path(state, path)
    try:
        content = _read_text_range(target)
    except FileNotFoundError:
        return _serialize_result(_retryable_error(
            state, "readFile", "NOT_FOUND", f"找不到本地文件： {path}"
        ))
    except PermissionError as e:
        raise FatalToolError(f"PERMISSION_DENIED: {e}") from e
    except (OSError, UnicodeError) as e:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {e}") from e
    state.failure_counts.pop("readFile_NOT_FOUND", None)
    state.failure_counts.pop("readFile_INVALID_ARGUMENT", None)
    print(f'成功读取本地文件 {path} 内容')

    return content


@tool(failure_error_function=_tool_input_error)
def readFileRange(ctx: RunContextWrapper[EvalContext], path: str,
                  start_line: int, end_line: int, start_column: int = 1) -> str:
    """读取指定行段，两端包含，行列从 1 开始；最多请求 200 行。

    start_column 仅作用于首行，按字符计数。结果可能停在行中间，
    此时使用返回的 next_line、next_column 继续读取，直到 complete=true。
    超出文件末尾的行段返回空内容，不表示文件不存在。
    """
    state = ctx.context
    target = resolve_path(state, path)
    try:
        if start_line < 1 or end_line < start_line or start_column < 1:
            raise ValueError("行列必须从 1 开始，end_line 不能小于 start_line")
        if end_line - start_line + 1 > _MAX_RANGE_LINES:
            raise ValueError("每次最多读取 200 行")
        content = _read_text_range(target, start_line, end_line, start_column)
    except FileNotFoundError:
        return _serialize_result(_retryable_error(
            state, "readFileRange", "NOT_FOUND", f"找不到本地文件： {path}"
        ))
    except UnicodeError as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {exc}") from exc
    except ValueError as exc:
        return _serialize_result(_retryable_error(
            state, "readFileRange", "INVALID_ARGUMENT", str(exc)
        ))
    except PermissionError as exc:
        raise FatalToolError(f"PERMISSION_DENIED: {exc}") from exc
    except OSError as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {exc}") from exc
    state.failure_counts.pop("readFileRange_NOT_FOUND", None)
    state.failure_counts.pop("readFileRange_INVALID_ARGUMENT", None)
    return content


@tool(failure_error_function=_tool_input_error)
def searchFile(ctx: RunContextWrapper[EvalContext], path: str, query: str,
               offset: int = 0, max_matches: int = 20) -> str:
    """区分大小写的字面量搜索，返回真实行号和命中附近原文，不使用正则。

    total_matches 是全文件命中行数，同一行多次出现只计一行。
    query 必须显式提供非空字符串，长度为 1 到 200，不能省略或用 null 代替。
    offset 从 0 开始按命中行分页；max_matches 为 1 到 100。
    has_more=true 时按 next_offset 继续；单行片段截断时可按行列补读。
    """
    state = ctx.context
    target = resolve_path(state, path)
    if not query or len(query) > 200 or offset < 0 or not 1 <= max_matches <= _MAX_PAGE_ITEMS:
        return _serialize_result(_retryable_error(
            state, "searchFile", "INVALID_ARGUMENT",
            "query 长度必须为 1 到 200，offset 必须非负，max_matches 必须为 1 到 100",
        ))
    total = 0
    matches = []
    try:
        with open(target, "r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, 1):
                position = line.find(query)
                if position < 0:
                    continue
                if offset <= total < offset + max_matches:
                    text = line.rstrip("\n")
                    begin = max(0, position - 40)
                    excerpt = text[begin:begin + _MAX_MATCH_EXCERPT_CHARS]
                    matches.append({
                        "line": line_number, "start_column": begin + 1,
                        "excerpt": excerpt,
                        "truncated": begin > 0 or begin + len(excerpt) < len(text),
                    })
                total += 1
    except FileNotFoundError:
        return _serialize_result(_retryable_error(
            state, "searchFile", "NOT_FOUND", f"找不到本地文件： {path}"
        ))
    except PermissionError as exc:
        raise FatalToolError(f"PERMISSION_DENIED: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {exc}") from exc
    state.failure_counts.pop("searchFile_NOT_FOUND", None)
    state.failure_counts.pop("searchFile_INVALID_ARGUMENT", None)
    return _limit_page_result({
        "success": True, "source": str(target), "query": query,
        "offset": offset, "total_matches": total, "matches": matches,
    }, "matches", offset, total)


@tool(failure_error_function=_tool_input_error)
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
    state.failure_counts.pop("writeFile_INVALID_ARGUMENT", None)
    print(f'成功输出内容到 {path}')


@tool(failure_error_function=_tool_input_error)
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
    state.failure_counts.pop("copyFile_INVALID_ARGUMENT", None)
    print(f'成功复制文件到{target}')
    return True


@tool(failure_error_function=_tool_input_error, docstring_style="google")
def applyPatch(ctx: RunContextWrapper[EvalContext], patch: str,
               workspace: str) -> str:
    """
    将 Git 格式的 Unified Diff 补丁应用到指定项目目录。

    每个文件必须包含以下三行文件头，其中相对路径保持一致，a/、b/ 是补丁标记：
        diff --git a/path/to/file b/path/to/file
        --- a/path/to/file
        +++ b/path/to/file
    后面跟 @@ -旧起始行,旧行数 +新起始行,新行数 @@ 和变更正文。
    旧行数 = 上下文行数 + 删除行数；新行数 = 上下文行数 + 新增行数。
    上下文行以空格开头，删除行以 - 开头，新增行以 + 开头。
    每个变更区块通常保留变更前后各 3 行原文作为上下文，文件边界除外。
    不要只提供删除行和新增行；区块末尾也应保留后续上下文，除非已到文件末尾。
    补丁末尾必须有换行符；按最新文件内容生成小范围补丁，避免算错行数。

    仅支持修改已有的普通文件，不支持新增、删除、重命名文件及修改文件权限。
    同一类错误累计达到工具重试上限时会抛出 FatalToolError。

    Args:
        patch: 按上述 Git 格式提供的补丁文本，文件路径相对于 workspace 指定目录，
            不要重复添加项目目录前缀。
        workspace: 项目目录，可用相对于授权根目录的路径；a/、b/ 不是真实目录。

    Returns:
        JSON 字符串，success 表示是否修改成功，output 是结果或带修正建议的错误信息。
        错误可重试时 retryable=true，应先根据 message 修正补丁。
    """

    state = ctx.context
    try:
        # 1. 获取工作目录的绝对路径
        root = resolve_path(state, workspace)

        if not root.is_dir():
            return _serialize_result(_retryable_error(
                state, "applyPatch", "NOT_FOUND", "workspace 不存在或不是目录"
            ))

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
        for line_number, line in enumerate(patch.splitlines(), 1):

            if line.startswith(forbidden):
                raise FatalToolError("PERMISSION_DENIED: 补丁包含不允许的文件操作")

            if line.startswith("diff --git "):
                match = re.fullmatch(
                    r"diff --git a/([^\s]+) b/([^\s]+)",
                    line,
                )

                if not match:
                    raise ValueError(
                        f"第 {line_number} 行 Git 文件头格式错误；应为 diff --git a/<相对路径> b/<相对路径>，路径不能含空白"
                    )
                if match.group(1) != match.group(2):
                    raise ValueError(
                        f"第 {line_number} 行新旧文件路径不同；仅支持修改同一个已有文件，不支持重命名"
                    )

                diff_paths.append(match.group(1))

            elif line.startswith("--- "):
                if not line.startswith("--- a/"):
                    raise ValueError(
                        f"第 {line_number} 行原文件头格式错误；应为 --- a/<相对路径>，路径相对于 workspace 指定目录"
                    )
                old_paths.append(line[6:])

            elif line.startswith("+++ "):
                if not line.startswith("+++ b/"):
                    raise ValueError(
                        f"第 {line_number} 行新文件头格式错误；应为 +++ b/<相对路径>，路径相对于 workspace 指定目录"
                    )
                new_paths.append(line[6:])

        # 要求补丁中三种文件路径声明一致
        if not diff_paths:
            raise ValueError(
                "缺少 diff --git 文件头；每个文件必须以 diff --git a/<相对路径> b/<相对路径> 开始，随后提供 --- a/ 和 +++ b/ 文件头"
            )
        if sorted(diff_paths) != sorted(old_paths) or sorted(diff_paths) != sorted(new_paths):
            raise ValueError("文件头路径不一致；diff --git、--- a/、+++ b/ 必须声明相同的文件集合和相对路径")

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
                return _serialize_result(_retryable_error(
                    state, "applyPatch", "NOT_FOUND", f"文件不存在或不是普通文件: {name}"
                ))

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
            git_message = check.stderr or check.stdout
            # 不自动重算行数或修改补丁，让 Agent 根据具体错误重新生成。
            if "corrupt patch" in git_message or "patch fragment without header" in git_message:
                message = (
                    "补丁语法错误：检查 @@ 区块行数（旧=上下文+删除，新=上下文+新增）及末尾换行。Git: "
                    + git_message
                )
            elif "patch does not apply" in git_message or "patch failed" in git_message:
                message = (
                    "补丁上下文不匹配或缺少上下文行：请重新读取目标文件，按当前内容生成补丁；"
                    "变更前后通常各保留 3 行原文（文件边界除外）。Git: " + git_message
                )
            else:
                message = git_message
            error = _retryable_error(
                state, "applyPatch", "PATCH_FAILED", message
            )
            error["output"] = error["message"]
            return _serialize_result(error)

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
        state.failure_counts.pop("applyPatch_INVALID_ARGUMENT", None)
        print(f'成功应用补丁到 {patch}')

        return _serialize_result(_limit_text_result({
            "success": result.returncode == 0,
            "output": (
                result.stderr
                or result.stdout
                or "补丁应用成功"
            ),
            "truncated": False,
        }, text_key="output"))
    except ValueError as exc:
        error = _retryable_error(state, "applyPatch", "INVALID_PATCH", str(exc))
        error["output"] = error["message"]
        return _serialize_result(error)
    except subprocess.TimeoutExpired as exc:
        raise FatalToolError("TIMEOUT: 补丁执行超时（15 秒）") from exc
    except OSError as exc:
        raise FatalToolError(f"ENVIRONMENT_ERROR: {exc}") from exc


@tool(failure_error_function=_tool_input_error)
def runSafeTest(ctx: RunContextWrapper[EvalContext], project_path: str,
                test_path: str) -> str:
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
        truncated: 是否限制了输出；为 true 时 source 指向完整日志文件
    """
    state = ctx.context
    project = resolve_path(state, project_path)
    tests = resolve_path(state, test_path)

    if not project.is_dir() or not tests.is_dir():
        error = _retryable_error(
            state, "runSafeTest", "NOT_FOUND", "源码目录或测试目录不存在"
        )
        error.update(status="error", exit_code=None, output=error["message"])
        return _serialize_result(error)

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
    state.failure_counts.pop("runSafeTest_INVALID_ARGUMENT", None)
    print(f'测试结束：{status}')

    response = {
        "status": status,
        "success": status == "passed",
        "exit_code": result.returncode,
        "output": result.stdout + result.stderr,
        "truncated": False,
    }
    if len(_serialize_result(response)) > _MAX_TOOL_RESULT_CHARS:
        response["source"] = _save_tool_output(state, response["output"])
    return _serialize_result(_limit_text_result(response, text_key="output", keep_tail=True))
