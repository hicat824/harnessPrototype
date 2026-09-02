from typing import Annotated, Any
from urllib.parse import urlencode

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from httpUtil import HttpRequestError, httpReqWithJsonResp
from resultCache import project_and_cache

__all__ = [
    "csArgoFindDomainsByTaskSeq",
    "csArgoFindDomainsByTaskId",
]

_CS_ARGO_URL = "http://cs-argo-dispatcher.gzproduction.com/cs-argo-dispatcher"


def _require_non_empty(name: str, value: str | None) -> str:
    if value is None:
        raise ToolError(f"{name} must not be None")
    text = str(value).strip()
    if not text:
        raise ToolError(f"{name} must not be empty")
    return text


def _parse_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{name} must be a numeric id, got {value!r}") from exc
    if parsed <= 0:
        raise ToolError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _extract_job_list(payload: Any, identifier: str) -> list:
    if isinstance(payload, list):
        return payload
    raise ToolError(
        f"findDomains returned invalid JobInfo list for {identifier}: {type(payload).__name__}"
    )


def _find_domains(path: str, query: dict, identifier: str) -> dict:
    url = f"{_CS_ARGO_URL}/job/{path}?{urlencode(query)}"
    try:
        payload = httpReqWithJsonResp(url)
    except HttpRequestError as exc:
        raise ToolError(f"failed to query job domains for {identifier}: {exc}") from exc
    jobs = _extract_job_list(payload, identifier)
    return project_and_cache("find_domains", jobs, identifier)


def csArgoFindDomainsByTaskSeq(
    task_seq: Annotated[str, Field(description="全字母的任务 id，例如 kk1imK")],
) -> dict:
    """
    通过字母 taskSeq 查询该任务各个环节的运行信息。

    对同一个任务，返回信息与 csArgoFindDomainsByTaskId 一致。
    taskId 是全数字任务 id，taskSeq 是全字母任务 id。

    返回 {items, cache}。engineCode 使用公司统一的 engineCode 编码体系。
    如果当前请求已经明确目标流程，优先选择 items 中 obj["engineCode"] 精确匹配的对象；
    列表中没有目标 engineCode 时说明未找到对应环节，不要按数组顺序猜测。

    items 中每个 Job 只含：
    - taskBusinessId：对应 Job 的 taskId
    - taskSeq：对应 Job 的 taskSeq
    - engineCode：对应环节的 engineCode
    - state：任务状态。"0" 排队，"1" 运行中，"4" 运行失败，"9" 运行成功，其他取值表示任务异常
    - tasks：精简后的 Task 列表，每项含 state、podName、workflowName、stateAt，以及 params_keys（不含值）

    完整 JobInfo（含 tasks[].params）按 {taskSeq}_{engineCode}.json 落盘，路径在 cache[].path。
    需要 params 或其他字段时调用 readCachedResult(path, json_path)，
    例如 tasks.0.params 或 tasks.0.params.lidarFiles。
    """
    parsed_seq = _require_non_empty("task_seq", task_seq)
    return _find_domains(
        "findDomainsByTaskSeq",
        {"taskSeq": parsed_seq},
        f"task_seq={parsed_seq}",
    )


def csArgoFindDomainsByTaskId(
    task_id: Annotated[str, Field(description="全数字的任务 id，例如 410460")],
) -> dict:
    """
    通过数字 taskId 查询该任务各个环节的运行信息。

    对同一个任务，返回信息与 csArgoFindDomainsByTaskSeq 一致。
    taskId 是全数字任务 id，taskSeq 是全字母任务 id。

    返回 {items, cache}。engineCode 使用公司统一的 engineCode 编码体系。
    如果当前请求已经明确目标流程，优先选择 items 中 obj["engineCode"] 精确匹配的对象；
    列表中没有目标 engineCode 时说明未找到对应环节，不要按数组顺序猜测。

    items 中每个 Job 只含：
    - taskBusinessId：对应 Job 的 taskId
    - taskSeq：对应 Job 的 taskSeq
    - engineCode：对应环节的 engineCode
    - state：任务状态。"0" 排队，"1" 运行中，"4" 运行失败，"9" 运行成功，其他取值表示任务异常
    - tasks：精简后的 Task 列表，每项含 state、podName、workflowName、stateAt，以及 params_keys（不含值）

    完整 JobInfo（含 tasks[].params）按 {taskSeq}_{engineCode}.json 落盘，路径在 cache[].path。
    需要 params 或其他字段时调用 readCachedResult(path, json_path)，
    例如 tasks.0.params 或 tasks.0.params.lidarFiles。
    """
    parsed_id = _parse_positive_int("task_id", task_id)
    return _find_domains(
        "findDomainsByTaskId",
        {"taskId": parsed_id},
        f"task_id={parsed_id}",
    )
