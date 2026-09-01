from typing import Annotated, Any
from urllib.parse import urlencode

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from httpUtil import HttpRequestError, httpReqWithJsonResp

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


def _find_domains(path: str, query: dict, identifier: str) -> list:
    url = f"{_CS_ARGO_URL}/job/{path}?{urlencode(query)}"
    try:
        payload = httpReqWithJsonResp(url)
    except HttpRequestError as exc:
        raise ToolError(f"failed to query job domains for {identifier}: {exc}") from exc
    return _extract_job_list(payload, identifier)


def csArgoFindDomainsByTaskSeq(
    task_seq: Annotated[str, Field(description="全字母的任务 id，例如 kk1imK")],
) -> list:
    """
    通过字母 taskSeq 查询该任务各个环节的运行信息。

    对同一个任务，返回信息与 csArgoFindDomainsByTaskId 一致。
    taskId 是全数字任务 id，taskSeq 是全字母任务 id。

    返回值为 JobInfo 列表。engineCode 使用公司统一的 engineCode 编码体系。
    如果当前请求已经明确目标流程，优先选择 obj["engineCode"] 精确匹配目标流程的对象；
    列表中没有目标 engineCode 时说明未找到对应环节，不要按数组顺序猜测。

    每个 JobInfo 需要关注的字段：
    - taskBusinessId：对应 Job 的 taskId
    - taskSeq：对应 Job 的 taskSeq
    - engineCode：对应环节的 engineCode
    - state：任务状态。"0" 表示排队，"1" 表示运行中，"4" 表示运行失败，"9" 表示运行成功，其他取值表示任务异常
    - tasks：该 Job 包含的 Task 列表，每个 taskInfo 常用字段：
        - state：任务状态，含义同上
        - podName：k8s 中的 pod name，可用于查询 pod 日志
        - workflowName：k8s 中的 workflowName，可用于查询日志
        - stateAt：任务开始时间，毫秒时间戳
        - params：任务参数，不同任务可能不一样，字段定义与 csKtsShowByTaskId 返回的 tags 一致
    """
    parsed_seq = _require_non_empty("task_seq", task_seq)
    return _find_domains(
        "findDomainsByTaskSeq",
        {"taskSeq": parsed_seq},
        f"task_seq={parsed_seq}",
    )


def csArgoFindDomainsByTaskId(
    task_id: Annotated[str, Field(description="全数字的任务 id，例如 410460")],
) -> list:
    """
    通过数字 taskId 查询该任务各个环节的运行信息。

    对同一个任务，返回信息与 csArgoFindDomainsByTaskSeq 一致。
    taskId 是全数字任务 id，taskSeq 是全字母任务 id。

    返回值为 JobInfo 列表。engineCode 使用公司统一的 engineCode 编码体系。
    如果当前请求已经明确目标流程，优先选择 obj["engineCode"] 精确匹配目标流程的对象；
    列表中没有目标 engineCode 时说明未找到对应环节，不要按数组顺序猜测。

    每个 JobInfo 需要关注的字段：
    - taskBusinessId：对应 Job 的 taskId
    - taskSeq：对应 Job 的 taskSeq
    - engineCode：对应环节的 engineCode
    - state：任务状态。"0" 表示排队，"1" 表示运行中，"4" 表示运行失败，"9" 表示运行成功，其他取值表示任务异常
    - tasks：该 Job 包含的 Task 列表，每个 taskInfo 常用字段：
        - state：任务状态，含义同上
        - podName：k8s 中的 pod name，可用于查询 pod 日志
        - workflowName：k8s 中的 workflowName，可用于查询日志
        - stateAt：任务开始时间，毫秒时间戳
        - params：任务参数，不同任务可能不一样，字段定义与 csKtsShowByTaskId 返回的 tags 一致
    """
    parsed_id = _parse_positive_int("task_id", task_id)
    return _find_domains(
        "findDomainsByTaskId",
        {"taskId": parsed_id},
        f"task_id={parsed_id}",
    )
