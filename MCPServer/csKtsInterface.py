from typing import Annotated
from urllib.parse import urlencode

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from httpUtil import HttpRequestError, httpReqWithJsonResp
from resultCache import project_and_cache

__all__ = [
    "csKtsShowByTaskId",
    "csKtsShowByTaskSeq",
    "csKtsInvokeQuery",
    "csKtsInsertOrUpdateTags",
    "csKtsFindRelationByCsKtsTaskId",
    "csKtsFindRelationByLabelTaskId",
]

_CS_KTS_URL = "http://cs-kts.gzproduction.com/cs-kts"
_DEFAULT_LP_SRC = "pod"
_RELATION_REQUIRED_KEYS = ("lpSrc", "csKtsTaskId", "osmTaskId", "labelTaskId")


def _parse_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{name} must be a numeric id, got {value!r}") from exc
    if parsed <= 0:
        raise ToolError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _parse_task_id(task_id: str) -> int:
    return _parse_positive_int("task_id", task_id)


def _require_non_empty(name: str, value: str | None) -> str:
    if value is None:
        raise ToolError(f"{name} must not be None")
    text = str(value).strip()
    if not text:
        raise ToolError(f"{name} must not be empty")
    return text


def _extract_task_tags(response: dict, identifier: str) -> dict:
    try:
        tags = response["tags"]
    except (TypeError, KeyError) as exc:
        raise ToolError(f"tags not found in task info response for {identifier}") from exc
    if not isinstance(tags, dict):
        raise ToolError(
            f"tags is not an object in task info response for {identifier}: {type(tags).__name__}"
        )
    return tags


def _query_task_tags(url: str, identifier: str) -> dict:
    try:
        response = httpReqWithJsonResp(url)
    except HttpRequestError as exc:
        raise ToolError(f"failed to query task info for {identifier}: {exc}") from exc
    return _extract_task_tags(response, identifier)


def csKtsShowByTaskId(
    task_id: Annotated[str, Field(description="全数字的任务 id，例如 441152")],
) -> dict:
    """
    通过数字 taskId 查询任务详细信息。

    taskId 与 taskSeq 都是任务的唯一标识：
    - taskId：全数字
    - taskSeq：全字母

    返回值就是任务的 tags 字典本身，不是 {"tags": ...} 包装对象。
    tags 是任务的动态扩展属性集合，不同任务包含的字段可能不同。

    常用字段包括：
    - 入库任务：
        - 数据来源 dataSource
        - 关联轨迹长度 originalTotalTrackMileage
        - HDFS存储路径 inputHdfsPath
    - 预标注自动化任务：
        - 项目ID projectId
        - 轨迹ID trackIds
        - lidar文件HDFS地址 lidarFiles
        - 是否启用点云赋彩 enableLidarColorization

    当用户询问的属性未出现在上述常用字段中时：
    1. 检查返回的 tags 字典中是否存在与用户需求语义匹配的字段；
    2. 优先匹配字段名完全一致或含义明确相近的字段；
    3. 不要凭空推断不存在的字段；
    4. 如果存在多个可能匹配的字段且无法确定，应向用户说明候选字段，而不是自行猜测。
    """
    parsed_id = _parse_task_id(task_id)
    query = urlencode({"id": parsed_id})
    url = f"{_CS_KTS_URL}/task/show?{query}"
    return _query_task_tags(url, f"task_id={parsed_id}")


def csKtsShowByTaskSeq(
    task_seq: Annotated[str, Field(description="全字母的任务 id，例如 kk1qlM")],
) -> dict:
    """
    通过字母 taskSeq 查询任务详细信息。

    taskId 与 taskSeq 都是任务的唯一标识：
    - taskId：全数字
    - taskSeq：全字母

    对同一个任务，返回信息与 csKtsShowByTaskId 完全一致。
    应在用户只提供 taskSeq、或上下文中只有全字母任务 id 时使用本工具。

    返回值就是任务的 tags 字典本身，不是 {"tags": ...} 包装对象。
    tags 是任务的动态扩展属性集合，不同任务包含的字段可能不同。

    当用户询问的属性未出现在常用字段中时：
    1. 检查返回的 tags 字典中是否存在与用户需求语义匹配的字段；
    2. 优先匹配字段名完全一致或含义明确相近的字段；
    3. 不要凭空推断不存在的字段；
    4. 如果存在多个可能匹配的字段且无法确定，应向用户说明候选字段，而不是自行猜测。
    """
    parsed_seq = _require_non_empty("task_seq", task_seq)
    query = urlencode({"taskSeq": parsed_seq})
    url = f"{_CS_KTS_URL}/task/showByTaskSeq?{query}"
    return _query_task_tags(url, f"task_seq={parsed_seq}")


def csKtsInvokeQuery(
    engine_code: Annotated[
        str,
        Field(description="环节/流程编码，使用公司统一的 engineCode，例如 oem-bev-s2-frame1"),
    ],
    task_id: Annotated[str, Field(description="全数字的任务 id，例如 441152")],
) -> dict:
    """
    查询指定任务指定环节的 JobInfo json。

    参数：
    - engine_code：环节/流程编码，使用公司统一的 engineCode 编码体系。
      当用户给出中文流程名（如“bev作业任务资料生成”）时，应先映射为对应 engineCode 再调用。
    - task_id：全数字的任务 id。

    返回 {items, cache}。items 通常只有一条，只含：
    - taskSeq：全字母的任务 id，可继续用于 csKtsShowByTaskSeq
    - name：环节名称，对应 engineCode
    - taskBusinessId：数字 taskId
    - bizCode：业务编码
    - params_keys：流程控制参数的 k 列表，不含 v
    - tasks：精简后的 taskInfo，每项含 taskId 和 params_keys（来自 input.params 的 k 列表）

    完整 JobInfo（含 params 的 v、tasks[].input.params）已落盘，路径在 cache[].path。
    需要具体参数时调用 readCachedResult(path, json_path)，
    例如 params.directParam-up-image-version 或 tasks.0.input.params.trackIds。
    读取 params 时按 k 精确匹配，不要凭空推断不存在的字段。
    """
    parsed_engine = _require_non_empty("engine_code", engine_code)
    parsed_id = _parse_task_id(task_id)
    url = f"{_CS_KTS_URL}/invoke/query"
    payload = {
        "engineCode": parsed_engine,
        "taskId": parsed_id,
    }
    try:
        response = httpReqWithJsonResp(url, payload, method="POST")
    except HttpRequestError as exc:
        raise ToolError(
            f"failed to query JobInfo for engine_code={parsed_engine!r}, "
            f"task_id={parsed_id}: {exc}"
        ) from exc

    if not isinstance(response, dict):
        raise ToolError(
            f"invoke/query returned invalid JobInfo for engine_code={parsed_engine!r}, "
            f"task_id={parsed_id}: {response!r}"
        )
    return project_and_cache(
        "invoke_query",
        response,
        f"engine_code={parsed_engine!r}, task_id={parsed_id}",
    )


def csKtsInsertOrUpdateTags(
    task_id: Annotated[str, Field(description="全数字的任务 id，例如 441152")],
    k: Annotated[str, Field(description="要写入的参数名，例如 testKey")],
    v: Annotated[str, Field(description="要写入的参数值，按字符串保存，例如 testVal")],
) -> dict:
    """
    往指定任务信息中添加或更新参数。

    添加成功后，该字段会以 kv 形式出现在任务信息查询接口
    （csKtsShowByTaskId / csKtsShowByTaskSeq）返回的 tags 中。

    参数：
    - task_id：全数字的任务 id。
    - k：要写入的参数名。
    - v：要写入的参数值，按字符串保存。

    返回 success=true 表示操作成功。失败时抛出异常，异常信息包含接口返回的失败原因。
    """
    parsed_id = _parse_task_id(task_id)
    parsed_k = _require_non_empty("k", k)
    if v is None:
        raise ToolError("v must not be None")
    parsed_v = str(v)

    url = f"{_CS_KTS_URL}/task/insertOrUpdateTags"
    payload = {
        "taskBusinessTags": [
            {
                "id": 0,
                "k": parsed_k,
                "taskId": parsed_id,
                "v": parsed_v,
            }
        ],
        "taskIds": str(parsed_id),
    }
    try:
        data = httpReqWithJsonResp(url, payload, method="POST")
    except HttpRequestError as exc:
        raise ToolError(
            f"failed to insert or update tags for task_id={parsed_id}, k={parsed_k!r}: {exc}"
        ) from exc

    return {
        "success": True,
        "task_id": str(parsed_id),
        "k": parsed_k,
        "v": parsed_v,
        "data": data,
    }


def _query_label_platform_relation(url: str, identifier: str) -> dict:
    try:
        response = httpReqWithJsonResp(url)
    except HttpRequestError as exc:
        raise ToolError(
            f"failed to query label platform relation for {identifier}: {exc}"
        ) from exc
    if not isinstance(response, dict):
        raise ToolError(
            f"label platform relation is invalid for {identifier}: {response!r}"
        )
    missing = [key for key in _RELATION_REQUIRED_KEYS if key not in response]
    if missing:
        raise ToolError(
            f"label platform relation missing fields {missing} for {identifier}: {response!r}"
        )
    return response


def csKtsFindRelationByCsKtsTaskId(
    task_id: Annotated[
        str,
        Field(description="4dBEV 导入流程任务 taskId，全数字，例如 440448"),
    ],
) -> dict:
    """
    通过 4dBEV 导入流程任务 taskId 查询标注平台任务关联关系。

    返回字段：
    - lpSrc：标注平台任务类型 labelplatformSource
    - csKtsTaskId：导入流程任务 id
    - osmTaskId：矢量数据对应任务 id
    - labelTaskId：标注平台任务 id
    """
    parsed_id = _parse_task_id(task_id)
    query = urlencode({"csKtsTaskId": parsed_id})
    url = f"{_CS_KTS_URL}/labelPlatform/findRelationByCsKtsTaskId?{query}"
    return _query_label_platform_relation(url, f"csKtsTaskId={parsed_id}")


def csKtsFindRelationByLabelTaskId(
    label_task_id: Annotated[
        str,
        Field(description="标注平台任务 id，全数字，例如 49643"),
    ],
    lp_src: Annotated[
        str,
        Field(description="标注平台任务类型 labelplatformSource，默认 pod"),
    ] = _DEFAULT_LP_SRC,
) -> dict:
    """
    通过标注平台任务 id 查询与 CS-KTS 导入流程任务的关联关系。

    lp_src 为标注平台任务类型 labelplatformSource，默认 pod，通常不需要传入。

    返回字段：
    - lpSrc：标注平台任务类型 labelplatformSource
    - csKtsTaskId：导入流程任务 id
    - osmTaskId：矢量数据对应任务 id
    - labelTaskId：标注平台任务 id
    """
    parsed_label_id = _parse_positive_int("label_task_id", label_task_id)
    parsed_lp_src = _require_non_empty("lp_src", lp_src)
    query = urlencode({
        "labelTaskId": parsed_label_id,
        "lpSrc": parsed_lp_src,
    })
    url = f"{_CS_KTS_URL}/labelPlatform/findRelationByLabelTaskId?{query}"
    return _query_label_platform_relation(
        url, f"labelTaskId={parsed_label_id}, lpSrc={parsed_lp_src!r}"
    )
