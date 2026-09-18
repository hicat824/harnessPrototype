from utils.httpUtil import *
from typing import Annotated
from pydantic import Field
from urllib.parse import urlencode
from mcp.server.mcpserver.exceptions import ToolError

from mcp.server import MCPServer

mcp = MCPServer("csktsTools")

_CS_KTS_URL = "http://cs-kts.gzproduction.com/cs-kts"


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


@mcp.tool()
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


if __name__ == '__main__':
    mcp.run(transport="stdio")