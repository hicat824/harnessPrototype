from typing import Annotated, Any
from urllib.parse import urlencode

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from httpUtil import HttpRequestError, httpReqWithJsonResp

__all__ = [
    "kdMaterialQueryTrack",
]

_KD_MATERIAL_URL = "http://kd-material-manager.gzproduction.com/kd-material-manager"


def _require_non_empty(name: str, value: str | None) -> str:
    if value is None:
        raise ToolError(f"{name} must not be None")
    text = str(value).strip()
    if not text:
        raise ToolError(f"{name} must not be empty")
    return text


def _extract_tracks(payload: Any, track_id: str) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "tracks" in payload:
        tracks = payload["tracks"]
        if not isinstance(tracks, list):
            raise ToolError(
                f"queryTrack tracks is not a list for track_id={track_id!r}: {type(tracks).__name__}"
            )
        return tracks
    raise ToolError(
        f"queryTrack returned invalid payload for track_id={track_id!r}: {type(payload).__name__}"
    )


def kdMaterialQueryTrack(
    track_id: Annotated[
        str,
        Field(description="轨迹 id，例如 kk1imK_1785211909114。通常 trackId.split('_')[0] 为对应入库任务的 taskSeq"),
    ],
) -> list:
    """
    查询已经入库轨迹的信息。

    返回值为 tracks 列表。空列表表示该轨迹没有成功入库过。
    列表中每个对象对应该轨迹在某个 pose 库下的轨迹信息，常用字段：
    - verTimestamps：当前 pose 库中轨迹的版本号，毫秒时间戳
    - taskId：对应入库任务的 taskSeq
    - poseTag：对应 pose 库名称
    - geometry：WKT 格式的轨迹形态
    - totalMileage：轨迹总长度
    - otherFiles：HDFS 杂项数据存放目录，其上一级即为对应入库数据 HDFS 存储根目录
    - previousTrackId：同一批数据下前一个轨迹 id，允许为空
    - nextTrackId：同一批数据下后一个轨迹 id，允许为空
    - sourceTag：轨迹来源标识

    通常 trackId.split('_')[0] 为对应入库任务的 taskSeq。
    """
    parsed_track_id = _require_non_empty("track_id", track_id)
    query = urlencode({"trackId": parsed_track_id})
    url = f"{_KD_MATERIAL_URL}/track/queryTrack?{query}"
    try:
        payload = httpReqWithJsonResp(url)
    except HttpRequestError as exc:
        raise ToolError(
            f"failed to query track for track_id={parsed_track_id!r}: {exc}"
        ) from exc
    return _extract_tracks(payload, parsed_track_id)
