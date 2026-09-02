from typing import Annotated, Any
from urllib.parse import urlencode

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from httpUtil import HttpRequestError, httpReqWithJsonResp
from resultCache import project_and_cache

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
) -> dict:
    """
    查询已经入库轨迹的信息。

    返回 {items, cache}。items 为空列表表示该轨迹没有成功入库过。
    items 中每条只含：
    - poseTag：pose 库名称
    - totalMileage：轨迹总长度
    - taskId：对应入库任务的 taskSeq

    其余字段（geometry、otherFiles、verTimestamps、previousTrackId、nextTrackId、sourceTag 等）
    已按 {trackId}_{poseTag}.json 落盘，路径在 cache[].path。
    需要详情时调用 readCachedResult(path, json_path)，例如 otherFiles、geometry。

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
    tracks = _extract_tracks(payload, parsed_track_id)
    normalized: list[dict] = []
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ToolError(
                f"queryTrack item is not an object for track_id={parsed_track_id!r}[{index}]: "
                f"{type(track).__name__}"
            )
        if "trackId" not in track:
            track = dict(track)
            track["trackId"] = parsed_track_id
        normalized.append(track)
    return project_and_cache(
        "query_tracks",
        normalized,
        f"track_id={parsed_track_id}",
    )
