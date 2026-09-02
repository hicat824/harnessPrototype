import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

__all__ = [
    "get_by_path",
    "project_and_cache",
    "readCachedResult",
]

_logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".mcp_result_cache"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_TEMPLATE_FIELD = re.compile(r"\{(\w+)\}")


class JsonPathNotFound(ToolError):
    """Raised when a JSON path segment does not exist."""


@dataclass(frozen=True)
class NestedExtractRule:
    keep: tuple[str, ...]
    keys_from: str | None = None
    keys_output: str = "params_keys"


@dataclass(frozen=True)
class ExtractRule:
    list_path: str | None
    keep: tuple[str, ...]
    cache_as: str
    nested: dict[str, NestedExtractRule] = field(default_factory=dict)
    kv_key_fields: dict[str, str] = field(default_factory=dict)
    cache_index_keys: tuple[str, ...] = ()


EXTRACT_RULES = {
    "query_tracks": ExtractRule(
        list_path=".",
        keep=("poseTag", "totalMileage", "taskId"),
        cache_as="{trackId}_{poseTag}.json",
        cache_index_keys=("trackId", "poseTag"),
    ),
    "find_domains": ExtractRule(
        list_path=".",
        keep=("engineCode", "state", "taskSeq", "taskBusinessId"),
        nested={
            "tasks": NestedExtractRule(
                keep=("state", "podName", "workflowName", "stateAt"),
                keys_from="params",
            ),
        },
        cache_as="{taskSeq}_{engineCode}.json",
        cache_index_keys=("taskSeq", "engineCode"),
    ),
    "invoke_query": ExtractRule(
        list_path=None,
        keep=("taskSeq", "name", "taskBusinessId", "bizCode"),
        kv_key_fields={"params": "params_keys"},
        nested={
            "tasks": NestedExtractRule(
                keep=("taskId",),
                keys_from="input.params",
            ),
        },
        cache_as="{taskSeq}_{name}.json",
        cache_index_keys=("taskSeq", "name"),
    ),
}


def _require_non_empty(name: str, value: str | None) -> str:
    if value is None:
        raise ToolError(f"{name} must not be None")
    text = str(value).strip()
    if not text:
        raise ToolError(f"{name} must not be empty")
    return text


def _safe_name(value: Any) -> str:
    text = _SAFE_NAME.sub("_", str(value)).strip("._")
    if not text:
        raise ToolError(f"cannot build cache file name from {value!r}")
    return text


def _is_kv_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and "k" in item and "v" in item for item in value)
    )


def get_by_path(data: Any, path: str) -> Any:
    """Resolve a dotted JSON path.

    List indexes use digits. If the current node is a list of {k, v} objects
    and the next segment is not a digit, look up the pair whose k matches.
    """
    if path is None:
        raise ToolError("json_path must not be None")
    stripped = path.strip()
    if stripped == "" or stripped == ".":
        return data

    current = data
    walked: list[str] = []
    for part in stripped.split("."):
        walked.append(part)
        here = ".".join(walked)
        if isinstance(current, list):
            if part.isdigit():
                index = int(part)
                if index < 0 or index >= len(current):
                    raise ToolError(
                        f"json_path index out of range at {here}: {path}"
                    )
                current = current[index]
                continue
            if _is_kv_list(current):
                matched = [item for item in current if item.get("k") == part]
                if not matched:
                    raise JsonPathNotFound(
                        f"json_path kv key {part!r} not found in {path}"
                    )
                if len(matched) > 1:
                    raise ToolError(
                        f"json_path kv key {part!r} is ambiguous in {path}"
                    )
                current = matched[0].get("v")
                continue
            raise ToolError(
                f"json_path {part!r} is not a list index at {here} in {path}"
            )
        if isinstance(current, dict):
            if part not in current:
                raise JsonPathNotFound(
                    f"json_path {part!r} not found at {here} in {path}"
                )
            current = current[part]
            continue
        raise ToolError(
            f"json_path cannot descend into {type(current).__name__} at {here} in {path}"
        )
    return current


def _resolve_list(payload: Any, list_path: str | None, identifier: str) -> list:
    if list_path is None:
        if not isinstance(payload, dict):
            raise ToolError(
                f"expected object payload for {identifier}: {type(payload).__name__}"
            )
        return [payload]
    if list_path == ".":
        node = payload
    else:
        node = get_by_path(payload, list_path)
    if not isinstance(node, list):
        raise ToolError(
            f"extract list at {list_path!r} is not a list for {identifier}: "
            f"{type(node).__name__}"
        )
    return node


def _collect_keys(node: Any, field_name: str, identifier: str) -> list[str]:
    if isinstance(node, dict):
        return list(node.keys())
    if isinstance(node, list):
        return _extract_kv_keys(node, field_name, identifier)
    raise ToolError(
        f"cannot collect keys from {type(node).__name__} for {identifier}.{field_name}"
    )


def _project_nested_item(
    item: Any,
    nested_rule: NestedExtractRule,
    identifier: str,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ToolError(
            f"nested item is not an object for {identifier}: {type(item).__name__}"
        )
    projected: dict[str, Any] = {}
    for key in nested_rule.keep:
        if key in item:
            projected[key] = item[key]
    if nested_rule.keys_from:
        try:
            node = get_by_path(item, nested_rule.keys_from)
        except JsonPathNotFound as exc:
            _logger.info(
                "optional keys_from %s missing for %s: %s",
                nested_rule.keys_from,
                identifier,
                exc,
            )
            node = None
        if node is not None:
            projected[nested_rule.keys_output] = _collect_keys(
                node, nested_rule.keys_from, identifier
            )
    return projected


def _extract_kv_keys(value: Any, field_name: str, identifier: str) -> list[str]:
    if not isinstance(value, list):
        raise ToolError(
            f"{field_name} is not a kv list for {identifier}: {type(value).__name__}"
        )
    keys: list[str] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, dict) or "k" not in pair:
            raise ToolError(
                f"{field_name}[{index}] is not a {{k,v}} object for {identifier}"
            )
        keys.append(pair["k"])
    return keys


def _project_item(
    item: Any,
    rule: ExtractRule,
    identifier: str,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ToolError(
            f"extract item is not an object for {identifier}: {type(item).__name__}"
        )
    projected: dict[str, Any] = {}
    for key in rule.keep:
        if key not in item:
            raise ToolError(f"missing field {key!r} for {identifier}")
        projected[key] = item[key]
    for nested_key, nested_rule in rule.nested.items():
        if nested_key not in item:
            _logger.info("missing nested list %s for %s, using []", nested_key, identifier)
            projected[nested_key] = []
            continue
        raw_list = item[nested_key]
        if not isinstance(raw_list, list):
            raise ToolError(
                f"{nested_key} is not a list for {identifier}: {type(raw_list).__name__}"
            )
        projected[nested_key] = [
            _project_nested_item(
                nested_item,
                nested_rule,
                f"{identifier}.{nested_key}[{index}]",
            )
            for index, nested_item in enumerate(raw_list)
        ]
    for source_field, output_field in rule.kv_key_fields.items():
        if source_field not in item:
            _logger.info("missing kv field %s for %s, using []", source_field, identifier)
            projected[output_field] = []
            continue
        projected[output_field] = _extract_kv_keys(
            item[source_field],
            source_field,
            identifier,
        )
    return projected


def _format_cache_name(template: str, item: dict[str, Any], identifier: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in item:
            raise ToolError(
                f"cache file template field {key!r} missing for {identifier}"
            )
        return _safe_name(item[key])

    return _TEMPLATE_FIELD.sub(replace, template)


def _cache_index_entry(
    item: dict[str, Any],
    rule: ExtractRule,
    path: Path,
    identifier: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": str(path)}
    for key in rule.cache_index_keys:
        if key not in item:
            raise ToolError(f"cache index field {key!r} missing for {identifier}")
        entry[key] = item[key]
    return entry


def _to_path(raw: str) -> Path:
    try:
        return Path(raw)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"invalid cache path {raw!r}: {exc}") from exc


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ToolError(f"failed to resolve path {path}: {exc}") from exc


def _write_cache_file(filename: str, payload: Any) -> Path:
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"failed to create cache directory {_CACHE_ROOT}: {exc}") from exc
    cache_root = _resolve_path(_CACHE_ROOT)
    path = _resolve_path(cache_root / filename)
    if not path.is_relative_to(cache_root):
        raise ToolError(f"cache path is outside cache directory: {filename}")
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            _logger.error("failed to remove temp cache file %s: %s", tmp_path, cleanup_exc)
        raise ToolError(f"failed to write cache file {path}: {exc}") from exc
    _logger.info("wrote result cache %s", path)
    return path


def project_and_cache(
    rule_name: str,
    payload: Any,
    identifier: str,
) -> dict[str, Any]:
    """Project keep fields for context and cache full items to workspace files."""
    try:
        rule = EXTRACT_RULES[rule_name]
    except KeyError as exc:
        raise ToolError(f"unknown extract rule: {rule_name}") from exc

    items = _resolve_list(payload, rule.list_path, identifier)
    prepared: list[tuple[dict[str, Any], str, dict[str, Any], str]] = []
    for index, item in enumerate(items):
        item_id = f"{identifier}[{index}]"
        projected = _project_item(item, rule, item_id)
        filename = _format_cache_name(rule.cache_as, item, item_id)
        prepared.append((projected, filename, item, item_id))

    projected_items: list[dict[str, Any]] = []
    cache_entries: list[dict[str, Any]] = []
    for projected, filename, item, item_id in prepared:
        cache_path = _write_cache_file(filename, item)
        projected_items.append(projected)
        cache_entries.append(_cache_index_entry(item, rule, cache_path, item_id))
    return {"items": projected_items, "cache": cache_entries}


def _resolve_cache_file(path: str) -> Path:
    raw = _to_path(_require_non_empty("path", path))
    cache_root = _resolve_path(_CACHE_ROOT)
    resolved = _resolve_path(raw if raw.is_absolute() else cache_root / raw)
    if not resolved.is_relative_to(cache_root):
        raise ToolError(f"cache path is outside cache directory: {path}")
    if not resolved.is_file():
        raise ToolError(f"cache file not found: {resolved}")
    return resolved


def _inventory(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return {"type": "object", "keys": list(data.keys())}
    if isinstance(data, list):
        info: dict[str, Any] = {"type": "list", "length": len(data)}
        dict_items = [item for item in data if isinstance(item, dict)]
        if dict_items:
            keys: list[str] = []
            seen: set[str] = set()
            for item in dict_items:
                for key in item:
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
            info["item_keys"] = keys
        if _is_kv_list(data):
            info["kv_keys"] = [item["k"] for item in data]
        return info
    return {"type": type(data).__name__}


def readCachedResult(
    path: Annotated[
        str,
        Field(description="缓存文件路径，来自查询工具返回的 cache[].path"),
    ],
    json_path: Annotated[
        str | None,
        Field(
            description=(
                "可选 JSON 路径，例如 otherFiles、tasks.0.params.lidarFiles、params.trackIds。"
                "不传则只返回该文件的字段名/列表长度，不返回全文。"
            ),
        ),
    ] = None,
) -> Any:
    """读取查询工具落盘的完整结果。

    不传 json_path 时只返回目录摘要（keys / length），避免把全文灌回上下文。
    需要具体字段时再传 json_path。对 [{k,v}, ...] 列表，路径段可直接使用 k 名。
    """
    cache_file = _resolve_cache_file(path)
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolError(f"failed to read cache file {cache_file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"invalid JSON in cache file {cache_file}: {exc}") from exc

    stripped_path = None if json_path is None else str(json_path).strip()
    if not stripped_path or stripped_path == ".":
        return {"path": str(cache_file), **_inventory(payload)}
    return get_by_path(payload, stripped_path)
