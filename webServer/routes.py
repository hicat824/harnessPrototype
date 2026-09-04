from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from agentCore.agentCore import AgentBusyError, agentLite

_INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


async def _read_json_object(request: Request, max_body_bytes: int) -> dict:
    """Read JSON with a byte cap.

    Starlette ``max_body_size`` returns plaintext 413 and overwrites a JSON
    envelope when ``Content-Length`` is present, so the limit is enforced here.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise HTTPException(400, "invalid Content-Length") from exc
        if declared > max_body_bytes:
            raise HTTPException(413, "Content Too Large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_body_bytes:
            raise HTTPException(413, "Content Too Large")
        chunks.append(chunk)
    raw = b"".join(chunks)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body must be an object")
    return body


def _agent(request: Request) -> agentLite:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise RuntimeError("agent is not initialized")
    return agent


async def index(request: Request) -> FileResponse:
    return FileResponse(_INDEX_HTML)


async def health(request: Request) -> JSONResponse:
    agent = _agent(request)
    settings = request.app.state.settings
    mcp_ready = (
        agent._persistent
        and agent._mcp_client is not None
        and not agent._mcp_unhealthy
    )
    return JSONResponse({
        "status": "ok" if mcp_ready else "degraded",
        "busy": agent._busy,
        "mcp_ready": mcp_ready,
        "timeout_seconds": settings.agent_timeout_seconds,
    })


async def post_chat(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise HTTPException(400, "Content-Type must be application/json")
    body = await _read_json_object(request, settings.max_body_bytes)
    extra = set(body) - {"message"}
    if extra:
        raise HTTPException(400, f"unknown fields: {sorted(extra)}")
    message = body.get("message")
    if not isinstance(message, str):
        raise HTTPException(400, "message must be a string")
    message = message.strip()
    if not message:
        raise HTTPException(400, "message must be a non-empty string")
    if len(message) > settings.message_max_chars:
        raise HTTPException(400, "message exceeds max length")

    logging.info("POST /api/chat start: preview=%r length=%d", message[:200], len(message))
    agent = _agent(request)
    t0 = time.monotonic()
    try:
        answer = await asyncio.wait_for(
            agent.callLLM(message),
            timeout=settings.agent_timeout_seconds,
        )
    except AgentBusyError:
        raise
    except (asyncio.TimeoutError, TimeoutError) as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logging.exception(
            "agent timeout: duration_ms=%d input_preview=%r",
            duration_ms,
            message[:200],
        )
        raise HTTPException(504, "agent timed out") from exc

    if not isinstance(answer, str) or not answer.strip():
        logging.error(
            "callLLM returned empty or non-str: type=%r value=%r",
            type(answer),
            answer,
        )
        raise RuntimeError("empty model output_text")
    duration_ms = int((time.monotonic() - t0) * 1000)
    logging.info(
        "POST /api/chat done: duration_ms=%d answer_len=%d",
        duration_ms,
        len(answer),
    )
    return JSONResponse({"answer": answer})
