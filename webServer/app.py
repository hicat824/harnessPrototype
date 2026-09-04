from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from agentCore.agentCore import AgentBusyError, agentLite
from webServer.routes import health, index, post_chat

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_HTTP_ERROR_CODES = {
    400: "invalid_request",
    405: "method_not_allowed",
    413: "payload_too_large",
    504: "agent_timeout",
}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    agent_timeout_seconds: float
    message_max_chars: int
    graceful_shutdown_seconds: float
    max_body_bytes: int = 64 * 1024


def _env_raw(name: str) -> str | None:
    return os.environ.get(name)


def detect_lan_ipv4() -> str:
    """Return the IPv4 used for the default route (UDP connect does not send)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 80))
        ip = sock.getsockname()[0]
    except OSError as exc:
        logging.error("failed to detect LAN IPv4: %s; set --host or HARNESS_HOST", exc)
        raise SystemExit(1) from exc
    finally:
        sock.close()
    if not ip or ip.startswith("127."):
        logging.error(
            "failed to detect LAN IPv4, got %r; set --host or HARNESS_HOST", ip
        )
        raise SystemExit(1)
    return ip


def _positive_number(name: str, raw: str, as_int: bool) -> int | float:
    try:
        value = int(raw) if as_int else float(raw)
    except ValueError as exc:
        logging.error("%s is not a number: %r", name, raw)
        raise SystemExit(1) from exc
    if not math.isfinite(float(value)) or value <= 0:
        logging.error("%s must be a finite number > 0, got %r", name, raw)
        raise SystemExit(1)
    return value


def parse_settings(argv: list[str] | None = None) -> Settings:
    host_default = _env_raw("HARNESS_HOST")
    if host_default is not None and not host_default.strip():
        logging.error("HARNESS_HOST is empty")
        raise SystemExit(1)
    if host_default is None:
        host_default = detect_lan_ipv4()
    else:
        host_default = host_default.strip()

    port_default = 8000
    port_env = _env_raw("HARNESS_PORT")
    if port_env is not None:
        try:
            port_default = int(port_env)
        except ValueError as exc:
            logging.error("HARNESS_PORT is not an integer: %r", port_env)
            raise SystemExit(1) from exc

    timeout_default: float = 180
    timeout_env = _env_raw("HARNESS_AGENT_TIMEOUT")
    if timeout_env is not None:
        timeout_default = float(
            _positive_number("HARNESS_AGENT_TIMEOUT", timeout_env, as_int=False)
        )

    max_chars_default = 8000
    max_chars_env = _env_raw("HARNESS_MAX_MESSAGE_CHARS")
    if max_chars_env is not None:
        max_chars_default = int(
            _positive_number("HARNESS_MAX_MESSAGE_CHARS", max_chars_env, as_int=True)
        )

    graceful_default: float = 5
    graceful_env = _env_raw("HARNESS_GRACEFUL_SHUTDOWN")
    if graceful_env is not None:
        graceful_default = float(
            _positive_number("HARNESS_GRACEFUL_SHUTDOWN", graceful_env, as_int=False)
        )

    parser = argparse.ArgumentParser(description="harnessPrototype web service")
    parser.add_argument(
        "--host",
        default=host_default,
        help="bind address (default: this machine's LAN IPv4)",
    )
    parser.add_argument("--port", type=int, default=port_default)
    parser.add_argument("--timeout", type=float, default=timeout_default)
    parser.add_argument("--max-message-chars", type=int, default=max_chars_default)
    parser.add_argument("--graceful-shutdown", type=float, default=graceful_default)
    args = parser.parse_args(argv)

    if not args.host.strip():
        logging.error("host must be a non-empty string, got %r", args.host)
        raise SystemExit(1)
    if args.port < 1 or args.port > 65535:
        logging.error("port must be 1-65535, got %r", args.port)
        raise SystemExit(1)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        logging.error("timeout must be a finite number > 0, got %r", args.timeout)
        raise SystemExit(1)
    if args.max_message_chars <= 0:
        logging.error(
            "max-message-chars must be > 0, got %r", args.max_message_chars
        )
        raise SystemExit(1)
    if not math.isfinite(args.graceful_shutdown) or args.graceful_shutdown <= 0:
        logging.error(
            "graceful-shutdown must be a finite number > 0, got %r",
            args.graceful_shutdown,
        )
        raise SystemExit(1)

    return Settings(
        host=args.host.strip(),
        port=args.port,
        agent_timeout_seconds=args.timeout,
        message_max_chars=args.max_message_chars,
        graceful_shutdown_seconds=args.graceful_shutdown,
    )


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = _HTTP_ERROR_CODES.get(exc.status_code, "invalid_request")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if exc.status_code >= 500:
        logging.exception("http error %s: %s", exc.status_code, detail)
    else:
        logging.info("http %s %s: %s", exc.status_code, code, detail)
    headers = {}
    if exc.status_code in {400, 413}:
        headers["connection"] = "close"
    return JSONResponse(
        _error_body(code, detail),
        status_code=exc.status_code,
        headers=headers,
    )


async def agent_busy_handler(request: Request, exc: AgentBusyError) -> JSONResponse:
    logging.info("agent busy: %s", exc)
    return JSONResponse(
        _error_body("agent_busy", str(exc)),
        status_code=429,
    )


async def timeout_handler(request: Request, exc: TimeoutError) -> JSONResponse:
    logging.exception("agent timeout: %s", exc)
    return JSONResponse(
        _error_body("agent_timeout", str(exc) or "agent timed out"),
        status_code=504,
    )


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.exception("unhandled error: %s", exc)
    return JSONResponse(
        _error_body("internal_error", str(exc)),
        status_code=500,
    )


def create_app(settings: Settings) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette):
        agent = agentLite()
        try:
            await agent.start_mcp()
        except BaseException:
            await agent.stop_mcp()
            raise
        app.state.agent = agent
        try:
            yield
        finally:
            await agent.stop_mcp()

    if not _STATIC_DIR.is_dir():
        raise FileNotFoundError(f"static directory not found: {_STATIC_DIR}")

    app = Starlette(
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/api/chat", post_chat, methods=["POST"]),
            Mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static"),
        ],
        lifespan=lifespan,
        exception_handlers={
            HTTPException: http_exception_handler,
            AgentBusyError: agent_busy_handler,
            TimeoutError: timeout_handler,
            asyncio.TimeoutError: timeout_handler,
            Exception: unhandled_handler,
        },
    )
    app.state.settings = settings
    return app
