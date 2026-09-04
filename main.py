import logging
import os
import sys
from pathlib import Path

import uvicorn

from webServer.app import create_app, parse_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] >>> %(levelname)s %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M",
    )
    settings = parse_settings(sys.argv[1:])
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or not str(api_key).strip():
        logging.error("DEEPSEEK_API_KEY is not set; refusing to start")
        raise SystemExit(1)
    user_id = os.environ.get("HARNESS_USER_ID")
    if user_id is None or not str(user_id).strip():
        logging.error("HARNESS_USER_ID is not set; refusing to start")
        raise SystemExit(1)
    logging.info("HARNESS_USER_ID is set (len=%d)", len(str(user_id).strip()))
    if settings.host not in {"127.0.0.1", "localhost", "::1"}:
        logging.warning(
            "binding %s is not loopback; this prototype has no auth and MCP tools "
            "can reach internal APIs",
            settings.host,
        )
    mcp_script = Path(__file__).resolve().parent / "MCPServer" / "server.py"
    page_url = f"http://{settings.host}:{settings.port}/"
    logging.info(
        "starting web server host=%s port=%s timeout=%s mcp_script=%s python=%s",
        settings.host,
        settings.port,
        settings.agent_timeout_seconds,
        mcp_script,
        sys.executable,
    )
    logging.info("intranet page: %s", page_url)
    app = create_app(settings)
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            workers=1,
            timeout_keep_alive=int(settings.agent_timeout_seconds),
            timeout_graceful_shutdown=settings.graceful_shutdown_seconds,
            log_config=None,
            log_level="info",
        )
    except SystemExit:
        raise
    except Exception as exc:
        logging.exception("web server failed to start: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
