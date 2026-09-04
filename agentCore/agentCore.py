import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests
from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI

__all__ = ["agentLite", "AgentBusyError"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MCP = _REPO_ROOT / "MCPServer" / "server.py"


class AgentBusyError(RuntimeError):
    """Raised when a second callLLM arrives while one is in-flight (persistent mode)."""


class agentLite:
    def __init__(self, mcp_server_script: Path | None = None) -> None:
        self._mcp_client: Client | None = None
        self._persistent = False
        self._mcp_unhealthy = False
        self._busy = False
        self._tools: list[dict[str, Any]] | None = None

        user_id = os.environ.get("HARNESS_USER_ID")
        if user_id is None or not str(user_id).strip():
            raise RuntimeError("HARNESS_USER_ID is not set")
        self._user_id = str(user_id).strip()
        logging.info("HARNESS_USER_ID is set (len=%d)", len(self._user_id))

        script = Path(mcp_server_script or _DEFAULT_MCP).resolve()
        if not script.is_file():
            raise FileNotFoundError(f"MCP server script not found: {script}")
        self._local_mcp_server = StdioServerParameters(
            command=sys.executable,
            args=[str(script)],
            cwd=str(script.parent),
        )

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key or not str(api_key).strip():
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self._openai_client = AsyncOpenAI(
            api_key=str(api_key).strip(),
            base_url="https://api.deepseek.com",
        )
        self._extra_body = {
            "thinking": {"type": "enabled"},
            "user_id": self._user_id,
        }

    async def start_mcp(self) -> None:
        """Lifespan startup: enter persistent mode and spawn a new Client."""
        self._persistent = True
        await self._ensure_mcp_client()

    async def stop_mcp(self) -> None:
        """Lifespan shutdown: leave persistent mode and reap the subprocess."""
        self._persistent = False
        await self._close_mcp_client()
        logging.info("MCP stdio stopped")

    async def _close_mcp_client(self) -> None:
        client = self._mcp_client
        self._mcp_client = None
        self._mcp_unhealthy = True
        self._tools = None
        if client is None:
            return
        try:
            await client.__aexit__(None, None, None)
        except Exception as exc:
            logging.exception(
                "MCP Client __aexit__ failed; subprocess may linger: %s", exc
            )
            raise

    async def _ensure_mcp_client(self) -> None:
        """Must run in a non-cancelled task (next request or lifespan startup)."""
        if self._mcp_client is not None and not self._mcp_unhealthy:
            if self._tools is None:
                raise RuntimeError(
                    "persistent MCP client is healthy but tools cache is missing"
                )
            return
        await self._close_mcp_client()
        script = Path(self._local_mcp_server.args[0])
        if not script.is_file():
            raise FileNotFoundError(f"MCP server script not found: {script}")
        client = Client(self._local_mcp_server)
        try:
            await client.__aenter__()
        except Exception as exc:
            logging.exception("failed to start MCP stdio client: script=%s", script)
            raise RuntimeError(f"failed to start MCP stdio client: {script}") from exc
        self._mcp_client = client
        self._mcp_unhealthy = False
        try:
            tools = await self._load_mcp_tools(client)
        except Exception:
            logging.exception("list_tools failed after MCP start: script=%s", script)
            try:
                await self._close_mcp_client()
            except Exception as close_exc:
                logging.exception(
                    "MCP close after list_tools failure also failed: %s", close_exc
                )
            raise
        if not tools:
            try:
                await self._close_mcp_client()
            except Exception as close_exc:
                logging.exception(
                    "MCP close after empty list_tools also failed: %s", close_exc
                )
            raise RuntimeError("MCP list_tools returned no tools")
        self._tools = tools
        logging.info(
            "MCP stdio started: script=%s python=%s tools=%d",
            script,
            sys.executable,
            len(tools),
        )

    async def callLLM(self, input_msg: str) -> str:
        if not isinstance(input_msg, str):
            raise TypeError(
                f"input_msg must be str, got {type(input_msg).__name__}"
            )
        if self._persistent:
            if self._busy:
                raise AgentBusyError(
                    "agent is already processing a request; retry after it finishes"
                )
            self._busy = True
            try:
                await self._ensure_mcp_client()
                mcp_client = self._mcp_client
                if mcp_client is None:
                    raise RuntimeError("persistent MCP client is missing after ensure")
                return await self._invoke(mcp_client, input_msg)
            except asyncio.CancelledError:
                self._mcp_unhealthy = True
                self._tools = None
                logging.warning(
                    "callLLM cancelled: input_preview=%r", input_msg[:200]
                )
                raise
            except BaseException:
                self._mcp_unhealthy = True
                self._tools = None
                raise
            finally:
                self._busy = False
        else:
            async with Client(self._local_mcp_server) as mcp_client:
                return await self._invoke(mcp_client, input_msg)

    async def _getUserBalance(self) -> str:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        def _fetch() -> requests.Response:
            return requests.get(
                "https://api.deepseek.com/user/balance",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                timeout=30,
            )

        try:
            response = await asyncio.to_thread(_fetch)
        except requests.RequestException as exc:
            logging.exception("DeepSeek balance HTTP failed")
            raise RuntimeError("Failed to get balance") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get balance: {response.status_code} - {response.text[:500]}"
            )
        try:
            balance = response.json()["balance_infos"][0]["total_balance"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Error parsing balance from response") from exc
        return f"当前账户余额为 {balance} 元"

    async def _invoke(self, mcp_client: Client, input_msg: str) -> str:
        try:
            answer = await self._simpleAgentLoop(
                self._openai_client,
                mcp_client,
                input_msg,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("empty model output_text")
            logging.info("[answer]: %s", answer)
            return answer
        except Exception:
            logging.exception(
                "agent invoke failed: input_preview=%r", input_msg[:200]
            )
            raise

    async def _load_mcp_tools(self, mcp_client: Client) -> list[dict[str, Any]]:
        result = await mcp_client.list_tools()
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            for tool in result.tools
        ]

    async def _simpleAgentLoop(
        self,
        openai_client: AsyncOpenAI,
        mcp_client: Client,
        input_msg: str,
    ) -> str:
        if input_msg == "/balance":
            return await self._getUserBalance()

        loop_context = [
            {
                "role": "system",
                "content": (
                    "仅输出与用户最终问题直接相关的答案，不要输出中间分析、"
                    "推理过程、工具调用过程、临时结果或与最终答案无关的内容。"
                ),
            },
            {
                "role": "user",
                "content": input_msg,
            },
        ]

        if self._persistent:
            if not self._tools:
                raise RuntimeError("persistent MCP tools cache is empty")
            tools = self._tools
        else:
            tools = await self._load_mcp_tools(mcp_client)

        response = await openai_client.responses.create(
            model="deepseek-v4-flash",
            input=loop_context,
            tools=tools,
            stream=False,
            extra_body=self._extra_body,
        )

        while True:
            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            if not tool_calls:
                return response.output_text

            loop_context.extend(response.output)

            for call in tool_calls:
                try:
                    args = json.loads(call.arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid tool arguments for {call.name}: {exc}"
                    ) from exc
                result = await mcp_client.call_tool(
                    call.name,
                    args,
                )
                loop_context.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                })

            response = await openai_client.responses.create(
                model="deepseek-v4-flash",
                input=loop_context,
                tools=tools,
                stream=False,
                extra_body=self._extra_body,
            )
