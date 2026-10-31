"""Week6 Day4：Agents SDK 连接本地业务 MCP Server。

Init / showMcpTools / callMcpTool 只访问本地 Mock，不需要 API Key。
callLLM 首次调用时才创建 DeepSeek 客户端，沿用现有 Responses 模型配置。
调用方在同一异步任务内依次 await Init()，执行查询，最后在 finally 中 close()。
"""

import os
from pathlib import Path
import sys

from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig, OpenAIResponsesModel
from agents.mcp import MCPServerStdio

from week6 import BusinessResult


_PROJECT_ROOT_PATH = Path(__file__).resolve().parent
_AGENT_INSTRUCTION = """
你是只读业务查询助手，使用 MCP 工具查询任务信息。

- 按任务 ID 查询详情，按 engine_code 查询任务列表。
- 缺少必要参数时先询问用户，不猜测。
- 仅依据工具结果回答，简短说明结论和数据来源。
- 区分查询成功、空结果和失败；结果被截断时明确说明。
- 不编造任务信息，不将任务运行失败误认为查询失败。
"""


class fakeAgentLite:
    """保留原类名，提供本地 MCP 查询及可选的真实 LLM 查询。"""

    def __init__(self):
        self._mcp_server = None
        self._client = None  # Agents SDK Agent，不是 HTTP 客户端。
        self._openai_client = None
        self.last_result = None  # 最近一次成功的 Runner 结果，供测试读取 new_items。

    async def Init(self):
        """启动并连接本地 MCP；重复调用不会再启动一个子进程。"""
        if self._mcp_server is not None:
            return

        self._mcp_server = MCPServerStdio(
            params={
                "command": sys.executable,
                "args": ["-B", str(_PROJECT_ROOT_PATH / "week6.py")],
                "cwd": str(_PROJECT_ROOT_PATH),
            },
            name="week6_business",
            client_session_timeout_seconds=10,
            # 本服务返回 BusinessResult，模型直接使用其结构化 JSON。
            use_structured_content=True,
        )
        try:
            await self._mcp_server.__aenter__()
        except BaseException:
            # 包括取消初始化的情况，清理已创建的连接后继续抛出原异常。
            await self.close()
            raise

    async def showMcpTools(self) -> list[str]:
        """返回服务当前暴露的工具名称列表；调用前必须 await Init()。"""
        if self._mcp_server is None:
            raise RuntimeError("请先 await Init() 连接 MCP 服务")
        tools = await self._mcp_server.list_tools()
        return [tool.name for tool in tools]

    async def callMcpTool(self, tool_name: str, args: dict) -> BusinessResult:
        """输入工具名和参数字典，返回经过校验的业务结果。

        MCP 执行失败或缺少结构化响应时抛异常；业务查询的 error/empty/partial
        仍作为正常 BusinessResult 返回，不与 MCP 协议错误混淆。
        """
        if tool_name not in await self.showMcpTools():
            raise ValueError(f"unknown tool: {tool_name}")
        result = await self._mcp_server.call_tool(tool_name=tool_name, arguments=args)
        if result.is_error:
            raise RuntimeError(f"MCP 工具执行失败：{tool_name}")
        if result.structured_content is None:
            raise RuntimeError(f"MCP 工具未返回结构化业务结果：{tool_name}")
        return BusinessResult.model_validate(result.structured_content)

    async def callLLM(self, prompt: str) -> str:
        """输入自然语言问题，返回最终文本；真实调用需要 OPENAI_API_KEY。

        本方法不设置 output_type，所以最终回答是字符串。last_result 保留本次
        Runner 返回值，可查看 new_items 中的工具调用、结果以及实际 usage。
        请求失败时保留原异常类型，不自动重跑整个 Agent。
        """
        self.last_result = None
        if self._mcp_server is None:
            raise RuntimeError("请先 await Init() 连接 MCP 服务")
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("真实 LLM 查询需要配置 OPENAI_API_KEY")
            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                timeout=60.0,
            )
            try:
                model = OpenAIResponsesModel(
                    model="deepseek-flash", openai_client=self._openai_client,
                )
                # MCP 已连接，再将同一个服务对象交给 Agent；不重复注册 Function Tools。
                self._client = Agent(
                    name="BusinessQueryAgent", model=model,
                    instructions=_AGENT_INSTRUCTION, mcp_servers=[self._mcp_server],
                )
            except BaseException:
                await self._openai_client.close()
                self._openai_client = None
                raise

        response = await Runner.run(
            self._client, prompt, max_turns=10,
            run_config=RunConfig(tracing_disabled=True),
        )
        self.last_result = response
        return response.final_output

    async def close(self):
        """关闭 MCP 子进程和模型 HTTP 客户端；允许重复调用。"""
        server = self._mcp_server
        http_client = self._openai_client
        self._mcp_server = None
        self._client = None
        self._openai_client = None
        try:
            if server is not None:
                await server.__aexit__(None, None, None)
        finally:
            # 即使 MCP 清理报错，也继续关闭 HTTP 连接。
            if http_client is not None:
                await http_client.close()
