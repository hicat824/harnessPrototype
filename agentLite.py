import os, sys
import asyncio
from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    set_tracing_disabled,
    SQLiteSession,
    trace,
    handoff, RunContextWrapper, TContext, Tool,
)
from agents.lifecycle import TAgent
from agents.mcp import MCPServerStdio
from agents.realtime import handoffs
from openai import AsyncOpenAI
from tools import (
    EvalContext, FatalToolError, readFile, searchFile, readFileRange,
    listFiles, applyPatch, runSafeTest, limit_expert_output,
)
from agentPrompts import (
    SUPERVISOR_INSTRUCTIONS,
    LOCAL_AGENT_INSTRUCTIONS,
    PYTHON_EXPERT_DESCRIPTION,
    PYTHON_EXPERT_INSTRUCTIONS,
    CPP_EXPERT_DESCRIPTION,
    CPP_EXPERT_INSTRUCTIONS,
)
from typing import Optional
import openai
import uuid
from agents import AgentHooks
from agents.tool_context import ToolContext
from agents.exceptions import ModelTimeoutError


class supervisorHooks(AgentHooks):
    async def on_tool_start(
            self,
            context: RunContextWrapper[TContext],
            agent: TAgent,
            tool: Tool,
    ) -> None:
        call_id = (
            context.tool_call_id
            if isinstance(context, ToolContext)
            else "unknown"
        )
        arguments = (
            context.tool_arguments
            if isinstance(context, ToolContext)
            else ""
        )

        print(
            f"[Supervisor] 开始调用 {tool.name}, "
            f"call_id={call_id}, arguments={arguments}"
        )

    async def on_tool_end(
            self,
            context: RunContextWrapper[TContext],
            agent: TAgent,
            tool: Tool,
            result: object,
    ) -> None:
        call_id = (
            context.tool_call_id
            if isinstance(context, ToolContext)
            else "unknown"
        )
        arguments = (
            context.tool_arguments
            if isinstance(context, ToolContext)
            else ""
        )

        print(
            f"[Supervisor] {tool.name} 调用结束, "
            f"call_id={call_id}, arguments={arguments}"
        )


class localAgentHooks(AgentHooks):
    async def on_start(self, context, agent):
        print(f"[{agent.name}] 开始运行")

    async def on_end(self, context, agent, output):
        output_text = str(output)
        if len(output_text) > 2000:
            output_text = output_text[:2000] + "...[truncated]"

        print(
            f"[{agent.name}] 运行结束, "
            f"output={output_text}"
        )


async def _limited_expert_output(run_result):
    """在专家结论成为 Supervisor 工具结果前限长，控制台 Hook 仅负责打印。"""
    return limit_expert_output(
        run_result.context_wrapper.context,
        run_result.final_output,
        run_result.last_agent.name,
    )


class AgentLite:
    def __init__(self, session=None):
        self._client = None
        self._mcp_server = None
        self._python_agent = None
        self._cpp_agent = None
        self._tools = [
            readFile,
            searchFile,
            readFileRange,
            listFiles,
            applyPatch,
            runSafeTest
        ]
        # self._client = self._initDSClient()
        self._thread_id = os.environ["THREAD_ID"]
        # 测试可传入独立 Session，避免覆盖日常对话数据库。
        self._session_id = session.session_id if session is not None else "testSession"
        self._session = session if session is not None else SQLiteSession(
            self._session_id, "/media/xumaozhou/data/AI/local_agent_test/session.db"
        )

    async def initDSClientWithMCP(self):
        self._session_id = f'session_{uuid.uuid4().hex[:12]}'
        # init mcp server loop
        self._mcp_server = MCPServerStdio(
            params={
                "command": "python",
                "args": ["/home/xumaozhou/Music/harnessPrototype/mcpServer.py"],
            }
        )
        # 手动进入 async context
        await self._mcp_server.__aenter__()

        tools = await self._mcp_server.list_tools()

        print("MCP tools:")
        for tool in tools:
            print(tool.name)

        # 1. 配置 DeepSeek 客户端
        client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.deepseek.com",
        )

        # 2. 配置模型
        model = OpenAIResponsesModel(
            model="deepseek-flash",
            openai_client=client,
        )

        # test multi agent
        self._python_agent = Agent(
            name="PythonExpert",
            instructions=PYTHON_EXPERT_INSTRUCTIONS,
            model=model,
            tools=self._tools,
            mcp_servers=[self._mcp_server],
            hooks=localAgentHooks(),
        )

        self._cpp_agent = Agent(
            name="CppExpert",
            instructions=CPP_EXPERT_INSTRUCTIONS,
            model=model,
            tools=self._tools,
            mcp_servers=[self._mcp_server],
            hooks=localAgentHooks(),
        )

        self._client = Agent(
            name="Supervisor",
            model=model,
            instructions=SUPERVISOR_INSTRUCTIONS,
            tools=[
                self._python_agent.as_tool(
                    tool_name="PythonExpert",
                    tool_description=PYTHON_EXPERT_DESCRIPTION,
                    failure_error_function=None,
                    custom_output_extractor=_limited_expert_output,
                    session=self._session
                ),
                self._cpp_agent.as_tool(
                    tool_name="CppExpert",
                    tool_description=CPP_EXPERT_DESCRIPTION,
                    failure_error_function=None,
                    custom_output_extractor=_limited_expert_output,
                    session=self._session
                ),
            ],
            mcp_servers=[self._mcp_server],
            hooks=supervisorHooks()
        )

    async def close(self):
        if self._mcp_server is not None:
            await self._mcp_server.__aexit__(None, None, None)

    async def simpleCallLLM(
            self, prompt: str, eval_context: Optional[EvalContext] = None,
            *, debug_info: Optional[dict] = None, max_turns: Optional[int] = None,
    ):
        """执行一次请求，仍返回 bool；调试时额外收集结论、usage 和错误。

        debug_info 由调用方提供，便于测试保存结果；调试请求禁用远程 tracing。
        max_turns 未指定时沿用 SDK 默认值。历史选择由调用方在请求前完成。
        """
        if debug_info is not None:
            debug_info.update(final_output=None, usage=None, error=None)
        with trace(
            workflow_name="Conversation", group_id=self._thread_id,
            disabled=debug_info is not None,
        ):
            try:
                # 默认每个任务使用独立计数；多轮任务可显式复用上下文。
                context = eval_context if eval_context is not None else EvalContext()
                run_options = {"session": self._session, "context": context}
                if max_turns is not None:
                    run_options["max_turns"] = max_turns
                if debug_info is not None:
                    run_options["run_config"] = RunConfig(tracing_disabled=True)
                response = await Runner.run(self._client, prompt, **run_options)
                print(f"answer = {response.final_output}")

                usage = response.context_wrapper.usage
                if debug_info is not None:
                    debug_info["final_output"] = response.final_output
                    debug_info["usage"] = {
                        "requests": usage.requests,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                print("请求次数:", usage.requests)
                print("输入 Token:", usage.input_tokens)
                print("输出 Token:", usage.output_tokens)
                print("总 Token:", usage.total_tokens)
                return True
            except ModelTimeoutError as e:
                if debug_info is not None:
                    debug_info["error"] = f"{type(e).__name__}: {e}"
                print("Agents SDK 模型超时:", e)

            except openai.APITimeoutError as e:
                if debug_info is not None:
                    debug_info["error"] = f"{type(e).__name__}: {e}"
                print("OpenAI SDK HTTP超时:", e)
                print("底层异常:", repr(e.__cause__))

            except FatalToolError as e:
                if debug_info is not None:
                    debug_info["error"] = f"{type(e).__name__}: {e}"
                print(f"工具调用失败：{e}")

            except Exception as e:
                if debug_info is not None:
                    debug_info["error"] = f"{type(e).__name__}: {e}"
                raise

        return False

    async def callLLMWithMcpServer(self):
        with trace(workflow_name="Conversation", group_id=self._thread_id):
            while True:
                prompt = input()
                if prompt == "/exit":
                    return "退出对话"
                if prompt == "/clear":
                    await self._session.clear_session()
                    print("清理对话完成")
                    continue

                try:
                    response = await Runner.run(self._client, prompt, session=self._session,
                                                context=EvalContext())
                    print(f"answer = {response.final_output}")

                    usage = response.context_wrapper.usage
                    print("请求次数:", usage.requests)
                    print("输入 Token:", usage.input_tokens)
                    print("输出 Token:", usage.output_tokens)
                    print("总 Token:", usage.total_tokens)
                except ModelTimeoutError as e:
                    print("Agents SDK 模型超时:", e)

                except openai.APITimeoutError as e:
                    print("OpenAI SDK HTTP超时:", e)
                    print("底层异常:", repr(e.__cause__))

                except FatalToolError as e:
                    print(f"工具调用失败：{e}")

    def _initDSClient(self):
        # 1. 配置 DeepSeek 客户端
        client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.deepseek.com",
        )

        # 2. 配置模型
        model = OpenAIChatCompletionsModel(
            model="deepseek-flash",
            openai_client=client,
        )

        agent = Agent(
            name="testAgent",
            instructions=LOCAL_AGENT_INSTRUCTIONS,
            model=model,
            tools=[readFile, searchFile, readFileRange, listFiles],
        )

        return agent

    def callLLMWithLocalSession(self) -> str:
        with trace(workflow_name="Conversation", group_id=self._thread_id):
            while True:
                prompt = input()
                if prompt == "/exit":
                    return "退出对话"

                try:
                    response = Runner.run_sync(self._client, prompt, session=self._session,
                                               context=EvalContext())
                    print(response.final_output)
                except FatalToolError as e:
                    print(f"工具调用失败：{e}")

    def callLLMWithPreRespId(self) -> str:
        """
        FIXME ： deepseek接口不支持传response_id或者conversion_id
        """
        previous_response_id = None
        while True:
            prompt = input()
            if prompt == "/exit":
                return "退出对话"

            try:
                response = Runner.run_sync(self._client, prompt,
                                           previous_response_id=previous_response_id,
                                           auto_previous_response_id=True,
                                           context=EvalContext())
                previous_response_id = response.last_response_id
                print(response.final_output)
            except FatalToolError as e:
                print(f"工具调用失败：{e}")
