"""Week6 Day4～Day6：Agents SDK 连接本地业务 MCP Server。

Init / showMcpTools / callMcpTool 只访问本地 Mock，不需要 API Key。
callLLM 首次调用时才创建 DeepSeek 客户端，沿用现有 Responses 模型配置。
调用方在同一异步任务内依次 await Init()，执行查询，最后在 finally 中 close()。
Day6 的三个函数及验收方法见 docs/week6_day6.md。
"""

import os
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Literal

from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig, OpenAIResponsesModel, ModelSettings, RunContextWrapper
from agents import ToolsToFinalOutputResult
from agents.exceptions import MaxTurnsExceeded, ToolInputGuardrailTripwireTriggered
from agents.mcp import MCPServerStdio
from agents.models.interface import Model
from agents.tool import FunctionToolResult
from agents.tool_guardrails import (
    tool_input_guardrail, tool_output_guardrail, ToolGuardrailFunctionOutput,
    ToolInputGuardrailData, ToolOutputGuardrailData,
)
from pydantic import BaseModel, ConfigDict, Field

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
            # HTTP 只读重试由服务端 Client 负责，MCP 层不叠加自动重试。
            max_retry_attempts=0,
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
        MCP 断连保留 SDK 异常向上抛出，不伪装成业务 empty/error，不自动重连。
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


# Day6：只增加本日的结构化输出、调用保护与单次运行框架。
# 不修改 Day4 callLLM 的文本输出，函数契约与验收方法见 docs/week6_day6.md。
class Day6Answer(BaseModel):
    """Agent 的最终回答；BusinessResult 仍只表示工具返回的数据。

    详情任务 ID 放 task_ids，状态与流程编码原值写入 summary。
    total 仅用于列表匹配总数；未查询时 source=None，不虚构数据来源。
    needs_input 必须给出 missing_fields，其他状态使用空列表。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["success", "empty", "partial", "error", "needs_input"]
    summary: str
    task_ids: list[str]
    total: int | None
    source: str | None
    truncated: bool
    error_code: str | None
    missing_fields: list[str]


class Day6QueryParameters(BaseModel):
    """模型准备的查询参数；通过程序校验之前不允许据此查询 MCP。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    tool_name: Literal["query_task", "query_tasks_by_engine"]
    task_id: str | None = None
    engine_code: str | None = None
    limit: int = Field(default=100, ge=1, le=100)


DAY6_PARAMETER_INSTRUCTIONS = """
只准备一条任务查询的结构化参数，不执行查询，不填写示例或占位值。
按任务 ID 查询时 tool_name="query_task"；按流程编码列任务时为 "query_tasks_by_engine"。
task_id、engine_code 只能逐字提取用户本轮明确提供的值，缺少时填写 null；
不适用的字段也填 null，不推断、省略或改写编码，不把条数当成任务 ID 或流程编码。
limit 使用用户指定的条数，未指定时为 100；仅提供 limit 不代表已经提供 engine_code。
"""


def validate_day6_query_parameters(parameters: Day6QueryParameters, question: str) -> list[str]:
    """校验必填标识符及原文依据，返回需要用户补充/确认的字段名。

    输入：模型提取的结构化参数与本轮用户原文。输出：通过为 []，否则为
    ['task_id'] 或 ['engine_code']。不查询 API、不读取测试答案或 Mock 数据。
    当前只支持显式给出的原值，不处理多轮指代、别名和否定语义。
    limit 的类型和范围由参数模型校验；本函数核验查询标识符的来源。
    """
    field_name = "task_id" if parameters.tool_name == "query_task" else "engine_code"
    value = getattr(parameters, field_name)
    if value is None or not value.strip():
        return [field_name]
    if field_name == "task_id" and re.fullmatch(r"[0-9]+", value) is None:
        return [field_name]
    # 标识符须完整出现在原文中，不能把 123 当作 12345 的依据，
    # 也不能把 import_std_extra 中的一部分当作 import_std。
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])"
    return [] if re.search(pattern, question) else [field_name]


def day6_needs_input_answer(fields: list[str]) -> Day6Answer:
    """程序生成缺参追问，不把没有依据的标识符传给工具。"""
    return Day6Answer(
        status="needs_input", summary=f"请提供或确认 {', '.join(fields)} 后再查询。",
        task_ids=[], total=None, source=None, truncated=False,
        error_code="MISSING_PARAMETER", missing_fields=fields,
    )


@dataclass
class Day6Context:
    """每次 Runner.run 独立创建；不传给模型，也不作为 MCP 工具参数。"""

    max_tool_calls: int = 2
    tool_calls: list[dict] = field(default_factory=list)
    blocked_calls: list[dict] = field(default_factory=list)
    question: str = ""
    parameter_validation: dict | None = None


DAY6_INSTRUCTIONS = """
你是只读任务查询助手，通过 MCP 工具获取事实。
- 从用户问题提取 task_id 或 engine_code/limit，选择对应工具，不猜参数。
- 仅当用户本轮明确提供实际 engine_code 时，才能调用列表查询工具。
  只提供 limit 不代表 engine_code 已知；缺少编码时直接追问，不调用工具探测或确认缺参。
  禁止用占位符、示例值或自行编造的值填充必填参数。
- 缺少必要参数时不调用工具：status="needs_input"，error_code="MISSING_PARAMETER"，
  missing_fields 填写缺少的字段名，在 summary 中简短追问；task_ids=[]、total=null、
  source=null、truncated=false，然后结束本次查询。
- 只依据本次工具结果回答；source 使用工具原值，详情状态和流程编码保留原值。
- task_ids 保持返回顺序；total 使用列表原匹配总数，详情为 null。
- 区分 success、empty、partial、error；保留 error_code 和 truncated。
- 工具返回 status="error" 时，由程序根据实际错误结果直接生成最终回答，
  保留 error_code、source 和 truncated 原值，不再请求模型总结或选择工具。
- 工具失败后禁止再次调用任何工具。HTTP 内部有限重试已由客户端处理，
  用户要求反复查询也不能覆盖此规则；其他情况下也不得重复相同调用。
- 没有工具结果时不编造来源、任务或统计；只给简短结论。
"""


def check_day6_tool_call(
    history: list[dict], tool_name: str, arguments: dict, max_calls: int = 2,
) -> str | None:
    """用已有 Trace 判断是否允许下一次调用，不修改输入。

    输入：本次已放行调用的 history，每条有 name/arguments/result；
    result 为 None（尚未完成）或 BusinessResult 字典。max_calls 是本次允许
    放行的总调用数，仅允许 1～3 的真正 int，否则抛 ValueError。

    按顺序返回第一个阻止原因，没有问题则返回 None：
    1. history 中已收到 status='error' → 'TERMINAL_TOOL_ERROR'。
    2. 相同工具名且参数字典内容相等 → 'REPEATED_TOOL_CALL'，与键顺序无关。
    3. len(history) >= max_calls → 'TOOL_CALL_LIMIT'。
    不在此请求工具、不修改 history、不为 HTTP 内部重试额外计数。
    """
    # bool 是 int 的子类，因此这里用 type，避免把 True 当作上限 1。
    if type(max_calls) is not int or not 1 <= max_calls <= 3:
        raise ValueError("max_calls 必须是 1～3 的整数")

    # 输出护栏通过 model_dump() 保存结果，所以 result 是字典；
    # 尚未完成的调用为 None，不能直接读取 result.status。
    for data in history:
        result = data.get("result")
        if result is not None and result.get("status") == "error":
            return "TERMINAL_TOOL_ERROR"

    # 分开遍历，保证后面出现的终止错误也优先于前面的重复调用。
    for data in history:
        if data.get("name") == tool_name and data.get("arguments") == arguments:
            return "REPEATED_TOOL_CALL"

    if len(history) >= max_calls:
        return "TOOL_CALL_LIMIT"

    return None


@tool_input_guardrail
def day6_input_guardrail(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """已提供的 SDK 接线：执行前校验并记录，不要求用户编写护栏适配代码。"""
    context = data.context.context
    if not isinstance(context, Day6Context):
        return ToolGuardrailFunctionOutput.allow()
    name = data.context.tool_name
    arguments = json.loads(data.context.tool_arguments or "{}")
    code = check_day6_tool_call(context.tool_calls, name, arguments, context.max_tool_calls)
    entry = {"call_id": data.context.tool_call_id, "name": name, "arguments": arguments}
    missing_fields = []
    if code is None:
        # 准备阶段通过后，仍复核实际调用参数；不能默认模型一定沿用先前提取值。
        parameters = Day6QueryParameters(tool_name=name, **arguments)
        missing_fields = validate_day6_query_parameters(parameters, context.question)
        if missing_fields:
            code = "UNCONFIRMED_PARAMETER"
    if code is not None:
        blocked = {**entry, "error_code": code}
        if missing_fields:
            blocked["missing_fields"] = missing_fields
        context.blocked_calls.append(blocked)
        return ToolGuardrailFunctionOutput.raise_exception({"error_code": code})
    # 检查与登记之间没有 await，同轮另一个调用能看到已登记的历史。
    context.tool_calls.append({**entry, "result": None})
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def day6_output_guardrail(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """已提供：将实际 MCP 业务结果按 call_id 写回 Trace。"""
    context = data.context.context
    if isinstance(context, Day6Context):
        # MCP use_structured_content=True 时，SDK 的工具输出是业务 JSON 文本。
        result = BusinessResult.model_validate_json(data.output).model_dump()
        for entry in context.tool_calls:
            if entry["call_id"] == data.context.tool_call_id:
                entry["result"] = result
                break
    return ToolGuardrailFunctionOutput.allow()


def day6_tool_use_behavior(
    context: RunContextWrapper, tool_results: list[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """工具业务失败时直接结束 Runner，其他结果继续交给模型。

    输入：SDK 提供的本次运行上下文和本轮实际工具结果。
    输出：error 时返回 is_final_output=True 和 Day6Answer；其余返回 False。
    两个 MCP 工具均返回 BusinessResult JSON，格式错误继续抛出校验异常，
    不伪装成业务失败，也不重新调用 API。context 由 SDK 传入，此处无需使用。
    """
    for tool_result in tool_results:
        result = BusinessResult.model_validate_json(tool_result.output)
        if result.status == "error":
            # 在下一次请求模型之前结束，避免模型忽略提示词再次查询。
            # 这里保留真实业务错误，与护栏中止的 TERMINAL_TOOL_ERROR 分开。
            answer = Day6Answer(
                status="error", summary=f"查询失败：{result.error_code}，来源：{result.source}。",
                task_ids=[], total=None, source=result.source, truncated=result.truncated,
                error_code=result.error_code, missing_fields=[],
            )
            return ToolsToFinalOutputResult(is_final_output=True, final_output=answer)
    # success、empty、partial 仍由模型组织回答；没有对所有工具一律直接返回。
    return ToolsToFinalOutputResult(is_final_output=False)


def build_day6_agent(model: Model, mcp_server: MCPServerStdio) -> Agent:
    """创建一个用于全部固定问题的 Agent，不启动/关闭连接。

    输入：已配置的 SDK Model（离线 ScriptedModel 或真实 DeepSeek 模型），
    已连接的同一个 MCPServerStdio。输出：Agent 对象。
    - 给 mcp_server.tool_input_guardrails / tool_output_guardrails 分别设置
      [day6_input_guardrail]、[day6_output_guardrail]，使用提供好的 SDK 适配器。
    - 确保 mcp_server.use_structured_content=True。
    - Agent 使用 DAY6_INSTRUCTIONS、output_type=Day6Answer、mcp_servers=[mcp_server]。
    - model_settings=ModelSettings(parallel_tool_calls=False)，不增加 Function Tools。
    - tool_use_behavior=day6_tool_use_behavior，业务错误由执行层直接结束。
    不读取测试预期、不手写自然语言意图解析；实体抽取与工具选择交给 LLM。
    """
    # 输入护栏在调用前执行限制，输出护栏将实际业务结果写回本次 Trace。
    mcp_server.tool_input_guardrails = [day6_input_guardrail]
    mcp_server.tool_output_guardrails = [day6_output_guardrail]
    mcp_server.use_structured_content = True

    # 同一个 MCP 服务已由调用方连接，不再重复注册同名 Function Tools。
    return Agent(
        name="day6_fake_agent",
        model=model,
        instructions=DAY6_INSTRUCTIONS,
        output_type=Day6Answer,
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(parallel_tool_calls=False),
        tool_use_behavior=day6_tool_use_behavior,
    )


def day6_failure_answer(code: str) -> Day6Answer:
    """已提供：执行边界阻止查询时的答复，不冒充 API 业务结果。"""
    return Day6Answer(
        status="error", summary=f"本次查询已停止：{code}", task_ids=[], total=None,
        source=None, truncated=False, error_code=code, missing_fields=[],
    )


async def run_day6_query(agent: Agent, question: str, max_tool_calls: int = 2) -> dict:
    """运行一条问题，返回答复、工具 Trace、usage 和耗时。

    输入：同一个已构建 Agent、自然语言问题、1～3 的真正 int 调用上限。
    不合法的 max_tool_calls 应在调用模型前抛 ValueError。
    每次都创建 Day6Context(max_tool_calls=..., question=question)，再用 RunContextWrapper(context=...)
    包装。不能复用上一条问题的 Trace 或计数，也不注入参考答案和 Session。

    先使用同一模型、不带工具的参数准备 Agent，最多一轮。程序校验提取值的
    原文依据，未通过时直接追问；通过后才执行业务 Agent，max_turns=6。
    两阶段共用 wrapper，usage 和耗时包含参数准备；不重复创建 MCP 连接。
    正常完成时 final_output 是 Day6Answer，runner_completed=True。
    若捕获 ToolInputGuardrailTripwireTriggered，取 context.blocked_calls 最后一条
    的 error_code；无依据参数返回 needs_input，其他边界调用 day6_failure_answer，
    runner_completed=False。被阻止的请求仍保留在 blocked_calls，不隐藏轨迹偏差。
    若捕获 MaxTurnsExceeded，使用 'MAX_TURNS_EXCEEDED'；其他异常保留原样抛出。

    返回普通字典，字段必须包含：
    - question: 原问题；answer: Day6Answer.model_dump()；runner_completed: bool。
    - tool_calls / blocked_calls: context 中实际登记的两份列表。
    - tool_call_count / blocked_call_count: 两份列表的长度。
    - usage: wrapper.usage 的 requests/input_tokens/output_tokens/total_tokens。
      使用 wrapper 可在受控中止后仍读取已完成模型请求的 usage，不能自行估算。
    - elapsed_seconds: perf_counter 的差值，包含本次工具发现/调用，不含服务启动。
    - parameter_validation: 提议的参数、待确认字段和是否通过，便于区分准备阶段与工具执行。

    不在这里根据问题写 if/正则去选工具，不自动重跑整个 Runner，不写评估器。
    模型是 ScriptedModel 时 usage 为测试合成值；真实与模拟的标记由测试报告负责。
    """

    # 在请求模型前校验配置，字符串和 bool 也不能当作合法次数。
    if type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 3:
        raise ValueError("max_tool_calls 必须是 1～3 的整数")

    # Agent 可以复用，但 Trace、调用次数和 usage 必须按本次问题重新创建。
    context = Day6Context(max_tool_calls=max_tool_calls, question=question)
    wrapper = RunContextWrapper(context=context)
    started = time.perf_counter()
    runner_completed = False

    try:
        # 准备 Agent 仅提取参数，不挂载 MCP 或本地工具，因此缺参阶段无法发业务请求。
        preparation_agent = Agent(
            name="Day6ParameterPreparation", model=agent.model,
            instructions=DAY6_PARAMETER_INSTRUCTIONS, output_type=Day6QueryParameters,
        )
        prepared = await Runner.run(
            preparation_agent, question, context=wrapper, max_turns=1,
            run_config=RunConfig(tracing_disabled=True),
        )
        if not isinstance(prepared.final_output, Day6QueryParameters):
            raise TypeError("参数准备输出必须是 Day6QueryParameters")
        parameters = prepared.final_output
        missing_fields = validate_day6_query_parameters(parameters, question)
        context.parameter_validation = {
            "proposed": parameters.model_dump(), "passed": not missing_fields,
            "missing_fields": missing_fields,
        }
        if missing_fields:
            answer = day6_needs_input_answer(missing_fields)
        else:
            # 将已核验的参数交给业务 Agent，避免它只能重新从原问题开始猜测。
            # 输入护栏仍以原 question 校验，不能把程序追加的文本当作用户来源。
            business_prompt = (
                f"{question}\n\n程序已校验的本轮查询参数：\n{parameters.model_dump_json()}\n"
                "请按上述参数调用对应工具，保持标识符原值。"
            )
            result = await Runner.run(agent, business_prompt, context=wrapper, max_turns=6,
                                      run_config=RunConfig(tracing_disabled=True))
            if not isinstance(result.final_output, Day6Answer):
                raise TypeError("Agent 最终输出必须是 Day6Answer")
            answer = result.final_output
        # 工具业务失败也可以正常结束 Runner；业务状态由 answer.status 表示。
        runner_completed = True
    except ToolInputGuardrailTripwireTriggered:
        # 输入护栏先登记阻止原因，再中止执行；被阻止的请求不会到达 MCP。
        # 如果异常并非来自本日护栏，保留原异常，避免误报阻止原因。
        if not context.blocked_calls:
            raise
        blocked = context.blocked_calls[-1]
        if blocked["error_code"] == "UNCONFIRMED_PARAMETER":
            answer = day6_needs_input_answer(blocked["missing_fields"])
        else:
            answer = day6_failure_answer(blocked["error_code"])
    except MaxTurnsExceeded:
        answer = day6_failure_answer("MAX_TURNS_EXCEEDED")
    # 只处理上面两种受控中止。模型、网络或程序错误自然向上抛出，不自动重跑。

    # 正常结束与受控中止统一组装报告；中止前已完成请求的 usage 仍在 wrapper 中。
    elapsed_seconds = time.perf_counter() - started
    usage = wrapper.usage
    return {
        "question": question,
        "answer": answer.model_dump(),
        "runner_completed": runner_completed,
        "tool_calls": context.tool_calls,
        "blocked_calls": context.blocked_calls,
        "tool_call_count": len(context.tool_calls),
        "blocked_call_count": len(context.blocked_calls),
        "usage": {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
        "elapsed_seconds": elapsed_seconds,
        "parameter_validation": context.parameter_validation,
    }
