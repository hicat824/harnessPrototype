"""Week8 Day1：Models、Messages 和一次模型调用练习。

独立学习代码，不修改 AgentLite，也不提前重构 RAG。
输入是纯文本消息，输出是 AIMessage；与 SDK 的对照只展示参数，不执行 Runner。
运行与测试说明见 langchain_learning/week8.md。
"""

import copy
import json
import os
from pathlib import Path
import time

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context


DEFAULT_INSTRUCTIONS = "你是一个学习助手。根据用户提供的信息简洁回答，不编造事实。"


def _text(value, name: str) -> None:
    """校验纯文本但不 strip 返回值，保留用户原来的换行、空格和标识符。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空白字符串")


def build_messages(instructions: str, question: str, history: list[dict] | None = None) -> list[BaseMessage]:
    """把本次输入转换为 LangChain Message 列表。

    输入：instructions 为规则文本；question 为本轮问题；history 为历史文本消息，
    仅接受 {"role": "user" 或 "assistant", "content": 非空字符串}，None 等价于 []。
    输出：[SystemMessage, *历史 HumanMessage/AIMessage, 本轮 HumanMessage]。
    顺序和原文保持不变，不修改 history。SystemMessage 是规则，不把它放入用户历史。
    本日不转换工具调用、多模态或 SDK 的所有 Session 条目；不支持的输入抛 ValueError。

    历史中的每一项已经是带角色的消息对象，组装时直接展开，保持一条记录对应一条消息。
    """
    _text(instructions, "instructions")
    _text(question, "question")
    if history is not None and not isinstance(history, list):
        raise ValueError("history 必须是列表或 None")
    history_messages = []
    for item in history if history is not None else []:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError("历史消息只允许 role 和 content")
        if item["role"] not in ("user", "assistant"):
            raise ValueError("历史消息 role 只能是 user/assistant")
        _text(item["content"], "history.content")
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        history_messages.append(message_type(content=item["content"]))

    # * 将历史列表逐项展开，不把多条历史塞进一条 BaseMessage 的 content。
    # 历史为空时展开零项，最终仍为“规则 + 本轮问题”；整个过程不修改原列表。
    messages = [
        SystemMessage(content=instructions),
        *history_messages,
        HumanMessage(content=question),
    ]
    return messages


def to_sdk_request(messages: list[BaseMessage]) -> dict:
    """已提供：展示相同文本在 Agents SDK 中如何表达，不发起 SDK 请求。

    输入须以一个 SystemMessage 开始，后续只允许 HumanMessage/AIMessage。
    输出 {"instructions": str, "input": [{"role": "user/assistant", "content": str}, ...]}。
    instructions 对应 Agent(instructions=...)，input 对应 Runner.run(agent, input=...)。
    这只是本日纯文本子集的对照，不是通用 Session/工具结果转换器。
    无效类型或空正文抛 ValueError，返回独立字典，不修改消息。
    """
    if not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[0], SystemMessage):
        raise ValueError("消息须以 SystemMessage 开始，且至少包含一条对话消息")
    _text(messages[0].content, "instructions")
    items = []
    for message in messages[1:]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            role = "assistant"
        else:
            raise ValueError("本日只比较纯文本 HumanMessage/AIMessage")
        _text(message.content, "message.content")
        items.append({"role": role, "content": message.content})
    return {"instructions": messages[0].content, "input": items}


def build_model(*, api_key: str | None = None, model: str = "deepseek-flash",
                http_client: httpx.Client | None = None,
                http_async_client: httpx.AsyncClient | None = None) -> ChatOpenAI:
    """已提供：创建 DeepSeek 的兼容接口适配器，创建本身不请求模型。

    沿用既有 OPENAI_API_KEY（本项目中存放 DeepSeek Key），不读取 IDE 配置或打印密钥。
    使用 Chat Completions、关闭自动重试；底层 HTTP 客户端可由调用方注入 Mock。
    不绑定 tools，不创建 Agent、不添加 Memory。返回 ChatOpenAI，供 run_query 调用。
    密钥缺失抛 RuntimeError，模型名为空抛 ValueError。
    注入的 HTTP 客户端由调用方负责关闭。
    """
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("真实调用需要项目既有的 DeepSeek OPENAI_API_KEY")
    _text(model, "model")
    return ChatOpenAI(
        model=model, api_key=key, base_url="https://api.deepseek.com", timeout=60,
        max_retries=0, max_tokens=512, temperature=0, use_responses_api=False,
        stream_usage=False, extra_body={"thinking": {"type": "disabled"}},
        http_client=http_client, http_async_client=http_async_client,
    )


async def run_query(model, instructions: str, question: str, *, history: list[dict] | None = None,
                    output_path: str | Path | None = None) -> dict:
    """执行一次模型调用，返回可用于观察消息及 SDK 对照的报告。

    输入：model 提供 async ainvoke(messages) -> AIMessage，可是真实 ChatOpenAI
    或离线替身；其他输入见 build_messages。output_path 为可选 JSON 输出位置。
    输出：输入 messages 的 type/content、SDK 对照、响应的 type/content、usage、
    部分响应元数据和 elapsed_seconds。usage 缺失保留 None，不猜测 Token 数量。

    await model.ainvoke(messages) 接收 LangChain 消息列表，sdk_request 只用于接口对照。
    ainvoke 返回的是一条模型消息，不能把它当成 SDK 的 RunResult.final_output。

    本日只接收非空纯文本 AIMessage；工具调用或内容块输出明确报错，不冒充已执行。
    模型、HTTP、超时或响应异常原样传播，不自动重试、不改写为成功回答。
    仅成功时保存报告；旧同名文件可能仍在，失败不能用旧文件证明本次成功。
    """
    messages = build_messages(instructions, question, history)
    sdk_request = to_sdk_request(messages)
    started = time.perf_counter()
    # 学习过程不上传额外的 LangSmith trace，离线替身也使用同一调用路径。
    with tracing_context(enabled=False):
        # 模型接收实际的 LangChain 消息；SDK 对照字典不能代替该输入。
        # await 等待一次异步调用，返回 AIMessage，后面再读取正文和 usage。
        response = await model.ainvoke(messages)

    if not isinstance(response, AIMessage):
        raise ValueError("聊天模型应返回 AIMessage，而不是字符串或 SDK RunResult")
    if response.tool_calls:
        raise ValueError("本日只处理文本回答，不执行模型提出的工具调用")
    _text(response.content, "response.content")
    report = {
        "messages": [{"type": message.type, "content": message.content} for message in messages],
        "sdk_request": sdk_request,
        "response": {"type": response.type, "content": response.content},
        "usage": copy.deepcopy(response.usage_metadata),
        "response_metadata": {key: copy.deepcopy(response.response_metadata[key])
                              for key in ("model_name", "finish_reason") if key in response.response_metadata},
        "elapsed_seconds": time.perf_counter() - started,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Week8 Day1：LangChain 纯文本模型与消息")
    parser.add_argument("--real-llm", action="store_true", help="显式允许调用既有 DeepSeek 接口")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--output", default="langchain_learning/artifacts/day1/real_model/report.json")
    args = parser.parse_args()
    if not args.real_llm:
        parser.error("真实调用需要显式添加 --real-llm；先完成两个步骤并运行离线测试")

    async def main():
        # 测试时注入 MockTransport；真实 CLI 忽略代理变量并显式关闭客户端。
        with httpx.Client(trust_env=False) as client:
            async with httpx.AsyncClient(trust_env=False) as async_client:
                model = build_model(model=args.model, http_client=client, http_async_client=async_client)
                report = await run_query(model, DEFAULT_INSTRUCTIONS, args.question, output_path=args.output)
        print(report["response"]["content"])
        print(f"Week8 Day1 产物：{Path(args.output).resolve()}")

    asyncio.run(main())
