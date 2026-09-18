from agents import set_tracing_disabled
import os
from agentLite import AgentLite
import asyncio

# 不使用 OpenAI 的 tracing 服务
set_tracing_disabled(True)


async def main():
    agent = AgentLite()

    await agent.initDSClientWithMCP()

    result = await agent.callLLMWithMcpServer()

    print(result)

    await agent.close()

if __name__ == '__main__':
    asyncio.run(main())

    # runner = AgentLite()
    # runner.callLLMWithPreRespId()
    # runner.callLLMWithLocalSession()