import os, sys
import logging
from openai import OpenAI
from mcp import Client, StdioServerParameters
import uuid
import json
import requests

__all__ = ["agentLite"]


class agentLite:
    def __init__(self):
        self._user_id = "8f3c1a7d6b2e4f90a5c8d1e7b3f6492c" # FIXME
        self._openai_client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self._local_mcp_server = StdioServerParameters(
            command="python",
            args=[
                "/home/xumaozhou/Music/harnessPrototype/MCPServer/server.py",
            ],
        )
        self._extra_body = {
            "thinking": {"type": "enabled"},
            "user_id": self._user_id
        }


    async def _getUserBalance(self):
        """
            Get the user's balance from the DeepSeek API.
            """
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

        payload = {}
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        response = requests.get("https://api.deepseek.com/user/balance", headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to get balance: {response.status_code} - {response.text}")

        j_balance = response.json()
        try:
            balance_infos = j_balance.get("balance_infos")
            balance = balance_infos[0].get("total_balance")
        except Exception as e:
            raise Exception(f"Error parsing balance from response: {e}")

        return f'当前账户余额为 {balance} 元'


    async def callLLM(self, input_msg):
        async with Client(self._local_mcp_server) as mcp_client:
            answer = await self._simpleAgentLoop(
                self._openai_client,
                mcp_client,
                input_msg,
            )

            logging.info(f'[answer]: {answer}]')


    async def _load_mcp_tools(self, mcp_client):
        result = await mcp_client.list_tools()

        tools = []

        for tool in result.tools:
            tools.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            })

        return tools

    async def _simpleAgentLoop(self, openai_client, mcp_client, input_msg):
        if input_msg == '/balance':
            return await self._getUserBalance()

        loop_context = [
            {
                "role": "system",
                "content": "仅输出与用户最终问题直接相关的答案，不要输出中间分析、推理过程、工具调用过程、临时结果或与最终答案无关的内容。"
            },
            {
                "role": "user",
                "content": input_msg
            }
        ]

        tools = await self._load_mcp_tools(mcp_client)

        response = openai_client.responses.create(
            model="deepseek-v4-flash",
            input=loop_context,
            tools=tools,
            stream=False,
            extra_body=self._extra_body
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
                args = json.loads(call.arguments)
                result = await mcp_client.call_tool(
                    call.name,
                    args,
                )
                loop_context.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                })

            response = openai_client.responses.create(
                model="deepseek-v4-flash",
                input=loop_context,
                tools=tools,
                stream=False,
                extra_body=self._extra_body
            )