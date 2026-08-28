import logging

from balanceApi import *
from tools import *

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] >>> %(message)s',
    datefmt='%Y/%m/%d %H:%M'
)

TEST_USR_ID = "8f3c1a7d6b2e4f90a5c8d1e7b3f6492c"


def testAgentTools():
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    input_tokens = 0
    output_tokens = 0

    context = [
        {
            "role": "user",
            "content": "检索 /media/xumaozhou/data/harnessPrototype_workspace/test_case 目录下的所有文件，读取S2车型对应clip的meta.json，根据metaInfo中的城市信息查询并输出对应城市的天气",
        }
    ]

    response = client.responses.create(
        model="deepseek-v4-flash",
        input=context,
        tools=tools,
        stream=False,
        extra_body={
            "thinking": {"type": "enabled"},
            "user_id": TEST_USR_ID
        }
    )

    while True:
        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        if not tool_calls:
            logging.info(
                f"[RESPONSE][{response.id}]: {response.output_text} . 输入token数： {input_tokens}, 输出token数： {output_tokens}")
            break

        context.extend(response.output)

        for tool_call in tool_calls:
            print(
                f"[agent] call tool: "
                f"{tool_call.name}({tool_call.arguments})"
            )
            arguments = json.loads(tool_call.arguments)
            result = call_tool(
                tool_call.name,
                arguments,
            )
            context.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                }
            )

            response = client.responses.create(
                model="deepseek-v4-flash",
                input=context,
                tools=tools,
                stream=False,
                extra_body={
                    "thinking": {"type": "enabled"},
                    "user_id": TEST_USR_ID
                }
            )


def testReflection():
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    input_tokens = 0
    output_tokens = 0

    context = [
        {
            "role": "user",
            "content": "使用PYTHON编写一个检查素数的函数",
        }
    ]

    response = client.responses.create(
        model="deepseek-v4-flash",
        input=context,
        stream=False,
        instructions="你是一名 Python 开发助手，回答尽量简洁。", # 系统级命令
        extra_body={
            "thinking": {"type": "enabled"},
            "user_id": TEST_USR_ID
        }
    )

    input_tokens += response.usage.input_tokens
    output_tokens += response.usage.output_tokens
    logging.info(response.output_text)

    context = [
        {
            "role": "user",
            "content": "检查这个函数是否有问题",
        }
    ]
    context.extend(response.output)

    response = client.responses.create(
        model="deepseek-v4-flash",
        input=context,
        stream=False,
        instructions="你是一名 Python 开发助手，回答尽量简洁。",  # 系统级命令
        extra_body={
            "thinking": {"type": "enabled"},
            "user_id": TEST_USR_ID
        }
    )

    input_tokens += response.usage.input_tokens
    output_tokens += response.usage.output_tokens
    logging.info(response.output_text)

if __name__ == "__main__":
    logging.info("---------------------> START <---------------------")
    testReflection()
    # testAgentTools()