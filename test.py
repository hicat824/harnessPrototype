import asyncio
import logging

from agentCore.agentCore import agentLite

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] >>> %(message)s",
    datefmt="%Y/%m/%d %H:%M",
)


if __name__ == "__main__":
    agent = agentLite()
    while True:
        input_msg = input()

        if input_msg == "exit" or input_msg == "exit()":
            logging.fatal("Exiting...")
            break

        logging.info("input_msg: %s", input_msg)
        answer = asyncio.run(agent.callLLM(input_msg=input_msg))
        logging.debug("callLLM returned %s chars", len(answer))
