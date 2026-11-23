"""Week11 Day4：通过 SQLite Checkpointer 保存、查看和回读线程状态。"""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from langgraph_learning.day1 import build_day1_graph
from week6 import BusinessResult


def checkpoint_config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    """W11D4-1 构造图执行和快照读取共用的 LangGraph 配置。

    thread_id 是当前线程的存储键。将原值放入外层配置的
    configurable.thread_id。传入 checkpoint_id 时，再将它放入同一个
    configurable 字典的 checkpoint_id 字段。为 None 时省略该字段，
    graph.get_state 会读取该线程的最新快照。

    例如输入 thread_id="task-a"，返回 {"configurable": {"thread_id": "task-a"}}。
    两个标识都由调用方提供，保留原值。空白标识会触发 ValueError。
    """
    if not thread_id.strip():
        raise ValueError("thread_id 不能为空")
    if checkpoint_id is not None and not checkpoint_id.strip():
        raise ValueError("checkpoint_id 不能为空")

    configurable = {"thread_id": thread_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return RunnableConfig(configurable=configurable)


@contextmanager
def open_task_checkpointer(db_path: str | Path) -> Iterator[SqliteSaver]:
    """为任务查询图打开 SQLite Checkpointer，退出时关闭连接。"""
    # 任务查询 State 包含 BusinessResult 实例，精确允许反序列化这一个业务类型。
    serializer = JsonPlusSerializer(allowed_msgpack_modules=[("week6", "BusinessResult")])
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as connection:
        yield SqliteSaver(connection, serde=serializer)


@contextmanager
def open_day4_graph(
    db_path: str | Path, query_task: Callable[[str], BusinessResult],
) -> Iterator[CompiledStateGraph]:
    """打开 SQLite 文件，将 Checkpointer 注入 Day1 图，退出时关闭连接。

    调用方在 with 块内 invoke、get_state 或 get_state_history。以同一个
    SQLite 文件和 thread_id 重新打开图，可以读取此前保存的快照。
    """
    with open_task_checkpointer(db_path) as checkpointer:
        yield build_day1_graph(query_task, checkpointer=checkpointer)
