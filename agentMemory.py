"""任务状态、长期记忆的数据结构与本地 JSON 读写，不调用模型。

每个 Session 的状态文件保存一份 TaskState，memory.json 保存 MemoryEntry 列表。
默认保存到本模块所在目录下的 agent_data，也可以在调用时指定目录。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


_DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "agent_data"

# Literal 用于类型提示；dataclass 不会在运行时自动校验这些取值。
TaskStatus = Literal["pending", "running", "completed", "failed"]
VerificationStatus = Literal["not_run", "passed", "failed"]
MemoryType = Literal["user_preference", "project_convention", "experience"]


def _task_state_path(session_id: str, directory: str | Path) -> Path:
    """将会话标识用于文件名，不允许它引入额外的目录层级。"""
    if not isinstance(session_id, str) or not session_id or "/" in session_id or "\\" in session_id:
        raise ValueError("session_id 必须是非空字符串，且不能包含路径分隔符")
    return Path(directory) / f"state_{session_id}.json"


def _save_json(path: Path, data) -> Path:
    """保存可直接阅读的中文 JSON，返回实际写入的文件路径。"""
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@dataclass
class VerificationResult:
    """实际验证结果，与任务执行状态分开记录，避免将执行结束视为验证通过。"""

    status: VerificationStatus = "not_run"
    summary: str = ""


@dataclass
class TaskState:
    """一个 Session 中当前任务的最新状态，不存储完整聊天历史。"""

    # 路径使用字符串，方便直接转为 JSON；填写工具实际使用的工作目录。
    workspace: str
    session_id: str
    goal: str
    status: TaskStatus = "pending"
    current_step: str = ""
    # 每个实例独立创建列表和验证结果，避免不同任务相互影响。
    constraints: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    verification: VerificationResult = field(default_factory=VerificationResult)

    def save(self, directory: str | Path = _DEFAULT_DIRECTORY) -> Path:
        """保存当前状态到 state_<session_id>.json，同一会话再次保存会覆盖旧状态。"""
        return _save_json(_task_state_path(self.session_id, directory), asdict(self))

    @classmethod
    def load(cls, session_id: str, directory: str | Path = _DEFAULT_DIRECTORY) -> "TaskState":
        """加载指定会话的状态；文件不存在或内容无效时向调用方抛出异常。"""
        path = _task_state_path(session_id, directory)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("任务状态 JSON 顶层必须是对象")
        if data.get("session_id") != session_id:
            raise ValueError("状态文件中的 session_id 与请求的会话不一致")
        # JSON 不保留类信息，需要显式还原嵌套的验证结果。
        data["verification"] = VerificationResult(**data.get("verification", {}))
        return cls(**data)


@dataclass
class MemoryEntry:
    """一条可跨任务复用的长期记忆；单个文件可以保存多条不同类型的记忆。"""

    type: MemoryType
    # global 表示全局适用；项目记忆填写项目路径，读取时由调用方按范围筛选。
    scope: str
    content: str
    # 默认记录创建时的 UTC 时间；修改内容时，由调用方同步更新该字段。
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    # 经验类记忆可记录来源文件或验证结果；没有来源时留空。
    source: str = ""

    @staticmethod
    def save(entries: list["MemoryEntry"], directory: str | Path = _DEFAULT_DIRECTORY) -> Path:
        """将全部记忆写入 memory.json，覆盖旧列表，不自动追加或更新时间。

        调用方式为 MemoryEntry.save(entries, directory)。添加记忆时，先 load，
        向列表追加条目后再 save；修改记忆时由调用方更新对应条目的 updated_at。
        """
        return _save_json(Path(directory) / "memory.json", [asdict(entry) for entry in entries])

    @classmethod
    def load(cls, directory: str | Path = _DEFAULT_DIRECTORY) -> list["MemoryEntry"]:
        """加载记忆列表；尚无文件时返回空列表，其他读写或格式错误继续抛出。"""
        path = Path(directory) / "memory.json"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("长期记忆 JSON 顶层必须是数组")
        return [cls(**item) for item in data]


def select_memories(entries: list[MemoryEntry], workspace: str | Path) -> list[MemoryEntry]:
    """选择全局记忆和当前项目记忆，返回新列表，按原顺序复用原条目。

    scope 为 global 时始终保留；其他非空 scope 作为项目路径，展开 ~ 并转成
    绝对路径后与 workspace 比较。相对路径按当前进程工作目录解析。
    仅匹配同一项目目录，不自动继承父目录或子目录的记忆。
    本函数不修改条目、不读写记忆文件，也不调用模型。
    """
    if not workspace:
        raise ValueError("workspace 不能为空")
    workspace_path = Path(workspace).expanduser().resolve()
    selected = []
    for entry in entries:
        if entry.scope == "global":
            selected.append(entry)
        elif entry.scope and Path(entry.scope).expanduser().resolve() == workspace_path:
            selected.append(entry)
    return selected
