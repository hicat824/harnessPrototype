"""仅使用临时目录验证状态与记忆的本地读写，不调用 LLM。"""

import json
from pathlib import Path
import pytest
from pygments.lexers import verification

from agentMemory import MemoryEntry, TaskState, VerificationResult, select_memories

_LOCAL_TEST_ROOT = Path("/media/xumaozhou/data/AI/local_agent_test/test_task_state_and_memory").resolve()

@pytest.mark.parametrize(
    "status, current_step, todos, verification",
    [
        ("pending", "任务初始化", ["分析 a.py"], VerificationResult(status="not_run", summary="")),
        ("running", "分析 a.py", ["修改 a.py"], VerificationResult(status="not_run", summary="")),
        ("running", "修改 a.py", ["测试 a.py"], VerificationResult(status="not_run", summary="")),
        ("running", "测试 a.py", [], VerificationResult(status="failed", summary="一项断言失败")),
        ("running", "修改 a.py", ["测试 a.py"], VerificationResult(status="not_run", summary="")),
        ("running", "测试 a.py", [], VerificationResult(status="passed", summary="")),
        ("completed", "提交任务", [], VerificationResult(status="not_run", summary="")),
    ]
)
def test_task_state_and_memory_processing(tmp_path, status, current_step, todos, verification):
    state = TaskState(
        workspace=str(_LOCAL_TEST_ROOT), session_id="session_a", goal="修复 a.py",
        status=status, current_step=current_step, todos=todos, constraints=["必须使用 Python 3.10"],
        verification=verification
    )
    directory = tmp_path / "agent_data"
    path = state.save(directory)
    assert path == directory / "state_session_a.json"
    assert "修复 a.py" in path.read_text(encoding="utf-8")
    restored = TaskState.load("session_a", directory)
    assert restored == state
    assert isinstance(restored.verification, VerificationResult)



def test_task_state_round_trip_restores_nested_result(tmp_path):
    state = TaskState(
        workspace="/workspace", session_id="session_a", goal="修复 a.py",
        status="running", current_step="运行测试",
        constraints=["必须使用 Python 3.10"], todos=["检查测试结果"],
        verification=VerificationResult(status="failed", summary="一项断言失败"),
    )
    directory = tmp_path / "agent_data"
    path = state.save(directory)
    assert path == directory / "state_session_a.json"
    assert "修复 a.py" in path.read_text(encoding="utf-8")
    restored = TaskState.load("session_a", directory)
    assert restored == state
    assert isinstance(restored.verification, VerificationResult)


def test_state_update_does_not_overwrite_another_session(tmp_path):
    first = TaskState(workspace="/workspace", session_id="first", goal="修复 a.py")
    second = TaskState(workspace="/workspace", session_id="second", goal="修复 b.py")
    first.save(tmp_path)
    second.save(tmp_path)
    first.status = "completed"
    first.verification = VerificationResult(status="passed", summary="测试通过")
    first.save(tmp_path)
    assert TaskState.load("first", tmp_path) == first
    assert TaskState.load("second", tmp_path) == second


def test_memory_list_round_trip_and_append_preserve_previous_entries(tmp_path):
    first = MemoryEntry(
        type="user_preference", scope="global", content="优先采用简单实现",
        updated_at="2026-10-16T10:00:00+08:00",
    )
    directory = tmp_path / "agent_data"
    path = MemoryEntry.save([first], directory)
    assert path == directory / "memory.json"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)
    assert "优先采用简单实现" in path.read_text(encoding="utf-8")
    entries = MemoryEntry.load(directory)
    second = MemoryEntry(
        type="experience", scope="/workspace", content="补丁需要上下文",
        source="补丁应用验证结果",
    )
    entries.append(second)
    MemoryEntry.save(entries, directory)
    assert MemoryEntry.load(directory) == [first, second]
    # 保存不会给已有记忆重新生成更新时间。
    assert MemoryEntry.load(directory)[0].updated_at == first.updated_at


def test_missing_memory_file_returns_empty_list_without_creating_directory(tmp_path):
    directory = tmp_path / "not_created"
    assert MemoryEntry.load(directory) == []
    assert not directory.exists()


def test_empty_memory_list_can_be_saved(tmp_path):
    MemoryEntry.save([], tmp_path)
    assert json.loads((tmp_path / "memory.json").read_text(encoding="utf-8")) == []
    assert MemoryEntry.load(tmp_path) == []


def test_missing_task_state_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TaskState.load("missing", tmp_path)


@pytest.mark.parametrize("filename,loader", [
    ("memory.json", lambda directory: MemoryEntry.load(directory)),
    ("state_sample.json", lambda directory: TaskState.load("sample", directory)),
])
def test_corrupt_json_is_not_treated_as_empty_data(tmp_path, filename, loader):
    path = tmp_path / filename
    path.write_text('{"broken":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        loader(tmp_path)
    assert path.read_text(encoding="utf-8") == '{"broken":'


@pytest.mark.parametrize("filename,content,loader", [
    ("memory.json", "{}", lambda directory: MemoryEntry.load(directory)),
    ("state_sample.json", "[]", lambda directory: TaskState.load("sample", directory)),
])
def test_wrong_json_top_level_is_rejected(tmp_path, filename, content, loader):
    (tmp_path / filename).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        loader(tmp_path)


@pytest.mark.parametrize("session_id", ["", "../outside", "nested/session", "nested\\session"])
def test_session_id_cannot_change_storage_directory(tmp_path, session_id):
    state = TaskState(workspace="/workspace", session_id=session_id, goal="修复 bug")
    with pytest.raises(ValueError, match="session_id"):
        state.save(tmp_path)
    with pytest.raises(ValueError, match="session_id"):
        TaskState.load(session_id, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_mismatched_session_id_is_rejected(tmp_path):
    state = TaskState(workspace="/workspace", session_id="first", goal="修复 bug")
    path = state.save(tmp_path)
    path.rename(tmp_path / "state_second.json")
    with pytest.raises(ValueError, match="不一致"):
        TaskState.load("second", tmp_path)


def test_select_memories_keeps_global_and_current_project_in_order(tmp_path):
    project = tmp_path / "project"
    global_entry = MemoryEntry(type="user_preference", scope="global", content="简单实现")
    other_entry = MemoryEntry(type="project_convention", scope=str(tmp_path / "other"), content="其他项目")
    project_entry = MemoryEntry(type="experience", scope=str(project), content="项目经验")
    another_global = MemoryEntry(type="experience", scope="global", content="通用经验")
    entries = [global_entry, other_entry, project_entry, another_global]
    before = [(entry.scope, entry.content, entry.updated_at) for entry in entries]

    selected = select_memories(entries, project)
    assert selected == [global_entry, project_entry, another_global]
    assert selected is not entries
    assert selected[0] is global_entry
    assert [(entry.scope, entry.content, entry.updated_at) for entry in entries] == before
    selected.clear()
    assert len(entries) == 4


def test_select_memories_normalizes_project_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    entries = [
        MemoryEntry(type="project_convention", scope=str(project) + "/", content="绝对路径"),
        MemoryEntry(type="experience", scope="project/src/..", content="相对路径"),
    ]
    assert select_memories(entries, "project/.") == entries


def test_select_memories_does_not_inherit_parent_child_or_similar_project(tmp_path):
    project = tmp_path / "project"
    entries = [
        MemoryEntry(type="experience", scope=str(tmp_path), content="父目录"),
        MemoryEntry(type="experience", scope=str(project / "child"), content="子目录"),
        MemoryEntry(type="experience", scope=str(tmp_path / "project_extra"), content="名称相近的项目"),
    ]
    assert select_memories(entries, project) == []


def test_select_memories_handles_empty_list_and_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert select_memories([], tmp_path) == []
    invalid_entry = MemoryEntry(type="experience", scope="", content="未指定范围")
    assert select_memories([invalid_entry], tmp_path) == []
    with pytest.raises(ValueError, match="workspace"):
        select_memories([], "")


def test_day4_memory_selection(tmp_path):
    """Day4 第二项验收：共享一份记忆文件，两个项目分别选择适用记忆。"""
    project_a = tmp_path / "projects" / "project_a"
    project_b = tmp_path / "projects" / "project_b"
    directory = tmp_path / "agent_data"

    # 1. 明确区分用户全局偏好与两个项目各自的约定。
    global_preference = MemoryEntry(
        type="user_preference", scope="global", content="添加必要注释",
    )
    convention_a = MemoryEntry(
        type="project_convention", scope=str(project_a), content="使用 Python 3.10",
    )
    convention_b = MemoryEntry(
        type="project_convention", scope=str(project_b), content="使用 Python 3.12",
    )
    entries = [global_preference, convention_a, convention_b]

    # 2. 全部记忆保存到同一个 memory.json，再从文件重新加载。
    path = MemoryEntry.save(entries, directory)
    assert path == directory / "memory.json"
    saved_content = path.read_bytes()
    loaded = MemoryEntry.load(directory)
    assert loaded == entries

    # 3. 两个项目都能取得全局偏好，但不能混入另一个项目的约定。
    selected_a = select_memories(loaded, project_a)
    selected_b = select_memories(loaded, project_b)
    assert selected_a == [global_preference, convention_a]
    assert selected_b == [global_preference, convention_b]

    # 4. 没有专属记忆的项目仅取得全局偏好；选择操作不改变共享记忆。
    project_c = tmp_path / "projects" / "project_c"
    assert select_memories(loaded, project_c) == [global_preference]
    assert loaded == entries
    assert path.read_bytes() == saved_content
