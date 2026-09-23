import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from tools import runSafeTest
from agentLite import AgentLite
import asyncio
import uuid
import subprocess
from agents import set_tracing_disabled

# 不使用 OpenAI 的 tracing 服务
set_tracing_disabled(True)

@dataclass
class evalCase:
    id: str
    prompt: str
    sourceWorkspace: str
    testWorkspace: str
    status: int



def runEvalTest(project_path: str, test_path: str) -> dict:
    """
    在临时 Docker 容器中执行 pytest 验证代码改动是否正确。
    每次调用创建一个新容器，执行完成后自动删除。

    使用场景：
    1. 修复 Bug 或修改代码后，调用本工具验证修改结果。
    2. 测试失败时，根据返回的错误信息分析问题。
    3. 测试通过后，才能报告相关测试已通过。
    4. 未发现测试用例时，不得认为测试通过。

    输入：
        project_path: 被测试源码目录
        test_path: pytest 测试目录

    返回：
        status: passed、failed 或 no_tests
        success: 是否测试通过
        exit_code: 返回码
        output: pytest 的测试输出及错误信息
    """

    project = Path(project_path).resolve()
    tests = Path(test_path).resolve()

    if not project.is_dir() or not tests.is_dir():
        return {
            "status": "error",
            "success": False,
            "exit_code": None,
            "output": "源码目录或测试目录不存在",
        }

    # 为每次测试生成唯一容器名称
    container_name = f"agent-test-{uuid.uuid4().hex[:12]}"

    command = [
        "docker", "run", "--rm",
        "--name", container_name,
        

        # 基础隔离
        "--network", "none",
        "--user", "1000:1000",
        "--cap-drop", "ALL", # 移除 Linux 特权能力
        "--security-opt", "no-new-privileges", # 禁止获取新的特权
        "--memory", "512m",
        "--cpus", "1",
        "--pids-limit", "128",
        "--read-only",

        # 为 pytest 提供临时写入空间
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",

        # 源码和固定测试均为只读
        "--mount", f"type=bind,src={project},dst=/workspace,readonly",
        "--mount", f"type=bind,src={tests},dst=/tests",

        # 工作目录及 Python 导入路径
        "-w", "/workspace",
        "-e", "PYTHONPATH=/workspace",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "HOME=/tmp",

        # Docker 镜像
        "agent-python:3.10",

        # 容器内执行的 pytest 命令
        "python", "-m", "pytest",
        "/tests",
        "-q",
        "-x",
        "--tb=short",
        "-p", "no:cacheprovider",
    ]
    print(f'执行测试, dokcer: {container_name}')

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f'测试结束： {result}')

    except subprocess.TimeoutExpired:
        # 超时时强制删除容器，避免测试进程残留
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=15,
        )

        return {
            "status": "timeout",
            "success": False,
            "exit_code": None,
            "output": "测试执行超时（60 秒）",
        }

    except OSError as exc:
        return {
            "status": "error",
            "success": False,
            "exit_code": None,
            "output": str(exc),
        }

        # 根据 pytest 退出码判断结果
        # Docker 自身启动失败也可能返回非零退出码
    status_map = {
        0: "passed",
        1: "failed",
        2: "interrupted",
        3: "internal_error",
        4: "usage_error",
        5: "no_tests",
    }

    if result.returncode in (125, 126, 127):
        status = "execution_error"
    else:
        status = status_map.get(
            result.returncode,
            "execution_error",
        )

    return {
        "status": status,
        "success": status == "passed",
        "exit_code": result.returncode,
        "output": result.stdout + result.stderr,
    }


async def runEvaluate(case: evalCase):
    source = Path(case.sourceWorkspace).resolve()
    tests = Path(case.testWorkspace).resolve()

    # 1. 为当前用例创建独立源码副本
    workspace = Path(tests) / "workspace"
    shutil.copytree(source, workspace, dirs_exist_ok=True)

    # 2. 修复前运行测试
    before = runEvalTest(str(workspace), str(tests))

    # 3. 调用agent修改代码/
    case.prompt = f"扫描{workspace}下的代码。" + case.prompt
    agent = AgentLite()
    await agent.initDSClientWithMCP()
    res = await agent.simpleCallLLM(case.prompt)

    # 4. 修复后再次测试
    after = runEvalTest(str(workspace), str(tests))

    # 5. 在临时目录删除前，提取需要保留的结果
    result = {
        "before_exit_code": before["exit_code"],
        "after_exit_code": after["exit_code"],
        "passed": after["exit_code"] == 0,
        "output": after["output"],
    }

    return result


if __name__ == '__main__':
    case = evalCase(
        id="fake_case_1",
        prompt="修复 divide(a, b)：当 b == 0 时应抛出 ValueError；正常除法行为保持不变。不需要运行测试。",
        sourceWorkspace="/media/xumaozhou/data/AI/local_agent_test/evaluate_case_3_samples/case_01_divide_zero/workspace",
        testWorkspace="/media/xumaozhou/data/AI/local_agent_test/evaluate_case_3_samples/case_01_divide_zero/tests",
        status=0,
    )
    eval_res = asyncio.run(runEvaluate(case))
    print(f"evaluate result = {eval_res}")