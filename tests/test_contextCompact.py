import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import asyncio
import copy
import hashlib
import os
import shutil
import uuid
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace
from agents import Agent, AgentHooks, SQLiteSession, OpenAIChatCompletionsModel
from agents.models.interface import Model
from agents.tool_context import ToolContext
from openai import AsyncOpenAI

import agentLite
from contextCompact import compact_session, select_history, _is_summary, summarize_history
from tools import _read_text_range

_WORKSPACE = "/media/xumaozhou/data/AI/local_agent_test/week5_day3_context_pruning"
# 真实对话预算与固定样本分开；这里只限制历史 JSON，不代表整个模型 Context。
_DEFAULT_REAL_HISTORY_CHAR_LIMIT = 48000


def _real_history_char_limit():
    """读取真实四轮验证的独立字符预算，可在 PyCharm 环境变量中配置。"""
    limit = int(os.environ.get("CONTEXT_TEST_HISTORY_CHAR_LIMIT", str(_DEFAULT_REAL_HISTORY_CHAR_LIMIT)))
    if limit < 3:
        raise ValueError("CONTEXT_TEST_HISTORY_CHAR_LIMIT 必须至少为 3")
    return limit


def _directory_instructions(workspace, allowed_root):
    """明确本次验证的项目目录和工具授权根，供三个 Agent 一致使用。"""
    relative_project = Path(workspace).relative_to(allowed_root).as_posix()
    return (
        f"\n本次项目相对目录：{relative_project}\n"
        f"工具授权根目录：{allowed_root}\n"
        f"项目扫描：{relative_project}；源码：{relative_project}/src/pipeline.py；"
        f"数据：{relative_project}/data/tasks_large.json、{relative_project}/data/processing_large.log。\n"
        "固定测试目录：eval_tests；专家日志：.tool_results/文件名。\n"
    )


class FakeSession:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = Path(input_file_path)
        self.output_file_path = Path(output_file_path)

    async def get_items(self, limit=None):
        # 每次调用都从本地文件加载数据
        content = json.loads(
            self.input_file_path.read_text(encoding="utf-8")
        )
        items = content["items"]

        if not isinstance(items, list):
            raise ValueError("Session 数据必须是列表")

        if limit is None:
            return items

        if limit <= 0:
            return []

        return items[-limit:]


    async def add_items(self, item):
        if self.output_file_path.exists():
            content = json.loads(
                self.output_file_path.read_text(encoding="utf-8")
            )
        else:
            content = {"items": []}

        items = content.get("items", [])

        if not isinstance(items, list):
            raise ValueError("Session 数据必须是列表")

        if items is None or len(items) == 0:
            items = []
        if not isinstance(item, list):
            raise ValueError("item 必须是列表")

        items.extend(item)
        content["items"] = items

        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))


    async def clear_session(self):
        content = {"items": []}
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False))


@pytest.mark.asyncio
async def test_compact_session():
    fake_summarize_fn = AsyncMock()
    fake_summarize_fn.return_value = "[摘要信息]"

    input_session_json = f'{_WORKSPACE}/session_history.json'
    output_session_json = f'{_WORKSPACE}/selected_context.json'
    session = FakeSession(input_session_json, output_session_json)

    result = await compact_session(session, fake_summarize_fn)
    assert result.get("compacted", False) == True



@pytest.mark.asyncio
async def test_select_history():
    input_session_json = f'{_WORKSPACE}/session_history.json'
    output_session_json = f'{_WORKSPACE}/selected_context.json'
    session = FakeSession(input_session_json, output_session_json)

    select_items = await select_history(session=session, priority_indices=(0,))
    assert len(select_items) > 0

    await session.clear_session()
    await session.add_items(select_items)
    assert Path(output_session_json).exists()



@pytest.mark.parametrize(
    "path, limit",
    [
        (f'{_WORKSPACE}/workspace/data/processing_large.log', 2000),
        (f'{_WORKSPACE}/workspace/data/tasks_large.json', 2000),
    ]
)
def test_read_text_range(path, limit):
    encoded_result = _read_text_range(target=Path(path))
    assert len(encoded_result) <= limit


def _save_debug_json(path, value):
    """保存调试记录，不保存客户端对象或 API Key。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )


def _file_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(root).rglob("*") if path.is_file()
    }


def _recent_two_turns(items):
    """从实际历史提取预期记录，不依赖固定样本的 call_id 或条目数量。"""
    starts = [index for index, item in enumerate(items)
              if item.get("role") == "user" and not _is_summary(item)]
    return items[starts[-2] if len(starts) >= 2 else starts[0]:] if starts else []


def _contains_in_order(items, expected):
    position = 0
    for item in expected:
        try:
            position = items.index(item, position) + 1
        except ValueError:
            return False
    return True


def _tool_records_paired(items):
    pending = set()
    for item in items:
        if item.get("type") == "function_call":
            pending.add(item["call_id"])
        elif item.get("type") == "function_call_output":
            if item["call_id"] not in pending:
                return False
            pending.remove(item["call_id"])
    return not pending


class RecordingModel(Model):
    """转发真实模型调用，记录 SDK 输入；这些记录不是 HTTP 原始报文。"""

    def __init__(self, model, agent_name, requests):
        self.model = model
        self.agent_name = agent_name
        self.requests = requests
        self.round_number = 0

    async def get_response(self, *args, **kwargs):
        record = {
            "round": self.round_number,
            "agent": self.agent_name,
            "instructions": kwargs.get("system_instructions"),
            "input": copy.deepcopy(kwargs["input"]),
            "tools": [{"name": tool.name, "schema": tool.params_json_schema}
                      for tool in kwargs["tools"]],
        }
        self.requests.append(record)
        try:
            response = await self.model.get_response(*args, **kwargs)
            record["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def stream_response(self, *args, **kwargs):
        # 本验证使用非流式 Runner；避免流式调用漏掉调试记录。
        raise NotImplementedError("四轮验证仅支持非流式调用")


class RecordingHooks(AgentHooks):
    """开始事件记录参数，结束事件记录结果；失败调用也能留下开始记录。"""

    def __init__(self, events, starts=None):
        self.events = events
        # 分开保存，避免只有开始事件的失败调用干扰工具结果限长检查。
        self.starts = starts if starts is not None else []
        self.round_number = 0

    async def on_tool_start(self, context, agent, tool):
        record = {
            "round": self.round_number,
            "agent": agent.name,
            "tool": tool.name,
            "call_id": getattr(context, "tool_call_id", None),
            "arguments": getattr(context, "tool_arguments", ""),
        }
        self.starts.append(record)
        print(
            f"[{agent.name}] 开始调用 {tool.name}, "
            f"call_id={record['call_id']}, arguments={record['arguments']}"
        )

    async def on_tool_end(self, context, agent, tool, result):
        self.events.append({
            "round": self.round_number,
            "agent": agent.name,
            "tool": tool.name,
            "call_id": getattr(context, "tool_call_id", None),
            "arguments": getattr(context, "tool_arguments", ""),
            "output": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
        })


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_CONTEXT_TEST") != "1",
    reason="真实 LLM 验证需显式设置 RUN_REAL_LLM_CONTEXT_TEST=1",
)
@pytest.mark.asyncio
async def test_four_round_real_llm():
    """四轮真实对话验收；默认跳过，禁止在普通测试中自动调用 LLM。

    显式启用后，在练习目录下创建独立副本和 SQLiteSession，保留调试产物。
    每轮先选择历史，再调用 AgentLite；第4轮降低选择阈值以覆盖裁剪分支。
    专家不共用 Supervisor Session，避免委派输入干扰真实用户轮次。
    本测试只使用已有本地文件工具，不启动与本练习无关的 MCP 服务。
    历史预算通过 CONTEXT_TEST_HISTORY_CHAR_LIMIT 配置，默认 48000 字符，
    不使用固定样本 manifest 中的 history_char_limit。
    需要可用的 DeepSeek 配置、Docker 和 agent-python:3.10 镜像。
    """
    # 装饰器之外再检查一次，直接调用函数也必须显式启用。
    if os.environ.get("RUN_REAL_LLM_CONTEXT_TEST") != "1":
        pytest.skip("请先显式启用真实 LLM 验证")
    history_char_limit = _real_history_char_limit()
    assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY"

    from agentLite import AgentLite, _limited_expert_output
    from agentPrompts import (
        SUPERVISOR_INSTRUCTIONS, PYTHON_EXPERT_INSTRUCTIONS, CPP_EXPERT_INSTRUCTIONS,
        PYTHON_EXPERT_DESCRIPTION, CPP_EXPERT_DESCRIPTION,
    )
    from tools import EvalContext, runSafeTest

    source = Path(_WORKSPACE)
    manifest = json.loads((source / "eval_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((source / "expected_results.json").read_text(encoding="utf-8"))
    required_texts = json.loads((source / "session_history.json").read_text(encoding="utf-8"))["required_texts"]
    prompts = [line.strip() for line in (source / "prompt.txt").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(prompts) == 4, "练习必须包含四轮提示词"

    run_id = "real_llm_" + uuid.uuid4().hex[:12]
    root = source / "real_llm_runs" / run_id
    root.mkdir(parents=True)
    workspace = root / "workspace"
    test_dir = root / "eval_tests"
    shutil.copytree(source / "workspace", workspace)
    shutil.copytree(source / "eval_tests", test_dir)
    before_files = _file_hashes(workspace)
    fixed_test_hashes = _file_hashes(test_dir)
    report = {
        "passed": False, "run_id": run_id, "history_char_limit": history_char_limit,
        "rounds": [], "checks": {},
    }
    requests, events, starts = [], [], []
    session = SQLiteSession(run_id, str(root / "session.db"))
    client = None

    async def fixed_tests():
        # 使用既有 Docker 工具独立验收，不依据 Agent 的“测试通过”自评。
        payload = json.dumps({"project_path": str(workspace), "test_path": str(test_dir)})
        result = await runSafeTest.on_invoke_tool(
            ToolContext(
                context=EvalContext(workspace=root), tool_name="runSafeTest",
                tool_call_id=uuid.uuid4().hex, tool_arguments=payload,
            ),
            payload,
        )
        return json.loads(result)

    try:
        report["baseline_test"] = await fixed_tests()
        assert report["baseline_test"]["exit_code"] == manifest["expected_before_pytest_exit_code"], \
            "基线测试不符合预期，请检查练习副本或 Docker 环境"

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com")
        real_model = OpenAIChatCompletionsModel(
            model=os.environ.get("CONTEXT_TEST_MODEL", "deepseek-flash"), openai_client=client,
        )
        models = {name: RecordingModel(real_model, name, requests)
                  for name in ("Supervisor", "PythonExpert", "CppExpert")}
        hooks = RecordingHooks(events, starts)
        directory_instructions = _directory_instructions(workspace, root)
        with patch.dict(os.environ, {"THREAD_ID": run_id}):
            lite = AgentLite(session=session)
        python_expert = Agent(
            name="PythonExpert", instructions=PYTHON_EXPERT_INSTRUCTIONS + directory_instructions,
            model=models["PythonExpert"], tools=lite._tools, hooks=hooks,
        )
        cpp_expert = Agent(
            name="CppExpert", instructions=CPP_EXPERT_INSTRUCTIONS + directory_instructions,
            model=models["CppExpert"], tools=lite._tools, hooks=hooks,
        )
        lite._client = Agent(
            name="Supervisor", instructions=SUPERVISOR_INSTRUCTIONS + directory_instructions,
            model=models["Supervisor"], hooks=hooks,
            tools=[
                python_expert.as_tool(
                    tool_name="PythonExpert", tool_description=PYTHON_EXPERT_DESCRIPTION,
                    custom_output_extractor=_limited_expert_output, failure_error_function=None,
                    max_turns=30,
                ),
                cpp_expert.as_tool(
                    tool_name="CppExpert", tool_description=CPP_EXPERT_DESCRIPTION,
                    custom_output_extractor=_limited_expert_output, failure_error_function=None,
                    max_turns=30,
                ),
            ],
        )
        context = EvalContext(workspace=root)

        for number, prompt in enumerate(prompts, 1):
            for model in models.values():
                model.round_number = number
            hooks.round_number = number
            before = await session.get_items()
            recent = copy.deepcopy(_recent_two_turns(before))
            # 独立预算以字符计；预留一个字符，兼容 // 2 估算的取整边界。
            budget = (history_char_limit - 1) // 2
            if number == 4 and before:
                minimal = [before[0], *recent]
                budget = min(budget, len(json.dumps(minimal, ensure_ascii=False)) // 2)
            selected = await select_history(session, history_tokens_thre=budget)
            # 请求之间顺序替换；写入失败时恢复，避免留下半份历史。
            if selected != before:
                try:
                    await session.clear_session()
                    await session.add_items(selected)
                except Exception:
                    await session.clear_session()
                    await session.add_items(before)
                    raise
            # 工具以 context.workspace（运行根目录）解析相对路径，不让模型重写长前缀。
            prompt = prompt.replace(str(source / "workspace"), "workspace")
            prompt = prompt.replace(str(source / "eval_tests"), "eval_tests")
            if number == 4:
                prompt += (
                    "\n最终回答只输出 JSON 对象，字段为 true_error_count、error_line_numbers、"
                    "matching_task_count、matching_task_ids、changed_files、test_status。"
                )
            row = {
                "round": number, "prompt": prompt, "history_tokens_threshold": budget,
                "selected_history_chars": len(json.dumps(selected, ensure_ascii=False)),
                "history_reduced": selected != before,
                "recent_turns_complete": _contains_in_order(selected, recent),
                "priority_constraints_kept": not before or all(
                    text in json.dumps(selected, ensure_ascii=False) for text in required_texts
                ),
            }
            report["rounds"].append(row)
            _save_debug_json(root / f"round_{number}_before.json", {"items": before})
            _save_debug_json(root / f"round_{number}_selected.json", {"items": selected})
            row["runner_completed"] = await asyncio.wait_for(
                lite.simpleCallLLM(prompt, context, debug_info=row, max_turns=30), timeout=300,
            )
            after = await session.get_items()
            _save_debug_json(root / f"round_{number}_after.json", {"items": after})
            outer_requests = [request for request in requests
                              if request["round"] == number and request["agent"] == "Supervisor"]
            row["selected_history_in_model_input"] = bool(outer_requests) and _contains_in_order(
                outer_requests[0]["input"], selected,
            )
            current_files = _file_hashes(workspace)
            row["changed_files"] = sorted(
                path for path in before_files.keys() | current_files.keys()
                if before_files.get(path) != current_files.get(path)
            )
            assert row["runner_completed"], f"第{number}轮执行失败：{row['error']}"
            assert _file_hashes(test_dir) == fixed_test_hashes, "禁止修改固定测试"
            if number <= 2:
                assert not row["changed_files"], "前两轮只允许分析"
            assert set(row["changed_files"]) <= set(manifest["allowed_changed_paths"]), "修改范围超出约定"

        report["final_test"] = await fixed_tests()
        try:
            answer = json.loads(report["rounds"][-1]["final_output"])
        except (TypeError, ValueError):
            answer = {}
        model_tool_results = [
            item["output"] for request in requests for item in request["input"]
            if item.get("type") == "function_call_output"
        ]
        report["checks"] = {
            "four_rounds_completed": len(report["rounds"]) == 4,
            "history_selection_exercised": any(row["history_reduced"] for row in report["rounds"]),
            "history_within_budget": all(row["selected_history_chars"] <= history_char_limit
                                         for row in report["rounds"]),
            "priority_constraints_kept": all(row["priority_constraints_kept"] for row in report["rounds"]),
            "recent_turns_complete": all(row["recent_turns_complete"] for row in report["rounds"]),
            "selected_history_in_model_input": all(row["selected_history_in_model_input"]
                                                   for row in report["rounds"]),
            "tool_calls_and_outputs_paired": all(_tool_records_paired(request["input"]) for request in requests),
            "tool_results_within_budget": bool(events) and all(
                len(event["output"]) <= manifest["per_tool_result_char_limit"] for event in events
            ),
            "model_tool_results_within_budget": bool(model_tool_results) and all(
                isinstance(output, str) and len(output) <= manifest["per_tool_result_char_limit"]
                for output in model_tool_results
            ),
            "final_tests_passed": report["final_test"]["exit_code"] == manifest["expected_after_pytest_exit_code"],
            "final_facts_correct": isinstance(answer, dict) and all(
                answer.get(key) == expected[key] for key in (
                    "true_error_count", "error_line_numbers", "matching_task_count", "matching_task_ids",
                )
            ),
        }
        for filename in ("processing_large.log", "tasks_large.json"):
            matched = []
            for event in events:
                if event["tool"] != "readFile":
                    continue
                arguments = json.loads(event["arguments"])
                if str(arguments.get("path", "")).endswith("data/" + filename):
                    matched.append(json.loads(event["output"]))
            report["checks"][filename + "_truncation_and_source"] = any(
                body.get("truncated") is True and bool(body.get("excerpt"))
                and str(body.get("source", "")).endswith("data/" + filename)
                for body in matched
            )
        report["passed"] = all(report["checks"].values())
        assert report["passed"], "存在未通过的验收项，查看 report.json"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # 无论成功或失败都保存已收集数据，保留副本供人工调试。
        try:
            _save_debug_json(root / "model_requests.json", requests)
            _save_debug_json(root / "tool_starts.json", starts)
            _save_debug_json(root / "tool_results.json", events)
            _save_debug_json(root / "report.json", report)
            print(f"四轮验证产物：{root}")
        finally:
            session.close()
            if client is not None:
                await client.close()


_DAY6_WORKSPACE = Path("/media/xumaozhou/data/AI/local_agent_test/week5_day6_structured_output")


def _day6_append_event(path, record):
    """即时保存 SDK 事件；原始记录不放入 Agent 可读取的 task 目录。"""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class Day6StreamingModel(Model):
    """转发实际模型流，按单次请求保存原始增量，不把最终 JSON 人为切片。"""

    def __init__(self, model, name, raw_path):
        self.model, self.name, self.raw_path = model, name, raw_path
        self.responses = []

    async def get_response(self, *args, **kwargs):
        return await self.model.get_response(*args, **kwargs)

    def get_retry_advice(self, request):
        return self.model.get_retry_advice(request)

    async def stream_response(self, *args, **kwargs):
        record = {"deltas": [], "completed": False, "has_tool_calls": False, "has_usage": False}
        self.responses.append(record)
        request_id = len(self.responses)
        tools = kwargs.get("tools", args[3] if len(args) > 3 else [])
        _day6_append_event(self.raw_path, {
            "kind": "model_request", "agent": self.name, "request": request_id,
            "tools": [tool.name for tool in tools],
        })
        try:
            async for event in self.model.stream_response(*args, **kwargs):
                _day6_append_event(self.raw_path, {
                    "kind": "model_stream_event", "agent": self.name, "request": request_id,
                    "data": event.model_dump(mode="json"),
                })
                if event.type == "response.output_text.delta":
                    record["deltas"].append({"type": "text_delta", "data": event.delta})
                elif event.type == "response.completed":
                    record["completed"] = event.response.status == "completed"
                    record["has_usage"] = event.response.usage is not None
                    record["has_tool_calls"] = any(
                        item.type == "function_call" for item in event.response.output
                    )
                yield event
        except Exception as exc:
            _day6_append_event(self.raw_path, {
                "kind": "model_error", "agent": self.name, "request": request_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise


async def _day6_stream_attempt(agent, prompt, context, session, model, timeout):
    """执行一次真实 SDK 流；返回结果、最后请求的真实增量、工具通知、Schema 异常。

    context 是工具上下文，session 仅主任务使用；model 必须为 Day6StreamingModel。
    仅将已收到模型完成事件的最终回答 Schema 异常交给调用方判断是否纠错。
    网络、工具、取消、超时等执行异常直接传播，不重跑工具流程。
    """
    from agents import Runner, RunConfig
    from agents.exceptions import ModelBehaviorError

    result = Runner.run_streamed(
        agent, prompt, context=context, session=session, max_turns=30 if agent.tools else 1,
        run_config=RunConfig(tracing_disabled=True),
    )
    tool_events, tool_names = [], {}

    async def consume():
        async for event in result.stream_events():
            if event.type != "run_item_stream_event":
                continue
            if event.name == "tool_called":
                raw = event.item.raw_item
                tool_names[raw.call_id] = raw.name
                tool_events.append({"type": "tool_start", "tool_name": raw.name, "call_id": raw.call_id})
            elif event.name == "tool_output":
                raw = event.item.raw_item
                tool_events.append({
                    "type": "tool_end", "tool_name": tool_names.get(raw["call_id"]),
                    "call_id": raw["call_id"], "data": raw["output"],
                })
        if result.run_loop_exception:
            raise result.run_loop_exception

    schema_error = None
    try:
        await asyncio.wait_for(consume(), timeout=timeout)
    except ModelBehaviorError as exc:
        last = model.responses[-1] if model.responses else {}
        if not last.get("completed") or last.get("has_tool_calls"):
            raise
        schema_error = exc
    except BaseException:
        result.cancel()
        raise

    if not model.responses:
        raise ValueError("未收到模型响应")
    last = model.responses[-1]
    if not last["completed"] or not last["has_usage"]:
        raise ValueError("模型流未正常完成或未提供实际 usage")
    if not last["deltas"]:
        raise ValueError("未收到真实文本增量，不能以最终对象伪造流式片段")
    if schema_error is None and (not result.is_complete or result.interruptions):
        raise ValueError("Agent 流未完整结束或存在待处理审批")
    return result, last["deltas"], tool_events, schema_error


async def _day6_stream_with_retry(agent, prompt, context, session, model, adapter, make_repair, report, run_dir):
    """主工具任务只执行一次；最多用无工具 Agent 纠正一次最终回答。

    adapter 提供 validate_output/collect_stream；make_repair(raw, error) 返回
    (纠错 Agent, 纠错 Prompt, 流式模型)。report 原地记录次数、错误及实际累计 usage。
    返回 (四字段回答字典, 归一化事件列表)，失败则保留尝试记录并抛出异常。
    """
    tool_events, total_usage = [], {key: 0 for key in (
        "requests", "input_tokens", "output_tokens", "total_tokens",
    )}
    report.update(tool_workflow_runs=1, retry_count=0, output_errors=[])
    for attempt in range(2):
        result, deltas, notifications, schema_error = await _day6_stream_attempt(
            agent, prompt, context, session, model, timeout=600 if attempt == 0 else 120,
        )
        tool_events.extend(notifications)
        # 主任务 usage 已包含专家，不能再次累加专家结果；独立纠错再加一次。
        usage = result.context_wrapper.usage
        for key in total_usage:
            total_usage[key] += getattr(usage, key)
        report["usage"] = dict(total_usage)
        raw = "".join(event["data"] for event in deltas)
        try:
            output = adapter["validate_output"](raw)
        except ValueError as exc:
            report["output_errors"].append(str(exc))
            _save_debug_json(Path(run_dir) / f"attempt_{attempt + 1}_events.json", [
                *notifications, *deltas, {"type": "error", "message": str(exc)},
            ])
            if attempt == 1:
                raise ValueError("最终回答纠错一次后仍不合法") from exc
            agent, prompt, model = make_repair(raw, str(exc))
            if agent.tools or agent.mcp_servers:
                raise ValueError("纠错 Agent 必须没有工具和 MCP 服务")
            report["retry_count"] = 1
            # 使用全新的独立上下文，不带主任务 Session，不重复工具流程。
            context, session = None, None
            continue
        if schema_error is not None:
            # 文本本身合格时，SDK 异常并非输出格式错误，不能吞掉。
            raise schema_error
        if output != result.final_output.model_dump():
            raise ValueError("真实增量与 SDK 最终结构化结果不一致")
        events = [*tool_events, *deltas, {"type": "completed", "usage": dict(total_usage)}]
        collected = adapter["collect_stream"](iter(events))
        if collected != {"output": output, "usage": total_usage}:
            raise ValueError("流式适配结果与最终回答或实际 usage 不一致")
        _save_debug_json(Path(run_dir) / f"attempt_{attempt + 1}_events.json", [
            *notifications, *deltas, {"type": "completed", "usage": dict(total_usage)},
        ])
        return output, events


@pytest.mark.skipif(
    os.environ.get("RUN_DAY6_STREAM_TEST") != "1",
    reason="测试五真实流式验证需显式设置 RUN_DAY6_STREAM_TEST=1",
)
@pytest.mark.asyncio
async def test_day6_stream_real_llm():
    """测试五：真实流、至多一次无工具纠错、实际 usage 和独立业务验收。

    数据来自 Day6 的 task 模板，事件来自实际 DeepSeek SDK 流，不使用离线
    stream_cases.json 冒充真实事件。需要 Docker 和 agent-python:3.10 镜像。
    默认跳过；即使直接调用也检查开关。只有本测试注入场景提示，不修改通用提示词。
    """
    if os.environ.get("RUN_DAY6_STREAM_TEST") != "1":
        pytest.skip("请先显式启用测试五真实流式验证")
    assert os.environ.get("OPENAI_API_KEY"), "缺少现有 DeepSeek 配置 OPENAI_API_KEY"
    from agents import OpenAIResponsesModel
    from agentPrompts import SUPERVISOR_INSTRUCTIONS, PYTHON_EXPERT_INSTRUCTIONS, PYTHON_EXPERT_DESCRIPTION
    from baseModel import AgentInput, AgentOutput, build_prompt
    from tools import EvalContext

    case_dir = _DAY6_WORKSPACE
    manifest = json.loads((case_dir / "eval_tests/manifest.json").read_text(encoding="utf-8"))
    assert _file_hashes(case_dir / "task/workspace") == manifest["workspace_hashes"]
    assert _file_hashes(case_dir / "task/eval_tests") == manifest["test_hashes"]
    adapter = runpy.run_path(str(case_dir / "exercise_adapter.py"))
    logger_class = runpy.run_path(str(case_dir / "run_non_stream.py"))["ToolCallLogger"]
    run_id = "stream_" + uuid.uuid4().hex[:12]
    run_dir = case_dir / "runs" / run_id
    task_root = run_dir / "task"
    shutil.copytree(case_dir / "task", task_root)
    tool_log = logger_class(run_dir / "tool_calls.jsonl")
    session = SQLiteSession(run_id, str(run_dir / "session.db"))
    report = {
        "mode": "real_llm", "response_mode": "stream", "status": "failed", "error": None,
        "usage": None, "retry_count": 0, "tool_workflow_runs": 0,
    }
    started, client, lite = time.perf_counter(), None, None
    try:
        task = AgentInput(
            goal="修复 workspace/src/report.py 的筛选逻辑，并运行固定测试。",
            workspace="workspace",
            constraints=[
                "Python 3.10，只允许修改 workspace/src/report.py，不新增依赖。",
                "engineCode 精确等于 engine_code 且 status 精确等于 done。",
                "保持输入顺序、不修改输入记录，保持函数签名及报表字段。",
                "workspace/data 和 eval_tests 只读，只访问本次 task 授权目录。",
            ],
            verification="调用 runSafeTest，project_path=workspace，test_path=eval_tests，报告实际结果。",
        )
        prompt = build_prompt(task)
        _save_debug_json(run_dir / "task_input.json", task.model_dump())
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        with patch.dict(os.environ, {"THREAD_ID": run_id}):
            lite = agentLite.AgentLite(session=session)
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com")
        base_model = OpenAIResponsesModel(
            model=os.environ.get("DAY6_TEST_MODEL", "deepseek-flash"), openai_client=client,
        )
        directory = (
            f"\n工具授权根目录：{task_root}；项目相对目录 workspace；固定测试目录 eval_tests。"
            "工具参数优先使用相对此根目录的路径，changed_files 相对项目目录。\n"
        )
        expert = Agent(
            name="PythonExpert", model=base_model,
            instructions=PYTHON_EXPERT_INSTRUCTIONS + directory,
            tools=[tool_log.wrap(tool, "PythonExpert") for tool in lite._tools],
        )
        model = Day6StreamingModel(base_model, "Supervisor", run_dir / "sdk_events.jsonl")
        supervisor = Agent(
            name="Supervisor", model=model, output_type=AgentOutput,
            instructions=SUPERVISOR_INSTRUCTIONS + directory,
            tools=[tool_log.wrap(expert.as_tool(
                tool_name="PythonExpert", tool_description=PYTHON_EXPERT_DESCRIPTION,
                custom_output_extractor=agentLite._limited_expert_output,
                failure_error_function=None, max_turns=30,
            ), "Supervisor")],
        )

        def make_repair(raw, error):
            records = [json.loads(line) for line in tool_log.path.read_text(encoding="utf-8").splitlines()]
            actual_files = _file_hashes(task_root / "workspace")
            before = manifest["workspace_hashes"]
            facts = {
                "goal": task.goal, "constraints": task.constraints,
                "changed_files": sorted(name for name in before.keys() | actual_files.keys()
                                        if before.get(name) != actual_files.get(name)),
                "tool_results": [row for row in records if row["event"] == "tool_end"
                                 and row["tool"] in ("applyPatch", "runSafeTest")],
            }
            repair_model = Day6StreamingModel(base_model, "OutputRepair", run_dir / "sdk_events.jsonl")
            repair = Agent(
                name="OutputRepair", model=repair_model, output_type=AgentOutput, tools=[],
                instructions="只纠正最终回答的结构。依据提供的实际执行事实，不改变测试结果或虚构执行。"
                             "按程序 Schema 输出，不添加围栏；无法完成的事项如实保留。",
            )
            repair_prompt = json.dumps({"actual_facts": facts, "invalid_output": raw, "validation_error": error}, ensure_ascii=False)
            _save_debug_json(run_dir / "repair_input.json", json.loads(repair_prompt))
            return repair, repair_prompt, repair_model

        answer, events = await _day6_stream_with_retry(
            supervisor, prompt, EvalContext(workspace=task_root), session, model,
            adapter, make_repair, report, run_dir,
        )
        _save_debug_json(run_dir / "answer.json", answer)
        _save_debug_json(run_dir / "events.json", events)
        report["status"] = "completed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        _save_debug_json(run_dir / "events.json", [{"type": "error", "message": report["error"]}])
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        try:
            _save_debug_json(run_dir / "run_report.json", report)
            _save_debug_json(run_dir / "session_after.json", {"items": await session.get_items()})
            print(f"Day6 测试五产物：{run_dir}")
        finally:
            session.close()
            if lite is not None:
                await lite.close()
            if client is not None:
                await client.close()

    # 独立检查器读取自己的原始测试，不把参考答案交给 Agent。
    checked = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-B", str(case_dir / "check_real_run.py"), "--mode", "stream",
         "--repo", str(Path(__file__).resolve().parents[1]), "--run-dir", str(run_dir),
         "--adapter", str(case_dir / "exercise_adapter.py")],
        capture_output=True, text=True, timeout=60,
    )
    evaluation = json.loads(checked.stdout)
    _save_debug_json(run_dir / "evaluation.json", evaluation)
    assert checked.returncode == 0 and evaluation["passed"], evaluation


@pytest.mark.parametrize("scenario", [
    "valid", "repair", "still_invalid", "tool_error", "disconnect", "missing_usage", "no_delta",
])
@pytest.mark.asyncio
async def test_day6_stream_support_offline(tmp_path, monkeypatch, scenario):
    """真实 SDK + 预设模型流验证测试五辅助逻辑，不访问网络或启动 Docker。

    检查正常流、纠错、停止、工具错误、断流、用量缺失和无真实增量；主工具只执行一次。
    """
    from agents import function_tool, set_tracing_disabled
    from agents.tracing import setup as tracing_setup
    from agents.tracing.provider import DefaultTraceProvider
    from openai.types.responses import (
        Response, ResponseCompletedEvent, ResponseFunctionToolCall,
        ResponseOutputMessage, ResponseOutputText, ResponseTextDeltaEvent, ResponseUsage,
    )
    from baseModel import AgentOutput
    from tools import FatalToolError

    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    set_tracing_disabled(True)
    adapter = runpy.run_path(str(_DAY6_WORKSPACE / "exercise_adapter.py"))
    valid = AgentOutput(
        summary="离线流验证完成", changed_files=[], test_status="not_run", unfinished_work=[],
    ).model_dump_json()
    calls, repairs = [], []

    @function_tool(failure_error_function=None)
    async def localWork() -> str:
        calls.append("executed")
        if scenario == "tool_error":
            raise FatalToolError("离线模拟工具失败")
        return '{"success":true,"summary":"工具文本不能拼进最终回答"}'

    class OfflineStreamModel(Model):
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = 0

        async def get_response(self, *args, **kwargs):
            raise AssertionError("本测试必须调用真实 SDK 流式路径")

        async def stream_response(self, *args, **kwargs):
            self.calls += 1
            payload = next(self.responses)
            if isinstance(payload, Exception):
                raise payload
            output = []
            if payload == "tool":
                # Supervisor 的中间说明也不是最终回答，不应与最后一轮文本混合。
                yield ResponseTextDeltaEvent(
                    type="response.output_text.delta", content_index=0, output_index=0,
                    item_id="planning", delta="准备调用工具", sequence_number=0, logprobs=[],
                )
                output = [ResponseOutputMessage(
                    id="planning", role="assistant", type="message", status="completed",
                    content=[ResponseOutputText(type="output_text", text="准备调用工具", annotations=[])],
                ), ResponseFunctionToolCall(
                    id="call_1", call_id="call_1", type="function_call", name="localWork", arguments="{}",
                )]
            else:
                # 这些是明确标注的离线模拟事件，真实入口不会切分最终回答来伪造流。
                for index, chunk in enumerate((payload[:7], payload[7:])):
                    if scenario == "no_delta":
                        continue
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta", content_index=0, output_index=0,
                        item_id="message_1", delta=chunk, sequence_number=index, logprobs=[],
                    )
                output = [ResponseOutputMessage(
                    id="message_1", role="assistant", type="message", status="completed",
                    content=[ResponseOutputText(type="output_text", text=payload, annotations=[])],
                )]
            response = Response.model_construct(
                id=f"response_{self.calls}", object="response", created_at=0, model="offline",
                status="completed", output=output,
                usage=None if scenario == "missing_usage" and payload != "tool" else ResponseUsage(
                    input_tokens=10, output_tokens=5, total_tokens=15,
                    input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                ),
            )
            yield ResponseCompletedEvent(type="response.completed", response=response, sequence_number=3)

    raw_model = OfflineStreamModel([
        "tool", ConnectionError("离线模拟断流") if scenario == "disconnect"
        else valid if scenario in ("valid", "missing_usage", "no_delta") else '{"summary":',
    ])
    model = Day6StreamingModel(raw_model, "Supervisor", tmp_path / "sdk_events.jsonl")
    agent = Agent(name="Supervisor", model=model, output_type=AgentOutput, tools=[localWork])
    repair_model = None

    def make_repair(raw, error):
        nonlocal repair_model
        repairs.append({"raw": raw, "error": error})
        repair_model = OfflineStreamModel(["[]" if scenario == "still_invalid" else valid])
        wrapped = Day6StreamingModel(repair_model, "OutputRepair", tmp_path / "sdk_events.jsonl")
        repair = Agent(name="OutputRepair", model=wrapped, output_type=AgentOutput, tools=[])
        return repair, json.dumps(repairs[-1]), wrapped

    report = {}
    if scenario in ("still_invalid", "tool_error", "disconnect", "missing_usage", "no_delta"):
        error_type = {"tool_error": FatalToolError, "disconnect": ConnectionError}.get(scenario, ValueError)
        with pytest.raises(error_type):
            await _day6_stream_with_retry(agent, "离线任务", None, None, model, adapter, make_repair, report, tmp_path)
    else:
        output, events = await _day6_stream_with_retry(
            agent, "离线任务", None, None, model, adapter, make_repair, report, tmp_path,
        )
        assert output == json.loads(valid)
        assert adapter["collect_stream"](iter(events)) == {"output": output, "usage": report["usage"]}
        assert report["usage"] == {
            "requests": 2 if scenario == "valid" else 3,
            "input_tokens": 20 if scenario == "valid" else 30,
            "output_tokens": 10 if scenario == "valid" else 15,
            "total_tokens": 30 if scenario == "valid" else 45,
        }
    assert calls == ["executed"]
    expected_repairs = int(scenario in ("repair", "still_invalid"))
    assert len(repairs) == expected_repairs
    assert report["retry_count"] == expected_repairs
    assert report["tool_workflow_runs"] == 1
    if repair_model:
        assert repair_model.calls == 1
        raw_records = [json.loads(line) for line in (tmp_path / "sdk_events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(row["tools"] == [] for row in raw_records
                   if row["kind"] == "model_request" and row["agent"] == "OutputRepair")


@pytest.mark.parametrize("problem,failed_check", [
    ("old_history", "first_model_input_is_compacted_history_plus_one_prompt"),
    ("split_prompt", "first_model_input_is_compacted_history_plus_one_prompt"),
    ("discard_marker", "no_irrelevant_old_history_in_model_inputs"),
    ("unpaired_tool", "model_tool_records_paired"),
])
def test_day5_support_rejects_invalid_model_inputs(problem, failed_check):
    """只验证输入检查器，不执行真实摘要或任务续接函数。"""
    compacted = [{"role": "user", "content": "历史摘要"}]
    prompt = "继续任务\n完成后报告结果"
    requests = [{"agent": "Supervisor", "input": [
        *compacted, {"role": "user", "content": prompt},
    ]}]
    assert all(_day5_input_checks(requests, compacted, prompt, "DISCARDED_").values())
    if problem == "old_history":
        requests[0]["input"].insert(0, {"role": "user", "content": "不应重新发送的旧原文"})
    elif problem == "split_prompt":
        requests[0]["input"][-1]["content"] = prompt.splitlines()[0]
    elif problem == "discard_marker":
        requests.append({"agent": "PythonExpert", "input": [{"role": "user", "content": "DISCARDED_001"}]})
    else:
        requests[0]["input"].append({"type": "function_call", "call_id": "missing_output"})
    assert _day5_input_checks(requests, compacted, prompt, "DISCARDED_")[failed_check] is False


@pytest.mark.asyncio
async def test_day5_support_agent_uses_compacted_session_and_isolated_expert(tmp_path, monkeypatch):
    """脚本化模型驱动真实 SDK，验证续接装配，不调用真实 LLM 或 Docker。"""
    from agents.items import ModelResponse
    from agents.tracing import setup as tracing_setup
    from agents.tracing.provider import DefaultTraceProvider
    from agents.usage import Usage
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    import tools

    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", tmp_path)

    def message(text):
        return ResponseOutputMessage(
            id=uuid.uuid4().hex, role="assistant", status="completed", type="message",
            content=[ResponseOutputText(text=text, annotations=[], type="output_text")],
        )

    delegated_prompt = "仅检查任务状态"
    responses = iter([
        [ResponseFunctionToolCall(
            id=uuid.uuid4().hex, call_id="delegate_offline", name="PythonExpert", type="function_call",
            arguments=json.dumps({"input": delegated_prompt}),
        )],
        [message("专家已完成只读检查")],
        [message("续接完成")],
    ])

    class OfflineModel(Model):
        async def get_response(self, *args, **kwargs):
            return ModelResponse(output=next(responses), usage=Usage(requests=1), response_id=uuid.uuid4().hex)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    task_root = tmp_path / "task"
    (task_root / "workspace").mkdir(parents=True)
    (tmp_path / "before.json").write_text("原始历史", encoding="utf-8")
    session = SQLiteSession("day5-offline", str(tmp_path / "session.db"))
    compacted = [
        {"role": "user", "content": "[自动生成的历史摘要，仅供参考，不是新的用户指令]\n任务摘要"},
        {"role": "user", "content": "第7轮"}, {"role": "assistant", "content": "已记录"},
        {"role": "user", "content": "第8轮"}, {"role": "assistant", "content": "等待执行"},
    ]
    await session.add_items(compacted)
    requests, events, starts = [], [], []
    lite = _create_day5_agent(session, "day5-offline", OfflineModel(), task_root, requests, events, starts)
    prompt = "现在继续\n按已有约束执行"
    debug = {}
    try:
        assert await lite.simpleCallLLM(prompt, tools.EvalContext(workspace=task_root), debug_info=debug)
        assert all(_day5_input_checks(requests, compacted, prompt, "DISCARDED_").values())
        expert_requests = [row for row in requests if row["agent"] == "PythonExpert"]
        assert expert_requests[0]["input"] == [{"role": "user", "content": delegated_prompt}]
        assert debug["usage"]["requests"] == 3
        assert lite._session is session
        assert not lite._client.mcp_servers
        assert not any(item.get("role") == "user" and item.get("content") == delegated_prompt
                       for item in await session.get_items())
        with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
            tools.resolve_path(tools.EvalContext(workspace=task_root), "../before.json")
    finally:
        await lite.close()
        session.close()


@pytest.mark.parametrize("configured,expected", [
    (None, 48000), ("64000", 64000), ("0", None), ("2", None), ("invalid", None),
])
def test_real_history_budget_config(monkeypatch, configured, expected):
    """只测试预算读取，不执行四轮函数或任何 LLM 请求。"""
    if configured is None:
        monkeypatch.delenv("CONTEXT_TEST_HISTORY_CHAR_LIMIT", raising=False)
    else:
        monkeypatch.setenv("CONTEXT_TEST_HISTORY_CHAR_LIMIT", configured)
    if expected is None:
        with pytest.raises(ValueError):
            _real_history_char_limit()
    else:
        assert _real_history_char_limit() == expected


def test_tool_start_keeps_arguments_when_path_check_fails(tmp_path, monkeypatch):
    """实际触发路径校验失败，验证失败前的参数仍被记录；不调用 LLM。"""
    import tools
    root = tmp_path / "authorized"
    root.mkdir()
    monkeypatch.setattr(tools, "_SAFE_WORKSPACE", root)
    payload = json.dumps({"path": "../outside.txt"})
    context = ToolContext(
        context=tools.EvalContext(workspace=root), tool_name="readFile",
        tool_call_id="failed_call", tool_arguments=payload,
    )
    starts, events = [], []
    hooks = RecordingHooks(events, starts)
    hooks.round_number = 2
    agent, tool = SimpleNamespace(name="PythonExpert"), SimpleNamespace(name="readFile")

    async def run():
        await hooks.on_tool_start(context, agent, tool)
        tools.resolve_path(context.context, "../outside.txt")
        await hooks.on_tool_end(context, agent, tool, "不应执行到这里")

    with pytest.raises(tools.FatalToolError, match="PERMISSION_DENIED"):
        asyncio.run(run())
    assert starts == [{
        "round": 2, "agent": "PythonExpert", "tool": "readFile",
        "call_id": "failed_call", "arguments": payload,
    }]
    assert events == []


def test_successful_tool_has_separate_start_and_result_records():
    payload = '{"path":"workspace/src/pipeline.py"}'
    context = ToolContext(
        context=None, tool_name="readFile", tool_call_id="successful_call", tool_arguments=payload,
    )
    starts, events = [], []
    hooks = RecordingHooks(events, starts)
    agent, tool = SimpleNamespace(name="PythonExpert"), SimpleNamespace(name="readFile")

    async def run():
        await hooks.on_tool_start(context, agent, tool)
        await hooks.on_tool_end(context, agent, tool, '{"excerpt":"代码内容"}')

    asyncio.run(run())
    assert starts[0]["call_id"] == events[0]["call_id"] == "successful_call"
    assert "output" not in starts[0]
    assert events[0]["output"] == '{"excerpt":"代码内容"}'


@pytest.mark.skipif(
    os.environ.get("RUN_DAY5_REAL_SUMMARY") != "1",
    reason="需要手动启用真实 LLM 摘要测试",
)
@pytest.mark.asyncio
async def test_day5_real_summary():
    from contextCompact import summarize_history

    case_dir = Path(
        "/media/xumaozhou/data/AI/local_agent_test/week5_day5_compaction"
    )
    run_id = "summary_" + uuid.uuid4().hex[:12]
    run_dir = case_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    def save_json(filename, data):
        (run_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 每次使用独立 Session，避免混入上次实验的数据。
    session = SQLiteSession(run_id, str(run_dir / "session.db"))
    history = json.loads(
        (case_dir / "session_history.json").read_text(encoding="utf-8")
    )["items"]
    await session.add_items(history)

    before = await session.get_items()
    save_json("before.json", {"items": before})

    async def summarize_and_record(old_history):
        # 保存真正交给摘要 Agent 的旧历史，方便确认压缩范围。
        save_json("summary_input.json", {"items": json.loads(old_history)})

        summary = await summarize_history(old_history)
        (run_dir / "summary.txt").write_text(summary.get("summary", ""), encoding="utf-8")
        return summary

    result = await compact_session(
        session,
        summarize_fn=summarize_and_record,
        threshold=8000,
        reserved_tokens=2500,
        keep_recent_turns=2,
    )

    after = await session.get_items()
    save_json("compacted_context.json", {"items": after})
    save_json("compaction_result.json", result)
    print(f"本次测试产物：{run_dir}")

    assert result["compacted"], result
    assert result["after"] < result["before"]



def _create_day5_agent(session, run_id, model, task_root, requests, events, starts):
    """复用 AgentLite 和通用提示词，只在测试中注入本次目录与记录器。"""
    from agentLite import AgentLite, _limited_expert_output
    from agentPrompts import (
        SUPERVISOR_INSTRUCTIONS, PYTHON_EXPERT_INSTRUCTIONS, CPP_EXPERT_INSTRUCTIONS,
        PYTHON_EXPERT_DESCRIPTION, CPP_EXPERT_DESCRIPTION,
    )

    # task_root 只包含工作副本和固定测试，原历史、答案和验收配置放在其外。
    directory_instructions = (
        f"\n本次工具授权根目录：{task_root}\n"
        "项目相对目录：workspace；源码入口：workspace/src/report.py；"
        "数据文件：workspace/data/tasks.json；固定测试目录：eval_tests。\n"
        "工具参数优先使用上述相对路径；changed_files 使用相对于项目目录的路径。\n"
    )
    with patch.dict(os.environ, {"THREAD_ID": run_id}):
        lite = AgentLite(session=session)
    models = {name: RecordingModel(model, name, requests)
              for name in ("Supervisor", "PythonExpert", "CppExpert")}
    hooks = RecordingHooks(events, starts)
    lite._python_agent = Agent(
        name="PythonExpert", instructions=PYTHON_EXPERT_INSTRUCTIONS + directory_instructions,
        model=models["PythonExpert"], tools=lite._tools, hooks=hooks,
    )
    lite._cpp_agent = Agent(
        name="CppExpert", instructions=CPP_EXPERT_INSTRUCTIONS + directory_instructions,
        model=models["CppExpert"], tools=lite._tools, hooks=hooks,
    )
    lite._client = Agent(
        name="Supervisor", instructions=SUPERVISOR_INSTRUCTIONS + directory_instructions,
        model=models["Supervisor"], hooks=hooks,
        tools=[
            expert.as_tool(
                tool_name=expert.name, tool_description=description,
                custom_output_extractor=_limited_expert_output, failure_error_function=None,
                max_turns=30,
                # 专家不共用 Supervisor 的 Session，避免委派输入混入用户历史。
            )
            for expert, description in (
                (lite._python_agent, PYTHON_EXPERT_DESCRIPTION),
                (lite._cpp_agent, CPP_EXPERT_DESCRIPTION),
            )
        ],
    )
    return lite


def _day5_input_checks(requests, compacted, prompt, discard_marker):
    """核对 SDK 模型入口，避免只检查 Session 文件却重新发送了完整旧历史。"""
    supervisor_requests = [row for row in requests if row["agent"] == "Supervisor"]
    expected_input = [*compacted, {"role": "user", "content": prompt}]
    return {
        "first_model_input_is_compacted_history_plus_one_prompt": bool(supervisor_requests)
        and supervisor_requests[0]["input"] == expected_input,
        "no_irrelevant_old_history_in_model_inputs": bool(requests) and all(
            discard_marker not in json.dumps(row["input"], ensure_ascii=False) for row in requests
        ),
        "model_tool_records_paired": bool(requests) and all(
            _tool_records_paired(row["input"]) for row in requests
        ),
    }


@pytest.mark.skipif(
    os.environ.get("RUN_DAY5_FINAL_TEST") != "1",
    reason="真实摘要与任务续接验收需显式设置 RUN_DAY5_FINAL_TEST=1",
)
@pytest.mark.asyncio
async def test_day5_final_test():
    """Day5 真实续接：摘要压缩、一次完整续接输入、独立产物验收。

    默认跳过；不依赖 Day5 第二步的旧产物，不使用 TaskState 或 Memory 补回历史。
    每次创建独立副本，调用现有 DeepSeek 配置与本地工具，不启动 MCP 服务。
    需要 Docker、agent-python:3.10 镜像和项目虚拟环境中的 pytest。
    """
    if os.environ.get("RUN_DAY5_FINAL_TEST") != "1":
        pytest.skip("请先显式启用 Day5 真实任务续接验收")
    assert os.environ.get("OPENAI_API_KEY"), "缺少 OPENAI_API_KEY"
    from tools import EvalContext, runSafeTest

    case_dir = Path("/media/xumaozhou/data/AI/local_agent_test/week5_day5_compaction")
    manifest = json.loads((case_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    context_checker = runpy.run_path(str(case_dir / "check_compaction.py"))["check"]
    run_id = "final_" + uuid.uuid4().hex[:12]
    run_dir = case_dir / "runs" / run_id
    # 将可供工具访问的目录与调试产物分开，不能补读原历史或参考答案。
    task_root = run_dir / "task"
    workspace = task_root / "workspace"
    test_dir = task_root / "eval_tests"
    test_dir.mkdir(parents=True)
    shutil.copytree(case_dir / "workspace", workspace)
    shutil.copy2(case_dir / "eval_tests/test_report.py", test_dir / "test_report.py")
    baseline_files = _file_hashes(workspace)
    fixed_test_hashes = _file_hashes(test_dir)
    session = SQLiteSession(run_id, str(run_dir / "session.db"))
    report = {"passed": False, "run_id": run_id, "checks": {}}
    requests, events, starts = [], [], []
    client, lite = None, None

    try:
        assert baseline_files == manifest["workspace_hashes"], \
            "练习 workspace 已偏离初始版本，请先恢复初始文件"
        # 在调用 LLM 前检查 Docker 和基线，环境问题不消耗真实摘要请求。
        payload = json.dumps({"project_path": "workspace", "test_path": "eval_tests"})
        baseline = await runSafeTest.on_invoke_tool(
            ToolContext(
                context=EvalContext(workspace=task_root), tool_name="runSafeTest",
                tool_call_id=uuid.uuid4().hex, tool_arguments=payload,
            ), payload,
        )
        report["baseline_test"] = json.loads(baseline)
        assert report["baseline_test"]["exit_code"] == manifest["expected_baseline_pytest_exit_code"], \
            "基线结果不符合预期，请检查 Docker、依赖或初始文件"

        history = json.loads((case_dir / "session_history.json").read_text(encoding="utf-8"))["items"]
        await session.add_items(history)
        before = await session.get_items()
        recent = copy.deepcopy(_recent_two_turns(before))
        summary_inputs = []
        _save_debug_json(run_dir / "before.json", {"items": before})

        async def summarize_and_record(old_history):
            summary_inputs.append(json.loads(old_history))
            _save_debug_json(run_dir / "summary_input.json", {"items": summary_inputs[-1]})
            # 当前接口返回 {summary, tokens}，直接交回 compact_session。
            summary = await summarize_history(old_history)
            (run_dir / "summary.txt").write_text(summary["summary"], encoding="utf-8")
            return summary

        started = time.perf_counter()
        report["compaction"] = await asyncio.wait_for(compact_session(
            session, summarize_fn=summarize_and_record,
            threshold=manifest["threshold"], reserved_tokens=manifest["reserved_tokens"],
            keep_recent_turns=manifest["keep_recent_turns"],
        ), timeout=180)
        report["summary_seconds"] = time.perf_counter() - started
        report["summary_total_tokens"] = report["compaction"].get("summary_tokens", 0)
        compacted = await session.get_items()
        _save_debug_json(run_dir / "compaction_result.json", report["compaction"])
        _save_debug_json(run_dir / "compacted_context.json", {"items": compacted})
        assert report["compaction"]["compacted"], report["compaction"]
        report["context_check"] = context_checker(compacted, before, manifest)
        _save_debug_json(run_dir / "context_check.json", report["context_check"])
        report["checks"]["summary_receives_only_old_history"] = summary_inputs == [before[:len(before) - len(recent)]]
        report["checks"]["compacted_context_passes"] = report["context_check"]["passed"]
        assert all(report["checks"].values()), "压缩结果未通过检查，查看 context_check.json 和 summary.txt"

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.deepseek.com")
        model = OpenAIChatCompletionsModel(
            model=os.environ.get("CONTEXT_TEST_MODEL", "deepseek-flash"), openai_client=client,
        )
        lite = _create_day5_agent(session, run_id, model, task_root, requests, events, starts)
        # continuation_prompt.txt 是一条多行消息，不能按行拆成多轮请求。
        prompt = (case_dir / "continuation_prompt.txt").read_text(encoding="utf-8").strip()
        report["continuation"] = {"prompt": prompt}
        started = time.perf_counter()
        completed = await asyncio.wait_for(lite.simpleCallLLM(
            prompt, EvalContext(workspace=task_root), debug_info=report["continuation"], max_turns=30,
        ), timeout=600)
        report["continuation_seconds"] = time.perf_counter() - started
        report["checks"].update(_day5_input_checks(requests, compacted, prompt, manifest["discard_marker"]))
        report["checks"]["runner_completed"] = completed
        assert completed, report["continuation"]["error"]
        assert all(report["checks"].values()), "模型输入不符合压缩续接要求，查看 model_requests.json"

        answer = json.loads(report["continuation"]["final_output"])
        _save_debug_json(run_dir / "answer.json", answer)
        # 调用已有检查器独立运行固定 pytest、比较实际文件哈希与最终 JSON。
        evaluation = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-B", str(case_dir / "check_continuation.py"),
             str(run_dir / "answer.json"), "--workspace", str(workspace)],
            capture_output=True, text=True, timeout=60,
        )
        report["evaluation_process"] = {
            "returncode": evaluation.returncode, "stdout": evaluation.stdout, "stderr": evaluation.stderr,
        }
        report["continuation_check"] = json.loads(evaluation.stdout)
        _save_debug_json(run_dir / "continuation_check.json", report["continuation_check"])
        report["checks"]["independent_continuation_check_passed"] = (
            evaluation.returncode == 0 and report["continuation_check"]["passed"]
        )
        report["checks"]["fixed_tests_unchanged"] = _file_hashes(test_dir) == fixed_test_hashes
        report["checks"]["original_workspace_unchanged"] = _file_hashes(case_dir / "workspace") == baseline_files
        report["passed"] = all(report["checks"].values())
        assert report["passed"], "续接产物未通过独立验收，查看 report.json"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # 无论在哪个阶段失败都保留已有证据；原始历史不放入工具授权目录。
        try:
            _save_debug_json(run_dir / "session_after.json", {"items": await session.get_items()})
            _save_debug_json(run_dir / "model_requests.json", requests)
            _save_debug_json(run_dir / "tool_starts.json", starts)
            _save_debug_json(run_dir / "tool_results.json", events)
            _save_debug_json(run_dir / "report.json", report)
            print(f"Day5 真实续接验收产物：{run_dir}")
        finally:
            session.close()
            if lite is not None:
                await lite.close()
            if client is not None:
                await client.close()
