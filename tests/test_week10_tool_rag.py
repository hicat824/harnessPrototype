"""Week10 验收。离线路径和 Day3 真实模型路径通过各自的开关启用。"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
from agents import SQLiteSession
from agents.testing.model import ScriptedModel, assistant_message, function_call
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

from week6 import BusinessAPIClient, BusinessResult, BusinessService, create_mock_http_client
from week10_tool_rag import (
    Day1Request, KnowledgeEvidence, build_day2_context, prepare_knowledge_query, run_day1,
    build_day3_agent, query_task_tool, run_day3_agent, search_knowledge_tool,
)


DATA_ROOT = Path("/media/xumaozhou/data/AI/local_agent_test")
DATASET = Path(os.environ.get("WEEK10_DATASET", DATA_ROOT / "week10_tool_rag"))
KNOWLEDGE_ROOT = Path(os.environ.get("WEEK9_RAG_DATASET", DATA_ROOT / "week9_rag_quality"))
OUTPUT = Path(os.environ.get("WEEK10_DAY1_OUTPUT", DATASET / "runs/day1"))
DAY1 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY1") != "1",
                         reason="完整路由验收需设置 RUN_WEEK10_DAY1=1")
DAY2_OUTPUT = Path(os.environ.get("WEEK10_DAY2_OUTPUT", DATASET / "runs/day2"))
DAY2 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY2") != "1",
                         reason="完整 Context 验收需设置 RUN_WEEK10_DAY2=1")
DAY3_OUTPUT = Path(os.environ.get("WEEK10_DAY3_OUTPUT", DATASET / "runs/day3"))
DAY3 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY3") != "1",
                         reason="Day3 离线验收需设置 RUN_WEEK10_DAY3=1")


@pytest.fixture(scope="module")
def scenarios():
    path = DATASET / "mock_scenarios.json"
    assert path.is_file(), f"找不到 Day1 数据，请检查 WEEK10_DATASET：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence(scenarios):
    config = scenarios["knowledge"]
    path = KNOWLEDGE_ROOT / config["source"]
    line_number = config["line"]
    text = path.read_text(encoding="utf-8").splitlines()[line_number - 1]
    return KnowledgeEvidence(config["doc_id"], f"{config['source']}#L{line_number}", text)


def test_day1_mock_sources_match_fixtures(scenarios, evidence):
    """直接核对 Week6 Mock 和 Week9 证据原文，独立于学生代码。"""
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        for probe in scenarios["task_probes"]:
            result = service.query_task(probe["task_id"])
            assert result.status == probe["status"]
            assert (result.data or {}).get("state") == probe["state"]
            assert result.error_code == probe["error_code"]
    assert all(topic in evidence.text for topic in scenarios["knowledge"]["topics"])


@DAY1
@pytest.mark.parametrize("state", ["SUCCESS", "FAILED", None])
def test_day1_combined_query_uses_the_observed_state(state):
    """改变上游返回值，检查组合路由是否跟随实际状态，以及无状态时是否停下。"""
    request = Day1Request("任务 12345 还是 RUNNING 吗？解释查到的状态。", True, True, "12345")
    query = prepare_knowledge_query(request, "tool_rag", state)
    if state is None:
        assert query is None
    else:
        assert isinstance(query, str) and state in query and "RUNNING" not in query


def _grade_case(case, service, knowledge, topics):
    # Agent 只接收 input。参考答案在本轮执行完成后用于评分。
    request = Day1Request(**case["input"])
    calls = []

    def query_task(task_id):
        calls.append({"tool": "query_task", "task_id": task_id})
        return service.query_task(task_id)

    def search_knowledge(query):
        calls.append({"tool": "search_knowledge", "query": query})
        return (knowledge,) if any(topic in query for topic in topics) else ()

    result = run_day1(request, query_task=query_task, search_knowledge=search_knowledge)
    expected = case["expected"]
    queries = [call["query"] for call in calls if call["tool"] == "search_knowledge"]
    expected_task_ids = [request.task_id] if "query_task" in expected["tool_order"] else []
    query_text = "\n".join(queries)
    checks = {
        "route": result.route == expected["route"],
        "outcome": (result.status, result.reason) == (expected["status"], expected["reason"]),
        "task_fact": (result.task_state, result.task_source) == (
            expected["task_state"], expected["task_source"],
        ),
        "knowledge_sources": [item.doc_id for item in result.evidence] == expected["doc_ids"],
        "tool_order": [call["tool"] for call in calls] == expected["tool_order"],
        "task_arguments": [call["task_id"] for call in calls if call["tool"] == "query_task"]
        == expected_task_ids,
        "knowledge_arguments": (
            (expected["route"] != "rag_only" or queries == [request.question])
            and all(term in query_text for term in expected.get("query_contains", []))
            and all(term not in query_text for term in expected.get("query_excludes", []))
        ),
    }
    return {"case_id": case["case_id"], "input": asdict(request), "result": asdict(result),
            "calls": calls, "checks": checks, "passed": all(checks.values())}


@DAY1
def test_day1_routes_and_stop_conditions(scenarios, evidence):
    """运行三个正常场景和两个停止场景，报告实际工具参数与调用顺序。"""
    data = json.loads((DATASET / "eval_cases.json").read_text(encoding="utf-8"))
    cases = [case for case in data["cases"] if case["day"] == 1]
    assert len(cases) == len({case["case_id"] for case in cases}) == 5
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        rows = [_grade_case(case, service, evidence, scenarios["knowledge"]["topics"]) for case in cases]

    failed = {row["case_id"]: [key for key, passed in row["checks"].items() if not passed]
              for row in rows if not row["passed"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "routing_report.json"
    path.write_text(json.dumps({"passed": not failed, "cases": rows}, ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")
    print(f"Day1 路由验收报告：{path}")
    assert not failed, failed


@pytest.fixture(scope="module")
def day2_cases():
    path = DATASET / "day2_context_cases.json"
    assert path.is_file(), f"找不到 Day2 样例，请检查 WEEK10_DATASET：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def day2_evidence(day2_cases):
    """两条 Week9 原文夹住一条偏长的合成片段，检验跳过中间项后继续选取。"""
    short_chunks = []
    for entry in day2_cases["knowledge"]:
        line = entry["line"]
        source = entry["source"]
        text = (KNOWLEDGE_ROOT / source).read_text(encoding="utf-8").splitlines()[line - 1]
        short_chunks.append(KnowledgeEvidence(entry["doc_id"], f"{source}#L{line}", text))
    stress = day2_cases["budget_stress_chunk"]
    long_chunk = KnowledgeEvidence(stress["doc_id"], "day2_context_cases.json#budget_stress_chunk",
                                   stress["fragment"] * stress["repeat"])
    return (short_chunks[0], long_chunk, short_chunks[1])


@pytest.fixture(scope="module")
def day2_task_results(scenarios, day2_cases):
    """经 Week6 Mock 服务查一次可见任务和一次不存在的任务。"""
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        return (service.query_task(day2_cases["task_id"]),
                service.query_task(day2_cases["missing_task_id"]))


def test_day2_fixture_sources(day2_cases, day2_evidence, day2_task_results):
    """检查样例来源以及长短差异，无需先实现填空。"""
    success, missing = day2_task_results
    assert success.status == "success" and success.data["state"] == "RUNNING"
    assert missing.status == "error" and missing.error_code == "NOT_FOUND"
    first, long_chunk, last = day2_evidence
    assert "RUNNING" in first.text and "实时 API" in last.text
    assert len(long_chunk.text) > day2_cases["budgets"]["tight"]


@DAY2
def test_day2_task_summary_keeps_verified_fields(day2_task_results):
    """保留业务结果和上游截断标记，从成功结果投影可引用的事实。"""
    success, _ = day2_task_results
    upstream = success.model_copy(update={"truncated": True})
    before = upstream.model_dump()
    context = build_day2_context(upstream, ())
    payload = json.loads(context.text)
    assert payload == {
        "task_query": {
            "status": "success", "error_code": None, "source_truncated": True,
            "fact": {
                "evidence_id": "T1", "task_id": "12345", "state": "RUNNING",
                "source": upstream.source, "fetched_at": upstream.fetched_at,
            },
        },
        "knowledge": [], "knowledge_truncated": False,
    }
    assert context.omitted_ids == () and upstream.model_dump() == before


@DAY2
def test_day2_missing_or_invalid_task_fact_has_no_citation(day2_task_results, day2_evidence):
    """没查任务、查不到任务和缺有效字段时，仍可保留各自允许的证据。"""
    _, missing = day2_task_results
    evidence = day2_evidence[:1]
    rag_only = json.loads(build_day2_context(None, evidence).text)
    assert rag_only["task_query"] is None
    assert [row["evidence_id"] for row in rag_only["knowledge"]] == ["K1"]

    failed = json.loads(build_day2_context(missing, ()).text)
    assert failed["task_query"] == {
        "status": "error", "error_code": "NOT_FOUND", "source_truncated": False, "fact": None,
    }

    for data in (
        {"state": "RUNNING"},
        {"task_id": "12345", "state": " "},
        {"task_id": None, "state": "RUNNING"},
        {"task_id": "12345", "state": 1},
    ):
        malformed = BusinessResult(status="success", data=data)
        payload = json.loads(build_day2_context(malformed, ()).text)
        assert payload["task_query"]["fact"] is None


@DAY2
def test_day2_packs_complete_evidence_within_budget(day2_cases, day2_evidence, day2_task_results):
    """同时检查完整、限长、恰好放下和最小结构超限，输出可复核的预算报告。"""
    success, _ = day2_task_results
    limits = day2_cases["budgets"]
    full = build_day2_context(success, day2_evidence, max_chars=limits["full"])
    tight = build_day2_context(success, day2_evidence, max_chars=limits["tight"])
    exact = build_day2_context(success, day2_evidence, max_chars=len(full.text))
    baseline = build_day2_context(success, ())
    with pytest.raises(ValueError, match="预算"):
        build_day2_context(success, (), max_chars=len(baseline.text) - 1)

    expected_chunks = [
        {"evidence_id": f"K{index}", **asdict(item)}
        for index, item in enumerate(day2_evidence, start=1)
    ]
    rows = []
    for name, context, expected, omitted in (
        ("full", full, expected_chunks, ()),
        ("tight", tight, [expected_chunks[0], expected_chunks[2]], ("K2",)),
        ("exact", exact, expected_chunks, ()),
    ):
        payload = json.loads(context.text)
        assert len(context.text) <= context.max_chars
        assert payload["task_query"] == json.loads(baseline.text)["task_query"]
        assert payload["knowledge"] == expected
        assert payload["knowledge_truncated"] is bool(omitted) and context.omitted_ids == omitted
        rows.append({"case": name, "max_chars": context.max_chars, "used_chars": len(context.text),
                     "omitted_ids": omitted, "context": payload})
    assert len(exact.text) == exact.max_chars

    DAY2_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DAY2_OUTPUT / "context_report.json"
    path.write_text(json.dumps({"passed": True, "cases": rows}, ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")
    print(f"Day2 Context 验收报告：{path}")


def test_day3_tool_schemas():
    """SDK 只把业务参数暴露给模型，本地依赖通过 Context 注入。"""
    for tool, name, parameter in (
        (query_task_tool, "query_task", "task_id"),
        (search_knowledge_tool, "search_knowledge", "query"),
    ):
        assert tool.name == name
        assert set(tool.params_json_schema["properties"]) == {parameter}
        assert tool.params_json_schema["properties"][parameter]["type"] == "string"
        assert tool.params_json_schema["required"] == [parameter]


def _day3_last_tool_context(call):
    return json.loads([item["output"] for item in call.input
                       if item.get("type") == "function_call_output"][-1])


@pytest.fixture
def day3_local_tracing(monkeypatch):
    """复用 Week6 的离线追踪设置，测试期间使用没有远程导出器的 provider。"""
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())


def _day3_script(case_input):
    """模拟模型发起调用，组合路径从 SDK 真正送回的工具结果获取状态。"""
    needs_state, needs_rule = case_input["needs_current_state"], case_input["needs_rule"]
    task_id = case_input["task_id"]
    if needs_state and task_id is None:
        return ScriptedModel([[assistant_message("请提供任务编号。")]])

    def respond(call):
        context = _day3_last_tool_context(call)
        task = context["task_query"]
        if task is not None and task["fact"] is None:
            return [assistant_message(f"任务查询失败（{task['error_code']}）。")]
        citations = [item["evidence_id"] for item in context["knowledge"]]
        if task is not None:
            citations.insert(0, task["fact"]["evidence_id"])
        return [assistant_message("依据本轮证据 " + "、".join(citations))]

    def search_after_query(call):
        task = _day3_last_tool_context(call)["task_query"]
        if task["fact"] is None:
            return respond(call)
        state = task["fact"]["state"]
        return [function_call("search_knowledge", {"query": f"{state} 状态的业务含义是什么？"},
                              call_id="search")]

    steps = []
    if needs_state:
        steps.append([function_call("query_task", {"task_id": task_id}, call_id="task")])
    if needs_rule:
        if needs_state:
            steps.append({"responder": search_after_query})
        else:
            steps.append([function_call("search_knowledge", {"query": case_input["question"]},
                                        call_id="search")])
    steps.append({"responder": respond})
    return ScriptedModel(steps)


def _day3_cases():
    cases = json.loads((DATASET / "eval_cases.json").read_text(encoding="utf-8"))["cases"]
    return [case for case in cases if case["day"] == 1]


async def _check_day3_sdk_cases(cases, scenarios, evidence, model_for_case):
    topics = scenarios["knowledge"]["topics"]
    rows = []
    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        for case in cases:
            case_input, expected = case["input"], case["expected"]
            model = model_for_case(case_input)
            session = SQLiteSession(case["case_id"])
            try:
                result = await run_day3_agent(
                    build_day3_agent(model), case_input["question"], query_task=service.query_task,
                    search_knowledge=lambda query: (evidence,) if any(word in query for word in topics) else (),
                    session=session,
                )
            finally:
                session.close()
            items = [item.to_input_item() for item in result.new_items]
            calls = [item for item in items if item["type"] == "function_call"]
            outputs = [json.loads(item["output"]) for item in items if item["type"] == "function_call_output"]
            assert [item["name"] for item in calls] == expected["tool_order"], case["case_id"]
            assert len(outputs) == len(calls)
            for call in calls:
                args = json.loads(call["arguments"])
                if call["name"] == "query_task":
                    assert args["task_id"] == case_input["task_id"]
                elif expected["route"] == "rag_only":
                    assert args["query"] == case_input["question"]
                else:
                    assert all(term in args["query"] for term in expected["query_contains"])
                    assert all(term not in args["query"] for term in expected["query_excludes"])
            last = outputs[-1] if outputs else {}
            task = last.get("task_query") or {}
            fact = task.get("fact") or {}
            assert (fact.get("state"), fact.get("source")) == (expected["task_state"], expected["task_source"])
            assert [item["doc_id"] for item in last.get("knowledge", [])] == expected["doc_ids"]
            assert isinstance(result.final_output, str) and result.final_output.strip()
            if expected["status"] == "error":
                assert task["error_code"] == expected["reason"] and expected["reason"] in result.final_output
            if expected["status"] == "needs_input":
                assert "任务编号" in result.final_output
            rows.append({"case_id": case["case_id"], "question": case_input["question"],
                         "route": expected["route"], "calls": calls,
                         "contexts": outputs, "answer": result.final_output})
    return rows


def _save_day3_report(filename, model, rows):
    DAY3_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DAY3_OUTPUT / filename
    path.write_text(json.dumps({"passed": True, "model": model, "cases": rows},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Day3 SDK 验收报告：{path}")


@DAY3
@pytest.mark.usefixtures("day3_local_tracing")
@pytest.mark.asyncio
async def test_day3_sdk_tool_loop(scenarios, evidence):
    """走真实 Runner，检查五种场景的工具参数、顺序与返回证据。"""
    rows = await _check_day3_sdk_cases(_day3_cases(), scenarios, evidence, _day3_script)
    assert len(rows) == 5
    _save_day3_report("sdk_report.json", "scripted", rows)


@DAY3
@pytest.mark.usefixtures("day3_local_tracing")
@pytest.mark.asyncio
async def test_day3_session_restores_previous_tool_result(tmp_path, scenarios):
    """使用同一 SQLite Session 续问上一轮，核对历史工具结果确实进入模型输入。"""
    question = "任务 12345 现在是什么状态？"

    def follow_up(call):
        fact = _day3_last_tool_context(call)["task_query"]["fact"]
        return [assistant_message(f"上一轮查询的是任务 {fact['task_id']}。")]

    model = ScriptedModel([
        [function_call("query_task", {"task_id": "12345"}, call_id="previous-task")],
        [assistant_message("已查询任务状态。")],
        {"responder": follow_up},
    ])
    agent = build_day3_agent(model)
    session = SQLiteSession("week10-day3", tmp_path / "session.sqlite")
    try:
        with create_mock_http_client(scenarios["subject_id"]) as client:
            service = BusinessService(BusinessAPIClient(client))
            tools = dict(query_task=service.query_task, search_knowledge=lambda _: (), session=session)
            await run_day3_agent(agent, question, **tools)
            result = await run_day3_agent(agent, "我们上一轮查的是哪个任务？", **tools)
        assert "12345" in result.final_output
        assert all(item.to_input_item()["type"] != "function_call" for item in result.new_items)
        history = await session.get_items()
        assert any(item.get("type") == "function_call_output" for item in history)
    finally:
        session.close()


@pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY3_LLM") != "1",
                    reason="真实模型验收需设置 RUN_WEEK10_DAY3_LLM=1")
@pytest.mark.usefixtures("day3_local_tracing")
@pytest.mark.asyncio
async def test_day3_real_model_routes(scenarios, evidence):
    """把三条正常路径的原问题交给真实模型选择工具，业务和知识数据沿用本地样例。"""
    from agents import OpenAIResponsesModel
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "缺少项目现有的 DeepSeek OPENAI_API_KEY"
    cases = [case for case in _day3_cases() if case["expected"]["status"] == "answered"]
    assert {case["expected"]["route"] for case in cases} == {"tool_only", "rag_only", "tool_rag"}
    model_name = "deepseek-flash"
    async with AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com",
                           timeout=60.0, max_retries=0) as client:
        model = OpenAIResponsesModel(model=model_name, openai_client=client)
        rows = await _check_day3_sdk_cases(cases, scenarios, evidence, lambda _: model)
    _save_day3_report("real_model_report.json", model_name, rows)
