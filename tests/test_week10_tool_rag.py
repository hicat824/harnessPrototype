"""Week10 验收。离线路径和 Day3 真实模型路径通过各自的开关启用。"""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import pytest
from agents import SQLiteSession
from agents.testing.model import ScriptedModel, assistant_message, function_call
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

from week6 import BusinessAPIClient, BusinessResult, BusinessService, create_mock_http_client
from week10_tool_rag import (
    Day1Request, KnowledgeEvidence, build_day2_context, prepare_knowledge_query, run_day1,
    build_day3_agent, query_task_tool, run_day3_agent, search_knowledge_tool,
    Day4Answer, Day4GroundedStatement, Day4TaskClaim, validate_day4_answer,
    Day5TaskState, run_day5_turn,
    Day6ExpectedTurn, Day6ToolCall, grade_day6_outcome, grade_day6_trajectory,
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


DAY4 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY4") != "1",
                         reason="Day4 离线验收需设置 RUN_WEEK10_DAY4=1")


@pytest.fixture
def day4_answer():
    """与 Mock 的 RUNNING 状态及 KB006 的规则对应的结构化回答。"""
    return Day4Answer(
        current_task=Day4TaskClaim(task_id="12345", state="RUNNING", evidence_id="T1"),
        rules=[Day4GroundedStatement(text="RUNNING 表示任务执行中。", evidence_ids=["K1"])],
        inferences=[Day4GroundedStatement(text="据此推断任务尚未完成。", evidence_ids=["T1", "K1"])],
        unknowns=["预计完成时间未知。"],
        next_steps=["如需确认完成情况，可稍后再次查询任务。"],
    )


def test_day4_answer_schema(day4_answer):
    """推断与规则都有引用，未知项和后续建议分别记录。"""
    data = day4_answer.model_dump()
    assert Day4Answer.model_validate(data) == day4_answer
    data["inferences"][0]["evidence_ids"] = []
    with pytest.raises(ValueError, match="evidence_ids"):
        Day4Answer.model_validate(data)


@DAY4
def test_day4_accepts_current_fact_and_cited_inference(day4_answer, day2_task_results, evidence):
    success, _ = day2_task_results
    context = build_day2_context(success, (evidence,))
    validate_day4_answer(day4_answer, context, requested_task_id="12345")

    # 只问规则时，来自同一知识片段的说明仍可以独立成立。
    rule_answer = day4_answer.model_copy(update={"current_task": None, "inferences": []})
    validate_day4_answer(rule_answer, build_day2_context(None, (evidence,)), requested_task_id=None)


@DAY4
def test_day4_current_claim_must_match_both_request_and_api(day4_answer, day2_task_results, evidence):
    success, _ = day2_task_results
    context = build_day2_context(success, (evidence,))
    assert day4_answer.current_task is not None
    for mismatch in ({"evidence_id": "K1"}, {"task_id": "98765"}, {"state": "FAILED"}):
        claim = day4_answer.current_task.model_copy(update=mismatch)
        wrong_answer = day4_answer.model_copy(update={"current_task": claim})
        with pytest.raises(ValueError, match="当前状态"):
            validate_day4_answer(wrong_answer, context, requested_task_id="12345")

    for requested in ("98765", None):
        with pytest.raises(ValueError, match="当前状态"):
            validate_day4_answer(day4_answer, context, requested_task_id=requested)


@DAY4
def test_day4_historical_case_cannot_establish_current_state():
    # 合成的历史案例故意包含同一任务编号和明确状态，模拟看似相关的检索结果。
    history = KnowledgeEvidence("W10D4-HISTORY", "fixture:historical_task",
                                "历史案例：任务 12345 在 2025 年记录的状态为 FAILED。")
    missing = BusinessResult(status="error", data=None, error_code="NOT_FOUND")
    for task_result in (None, missing):
        context = build_day2_context(task_result, (history,))
        for evidence_id in ("K1", "T1"):
            wrong = Day4Answer(
                current_task=Day4TaskClaim(task_id="12345", state="FAILED", evidence_id=evidence_id),
                rules=[], inferences=[], unknowns=[], next_steps=[],
            )
            with pytest.raises(ValueError, match="当前状态"):
                validate_day4_answer(wrong, context, requested_task_id="12345")

        unknown = Day4Answer(
            current_task=None, rules=[], inferences=[], unknowns=["缺少任务 12345 的本轮查询结果。"],
            next_steps=["通过任务 API 查询当前状态。"],
        )
        validate_day4_answer(unknown, context, requested_task_id="12345")
        with pytest.raises(ValueError, match="unknowns"):
            validate_day4_answer(unknown.model_copy(update={"unknowns": [" "]}), context,
                                 requested_task_id="12345")


@DAY4
def test_day4_references_must_use_visible_and_allowed_sources(
    day4_answer, day2_cases, day2_evidence, day2_task_results,
):
    success, _ = day2_task_results
    context = build_day2_context(success, day2_evidence, max_chars=day2_cases["budgets"]["tight"])
    assert context.omitted_ids == ("K2",)
    for field, evidence_id in (("rules", "T1"), ("rules", "K99"), ("inferences", "K2")):
        claim = Day4GroundedStatement(text="把未核实的内容写成结论。", evidence_ids=["K1", evidence_id])
        wrong = day4_answer.model_copy(update={field: [claim]})
        with pytest.raises(ValueError, match="引用"):
            validate_day4_answer(wrong, context, requested_task_id="12345")


DAY5 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY5") != "1",
                         reason="Day5 离线验收需设置 RUN_WEEK10_DAY5=1")


@pytest.fixture
def day5_tools(scenarios, evidence):
    """复用 Week6 Mock API 和 KB006 文档，记录调用次数与参数。"""
    from unittest.mock import Mock

    with create_mock_http_client(scenarios["subject_id"]) as client:
        service = BusinessService(BusinessAPIClient(client))
        yield {"query_task": Mock(wraps=service.query_task),
               "search_knowledge": Mock(return_value=(evidence,))}


def test_day5_task_state_starts_with_an_unused_budget():
    request = Day1Request("任务 12345 现在是什么状态？", True, False, "12345")
    state = Day5TaskState(request, max_tool_calls=0)
    assert state.tool_calls == 0 and state.last_result is None
    with pytest.raises(ValueError, match="max_tool_calls"):
        Day5TaskState(request, max_tool_calls=-1)


@DAY5
def test_day5_asks_then_resumes_with_the_supplied_task_id(day5_tools):
    request = Day1Request("任务现在是什么状态？也解释一下状态的含义。", True, True)
    state = Day5TaskState(request)
    first = run_day5_turn(state, **day5_tools)
    assert first.status == "needs_input" and first.reason == "MISSING_PARAMETER"
    assert run_day5_turn(state, supplied_task_id=" ", **day5_tools).status == "needs_input"
    assert state.tool_calls == 0
    day5_tools["query_task"].assert_not_called()
    day5_tools["search_knowledge"].assert_not_called()

    result = run_day5_turn(state, supplied_task_id=" 12345 ", **day5_tools)
    assert (result.route, result.status, result.task_state) == ("tool_rag", "answered", "RUNNING")
    assert state.request.question == request.question
    assert state.request.needs_current_state and state.request.needs_rule
    assert state.request.task_id == "12345" and state.tool_calls == 2
    day5_tools["query_task"].assert_called_once_with("12345")
    day5_tools["search_knowledge"].assert_called_once()
    assert "RUNNING" in day5_tools["search_knowledge"].call_args.args[0]

    assert run_day5_turn(state, supplied_task_id="99999", **day5_tools) == result
    assert state.request.task_id == "12345"
    assert day5_tools["query_task"].call_count == day5_tools["search_knowledge"].call_count == 1


@DAY5
@pytest.mark.parametrize("max_calls, observed_state", [(0, None), (1, "RUNNING")])
def test_day5_budget_stops_before_the_next_tool(max_calls, observed_state, day5_tools):
    request = Day1Request("任务 12345 当前是什么状态？解释这个状态。", True, True, "12345")
    state = Day5TaskState(request, max_tool_calls=max_calls)
    result = run_day5_turn(state, **day5_tools)
    assert (result.status, result.reason, result.task_state) == (
        "error", "TOOL_BUDGET_EXHAUSTED", observed_state,
    )
    assert state.tool_calls == day5_tools["query_task"].call_count == max_calls
    day5_tools["search_knowledge"].assert_not_called()


@DAY5
def test_day5_permission_failure_ends_the_task_without_retry(day5_tools):
    # Week6 Mock 使用任务编号 33333 触发 HTTP 403。
    request = Day1Request("任务 33333 是什么状态？解释查到的状态。", True, True, "33333")
    state = Day5TaskState(request)
    result = run_day5_turn(state, **day5_tools)
    assert (result.status, result.reason, result.task_state) == ("error", "FORBIDDEN", None)
    assert run_day5_turn(state, supplied_task_id="12345", **day5_tools) == result
    assert state.tool_calls == 1
    day5_tools["query_task"].assert_called_once_with("33333")
    day5_tools["search_knowledge"].assert_not_called()


@DAY5
@pytest.mark.asyncio
async def test_day5_resumes_after_compacting_session_history(day5_tools):
    from contextCompact import compact_session

    request = Day1Request("查一个任务的当前状态，顺带解释状态含义。", True, True)
    state = Day5TaskState(request)
    session = SQLiteSession("week10-day5")
    try:
        first = run_day5_turn(state, **day5_tools)
        assert first.status == "needs_input"
        await session.add_items([
            {"role": "user", "content": request.question + "这是补充背景。" * 40},
            {"role": "assistant", "content": "请提供任务编号。"},
            {"role": "user", "content": "稍等。"},
            {"role": "assistant", "content": "好的。"},
        ])

        async def summarize(old_history):
            assert request.question in old_history
            return {"summary": "此前讨论了一项任务。", "tokens": 8}

        report = await compact_session(session, summarize, keep_recent_turns=1, force=True, reserved_tokens=0)
        assert report["compacted"] and report["after"] < report["before"]
        assert request.question not in json.dumps(await session.get_items(), ensure_ascii=False)

        result = run_day5_turn(state, supplied_task_id="12345", **day5_tools)
        assert (result.status, result.route, result.task_state) == ("answered", "tool_rag", "RUNNING")
        assert state.request.question == request.question and state.tool_calls == 2
        day5_tools["query_task"].assert_called_once_with("12345")
        day5_tools["search_knowledge"].assert_called_once()
    finally:
        session.close()

DAY6 = pytest.mark.skipif(os.environ.get("RUN_WEEK10_DAY6") != "1",
                         reason="Day6 离线验收需设置 RUN_WEEK10_DAY6=1")
DAY6_OUTPUT = Path(os.environ.get("WEEK10_DAY6_OUTPUT", DATASET / "runs/day6"))


@pytest.fixture(scope="module")
def day6_dataset():
    path = DATASET / "day6_eval_cases.json"
    assert path.is_file(), f"找不到 Day6 样例，请检查 WEEK10_DATASET：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _day6_catalog(dataset):
    # 和 Day1 一样从 Week9 文档加载原文。Mock 词表决定召回，评分答案不参与检索。
    catalog = []
    for entry in dataset["knowledge"]:
        line = entry["line"]
        text = (KNOWLEDGE_ROOT / entry["path"]).read_text(encoding="utf-8").splitlines()[line - 1]
        item = KnowledgeEvidence(entry["doc_id"], f"{entry['path']}#L{line}", text)
        catalog.append((item, entry["query_terms"]))
    return catalog


def _day6_mock_answer(result, context):
    """仅用预算内的证据生成固定答复。复制规则原文，便于程序确定性核对。"""
    payload = json.loads(context.text)
    task_query = payload["task_query"]
    fact = task_query["fact"] if task_query is not None else None
    current_task = Day4TaskClaim(task_id=fact["task_id"], state=fact["state"],
                                 evidence_id=fact["evidence_id"]) if fact else None
    rules = [Day4GroundedStatement(text=item["text"], evidence_ids=[item["evidence_id"]])
             for item in payload["knowledge"]]
    unknowns = [result.reason] if result.reason else []
    if payload["knowledge_truncated"]:
        unknowns.append("部分知识证据未进入本轮 Context。")
    return Day4Answer(current_task=current_task, rules=rules, inferences=[],
                      unknowns=unknowns, next_steps=[])


def _observe_day6_case(inputs, service, catalog):
    """执行 Day5 多轮路径，逐轮记录工具、召回、证据和固定答复。"""
    state = Day5TaskState(Day1Request(**inputs["request"]), max_tool_calls=inputs.get("max_tool_calls", 2))
    active_sources = inputs.get("available_sources")
    observations = []
    task_result = None

    def query_task(task_id):
        nonlocal task_result
        calls.append(Day6ToolCall("query_task", {"task_id": task_id}))
        task_result = service.query_task(task_id)
        return task_result

    def search_knowledge(query):
        hits = tuple(item for item, terms in catalog
                     if (active_sources is None or item.source in active_sources)
                     and any(term in query for term in terms))
        calls.append(Day6ToolCall("search_knowledge", {"query": query}, tuple(item.source for item in hits)))
        return hits

    for supplied_task_id in [None, *inputs.get("follow_ups", [])]:
        # 只清空本轮轨迹。任务状态及已取得的任务事实仍可被续轮使用。
        calls = []
        result = run_day5_turn(state, query_task=query_task, search_knowledge=search_knowledge,
                               supplied_task_id=supplied_task_id)
        context = build_day2_context(task_result, result.evidence, max_chars=inputs.get("max_context_chars", 6000))
        answer = _day6_mock_answer(result, context)
        requested_id = state.request.task_id if state.request.needs_current_state else None
        observations.append((result, calls, context, answer, requested_id))
    return observations


def _day6_metrics(rows):
    # 逐题统计，每个适用的轮次都通过才记一题通过。同题轮次多也只计一题。
    metrics = {}
    for layer in ("trajectory", "outcome"):
        names = rows[0]["turns"][0][layer].keys()
        metrics[layer] = {}
        for name in names:
            case_scores = []
            for row in rows:
                scores = [turn[layer][name] for turn in row["turns"] if turn[layer][name] is not None]
                if scores:
                    case_scores.append(all(scores))
            applicable = len(case_scores)
            passed = sum(case_scores)
            metrics[layer][name] = {"passed": passed, "applicable": applicable,
                                    "rate": passed / applicable if applicable else None}
        passed = sum(all(score is not False for turn in row["turns"] for score in turn[layer].values())
                     for row in rows)
        metrics[layer]["overall"] = {"passed": passed, "applicable": len(rows), "rate": passed / len(rows)}
    return metrics


@DAY6
def test_day6_dataset_has_thirty_mixed_cases(day6_dataset):
    """单独核对样例数量、类别和轮次，填写评分函数前也能运行。"""
    cases = day6_dataset["cases"]
    assert len(cases) >= 30 and len({case["case_id"] for case in cases}) == len(cases)
    assert {case["category"] for case in cases} == {
        "tool_only", "rag_only", "tool_rag", "stops", "continuation",
    }
    for case in cases:
        assert len(case["expected_turns"]) == 1 + len(case["input"].get("follow_ups", [])), case["case_id"]


@DAY6
def test_day6_trajectory_distinguishes_wrong_calls_from_a_retrieval_hit():
    source = "knowledge/KB006.md#L6"
    expected = Day6ExpectedTurn(["query_task", "search_knowledge"], "answered", task_ids=("12345",),
                                query_contains=("RUNNING",), query_excludes=("FAILED",),
                                retrieved_sources=[source])
    task_call = Day6ToolCall("query_task", {"task_id": "12345"})
    wrong_query = Day6ToolCall("search_knowledge", {"query": "RUNNING 还是 FAILED？"}, (source,))
    assert grade_day6_trajectory([task_call, wrong_query], expected) == {
        "tool_order": True, "tool_arguments": False, "retrieval": True,
    }
    assert grade_day6_trajectory([wrong_query, task_call], expected)["tool_order"] is False
    assert grade_day6_trajectory([task_call], expected)["retrieval"] is False
    wrong_task = Day6ToolCall("query_task", {"task_id": "23456"})
    search = Day6ToolCall("search_knowledge", {"query": "解释 RUNNING"}, (source,))
    assert grade_day6_trajectory([wrong_task, search], expected)["tool_arguments"] is False
    assert grade_day6_trajectory([], Day6ExpectedTurn([], "needs_input")) == {
        "tool_order": True, "tool_arguments": None, "retrieval": None,
    }


@DAY6
def test_day6_outcome_distinguishes_source_support_from_answer_completeness(evidence):
    task = BusinessResult(status="success", data={"task_id": "12345", "state": "RUNNING"})
    context = build_day2_context(task, (evidence,))
    result = run_day1(Day1Request("查询并解释任务 12345 的状态", True, True, "12345"),
                      query_task=lambda _: task, search_knowledge=lambda _: (evidence,))
    expected = Day6ExpectedTurn(["query_task", "search_knowledge"], "answered",
                                task_id="12345", task_state="RUNNING", rule_sources=[evidence.source])
    answer = _day6_mock_answer(result, context)
    assert all(grade_day6_outcome(result, answer, context, expected, requested_task_id="12345").values())

    missing = answer.model_copy(update={"current_task": None, "unknowns": ["状态未说明"]})
    assert grade_day6_outcome(result, missing, context, expected, requested_task_id="12345") == {
        "status": True, "source_support": True, "final_answer": False,
    }
    needs_explanation = replace(expected, requires_unknowns=True)
    scores = grade_day6_outcome(result, answer, context, needs_explanation, requested_task_id="12345")
    assert scores["final_answer"] is False

    unsupported = answer.model_copy(update={"rules": [Day4GroundedStatement(
        text="RUNNING 表示已完成", evidence_ids=["K1"],
    )]})
    assert grade_day6_outcome(result, unsupported, context, expected, requested_task_id="12345")["source_support"] is False


@DAY6
def test_day6_mock_evals_report_outcome_and_trajectory(day6_dataset):
    catalog = _day6_catalog(day6_dataset)
    rows = []
    for case in day6_dataset["cases"]:
        inputs = case["input"]
        started = perf_counter()
        with create_mock_http_client(inputs.get("subject_id", "alice")) as client:
            observations = _observe_day6_case(inputs, BusinessService(BusinessAPIClient(client)), catalog)
        elapsed = perf_counter() - started
        assert len(observations) == len(case["expected_turns"]), case["case_id"]
        turns = []
        for (result, calls, context, answer, task_id), reference in zip(observations, case["expected_turns"]):
            expected = Day6ExpectedTurn(**reference)
            checks = {"trajectory": grade_day6_trajectory(calls, expected), "outcome": grade_day6_outcome(
                result, answer, context, expected, requested_task_id=task_id,
            )}
            turns.append({"calls": [asdict(call) for call in calls], "route": result.route,
                          "status": result.status, "reason": result.reason,
                          "omitted_ids": context.omitted_ids, "answer": answer.model_dump(), **checks})
        rows.append({"case_id": case["case_id"], "category": case["category"], "elapsed_seconds": elapsed,
                     "turns": turns})

    failed = {row["case_id"]: [f"turn{index}.{layer}.{name}"
                              for index, turn in enumerate(row["turns"], 1)
                              for layer in ("trajectory", "outcome")
                              for name, score in turn[layer].items() if score is False]
              for row in rows}
    failed = {case_id: checks for case_id, checks in failed.items() if checks}
    report = {"mode": "mock", "passed": not failed, "case_count": len(rows), "metrics": _day6_metrics(rows),
              "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows), "usage": None,
              "failed_checks": failed, "cases": rows}
    DAY6_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DAY6_OUTPUT / "mock_eval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Day6 Mock 分层评估报告：{path}")
    assert not failed, failed
