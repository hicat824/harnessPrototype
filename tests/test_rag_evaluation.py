"""Week7 Day6：评估器的离线回归与显式真实批量评估。

默认自检用例加载、执行轨迹和报告汇总；RUN_WEEK7_DAY6=1 启用三个练习步骤。
真实 15 条问题还需 RUN_WEEK7_DAY6_LLM=1，普通测试不调用真实模型。
"""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from rag.answering import DeepSeekGenerator
from rag.embeddings import OllamaEncoder
from rag.evaluation import (
    DEFAULT_CASES, _write_batch, apply_reviews, check_versions, classify_failures,
    collect_trace, document_group_recall, grade_trace, load_cases, run_evaluations,
)
from rag.preprocessing import KnowledgeDocument, make_chunk
from rag.retrieval import KnowledgeIndex, load_index


DAY6 = pytest.mark.skipif(os.environ.get("RUN_WEEK7_DAY6") != "1",
                          reason="填写三个评估步骤后设置 RUN_WEEK7_DAY6=1")
REAL_LLM = pytest.mark.skipif(
    os.environ.get("RUN_WEEK7_DAY6") != "1" or os.environ.get("RUN_WEEK7_DAY6_LLM") != "1",
    reason="真实批量评估需另行设置 RUN_WEEK7_DAY6_LLM=1",
)
REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("RAG_DAY6_OUTPUT", str(REPO / "rag/artifacts/day6")))
TEXT = "查询未找到任务，不代表任务执行失败。"
OTHER_TEXT = "另一个流程需要检查输入清单。"


class FixedEncoder:
    model = "fixed/day6"
    query_instruction = "offline query instruction"

    def __init__(self):
        self.vector, self.calls = [1, 0], []

    def encode_query(self, texts):
        self.calls.append(list(texts))
        return [list(self.vector)]

    def encode_document(self, texts):
        raise AssertionError("评估查询不能重编码正文")


def _answer(quote=TEXT, claim="不能把未找到当成任务失败。"):
    return {"status": "answered", "claims": [
        {"text": claim, "citations": [{"source_id": "S1", "quote": quote}]}
    ], "missing_information": ""}


class FixedGenerator:
    """手写回答替身，不能用它的通过结果证明真实 RAG 的语义质量。"""

    def __init__(self, raw=None, error=None):
        self.raw = json.dumps(_answer(), ensure_ascii=False) if raw is None else raw
        self.error, self.calls = error, []

    async def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        return self.raw


@pytest.fixture
def case():
    return {"case_id": "SYNTH-01", "question": "未找到任务是否代表任务失败？",
            "filters": {"knowledge_status": "current"}, "expected_status": "answered",
            "expected_doc_groups": [["D1"]], "allowed_versions": ["2.0"],
            "allowed_knowledge_statuses": ["current"], "reference_points": ["GOLD_ONLY_DO_NOT_SEND"]}


@pytest.fixture
def index():
    chunks = []
    for number, text in enumerate((TEXT, OTHER_TEXT), 1):
        doc = KnowledgeDocument(f"D{number}", text, {"format": "text", "source": f"synthetic/D{number}.txt",
                                "version": "2.0", "knowledge_status": "current", "scope": ["learning"],
                                "engine_code": "import_std" if number == 1 else "export_std"})
        chunks.append(make_chunk(doc, text, 0, len(text), 0, []))
    return KnowledgeIndex(FixedEncoder.model, FixedEncoder.query_instruction, chunks, [[1, 0], [0, 1]])


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_provided_fifteen_cases_match_knowledge_manifest():
    cases = load_cases()
    assert len(cases) == 15 and len({item["case_id"] for item in cases}) == 15
    root = Path(os.environ.get("WEEK7_RAG_DATASET", "/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics"))
    documents = {item["doc_id"]: item for item in json.loads((root / "manifest.json").read_text(encoding="utf-8"))["documents"]}
    for item in cases:
        for group in item["expected_doc_groups"]:
            for doc_id in group:
                assert doc_id in documents
                assert documents[doc_id]["version"] in item["allowed_versions"]
                assert documents[doc_id]["knowledge_status"] in item["allowed_knowledge_statuses"]
                assert (root / documents[doc_id]["source"]).is_file()
    assert sum(item["expected_status"] == "insufficient" for item in cases) == 2


@pytest.mark.parametrize("problem", ["version", "duplicate", "missing_question", "bad_groups", "bad_filter"])
def test_provided_case_loader_rejects_bad_data(case, tmp_path, problem):
    data = {"schema_version": 1, "cases": [case]}
    if problem == "version":
        data["schema_version"] = True
    elif problem == "duplicate":
        data["cases"].append(copy.deepcopy(case))
    elif problem == "missing_question":
        del case["question"]
    elif problem == "bad_groups":
        case["expected_doc_groups"] = []
    else:
        case["filters"] = None
    path = tmp_path / "cases.json"
    _write(path, data)
    with pytest.raises(ValueError):
        load_cases(path)


@pytest.mark.asyncio
async def test_provided_trace_records_stages_without_sending_gold(case, index):
    encoder, generator = FixedEncoder(), FixedGenerator()
    trace = await collect_trace(case, index, encoder, generator, k=1)
    assert trace["execution_error"] is None and trace["answer"]["status"] == "answered"
    assert trace["retrieved_hits"][0]["chunk"] == asdict(index.chunks[0])
    assert trace["raw_answer"] == generator.raw and trace["messages"] == generator.calls[0]
    assert "GOLD_ONLY_DO_NOT_SEND" not in json.dumps(generator.calls, ensure_ascii=False)
    assert len(encoder.calls) == len(generator.calls) == 1


@pytest.mark.parametrize("invalid_k", [0, -1, True, 1.5, "10", None])
def test_provided_case_rejects_invalid_k(case, tmp_path, invalid_k):
    """按用例覆盖的 k 也必须是正整数，防止布尔值或字符串被隐式接受。"""
    case["k"] = invalid_k
    path = tmp_path / "cases.json"
    _write(path, {"schema_version": 1, "cases": [case]})
    with pytest.raises(ValueError, match="k 必须是正整数"):
        load_cases(path)


@DAY6
@pytest.mark.asyncio
async def test_case_k_overrides_batch_default(case, index, tmp_path):
    """两条合成用例共享索引，只有指定 k 的用例扩大召回；记录实际使用值。"""
    expanded = copy.deepcopy(case)
    expanded.update(case_id="SYNTH-EXPANDED", k=2)
    result = await run_evaluations(
        [case, expanded], index, FixedEncoder(), FixedGenerator(), tmp_path, k=1,
    )
    assert [row["retrieval_k"] for row in result["rows"]] == [1, 2]
    assert [len(row["retrieved_hits"]) for row in result["rows"]] == [1, 2]
    saved = json.loads((tmp_path / "cases/SYNTH-EXPANDED.json").read_text(encoding="utf-8"))
    assert saved["retrieval_k"] == 2
    assert "k" not in case  # 覆盖配置不回写其他用例。


@pytest.mark.asyncio
@pytest.mark.parametrize("problem,category", [
    ("retrieval", "retrieval_execution_error"), ("generation", "generation_error"),
    ("format", "answer_format_error"), ("citation", "citation_error"),
])
async def test_provided_trace_preserves_partial_results_and_errors(case, index, problem, category):
    encoder, generator = FixedEncoder(), FixedGenerator()
    if problem == "retrieval":
        def fail(texts):
            raise RuntimeError("模拟检索编码失败")
        encoder.encode_query = fail
    elif problem == "generation":
        generator.error = httpx.ReadTimeout("模拟生成超时")
    elif problem == "format":
        generator.raw = "{broken"
    else:
        payload = _answer()
        payload["claims"][0]["citations"][0]["source_id"] = "S99"
        generator.raw = json.dumps(payload, ensure_ascii=False)
    trace = await collect_trace(case, index, encoder, generator, k=1)
    assert trace["execution_error"]["category"] == category
    assert trace["execution_error"]["error_type"]
    if problem == "retrieval":
        assert trace["retrieved_hits"] is None and trace["context"] is None
    else:
        assert trace["retrieved_hits"] and trace["context"]
    if problem in ("format", "citation"):
        assert trace["raw_answer"] == generator.raw
    if problem == "citation":
        assert trace["answer"] is not None


def test_provided_summary_excludes_unknown_recall_and_pending_passes(tmp_path):
    rows = []
    for recall, status in [(0.5, "needs_review"), (None, "passed")]:
        rows.append({"metrics": {"retrieval_doc_group_recall": recall}, "execution_error": None,
                     "automated_passed": True, "review": {"verdict": "pending" if status == "needs_review" else "supported"},
                     "final_status": status, "error_tags": []})
    summary = _write_batch(tmp_path, rows)
    assert summary["mean_doc_group_recall"] == 0.5 and summary["recall_case_count"] == 1
    assert summary["passed_count"] == 1 and summary["manual_pending_count"] == 1
    assert _write_batch(tmp_path, [rows[1]])["mean_doc_group_recall"] is None


def test_provided_cli_requires_explicit_real_flag():
    result = subprocess.run([sys.executable, "-B", "-m", "rag.evaluation", "run"], cwd=REPO,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--real-models" in result.stderr


@DAY6
@pytest.mark.parametrize("groups,doc_ids,expected", [
    ([["D1"]], ["D1", "D1"], 1), ([["D1", "D2"]], ["D2"], 1),
    ([["D1", "D2"], ["D3"]], ["D1"], 0.5), ([["D1"]], [], 0), ([], ["D1"], None),
])
def test_document_group_recall_counts_groups_not_chunks(groups, doc_ids, expected):
    assert document_group_recall(groups, [{"doc_id": doc_id} for doc_id in doc_ids]) == expected


@DAY6
@pytest.mark.parametrize("versions,statuses,expected", [
    (["2.0", "2.0"], ["current", "current"], True),
    (["2.0", "1.0"], ["current", "current"], False),
    (["2.0"], ["historical"], False), ([], [], True),
])
def test_version_check_requires_all_chunks_to_apply(case, versions, statuses, expected):
    chunks = [{"metadata": {"version": version, "knowledge_status": status}} for version, status in zip(versions, statuses)]
    assert check_versions(chunks, case) is expected


@DAY6
def test_failure_labels_distinguish_false_from_unknown():
    checks = {"retrieval_documents_ok": False, "context_documents_ok": None,
              "versions_ok": False, "answer_status_ok": True, "citations_ok": None}
    assert classify_failures(checks) == ["retrieval_error", "version_error"]
    assert classify_failures({}) == []
    with pytest.raises(ValueError):
        classify_failures({"retrieval_documents_ok": 0})


@DAY6
@pytest.mark.asyncio
async def test_automatic_pass_still_requires_manual_review(case, index):
    trace = await collect_trace(case, index, FixedEncoder(), FixedGenerator(), k=1)
    before = copy.deepcopy(trace)
    row = grade_trace(trace)
    assert row["automated_passed"] is True and row["final_status"] == "needs_review"
    assert row["metrics"]["retrieval_doc_group_recall"] == 1
    assert grade_trace(trace, {"verdict": "supported", "notes": "已逐条对照原文"})["final_status"] == "passed"
    assert trace == before


@DAY6
@pytest.mark.asyncio
async def test_incomplete_trace_is_not_automatically_passed(case, index):
    trace = await collect_trace(case, index, FixedEncoder(), FixedGenerator(), k=1)
    trace.update(retrieved_hits=None, context=None, answer=None)
    row = grade_trace(trace)
    assert row["automated_passed"] is False and "incomplete_trace" in row["error_tags"]


@DAY6
@pytest.mark.asyncio
async def test_manual_review_detects_hallucination_even_with_valid_quote(case, index):
    generator = FixedGenerator(raw=json.dumps(_answer(claim="所有任务都已经完成。"), ensure_ascii=False))
    trace = await collect_trace(case, index, FixedEncoder(), generator, k=1)
    assert grade_trace(trace)["automated_passed"] is True
    row = grade_trace(trace, {"verdict": "hallucination", "notes": "摘录只解释未找到，不能支持所有任务完成"})
    assert row["final_status"] == "failed" and "answer_hallucination" in row["error_tags"]


@DAY6
@pytest.mark.asyncio
@pytest.mark.parametrize("problem,tag", [("retrieval", "retrieval_error"), ("context", "context_error"),
                                       ("version", "version_error"), ("status", "answer_status_error")])
async def test_error_tags_locate_retrieval_context_version_and_answer(case, index, problem, tag):
    encoder, generator, budget = FixedEncoder(), FixedGenerator(), 9000
    if problem == "retrieval":
        encoder.vector = [0, 1]
        generator.raw = json.dumps(_answer(quote=OTHER_TEXT, claim="应核对输入清单。"), ensure_ascii=False)
    elif problem == "context":
        budget = 1
    elif problem == "version":
        index.chunks[0].metadata["version"] = "1.0"
    else:
        generator.raw = json.dumps({"status": "insufficient", "claims": [], "missing_information": "模拟漏答"})
    trace = await collect_trace(case, index, encoder, generator, k=1, max_context_chars=budget)
    row = grade_trace(trace)
    assert tag in row["error_tags"] and row["automated_passed"] is False
    if problem == "retrieval":
        assert row["checks"]["context_documents_ok"] is None


@DAY6
@pytest.mark.asyncio
async def test_abstention_case_has_no_fake_retrieval_score(case, index):
    case.update(expected_status="insufficient", expected_doc_groups=[], filters={"engine_code": "missing"})
    trace = await collect_trace(case, index, FixedEncoder(), FixedGenerator(), k=1)
    row = grade_trace(trace)
    assert row["metrics"]["retrieval_doc_group_recall"] is None
    assert row["checks"]["retrieval_documents_ok"] is None and row["automated_passed"] is True


@DAY6
@pytest.mark.asyncio
async def test_offline_batch_saves_reports_and_applies_reviews(case, index):
    second = copy.deepcopy(case)
    second.update(case_id="SYNTH-02", expected_status="insufficient", expected_doc_groups=[], filters={"engine_code": "missing"})
    output = OUTPUT_ROOT / "offline_fixture"
    result = await run_evaluations([case, second], index, FixedEncoder(), FixedGenerator(), output, k=1)
    assert result["summary"]["case_count"] == 2 and result["summary"]["automated_passed_count"] == 2
    assert result["summary"]["passed_count"] == 0 and result["summary"]["manual_pending_count"] == 2
    reviews = output / "manual_reviews.json"
    _write(reviews, {"reviews": [{"case_id": item["case_id"], "verdict": "supported", "notes": "固定答案离线对照通过"}
                                  for item in (case, second)]})
    reviewed = apply_reviews(output / "evaluation_report.json", reviews)
    assert reviewed["summary"]["passed_count"] == 2 and reviewed["summary"]["manual_pending_count"] == 0
    print(f"Day6 离线评估产物：{output}")


@DAY6
@pytest.mark.asyncio
async def test_batch_continues_after_generation_error(case, index, tmp_path):
    class SometimesFails(FixedGenerator):
        async def __call__(self, messages):
            self.calls.append(copy.deepcopy(messages))
            if len(self.calls) == 1:
                raise RuntimeError("模拟首条生成失败")
            return self.raw

    second = copy.deepcopy(case)
    second["case_id"] = "SYNTH-02"
    result = await run_evaluations([case, second], index, FixedEncoder(), SometimesFails(), tmp_path, k=1)
    assert result["summary"]["case_count"] == 2 and result["summary"]["execution_error_count"] == 1
    assert result["rows"][1]["automated_passed"] is True
    first = json.loads((tmp_path / "cases/SYNTH-01.json").read_text(encoding="utf-8"))
    assert first["execution_error"]["stage"] == "generation" and first["context"]


@DAY6
@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["unknown", "duplicate", "empty_reason"])
async def test_manual_review_rejects_inconsistent_input(case, index, tmp_path, problem):
    await run_evaluations([case], index, FixedEncoder(), FixedGenerator(), tmp_path, k=1)
    review = {"case_id": case["case_id"], "verdict": "supported", "notes": ""}
    reviews = [review]
    if problem == "unknown":
        review["case_id"] = "UNKNOWN"
    elif problem == "duplicate":
        reviews.append(copy.deepcopy(review))
    else:
        review["verdict"] = "hallucination"
    path = tmp_path / "reviews.json"
    _write(path, {"reviews": reviews})
    with pytest.raises(ValueError):
        apply_reviews(tmp_path / "evaluation_report.json", path)


@REAL_LLM
@pytest.mark.asyncio
async def test_real_rag_evaluation_records_fifteen_cases():
    assert bool(os.environ.get("OPENAI_API_KEY")), "真实生成需要既有 DeepSeek OPENAI_API_KEY"
    cases = load_cases(os.environ.get("RAG_DAY6_CASES", str(DEFAULT_CASES)))
    assert len(cases) == 15, "本日真实验收使用固定的 15 条问题"
    index = load_index(os.environ.get("RAG_DAY6_INDEX", str(REPO / "rag/artifacts/day4/local_model/knowledge_index.json")))
    encoder = OllamaEncoder(index.model_name, os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                            query_instruction=index.query_instruction)
    generate = DeepSeekGenerator(os.environ.get("RAG_GENERATION_MODEL", "deepseek-flash"))
    output = OUTPUT_ROOT / "real_model"
    result = await run_evaluations(cases, index, encoder, generate, output)
    assert result["summary"]["case_count"] == 15
    assert result["summary"]["execution_error_count"] == 0, "请查看逐条报告中的阶段、异常类型和原始生成文本"
    assert result["summary"]["manual_pending_count"] == 15
    print(f"Day6 真实评估产物：{output}；pytest 通过不表示语义全通过，需逐条人工复核")
