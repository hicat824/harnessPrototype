"""Week7 Day6：固定问题集评估与分阶段错误分析。

三个 TODO(W7D6) 分别练习文档召回、版本适用性、False 与未检查 None 的区分。
运行、逐条落盘、异常记录和人工复核流程均已提供，不修改 Day2～Day5 主流程。
"""

from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

import httpx
from pydantic import ValidationError

from rag.answering import ContextBundle, DeepSeekGenerator, RagAnswer, answer_from_context, build_context, validate_answer
from rag.embeddings import OllamaEncoder, _save_json
from rag.retrieval import KnowledgeIndex, _check_search_options, load_index, search_knowledge


DEFAULT_CASES = Path(__file__).with_name("eval_cases_day6.json")


def _strings(values, *, allow_empty: bool = False) -> None:
    """已提供：检查非空字符串列表，允许有明确含义的空列表。"""
    if (not isinstance(values, list) or (not allow_empty and not values)
            or any(not isinstance(value, str) or not value.strip() for value in values)):
        raise ValueError("预期为非空白字符串列表")


def _check_case(case: dict) -> None:
    """已提供：用例格式校验；filters 是请求条件，参考答案只用于评估。"""
    if not isinstance(case, dict):
        raise ValueError("用例必须为字典")
    required = {"case_id", "question", "filters", "expected_status", "expected_doc_groups",
                "allowed_versions", "allowed_knowledge_statuses", "reference_points"}
    if not required <= case.keys():
        raise ValueError("评估用例缺少必要字段")
    if not isinstance(case["case_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case["case_id"]):
        raise ValueError("case_id 只能包含字母、数字、下划线和连字符")
    if not isinstance(case["question"], str) or not case["question"].strip():
        raise ValueError("问题不能为空")
    if not isinstance(case["filters"], dict):
        raise ValueError("filters 必须是明确的字典，允许空字典")
    # 用例可单独指定检索数量；省略时由 collect_trace 使用调用方的默认值。
    _check_search_options(case.get("k", 1), case["filters"], None)
    if case["expected_status"] not in ("answered", "insufficient"):
        raise ValueError("expected_status 必须为 answered/insufficient")
    groups = case["expected_doc_groups"]
    if not isinstance(groups, list):
        raise ValueError("expected_doc_groups 必须为列表")
    for group in groups:
        _strings(group)
    if case["expected_status"] == "answered" and not groups:
        raise ValueError("有依据回答用例必须指定至少一个预期文档组")
    for name in ("allowed_versions", "allowed_knowledge_statuses"):
        _strings(case[name], allow_empty=True)
    _strings(case["reference_points"])


def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    """已提供：读取版本 1 的评估用例，拒绝重复 ID，不把参考项加入知识索引。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or not isinstance(data.get("cases"), list) or not data["cases"]):
        raise ValueError("用例文件必须包含 schema_version=1 和非空 cases 列表")
    ids = set()
    for case in data["cases"]:
        _check_case(case)
        if case["case_id"] in ids:
            raise ValueError("case_id 不能重复")
        ids.add(case["case_id"])
    return data["cases"]


def document_group_recall(expected_groups: list[list[str]], chunks: list[dict]) -> float | None:
    """计算文档组召回率：命中的预期组数 / 预期组数。

    每组是可替代来源，例如 [[KB008,KB014],[KB007]] 代表需要两类依据，
    第一类命中 KB008 或 KB014 任意一个即可。相同文档的多个 Chunk 不能重复加分。
    无预期组返回 None，表示不适用，不能伪造成满分或零分；有组但没召回为 0。
    输入由用例加载器及 Day4 检索校验，输出范围 [0,1]，不修改输入。

    练习 1：利用 present_docs 判断每组是否至少命中一个文档，计算 matched_groups。
    """
    if not expected_groups:
        return None
    # 同一文档的多个 Chunk 只保留一个文档 ID。
    present_docs = {chunk["doc_id"] for chunk in chunks}
    matched_groups = 0
    for group in expected_groups:
        # 组内任意文档被召回即命中；逐组计数，每组最多加一次。
        if any(doc_id in present_docs for doc_id in group):
            matched_groups += 1
    return matched_groups / len(expected_groups)


def _version_matches(chunk: dict, case: dict) -> bool:
    """已提供：版本和 current/historical 必须都适用；空允许列表表示不限。"""
    metadata = chunk.get("metadata", {})
    return (
        (not case["allowed_versions"] or metadata.get("version") in case["allowed_versions"])
        and (not case["allowed_knowledge_statuses"]
             or metadata.get("knowledge_status") in case["allowed_knowledge_statuses"])
    )


def check_versions(chunks: list[dict], case: dict) -> bool:
    """判断所有检索/Context 片段的版本是否适用，而不是只看其中一条。

    chunks 是 Chunk 字典列表，case 提供允许版本和知识状态；不按正文猜版本。
    任何一条不适用就为 False；空列表为 True，但这不代表文档已召回。
    练习 2：复用 _version_matches 计算 versions_ok。
    """
    # TODO(W7D6-2)：versions_ok = ...，检查全部片段，空列表也需正确处理。
    versions_ok = True
    for chunk in chunks:
        if not _version_matches(chunk, case):
            versions_ok = False
            break
    return versions_ok


CHECK_ERRORS = {
    "retrieval_documents_ok": "retrieval_error",
    "context_documents_ok": "context_error",
    "versions_ok": "version_error",
    "scope_ok": "scope_error",
    "answer_status_ok": "answer_status_error",
    "citations_ok": "citation_error",
    "empty_context_skips_generation": "unnecessary_generation",
}


def classify_failures(checks: dict[str, bool | None]) -> list[str]:
    """把失败检查映射成错误标签，允许一条用例有多个错误。

    True 表示通过，False 表示失败，None 表示未检查或不适用。
    不能用 not value 把 None 算作失败，也不能把“未检查”当成通过。
    练习 3：只在 value 确实为 False 时令 failed 为 True。
    """
    failures = []
    for name, tag in CHECK_ERRORS.items():
        value = checks.get(name)
        if value is not None and type(value) is not bool:
            raise ValueError("检查值只能为 True/False/None")
        # TODO(W7D6-3)：failed = ...，区分失败与不适用/未知。
        failed = value is False
        if failed:
            failures.append(tag)
    return failures


async def collect_trace(case: dict, index: KnowledgeIndex, encoder, generate, *,
                        k: int = 5, max_context_chars: int = 9000) -> dict:
    """已提供：复用 Day4/Day5 收集一次执行记录，不给模型发送评估参考项。

    返回检索片段、Context、实际消息、原始生成文本、已解析回答及阶段异常。
    case 中可选的正整数 k 优先于函数参数 k，实际取值记录为 retrieval_k。
    模型输出无效时仍保留原始文本；异常仅记录阶段、类型和可用 HTTP 状态码，
    不记录可能带密钥的任意异常正文。不自动重试，不把故障当成信息不足。
    """
    _check_case(case)
    retrieval_k = case.get("k", k)
    trace = {"case_id": case["case_id"], "case": copy.deepcopy(case), "retrieved_hits": None,
             "retrieval_k": retrieval_k,
             "context": None, "messages": [], "raw_answer": None, "answer": None,
             "resolved_citations": [], "generation_called": False, "execution_error": None}
    stage, started = "retrieval", time.perf_counter()
    try:
        hits = search_knowledge(case["question"], index, encoder, k=retrieval_k, filters=case["filters"])
        trace["retrieved_hits"] = copy.deepcopy(hits)
        stage = "context"
        context = build_context(hits, max_chars=max_context_chars)
        trace["context"] = asdict(context)

        async def recording_generate(messages):
            nonlocal stage
            trace["messages"] = copy.deepcopy(messages)
            trace["generation_called"] = True
            stage = "generation"
            raw = await generate(messages)
            trace["raw_answer"] = raw
            stage = "answer_validation"
            return raw

        stage = "answer_pipeline"
        result = await answer_from_context(case["question"], context, recording_generate)
        for name in ("messages", "answer", "resolved_citations", "generation_called"):
            trace[name] = result[name]
    except Exception as error:
        category = {"retrieval": "retrieval_execution_error", "context": "context_build_error",
                    "generation": "generation_error", "answer_pipeline": "answer_pipeline_error"}.get(stage)
        if stage == "answer_validation":
            category = "answer_format_error" if isinstance(error, ValidationError) else "citation_error"
            # 引用验证失败时，仍保留已经满足结构的回答，以便定位错误摘录或编号。
            try:
                trace["answer"] = RagAnswer.model_validate_json(trace["raw_answer"]).model_dump()
            except (ValidationError, TypeError):
                pass
        trace["execution_error"] = {"stage": stage, "error_type": type(error).__name__,
                                    "category": category}
        if isinstance(error, httpx.HTTPStatusError):
            trace["execution_error"]["http_status"] = error.response.status_code
    trace["elapsed_seconds"] = time.perf_counter() - started
    return trace


def _check_review(review: dict) -> None:
    """已提供：人工评语必须区分待复核、支持、幻觉和遗漏；失败须说明原因。"""
    if (not isinstance(review, dict) or review.get("verdict") not in
            ("pending", "supported", "hallucination", "incomplete") or not isinstance(review.get("notes"), str)):
        raise ValueError("人工复核需包含合法 verdict 和字符串 notes")
    if review["verdict"] in ("hallucination", "incomplete") and not review["notes"].strip():
        raise ValueError("判定幻觉或遗漏时，notes 必须说明原因")


def grade_trace(trace: dict, review: dict | None = None) -> dict:
    """已提供：自动指标与人工语义结论分开，返回独立复制的评估记录。

    文档组召回只说明命中来源文档，不保证具体 Chunk 足以回答。
    无 gold 文档组时，召回与对应检查保留 None；缺少执行结果也不假装算过分。
    自动通过但人工 pending 时，final_status 为 needs_review，而不是 passed。
    """
    row, case = copy.deepcopy(trace), trace["case"]
    retrieved = None if trace["retrieved_hits"] is None else [hit["chunk"] for hit in trace["retrieved_hits"]]
    context = None if trace["context"] is None else [source["chunk"] for source in trace["context"]["sources"]]
    retrieval_recall = None if retrieved is None else document_group_recall(case["expected_doc_groups"], retrieved)
    context_recall = None if context is None else document_group_recall(case["expected_doc_groups"], context)
    retrieval_ok = None if retrieval_recall is None else retrieval_recall == 1
    # 未召回目标时，Context 缺失是上游结果，不再直接归因给 Context 选择。
    context_ok = None if context_recall is None or retrieval_ok is False else context_recall == 1
    chunks = (retrieved or []) + (context or [])
    engine = case["filters"].get("engine_code")
    checks = {
        "retrieval_documents_ok": retrieval_ok, "context_documents_ok": context_ok,
        "versions_ok": None if retrieved is None else check_versions(chunks, case),
        "scope_ok": None if retrieved is None or engine is None else
            all(chunk["metadata"].get("engine_code") == engine for chunk in chunks),
        "answer_status_ok": None if trace["answer"] is None else trace["answer"]["status"] == case["expected_status"],
        "citations_ok": None,
        "empty_context_skips_generation": None if context is None else bool(context) or not trace["generation_called"],
    }
    if trace["answer"] is not None and trace["context"] is not None:
        try:
            parsed = RagAnswer.model_validate(trace["answer"])
            validate_answer(parsed, ContextBundle(**trace["context"]))
            checks["citations_ok"] = True if parsed.status == "answered" else None
        except (ValueError, TypeError):
            checks["citations_ok"] = False
    failures = classify_failures(checks)
    if trace["execution_error"] is not None:
        failures.append(trace["execution_error"]["category"])
    elif retrieved is None or context is None or trace["answer"] is None:
        # 没有异常记录也不代表执行完整；不能把一份未完成 trace 判为自动通过。
        failures.append("incomplete_trace")
    failures = list(dict.fromkeys(failures))
    automatic_ok = not failures
    review = copy.deepcopy(review if review is not None else {"verdict": "pending", "notes": ""})
    _check_review(review)
    if review["verdict"] in ("hallucination", "incomplete"):
        failures.append("answer_hallucination" if review["verdict"] == "hallucination" else "answer_incomplete")
    final_status = ("failed" if failures else "passed" if review["verdict"] == "supported" else "needs_review")
    row.update(metrics={"retrieval_doc_group_recall": retrieval_recall, "context_doc_group_recall": context_recall},
               checks=checks, error_tags=failures, automated_passed=automatic_ok,
               review=review, final_status=final_status)
    return row


def _write_batch(directory: Path, rows: list[dict], report_name: str = "evaluation_report.json") -> dict:
    """已提供：按已完成条目汇总，不把 None 当零，也不把待人工复核当通过。"""
    recalls = [row["metrics"]["retrieval_doc_group_recall"] for row in rows
               if row["metrics"]["retrieval_doc_group_recall"] is not None]
    summary = {
        "case_count": len(rows), "execution_error_count": sum(row["execution_error"] is not None for row in rows),
        "automated_passed_count": sum(row["automated_passed"] for row in rows),
        "manual_pending_count": sum(row["review"]["verdict"] == "pending" for row in rows),
        "passed_count": sum(row["final_status"] == "passed" for row in rows),
        "failed_count": sum(row["final_status"] == "failed" for row in rows),
        "recall_case_count": len(recalls), "mean_doc_group_recall": sum(recalls) / len(recalls) if recalls else None,
        "error_counts": dict(Counter(tag for row in rows for tag in row["error_tags"])),
    }
    directory.mkdir(parents=True, exist_ok=True)
    _save_json(directory / "summary.json", summary)
    _save_json(directory / report_name, {"summary": summary, "rows": rows})
    return summary


async def run_evaluations(cases: list[dict], index: KnowledgeIndex, encoder, generate,
                          output_dir: str | Path, *, k: int = 5, max_context_chars: int = 9000) -> dict:
    """已提供：顺序评估，逐条保存 trace 和结果；单条模型故障记录后继续。

    不吞掉评估器自身的 TODO/实现异常。真实使用前先跑完整离线测试。
    参考项保存在评估报告，绝不传入模型；每次查询仍只编码问题。
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须为非空列表")
    for case in cases:
        _check_case(case)
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("case_id 不能重复")
    directory = Path(output_dir)
    records = directory / "cases"
    records.mkdir(parents=True, exist_ok=True)
    _save_json(directory / "manual_reviews_template.json", {"reviews": [
        {"case_id": case["case_id"], "verdict": "pending", "notes": ""} for case in cases
    ]})
    rows = []
    _write_batch(directory, rows)
    for case in cases:
        trace = await collect_trace(case, index, encoder, generate, k=k, max_context_chars=max_context_chars)
        _save_json(records / f"{case['case_id']}.json", trace)
        row = grade_trace(trace)
        rows.append(row)
        _save_json(records / f"{case['case_id']}.json", row)
        _write_batch(directory, rows)
        print(f"{case['case_id']}: automatic={row['automated_passed']}, final={row['final_status']}")
    return {"summary": _write_batch(directory, rows), "rows": rows}


def apply_reviews(report_path: str | Path, reviews_path: str | Path) -> dict:
    """已提供：离线合并人工复核并重新评分，不请求任意模型。

    verdict 可为 pending/supported/hallucination/incomplete；未知或重复 case_id 报错。
    允许只复核一部分条目，其他保留原状态。同一报告重跑生成后须重新核对回答，
    不能直接复用旧回答的人工结论。
    """
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reviews = json.loads(Path(reviews_path).read_text(encoding="utf-8"))["reviews"]
    existing_ids = {row["case_id"] for row in report["rows"]}
    selected = {}
    for review in reviews:
        _check_review(review)
        case_id = review.get("case_id")
        if case_id not in existing_ids or case_id in selected:
            raise ValueError("人工复核出现未知或重复 case_id")
        selected[case_id] = {"verdict": review["verdict"], "notes": review["notes"]}
    rows = [grade_trace(row, selected.get(row["case_id"], row.get("review"))) for row in report["rows"]]
    for row in rows:
        _save_json(report_path.parent / "cases" / f"{row['case_id']}.json", row)
    return {"summary": _write_batch(report_path.parent, rows, report_path.name), "rows": rows}


if __name__ == "__main__":
    import argparse
    import asyncio
    import os

    parser = argparse.ArgumentParser(description="Day6：15 条 RAG 评估与离线人工复核")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="显式调用 Ollama + DeepSeek 批量评估")
    run.add_argument("--real-models", action="store_true", help="确认调用真实模型及产生费用")
    run.add_argument("--cases", default=str(DEFAULT_CASES))
    run.add_argument("--index", default="rag/artifacts/day4/local_model/knowledge_index.json")
    run.add_argument("--output", default="rag/artifacts/day6/real_model")
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--context-chars", type=int, default=9000)
    review = commands.add_parser("review", help="合并人工复核，不调用模型")
    review.add_argument("--report", default="rag/artifacts/day6/real_model/evaluation_report.json")
    review.add_argument("--reviews", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not args.real_models:
            parser.error("真实评估需要显式添加 --real-models；先完成三个步骤并运行离线验收")
        cases, index = load_cases(args.cases), load_index(args.index)
        encoder = OllamaEncoder(index.model_name, os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                                query_instruction=index.query_instruction)
        generate = DeepSeekGenerator(os.environ.get("RAG_GENERATION_MODEL", "deepseek-flash"))
        result = asyncio.run(run_evaluations(cases, index, encoder, generate, args.output,
                                              k=args.k, max_context_chars=args.context_chars))
    else:
        result = apply_reviews(args.report, args.reviews)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
