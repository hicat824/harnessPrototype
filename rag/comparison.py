"""Week8 Day7：同一知识快照下，对照原生 Python RAG 与 LangChain RAG。

已完成两条 RAG 路径调用和 Token 指标判断，不改写 Week7/Day5 的检索与回答逻辑。
报告区分执行成功、自动检查和人工语义复核，不用单次耗时决定框架优劣。
逐项测试说明见 langchain_learning/week8.md#day7。
"""

import ast
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.models.interface import Model

from rag.answering import Generate, ask_knowledge
from rag.evaluation import _check_case, check_versions, document_group_recall
from rag.langchain_rag import _save_json, _store_info, document_chunk, run_rag
from rag.retrieval import KnowledgeIndex, _encoder_config, _validate_index


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'langchain_learning/artifacts/day7'


def native_index_from_store(store) -> KnowledgeIndex:
    """已提供：从 Day5 向量库恢复原生索引，不重新读取文档或编码正文。

    两条路径使用同一份 ID、正文、元数据、向量和编码配置，排除建库差异。
    返回独立副本，后续查询不会改动原向量库。
    """
    _store_info(store)
    model, instruction = _encoder_config(store.embedding.encoder)
    documents = store.get_by_ids(list(store.store))
    index = KnowledgeIndex(model, instruction, [document_chunk(doc) for doc in documents],
                           [copy.deepcopy(store.store[doc.id]['vector']) for doc in documents])
    _validate_index(index)
    return index


def snapshot_id(index: KnowledgeIndex) -> str:
    """已提供：为正文、元数据、向量及模型配置计算快照标识，不把向量写进对照报告。"""
    text = json.dumps(asdict(index), sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class OneShotSDKGenerator:
    """已提供：两条 RAG 路径共用的单次生成适配器，保留 SDK 实际报告的 usage。

    输入仍为 Week7 的 list[dict] 消息，输出仍为 JSON 原文字符串。
    SDK 在此只执行无工具的一次生成；不走 Day6 的检索工具闭环。
    两边使用相同 Model、消息转换和生成设置，避免把不同调用适配器的差异算给检索框架。
    模型由调用方管理；离线用 ScriptedModel，真实用既有 DeepSeek 适配器。
    """

    def __init__(self, model: Model, *, model_name: str):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError('必须明确记录生成模型名称')
        self.backend, self.model = model, model_name
        self.last_usage = None

    async def __call__(self, messages: list[dict]) -> str:
        self.last_usage = None  # 失败时不能沿用上一条请求的用量。
        agent = Agent(name='ComparisonAnswer', model=self.backend, instructions=messages[0]['content'],
                      model_settings=ModelSettings(temperature=0))
        result = await Runner.run(agent, copy.deepcopy(messages[1:]), max_turns=1,
                                  run_config=RunConfig(tracing_disabled=True))
        usage = result.context_wrapper.usage
        self.last_usage = {name: getattr(usage, name)
                           for name in ('requests', 'input_tokens', 'output_tokens', 'total_tokens')}
        if not isinstance(result.final_output, str):
            raise ValueError('单次生成必须返回文本，结构和引用继续由已有回答层校验')
        return result.final_output


def token_delta(native_usage: dict | None, langchain_usage: dict | None) -> int | None:
    """计算 LangChain 减原生路径的 total_tokens；任一边未知则返回 None。

    输入：实际报告的 usage 字典或 None；字典缺少 total_tokens 也表示未知。
    输出：整数差值，允许为负数；未知返回 None，不能补成 0。真实的 0 仍应参与计算。
    仅接受非负整数计数，拒绝 bool；学习点 W8D7-3 是区分未知值与真实的零。
    """
    values = []
    for usage in (native_usage, langchain_usage):
        if usage is not None and not isinstance(usage, dict):
            raise ValueError('usage 必须为字典或 None')
        value = usage.get('total_tokens') if usage is not None else None
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('total_tokens 必须为非负整数或 None')
        values.append(value)
    native_total, langchain_total = values
    # 两边都明确报告计数才相减；真实的零不会被当成未知值。
    return None if (native_total is None or langchain_total is None) else langchain_total - native_total


async def _run_path(path: str, case: dict, index: KnowledgeIndex, store, generate: Generate,
                    max_context_chars: int) -> dict:
    """运行单条已有路径，保留实际消息、回答和失败阶段。

    case 只把 question、k、filters 传入运行层，参考答案只在后续评分时使用。
    输入的 generate 接收消息并返回 JSON 文本；可选 last_usage 提供实际用量。
    输出含 details（原 RAG 报告）、metrics、error。普通执行异常保存后返回 failed，
    便于批量继续；NotImplementedError 直接抛出，避免隐藏未实现的接口。
    """
    question, k, filters = case['question'], case.get('k', 5), copy.deepcopy(case['filters'])
    metrics = dict(generation_calls=0, generation_seconds=0.0, usage=None)
    stage, raw, messages = 'retrieval_or_context', None, []
    started = time.perf_counter()

    async def observed_generate(actual_messages):
        nonlocal stage, raw, messages
        stage = 'generation'
        messages = copy.deepcopy(actual_messages)
        metrics['generation_calls'] += 1
        generation_started = time.perf_counter()
        try:
            raw = await generate(actual_messages)
        finally:
            metrics['generation_seconds'] += time.perf_counter() - generation_started
            metrics['usage'] = copy.deepcopy(getattr(generate, 'last_usage', None))
        stage = 'answer_validation'
        return raw

    row = dict(path=path, execution_status='running', details=None, error=None,
               generation_model=getattr(generate, 'model', None), metrics=metrics)
    try:
        if path == 'native':
            # 学习点 W8D7-1：直接复用 Week7 的原生 RAG，使用同一编码器和生成适配器。
            report = await ask_knowledge(
                question, index, store.embedding.encoder, observed_generate,
                k=k, filters=filters, max_context_chars=max_context_chars,
            )
        elif path == 'langchain':
            # 学习点 W8D7-2：执行 Day5 的 Runnable 链，其他检索配置和预算保持一致。
            report = await run_rag(
                question, store, observed_generate,
                k=k, filters=filters, max_context_chars=max_context_chars,
            )
        else:
            raise ValueError('未知对照路径')
        row.update(execution_status='succeeded', details=report)
    except NotImplementedError:
        raise
    except Exception as error:
        row.update(execution_status='failed', error=dict(stage=stage, type=type(error).__name__))
    finally:
        metrics['elapsed_seconds'] = time.perf_counter() - started
        row.update(actual_messages=messages, raw_answer=raw)
    return row


def _grade(row: dict, case: dict, by_id: dict) -> dict:
    """已提供：复用 Week7 的文档组与版本检查，语义支持性仍需人工复核。"""
    if row['execution_status'] != 'succeeded':
        return dict(execution_ok=False, automatic_passed=False)
    details = row['details']
    retrieved = [by_id[key] for key in details['retrieved_chunk_ids']]
    selected = [item['chunk'] for item in details['context']['sources']]
    retrieval_recall = document_group_recall(case['expected_doc_groups'], retrieved)
    context_recall = document_group_recall(case['expected_doc_groups'], selected)
    checks = dict(execution_ok=True, retrieval_recall=retrieval_recall, context_recall=context_recall,
                  versions_ok=check_versions(retrieved + selected, case),
                  answer_status_ok=details['answer']['status'] == case['expected_status'])
    checks['automatic_passed'] = (checks['versions_ok'] and checks['answer_status_ok']
                                  and retrieval_recall in (None, 1.0) and context_recall in (None, 1.0))
    return checks


def code_profile() -> dict:
    """已提供：记录实现文件的行数/函数数，作为阅读导航，不把行数当代码质量评分。"""
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name in ('rag/retrieval.py', 'rag/langchain_rag.py', 'rag/answering.py'):
        text = (root / name).read_text(encoding='utf-8')
        functions = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(ast.parse(text)))
        result[name] = dict(physical_lines=len(text.splitlines()), functions=functions)
    return result


def _cell(value) -> str:
    return ('未提供' if value is None else str(value)).replace('|', '\\|').replace('\n', ' ')


def render_report(report: dict) -> str:
    """已提供：生成人可读的对照 Markdown；原始消息、原文与引用保留在 JSON。"""
    lines = ['# Week8 Day7：原生 Python 与 LangChain RAG 对照', '',
             f"数据模式：`{report['mode']}`；知识快照：`{report['snapshot_id']}`。", '',
             '耗时不含建库；两条路径顺序执行，可能受缓存和网络波动影响，不据此认定框架快慢。', '',
             '| 用例 | 路径 | 执行 | 自动检查 | 生成调用 | 总 Token | 耗时/秒 | 失败阶段 |',
             '|---|---|---|---|---|---|---|---|']
    for case in report['cases']:
        for name in ('native', 'langchain'):
            row = case[name]
            usage = row['metrics']['usage']
            tokens = usage.get('total_tokens') if isinstance(usage, dict) else None
            values = [case['case_id'], name, row['execution_status'], row['checks']['automatic_passed'],
                      row['metrics']['generation_calls'], tokens, round(row['metrics']['elapsed_seconds'], 4),
                      row['error']['stage'] if row['error'] else '—']
            lines.append('| ' + ' | '.join(_cell(value) for value in values) + ' |')
    lines += ['', '## 文件规模与职责', '',
              '| 文件 | 行数 | 函数数 |', '|---|---|---|']
    for name, info in report['code_profile'].items():
        lines.append(f"| {name} | {info['physical_lines']} | {info['functions']} |")
    lines += ['', '`rag/answering.py` 是双方共用的预算、消息和引用层，不应重复计算为某一方独有实现。', '',
              '## 阅读后补充的结论', '',
              '- 两种检索写法的差别，以及框架替你管理了哪些接口：待补充。',
              '- 哪种情况下保留手写实现，哪种情况下使用 LangChain 组件：待补充。',
              '- 实际回答是否被引用支持，实时问题是否正确拒答：待复核。', '',
              '离线替身只证明数据流；真实回答自动检查通过后仍需语义复核。未知用量保留为空，不按零计算。', '']
    return '\n'.join(lines)


async def run_comparison(cases: list[dict], store, generator_factory: Callable[[], Generate], *,
                          mode: str = 'offline_fixture', max_context_chars: int = 12000,
                          output_dir: str | Path = DEFAULT_OUTPUT / 'offline') -> dict:
    """批量对照并逐条保存，返回 report；每条路径使用 factory 新建的生成器。

    cases 沿用 Week7 格式；只有 question/k/filters 进入运行层，不向模型提供参考项。
    store 为同一 Day5 已建好的索引；不重新编码正文，每条有候选的路径各编码一次问题。
    mode 只能为 offline_fixture/real_model，明确区分脚本用量和真实用量。
    自动检查只覆盖执行、召回、版本与回答状态；真实语义结果标为 needs_review。
    单路径失败后仍执行另一条路径及后续问题；保留失败阶段，不伪造成功答案。
    """
    if mode not in ('offline_fixture', 'real_model') or not isinstance(cases, list) or not cases:
        raise ValueError('必须提供非空 cases 和明确的数据模式')
    ids = set()
    for case in cases:
        _check_case(case)
        if case['case_id'] in ids:
            raise ValueError('用例 ID 不能重复')
        ids.add(case['case_id'])
    index = native_index_from_store(store)
    by_id = {chunk.chunk_id: asdict(chunk) for chunk in index.chunks}
    output_dir = Path(output_dir)
    report = dict(mode=mode, snapshot_id=snapshot_id(index), embedding_model=index.model_name,
                  cases=[], code_profile=code_profile(), automatic_passed=False)
    for case in cases:
        pair = dict(case_id=case['case_id'], question=case['question'], k=case.get('k', 5),
                    filters=copy.deepcopy(case['filters']))
        for name in ('native', 'langchain'):
            row = await _run_path(name, case, index, store, generator_factory(), max_context_chars)
            row['checks'] = _grade(row, case, by_id)
            pair[name] = row
            _save_json(output_dir / 'cases' / f"{case['case_id']}_{name}.json", row)
        left, right = pair['native'], pair['langchain']
        succeeded = all(row['execution_status'] == 'succeeded' for row in (left, right))
        pair['same_ranked_ids'] = (left['details']['retrieved_chunk_ids'] == right['details']['retrieved_chunk_ids']) if succeeded else None
        pair['same_context'] = (left['details']['context'] == right['details']['context']) if succeeded else None
        pair['same_messages'] = (left['actual_messages'] == right['actual_messages']) if succeeded else None
        pair['token_delta'] = token_delta(left['metrics']['usage'], right['metrics']['usage'])
        names = [row['generation_model'] for row in (left, right)]
        pair['same_generation_model'] = names[0] == names[1] if all(isinstance(n, str) and n for n in names) else None
        pair['automatic_passed'] = all(row['checks']['automatic_passed'] for row in (left, right)) and pair['same_generation_model'] is True
        pair['semantic_review'] = ('needs_review' if mode == 'real_model' else 'fixture_only') if pair['automatic_passed'] else 'failed'
        report['cases'].append(pair)
        report['automatic_passed'] = all(row['automatic_passed'] for row in report['cases'])
        _save_json(output_dir / 'comparison_report.json', report)
        (output_dir / 'comparison_report.md').write_text(render_report(report), encoding='utf-8')
    return report
