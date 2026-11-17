"""Week9 Day1：冻结评估数据，校验参考证据，并分离运行输入与评分参考。

两个关键步骤练习参考适用性与运行输入隔离；不调用模型，不建立索引，也不计算 Day2 的检索指标。
知识正文继续复用 Week7 的 manifest 加载器。逐项说明见 rag/week9.md#day1。
"""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from rag.preprocessing import load_documents, read_manifest, resolve_source_path


DEFAULT_DATASET = Path('/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality')
CATEGORIES = ('common', 'ambiguous', 'unknown', 'historical', 'permission')
REFUSAL_REASONS = ('missing_knowledge', 'realtime_required', 'permission_denied', 'unpublished_knowledge')
FILTER_KEYS = {'engine_code', 'version', 'knowledge_status'}


def _text(value, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} 必须是非空白字符串')


def _strings(value, name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f'{name} 必须是列表')
    for item in value:
        _text(item, name)
    if len(value) != len(set(value)):
        raise ValueError(f'{name} 不能重复')


def _check_input(request: dict) -> None:
    """已提供：格式校验，不把正文中的身份声明当作可信授权。"""
    if not isinstance(request, dict) or request.keys() != {'question', 'filters', 'access_context'}:
        raise ValueError('input 只能包含 question、filters、access_context')
    _text(request['question'], 'question')
    filters = request['filters']
    if not isinstance(filters, dict) or not filters.keys() <= FILTER_KEYS:
        raise ValueError('filters 仅支持 engine_code、version、knowledge_status')
    for name, value in filters.items():
        _text(value, name)
    access = request['access_context']
    if not isinstance(access, dict) or access.keys() != {'subject_id', 'granted_scopes'}:
        raise ValueError('access_context 需要 subject_id 和 granted_scopes')
    _text(access['subject_id'], 'subject_id')
    _strings(access['granted_scopes'], 'granted_scopes')


def _check_case(case: dict) -> None:
    """已提供：检查题目和答案的契约，不尝试用关键词自动评分语义。"""
    if not isinstance(case, dict) or case.keys() != {'case_id', 'category', 'input', 'expected'}:
        raise ValueError('用例需要 case_id、category、input、expected')
    if not isinstance(case['case_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', case['case_id']):
        raise ValueError('case_id 只能包含字母、数字、下划线和连字符')
    if case['category'] not in CATEGORIES:
        raise ValueError('未知的 category')
    _check_input(case['input'])
    expected = case['expected']
    fields = {'status', 'doc_groups', 'evidence', 'acceptable_points', 'refusal_condition',
              'refusal_reason', 'forbidden_doc_ids'}
    if not isinstance(expected, dict) or expected.keys() != fields:
        raise ValueError('expected 字段不完整或包含未知字段')
    if expected['status'] not in ('answered', 'insufficient'):
        raise ValueError('status 必须为 answered 或 insufficient')
    groups = expected['doc_groups']
    if not isinstance(groups, list):
        raise ValueError('doc_groups 必须是列表')
    seen = set()
    for group in groups:
        _strings(group, '文档组', allow_empty=False)
        if seen.intersection(group):
            raise ValueError('同一文档不能在多个文档组重复计分')
        seen.update(group)
    _strings(expected['acceptable_points'], 'acceptable_points', allow_empty=False)
    _strings(expected['forbidden_doc_ids'], 'forbidden_doc_ids')
    evidence = expected['evidence']
    if not isinstance(evidence, list):
        raise ValueError('evidence 必须为列表')
    for ref in evidence:
        if not isinstance(ref, dict) or ref.keys() != {'doc_id', 'segment_id', 'quote'}:
            raise ValueError('evidence 需要 doc_id、segment_id、quote')
        for key, value in ref.items():
            _text(value, key)
    if expected['status'] == 'answered':
        if not groups or not evidence:
            raise ValueError('answered 需要目标文档组和原文证据')
        if expected['refusal_condition'] is not None or expected['refusal_reason'] is not None:
            raise ValueError('answered 的拒答字段必须为 null')
    else:
        _text(expected['refusal_condition'], 'refusal_condition')
        if expected['refusal_reason'] not in REFUSAL_REASONS:
            raise ValueError('insufficient 必须说明合法的拒答原因')
        if groups:
            raise ValueError('本日拒答题无目标答案文档，doc_groups 应为空列表')


def load_benchmark(dataset_dir: str | Path = DEFAULT_DATASET) -> dict:
    """已提供：读取 eval_cases.json，检查结构和唯一 ID，不执行两个练习函数。

    输入：包含 manifest.json、knowledge/、eval_cases.json 的目录。
    输出：原始评估字典，含 schema_version、dataset_id、synthetic、cases 等字段。
    文件、JSON 错误原样抛出；契约错误抛 ValueError。此步骤不证明参考答案正确。
    """
    data = json.loads((Path(dataset_dir) / 'eval_cases.json').read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data['schema_version'] != 1 or data.get('synthetic') is not True):
        raise ValueError('需要 schema_version=1 的合成评估集')
    _text(data.get('dataset_id'), 'dataset_id')
    if not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('cases 必须是非空列表')
    ids = set()
    for case in data['cases']:
        _check_case(case)
        if case['case_id'] in ids:
            raise ValueError('case_id 不能重复')
        ids.add(case['case_id'])
    return data


def reference_is_eligible(document: dict, request: dict) -> bool:
    """练习 1：判断一个参考文档是否适用于本题，返回 bool，不修改输入。

    document：一条 manifest 记录，使用 version、knowledge_status、engine_code、
        required_scopes；scope 只是主题标签，不能用它判断访问权限。
    request：case['input']，含 filters 和可信 access_context.granted_scopes。

    判断条件：文档是 current/historical；每个指定 filter 与文档对应字段精确相等；
    required_scopes 中的全部权限都已授予。无 filter 不增加条件；无访问要求表示公开。
    例如 required=['ops','audit']、granted=['ops'] 仍不可用。
    这是评估参考的校验，不是对现有 Retriever 的权限改造。
    """
    _check_input(request)
    _strings(document.get('required_scopes'), 'required_scopes')
    # 学习点 W9D1-1：已发布状态与本题的每个过滤条件必须同时满足。
    # all() 会逐项求值；filters 为空时 all([]) 为 True，表示不增加过滤限制。
    metadata_ok = (
        document.get('knowledge_status') in ('current', 'historical')
        and all(document.get(key) == value for key, value in request['filters'].items())
    )

    # 权限从可信 access_context 读取，不从问题正文或 request 顶层获取。
    # 子集判断允许用户有额外权限，也不受列表顺序影响；空要求表示公开文档。
    required_scopes = set(document['required_scopes'])
    granted_scopes = set(request['access_context']['granted_scopes'])
    access_ok = required_scopes.issubset(granted_scopes)
    return metadata_ok and access_ok


def build_case_input(case: dict) -> dict:
    """练习 2：从完整用例提取一份独立复制的运行输入。

    输入：含 case_id/category/input/expected 的完整评估用例。
    输出：仅含 question、filters、access_context 的新字典；嵌套列表/字典也要隔离。
    不允许出现 expected、参考摘录、答案要点、拒答条件或禁止文档 ID。
    后续程序用 question 检索，用 filters/权限过滤；不要求将整个字典发送给 LLM。
    比如修改返回值的 granted_scopes，不应修改原用例或其他请求。
    """
    _check_case(case)
    # 学习点 W9D1-2：只提取 input；深拷贝同时隔离 filters 和权限列表。
    request = copy.deepcopy(case.get('input'))
    _check_input(request)
    return request


def validate_benchmark(dataset_dir: str | Path = DEFAULT_DATASET) -> dict:
    """已提供：验证整份题库，返回结构/来源报告，失败时抛 ValueError。

    检查 25～30 题与五类覆盖、知识文件哈希、文档组、逐字摘录、版本及权限。
    fingerprint 同时覆盖 manifest、题库和知识文件，供后续冻结基线比较。
    报告不包含检索指标或真实回答成绩，不能作为 RAG 质量通过结论。
    """
    root = Path(dataset_dir)
    data, documents = load_benchmark(root), read_manifest(root)
    if not 25 <= len(data['cases']) <= 30:
        raise ValueError('Day1 评估集需要 25～30 道题')
    categories = Counter(case['category'] for case in data['cases'])
    if set(categories) != set(CATEGORIES):
        raise ValueError('评估集必须覆盖五类问题')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('dataset_id') != data['dataset_id'] or manifest.get('synthetic') is not True:
        raise ValueError('知识与评估集的 dataset_id/synthetic 不一致')
    by_id = {document['doc_id']: document for document in documents}
    hashes = {}
    for document in documents:
        _strings(document.get('required_scopes'), 'required_scopes')
        raw = resolve_source_path(root, document['source']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if document.get('sha256') != digest:
            raise ValueError(f"知识文件哈希不一致：{document['doc_id']}")
        hashes[document['source']] = digest
    units = {(unit.doc_id, unit.segment_id): unit.text for unit in load_documents(root)}
    for case in data['cases']:
        expected = case['expected']
        refs = expected['evidence']
        positive = {ref['doc_id'] for ref in refs}
        groups = {doc_id for group in expected['doc_groups'] for doc_id in group}
        forbidden = set(expected['forbidden_doc_ids'])
        if not (positive | groups | forbidden) <= by_id.keys():
            raise ValueError(f"未知参考文档：{case['case_id']}")
        if forbidden.intersection(positive | groups):
            raise ValueError(f"参考文档与禁止来源冲突：{case['case_id']}")
        for doc_id in positive | groups:
            if not reference_is_eligible(by_id[doc_id], case['input']):
                raise ValueError(f"参考文档版本/状态/权限不适用：{case['case_id']} / {doc_id}")
        for group in expected['doc_groups']:
            if not positive.intersection(group):
                raise ValueError(f"目标文档组缺少原文证据：{case['case_id']}")
        for ref in refs:
            text = units.get((ref['doc_id'], ref['segment_id']))
            if text is None or ref['quote'] not in text:
                raise ValueError(f"参考摘录不属于指定正文单元：{case['case_id']}")
    # 文件路径排序后再计算整体摘要，避免清单顺序意外影响文件拼接。
    for name in ('manifest.json', 'eval_cases.json'):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode('utf-8')).hexdigest()
    return dict(dataset_id=data['dataset_id'], fingerprint=fingerprint, case_count=len(data['cases']),
                document_count=len(documents), category_counts=dict(categories),
                status_counts=dict(Counter(case['expected']['status'] for case in data['cases'])),
                checked_evidence_count=sum(len(case['expected']['evidence']) for case in data['cases']),
                file_hashes=hashes, validation_passed=True, model_called=False,
                scope='数据结构、来源和参考适用性校验；未验证检索质量或回答语义。')


def write_dataset_report(dataset_dir: str | Path, output_dir: str | Path) -> dict:
    """已提供：完整校验后落盘两份产物；输出目录不能覆盖知识或评分输入文件。

    返回 dataset_report.json 的内容。runtime_inputs.json 只保存 case_id 与运行输入，
    case_id 用于报告关联，不是参考答案。任何校验失败都不会先输出成功报告。
    """
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output == root or output == root / 'knowledge' or root / 'knowledge' in output.parents:
        raise ValueError('output_dir 不能覆盖知识或评分输入目录')
    report = validate_benchmark(root)
    inputs = [dict(case_id=case['case_id'], input=build_case_input(case))
              for case in load_benchmark(root)['cases']]
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (('dataset_report.json', report), ('runtime_inputs.json', inputs)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report
