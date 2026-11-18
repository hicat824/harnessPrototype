"""Week9 Day5：在检索候选阶段执行知识发布状态、版本与访问权限规则。"""

from rag.eval_dataset import reference_is_eligible
from rag.hybrid_search import make_search_retriever


def _allow_candidate(metadata: dict, request: dict) -> bool:
    """复用 Day1 规则判断当前片段是否可以参与检索。

    metadata 是来自本地可信知识索引的文档元数据，含 version、
    knowledge_status、engine_code 和 required_scopes。
    request 沿用 case.input，含问题、精确 filters 及由调用方提供的
    可信 access_context。授权使用 granted_scopes，问题正文不参与授权。

    返回 True 仅当状态为 current/historical、所有显式 filters 精确匹配，
    且文档的全部 required_scopes 已获授予。filters 未指定版本或状态时，
    两种已发布状态都可以参与，用于跨版本对照。

    Day1 的 reference_is_eligible 统一检查这三项。校验异常会向调用方传播，
    避免在权限或输入校验出错后继续检索。
    """
    return reference_is_eligible(metadata, request)


def make_policy_retriever(store, strategy: str = 'hybrid', *, candidate_k: int = 20,
                          rank_constant: int = 60):
    """构造 retrieve(request, k)，结果沿用 Day3 的 [{score, chunk, retrieval}]。

    在 dense/keyword 评分和截取 Top-K 前执行策略，空候选直接返回 []。
    再用该结果调用 Day4 重排或组装 Context。调用方需要提供可信授权上下文，
    本函数消费已授予的权限，不负责认证用户，也不修改索引或输入。
    """
    return make_search_retriever(store, strategy, candidate_k=candidate_k,
                                 rank_constant=rank_constant, candidate_filter=_allow_candidate)
