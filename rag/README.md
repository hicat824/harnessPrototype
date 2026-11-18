# RAG 学习文档

Week7 Day1～Day7 已完成，完整学习内容已汇总到 [week7.md](week7.md)。

- [系统工作流与 Mermaid 架构图](week7.md#workflow)
- [完成记录与验收边界](week7.md#completion)
- [统一运行命令](week7.md#commands)
- [Day1：RAG 概念与知识数据模型](week7.md#day1)
- [Day2：文档加载、清洗与分块](week7.md#day2)
- [Day3：Embedding、余弦相似度与 Top-K](week7.md#day3)
- [Day4：持久化索引与 Retriever](week7.md#day4)
- [Day5：Context 组装与有依据回答](week7.md#day5)
- [Day6：最小 RAG Evals 与错误分析](week7.md#day6)
- [Day7：端到端最小 RAG 周验收](week7.md#day7)

## Week8：LangChain 版知识库组件（已完成）

- [本周完成总结与框架选型结论](../langchain_learning/week8.md#week8-completion)

- [Day4：Document、Loader、Text Splitter 与 Embeddings（已完成）](../langchain_learning/week8.md#day4)
- 实现：[langchain_preparation.py](langchain_preparation.py)，测试：[test_week8_day4.py](../tests/test_week8_day4.py)。
- 沿用 Week7 同一知识集；实现放在 `rag/`，默认使用主项目 `.venv`。
- [Day5：Vector Store、Retriever 与检索回答链（已完成）](../langchain_learning/week8.md#day5)
- Day5 实现：[langchain_rag.py](langchain_rag.py)，测试：[test_week8_day5.py](../tests/test_week8_day5.py)。
- [Day6：将 Retriever 接入 Agents SDK（已完成）](../langchain_learning/week8.md#day6)
- Day6 实现：[sdk_rag.py](sdk_rag.py)，测试：[test_week8_day6.py](../tests/test_week8_day6.py)。
- [Day7：原生 Python 与 LangChain RAG 对照周验收（已完成）](../langchain_learning/week8.md#day7)
- Day7 实现：[comparison.py](comparison.py)，测试：[test_week8_day7.py](../tests/test_week8_day7.py)。

## Week9：RAG 质量优化与评估（Day1～Day5 已完成）

- [Day1：固定评估集、参考证据与输入隔离](week9.md#day1)。
- 实现：[eval_dataset.py](eval_dataset.py)，测试：[test_week9_day1.py](../tests/test_week9_day1.py)。
- 测试数据：`/media/xumaozhou/data/AI/local_agent_test/week9_rag_quality/`；30 道题、21 份合成知识。
- 参考适用性与运行输入隔离已完成；Day1 完整离线验收 **49 passed**。
- [Day2：Recall@K、MRR 与召回诊断](week9.md#day2)；实现：[retrieval_metrics.py](retrieval_metrics.py)，测试：[test_week9_day2.py](../tests/test_week9_day2.py)。
- Day2 指标与真实 Ollama 基线已验收；离线 **33 passed、1 skipped**，真实 30 题已记录，K=5 时平均 Recall 为 100%。版本/权限问题保留为后续优化基线。
- [Day3：关键词、精确标识符与 RRF 混合检索](week9.md#day3)；实现：[hybrid_search.py](hybrid_search.py)，测试：[test_week9_day3.py](../tests/test_week9_day3.py)。
- Day3 关键词、RRF 与标识符约束已通过离线验收，真实三策略质量尚待核验。
- [Day4：重排、来源内去重与 Context 预算（已完成）](week9.md#day4)。实现见 [reranking.py](reranking.py)，测试见 [test_week9_day4.py](../tests/test_week9_day4.py)。
- Day4 已完成离线测试、Ollama 30 题检索和 DeepSeek 3 题重排对照。两道可计分题的来源覆盖率保持 100%，原文复核确认重排选入了更直接的证据。
- [Day5：版本、状态与权限过滤（已完成）](week9.md#day5)。实现见 [knowledge_policy.py](knowledge_policy.py)，测试见 [test_week9_day5.py](../tests/test_week9_day5.py)。
- Day5 真实 Ollama 对照中，候选和 Context 含不合规来源的题数均从 4 降到 0，22 道可计分题的 Context Recall 保持 100%。下一阶段为 Day6 输出评估与错误归因。

本文件用于目录导航。Week7 学习内容维护在 `week7.md`，Week8 维护在 `langchain_learning/week8.md`，Week9 维护在 `week9.md`；各文档均按测试分别说明数据来源、目标和产物。
