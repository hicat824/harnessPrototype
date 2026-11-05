# Week7 Day3：Embedding、余弦相似度与 Top-K

本日将文本转换成向量，理解余弦相似度，排序取 Top-K，并输出相似度热力图。
实现位于 `rag/embeddings.py`，测试位于 `tests/test_rag_embeddings.py`。

> 学习状态（2026-11-05）：**Week7 Day3 已完成，当前进入 Day4。**
> 离线验收通过；真实 Ollama 模型验收由用户运行并确认通过。
> 本地报告记录：`qwen3-embedding:0.6b`，64 个 Chunk，1024 维向量。

## 学习目标与验收边界

- 能解释“文本 → 向量 → 相似度 → 排名”的处理过程。
- 能用手算样例验证同向、垂直、反向及向量长度对余弦相似度的影响。
- 使用一个实际本地 Embedding 模型编码问题与 Day2 Chunk，保存向量、分数和来源。
- 查看热力图与 Top-K 原文，说明语义匹配与字面关键词匹配的区别。

固定向量测试通过只说明计算与流程正确；完成 Day3 还需要运行真实模型实验，并人工检查语义相关性。
本日不实现持久化向量索引，不调用生成式 LLM，不根据二维降维坐标做检索。

## 三个关键原理回顾

三个关键表达式均已完成，复习时按下表定位。

| 位置 | 关键变量 | 当前实现 | 关键原理 |
|---|---|---|---|
| `embed_texts` | `encode`、`texts` | 一次调用 `encode(texts)` | 批量编码；每条文本对应一个向量 |
| `cosine_similarity` | `left_unit`、`right_unit` | 对应分量乘积求和 | 两个单位向量的点积等于余弦相似度 |
| `top_k` | `scores = [(行号, 分数), ...]` | `sorted` 排序后切片 | 分数降序；同分按行号升序，随后取前 K 项 |

各函数 docstring 已说明参数、返回值、异常与调用方式。输入检查、Ollama 适配器、单位化、矩阵构造、绘图、JSON 保存和测试替身均已提供。
`embed_texts` 只需把 `texts` 传给已选好的编码方法，不传模型 SDK 的专用选项。

## 数据来源与流程

| 数据 | 来源 | 用途 |
|---|---|---|
| 四个人工二维向量及手算矩阵 | 测试内 `VECTORS` / `EXPECTED_MATRIX` | 校验数学计算与绘图；不是模型输出 |
| 四句合成中文及对应固定向量 | 测试内 `FixedEncoder` | 校验完整流程和来源绑定；不证明语义效果 |
| Day2 Chunk | 已有 Week7 样例目录；运行时复用 Day2 重新加载和分块 | 真实模型实验，不依赖 `/tmp` 中的旧产物 |
| 问题 | 测试内的“任务查不到时应该先检查什么？”或命令行 `--query` | 编码后计算与 Chunk 的相似度 |

默认样例目录：`/media/xumaozhou/data/AI/local_agent_test/week7_rag_basics`。
可通过 `WEEK7_RAG_DATASET` 修改测试数据目录。

```text
Day2 加载和分块 → Chunk 正文 → encode_document → Chunk 向量
                                     ↓               ↓
问题文本 → encode_query → 问题向量 → 余弦相似度 → Top-K + 原文来源

Chunk 向量两两比较 → N×N 相似度矩阵 → PNG 热力图
```

问题与正文使用**同一个模型**。短问题检索较长片段属于非对称检索场景，因此保留 `encode_query` / `encode_document` 两个入口。
本日固定使用 `qwen3-embedding:0.6b`；按 [Qwen 官方示例](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage)，适配器只给问题添加以下格式，正文原样发送：

```text
Instruct: Given a user question, retrieve relevant passages that answer the question
Query:原始问题
```

这段说明由 `OllamaEncoder` 组装，原始问题和 Chunk 不会被修改。若以后更换模型，应按新模型要求检查说明格式。

## 安装与运行

以下命令均在仓库根目录执行。画图使用 Matplotlib；HTTP 请求复用项目已有的 `httpx`。真实模型由本地 Ollama 服务运行，无须安装 Sentence Transformers 或 Ollama Python SDK。

### 1. 运行框架自检与人工向量热力图预览

```bash
.venv/bin/python -m pip install 'matplotlib>=3.7,<4'
.venv/bin/python -B -m pytest tests/test_rag_embeddings.py -q -s -p no:cacheprovider
```

默认运行框架和 HTTP Mock 自检；算法和真实模型测试需通过后面的环境变量显式启用。
热力图保存在 `rag/artifacts/day3/preview/similarity_heatmap.png`，旁边的 `preview_data.json` 记录手写向量与矩阵。

### 2. 运行完整离线验收

```bash
RUN_WEEK7_DAY3=1 .venv/bin/python -B -m pytest tests/test_rag_embeddings.py -q -s -p no:cacheprovider
```

此命令使用固定向量，不下载模型、不联网。完整流程产物保存在 `rag/artifacts/day3/offline_fixture/`。
只检查某一原理可添加 `-k cosine_similarity`、`-k top_k` 或 `-k embed_texts`。

### 3. 检查本地 Ollama 服务

你已经通过 `ollama pull qwen3-embedding:0.6b` 拉取模型，无须再下载其他模型。
确认 Ollama 服务已启动，且能看到该模型：

```bash
ollama list
```

如果服务尚未运行，在另一个终端执行 `ollama serve`；已有服务时不重复启动。
默认访问 `http://localhost:11434`。代码使用 [Ollama `/api/embed` 接口](https://docs.ollama.com/api/embed)，一次请求传入一个文本列表，按返回顺序读取 `embeddings`。
请求设置 `truncate=false`，超出模型上下文时直接报错，不静默截断正文。

### 4. 运行真实模型验收

```bash
RUN_WEEK7_DAY3=1 RUN_WEEK7_DAY3_MODEL=1 \
.venv/bin/python -B -m pytest tests/test_rag_embeddings.py -q -s -k local_model_experiment -p no:cacheprovider
```

该测试连接本地 Ollama，不调用公网模型，不需要 API Key。硬件调度由 Ollama 管理。
产物保存在 `rag/artifacts/day3/local_model/`。模型缺失、连接失败、HTTP 错误、超时或响应格式不正确时直接失败，不自动拉取模型，也不退回固定向量。

可选环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 服务根地址 |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | 已拉取的模型名称 |
| `OLLAMA_EMBED_TIMEOUT` | `120` | 每次批量请求的超时秒数；首次加载较慢时可增大 |

也可直接运行完整实验：

```bash
.venv/bin/python -B -m rag.embeddings \
  --model qwen3-embedding:0.6b \
  --ollama-url http://localhost:11434 \
  --timeout 120 \
  --query '任务查不到时应该先检查什么？' \
  --output rag/artifacts/day3/local_model
```

测试产物根目录可通过 `RAG_DAY3_OUTPUT` 修改；同一目录下再次运行会覆盖同名结果。
`rag/artifacts/` 已设置为 Git 忽略，不会混入代码提交。Ollama 继续使用自己的模型存储目录。

## 如何阅读结果

| 文件 | 内容 |
|---|---|
| `similarity_heatmap.png` | Chunk×Chunk 余弦相似度；红色较高、蓝色较低，固定色标 -1～1 |
| `similarity_matrix.json` | 与图片相同顺序的短标签及完整数值矩阵 |
| `embedded_chunks.json` | 每个短标签对应的 Chunk、向量、`doc_id`、来源、标题和原文位置 |
| `day3_report.json` | 数据模式、模型名称、维数、问题向量、Top-K 分数及完整 Chunk 信息；Ollama 实验另记录服务地址和查询说明 |

热力图使用 `C01` 等短标签，避免依赖本地中文字体；用 `embedded_chunks.json` 找回中文原文。
对角线应接近 1，矩阵应对称。高相似度不保证规则适用，特别是历史版本与当前版本可能都很相似。
负分表示向量方向关系，不能直接解释成“业务含义相反”。

下列每节对应一个独立测试函数。

## 测试一：人工向量热力图预览

函数：`test_provided_heatmap_preview`，默认执行。

- **数据来源**：四个人工二维向量及手算矩阵。
- **流程**：直接把已知矩阵传给绘图函数，不依赖余弦计算函数。
- **检查点**：生成有效 PNG，数据文件明确标为人工向量，不宣称真实语义效果。
- **输出**：`preview/similarity_heatmap.png`、`preview/preview_data.json`。

## 测试二：热力图参数检查

函数：`test_provided_heatmap_rejects_bad_input`，默认执行。

- **数据来源**：测试内空矩阵、标签不匹配或重复、NaN、越界值和错误文件后缀。
- **流程**：调用绘图函数并捕获异常。
- **检查点**：非法参数抛出 `ValueError`。
- **输出**：pytest 结果，无正式产物。

## 测试三：Ollama 配置检查

函数：`test_provided_ollama_rejects_bad_config`，默认执行。

- **数据来源**：空模型名、无效服务地址、非法超时和说明参数。
- **流程**：构造 `OllamaEncoder`，不连接服务。
- **检查点**：非法配置抛 `ValueError`，不开始请求。
- **输出**：pytest 结果。

## 测试四：问题与正文编码入口

函数：`test_embed_texts_calls_correct_encoder`。

- **数据来源**：两句中文与 `FixedEncoder` 的固定向量。
- **流程**：分别以 query、document 模式批量编码。
- **检查点**：选对入口；只传文本列表；模型只被调用一次；条数、顺序正确，输入不变。
- **输出**：pytest 结果。

## 测试五：空文本批次与非法输入

函数：`test_embed_texts_empty_and_bad_inputs`。

- **数据来源**：空批次、空白文本、非字符串、非法 kind。
- **流程**：调用编码函数，检查测试替身的调用记录。
- **检查点**：空批次返回空列表；非法输入报错；两者都不调用模型。
- **输出**：pytest 结果。

## 测试六：模型输出校验

函数：`test_embed_texts_rejects_bad_model_output`。

- **数据来源**：固定编码器返回错误条数、零向量、NaN、Infinity、bool 或字符串。
- **流程**：编码后由已提供代码检查返回值。
- **检查点**：拒绝无法用于余弦计算的结果，不继续输出成功报告。
- **输出**：pytest 结果。

## 测试七：维数不齐与推理异常

函数：`test_embed_texts_rejects_mixed_dimensions_and_propagates_error`。

- **数据来源**：不同维数的固定向量及主动抛错的模型替身。
- **流程**：分别调用编码函数。
- **检查点**：维数不齐抛 `ValueError`，模型异常原样传播。
- **输出**：pytest 结果。

## 测试八：余弦相似度手算对照

函数：`test_cosine_similarity_matches_hand_calculation`。

- **数据来源**：同向、垂直、反向、长度不同及非整数分数的五组向量。
- **流程**：调用你填写的余弦计算，与手算值比较。
- **检查点**：结果约为 1、0、-1 或指定值；不把模长当相似度；输入不变。
- **输出**：pytest 结果。

## 测试九：非法向量

函数：`test_cosine_similarity_rejects_invalid_vectors`。

- **数据来源**：空向量、零向量、维数不同、非有限值及 bool。
- **流程**：调用余弦函数。
- **检查点**：明确抛 `ValueError`，不返回 NaN 或看似有效的分数。
- **输出**：pytest 结果。

## 测试十：两两相似度矩阵

函数：`test_pairwise_matrix_is_symmetric_and_matches_expected`。

- **数据来源**：四个人工向量与完整手算答案。
- **流程**：计算 4×4 矩阵，逐行检查。
- **检查点**：数值正确、矩阵对称、对角线约为 1；空批次得到空矩阵。
- **输出**：pytest 结果。

## 测试十一：Top-K 排名

函数：`test_top_k_orders_scores_and_breaks_ties_by_index`。

- **数据来源**：包含同向不同长度、垂直和反向的候选向量。
- **流程**：计算问题相似度，按你填写的规则排序取 K 项。
- **检查点**：降序、同分稳定、下标准确；K 超过候选数返回全部；空候选返回空列表；输入不变。
- **输出**：pytest 结果。

## 测试十二：K 参数边界

函数：`test_top_k_rejects_invalid_budget`。

- **数据来源**：0、负数、bool 和小数 K。
- **流程**：调用 Top-K 函数。
- **检查点**：抛 `ValueError`。
- **输出**：pytest 结果。

## 测试十三：离线完整流程

函数：`test_offline_pipeline_exports_sources_scores_and_heatmap`。

- **数据来源**：四句合成中文、测试侧构造的 Chunk 和固定编码器。
- **流程**：正文编码 → 问题编码 → 相似度 → Top-K → 矩阵 → 导出。
- **检查点**：已知排名与分数正确；向量、短标签与原文来源一一对应；输入不变。
- **输出**：`offline_fixture/` 下的四个标准产物。此处语义关系由测试人工设定，不是模型学出来的。

## 测试十四：真实本地模型实验

函数：`test_local_model_experiment`，必须另行显式启用。

- **数据来源**：Ollama 已拉取的 Qwen3 Embedding 模型、Day2 的 Week7 合成语料、固定中文问题。
- **流程**：重新生成 Chunk，经本地 `/api/embed` 编码正文与带检索说明的问题，计算并导出。
- **检查点**：实际向量有限且同维；来源保留；Top-K 数量与分数合法；矩阵对称、对角线约为 1。不同模型不强制同一语义分数或排名。
- **输出**：`local_model/` 下的四个标准产物。

## 测试十五：Ollama 请求格式与问题说明

函数：`test_provided_ollama_sends_batch_and_formats_query`，默认执行。

- **数据来源**：两句中文和 `httpx.MockTransport` 返回的固定向量。
- **流程**：直接调用适配器的两个入口，走实际 JSON 请求构造与响应解析。
- **检查点**：URL、POST 方法、模型名、批次顺序、超时、`truncate=false` 正确；只有问题添加说明，原输入不变。
- **输出**：pytest 结果，不连接真实服务。

## 测试十六：空批次与无效问题不请求服务

函数：`test_provided_ollama_empty_and_invalid_texts_never_request`，默认执行。

- **数据来源**：空批次、空白问题、非字符串和非列表输入。
- **流程**：直接调用适配器，若触发 HTTP 请求则测试失败。
- **检查点**：空批次返回空列表；无效原文直接报错，不能因添加说明而误通过检查。
- **输出**：pytest 结果。

## 测试十七：HTTP 错误传播

函数：`test_provided_ollama_propagates_http_error`，默认执行。

- **数据来源**：Mock 返回 404 和 500 JSON 错误。
- **流程**：请求一次并检查异常。
- **检查点**：保留 `HTTPStatusError` 和状态码，不重试、不自动下载、不返回人工向量。
- **输出**：pytest 结果。

## 测试十八：连接错误与超时

函数：`test_provided_ollama_propagates_transport_error`，默认执行。

- **数据来源**：Mock 分别抛 `ConnectError` 和 `ReadTimeout`。
- **流程**：调用一次批量编码。
- **检查点**：原异常类型传给调用方，不伪装成成功结果。
- **输出**：pytest 结果。

## 测试十九：HTTP 成功后的响应校验

函数：`test_provided_ollama_rejects_bad_response`，默认执行。

- **数据来源**：Mock 返回 HTTP 200，但内容缺字段、有 error、条数不符、零向量或非数字。
- **流程**：走实际响应结构与向量检查。
- **检查点**：非法结果抛 `ValueError`，HTTP 200 本身不代表编码成功。
- **输出**：pytest 结果。

## 测试二十：非法 JSON 响应

函数：`test_provided_ollama_rejects_invalid_json`，默认执行。

- **数据来源**：Mock 返回非 JSON 正文。
- **流程**：调用适配器并解析响应。
- **检查点**：保留 `JSONDecodeError`，不继续计算分数。
- **输出**：pytest 结果。

最后人工检查：打开热力图，选择一个高分片段对照原文，再查看问题的 Top-K，解释至少一个“词面不同但语义接近”或“词面相似但业务条件不同”的例子。测试通过不能代替这一步，也不意味着已完成后续检索质量验收。

绘图实现参考 [Matplotlib 热力图示例](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)。
