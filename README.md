# ReviewInsight

基于 RAG 的跨境电商评论分析 Agent（v0.2.0）。上传评论后，用自然语言提问，答案中的统计数字全部由代码从 SQLite 生成，模型只负责意图归类、工具选择与输出「条目名 + 原文证据」。

核心能力：

- 缺点 / 优点 / 整体评价三类问题，统计口径由代码决定，模型不生成数字
- 语义检索 + 方面词表混合检索（RRF），证据可回溯到原文
- LangGraph 工作流 + 工具调用（查统计 / 查全部方面 / 检索证据）
- 多轮追问（Redis 保存最近 3 轮问题摘要）、SSE 流式输出
- 多产品隔离、Redis 缓存、进程内指标与护栏

启动后打开 **http://localhost:8000/demo**：单文件演示页，`product_id` 输入框默认 `shopping_phone`，可勾选多轮会话，查看答案、统计、证据来源与节点耗时。API 文档在 `/docs`，指标在 `/metrics`。

## 技术栈

| 层 | 组件 | 版本 / 说明 |
| --- | --- | --- |
| 语言 | Python | 3.11（Dockerfile 与 CI 一致） |
| API | FastAPI + Uvicorn | 0.141.1 / 0.52.0 |
| Agent | LangGraph | `>=1.2.11,<1.3.0` |
| LLM | LangChain + DeepSeek | langchain 1.3.17 / langchain-deepseek 1.1.0，模型 `deepseek-chat` |
| RAG | ChromaDB | chromadb 1.5.9 / langchain-chroma 1.1.0 |
| 嵌入 | sentence-transformers | 5.5.0，`shibing624/text2vec-base-chinese`，CPU + 归一化 |
| 关系存储 | SQLite | `reviews` 表保存评论、产品、方面极性 |
| 缓存 / 会话 | Redis 7 | redis-py 5.2.1 |
| 数据 | pandas | 3.0.5（CSV 解析） |
| 配置 | pydantic-settings | 2.15.0 |
| 部署 | Docker Compose | `api` + `redis`，命名卷持久化 |

依赖以 [requirements.txt](requirements.txt) 为准；测试依赖见 [requirements-dev.txt](requirements-dev.txt)。

## 快速开始

### 1. 准备环境

- Docker / Docker Compose
- 一个可用的 DeepSeek API Key
- 首次构建需要网络下载依赖与嵌入模型（约 400 MB），之后镜像内离线可用

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=sk-xxx
CHROMA_PERSIST_DIR=./chroma_db
LOG_LEVEL=INFO
```

`DEEPSEEK_API_KEY` 没有默认值，未填写时应用无法启动。

### 2. 启动服务

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

`/health` 返回 200 表示 Redis 正常；Redis 不可用时返回 503（`status: degraded`），容器会显示 unhealthy。

服务与端口：

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| api | 8000 | FastAPI，容器名 `reviewinsight-api-1` |
| redis | 6379 | 仅容器内网可达，未映射到宿主机 |

容器名由 Compose 项目名（目录名小写）生成。目录改名后，下文 `docker exec` 中的 `reviewinsight-api-1` 需同步替换。

### 3. 导入数据（首次必做）

仓库附带 200 条手机评论样例（`data/real/手机_200.csv`），可直接跑通完整链路；下文“当前实测结果”中的 830 条指标来自公开数据集导入环境（`--limit 800`），两者口径一致、绝对数字不同。任选一种方式准备数据：

```bash
# 方式一：公开数据集（首次需装开发依赖）
docker exec -w /app reviewinsight-api-1 pip install -q -r requirements-dev.txt
docker exec -e HF_HUB_OFFLINE=0 -w /app reviewinsight-api-1 \
  python scripts/import_dataset.py --category 手机 --limit 800 --product-id shopping_phone

# 方式二：上传仓库内的抽样数据
curl -X POST http://localhost:8000/api/v1/analysis/upload \
  -F "file=@data/real/手机_200.csv" -F "product_id=shopping_phone"
```

导入后访问 `/metrics`，`index.reviews` 与 `index.vectors` 应相等。

### 4. 打开页面

- 演示页：http://localhost:8000/demo
- Swagger：http://localhost:8000/docs
- 指标：http://localhost:8000/metrics
- 健康检查：http://localhost:8000/health

### 本地开发（可选）

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
# 先准备 Redis，例如：docker run -d -p 6379:6379 redis:7-alpine
uvicorn app.main:app --reload --port 8000
```

本地运行时 `CHROMA_PERSIST_DIR`、`DB_PATH` 默认指向仓库内目录；首次请求会下载嵌入模型（容器内已烘焙，无需下载）。

## 目录结构

```text
app/
  main.py            FastAPI 入口：/、/demo、/health
  core/              配置、Redis 客户端、嵌入模型单例、进程内指标
  routers/           analysis（上传 / 问答 / 流式）、metrics
  services/          agent（LangGraph）、stats（SQLite 统计与方面抽取）
                     vector（Chroma + SQLite）、cache（问答缓存）、memory（会话历史）
  utils/             txt / csv 解析
  static/            单文件演示页 index.html
scripts/             可信度评测、检索评测、压测、数据导入、方面重算
data/                init_db.py（建表）、real/（抽样评论）
tests/               pytest 用例（52 个 test_ 函数）
.github/workflows/   CI
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/analysis/upload` | 上传 `.txt` / `.csv`，表单字段 `file`、`product_id` |
| POST | `/api/v1/analysis/analyze` | 同步问答，返回 `answer` / `sources` / `stats` / `meta` |
| POST | `/api/v1/analysis/analyze/stream` | SSE 流式问答，逐节点推送事件 |
| GET | `/metrics` | 进程内累计指标与 `index` 条数 |
| GET | `/health` | Redis 探活，正常 200、降级 503 |
| GET | `/demo` | 单文件演示页 |
| GET | `/` | 运行状态 JSON |
| GET | `/docs` | Swagger UI |

请求示例：

```bash
curl -X POST http://localhost:8000/api/v1/analysis/upload \
  -F "file=@reviews.csv" -F "product_id=jd_phone"

curl -X POST http://localhost:8000/api/v1/analysis/analyze \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"电池有什么缺点？\",\"product_id\":\"default\"}"
```

上传新文件时返回 `count` 与 `index`（`reviews` / `vectors` 条数）；同一文件重复上传到同一产品时只返回「已处理过」，不再返回 `index`。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无（必填） | DeepSeek 密钥 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | 向量持久化目录，容器内 `/data/chroma_db` |
| `DB_PATH` | `./data/reviews.db` | SQLite 路径，容器内 `/data/reviews.db` |
| `PROCESSED_FILES_LOG` | `./processed_files.json` | 文件指纹日志，容器内 `/data/processed_files.json` |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `DEFAULT_RETRIEVAL_K` | `10` | 检索默认 top-k |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `localhost` / `6379` / `0` | Redis 连接；Compose 内为 `redis` |
| `REDIS_TTL` | `86400` | 问答缓存过期秒数 |
| `MAX_TOOL_ROUNDS` | `4` | 工具调用轮次上限 |
| `LLM_TIMEOUT_SECONDS` | `60` | 单次模型调用超时 |
| `SESSION_MAX_TURNS` | `3` | 会话保留轮数 |
| `SESSION_TTL_SECONDS` | `1800` | 会话过期秒数 |

Compose 额外注入 `REDIS_HOST`、`REDIS_PORT`、`CHROMA_PERSIST_DIR`、`DB_PATH`、`PROCESSED_FILES_LOG`、`HF_HUB_OFFLINE=1`。注意：`HF_HUB_OFFLINE=1` 由 Compose 设置，镜像本身只烘焙 `HF_ENDPOINT` / `HF_HOME`；直接 `docker run` 时不会自动离线。

## 数据一致性与索引

答案中的统计数字全部由代码从 SQLite 生成，模型只输出缺点名与原文证据。

上传新文件时返回 `index` 字段（`reviews` 与 `vectors` 条数），两者应始终相等；若历史数据出现重复向量，可执行：

```bash
docker exec -w /app reviewinsight-api-1 python -c "from app.services.vector_service import rebuild_vector_index; print(rebuild_vector_index())"
```

## 测试与评测

```bash
# 容器内安装测试依赖并运行单元测试
docker exec -w /app reviewinsight-api-1 pip install -q -r requirements-dev.txt
docker cp tests reviewinsight-api-1:/app/tests
docker exec -w /app reviewinsight-api-1 python -m pytest -q

# 运行可信度评测（统计一致率 / 前缀一致率 / 自行计数率 / 结构化成功率）
docker cp scripts reviewinsight-api-1:/app/scripts
docker exec -w /app reviewinsight-api-1 python scripts/eval.py
```

镜像只复制 `app/`，因此 `tests/`、`scripts/` 需要 `docker cp` 进去。

### CI

`.github/workflows/ci.yml` 在 push / PR 时安装依赖并运行 `pytest -q`。测试使用假 Agent 与 `FakeEmbeddings`，不调用 DeepSeek、不下载嵌入模型、不依赖 Redis，因此 runner 上不需要任何密钥或外部服务。

### 当前实测结果

测量环境：单 uvicorn worker，容器内已导入 `default` 30 条 + `shopping_phone` 800 条评论（合计 830 条）。其中 800 条由 `scripts/import_dataset.py --limit 800` 从公开数据集导入，仓库内只附带 200 条样例（`data/real/手机_200.csv`）。

- 用 200 条样例复现时，`index` 应为 `230/230`（`default` 30 + 200），下表的绝对数字会随之变化，但指标定义与评测脚本完全一致。
- 需要复现 830 条环境，按“导入数据 → 方式一”重新导入 800 条即可。
- 任何情况下都以 `/metrics` 的 `index` 与评测脚本输出为准，不要手写数字。

| 项目 | 结果 |
| --- | --- |
| 单元测试 | 52 passed |
| 统计一致率 | 100%（12/12，答案中的数字与 SQLite 直查一致） |
| 前缀一致率 | 100%（12/12，缺点 / 优点 / 整体评价三类都参与校验） |
| 自行计数率 | 0%（12/12，无“频次 / 约 X 次”） |
| 结构化成功率 | 100%（11/11，无对应内容的题目不计入） |
| 平均耗时 | 约 3.0 秒/题（整体评价会多次检索证据，比单方面问答略慢） |
| 向量索引 | reviews 830 / vectors 830（default 30 + shopping_phone 800） |

三类问题（缺点 / 优点 / 整体评价）都由 `finalize` 节点用代码拼接 SQLite 统计，模型只输出结构化条目与原文证据：

| 类型 | 模型输出 | 代码拼接 |
| --- | --- | --- |
| 缺点 | `{"defects": [{"name", "evidence"}]}` | 统计句 + 「具体缺点：」列表 |
| 优点 | `{"merits": [{"name", "evidence"}]}` | 统计句（正面/负面）+ 「具体优点：」列表 |
| 整体评价 | `{"summary", "merits", "defects"}` | 全方面汇总（逐方面提及/正面/负面/占比）+ 结论 + 优缺点列表 |

整体评价有两种口径：问题点了具体方面（“续航表现怎么样”）只统计该方面；没点方面（“整体评价怎么样”）才调用 `overall_statistics()` 汇总全部方面：

```text
根据数据库统计（共 30 条评论，各方面提及 21 次，正面 13 次、负面 8 次）：
- 电池：提及 17 次，正面 10 次、负面 7 次（负面占比 41%）
- 外观设计：提及 3 次，正面 3 次、负面 0 次（负面占比 0%）
- 发热：提及 1 次，正面 0 次、负面 1 次（负面占比 100%）
```

- 命中 Redis 缓存时，`stats` 与首次回答一起返回（缓存值存的是 `{answer, stats}`）。
- 问题未命中具体方面（如“屏幕”）时，答案会显式写明「按该产品提及最多的方面统计」，`stats.matched=false`。

## 意图归类

`classify_question` 改为「规则优先、模型兜底」，规则命中时不调用模型，同类问法不会漂移：

- 整体评价：整体 / 全面 / 总结 / 概括 / 优缺点 / 值得买 / 值得入手 …
- 缺点：缺点 / 短板 / 问题 / 吐槽 / 差评 / 不满意 …
- 优点：优点 / 亮点 / 好处 / 好评 / 满意 / 好在哪里 …
- 中性问法：怎么样 / 如何 / 表现 / 好吗 / 行不行 … → 整体评价（按问题所指方面或全部方面统计）
- 以上都未命中才调用模型（带 3 个示例），日志会标明来源：`节点1：问题类型为 [缺点]（规则）`
- 单方面统计由代码按「用户原问题」决定：模型在工具调用里改写问题、或多次查询不同方面，都不会改变最终答案的数字口径

## 检索评估（hit@5 / MRR@5）

`scripts/eval_retrieval.py` 在 800 条真实手机评论上对比三种检索策略：

```bash
docker exec -w /app reviewinsight-api-1 python scripts/eval_retrieval.py
```

| 方法 | hit@5 | MRR@5 |
| --- | --- | --- |
| 纯向量 | 0.90 | 0.733 |
| 生产词表（`ASPECT_KEYWORDS`） | 0.40 | 0.325 |
| 混合（向量 + 词表 + RRF） | 1.00 | 0.783 |

标注是弱标注（评论包含主题同义词即视为相关），词表检索与标注部分同源，因此词表指标只作参照，真正的对比是「纯向量 vs 混合」。

- 数据扩到 800 条后 hit@5 出现饱和：电池相关 195 条、屏幕 247 条，弱标注覆盖了约 1/4 语料，top-5 很容易命中。此时 MRR 更有区分度：0.733 → 0.783。
- 混合补回了纯向量漏检的样本：`手机发热严重吗` 只有 2 条相关评论，向量没召回（RR=0），词表召回排第 1，融合后排第 2（RR=0.5）。
- 混合的增益集中在已建词表的方面。长尾主题 `物流发货快不快`（12 条相关）没有对应词表，词表召回漏检；向量和混合能进 top-5，但混合没有超过纯向量（RR 都是 0.333），说明加词表并不能提升长尾排序。
- 这也解释了为什么没有直接上重排模型：当前瓶颈是主题覆盖与标注质量，不是排序精度。

## 工具调用与流式输出

工作流：`classify_question → agent（可多轮调用工具）→ finalize`。

- 工具 `aspect_statistics(question)`：从 SQLite 取单个方面的真实统计，是唯一数字来源。
- 工具 `overall_statistics()`：从 SQLite 取所有方面的汇总统计，供整体评价使用。
- 工具 `search_review_evidence(query, k)`：向量检索评论文本作为证据，可多次调用。
- `finalize` 节点用代码拼接统计句并解析模型正文，模型依旧没有生成数字的权力。

流式接口逐个节点推送 SSE 事件（`classify_question` / `agent` / `tools` / `finalize` / `done`）：

```bash
curl -N -X POST http://localhost:8000/api/v1/analysis/analyze/stream \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"电池有什么缺点？\",\"product_id\":\"default\"}"
```

## 多轮会话

`analyze` / `analyze/stream` 都支持可选的 `session_id`；不传则保持单轮问答与缓存行为：

```bash
curl -X POST http://localhost:8000/api/v1/analysis/analyze \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"那优点呢\",\"product_id\":\"default\",\"session_id\":\"demo-1\"}"
```

- 历史存在 Redis，只保留最近 3 轮，30 分钟过期，按 `product_id` 隔离。
- 只存「问题、问题类型、方面」，不存完整答案：避免上一轮的统计数字被模型复述，本轮数字仍由代码查库生成。
- 追问用规则改写，不改写就退化为单轮，不让模型在改写阶段改统计口径：

| 上一轮 | 本轮 | 改写结果 |
| --- | --- | --- |
| 电池有什么缺点？ | 那优点呢 | 电池有什么优点？ |
| 电池有什么缺点？ | 那发热呢 | 发热有什么缺点？ |
| 电池有什么缺点？ | 它还有什么缺点 | 电池有什么缺点？ |
| 电池有什么缺点？ | 整体评价怎么样？ | 不改写（整体问题不缩小成单方面） |

- 多轮请求直接跳过单轮缓存：缓存 key 只有产品/版本/问题，不含上下文，命中会答非所问。`meta.memory_turns` 表示本轮依赖的历史轮数。
- 限制：不做“为什么？”这类长程指代，规则未覆盖时按单轮处理；会话过期后自动回到无上下文状态。

实测同一 session：`电池有什么缺点？`（memory_turns=0）→ `那优点呢`（跳到“电池有什么优点”，正面 10 / 负面 7）→ `那发热呢`（跳到“发热有什么缺点”，提及 1 次）→ `整体评价怎么样？`（未被缩窄，仍汇总全部方面）。单轮请求的 `cached=true` 行为不变。

## 可观测性与护栏

同步与流式接口都会返回 `meta`，把“这次回答花了多少代价、有没有被护栏截断”暴露给调用方：

| 字段 | 含义 |
| --- | --- |
| `cached` | 是否命中 Redis 缓存（缓存命中不产生 token） |
| `latency_ms` | 请求总耗时 |
| `tool_rounds` | 实际执行的工具轮次 |
| `guarded` | 是否因达到轮次上限被强制收敛 |
| `input_tokens` / `output_tokens` | 本次所有 LLM 调用的 token 汇总 |
| `memory_turns` | 本轮依赖的历史轮数（0 表示单轮问答） |
| `trace` | 逐节点明细，含 `node`、`seconds`、token |

```json
{"cached": false, "latency_ms": 1826.2, "tool_rounds": 1, "guarded": false,
 "input_tokens": 1717, "output_tokens": 136,
 "trace": [{"node": "classify_question", "seconds": 0.0},
           {"node": "llm", "input_tokens": 680, "output_tokens": 78},
           {"node": "agent", "seconds": 0.868},
           {"node": "tools", "seconds": 0.173},
           {"node": "llm", "input_tokens": 1037, "output_tokens": 58},
           {"node": "agent", "seconds": 0.777},
           {"node": "finalize", "seconds": 0.001}]}
```

`GET /metrics` 暴露进程内累计指标，便于压测和排查“入库成功但检索不到”：

```bash
curl http://localhost:8000/metrics
```

返回 `requests` / `errors` / `cache_hits` / `guardrail_hits` / `input_tokens` / `output_tokens` / `latency_ms_total` / `avg_latency_ms`，以及 `index`（`reviews` 与 `vectors` 条数）。实测单次 analyze 约 2-3 秒、消耗 `input/output tokens ≈1700/150`；冷启动首个请求约 7.9 秒（模型加载），`index` 稳定为 830/830（800 条导入环境，见上文测量环境说明）。流式接口的 `finalize` 事件同样带 `meta`，其中 `trace` 由各节点在流式过程中累加，避免在 `finalize` 里只剩最后一段。

护栏：

- 工具轮次上限 `settings.max_tool_rounds = 4`：达到上限即使模型还想调工具，也会路由到 `finalize`，答案末尾追加「工具调用已达上限，本次未能生成完整结论，以上数字仍来自数据库统计」。
- 模型超时 `settings.llm_timeout_seconds = 60`，避免请求无限挂起。
- 工具参数校验：`query`/`question` 为空直接返回 `{"error": ...}`，`k` 夹取到 `[1, 20]`。
- 指标是进程内计数，单进程 uvicorn 够用；多副本部署需换成 Prometheus client 或 Redis 计数器，否则各副本数据彼此独立。

## 压测与容量

`scripts/load_test.py` 用唯一后缀绕过 Redis 缓存，测的是完整链路（LLM 决策 + 检索 + 统计）：

```bash
docker exec -w /app reviewinsight-api-1 \
  python scripts/load_test.py --concurrency 10 --requests 20
```

实测（单 uvicorn worker，800 条评论，每组 20 次请求、0 缓存命中）：

| 场景 | 并发 | RPS | P50 | P95 | 错误率 |
| --- | --- | --- | --- | --- | --- |
| 修复前 | 10 | 0.29 | 33.2 s | 36.0 s | 0% |
| 修复后 | 10 | 2.02 | 2.7 s | 6.1 s | 0% |
| 修复后 | 20 | 2.78 | 3.7 s | 6.9 s | 0% |

修复前的瓶颈不是模型本身，而是两个实现问题：

- `/analyze` 在 async 接口里直接调用同步的 `agent_app.invoke`，会阻塞整个事件循环，20 个请求被完全串行化。改为 `run_in_threadpool` 后并发才真正生效。
- `get_embeddings()` 每次检索都新建 `HuggingFaceEmbeddings`，重复加载模型；改为 `@lru_cache(maxsize=1)` 进程级单例。

结论：当前瓶颈是每个请求 2 次 LLM 调用与单进程 worker，20 并发下 P95 约 6.9 秒、无错误；继续扩容应加 worker/请求队列，而不是继续改检索。对面试可以明确说明：这不是压测工具得出的“理论 QPS”，而是同一容器、同一模型、真实 LLM 调用下的结果。

## 多产品与真实数据

上传接口支持 `product_id`，统计、检索、缓存都按产品隔离：

```bash
curl -X POST http://localhost:8000/api/v1/analysis/upload \
  -F "file=@reviews.csv" -F "product_id=jd_phone"
```

`scripts/import_dataset.py` 用于导入公开中文评论数据，抽样后生成 `data/real/*.csv` 并上传：

```bash
# HuggingFace 数据集（默认 dirtycomputer/online_shopping_10_cats，10 类电商评论）
docker exec -w /app reviewinsight-api-1 pip install -q -r requirements-dev.txt
docker exec -e HF_HUB_OFFLINE=0 -w /app reviewinsight-api-1 \
  python scripts/import_dataset.py --category 手机 --limit 800 --product-id shopping_phone

# 本地已下载的天池/京东语料（CSV 需含 comment 或 review 列）
docker cp data/raw/jd.csv reviewinsight-api-1:/tmp/jd.csv
docker exec -w /app reviewinsight-api-1 \
  python scripts/import_dataset.py --source csv --csv /tmp/jd.csv --product-id jd_phone
```

脚本参数默认值：`--limit 200`、`--product-id shopping10`、`--hf-name dirtycomputer/online_shopping_10_cats`、输出到 `data/real/`；示例命令显式覆盖了条数与产品。

实测导入 800 条真实手机评论后：`index` 为 830/830（含默认产品 30 条），`shopping_phone` 电池方面提及 203 次、负面 112 次（55%），外观设计提及 196 次（24.5%，负面 19%），与 `default` 产品完全隔离。注意 `seamew/ChnSentiCorp` 是脚本型数据集，`datasets>=4` 已不再支持，脚本默认改用 parquet 格式的 `online_shopping_10_cats`。

### 关键词匹配增强

首版关键词只覆盖 4 个电池词，200 条真实手机评论里仅 32 条被识别为电池相关。扩充后的规则：

- 词表扩充：待机、掉电、耗电、快充、闪充、充满、毫安、mAh、蓄电，以及繁体变体（電池/續航/充電）
- 文本归一化：NFKC + 转小写 + 去掉汉字之间的空格（“电 池”也能命中）
- 按句判定极性：负面词只在命中方面词的同一句内生效。切句符是 `。！？!?；;\n`，**不含逗号**，所以 `电池很好，物流太慢` 仍属同一句，会被判负；写成 `电池很好。物流太慢` 才是正面
- 否定免疫：`不/没/无/未` 紧邻否定，以及 `不会/不能/没有` 允许隔几个字（“不会突然断电”不再判负）
- 负面方面词：`烫/发热` 这类词命中即负面

效果（最初 200 条样本，同一批数据重算；扩量到 800 条后电池提及 203 次、负面 112 次）：

| 指标 | 增强前 | 增强后 |
| --- | --- | --- |
| 电池提及 | 32（16.0%） | 49（24.5%） |
| 电池负面 | 20 | 25 |

已入库的历史数据用脚本重算，不需要重建向量索引（`aspects` 只存在 SQLite）：

```bash
docker exec -w /app reviewinsight-api-1 python scripts/recompute_aspects.py            # 先看差异
docker exec -w /app reviewinsight-api-1 python scripts/recompute_aspects.py --apply
docker exec reviewinsight-redis-1 redis-cli FLUSHDB                                    # 统计口径变了，清缓存
```

注意：镜像构建时已把嵌入模型烘焙进 `HF_HOME`；Compose 默认 `HF_HUB_OFFLINE=1`（运行时不回源），下载数据集时必须用 `-e HF_HUB_OFFLINE=0` 覆盖。

## 已知限制

- 方面词表只覆盖电池 / 发热 / 外观设计，长尾主题依赖向量检索，排序提升有限。
- 多轮改写只覆盖规则内的指代与承接问法，复杂长程指代按单轮处理。
- 指标为进程内计数，多副本部署需外置计数器；缓存与会话依赖 Redis，Redis 不可用时退化为单轮、无缓存。
- 容量瓶颈在单 worker 与每个请求 2 次 LLM 调用，扩容优先加 worker / 请求队列。
- 仓库不附带评论数据，首次运行必须上传或导入，否则演示页与统计均为空。
