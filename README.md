# RAG 企业知识任务智能体

[![普通测试](https://github.com/LilBin85/rag-enterprise-kb/actions/workflows/tests.yml/badge.svg)](https://github.com/LilBin85/rag-enterprise-kb/actions/workflows/tests.yml)
[![PostgreSQL/Qdrant 集成测试](https://github.com/LilBin85/rag-enterprise-kb/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/LilBin85/rag-enterprise-kb/actions/workflows/integration-tests.yml)

一个面向企业制度与技术规范的可审计 RAG/Agent 后端：在租户与 ACL 隔离下完成文档摄取、证据检索、只读研究和 HITL 受控写入。

## 架构

```mermaid
flowchart LR
    U[调用方] --> API[FastAPI /v1]
    API <--> PG[(PostgreSQL<br/>任务、恢复命令、权限与证据事实源)]
    PG <--> TW[task-worker]
    PG <--> IW[ingestion-worker]
    PG <--> PW[projection-worker]
    API --> Q[(Qdrant<br/>授权向量检索)]
    TW --> Q
    IW --> Q
    PW --> Q
    TW --> OBJ[(对象存储)]
    IW --> OBJ
    PW --> OBJ
    API --> LLM[DeepSeek]
    TW --> LLM
```

PostgreSQL 是任务、恢复命令和权限状态的唯一事实源，Qdrant 只是可重建、可收敛的检索投影；Agent 身份、工具和审批绑定均由服务端注入。队列继续由 PostgreSQL claim/lease/fencing 驱动，没有新增 Redis、Celery、Kafka 或其他消息中间件。

## Fast Path 与 Durable Path

`EnterpriseRetriever` 是唯一共享 Retrieval Core。它由 infrastructure factory 以同一组授权、Qdrant、BGE、候选/上下文限制、RRF、hybrid、阈值和证据充分性配置装配；两条路径只共享检索事实，不合并任务生命周期或响应语义。

| 场景 | 选择 | 时序 | 返回 |
| --- | --- | --- | --- |
| 单问题、低步骤数、无需澄清或审批 | Fast Path：`POST /v1/knowledge/query` | API → `KnowledgeQueryService` → Retrieval Core → Fast evidence ledger → 受约束回答与 citation 复核 | 同步 `answer` 或 `refusal` |
| 多文档研究、版本比较、澄清、审批或受控写入 | Durable Path：`POST /v1/tasks` | API → PostgreSQL task → task-worker/Agent → `search_knowledge` → Retrieval Core → Durable evidence ledger → ToolMessage/checkpoint | 创建与恢复返回 `202`，结果异步读取 |

Fast Path 不依赖 task-worker capability；Durable Path 保留 Agent、工具预算、checkpoint 与 HITL。检索 telemetry 仅用固定 `retrieval_consumer=fast|durable` 区分消费者，不记录 query、task、tenant 或 user 标签。

## 三分钟演示

前置：Docker Desktop 已启动，`.env` 已配置 `DEEPSEEK_API_KEY`、`DEMO_ROLE_TOKENS` 和本地 BGE-M3 缓存。已有镜像和模型缓存时，只需一条命令：

```powershell
docker compose up --build -d
```

服务就绪后打开 [Swagger 操作页](http://localhost:8000/docs)，按“初始化身份 → 上传文档 → 等待摄取成功 → 提问”完成演示；逐步说明见[使用指南](使用指南.md)，健康状态见 [`/health/ready`](http://localhost:8000/health/ready)。

## 核心指标

| 验收项 | 固定版本实测结果 |
| --- | ---: |
| 自动化测试 | 588 / 588 通过，0 失败，0 跳过 |
| 检索质量 | Recall@5 = 1.00，MRR = 1.00，无答案准确率 = 1.00 |
| 权限隔离 | ACL 泄漏率 = 0 |
| M1 只读 Research Agent | 通过；4 次工具调用，3 条引用 |
| M2 受控写入 | 通过；31 个任务，21 个场景 |

数据来自机器报告 [docs/acceptance/latest.json](docs/acceptance/latest.json)，绑定代码版本 `ed056ac`；它是固定版本快照，不自动代表后续工作树已通过。发布前必须重新生成报告，并以 `verdict`、`gates`、`unverified` 和检索质量门为准。

## 三个关键技术取舍

| 取舍 | 得到什么 | 付出什么 |
| --- | --- | --- |
| PostgreSQL 作为唯一事实源 | 状态、授权、证据和恢复语义可事务化审计 | 迁移与事务边界更严格 |
| Qdrant 作为带世代的授权投影 | 避免旧向量和越权结果进入答案 | 摄取需要构建、校验、切换和收敛 |
| 固定工具集 + HITL + checkpoint | Agent 可恢复，写操作可审批、可追责 | 比无约束同步 Agent 延迟更高、流程更长 |

详细边界见[架构说明](docs/architecture.md)、[威胁模型](docs/threat-model.md)和[迁移说明](docs/migration.md)。

## Windows 启动

前置：Conda 环境 `rag-kb`、PostgreSQL、Qdrant（默认 `http://localhost:6333`）。不要把密钥写入仓库；`.env` 仅供本地配置读取。

完整容器方式会启动 PostgreSQL、Qdrant、迁移、一次性 `qdrant-init`、API 和三个独立 Worker。
启动前应在 `.env` 中把 `DEMO_ROLE_TOKENS` 配置为 JSON 令牌映射；未配置时健康检查可用，
受保护接口固定拒绝未认证请求：

```powershell
docker compose up --build -d
```

本地演示完整观测链路时，改用以下命令。Grafana、Jaeger 和 Prometheus 仅绑定到
`127.0.0.1`；三个 Worker 的 `:9464`、`:9465`、`:9466` 指标端口只在 Compose 网络内供 Prometheus 抓取。

```powershell
docker compose --profile observability up --build -d
```

入口：Grafana `http://127.0.0.1:3000`、Jaeger `http://127.0.0.1:16686`、Prometheus
`http://127.0.0.1:9090`。观测配置不采集请求、正文、Prompt、模型输出或工具 payload。

本机未安装 Docker 时可使用下面的 Windows/Conda 方式：

```powershell
conda activate rag-kb
docker compose up -d qdrant
$env:RAG_KB_DATABASE_URL = "postgresql+psycopg://<用户>:<密码>@localhost:5432/<数据库>"
alembic upgrade head                 # 当前迁移目标：20260831_0015
python scripts/init_agent_checkpointer.py
python -m rag_kb.qdrant_init         # 幂等创建并校验企业 Qdrant schema
.\scripts\start_api.ps1             # FastAPI :8000
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --queue task --watch
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --queue ingestion --watch
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --queue projection --watch
```

三个命令需要分别在独立终端中持续运行；也可分别执行 `.\scripts\start_worker.ps1 -Queue task`、`.\scripts\start_worker.ps1 -Queue ingestion` 和 `.\scripts\start_worker.ps1 -Queue projection`。不存在默认轮询全部队列的模式。

| Worker | 职责与配置边界 |
| --- | --- |
| `task-worker` | 执行 Agent 初始任务、用户澄清恢复和审批恢复，发布 `task-agent/m2-fenced-v1` capability，收敛 task/tool/knowledge-query 超时和 task retention，并维护任务队列指标；需要 DeepSeek 配置。 |
| `ingestion-worker` | 解析文档、生成 embedding、写入索引、恢复上传预留并维护摄取队列指标；不需要 DeepSeek 配置。 |
| `projection-worker` | 执行 Qdrant 投影收敛、周期审计、修复与重建；不需要 DeepSeek 配置。 |

`GET /health/ready` 当前只检查 PostgreSQL、Alembic schema revision、Qdrant、Qdrant collection schema 和活动投影收敛状态，不直接检查任一 Worker。Worker 缺失的实际影响如下：

| 缺失服务 | 实际影响 |
| --- | --- |
| `task-worker` | capability lease 到期后新建 task 返回 `503`；已有任务和恢复命令继续保留在 PostgreSQL 中等待消费；同步 `/v1/knowledge/query` 不依赖 Agent Worker。 |
| `ingestion-worker` | 上传仍可能返回 `202`，但摄取任务积压，文档长期停在 `pending`。 |
| `projection-worker` | 周期投影审计和修复停止；只有活动投影实际不收敛时，`projection_convergence` 才会使 readiness 失败。 |

若本机没有 Docker，可由管理员单独启动 PostgreSQL/Qdrant；不要在 API 进程中执行迁移或下载模型。健康检查：`GET /health/live` 与 `GET /health/ready`。

2026-08-23 已完成本机 Docker Compose 真实构建、迁移、Qdrant 初始化、服务健康和 HTTP
认证边界验证；其他环境必须重新运行健康检查，不能直接继承本机结论。

## API

- `POST /v1/knowledge/query`：服务端身份问答。
- `POST /v1/documents`：multipart 上传新文档，必须带 `Idempotency-Key`，返回 `202`。
- `POST /v1/documents/{document_id}/versions`：上传同一文档的新版本；路径和 metadata 的 `document_id` 必须一致，并使用新的 `Idempotency-Key`。
- `GET /v1/documents/{document_id}`、`GET /v1/ingestion-jobs/{job_id}`：租户与 ACL 约束的查询。
- `POST /v1/tasks` 接受 `research` 与 `controlled_action`；后者还要求 feature gate 开启和兼容 Worker lease。`GET /v1/tasks/{task_id}/result` 返回服务端重验的结果；`POST /v1/tasks/{task_id}/continue` 只接受服务端确认的澄清恢复请求，成功返回 `202 Accepted`。
- `GET /v1/approvals`、`GET /v1/approvals/{approval_id}`、`POST /v1/approvals/{approval_id}/decision`：仅 `task:approve` 审批者可发现和决定待审批动作；决定成功返回 `202 Accepted`，同一决定重试保持幂等，相反决定返回稳定 `409` 冲突。
- `GET /v1/tasks/{task_id}/artifacts/{artifact_id}/content`：仅任务 owner 且拥有 `task:read` 时读取正文；未就绪为 `409`、过期或已清理为 `410`、完整性失败为 `500`，响应不暴露对象键或文件元数据。

身份来自 `Authorization: Bearer ...` 或本地演示 `X-Demo-Role-Token`，角色不能由请求正文提升。本地演示主体的初始化请运行 `python scripts/init_demo_principals.py`，它按令牌真实解析的 permissions attributes 写入，不要手工插入空 attributes 的记录。应用只提供版本化 `/v1` 业务接口。

上述两个恢复接口的 `202` 只表示 API 已完成认证、授权、版本校验并把恢复命令耐久提交到 PostgreSQL，不表示 Agent、工具或任务已经同步恢复完成；API 返回时任务仍可能是 `waiting_for_user` 或 `waiting_for_approval`。API 不构造 LangGraph `Command`、不运行 Agent、也不执行工具；只有 `task-worker` 能领取命令、重新校验服务端绑定并构造服务端恢复载荷。Worker 暂时离线不会撤销已提交命令，命令会留在 PostgreSQL 等待消费；响应不暴露恢复命令敏感载荷或内部执行绑定。

## 验证

GitHub Actions 将普通测试与 PostgreSQL/Qdrant 集成测试拆成两条流水线，并分别上传 JUnit 报告；60 例离线任务评测报告见 [docs/acceptance/enterprise-task-evaluation.json](docs/acceptance/enterprise-task-evaluation.json)。

```powershell
conda run -n rag-kb --no-capture-output python -m pytest -q `
  tests/test_database_migrations.py `
  tests/test_agent_checkpointer.py `
  tests/test_task_agent.py `
  tests/test_task_worker_recovery.py `
  tests/test_v1_documents_api.py

conda run -n rag-kb --no-capture-output python -m pytest -q

# 仅在 RAG_KB_TEST_POSTGRES_URL 已配置且 Qdrant 可达时运行
conda run -n rag-kb --no-capture-output python -m pytest -q `
  tests/test_postgres_agent_checkpointer.py `
  tests/test_postgres_qdrant_projection_convergence.py

# 真实模型验证：先创建独立 PostgreSQL 数据库、m1_e2e_ 前缀的 Qdrant collection
# 和独立对象目录；执行结束后删除这些临时资源。环境变量 LANGGRAPH_STRICT_MSGPACK=true
# 会同时启用 LangGraph 运行时 allowlist 与 schema 推导白名单；脚本还会核验
# checkpoint serializer 的 pickle_fallback=false 并拒绝 pickle 编码落库。
$env:LANGGRAPH_STRICT_MSGPACK = "true"
conda run -n rag-kb --no-capture-output python scripts/smoke_m1_research_e2e.py `
  --allow-live `
  --database-url "postgresql+psycopg://<用户>:<密码>@localhost:5432/<临时数据库>" `
  --qdrant-url "http://localhost:6333" `
  --qdrant-collection "m1_e2e_<唯一标识>" `
  --object-store-root "data/local/e2e/m1_e2e_<唯一标识>" `
  --hf-home "$env:HF_HOME" `
  --previous-source "data/corpus/sources/员工休假制度-v1.md" `
  --source "data/corpus/sources/员工休假制度-v2.md" `
  --report "docs/acceptance/m1-research-e2e.json"

conda run -n rag-kb --no-capture-output python scripts/benchmark_enterprise_retrieval.py `
  --dataset data/benchmarks/retrieval_acceptance_v2.yaml `
  --schema schemas/retrieval-acceptance-v2.schema.json `
  --output data/local/benchmarks/enterprise-retrieval-stage9-sufficiency-aligned.json
conda run -n rag-kb --no-capture-output python scripts/build_project_acceptance_report.py `
  --pytest-xml data/local/acceptance/pytest-stage9.xml `
  --retrieval-report data/local/benchmarks/enterprise-retrieval-stage9-sufficiency-aligned.json `
  --m1-e2e-report docs/acceptance/m1-research-e2e.json `
  --m2-e2e-report docs/acceptance/m2-controlled-write-e2e.json `
  --dataset data/benchmarks/retrieval_acceptance_v2.yaml `
  --config pyproject.toml `
  --output docs/acceptance/latest.json
```

报告中记录的 M1/M2 严格真实 E2E 曾在隔离资源上通过：M1 验证 V1/V2 上传、精确四个 READ 工具与结果引用；M2 验证 31 个受控任务、HITL、批准/拒绝、artifact 隔离、恢复窗口、严格 msgpack 与终态审计。其固定版本全量回归为 `528 passed / 0 failed / 0 skipped`，不是后续工作树结论；脚本不会从文档继承指标或填充缺失数字。

## 目录边界

`src/rag_kb/` 为新应用；`tests/` 为分层回归；`schemas/` 为机器契约；`scripts/` 为可复现入口；`data/benchmarks/` 为任务集。用户的 `.obsidian/`、`output/`、面试材料和历史验收报告均属于受保护资产，不在迁移中删除或覆盖。E2E 结束后，调用方必须删除本次创建的临时 PostgreSQL 数据库、专属 Qdrant collection 和对象目录；正式机器证据为 `docs/acceptance/m1-research-e2e.json`、`docs/acceptance/m2-controlled-write-e2e.json` 与 `docs/acceptance/latest.json`。
