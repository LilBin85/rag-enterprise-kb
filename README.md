# RAG 企业知识任务智能体

面向企业制度与技术规范的租户隔离知识任务智能体。`/v1` 接口以 PostgreSQL 事实源、Qdrant 授权检索、证据账本和可恢复任务状态为核心。

机器报告 [docs/acceptance/latest.json](docs/acceptance/latest.json) 保留其记录的 revision/worktree 的 M1/M2 验收快照，不代表当前工作树已经通过验收。发布前必须重新生成报告，并以 `verdict`、`gates`、`unverified` 和检索质量门为准，不能由文档补写缺失证据。

## 架构与运行模式

- `knowledge_query`：一次授权检索、证据账本记录、受约束回答。拒答分两层：数量/锚点等字面缺失在检索层确定性拦截，主谓语义充分性由回答模型 verdict 判定；引用一律以证据账本复核兜底。
- `research`：公开的只读任务模式，固定暴露四个 READ 工具。
- `controlled_action`：受 feature gate 和兼容 Worker capability 约束的公开任务模式；在同一 `thread_id = task_id` 中先经 HITL 暂停和服务端审批，再执行两个固定 WRITE adapter。客户端不能传入 checkpoint、resume 或工具绑定数据。

摄取采用三阶段投影切换：先按新 generation 分批构建并回读校验 Qdrant 投影，再发布新世代可见，最后在单个 SQL 事务中以原绑定为前置条件 CAS 激活版本并精确绑定新世代，绑定成功后才清理旧投影。切换窗口内，检索仍只使用 PostgreSQL 当前绑定的 `projection_key`；旧 worker 不能倒退当前世代。

worker 对任务、摄取和投影队列轮转调度，并周期执行有界的上传预留回收与正文保留收敛：过期的 waiting 任务自动终止，工具结果、任务 checkpoint/结果和固定前缀孤儿对象按 TTL 幂等清理且不改写历史执行终态（详细边界见[架构说明](docs/architecture.md)）。

Research Agent 只暴露 `search_knowledge`、`fetch_evidence`、`list_document_versions`、`compare_document_versions` 四个 READ 工具；controlled-action Agent 在此基础上增加两个固定 WRITE adapter。模型只能使用服务端注入的身份上下文，不能传递 tenant、权限、lease、审批绑定或对象存储路径。公开任务创建与领取要求 `task-agent/m2-fenced-v1` capability lease；无兼容 capability 时 `POST /v1/tasks` 返回 503，同步知识查询、文档接口和全局 readiness 不受影响。

详见：[架构说明](docs/architecture.md)、[威胁模型](docs/threat-model.md)、[迁移说明](docs/migration.md)、[使用指南](使用指南.md)。

## Windows 启动

前置：Conda 环境 `rag-kb`、PostgreSQL、Qdrant（默认 `http://localhost:6333`）。不要把密钥写入仓库；`.env` 仅供本地配置读取。

完整容器方式会启动 PostgreSQL、Qdrant、迁移、一次性 `qdrant-init`、API 和 worker。
启动前应在 `.env` 中把 `DEMO_ROLE_TOKENS` 配置为 JSON 令牌映射；未配置时健康检查可用，
受保护接口固定拒绝未认证请求：

```powershell
docker compose up --build -d
```

本地演示完整观测链路时，改用以下命令。Grafana、Jaeger 和 Prometheus 仅绑定到
`127.0.0.1`；Worker 的 `:9464/metrics` 只在 Compose 网络内供 Prometheus 抓取。

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
alembic upgrade head                 # 当前迁移目标：20260829_0014
python scripts/init_agent_checkpointer.py
python -m rag_kb.qdrant_init         # 幂等创建并校验企业 Qdrant schema
.\scripts\start_api.ps1             # FastAPI :8000
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --watch --include-tasks
```

`--include-tasks` 同时启用通用任务队列、过期工具调用和同步问答租约回收；摄取、投影收敛和上传预留恢复始终由 worker 处理。

若本机没有 Docker，可由管理员单独启动 PostgreSQL/Qdrant；不要在 API 进程中执行迁移或下载模型。健康检查：`GET /health/live` 与 `GET /health/ready`。

2026-08-23 已完成本机 Docker Compose 真实构建、迁移、Qdrant 初始化、服务健康和 HTTP
认证边界验证；其他环境必须重新运行健康检查，不能直接继承本机结论。

## API

- `POST /v1/knowledge/query`：服务端身份问答。
- `POST /v1/documents`：multipart 上传新文档，必须带 `Idempotency-Key`，返回 `202`。
- `POST /v1/documents/{document_id}/versions`：上传同一文档的新版本；路径和 metadata 的 `document_id` 必须一致，并使用新的 `Idempotency-Key`。
- `GET /v1/documents/{document_id}`、`GET /v1/ingestion-jobs/{job_id}`：租户与 ACL 约束的查询。
- `POST /v1/tasks` 接受 `research` 与 `controlled_action`；后者还要求 feature gate 开启和兼容 Worker lease。`GET /v1/tasks/{task_id}/result` 返回服务端重验的结果；`/continue` 只恢复服务端确认的澄清 interrupt。
- `GET /v1/approvals`、`GET /v1/approvals/{approval_id}`、`POST /v1/approvals/{approval_id}/decision`：仅 `task:approve` 审批者可发现和决定待审批动作；服务端构造 resume，重复同一决定幂等，相反决定返回 `409`。
- `GET /v1/tasks/{task_id}/artifacts/{artifact_id}/content`：仅任务 owner 且拥有 `task:read` 时读取正文；未就绪为 `409`、过期或已清理为 `410`、完整性失败为 `500`，响应不暴露对象键或文件元数据。

身份来自 `Authorization: Bearer ...` 或本地演示 `X-Demo-Role-Token`，角色不能由请求正文提升。本地演示主体的初始化请运行 `python scripts/init_demo_principals.py`，它按令牌真实解析的 permissions attributes 写入，不要手工插入空 attributes 的记录。应用只提供版本化 `/v1` 业务接口。

## 验证

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

报告中记录的 M1/M2 严格真实 E2E 曾在隔离资源上通过：M1 验证 V1/V2 上传、精确四个 READ 工具与结果引用；M2 验证 31 个受控任务、HITL、批准/拒绝、artifact 隔离、恢复窗口、严格 msgpack 与终态审计。其双库全量回归为 `550 passed / 0 failed / 0 skipped`，是历史快照而非当前工作树结论；脚本不会从文档继承指标或填充缺失数字。

## 目录边界

`src/rag_kb/` 为新应用；`tests/` 为分层回归；`schemas/` 为机器契约；`scripts/` 为可复现入口；`data/benchmarks/` 为任务集。用户的 `.obsidian/`、`output/`、面试材料和历史验收报告均属于受保护资产，不在迁移中删除或覆盖。E2E 结束后，调用方必须删除本次创建的临时 PostgreSQL 数据库、专属 Qdrant collection 和对象目录；正式机器证据为 `docs/acceptance/m1-research-e2e.json`、`docs/acceptance/m2-controlled-write-e2e.json` 与 `docs/acceptance/latest.json`。
