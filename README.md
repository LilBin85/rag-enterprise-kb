# RAG 企业知识任务智能体

面向企业制度与技术规范的租户隔离知识任务智能体。新版 `/v1` 接口以 PostgreSQL 事实源、Qdrant 授权检索、证据账本和可恢复任务状态为核心；旧接口仅作为薄兼容层，默认关闭。

验收唯一事实来源是机器报告 [docs/acceptance/latest.json](docs/acceptance/latest.json)。报告缺失、真实模型未运行或检索质量门失败时，文档不推断“全部通过”；请以报告中的 `verdict`、`gates` 和 `unverified` 为准。

## 架构与运行模式

- `knowledge_query`：一次授权检索、证据账本记录、受约束回答。拒答分两层：数量/锚点等字面缺失在检索层确定性拦截，主谓语义充分性由回答模型 verdict 判定；引用一律以证据账本复核兜底。
- `research`：可暂停、继续、取消的多步研究任务；每步由显式 `NextAction` 和持久 checkpoint 驱动。
- `controlled_action`：读工具检索后提出写动作；固定六工具注册表中的写工具必须绑定当前任务、计划、证据、主体、参数摘要和未过期批准。

摄取采用“对象落盘 → 解析 → SQL manifest → generation-specific Qdrant projection → receipt → active 发布”的顺序。lease、generation、幂等键和状态事件用于崩溃恢复；旧 worker 不能覆盖新世代。

固定工具为 `search_knowledge`、`fetch_evidence`、`list_document_versions`、`compare_document_versions`、`create_report_artifact`、`create_work_item_draft`。写工具只允许服务端生成对象键，不能接收路径、URL、shell、SQL 或任意外部调用。

详见：[架构说明](docs/architecture.md)、[威胁模型](docs/threat-model.md)、[迁移说明](docs/migration.md)、[使用指南](使用指南.md)。

## Windows 启动

前置：Conda 环境 `rag-kb`、PostgreSQL、Qdrant（默认 `http://localhost:6333`）。不要把密钥写入仓库；`.env` 仅供本地配置读取。

完整容器方式会启动 PostgreSQL、Qdrant、迁移、一次性 `qdrant-init`、API 和 worker。
启动前应在 `.env` 中把 `DEMO_ROLE_TOKENS` 配置为 JSON 令牌映射；未配置时健康检查可用，
受保护接口固定拒绝未认证请求：

```powershell
docker compose up --build -d
```

本机未安装 Docker 时可使用下面的 Windows/Conda 方式：

```powershell
conda activate rag-kb
docker compose up -d qdrant
$env:RAG_KB_DATABASE_URL = "postgresql+psycopg://<用户>:<密码>@localhost:5432/<数据库>"
alembic upgrade head                 # 当前迁移目标：20260822_0008
python -m rag_kb.qdrant_init         # 幂等创建并校验企业 Qdrant schema
.\scripts\start_api.ps1             # FastAPI :8000
conda run -n rag-kb --no-capture-output python -m rag_kb.worker --watch --include-tasks
```

若本机没有 Docker，可由管理员单独启动 PostgreSQL/Qdrant；不要在 API 进程中执行迁移或下载模型。健康检查：`GET /health/live` 与 `GET /health/ready`。

2026-08-23 已完成本机 Docker Compose 真实构建、迁移、Qdrant 初始化、服务健康和 HTTP
认证边界验证；其他环境必须重新运行健康检查，不能直接继承本机结论。

## API

- `POST /v1/knowledge/query`：服务端身份问答。
- `POST /v1/documents`：multipart 上传，必须带 `Idempotency-Key`，返回 `202`。
- `GET /v1/documents/{document_id}`、`GET /v1/ingestion-jobs/{job_id}`：租户与 ACL 约束的查询。
- 任务接口位于 `/v1/tasks`；暂停、继续、取消、批准和恢复由持久状态图处理。

身份来自 `Authorization: Bearer ...` 或本地演示 `X-Demo-Role-Token`，角色不能由请求正文提升。旧 `/questions`、`/policy-comparisons`、`/policy-rollout-tasks` 只有在 `create_app(include_legacy_routes=True)` 时装配，并返回 `Deprecation`、`Sunset`、`Link`。

## 验证

```powershell
conda run -n rag-kb --no-capture-output python -m pytest -q
conda run -n rag-kb --no-capture-output python scripts/benchmark_enterprise_retrieval.py `
  --dataset data/benchmarks/retrieval_acceptance_v2.yaml `
  --schema schemas/retrieval-acceptance-v2.schema.json `
  --output data/local/benchmarks/enterprise-retrieval-stage9-sufficiency-aligned.json
conda run -n rag-kb --no-capture-output python scripts/build_project_acceptance_report.py `
  --pytest-xml data/local/acceptance/pytest-stage9.xml `
  --retrieval-report data/local/benchmarks/enterprise-retrieval-stage9-sufficiency-aligned.json `
  --task-report data/local/benchmarks/enterprise-task-evaluation.json `
  --model-report data/local/acceptance/task-model-live-smoke.json `
  --dataset data/benchmarks/retrieval_acceptance_v2.yaml `
  --config pyproject.toml `
  --output docs/acceptance/latest.json
```

2026-08-24 的隔离真实组件套件为 `334 passed`；机器报告中的 pytest、检索质量、ACL
零泄漏、真实模型和任务契约门均为 `passed`。脚本仍以证据文件为准，不从文档继承指标，
也不会填充缺失数字。

## 目录边界

`src/rag_kb/` 为新应用；`tests/` 为分层回归；`schemas/` 为机器契约；`scripts/` 为可复现入口；`data/benchmarks/` 为任务集。用户的 `.obsidian/`、`output/`、面试材料和历史验收报告均属于受保护资产，不在迁移中删除或覆盖。
