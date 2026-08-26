# 重构任务进度记录

> 最后更新：2026-08-26
> 当前阶段：M1 Research Agent 二次修复，已完成 `M1_FIX_PLAN.md` 批次 0；批次 1-3 尚未完成，批次 4 尚未开始。
> 开发依据：`AGENT_REFACTOR_PLAN.md`、`M1_FIX_PLAN.md`

## 执行约束

- 只实现 M1 Research Agent，不进入 M2 WRITE、HITL 审批、多 Agent、MCP、代码执行或长期记忆。
- 按 `M1_FIX_PLAN.md` 固定顺序推进：批次 0 → 批次 1 → 批次 2 → 批次 3 → 批次 4。
- 0012 发布采用 stop-the-world：先停止任务 Worker，再迁移和部署，最后只启动兼容新协议的 Worker；禁止 0011 旧 Worker 与 0012 新 Worker 滚动混跑。
- 不使用破坏性 Git 操作，不覆盖已有未提交改动；敏感连接信息只从环境变量或容器读取。
- Python 验证统一使用：`conda run -n rag-kb --no-capture-output ...`。

## 总体状态

| 范围 | 状态 | 说明 |
| --- | --- | --- |
| P0 基线 | 已完成 | Compose loopback、演示身份、配置传播和基础验收材料已落地。 |
| M1 基础重构 | 已落地到当前工作树 | Research Agent、PostgreSQL checkpointer、四个 READ 工具、任务结果绑定和 0010/0011 基础迁移已实现。 |
| M1 批次 0 | 已完成并验证 | 0012 迁移、ORM/领域协议、历史回填、capability lease 基础方法及 SQLite/PostgreSQL round-trip 已完成。 |
| M1 批次 1 | 未完成 | fenced checkpointer、工具最终 fencing、capability 创建门和 Worker 心跳仍待实现。 |
| M1 批次 2 | 未完成 | deadline 数据库锁内终检、按任务预算构建 Agent、错误码边界和配置传播仍待实现。 |
| M1 批次 3 | 未完成 | 独立 retention service、状态/正文分离、过期读取契约、waiting reaper 和 orphan 扫描仍待实现。 |
| M1 批次 4 | 未开始 | 最终验收、文档全量同步和交付清理尚未进行。 |

## 已完成的 M1 基础能力

- 公开任务模式收敛为 `research`；任务线程使用 `thread_id = task_id`。
- Agent 只暴露 `search_knowledge`、`fetch_evidence`、`list_document_versions`、`compare_document_versions` 四个 READ 工具，当前不暴露 WRITE 工具。
- `agent_tasks` 只保存业务生命周期、中断摘要、租约、预算和结果引用；消息、ToolMessage、Todo、interrupt/resume 仍由 LangGraph checkpointer 保存。
- `TaskWorker` 在 Agent 调用期间续租任务租约，结果完成前重新校验 evidence/citations，并把结果 ref/hash 绑定到任务。
- checkpointer 使用显式 msgpack allowlist 和 `pickle_fallback=False`。
- 0010 完成旧 Runtime 字段收缩，0011 增加 checkpoint/result purge marker 和 retention 基础字段。
- `POST /v1/documents/{document_id}/versions` 已建立同一文档的新版本上传路径，并校验路径与 metadata 的 `document_id` 一致。

## 批次 0：0012 与协议骨架

### 已实现

- 新增 `migrations/versions/20260826_0012_m1_fencing_and_content_lifecycle.py`，revision 为 `20260826_0012`，上游为 `20260826_0011`。
- 新增 `agent_capability_leases` 表及 `(capability_name, protocol_version, lease_expires_at)` 过期索引。
- `agent_tasks` 新增 `content_expires_at`；`tool_calls` 新增 `result_purged_at`。
- Research 终态正文到期时间已纳入领域模型、ORM 和数据库约束；成功/拒绝结果保留 kind/hash，正文 purge 后允许 ref 为空。
- 工具成功结果约束已允许“执行状态仍为 succeeded、正文引用已清理”的结构，并保留 hash 与过期审计字段。
- 历史 Research 终态会回填 `updated_at + 7 days` 的正文到期时间；缺少不可变结果绑定的旧 succeeded/refused 会 fail-close 为 failed。
- 仅对可识别的旧 READ GC 错误行进行审计修复；旧 WRITE `unknown` 不会被推断为成功。
- capability 注册、续租、注销、查询已使用数据库时间；预留 `_has_active_capability_in_session()` 供后续任务创建门复用。
- `TaskState.result_available()` 已区分执行终态、正文过期和正文 purge；`finish_task()`、取消路径和清理路径已映射新增字段。
- readiness revision 已更新为 `20260826_0012`，与 Alembic head 和 ORM 结构保持一致。

### 批次 0 验证证据

- SQLite 迁移历史修复与 round-trip：包含 0011 → 0012、降级到 0011、再次升级到 0012。
- SQLite 约束反例：验证 `NULL` 不能绕过 Research 正文到期、结果 hash 和工具结果过期字段约束。
- PostgreSQL 隔离迁移：使用独立 Docker 临时库完成 0011 → 0012 → 0011 → 0012，结果为 `1 passed`；测试结束后容器已删除。
- 全量测试：`368 passed, 12 skipped`。
- 迁移/领域/仓储/任务 API 定向测试：最近一次变更后通过；数据库迁移定向测试为 `15 passed, 1 skipped`。
- `git diff --check` 已通过；当前提交前不会纳入真实 `.env`、生产连接信息或临时数据库文件。

## 后续批次未完成项

### 批次 1：并发事实源闭环

- 尚未实现 `FencedPostgresSaver`，checkpoint `put/put_writes` 还没有与任务行锁在同一事务中完成 generation fencing。
- TaskWorker 尚未把 tenant、worker、generation 等 fence 配置稳定注入每个 checkpoint 写入。
- 工具 `_complete()` 尚未统一执行最终租约/当前身份门，也没有完整的对象写入失败补偿。
- 幂等回放尚未在返回正文前重新解析当前服务端身份。
- capability 表尚未接入 API `create_task()` 的同事务创建门；Worker 尚未注册、续租和注销 `m1-fenced-v2` capability。

### 批次 2：停止边界与提交契约

- `finish_task()` 尚未在数据库锁内用数据库时钟完成 deadline、取消状态和当前租约的最终终检。
- Agent 仍主要按 Worker 启动配置构建，尚未按任务行 `budgets` 选择或缓存 Agent。
- ExecutionGuard、stale/auth 分类、稳定错误码透传和对象绑定失败补偿仍需按计划闭环。
- `.env.example`、Compose 和运行时预算/capability 配置传播尚未完成最终一致性收口。

### 批次 3：正文生命周期闭环

- 尚未新增独立 `TaskRetentionService`；工具和任务清理仍部分位于 `ToolExecutionService`/`TaskWorker`。
- 运行时工具 GC 仍可能把 READ `succeeded` 改写为 `failed`，这与批次 3 要求的执行状态/正文状态分离不一致。
- 任务结果 API 和 `TaskResultService` 尚未统一使用 expiry/purge 判断，也未完成明确的 expired/unavailable 分类。
- waiting deadline reaper、删除失败保留引用并重试、固定前缀 orphan grace 扫描尚未完成。
- 无调用方的 legacy approval 入口尚未删除。

### 批次 4：最终验收与文档同步

- 尚未完成所有定向、全量和隔离 PostgreSQL 门禁的最终汇总。
- 尚未完成严格 msgpack 真实 Research E2E 的最终交付复核。
- README、`docs/architecture.md`、`docs/migration.md`、使用指南和 acceptance artifact 尚未按全部批次最终同步；当前部分说明仍保留旧迁移目标。
- 尚未执行最终临时数据库、Qdrant collection、对象目录和调试脚本清理清单复核。
- 在上述事项完成前，不声明全文 M1 收口，也不进入 M2。

## 当前交付边界

- 本次交付目标是 M1 批次 0 的协议骨架和 0012 迁移，不是全文 M1 最终验收。
- `docs/acceptance/latest.json` 的整体质量 verdict 仍需以最新检索质量报告为依据；没有重新生成质量报告时，不得将其 `unverified` 解读为代码回归。
- 0012 的真实发布必须遵守 stop-the-world 顺序；当前进度记录不替代正式发布 runbook。
