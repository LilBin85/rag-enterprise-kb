# 重构任务进度记录

> 最后更新：2026-08-28
> 当前阶段：M2 进行中——批次 0（0013 与协议事实）、批次 1（HITL 暂停与审批事实）已完成并验证；批次 2-4 未实施。M1 保持全部完成与验证，未回归。
> 开发依据：`AGENT_REFACTOR_PLAN.md`、`M2_PLAN.md`（未跟踪实施方案权威）

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
| M1 批次 1 | 已完成并验证 | checkpoint/工具 fencing、当前身份重校验、capability 创建门和 Worker 心跳已由隔离 PostgreSQL 并发测试验证。 |
| M1 批次 2 | 已完成并验证 | 停止边界、任务行预算单源、deadline 锁内终检、错误码边界和 API/Worker 配置传播已完成。 |
| M1 批次 3 | 已完成并验证 | F1/F2/F5-F9 复核通过；F3 推进性（P1）与 F4 目录成本（P2）已按 `M1_BATCH3_FIX_PLAN.md` 二次修复节完成并重新执行全部验证门禁。 |
| M1 批次 4 | 已完成并验证 | 最终验收、文档全量同步、acceptance artifacts 更新（verdict=passed）和交付清理已完成；详见批次 4 小节。 |
| M2 批次 0 | 已完成并验证 | 0013 迁移、task 协议绑定（claim/resume/checkpoint 门）、approval checkpoint 绑定与 reject 审计列、artifact content 生命周期列、strict serializer 门禁；提交 `0ec0049`。 |
| M2 批次 1 | 已完成并验证 | 两个 WRITE 薄 adapter + HITL 精确暂停、interrupt 双分类严格分流、服务端最小 action binding（PlanRevision）、pending approval checkpoint 绑定幂等、发起人自批拒绝、FormatChecker；提交 `0f4474d`，详见 M2 批次 1 小节。 |
| M2 批次 2-4 | 未实施 | 双层恢复与副作用安全、公共 API 与正文生命周期、故障注入与最终验收均未开始。 |

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

- `FencedPostgresSaver` 在同一 PostgreSQL transaction 内锁定 `agent_tasks` 并验证 owner、generation、lease、状态和租户；`put()` 与 `put_writes()` 均拒绝抢占后的旧 generation。
- `TaskWorker` 为每个 checkpoint config 稳定注入 tenant、worker 与 generation fence 字段；跨 superstep 的 saver 返回 config 保留这些字段。
- 工具首次执行、幂等回放、对象读取返回前与最终绑定统一经过任务租约和当前身份门；抢占或撤权后不绑定终态/ref，并补偿本次未绑定对象。
- PostgreSQL 租约判定和工具时间戳使用 `clock_timestamp()`；SQLite 保留注入时钟以维持确定性单测。最终身份门锁定 principal 与 membership 行，演示主体更新也遵循同一锁协议。
- Research capability 已接入 API 创建事务门；Worker 在完整构建后登记、后台续租，并在正常退出或中断路径 best-effort 注销。
- 验证：定向测试 `73 passed`、隔离 PostgreSQL 并发测试 `11 passed`、全量（含隔离 PostgreSQL）`395 passed, 1 skipped`；`git diff --check` 通过。

### 批次 2：停止边界与提交契约

- ExecutionGuard 仅统计四个业务工具；连续失败从完整 AI/Tool 轨迹重算，Todo、结构化输出、未知工具和非受控 ToolMessage 均排除。
- 新任务由 Settings 派生完整 `TaskBudgets` 后持久化；Worker 按 `lease.budgets` 选择并缓存 Agent，middleware 与 recursion limit 使用同一预算。
- `finish_task()` 在任务行锁内以 PostgreSQL `clock_timestamp()`（SQLite 注入时钟）终检租约、状态、取消和 deadline；deadline 已过时原子收敛为 `failed/task_deadline_expired`，并补偿未绑定任务结果对象。
- 普通工具异常透传稳定 `exc.code`，`ToolExecutionService` 不再吞掉 BaseException；`.env.example` 与 Compose 已显式共享预算、超时和 capability TTL 配置。
- 验证：定向 SQLite 测试通过；独立 Docker PostgreSQL 迁移至 0012 后相关测试 `15 passed`，容器已删除；全量测试 `386 passed, 20 skipped`；`git diff --check` 通过。

### 批次 3：正文生命周期闭环（F3/F4 二次修复已完成并验证）

- 新增独立 `TaskRetentionService`，统一执行 waiting deadline reaper、工具正文 GC、任务 checkpoint/结果 GC 与固定前缀 orphan 扫描；`TaskWorker` 和 `ToolExecutionService` 不再承担 retention 周期职责。
- 工具/任务执行结论与正文可用性已分离：GC 不改写 succeeded/refused/failed/cancelled/unknown；删除失败保留 ref/marker，成功后才写 purge marker。
- `TaskResultService` 与任务 API 统一以 PostgreSQL `clock_timestamp()`（SQLite 注入时钟）判断正文可用性；到期/已清理为 410，未就绪为 409，完整性错误 fail-close 为 500。
- waiting resume 与执行终态均在数据库时钟边界复核 deadline；对象最终绑定在工具 deadline 内复核，提交确认丢失时先锁行确认绑定再决定补偿删除。
- orphan 扫描只处理 `tool-results/`、`task-results/`，具备 grace、运行中保护、前缀轮转、游标、有界失败重试和公平扫描；`approve_for_task_revision()` 已删除且源内零引用。
- 本会话修复：结果绑定在任务行锁内复核当前身份；结果读取在返回前锁内终检；孤儿扫描在枚举期跳过已引用对象并采用有界扫描；证据冲突按授权/完整性分流；保留窗口、创建 deadline 与 Worker 关闭顺序已与契约对齐。
- 2026-08-27 验收复核曾发现 F3 首窗全引用时游标停滞（P1）、F4 空目录不计扫描预算（P2）；当日按二次修复节完成修复：
  - F3：`iter_ref_entries()` 改为规范序（目录带尾 `/` 排序保证与 ref 字符串序一致）+ `after` 位置游标 + `(entries, exhausted, resume)` 三元组；retention 游标改 opaque str——截断即前进、扫尽一律重置；枚举预算照常约束全部 stat；重试队列 ref 同入 exclude，防止 limit=1 时排队项挤占窗口饿死同前缀后续孤儿。游标纪律：任何"已检验、有资格、未返回"的对象绝不落到游标之下——预算溢出点不作游标，limit 收满时游标回退到最后一个已返回候选，杜绝跳过夹缝。验收语义为"有界轮次内收敛"（引用数 R、预算 B，≤ ceil(R/B)+1 轮），实测越窗/重启收敛/夹缝复现用例均通过。
  - F4：`delete_exact()` 删文件后机会式 rmdir 父目录；扫描期对过期空目录 best-effort rmdir；`put_exclusive()` 对并发 rmdir 的 ENOENT 重试一次。不做硬性目录预算机制。
- 验证（F3/F4 二次修复后全门禁，含夹缝缺陷补修复跑）：SQLite 定向三套件（retention/task_agent/task_worker_recovery）`39 passed`；隔离 Docker PostgreSQL（容器 `rag-kb-f3fix-8k4t`，postgres:18.6 @ 127.0.0.1:25432）迁移至 `20260826_0012` 并初始化 checkpointer 后定向 `6 passed`；本机 Qdrant healthz 200；全量套件在隔离 PostgreSQL + Qdrant 环境通过：`438 passed, 1 skipped in 231.61s`；`git diff --check` 通过；临时容器已删除并确认不存在。

### 批次 4：最终验收与文档同步（已完成并验证）

- 修复严格真实 E2E 的阻塞缺陷：任务创建移到 Task Worker 构建（capability `m1-fenced-v2` 注册）之后；E2E 报告补齐 strict msgpack 运行时证据（`LANGGRAPH_STRICT_MSGPACK=true` 生效、`JsonPlusSerializer(pickle_fallback=False)`、allowlist 仅 `TaskResultCandidate`）、数据库观测版本数/工具状态/checkpoint 编码与结果可读证据；acceptance 汇总器相应强制这些字段。
- 根因修复全量复跑暴露的 Qdrant 健壮性缺陷：`ensure_document_collection` 在创建请求不确定后对服务端异步落地增加短复查窗口（此前客户端超时 + 立即 get 404 会误判为 `document_index_collection_unavailable`），并为反例保留最小单测。
- 文档已按当前代码行为同步：README（0012 head、capability 门 503、retention 收敛、最终命令与 E2E 后清理要求）、架构说明（四个 fencing 写边界、budgets 单一事实源、deadline 锁内终检、正文生命周期分离、strict serializer）、迁移说明（0011/0012、stop-the-world、禁止混跑）、使用指南（演示身份改用 `scripts/init_demo_principals.py`、503/409/410/500 行为）。
- 验证（2026-08-27，隔离 PostgreSQL 18.6 容器 `rag-kb-m1b4-fymawleh` @ 127.0.0.1:56848，三个独立临时库）：
  - SQLite/共享逻辑定向矩阵：`110 passed in 54.10s`。
  - 全部 PostgreSQL 条件矩阵（10 个测试文件，含 checkpointer/fencing/retention/task_repository/tool_protocol/document_component/Qdrant 三套件）：`24 passed in 25.38s`，0 skip。
  - 0012 独立空库 round-trip（SQLite 双向 + PostgreSQL upgrade/downgrade/upgrade）：`16 passed in 24.28s`。
  - 第一次全量（含全部隔离集成）：`441 passed in 237.97s`，0 skip。
  - 修复后最终全量 + JUnit：`441 passed in 260.17s`，0 skip 0 fail → `data/local/acceptance/pytest-stage9.xml`。
  - 严格真实 Research E2E（strict msgpack、独立库/collection/对象目录）：passed，轨迹精确四工具、document_versions=2（DB 观测）、citations=3 且服务端复核、结果可读 → `docs/acceptance/m1-research-e2e.json`。
  - 企业检索 benchmark（v2 数据集）：Recall@5=1.0、MRR=1.0、no-answer accuracy=1.0、ACL leak rate=0，cleanup collection/tenant/object_root 全部成功 → `data/local/benchmarks/enterprise-retrieval-stage9-sufficiency-aligned.json`。
  - 最终汇总报告 verdict=passed → `docs/acceptance/latest.json`；`git diff --check` 通过。
- 清理证据：本批容器及三个临时数据库随容器删除且确认不存在；`m1_e2e_b4fymawleh01` 与遗留 `stage3_component_123c8f3d...` collection 已删除，Qdrant 恢复基线三集合；`data/local/e2e/m1_e2e_*` 对象目录已删除；调试残留 `probe_enum7.py` 及两份 `.bak` 备份已删除。正式保留验收证据仅限 stage9 JUnit、stage9 检索报告与两个 acceptance JSON。

## M2 批次 0：0013 与协议事实（已完成并验证）

- 新增 `20260827_0013_m2_approval_and_artifact_lifecycle.py`：approvals 增加 thread/checkpoint_ns/checkpoint_id/interrupt_id/tool_call_id（LangGraph checkpoint action ID，非 tool_calls 外键，FK 循环已拆除）与 rejected_by/rejected_at/rejection_reason；artifacts 增加 content 生命周期列；agent_tasks 增加协议绑定与 interrupt_ref。
- `waiting_for_approval` 的 interrupt_kind/ref/summary CHECK 完整性、`uq_approvals_active_checkpoint_binding` 部分唯一索引（同 action 重复同步必须复用同一条 active approval）、执行侧 tool_calls→approvals 十列复合 FK（RESTRICT）均已落地。
- `claim_next`/`claim_user_resume`/`FencedPostgresSaver.put/put_writes` 校验 required capability/protocol（task-agent/m2-fenced-v1）；strict msgpack 由根包 bootstrap + serializer fail-close 双保险。
- 验证：全量 `445 passed / 0 failed / 0 skipped`（PG 条件测试实际执行）。

## M2 批次 1：HITL 暂停与审批事实（已完成并验证）

### 已实现

- `build_controlled_action_agent`：`HumanInTheLoopMiddleware(interrupt_on=两个 WRITE, allowed_decisions=[approve,reject])`，位于 SingleBusinessToolCall 之后；`build_research_agent` 与 M1 四 READ 轨迹零改动。
- 两个 WRITE 薄 LangChain adapter（`build_controlled_action_tools` = 4 READ + 2 WRITE）：控制字段只取 `AgentRunContext`；模型注入 tenant/roles/approval ID/idempotency key/thread/checkpoint/storage key 在工具边界 schema 拒绝。
- Worker interrupt 双分类严格分流（`_route_interrupt` 按 payload 形状）：clarification → waiting_for_user；HITL（action_requests+review_configs 精确两键）→ `_pause_for_approval`；未知形状/多 interrupt/多 action/未知工具/开放 decisions 全部 fail-closed。tool_call_id 从 checkpoint 最后一条 AI tool call 按 (name,args) 精确匹配派生。
- `ApprovalService.ensure_action_binding`：从 Evidence Ledger 服务端派生最小不可变 plan（仅 hash 引用，无原始参数），按 (plan_digest, evidence_digest) 幂等复用 revision。
- `request()` 信任边界收紧：tool/version 只从 ToolRegistry 派生（拒绝任意工具与伪造版本）；checkpoint 绑定幂等（同 binding 复用同一条 active approval，不符 `approval_checkpoint_conflict`）；`pause_for_approval` 同事务写 waiting_for_approval + 裁剪摘要 + interrupt_ref=approval_id。
- 发起人自批拒绝：`approve()` 内判定 `auth.user_id == agent_tasks.user_id` → `approval_self_rejected`，approval 保持 pending。
- `ToolRegistry` 接入 FormatChecker；worker 装配受控 Agent（惰性编译）。
- 事务边界（D6，本批做到第 2 步）：HITL checkpoint 落盘 → action binding + pending approval（各自 UnitOfWork，幂等）→ waiting_for_approval（state_version 乐观锁）；①②之间崩溃由新 Worker 从同一 interrupt 幂等补建。

### 验证证据（2026-08-28）

- 批次 1 五文件 `88 passed`（test_task_agent 19 / test_task_approval_recovery 15 新增 / test_approval_tool_execution 41 / test_tool_registry 6 / test_domain_contracts 7）。
- 锁定版本 HITL 契约以真实 `create_agent` + 假模型 + InMemorySaver 验证：interrupt 在 ToolNode 前触发、payload 形状精确、adapter 零调用。
- 全量 `495 passed / 0 failed / 0 skipped`（PG 条件测试实际执行：主测试库真实 Alembic 迁移至 0013，迁移库完全空库起步）；`git diff --check` 通过；临时 PG 测试库已删除。

### 已知遗留（批次 2 前置）

- `_reject_stale_approval`（tool_execution_service.py）自动拒绝缺 rejected_by/at/reason，对 checkpoint 绑定行触发 `ck_approvals_rejected_audit_complete` 失败——批次 1 暂停路径不触发（有测试证明），批次 2 必须首先修复。
- 受控任务若暂停为 waiting_for_user 暂无恢复入口（claim_user_resume 仅 RESEARCH）；批次 3 前无受控任务生产创建入口，属已知暂留态。

## 当前交付边界

- M1 批次 0-4 全部完成并验证，M2 批次 0-1 完成并验证（上述基线）；正式进度为 M2 批次 2 未开始。
- M2 批次 1 交付后 `waiting_for_approval` 是合法暂留态，恢复路径（受控 resume、D7 稳定 key、UNKNOWN 收敛）由批次 2 交付；未实现公开 approval API、controlled_action 创建入口、artifact content 路由与 retention。
- 0012/0013 的真实发布仍必须遵守 stop-the-world 顺序；本进度记录不替代正式发布 runbook。
