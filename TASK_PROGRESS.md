# 重构任务进度记录

> 最后更新：2026-08-28
> 当前阶段：M2 进行中——批次 0（0013 与协议事实）、批次 1（HITL 暂停与审批事实）、批次 2（双层恢复与副作用安全）、批次 3（公共 API 与正文生命周期）已完成并验证；批次 4（故障注入、严格真实 M2 E2E 与最终验收）未开始。M1 保持全部完成与验证，未回归。
> 开发依据：`AGENT_REFACTOR_PLAN.md`、`M2_PLAN.md`（未跟踪实施设计）；当前交付状态以本文为准。

## 执行约束

- M2 当前范围限定为受控 WRITE、HITL 审批、正文生命周期和最终验收；不进入 P2/P3、多 Agent、MCP、代码执行或长期记忆。
- 按 M2 计划固定顺序推进：批次 0 → 批次 1 → 批次 2 → 批次 3 → 批次 4。
- 0013 发布采用 stop-the-world：先停止任务 Worker，再迁移和部署，最后只启动兼容 `task-agent/m2-fenced-v1` 的 Worker；禁止旧协议 Worker 与 0013 新 Worker 滚动混跑。
- 不使用破坏性 Git 操作，不覆盖已有未提交改动；敏感连接信息只从环境变量或容器读取。
- Python 验证统一使用 `rag-kb` 环境的解释器，不混用全局 Python。

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
| M2 批次 2 | 已完成并验证 | 幂等 decision + 服务端受控 resume（同 thread）、D7 稳定 action key、claim_approval_resume/claim_next 受控重领取、三类崩溃窗口（真实 PG + FencedPostgresSaver）、失效矩阵 adapter 0、storage_key 全面收口（artifact_id）、TaskResultCandidate.artifact_ids 与受控 settle、reject 审计三元组；详见 M2 批次 2 小节。 |
| M2 批次 3 | 已完成并验证 | controlled_action 创建、审批 API、artifact content 与正文 TTL/purge 已交付；全量双库验证 `546 passed / 0 failed / 0 skipped`。 |
| M2 批次 4 | 未开始 | 故障注入、严格真实 M2 E2E、acceptance artifacts 与最终验收尚未执行。 |

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

## M2 批次 2：双层恢复与副作用安全（已完成并验证）

### 已实现

- **C-1 修复（含姊妹缺口）**：`_reject_stale_approval` 与 `approve()` 绑定失效分支均补齐 rejected_by/rejected_at/rejection_reason——自动失效用每租户固定服务端主体 `system:approval-reaper`（事务内确保 principals 行，满足 FK）+ 稳定 code `auto:<trigger>`；审批时发现失效用审批者身份。
- **幂等 decision**（ApprovalService）：approve 对 approved/executed 幂等返回当前状态；reject 新增（写完整审计三元组）；相反 decision 稳定冲突 `approval_decision_conflict`；过期稳定 `approval_expired`。新增 get/list/load_action_facts/load_approved_for_replay 查询能力（应用层，HTTP 留批次 3）。
- **服务端受控 resume**：`TaskWorker.resume_for_approval`——decision 先落库（幂等/冲突在此判定）→ `claim_approval_resume`（校验 waiting_for_approval + interrupt_kind=tool_approval + interrupt_ref==approval_id + m2 协议，领取即 generation+1/state_version+1，fence 通过）→ checkpoint 绑定逐项复核（checkpoint_ns/id、interrupt_id、tool_call_id 任一不符 fail-closed）→ 固定 `Command(resume={"decisions":[{"type":"approve"}]})` / 固定 reject 文案；客户端与模型零 payload。
- **D7 稳定 action key**：`stable_write_action_key` = canonical_fingerprint("agent_write_action", {task/thread/checkpoint_ns/checkpoint_id/interrupt_id/tool_call_id/tool_name/tool_version/arguments_hash/evidence_digest})，由 approval 行派生，不含 approval/run/worker/lease ID；经 AgentRunContext（新增 evidence_ids）服务端注入。
- **崩溃重放收敛**：`claim_next` 扩展受控任务领取（协议门隔离 M1/M2 Worker）；resume 已消费的重放路径由 `_replay_approval_facts` 从 checkpoint 待执行 tool call（tool_call_id 全局唯一，规避 LangGraph resume 写入的 checkpoint 副本 ID 漂移）反查 approved 审批，重建同一 context 与稳定 key；查不到则最终门 fail-closed。
- **storage_key 收口**：LocalArtifactToolAdapter 输出与 artifact_result schema 以 artifact_id 替换 storage_key（两处同步）；WRITE 专用 ToolMessage 渲染（只含 artifact_id/type/hash/status，修复批次 1 遗留 KeyError 面）；`agent_tasks.artifact_refs` 改写 artifact ID（列物理保留）；`ToolExecutionResult`/结构化结果/日志面无 storage_key。
- **结果契约**：TaskResultCandidate 新增 artifact_ids（候选声明，服务端不信任）；`_verified_artifacts` settle 前逐个校验 tenant/task 归属、status=ready、对象正文 sha256 与 content_hash 一致，任一失败 `task_artifact_binding_invalid`；`build_controlled_task_result` 服务端构造 result_kind="controlled_action" 与 artifacts 摘要；`finish_task` result_kind 按 task mode 最小放宽。
- **受控 reaper**：`expire_waiting_tasks` 扩展受控 waiting_for_approval deadline 超时 → failed（不设 content_expires_at，retention 留批次 3）。

### 事务边界与 D6 一致性窗口收敛（逐行落实）

| 窗口 | 收敛 | 幂等依据 |
|---|---|---|
| checkpoint 已写、approval 未写 | 批次 1 已交付：同 interrupt 幂等补建 | request() checkpoint 绑定幂等 |
| approve 已提交、resume 未调用 | 重复 decision 幂等 → 重新 claim/resume | 复用同 approval（active 复用）+ 同 action key |
| resume 已提交、ToolNode 前崩溃 | claim_next 重领取 → interrupt 重路由或 invoke(None) 重放 | `_replay_approval_facts` 反查 + 同 D7 key |
| tool 已执行、ToolMessage 未 checkpoint | 重放命中 tool_calls 唯一约束 → result_ref 回读 | uq_tool_calls_task_tool_idempotency + readback 校验 |
| checkpoint 完成、摘要未同步 | claim_next → settle 幂等修正 agent_tasks | finish_task 行锁原子终态 |
| approval 过期/撤权/evidence/参数/版本变化 | decision 阶段拒绝或最终门拒绝 | adapter 0；自动拒绝带完整审计 |

### 验证证据（2026-08-28）

- 先红后绿：批次 2 全部新增测试先以红灯落地（C-1 IntegrityError 复现、幂等 decision 报 approval_not_pending、四格真值表、resume 入口缺失、D7 key 缺失、storage_key 泄露断言、受控 finish research-only），再逐项转绿。
- 四格真值表 + 幂等/拒绝 decision + reject 固定 payload + 崩溃重领取 + D7 key context：test_task_approval_recovery.py `24 passed`。
- 受控 finish 幂等（task mode 放宽 + 终态不变）：test_task_worker_recovery.py 新增通过。
- 真实 PostgreSQL（双库协议：主库 0013 迁移 + 全空迁移库）+ FencedPostgresSaver + 真实 create_agent 契约假模型 + 真实 adapter：test_postgres_m2_agent_recovery.py `12 passed`——完整链路恰好一次、窗口 A（执行前崩溃恢复一次）、窗口 B（adapter 后 checkpoint 前崩溃 → result_ref 回读、adapter 总 1、同 tool_call/artifact）、窗口 C（checkpoint 后摘要前崩溃 → 只幂等修正 agent_tasks 不重跑图）、失效矩阵 6 场景（expiry/principal/ACL/evidence/参数/版本）adapter 0、WRITE timeout → UNKNOWN（迟到 adapter 不能恢复 ready、sweeper 0 重放、结构性不可重驱动）、并发 decision（恰好一条有效 action/一个 tool call/一个 artifact，另一方稳定冲突）、storage_key 不出现于结果正文与日志。
- result_ref 损坏 fail-closed（test_postgres_tool_protocol.py，tampered/missing 参数化）：WRITE 回读失败收敛 unknown、不凭 hash 伪装成功、adapter 不执行第二次；`525 passed / 0 failed / 0 skipped` 全量通过（PG 条件测试实际执行；基线 495 + 30 新增）；`git diff --check` 通过；临时 PG 测试库已删除。
- 实施中发现并修复批次 1 遗留 C-2 的真实形态：WRITE 成功结果渲染缺专用路径（artifact 字段缺失断言红灯证明）。

## 当前交付边界

- M1 批次 0-4 全部完成并验证，M2 批次 0-3 完成并验证；M2 批次 4 尚未开始，M2 尚未完成。
- 批次 3 已交付公开 controlled_action 创建、approval list/detail/decision、artifact content 路由与 retention；UNKNOWN 永不自动重放、客户端不能提交 graph resume payload。
- controlled_action 若进入 `waiting_for_user`，可通过 `/v1/tasks/{task_id}/continue` 在同一 `task_id` 恢复；批次 4 的故障注入与严格 E2E 必须覆盖该边界。
- 0013 的真实发布仍必须遵守 stop-the-world 顺序；本进度记录不替代正式发布 runbook。

## M2 批次 3：公共 API 与正文生命周期（已完成并验证）

- `POST /v1/tasks` 已公开 `controlled_action` 创建入口：feature gate 默认开启，创建事务同时要求兼容 `task-agent/m2-fenced-v1` capability；不可用时 503 且不插入任务。
- 新增 `/v1/approvals` 列表、详情与 decision API：独立 `approver` 演示身份仅持有 `task:approve`；审批仅返回安全摘要与绑定 hash，重复同决定返回当前状态，相反决定稳定 409，发起人自批为 403。
- 新增 owner + `task:read` 授权的 artifact content 读取；逐项复核 tenant/task/artifact 归属、ready、TTL/purge 与 SHA-256，正文响应不暴露对象键、filename、MIME、Range 或 Content-Disposition。所有请求验证错误使用脱敏 422，避免回显注入字段。
- controlled 终态、取消、waiting 超时和 WRITE UNKNOWN 均写入正文保留窗口；artifact purge 保留 hash/status 审计，递归 orphan 扫描覆盖 `artifacts/` 且保持 grace、游标、重试和前缀公平性。
- 验证：批次 3 六组 SQLite/HTTP 门禁全部通过；隔离 PostgreSQL retention、迁移 round-trip、真实 M2 approval/recovery/concurrency 路径通过；双库协议下全量 `546 passed in 308.91s`，`0 failed / 0 skipped`，`git diff --check` 通过。
- 上述是批次 3 回归验证，不代表 M2 已完成；严格真实 M2 E2E、acceptance artifacts 与最终机器报告仍待批次 4。
