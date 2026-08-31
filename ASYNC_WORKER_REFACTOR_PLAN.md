# API/Worker 执行权收敛与 Worker 进程拆分计划

> 文档状态：实施中
> 创建日期：2026-08-30
> 适用仓库：`rag-enterprise-kb`
> 范围：仅包含“API 不再构造或运行 Task Worker”与“将三类后台任务拆成三个进程模式”
> 状态事实源：本文件的“执行状态表”

## 0. 后续会话必须遵守的回读协议

主 Agent 必须完整遵守以下循环，不能仅依赖对话上下文或子 Agent 摘要：

1. 开始任何节点前，完整读取本文件。
2. 一次只能把一个节点标为 `进行中`。
3. 将该节点委派给符合复杂度要求的子 Agent；主 Agent负责读取实际改动并验证，不能只采信子 Agent 自述。
4. 节点验证结束后，立即更新本文件中的状态、验证证据、实际改动文件和遗留风险。
5. **完成一次节点更新后，主 Agent 必须再次从头完整读取本文件，再决定下一个节点。**
6. 如果实现事实与计划冲突，先在“决策记录”中记录原因并更新计划，再继续编码。
7. 不得同时推进会修改同一文件或同一数据库状态机的节点。

本协议的目的，是避免长会话压缩、子 Agent 切换或多轮实现后偏离两个核心目标。

## 1. 问题定义

### 1.1 API 内运行 Agent

当前 `POST /v1/tasks/{task_id}/continue` 和 `POST /v1/approvals/{approval_id}/decision` 通过 `build_task_resume_worker()` 在每次 HTTP 请求中构造完整 `TaskWorker`，然后同步调用 `resume_for_user()` 或 `resume_for_approval()`。

这使 API 与后台 Worker 同时拥有 Agent 执行权，并带来以下问题：

- HTTP 生命周期包含模型、检索、工具和 checkpoint 执行，响应时间不可控。
- API 请求重复构造模型、BGE、Qdrant、工具注册表和 PostgreSQL checkpointer。
- capability lease 由 API 临时发布，弱化“后台 Task Worker 是唯一执行者”的语义。
- 恢复命令从接收、持久化、领取、checkpoint 到最终执行之间缺少清晰的单一所有者。
- 崩溃窗口、幂等决策与 fencing 的组合更难解释和验证。

### 1.2 单 Worker 进程轮询三类任务

当前 `rag_kb.worker` 在一个循环中轮询：

1. Agent task；
2. document ingestion；
3. projection reconciliation。

轮询只在一次 `run_once()` 返回后切换。模型调用、解析或 embedding 长时间阻塞时，其他队列会发生队头阻塞，并且三个任务域无法独立重启、扩缩容或设置资源。

## 2. 目标与非目标

### 2.1 目标

- API 只负责认证、授权、输入校验、并发版本校验和耐久命令写入。
- 只有 `task` 模式 Worker 可以创建并运行 Agent、发布 task-agent capability、恢复 LangGraph。
- 用户澄清和审批决定在 API 成功返回前被耐久记录；API 返回后 Worker 可独立消费。
- 重复请求、并发请求和 Worker 崩溃不产生重复受控副作用。
- ingestion、projection、task 可作为三个独立进程运行，并继续使用 PostgreSQL claim/lease/fencing。
- 不新增 Redis、Celery、Kafka 或新的分布式基础设施。
- 保留现有公开 API 的路径、输入安全边界和稳定错误语义；如响应时序必须改变，应以最小兼容方式记录。

### 2.2 非目标

- 不重写 LangGraph Agent、检索算法、ACL、Evidence Ledger 或工具业务逻辑。
- 不顺手进行 application/infrastructure 全量分层重构。
- 不增加多 Agent、MCP、代码执行、WebSocket 或流式 UI。
- 不为“看起来整洁”拆分无关的大文件。
- 不向外部服务发送仓库语料、测试数据、`.env` 内容或验收报告正文。
- 未经用户明确授权，不提交 Git、不推送远程仓库、不执行真实 DeepSeek 验收。

## 3. 不可破坏的现有不变量

实施时必须保留：

- PostgreSQL 是任务与权限的唯一事实源；Qdrant 只是可重建投影。
- 身份由服务端认证上下文注入，客户端不能注入权限、resume payload 或审批绑定。
- 用户澄清只恢复 `user_clarification` interrupt。
- 审批决定只恢复绑定的 `tool_approval` interrupt，发起人不能自批。
- approval、task、checkpoint、tool call、artifact 的既有 binding 不得削弱。
- capability/protocol、generation、lease、state version 与 checkpoint fencing 必须继续 fail-closed。
- LangGraph resume payload 只能由服务端根据已验证的耐久命令构造。
- 写工具继续满足未批准调用为 0、重复副作用为 0、UNKNOWN fail-closed。
- 命令正文、用户答案、审批原因不得进入日志、指标标签、trace attribute 或错误消息。
- `.env`、本地对象目录、临时数据库、临时 Qdrant collection 和用户资料均为受保护资产。

## 4. 目标架构

```text
客户端
  │
  ▼
FastAPI
  ├─ POST /tasks                 ──写入任务──────────────┐
  ├─ POST /tasks/{id}/continue   ──写入恢复命令──────────┤
  └─ POST /approvals/{id}/decision──写入决定+恢复命令────┤
                                                        ▼
                                                PostgreSQL
                                                 │   │   │
                         ┌───────────────────────┘   │   └──────────────────┐
                         ▼                           ▼                      ▼
                  task worker                ingestion worker       projection worker
                  Agent/Tools/Checkpoint      Parse/Embed/Index       Convergence/Repair
```

### 4.1 唯一执行权

- API 不导入 `rag_kb.worker.build_task_worker`。
- Web 依赖中不存在 `build_task_resume_worker`。
- API 进程不创建 Agent model、TaskWorker 或 task capability lease。
- `task` Worker 是 Agent 初始执行、用户恢复和审批恢复的唯一执行者。

### 4.2 耐久恢复命令

在编码前必须通过节点 A1 决定命令存储形态。默认候选为最小 `task_resume_commands` inbox/outbox 表；不得未经比较直接增加复杂消息系统。

命令至少需要表达：

- `command_id`、`tenant_id`、`task_id`、`kind`；
- 服务端认证后的 actor/principal；
- 期望的 task state version；
- 用户答案，或审批 ID、决定与可选原因；
- 幂等键/唯一约束；
- pending、claimed、consumed/failed 生命周期；
- claim owner、generation/lease 或等价 fencing；
- 创建、领取、消费与到期时间。

用户答案属于敏感业务正文。设计必须说明：

- 为什么必须耐久化；
- 最小化保存哪些字段；
- 如何避免日志、trace 和事件泄漏；
- 何时清除正文、保留哪些不可变 hash/审计元数据；
- 在“领取后、checkpoint 前”和“checkpoint 后、标记消费前”崩溃时如何恢复。

## 5. Agent 分工规则

| 复杂度 | 分配规则 | 典型任务 |
| --- | --- | --- |
| 高难度 | `terra max` | 状态机、事务、迁移、崩溃恢复、fencing、并发与跨模块设计 |
| 简单 | `luna max` | 机械性配置修改、文档同步、静态检查、无歧义测试补充 |

- 子 Agent 每次只接收一个有边界的节点。
- 主 Agent 在委派提示中必须写明：TASK、EXPECTED OUTCOME、REQUIRED TOOLS、MUST DO、MUST NOT DO、CONTEXT。
- 子 Agent 不得读取或输出 `.env`，不得访问真实外部模型，不能执行 Git 提交或推送。
- 如果运行环境没有 `luna max`，主 Agent必须明确记录“模型不可用”，并亲自完成简单节点；不得假称使用了 Luna，也不得因此把简单任务升级为无边界重构。
- 高难度节点在 `terra max` 不可用时停止实施并向用户说明，不擅自降低推理等级。

## 6. 执行节点

### 0A：建立基线与受保护资产清单

- 复杂度：简单
- 执行者：Luna Max
- 目标：记录工作树状态、当前测试基线、迁移 head、Compose 服务和相关文件，不改业务代码。
- 必须检查：
  - `git status --short`，区分用户已有修改；
  - `docs/acceptance/latest.json` 的代码绑定与测试数字；
  - 当前 Alembic head；
  - API/Worker 相关定向测试；
  - `.env`、`data/local/`、`output/` 等受保护范围。
- 验收：本文件记录基线；不覆盖或吸收无关修改。

### A1：恢复命令协议与事务边界 ADR

- 复杂度：高难度
- 执行者：Terra Max
- 目标：在编码前确定 API 到 Task Worker 的耐久命令协议。
- 必须比较：
  1. 独立 `task_resume_commands` 表；
  2. 在 `agent_tasks` 上增加单槽 pending command；
  3. API 直接写 LangGraph checkpoint 但不运行 Agent。
- 决策标准：幂等、并发、隐私、崩溃恢复、审批事务原子性、迁移成本、可测试性。
- 预期倾向：独立最小命令 inbox；若选择其他方案，必须用代码事实解释。
- 输出：在本文件“决策记录”中写 ADR，包括状态机、唯一约束、清理策略、错误映射和崩溃矩阵。
- 验收：没有未定义的领取/重放语义后才能进入 A2。

### A2：数据库迁移与命令仓储

- 复杂度：高难度
- 执行者：Terra Max
- 前置：A1 完成。
- 目标：实现最小 schema、ORM/领域类型和事务仓储。
- 必须包含：
  - upgrade/downgrade 与 PostgreSQL/SQLite 约束对齐；
  - 租户、任务、命令类型和状态约束；
  - 幂等唯一约束；
  - 原子 enqueue/claim/consume/retry；
  - 领取 fencing 与过期恢复；
  - 正文清理和不可泄漏测试。
- 验收：迁移 round-trip、仓储单测、真实 PostgreSQL 并发测试通过。

### A3：用户澄清 API 改为只入队

- 复杂度：高难度
- 执行者：Terra Max
- 目标：`POST /v1/tasks/{task_id}/continue` 不再运行 Agent。
- 必须保留：权限、答案长度限制、interrupt kind、expected state version、租户隐藏、重复请求语义。
- API 成功条件：恢复命令已耐久写入；不代表 Agent 已完成。
- 必须决定并测试返回状态：优先保持响应 Schema 兼容，并准确表达 queued/resume-pending 语义。
- 验收：测试能证明 API 进程没有构造 model、checkpointer 或 TaskWorker。

### A4：审批决定与恢复命令原子化

- 复杂度：高难度
- 执行者：Terra Max
- 目标：`POST /v1/approvals/{approval_id}/decision` 只提交业务决定并耐久入队。
- 必须保留：服务端 approval binding、自批拒绝、同决定幂等、相反决定冲突、过期与错误映射。
- 决策和 enqueue 必须处于同一事务，不能出现“审批已落库但永远没有恢复命令”的窗口。
- 客户端仍不得传入 LangGraph `Command`、tool arguments 或 checkpoint binding。
- 验收：API 单测与 PostgreSQL 事务故障测试覆盖原子性。

### A5：Task Worker 统一消费初始任务和恢复命令

- 复杂度：高难度
- 执行者：Terra Max
- 目标：只有后台 Task Worker 执行初始任务、用户恢复与审批恢复。
- 必须实现：
  - 公平领取新任务和恢复命令；
  - 从耐久命令构造服务端固定 LangGraph resume payload；
  - 领取后重新校验当前身份、task/approval/checkpoint binding；
  - 命令 generation 与 task/checkpoint fencing；
  - 两个关键崩溃窗口的安全重放；
  - consume/failed/expired 的稳定审计。
- 禁止：把客户端 JSON 原样传给 `Command(resume=...)`。
- 验收：已有 `resume_for_user`/`resume_for_approval` 的安全语义被迁移而不是绕过。

### A6：移除 Web 中的 Worker 构造路径

- 复杂度：简单
- 执行者：Luna Max
- 前置：A3-A5 完成。
- 目标：删除 `build_task_resume_worker` 及 API 对 `TaskWorker` 的依赖注入。
- 必须验证：
  - `src/rag_kb/web/` 不再导入 `rag_kb.worker` 或 `TaskWorker`；
  - API 启动不要求 DeepSeek/BGE/checkpointer；
  - 现有测试 fake 从 Worker fake 改为 command service/repository fake。
- 验收：静态依赖检查和 API 定向测试通过。

### A7：任务一崩溃矩阵与回归门

- 复杂度：高难度
- 执行者：Terra Max
- 目标：证明异步恢复没有削弱 M1/M2 安全边界。
- 至少覆盖：
  - API enqueue 前/后失败；
  - 重复 continue；
  - 同审批决定重试与相反决定竞争；
  - command claim 后、LangGraph checkpoint 前崩溃；
  - checkpoint 后、command consume 前崩溃；
  - Worker 被抢占、lease 过期、generation 变化；
  - 撤权后已排队命令 fail-closed；
  - 未批准写调用、重复副作用和跨租户泄漏均为 0。
- 验收：定向单测、PostgreSQL 集成测试、任务 E2E 通过；临时资源残留为 0。

### B1：Worker CLI 改为显式单队列模式

- 复杂度：高难度
- 执行者：Terra Max
- 目标：将 `--include-tasks` 替换为显式模式，例如：

```text
python -m rag_kb.worker --queue task --watch
python -m rag_kb.worker --queue ingestion --watch
python -m rag_kb.worker --queue projection --watch
```

- 每个进程只构造本模式需要的重量级对象。
- `task` 模式独占 Agent model、checkpointer、task capability 和 task retention。
- `ingestion` 模式独占 parser/embedder/indexing 与 upload recovery。
- `projection` 模式独占 convergence/repair。
- 禁止保留“默认悄悄轮询全部队列”的生产行为；错误队列名 fail-fast。
- 验收：每种模式的构造测试证明不会初始化其他模式依赖。

### B2：拆分恢复、清理与可观测性所有权

- 复杂度：高难度
- 执行者：Terra Max
- 目标：拆开当前聚合 `_run_recovery_window()`，明确每类后台维护任务归属。
- 必须形成所有权表，至少包含：
  - task capability heartbeat；
  - task/tool/knowledge-query 超时与 retention；
  - upload reconciliation；
  - ingestion queue snapshot；
  - projection repair；
  - worker last-loop、错误和队列指标。
- 同一维护动作默认只允许一个 Worker 类型执行；需要多实例并发时依赖已有数据库锁/幂等而不是进程内假设。
- metrics 的 service/worker kind 标签必须可区分三类进程，且不得包含高基数字段。
- 验收：恢复职责既不重复，也不遗漏；相关故障测试通过。

### B3：Compose 拆成三个 Worker 服务

- 复杂度：简单
- 执行者：Luna Max
- 目标：将单个 `worker` 服务替换为 `task-worker`、`ingestion-worker`、`projection-worker`。
- 必须处理：
  - 每个服务的启动命令、环境变量、volume 和 depends_on；
  - 只给 task-worker 传 DeepSeek 所需配置；
  - 健康检查/重启策略保持合理；
  - 多容器指标端口与 Prometheus 抓取目标不冲突；
  - API capability readiness 只由 task-worker 满足。
- 禁止新增 Redis/Celery/Kafka。
- 验收：`docker compose config` 通过，三个服务命令与资源边界符合 B1/B2。

### B4：运行文档与本地命令同步

- 复杂度：简单
- 执行者：Luna Max
- 目标：只更新受本次行为影响的 README、架构图、运行命令和故障排查说明。
- 必须说明：
  - API 接受恢复命令不等于任务同步完成；
  - 如何分别启动三个 Worker；
  - 哪个 Worker 缺失会影响哪项 readiness/能力；
  - PostgreSQL 仍是队列与事实源，无新增中间件。
- 禁止手写未经最新机器报告支持的测试数字或提交哈希。
- 验收：文档命令可执行，且没有继续引用 `--include-tasks`。

### C1：最终验收与交付清理

- 复杂度：高难度
- 执行者：Terra Max
- 目标：从干净、可识别的工作树完成最终回归，不自动提交或推送。
- 最低验证门：
  - API 定向测试；
  - task/approval/worker/ingestion/projection 定向测试；
  - migration round-trip；
  - PostgreSQL/Qdrant 集成测试；
  - 全量 pytest；
  - `docker compose config`；
  - privacy scan；
  - `git diff --check`；
  - 临时数据库、Qdrant collection、对象目录残留为 0。
- 最终检查：
  - API 代码树无 TaskWorker 构造；
  - task Worker 是唯一 Agent 执行者；
  - 三个 Worker 模式可以独立启动；
  - 文档与机器证据一致；
  - 本次新增且不再使用的临时文件已清理。

## 7. 执行状态表

状态只允许：`待执行`、`进行中`、`已完成`、`阻塞`。

| 节点 | 状态 | 执行者 | 验证证据 | 实际改动文件 | 遗留风险 |
| --- | --- | --- | --- | --- | --- |
| 0A 基线 | 已完成 | 主 Agent（Luna Max 不可用） | `rag-kb` 定向基线 88 passed；Alembic `20260829_0014 (head)`；Compose 静态清单已核对 | `ASYNC_WORKER_REFACTOR_PLAN.md` | 当前环境无 Docker CLI，无法执行 `docker compose config`；`latest.json` 绑定旧于当前 HEAD |
| A1 命令协议 ADR | 已完成 | Terra Max | ADR-002/003/004 静态一致性门通过；候选比较、双 lease 状态机、两崩溃窗口、隐私与错误语义均已定义 | `ASYNC_WORKER_REFACTOR_PLAN.md` | A2 须以 PostgreSQL 并发测试证明双行 claim、固定锁序与 checkpoint marker 同事务 fencing |
| A2 迁移与仓储 | 已完成 | Terra Max | 主 Agent：67 passed、7 skipped；`py_compile`、隐私扫描、`git diff --check` 通过；Alembic `20260831_0015 (head)` | `migrations/versions/20260831_0015_task_resume_commands.py`；`src/rag_kb/domain/resume_commands.py`；`src/rag_kb/infrastructure/database/models.py`；`src/rag_kb/infrastructure/database/task_repository.py`；`src/rag_kb/web/dependencies.py`；`tests/test_task_resume_command_migration.py`；`tests/test_task_resume_command_repository.py`；`tests/test_database_migrations.py`；`tests/test_postgres_task_repository.py` | 未配置 `RAG_KB_TEST_POSTGRES_URL`，7 个真库测试跳过；`.tmp/pytest-a2-main` 删除被安全审查额度错误拒绝，保留待后续获准清理 |
| A3 用户澄清入队 | 已完成 | Terra Max | 主 Agent：API/仓储定向回归 48 passed；`py_compile`、Worker 依赖扫描、响应隐私扫描、`git diff --check` 通过 | `src/rag_kb/web/routers/tasks.py`；`src/rag_kb/web/schemas/tasks.py`；`tests/test_v1_tasks_api.py` | A6 前共享 dependencies 中仍保留 approval API 使用的 resume worker 构造器；A3 路由已无该依赖 |
| A4 审批决定原子入队 | 已完成 | Terra Max | 主 Agent：审批/命令/恢复回归 96 passed、1 skipped；`py_compile`、Worker 依赖扫描、隐私扫描、`git diff --check` 通过 | `src/rag_kb/application/approval_service.py`；`src/rag_kb/infrastructure/database/task_repository.py`；`src/rag_kb/web/routers/approvals.py`；`src/rag_kb/web/schemas/approvals.py`；`tests/test_v1_approvals_api.py`；`tests/test_approval_decision_enqueue.py`；`tests/test_postgres_approval_decision_enqueue.py` | 未配置 `RAG_KB_TEST_POSTGRES_URL`，真实 PostgreSQL 原子回滚用例 1 skipped，C1 前补验 |
| A5 Task Worker 消费 | 已完成 | Terra Max | 主 Agent：真实 PostgreSQL A5 全集 36 passed；迁移 round-trip 1 passed；SQLite/API 核心回归 104 passed；`py_compile`、隐私扫描、`git diff --check` 通过 | `migrations/versions/20260831_0015_task_resume_commands.py`；`src/rag_kb/agent/checkpointer.py`；`src/rag_kb/application/task_worker.py`；`src/rag_kb/infrastructure/database/models.py`；`src/rag_kb/infrastructure/database/task_repository.py`；`tests/test_postgres_m2_agent_recovery.py`；`tests/test_postgres_task_repository.py`；A5 相关测试 | 首轮 SQLite 复验受系统临时目录 ACL 影响，改用仓库内独立 basetemp 后全过；临时 PostgreSQL 容器及三个测试库已删除；`.tmp` 留待 C1 统一清理 |
| A6 移除 Web Worker 构造 | 已完成 | 主 Agent（Luna Max 不可用） | 主 Agent：API/启动定向 26 passed；Agent 公共导入兼容 27 passed；Web 静态依赖扫描、`py_compile`、`git diff --check` 通过 | `src/rag_kb/web/dependencies.py`；`src/rag_kb/agent/__init__.py`；`tests/test_v1_tasks_api.py`；`tests/test_new_app_factory.py`；`scripts/smoke_m2_controlled_write_e2e.py` | 隔离 M2 E2E 仍保留旧同步 `200`/终态断言，须由 A7 改为 `202` 后由 Task Worker 消费；A6 测试 basetemp 留待 C1 统一清理 |
| A7 任务一回归 | 已完成 | Terra Max（`max`） | SQLite 129 passed、隔离 PostgreSQL 5 passed；主 Agent复核 144 passed；补充 live fault 17/17、完整 smoke 31 tasks/21 scenarios passed、A6/A7 回归 56 passed；隐私/依赖扫描通过，隔离资源残留 0 | `scripts/smoke_m2_controlled_write_e2e.py`；`src/rag_kb/web/routers/approvals.py`；`src/rag_kb/web/schemas/approvals.py`；`tests/test_v1_tasks_api.py`；`tests/test_v1_approvals_api.py`；`tests/test_task_approval_recovery.py` | 本次补充 live E2E 无需代码修复；历史 A6/A7/B1 basetemp 仍按 C1 精确清理 |
| B1 单队列 CLI | 已完成 | Terra Max（`max`） | Terra Max：Worker runtime 13 passed；主 Agent：13 passed；`py_compile`、旧参数扫描、`git diff --check` 通过 | `src/rag_kb/worker.py`；`tests/test_worker_runtime.py` | `_run_recovery_window` 的恢复/队列/指标所有权仍聚合，按 B2 拆分；Compose/文档旧命令由 B3/B4 同步；B1 basetemp 留待 C1 清理 |
| B2 恢复与指标所有权 | 已完成 | Terra Max（`max`） | Terra Max：32 passed；主 Agent：32 passed；`py_compile`、旧聚合入口扫描、低基数标签复核、`git diff --check` 通过 | `src/rag_kb/worker.py`；`src/rag_kb/observability.py`；`tests/test_worker_runtime.py` | Compose 三服务、独立指标端口与 Prometheus targets 留给 B3；B2 basetemp 留待 C1 精确清理 |
| B3 三 Worker Compose | 已完成 | 主 Agent（Luna Max 不可用） | 部署静态测试 6 passed；旧观测 job 扫描 0 处；`COMPOSE_DISABLE_ENV_FILE=1` 下 `docker compose config --quiet`、`git diff --check` 通过 | `docker-compose.yml`；`ops/observability/prometheus/prometheus.yml`；`ops/observability/grafana/dashboards/system-overview.json`；`ops/observability/grafana/provisioning/alerting/rag-kb.yml`；`tests/test_deployment_acceptance_static.py`；`ASYNC_WORKER_REFACTOR_PLAN.md` | 同步 `/v1/knowledge/query` 仍在 API 内使用 DeepSeek，但 B3 按明确边界不再向 API 传 DeepSeek 配置；问答链执行权需另行设计，未在 B3 扩展重构 |
| B4 文档同步 | 待执行 | Luna Max | - | - | - |
| C1 最终验收 | 待执行 | Terra Max | - | - | - |

### 7.1 0A 基线记录

- 执行环境不支持 Luna Max；依据第 5 节规则，0A 由主 Agent 亲自完成，后续简单节点同样处理并明确记录，不假称使用 Luna。
- 初始工作树：`main`，HEAD `2de9cd06380b3652ea2d23dde6b54cccd605be56`；`git status --short` 仅有用户提供的未跟踪 `ASYNC_WORKER_REFACTOR_PLAN.md`，实施中必须保留。
- `docs/acceptance/latest.json`：生成于 `2026-08-30T23:16:26.333542+08:00`，代码绑定 `ac18d8200ae2f2a7d13c635617d6a81d993ddc24`、干净工作树 hash，记录 `530 passed`、0 failure/error/skip；该绑定旧于当前 HEAD，只作历史基线，不冒充本次最终证据。
- Alembic：`20260829_0014 (head)`。
- 定向测试：`tests/test_v1_tasks_api.py`、`tests/test_v1_approvals_api.py`、`tests/test_task_worker_recovery.py`、`tests/test_task_approval_recovery.py`、`tests/test_worker_runtime.py`、`tests/test_ingestion_worker.py`、`tests/test_projection_reconciler.py`、`tests/test_ingestion_projection_guard.py`，在 `rag-kb` 环境使用仓库内临时目录执行，结果 `88 passed in 38.54s`；临时目录已删除。
- Compose 静态服务清单：`postgres`、`qdrant`、`api`、`migrate`、`qdrant-init`、单一 `worker`、`jaeger`、`otel-collector`、`prometheus`、`grafana`。当前环境无 Docker CLI，0A 未执行 `docker compose config`，须在 B3/C1 补验。
- 受保护资产：`.env`、`data/local/`、`output/` 均存在；仅检查存在性和文件类型，未读取、输出或改动内容。重点 API、Task Worker、仓储、迁移、Compose 与观测文件均存在。

### 7.2 A1 验证记录

- Terra Max 完成 ADR-002/003/004；主 Agent 读取实际补丁并按现有 `UnitOfWork`、`claim_user_resume`/`claim_approval_resume`、`FencedPostgresSaver`、approval binding 与稳定副作用 key 逐项复核。
- 静态一致性门确认：三候选已比较；独立 inbox、原子 task/command 双领取、固定锁序、交叉 generation fencing、首次 checkpoint 前重领、checkpoint 后仅结算、应用 marker、API `202 Accepted`、正文清理、PostgreSQL/SQLite 语义和 A2-A5 落点均已定义。
- 代码事实修订：API 不得查询 LangGraph checkpoint；低敏 checkpoint/interrupt binding 由 Task Worker 在 pause 时写入 PostgreSQL task 行，API 只从锁定的 task/approval 行复制服务端 binding。
- 一名 Terra Max 在最后一次回复时触发使用额度限制，但该轮精确补丁已在报错前落盘；主 Agent 已直接读取并验证实际文件，没有依据失败摘要宣称完成。

### 7.3 A2 验证记录

- 实现：新增 `20260831_0015` 迁移、恢复命令领域契约、`task_resume_commands` ORM 与仓储原语；加入 task 侧低敏 binding/active command 门闩、幂等 enqueue、task/command 原子双领取、过期重领、双 lease 续约/重试、同事务 marker 原语、consume/fail/expire 清正文及普通 task claim 排除。
- 主 Agent 逐项读取实际 migration、ORM、领域类型、仓储与测试；发现并让 Terra Max 修正三点：移除可独立提交 marker 的快捷入口、双领取显式 capability/protocol fail-closed、approval FK 改为父链级联。
- 主 Agent 独立验证：`tests/test_task_resume_command_repository.py`、`tests/test_task_resume_command_migration.py`、`tests/test_task_repository.py`、`tests/test_task_worker_recovery.py`、`tests/test_database_migrations.py`、`tests/test_postgres_task_repository.py` 共 `67 passed, 7 skipped`；跳过原因均为未配置真实 PostgreSQL 测试 URL。
- `py_compile`、正文日志/trace 静态扫描与 `git diff --check` 通过；Alembic head 为 `20260831_0015`。
- 非阻塞验证缺口：新增 PostgreSQL 并发双 lease 用例已存在但未执行，必须在 C1 前具备 `RAG_KB_TEST_POSTGRES_URL` 时补验；否则最终交付继续明确为风险。
- 清理：主 Agent 创建的 `.tmp/pytest-a2-main` 仍存在；精确删除请求被自动安全审查因使用额度限制拒绝，未尝试绕过，后续获准时清理。

### 7.4 A3 验证记录

- 实现：`POST /v1/tasks/{task_id}/continue` 不再注入或调用 `TaskWorker`，只调用 A2 的 `enqueue_user_resume_command`；命令事务提交后返回 `202 Accepted`，保持既有 task 响应字段并新增低敏 `resume_command {command_id, status: accepted}`。
- 安全与兼容：保留 `task:read` 权限、租户/所有者隐藏、答案 2,000 字限制、`user_clarification` kind、客户端 `expected_state_version`、额外 resume 字段拒绝和同 target 幂等/不同正文冲突；task 保持真实 `waiting_for_user`，不伪装 Agent 已恢复或完成。
- 主 Agent 逐项读取 tasks 路由、响应 schema、API 假件与测试实际补丁，并核对 A2 UnitOfWork 在仓储方法返回前完成提交；未发现 A4/A5 或 Worker 运行链混改。
- 主 Agent 独立验证：`tests/test_v1_tasks_api.py`、`tests/test_v1_approvals_api.py`、`tests/test_task_resume_command_repository.py`、`tests/test_task_repository.py` 共 `48 passed`；`py_compile`、精确 TaskWorker/worker/model/checkpointer 依赖扫描、响应敏感字段扫描与 `git diff --check` 通过。
- 静态扫描首次使用过宽的 `model` 模式，误命中 FastAPI `response_model` 并以退出码 1 停止；收紧为具体构造/导入符号后通过，该次失败不是产品代码缺陷。
- 当前边界：A6 前 `src/rag_kb/web/dependencies.py` 仍因 approval 路径保留 `build_task_resume_worker`；A3 路由自身已无 Worker/Agent/checkpointer 依赖，A4-A6 按顺序收敛剩余 Web 构造路径。

### 7.5 A4 验证记录

- 实现：`ApprovalService.decide_and_enqueue` 在一个 `UnitOfWork` 内完成审批决定和 `tool_approval` 恢复命令写入；候选读取只定位 task，真实状态更新固定按 task→approval→command 加锁。审批路由不再注入或运行 `TaskWorker`，成功返回 `202 Accepted` 与低敏 `resume_command`。
- 事务与 binding：pending 决定、同决定幂等复用、相反决定冲突、历史已决定但缺命令的补写，均在锁定 task/approval 后复核 task state version、tool approval interrupt、checkpoint namespace/ID、interrupt、approval、plan/evidence/principal 与现有 approval 完整 binding；命令插入后故障会回滚审批字段和命令行。
- 安全与兼容：保留 approver 权限、租户隐藏、自批拒绝、到期与稳定冲突语义；客户端仍不能提交 tool arguments、checkpoint binding 或 LangGraph command，reason 只进入耐久命令正文且不进入响应、日志或 trace。
- 主 Agent 读取实际 service、repository、router、schema 与 SQLite/PostgreSQL 测试补丁；独立运行审批 API、原子事务、工具执行、恢复和命令仓储测试，结果 `96 passed, 1 skipped`，跳过原因仅为未配置 `RAG_KB_TEST_POSTGRES_URL`。
- `py_compile`、审批路由 TaskWorker/model/checkpointer 精确扫描、敏感正文日志/trace 扫描与 `git diff --check` 通过。新增真实 PostgreSQL 插入后故障回滚用例保留到 C1 补验。
- 子 Agent 新增 SQLite fixture 时因关联对象未显式 flush 导致两次 FK 插入顺序失败，已用局部 flush 修正；生产事务测试和主 Agent 独立回归均通过，未以该夹具问题掩盖产品失败。

### 7.6 A5 验证记录

- 实现：`TaskWorker.run_once` 在恢复命令与普通初始 task 间按最早 eligible time 选择并对同时间候选轮换；恢复只接受仓储原子领取的 `ResumeCommandLease`，私有推进链不再接受原始 answer/decision 旁路。兼容 `resume_for_user`/`resume_for_approval` 也先耐久入队并领取，再走同一 Worker 消费链。
- 执行复核：Worker 注入前由仓储按 task→approval→command 锁序重读双 lease、capability/protocol、当前 principal 权限、task state、checkpoint/interrupt binding 与审批完整 binding；撤权、过期、generation/owner/lease 或 binding 变化均 fail-closed。用户与审批 resume payload 仅由服务端从耐久类型化命令构造。
- pause 与根 namespace：新 pause 从已验证 snapshot 写入 checkpoint namespace/ID、interrupt ID 和需要时的 approval ID；按 ADR-006，合法根 namespace `""` 原样保存并与 checkpoint/approval 逐项相等，NULL 历史 binding 仍不可恢复。未发布的 0015 migration、ORM CHECK 与仓储完整性谓词已同步修订。
- checkpoint 崩溃边界：`FencedPostgresSaver.put` 在其 PostgreSQL checkpoint 事务内验证 task/command 交叉 fence，将 `resume_command_id` 写入 checkpoint metadata，同时写 command applied marker 并清 `active_resume_command_id`；仓储只保留 session 级 marker helper，生产源码无独立提交调用。checkpoint 前过期命令由双 lease 精确重领；marker 已存在时 Worker 只结算 consumed 并清正文，不重复注入。
- lease 与结算：首次 marker 前 heartbeat 同步续约 task/command；marker 后恢复普通 task lease；consume/failed/expired 通过条件 fence 结算并清敏感正文。受控审批测试替身同步为先 marker、后 ToolNode 副作用的生产顺序，保留写工具幂等与 stale fencing。
- 第一名 A5 Terra Max 在实现中触发使用额度限制，主 Agent 未把半成品视为完成；显式启动第二名 `gpt-5.6-terra`、reasoning `max` 逐项审计接管。接管时定向基线有 17 个失败，均已按实际断链修复，未降低模型要求或扩大到 A6。
- Terra Max 扩展回归 `135 passed, 19 skipped`；主 Agent 独立运行 Worker、approval、仓储、迁移与 checkpoint fencing 定向集，结果 `103 passed, 22 skipped`。跳过项均需真实 PostgreSQL URL；用户虽重启 Compose，但当前 Codex PowerShell 找不到 Docker CLI，且未读取 `.env` 绕过限制。
- 主 Agent 读取实际 Worker、checkpointer、仓储、迁移约束和测试改动；`py_compile`、精确正文日志/trace 与原始 resume 参数旁路扫描、生产独立 marker 调用扫描、`git diff --check` 通过。首次过宽静态模式误中错误码/常量后已收紧，未将误报当产品失败。
- 用户要求本节点完成后暂停；A6 仍为待执行，Web shared dependency 的最终删除不属于 A5。A5 验证产生的仓库内 `.tmp/pytest-a5-*` 保留到 C1 统一清理。
- 补充真库门（A5 重新打开）：用户授权后，主 Agent 通过 Docker Desktop 绝对路径创建全新 `postgres:18.6` 临时容器，仅绑定 `127.0.0.1:54900`，并创建主库、迁移库、E2E 库。迁移库 round-trip 集合通过前 21 项；首次主库测试因未先 upgrade head 出现缺表失败，执行 Alembic `upgrade head` 后重跑得到 `27 passed, 9 failed`。这 9 项是产品/测试链需诊断的真实 PostgreSQL 失败，A5 状态因此恢复为进行中，不能以 SQLite 通过替代。
- 真库修复收口：第二名 Terra Max 精确定位到 LangGraph 同一 invoke 的后续 superstep 会继续携带恢复 command fencing 配置，而首次 checkpoint marker 后审批执行可合法推进 task `state_version`。`FencedPostgresSaver` 因此在 marker 前保持 command×task state/generation/active-command 完整交叉 fence；marker 后要求 marker 已存在、active command 已清空且 task lease generation/owner/protocol/有效期仍成立，再退回普通 task fence，不放宽首次应用边界。测试侧仅修正恢复命令 FK 清理顺序，并精确化失效审批与并发 loser 的稳定错误断言。
- 主 Agent 读取并审查上述生产与测试差异后独立复验：真实 PostgreSQL 仓储、审批原子入队、checkpointer、checkpoint fencing、M2 恢复、retention 与 tool fencing 共 `36 passed`；隔离空迁移库的 Alembic upgrade→downgrade→upgrade 用例 `1 passed`；SQLite/Worker/API A5 核心集使用仓库内独立 basetemp 后 `104 passed`。首次 SQLite 复验的 `78 errors` 全部来自系统 `%TEMP%/pytest-of-26219` ACL，非业务失败；更换可写 basetemp 后无失败。
- 静态收口：相关模块 `py_compile` 与 `git diff --check` 通过，未残留临时诊断输出；恢复正文/隐私扫描继续通过。临时 `postgres:18.6` 容器仅绑定 loopback，验收完成后已删除，容器内主库、迁移库和 E2E 测试库不可恢复；未读取或修改 `.env`，未调用真实 DeepSeek，未启动 A6。

### 7.7 A6 验证记录

- 执行环境仍不支持 Luna Max；依据第 5 节既定规则，A6 由主 Agent 亲自完成，没有假称使用 Luna，也没有扩大到 A7 的崩溃矩阵或异步 E2E 改造。
- 删除 `build_task_resume_worker` 及其 Web→`rag_kb.worker` 构造路径；任务 API 测试继续直接注入已有命令仓储假件，隔离 M2 脚本中不再引用或覆盖已删除的 Worker 依赖。
- 启动检查首次发现 `rag_kb.agent.__init__` 会因 `TaskResultService` 的共享结果类型导入而急切加载 Agent factory/tools；已使用标准模块惰性导出保留既有公共 API，使 `create_app(Settings(_env_file=None))` 在禁止导入 Worker、Agent factory/tools、checkpointer、BGE 与 DeepSeek 模块的独立进程中成功。
- 主 Agent 独立验证：`tests/test_new_app_factory.py`、`tests/test_v1_tasks_api.py`、`tests/test_v1_approvals_api.py` 共 `26 passed`；`tests/test_task_agent.py`、`tests/test_agent_checkpointer.py` 共 `27 passed`；相关文件 `py_compile`、Web Worker/TaskWorker 静态扫描与 `git diff --check` 通过。
- A6 未运行需要外部资源的真实 E2E；`scripts/smoke_m2_controlled_write_e2e.py` 仍有 A3/A4 后已过时的同步 `200` 与立即终态断言，A7 必须改为 API `202 Accepted` 后显式由 Task Worker 消费并覆盖崩溃矩阵。A6 新增的仓库内 basetemp 与既有 `.tmp` 一并保留到 C1 精确清理。

### 7.8 A7 验证记录

- Terra Max（推理强度 `max`）完整审计 A7 覆盖矩阵后复用既有崩溃/fencing 用例，只补真实缺口：continue enqueue 前失败不留命令、enqueue 后响应构造失败可重试复用同一命令、审批命令入队后撤销 approver 时 Worker fail-closed 且 WRITE/tool call 为 0。
- 审计发现审批 `rejection_reason` 仍通过 HTTP schema/路由回显，违反审批原因不得进入响应的不变量；已从公开审批响应移除，耐久审计字段与内部完整性约束保持不变，并补 decision/detail 均不泄漏原因的 API 回归。
- 隔离 M2 E2E 已改为异步语义：API 返回 `202 Accepted` 后断言 task 仍为 waiting、重复请求复用同一 command 且副作用为 0，再由 `TaskWorker.run_once()` 唯一消费；最终核对 command 仅一条、状态 consumed、正文清除，并扫描答案、审批原因和 payload hash 不进入响应/日志/工具表面。
- Terra Max 验证：SQLite A7 定向集 `129 passed in 100.51s`；A6 启动兼容 `9 passed in 4.14s`。全新 `postgres:18.6` 容器仅绑定 `127.0.0.1:50741`，迁移至 `20260831_0015` 后，双领取、旧 generation checkpoint fencing、审批入队回滚、撤权零副作用、并发决定单副作用共 `5 passed in 7.75s`；容器使用 `--rm` 停止。
- 主 Agent 逐文件读取实际差异并独立复验新 API/撤权/两崩溃窗口/A6 启动共 `9 passed in 13.20s`；相关文件 `py_compile`、Web 审批原因扫描、Web Worker/TaskWorker 依赖扫描和 `git diff --check` 通过。主 Agent 在沙箱外只读复查 Docker 端口 `50741`，无容器残留。
- 本节点没有调用真实 DeepSeek，没有读取 `.env`，没有启动 B1。完整 live M2 smoke 还需要隔离 Qdrant 和 fault JUnit，按 C1 最终门执行；A7 的仓库内 basetemp 与历史 `.tmp` 留到 C1 精确清理。

### 7.9 B1 验证记录

- Terra Max（推理强度 `max`）将 Worker CLI 改为必填 `--queue task|ingestion|projection`；缺失、非法值和旧 `--include-tasks` 均由 argparse fail-fast，不再存在默认混合轮询生产入口。
- 删除 `build_workers()` 与三队列 round-robin；`task` 只构造 TaskWorker 和 task retention，`ingestion` 只构造 ingestion Worker 与 upload reconciler，`projection` 只构造 projection reconciler。测试通过 monkeypatch 实际记录 builder 调用，证明其他模式组合根没有初始化。
- 为守住 B1/B2 边界，聚合 `_run_recovery_window()` 未拆分恢复、清理、队列快照或指标所有权，只允许非 ingestion 模式没有 upload reconciler 时跳过该既有子动作；正式所有权拆分留给 B2。
- Terra Max 验证 `tests/test_worker_runtime.py` 为 `13 passed in 45.01s`。主 Agent 逐行读取 `worker.py` 与测试实际差异，并独立复验 `13 passed in 7.64s`；相关 `py_compile`、生产旧 `include_tasks`/`--include-tasks`/`build_workers` 扫描和 `git diff --check` 通过。
- 本节点没有读取 `.env`、没有调用真实 DeepSeek、没有修改 A6/A7、Compose 或 README。Compose/部署静态测试与运行文档仍引用旧命令，分别留给 B3/B4；B1 仓库内 basetemp 留待 C1 精确清理。

### 7.10 用户要求的 A6/A7 复核记录

- B1 完成并全文回读后，主 Agent 按用户要求暂停 B2，重新读取 A6/A7 全部实际差异。A6 的旧 `build_task_resume_worker` 删除、Web/脚本生产引用清零、Agent package 惰性导出和独立进程 API 启动门均真实存在；A6 结论保持已完成。
- A7 的 API enqueue 前后失败、审批原因不回显、排队后撤权 fail-closed、checkpoint 前后崩溃用例和异步 M2 E2E 脚本改造均真实存在；生产 `build_task_resume_worker` 与 Web `rejection_reason` 静态扫描为 0。主 Agent 独立运行 A6/A7 相关 11 个测试文件，结果 `144 passed in 90.64s`。
- 复核同时发现状态事实冲突：A7 节点验收要求“任务 E2E 通过”，但 7.8 明确记录完整 live M2 smoke 未运行，只完成脚本语义改造与编译。该缺口不能留到 C1 后仍宣称 A7 已完成，因此 A7 已重新标为进行中；必须由 Terra Max `max` 完成隔离 E2E、处理真实失败并确认临时 PostgreSQL/Qdrant/对象资源清理后，才能再次关闭 A7。
- 用户明确授权 Docker 后，主 Agent 创建唯一 `postgres:18.6` 与 `qdrant:v1.19.0` 容器，分别仅绑定 `127.0.0.1:54492`、`127.0.0.1:54496`；Terra Max 使用 replay/fake 模型实际生成 fault JUnit `17/17 passed`，并运行完整 controlled-write smoke，报告为 `status=passed`、`cleanup.completed=true`、31 tasks/21 scenarios。报告证明 `202 → waiting → TaskWorker`、审批前写 0/最终写 1、未批准/重复/UNKNOWN 副作用 0、跨租户/非所有者 404，以及答案、审批原因、resume body/hash 泄漏均为 0。
- 补充 live 门同时复验 A6/A7 API/recovery `56 passed`；`py_compile`、Web Worker/TaskWorker 依赖扫描、公开 `rejection_reason` 扫描与 `git diff --check` 通过，无需代码修复。smoke/fault 自建数据库、Qdrant collection、对象根和临时 JUnit/JSON 目录均已清理；主 Agent 停止两个 `--rm` 容器后按名称与端口复查为空。A7 至此重新关闭，验收与状态一致。

### 7.11 B2 验证记录

- 将聚合 `_run_recovery_window()` 拆为唯一所有权边界：task Worker 独占 capability heartbeat/availability、task/tool/knowledge-query 超时、task retention 与 task queue snapshot；ingestion Worker 独占 upload reconciliation 与 ingestion queue snapshot；projection Worker 仅通过 `ProjectionReconciler.run_once()` 执行 projection repair。三类进程分别记录自身 last-loop 与错误。
- recovery、queue、capability、last-loop 与 worker error 指标均增加或保持固定集合的 `worker_kind`；Worker OTEL/log service 名按 `rag-kb-worker-{task|ingestion|projection}` 区分。所有外部输入在指标层被界定为固定低基数值。
- 接管审计发现并修正一项真实误报：task 的正常 `finished`/`waiting_*` 结果不再计为 worker error，仅稳定 `error_code` 存在时记录；同时补齐 task capability availability 的固定 `worker_kind="task"`。
- Terra Max 定向验证 `tests/test_worker_runtime.py` 与 `tests/test_observability.py` 为 `32 passed`。主 Agent 逐项读取实际生产代码与测试，独立复验同一组测试为 `32 passed in 9.17s`；`py_compile`、旧 `_run_recovery_window` 扫描（0 处）、高基数 `worker_kind` 路径复核与全工作树 `git diff --check` 通过。
- 本节点未修改 Compose、Prometheus 或文档；三服务部署、独立 metrics 端口/抓取目标留给 B3，B2 仓库内 basetemp 留待 C1 精确清理。

### 7.12 B3 验证记录

- 当前环境仍无 Luna Max，按第 5 节规则由主 Agent 亲自完成简单配置节点，没有假称使用 Luna，也没有扩大到 B4/C1。旧单一 `worker` 已替换为 `task-worker`、`ingestion-worker`、`projection-worker`，分别显式使用 `--queue task|ingestion|projection --watch`；API 不依赖任一 Worker 启动，task capability 仅由 task Worker 的数据库 lease 发布。
- 三个 Worker 均依赖健康的 PostgreSQL/Qdrant、已完成的 Qdrant 初始化和迁移，共享必要对象存储与只读离线模型缓存，并使用 `restart: unless-stopped`。DeepSeek base URL、model 与 API key 只传给 `task-worker`；三个内部 metrics 端口分别为 9464、9465、9466，仅 `expose` 到 Compose 网络而未映射宿主端口。
- Prometheus 使用 `rag-kb-task-worker`、`rag-kb-ingestion-worker`、`rag-kb-projection-worker` 三个独立 job/target；Grafana 系统面板分别查询并展示三类 Worker。`rag-kb-target-down` 对 API、三个 Worker 和 Collector 逐个检查 `up == 0` 与 `absent(...)`，不再以聚合掩盖单个 Worker 缺失；运行观测配置中的旧单 Worker job 扫描为 0 处。
- 仓库没有可复用且能反映 Worker 循环业务健康的探针，因此未添加 `echo`、固定成功命令或仅证明 metrics 线程存活的伪 healthcheck。当前由容器主进程退出状态、`unless-stopped` 重启策略、三个 Prometheus 抓取目标及 B2 的 worker last-loop/error 指标承担存活观测；真正的 Worker 业务健康探针属于独立设计，不在 B3 新造第二套机制。
- 静态部署测试已同步验证三服务存在、旧服务不存在、命令、依赖、对象卷、DeepSeek 隔离、内部 metrics 端口、Prometheus targets、Grafana 三 Worker 查询及 target-down 的 down/absent 覆盖。主 Agent运行 `conda run -n rag-kb --no-capture-output python -m pytest -q tests/test_deployment_acceptance_static.py`，结果 `6 passed, 1 warning`；warning 仅为现有 `.pytest_cache` 目录 ACL 导致 pytest 无法写缓存，不影响断言结果。
- Windows PATH 现可解析 `C:\Users\26219\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`。为保证 Docker Compose 不自动读取受保护的 `.env`，设置 `COMPOSE_DISABLE_ENV_FILE=1` 后运行 `docker compose config --quiet`，退出成功且无输出；旧 job 精确扫描与全工作树 `git diff --check` 同样通过。
- 实现事实冲突与遗留风险：同步 `/v1/knowledge/query` 仍通过 API 内的 `DeepSeekAnswerModel` 使用 DeepSeek 配置，而 B3 明确要求 DeepSeek 配置只传给 `task-worker`。本节点继续遵守部署配置隔离，不把同步问答链迁移或重构扩大进简单 Compose 节点；该端点在 Compose 默认配置下缺少真实 DeepSeek 凭据的风险如实保留，不以回传配置给 API 掩盖冲突。

## 8. 决策记录

### ADR-001：不引入外部消息中间件

- 状态：已确定
- 决策：继续使用 PostgreSQL 作为耐久任务/命令事实源。
- 原因：当前目标是收敛执行权和隔离 Worker，不需要 Redis、Celery 或 Kafka 才能解决。

### ADR-002：恢复命令存储形态

- 状态：已决定（A1）
- 决策：新增**最小、独立的** PostgreSQL inbox 表 `task_resume_commands`；它是 API 到 `task` Worker 的唯一耐久恢复命令通道。它不是通用消息总线，不承载初始 task、ingestion 或 projection 工作，也不引入 Redis、Celery、Kafka。
- 选择比较：

| 候选 | 结论 | 关键原因 |
| --- | --- | --- |
| 独立 `task_resume_commands` inbox | 采用 | 可以为每次恢复保留独立的幂等键、命令 lease/generation、审计和正文清理点；审批决定与入队可在同一数据库事务提交；可在 checkpoint 事务中留下命令应用标记。 |
| `agent_tasks` 单槽 pending command | 拒绝 | 同一 task 的并发 continue 会竞争、覆盖或丢失正文；一个 task lease 无法同时表达命令领取 lease，难以区分“已恢复但尚未结算”的崩溃窗口；清除敏感正文会破坏 task 行语义，保留又会扩大泄漏面；也不能提供可审计的幂等重放记录与公平队列项。 |
| API 直接写 LangGraph checkpoint | 拒绝 | checkpoint 写入是 Agent 执行边界的一部分，需要 task capability、task generation、协议和副作用键 fencing。API 这样做仍须构造 checkpointer/Agent 执行上下文，既违反唯一执行权，也无法安全地产生服务端 `Command(resume=...)` 或证明审批 binding。 |

#### ADR-002.1：命令记录与幂等约束

`task_resume_commands` 只保存恢复所必需的、服务端验证后的字段；A2 必须把以下字段实现为 schema/领域类型/仓储契约，而不是允许客户端传入任意 JSON：

| 字段 | 约束与用途 |
| --- | --- |
| `command_id` | UUID 主键；只作不透明审计/响应标识，不能作为指标标签。 |
| `tenant_id`、`task_id`、`kind` | 非空；`kind` 仅为 `user_clarification` 或 `tool_approval`。所有查找均以 `tenant_id` 限定。 |
| `actor_principal_id`、`actor_kind` | 服务端认证后写入的最小主体标识；不是可被信任的权限快照。Worker 必须按该标识重新加载当前主体并重做授权。 |
| `expected_task_state_version`、`expected_interrupt_checkpoint_ns`、`expected_interrupt_checkpoint_id`、`expected_interrupt_id`、`expected_protocol_version` | 全部是服务端已验证的 fence。API 先在锁定的 `agent_tasks` 行比较客户端给出的 `expected_state_version`，匹配后才把该行的 state version、task 侧低敏 interrupt checkpoint namespace/ID、interrupt ID 和现有 protocol 值复制进 command；API 不构造、不读取也不查询 LangGraph checkpointer/snapshot。 |
| `approval_id`、`decision` | 仅 `tool_approval` 必填；`decision` 仅为已批准的决定枚举。审批的 tool call、arguments、evidence、发起人和 checkpoint binding 仍以锁定的既有 approval 行为唯一事实源；enqueue/双领取必须把它与 task 侧低敏 binding 交叉核对，不在命令中复制完整敏感 binding。 |
| `payload_body` | 仅用户答案，或审批可选原因；禁止保存原始请求 JSON、客户端 `Command`、tool arguments、证据正文或权限声明。`user_clarification` 必有答案且无审批字段；`tool_approval` 必有审批 ID/决定且无答案。 |
| `payload_sha256` | 对规范化后的最小正文及其 `kind`/决定计算的 SHA-256；用于完整性、幂等判定和终态审计，绝不写入日志、trace、事件或指标。 |
| `resume_target_key`、`idempotency_key` | 均由服务端计算。前者绑定 tenant、task、kind、服务端已验证的预期 version/checkpoint namespace+ID/interrupt 和 approval ID，**不含**正文；后者绑定 `resume_target_key` 与 `payload_sha256`/决定。 |
| `status`、`attempt_count`、`next_attempt_at`、`expires_at` | 命令状态、有限重试和服务器计算的到期边界。`expires_at` 取 task/interrupt/approval 的最早权威截止时间，且不晚于创建后 24 小时。 |
| `lease_owner`、`lease_generation`、`lease_expires_at`、`claimed_at` | 仅 Worker 的**双领取**设置。`lease_owner` 必须与关联 `agent_tasks.lease_owner` 相同；二者的 `lease_expires_at` 必须由同一次数据库时间计算出的同一值写入。generation 从 0 单调递增，任何后续写入都必须同时匹配 owner 与 generation。 |
| `claimed_task_state_version`、`claimed_task_lease_generation` | 双领取时写入 task 刚更新后的精确值：前者恒为 `expected_task_state_version + 1`，后者为该次 task lease 递增后的 generation。它们与 `lease_owner`/同源 expiry 共同构成 command↔task 的交叉 fence；每次合法重领都更新 task generation 值，但绝不第二次递增 state version。 |
| `agent_tasks.interrupt_checkpoint_ns`、`interrupt_checkpoint_id`、`interrupt_id`、`interrupt_approval_id` | task 侧低敏 resume binding。Task Worker 的 `pause_for_user`/`pause_for_approval` 只能从已验证的 snapshot 与当前 task lease 写入；前 3 项在任一 interrupt 下必填，其中 namespace 必须非 NULL 但允许 LangGraph 合法根命名空间空串，checkpoint ID 与 interrupt ID 必须非空白；`tool_approval` 另须 `interrupt_approval_id`，`user_clarification` 则该字段必须为 NULL。 |
| `agent_tasks.active_resume_command_id` | task 侧可空关联字段。双领取时设为 `command_id`，在首次 resume checkpoint 应用标记的同一 fenced 事务才可清空；只要非空，通用 task claim 不得绕过该命令正文。 |
| `applied_checkpoint_id`、`applied_at` | 耐久的 `command_id` 应用标记；仅可由首次 resume checkpoint 的 fenced 事务写入一次，并与清空 `active_resume_command_id` 原子提交。它存在时命令绝不能再次注入。 |
| `consumed_at`、`failed_at`、`expired_at`、`failure_code`、`body_purged_at`、`created_at`、`updated_at` | 低敏审计时间和稳定机器错误码；`failure_code` 不得包含异常详情、答案或审批原因。 |

为避免“只有 interrupt 摘要却无法安全定位恢复点”，A2 必须新增命名为 `interrupt_complete` 的 task 侧 `CHECK`：`interrupt_kind IS NULL` 时上述四个新 binding 字段全部为 NULL；`user_clarification` 时 checkpoint namespace 必须非 NULL（允许合法根命名空间空串）、checkpoint ID 与 interrupt ID 必须非 NULL 且非空白、`interrupt_approval_id` 必须为 NULL；`tool_approval` 时 namespace 必须非 NULL（同样允许空根 namespace），其余三项必须非 NULL 且非空白；其他组合一律拒绝。既有 `interrupt_summary/ref` 仍只作展示/引用，不能充当 resume binding。Task Worker 的 `pause_for_user`/`pause_for_approval` 在已验证 snapshot、现有 task lease fence 和（审批时）完整 approval binding 均成立后，原子写入 `waiting`、已有 interrupt 展示字段及这组低敏字段；API 永不写入或读取 checkpoint/snapshot。

约束必须同时覆盖“相同请求重试”和“不同正文抢同一 interrupt”：

- `UNIQUE (tenant_id, idempotency_key)` 使相同逻辑请求返回原 `command_id`，不创建第二条命令。
- `UNIQUE (tenant_id, resume_target_key)` 是比“仅活跃行部分唯一索引”更强的等价约束：同一 task version/interrupt/approval 终生只有一条恢复命令记录，终态审计仍可阻止旧请求在正文清空后重放。相同 target 但正文或决定不同，必须映射为稳定冲突，不能覆盖首条命令。
- A2 同时加入 kind/payload 形状、状态/时间和 `applied_checkpoint_id` 一次性写入的 `CHECK`/条件更新；PostgreSQL 与 SQLite 测试实现必须语义对齐。终态审计行至少保留到其父 task 的既有 retention 清理完成，之后才可随父审计链 purge，因此不会因早删唯一键产生重放孔洞。

#### ADR-002.2：全局锁序、enqueue 与原子双领取

凡是同时触及 task、approval 或恢复命令的路径，锁顺序固定为 **`agent_tasks → approvals → task_resume_commands`**。continue enqueue 锁 task 后锁 command；approval decision/enqueue 锁 task、对应 approval、再锁 command；取消、到期和撤权处理也必须按同一顺序锁住关联行后才更新终态或清理正文。不得从 command 反向锁 task。为同时保留 `SKIP LOCKED` 吞吐，Worker 可以先作不加锁的候选 ID 读取；真正的 claim UnitOfWork 必须依次锁 task（`FOR UPDATE SKIP LOCKED`）、必要时锁 approval、最后锁 command（`FOR UPDATE SKIP LOCKED`）并重新检查所有谓词。候选在重新检查前不构成领取，读到陈旧 ID 只会安全跳过。

API enqueue 在上述锁序内只通过 PostgreSQL 行完成静态验证和命令插入，**绝不构造、读取或查询 LangGraph checkpointer/snapshot**。客户端的 `expected_state_version` 仅是并发断言：API 必须先与锁定 `agent_tasks.state_version` 比较，匹配后才把该行的 version、`interrupt_checkpoint_ns`、`interrupt_checkpoint_id`、`interrupt_id` 和 protocol 值作为服务端已验证值复制到 command。approval decision 还必须在锁定 approval 行后，以 approval 的完整 binding 为事实源，交叉核对 task 侧的 namespace/ID/interrupt/approval ID；不能从 HTTP body 或 LangGraph 读取补齐。enqueue 提交后仍必须保持 `agent_tasks` 为 `waiting`：不发布 task capability、不写 task lease、不递增 task `lease_generation` 或 `state_version`，也不清除 interrupt。只有 Task Worker 的**原子双领取**才允许把 task 的当前绑定 interrupt 清除，并在同一 UnitOfWork 中完成以下操作；任一条件或更新失败都整笔回滚，正文绝不离开数据库：

1. 锁定后的 task 必须仍为 `waiting`、`state_version = expected_task_state_version`、满足 `interrupt_complete` 且不存在 `active_resume_command_id`；其 task 侧 checkpoint namespace/ID、interrupt ID 和 protocol 必须逐项等于 command 的服务端已验证字段。approval 命令还必须以锁定 approval 的完整 binding 为事实源，逐项与 task 侧低敏 binding 和 command 交叉核对，并通过决定、自批、发起人、tool/evidence/arguments binding 和当前授权复核。
2. 使用数据库当前时间一次性计算 `shared_lease_expires_at` 和同一个新的 `lease_owner`；将 task 改为 `running`，令 task `lease_generation + 1`、`state_version + 1`，写入 `active_resume_command_id = command_id`，并在同一更新中清除 `interrupt_kind/summary/ref` 及 `interrupt_checkpoint_ns`、`interrupt_checkpoint_id`、`interrupt_id`、`interrupt_approval_id`。除这组已验证 interrupt binding 外，不得清除或伪造其他 checkpoint 状态。
3. 在同一提交中将 command 从 `pending` 改为 `claimed`，令 command `lease_generation + 1`，写入与 task 完全相同的 owner/expiry、`claimed_task_state_version = expected_task_state_version + 1` 和 task 的新 lease generation。两行的 owner、expiry、关联 command ID 和交叉 generation 是一次 claim 的不可分割结果；首次应用标记前若需续约，也必须以同一 UnitOfWork 为两行写入同一个新 expiry，不能单边 heartbeat。

#### ADR-002.3：状态机、双 lease fencing 与崩溃恢复

`consumed` 表示恢复命令已被一个受 fence 保护的 checkpoint **应用**，不表示整个 Agent task 已完成。`pending` 只适用于尚未双领取的命令；一旦 task 已从 `waiting` 进入 `running`，不得把 command 单独退回 `pending`。

| 当前状态 | 条件/动作 | 下一状态 |
| --- | --- | --- |
| `pending` | 完成 ADR-002.2 的全部锁定复核和双行条件更新 | `claimed` |
| `pending` | 在 task 仍未领取时发现权威到期、取消、撤权或静态 binding 不可恢复；按全局锁序结算，且不清 interrupt | `expired` 或 `failed` |
| `claimed` | 当前 owner 在首次 checkpoint 前遇到可重试故障 | 保持 `claimed`；只能在持有两边同源 lease 时重试，或原子缩短两边 lease 等待恢复，绝不单独释放 command |
| `claimed` | 两边 lease 均已过期、应用标记为空，且满足下述精确重领条件 | 保持 `claimed`；由恢复命令路径原子换 owner、各自 generation `+1`，task state/version 不变 |
| `claimed` | 首次 resume checkpoint 与应用标记同一 fenced 事务成功提交 | `consumed`（可同事务或紧随其后的条件结算） |
| `claimed` | 发现已存在应用标记 | `consumed`（只结算，绝不再次构造 `Command`） |
| 无应用标记的 `claimed` | 取消、到期、撤权或不可恢复安全失败；按全局锁序同步处理 task/approval/command | `expired` 或 `failed` |

`consumed`、`failed`、`expired` 均为终态，不可回到 `pending`。`pending` 的首次领取/终态更新必须匹配其精确 target 与 `status = pending`；双领取后的 command 续约、缩短、应用标记、结算和终态更新必须匹配 `command_id + status = claimed + lease_owner + lease_generation`。与之配对的 task 更新还必须匹配 `task_id + running + state_version + lease_owner + lease_generation + active_resume_command_id`。被抢占或 lease 过期的旧 Worker 因而不能写 checkpoint 标记、consume 或覆盖新 owner 的结果。

**首次 checkpoint 前的唯一重领算法：**只有 `recover_claimed_resume` 命令恢复路径可以处理过期的 `claimed` 命令；通用 task crash recovery 不得处理它。该路径在同一 UnitOfWork 以固定锁序锁住两边，并仅在下列条件同时成立时重领：command 为 `claimed`、`applied_checkpoint_id IS NULL`、其 lease 已过期；task 为 `running`、`active_resume_command_id = command_id`、`state_version = expected_task_state_version + 1 = claimed_task_state_version`、task lease generation 等于 `claimed_task_lease_generation`、两边旧 owner 相同且两个 lease expiry 相同并均已过期。然后用一个新的共享 owner/expiry 原子更新两行，command 和 task 各自 generation `+1`，并把 command 的 `claimed_task_lease_generation` 更新为新的 task generation；task 保持 `running`，`state_version` 保持 `expected_task_state_version + 1`，不会回到 `waiting`、不会再次清 interrupt、也不会第二次递增 state version。任一侧不匹配时不得执行正文、不得改为普通 task claim 或尝试猜测恢复；保持 fail-closed，只有同锁序的取消/到期/撤权终态路径可安全解开该状态。

通用 `claim_next`（包括 running-task crash recovery）必须排除存在未终态、未应用恢复命令的 running task。实现至少需关联排除 `active_resume_command_id` 指向且 `status IN ('pending','claimed') AND applied_checkpoint_id IS NULL` 的 command；本 ADR 采用更强的规则：任何 `running AND active_resume_command_id IS NOT NULL` 的 task 一律不进入通用 claim。这样即使关联行异常缺失也 fail-closed，而不会绕过仍在 command 中的正文。普通初始 task claim 也不得选取仍为 `waiting` 且存在 pending resume command 的 task。

`FencedPostgresSaver` 写入**首次** resume checkpoint 时，必须在同一数据库事务内同时验证现有 task capability/协议 fence、task 的 owner/generation/共享 expiry/`active_resume_command_id`，以及 command 的 owner/generation、`claimed_task_state_version` 和 `claimed_task_lease_generation`。验证通过后，事务同时写入 checkpoint 元数据 `resume_command_id = command_id`、该 checkpoint ID 到 `task_resume_commands.applied_checkpoint_id`、`applied_at`，并清空 `agent_tasks.active_resume_command_id`。这两个互相可核对的字段与 task 侧解除门闩共同构成耐久应用标记；不能拆为 checkpoint 后的独立更新。此后 command 可按其 fence 结算为 `consumed` 并清空正文；task 已有可恢复 checkpoint，后续普通 running-task 崩溃恢复只继续 checkpoint，不再依赖 command 正文或再次注入 resume。

生产 PostgreSQL 的候选发现和锁定使用 `FOR UPDATE SKIP LOCKED`，但真实状态改变只能发生在上述按 task→approval→command 重新验证后的单一事务中；恢复扫描优先发现两 lease 皆过期的 `claimed` command，而不是把它单边改回 `pending`。SQLite 仅作为测试降级：用 `BEGIN IMMEDIATE` 串行化同样的锁序和条件 compare-and-set，不假装支持 `SKIP LOCKED`；真实并发、双 lease 抢占和 fencing 必须由 PostgreSQL 集成测试证明。

最低公平规则也属于 A5 的正确性契约：task Worker 在每次领取时必须在“可安全双领取的恢复命令”和“可普通领取的新初始 task”两类候选之间按最早 `eligible_at`（task 创建/重试就绪时间与 command `next_attempt_at`）选择；同一时间戳轮换 tie-break，不能让持续到达的恢复命令无限饿死初始 task，反之亦然。双 lease 条件不满足的恢复候选只能跳过或 fail-closed，不能被降级为普通 task；每次只领取实际将要执行的一项，不能预领另一项后让其 lease 空转。

### ADR-003：API 成功响应语义

- 状态：已决定（A1；A3/A4 按此实现）
- 决策：continue 与 approval decision 只在其事务已经持久提交恢复命令后返回 **`202 Accepted`**。`202` 只表示 API 已认证、授权、校验并耐久接受命令；不表示 Worker 已领取、LangGraph 已运行、审批工具已实际执行或 task 已完成。
- 兼容响应：保留现有成功响应模型的全部必填字段及 task 的真实当前状态；新增可选、向后兼容的 `resume_command` 子对象（至少包含不透明 `command_id` 与 `status: accepted`）。不得把尚未发生的 Agent 结果伪装成同步响应，也不得把答案、原因、hash、checkpoint/interrupt 或 tool binding 放入响应。相同幂等请求返回相同 `command_id` 和同样的接受语义。
- 稳定错误映射：现有认证、租户隐藏、权限、自批拒绝、输入长度、task/approval/interrupt/version/protocol 冲突的既有 HTTP 状态和错误码保持不变；A3/A4 必须继续复用现有领域异常映射。新增的同一 `resume_target_key` 不同正文、相反审批决定或终态不可恢复命令，统一映射到既有“不允许当前恢复/决定”的稳定冲突类别（不暴露是哪项 binding 失败）。Worker 不在线、排队或 lease 等待不是 API 错误，已经提交仍返回 `202`；事务未提交则不得返回成功，也不得留下“审批已决定但无命令”的可见状态。

### ADR-004：恢复命令正文保留与清理

- 状态：已决定（A1）
- 必要性：用户答案和审批原因不能只留在 HTTP 内存；API 返回后、Worker 尚未执行或崩溃重启时，只有耐久正文才能构造经过验证的服务端 resume。因此正文仅在 `task_resume_commands.payload_body` 中保存到“应用或安全终态”为止。
- 最小化与隔离：答案/原因之外不保存原始 request、headers、JWT、客户端 command、工具参数、证据正文或模型输出。ORM `repr`、审计事件、structured log、异常消息、trace attribute/span event 和 metrics label 都不得包含 `payload_body`、`payload_sha256`、`command_id`、approval 原因或用户答案；指标只能使用低基数的 kind/status/结果码。响应同样不得回显这些字段。
- 清理：`consumed`、`failed`、`expired` 在其状态转换的同一事务把 `payload_body` 置为 `NULL` 并写 `body_purged_at`；若接管者发现已应用标记，也在只结算为 `consumed` 时清空。保留 `payload_sha256`、逻辑键、actor/tenant/task/approval 标识、决定、时间、attempt、应用 checkpoint ID 和稳定失败码作为最小审计元数据，直至父 task 的既有 retention 清理；随后由同一受控清理链删除审计行。未终态正文的 `expires_at` 上限为权威业务截止时间与 24 小时中的较早者，避免无限期保留。
- 禁止的捷径：不能在 API 写 checkpoint、不能在 API 返回后再异步写命令、不能先清空正文再依赖日志或 trace 重建；也不能仅凭 command status 推断 checkpoint 是否已应用。唯一可靠的重放分界是由 `FencedPostgresSaver` 同事务写入的 `command_id` 应用标记。

### ADR-005：A2 到 A5 的 interrupt binding 迁移兼容

- 状态：已确定（A2 实现事实修订）
- 事实：A2 先于 A5；当前 `pause_for_user`/`pause_for_approval` 尚不会写入新增的低敏 checkpoint/interrupt binding。A2 若立即要求所有 waiting 行必须有完整新 binding，会使现有 pause 路径和基线测试在 A5 前失效。
- 决策：A2 的数据库 `CHECK` 只允许两种形状：四个新增 binding 字段**全部为空**（迁移前/过渡 pause 行），或按 ADR-002 kind **全量且相互一致**；任何半绑定一律拒绝。恢复命令 enqueue、首次双领取和重领的仓储谓词必须额外要求全量 binding，因此全空行绝不能进入新恢复通道，保持 fail-closed。
- 收敛：A5 必须让所有新 pause 写入全量 binding，并补测试证明新产生的 waiting 行不再全空；历史全空行不猜测 checkpoint、不自动恢复，只能按既有超时/取消路径安全终结。是否物理收紧 `CHECK` 留到历史空行清理后决定，不是本次架构目标的前置条件。
- 迁移绑定：A2 新增 Alembic revision 时，必须精确同步现有 `EXPECTED_DATABASE_REVISION` 常量；这是一行迁移一致性更新，不代表提前修改 API 执行行为。

### ADR-006：LangGraph 根 checkpoint namespace 的完整性语义

- 状态：已确定（A5 代码事实修订）。
- 事实：现有 LangGraph 根 checkpoint、审批恢复与 PostgreSQL checkpoint fencing 测试均使用 `configurable.checkpoint_ns = ""`；空字符串是合法根命名空间，不表示 binding 缺失。A2 的初版 `interrupt_complete` 实现若要求 `length(trim(namespace)) > 0`，会让 A5 将真实根 checkpoint 写入 task 行时违反约束。
- 决策：`interrupt_checkpoint_ns` 的完整性判断是 **非 NULL**，允许 `""` 并必须与 snapshot/approval binding 逐项相等；`interrupt_checkpoint_id`、`interrupt_id` 和需要时的 `interrupt_approval_id` 仍要求非 NULL 且非空白。不得用 sentinel 替换空 namespace，因为这会破坏 checkpoint、approval 与 task 的相等 binding。
- 落地：A5 在写入完整 pause binding 前，精确修订未发布的 `20260831_0015` 迁移 CHECK、ORM CHECK、仓储完整性谓词及相关迁移/仓储测试；迁移 revision/head 不变，不新增后续修补迁移。真实历史 NULL 仍按 ADR-005 fail-closed，NULL 与合法空串必须严格区分。

#### A1 后续实现落点（不在本节点编码）

| 后续节点 | 必须按 ADR 落地的边界 |
| --- | --- |
| A2 | 新增迁移、模型和仓储：命令字段、`agent_tasks.active_resume_command_id` 与 interrupt checkpoint namespace/ID、interrupt ID、approval ID 低敏 binding 列及 `interrupt_complete` CHECK；固定 task→approval→command 锁序的 enqueue/双领取/双 lease 重领；PostgreSQL `SKIP LOCKED` 与 SQLite 测试降级、正文 purge、交叉 fence 条件更新及 PostgreSQL 并发测试。 |
| A3 | 将 continue 改为仅执行“锁定 task 行并比较客户端 version → 从该行复制服务端 binding → enqueue → 提交 + 202”；API 不构造、读取或查询 Worker/Agent/checkpointer/snapshot，不改变客户端输入安全边界。 |
| A4 | 在单个 UnitOfWork 内按 task→approval→command 锁定/验证 approval 与 task，以 approval 完整 binding 为事实源并和 task 低敏 binding 交叉核对后写入同决定和命令；同决定重试返回既有命令，相反决定稳定冲突。对于历史上已决定但缺命令的异常记录，只能在锁定并完整复核 binding 后作一次可审计补命令，不能绕过校验。 |
| A5 | 让 task Worker 的 `pause_for_user`/`pause_for_approval` 从已验证 snapshot 与 task lease 填充 task 侧低敏 interrupt binding；再区分初始 task、首次命令双领取和过期命令双 lease 重领，执行复核/服务端 resume，把 task↔command 交叉 fence 与 `active_resume_command_id` 门闩传入 `FencedPostgresSaver` 的首次 checkpoint 事务，并按应用标记安全 settle，之后交还普通 running-task checkpoint recovery。 |

## 9. 重点文件索引

任务一核心：

- `src/rag_kb/web/dependencies.py`
- `src/rag_kb/web/routers/tasks.py`
- `src/rag_kb/web/routers/approvals.py`
- `src/rag_kb/application/task_worker.py`
- `src/rag_kb/application/approval_service.py`
- `src/rag_kb/infrastructure/database/task_repository.py`
- `src/rag_kb/infrastructure/database/models.py`
- `migrations/versions/`
- `tests/test_v1_tasks_api.py`
- `tests/test_v1_approvals_api.py`
- `tests/test_task_worker_recovery.py`
- `tests/test_task_approval_recovery.py`
- `tests/test_postgres_task_repository.py`
- `tests/test_postgres_m2_agent_recovery.py`

任务二核心：

- `src/rag_kb/worker.py`
- `docker-compose.yml`
- `ops/observability/`
- `tests/test_worker_runtime.py`
- `tests/test_ingestion_worker.py`
- `tests/test_projection_reconciler.py`
- `tests/test_ingestion_projection_guard.py`
- `tests/test_postgres_qdrant_projection_convergence.py`

## 10. 完成定义

只有同时满足以下条件，两个任务才算完成：

- Web/API 进程不构造、不注入、不运行 TaskWorker 或 Agent。
- continue/decision 的输入在 API 返回成功前已安全耐久化。
- Task Worker 能在崩溃与重启后唯一、安全地消费恢复命令。
- 现有审批、fencing、ACL、证据、artifact 与副作用安全不变量全部保留。
- task、ingestion、projection 三种 Worker 模式独立构造、运行、恢复和观测。
- Compose 不再使用单进程 `--include-tasks` 混合轮询。
- 不新增消息中间件或无关架构层。
- 全量与集成验证通过，临时资源清理完成。
- 本文件所有节点状态和证据已更新；主 Agent 在最后一个节点后也已完成一次全文回读。
