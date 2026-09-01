# Application 边界收敛与双路径 Retrieval Core 统一计划

> 文档状态：已完成（F1）
> 创建日期：2026-09-01
> 适用仓库：`rag-enterprise-kb`
> 前置完成项：API 已退出 TaskWorker 执行链；task、ingestion、projection 已拆为三个 Worker 模式
> 本文范围：第 3 项“单向化 application/infrastructure 依赖并增加边界测试”；第 4 项“明确 Fast Path 与 Durable Path，统一 Retrieval Core”
> 本轮实施范围：仅 L0、L1、L2、L3、L4、L5、L6、F1；R1、R2、R3 保持待执行且不在本轮改动。

## 0. 结论先行

这两项不应做成一次“大搬家”。当前代码的真实情况是：

1. `application` 对 `infrastructure` 有 **44 条直接导入，分布于 12 个文件**。
2. `infrastructure` 对 `application` 有 11 条导入，分布于 8 个文件；在端口/适配器架构中，这个方向本身可以成立，但当前部分导入指向具体用例模块而不是稳定契约。
3. 9 个 `application` 文件共有 11 条 SQLAlchemy 直接导入。即使只清除 `rag_kb.infrastructure` 字样，application 仍然没有真正脱离持久化实现。
4. `application` 与 `agent` 也存在小型环：application 有 5 条 agent 导入，agent 有 2 条 application 导入。若完全忽略它，最终依赖图仍会有环。
5. Fast Path 和 Durable Path **已经共同使用 `EnterpriseRetriever`**；现在重复的不是检索算法，而是：
   - Web 与 task Worker 各自装配 BGE、Qdrant、授权端口与检索参数；
   - 两条路径的职责名称和适用场景没有形成清晰的公开契约；
   - 缺少测试证明两条路径消费的是同一检索策略，未来可能发生配置漂移。

因此采用以下策略：

- 用 AST 架构测试先建立“只能减少、不能新增”的棘轮。
- 按业务能力提取窄端口，不创建万能 Repository、万能 UnitOfWork 或服务定位器。
- 先收敛 Retrieval Core，因为风险低、收益直接，也能建立正确的端口范式。
- 再按 query、document/ingestion、task/approval 三个风险层级逐批清除 application 的具体基础设施依赖。
- 保留 Fast/Durable 各自的任务生命周期和证据交付语义，只共享授权检索核心及其配置。

## 1. 实施前保护边界

### 1.1 当前受保护工作树

规划时发现以下已有变化，不属于本文任务，实施时不得覆盖、恢复或纳入提交：

- 已删除但尚未提交的 `ASYNC_WORKER_REFACTOR_CONTINUATION_PROMPT.md`；
- 未跟踪的 `.tmp/` 及其历史测试目录。

后续开始实施前必须重新执行 `git status --short`，以当时工作树为准补充保护清单。

### 1.3 实施时基线（2026-09-01）

- 当前 HEAD：`2aea5ea23cc5400af8e3c694fd33cc7e9ad35f1f`；工作树仍有已删除的 `ASYNC_WORKER_REFACTOR_CONTINUATION_PROMPT.md`，本文档本身为未跟踪文件；`.tmp/` 及其历史测试目录存在且保持不处理。
- `application -> infrastructure`：44 条直接导入，分布于 12 个模块；application 的 SQLAlchemy 直接导入为 11 条，分布于 9 个模块；application 对 Qdrant 和 psycopg 的直接导入均为 0。
- `application -> agent` 为 5 条直接导入，分布于 3 个模块；`agent -> application` 为 2 条直接导入，均在 `agent.tools`。这是规划统计中唯一需要修正的依赖图基线。
- `pytest --collect-only -q` 已在 `rag-kb` conda 环境收集 578 个测试；尚未执行完整基线回归。
- `docs/acceptance/latest.json` 绑定 `ac18d8200ae2f2a7d13c635617d6a81d993ddc24` 与空工作树哈希，均不对应当前 HEAD/工作树，不能作为本轮验收证据。
- Alembic 脚本 head 为 `20260831_0015`；未读取 `.env`，且当前进程未提供迁移数据库连接，因此 `alembic current` 无法读取数据库 revision（`migration_database_url_required`）。
- `docker compose config --quiet` 已通过。

### 1.4 F1 正式验收基线（2026-09-02）

- 基线时状态：进行中。开始 F1 时 HEAD 为 `2aea5ea23cc5400af8e3c694fd33cc7e9ad35f1f`（`main`）；工作树含 L0-L6 的大规模未提交源码、测试、脚本与本计划修改，尚未建立可绑定的源码提交。
- 受保护资产：已删除但未提交的 `ASYNC_WORKER_REFACTOR_CONTINUATION_PROMPT.md`，以及来源不明、当前有访问拒绝的 `.tmp/` 历史目录；两者均不恢复、不删除、不纳入提交。
- AST 扫描的 `application→infrastructure`、`application→SQLAlchemy/Qdrant/psycopg`、`application→rag_kb.agent`、`domain` 向上依赖、`infrastructure/agent→web/worker/main` 均为 0，`TEMPORARY_ALLOWLIST` 为空。`application/task_worker.py` 对 LangGraph/LangChain 的直接使用经架构决策明确允许：它只编排框架公开 API，不依赖项目内 Agent 具体实现；不为该框架隔离新增 runtime port。
- 初审发现两条旧维护路径的锁序与主路径不一致：`DatabaseToolExecutionLifecycle.expire_stale_running()` 为 tool→task，`complete()` 为 task→tool；`DatabaseApprovalLifecycle.reject()` 为 approval→task，而决定入队路径为 task→approval。先记录为 F1 风险，修复后以真实 PostgreSQL 回归验证。
- 当时已验证环境：Alembic script head 为 `20260831_0015`，`COMPOSE_DISABLE_ENV_FILE=1` 下 `docker compose config --quiet` 通过。后续验收结果见 F1 当前执行记录；不执行 R1/R2/R3。

### 1.2 安全约束

- 不读取、输出、提交 `.env`。
- 不把仓库语料、用户答案、审批原因、验收正文或凭据发送给外部服务。
- 不削弱租户/ACL、Evidence Ledger、task/checkpoint fencing、HITL、幂等或副作用恢复。
- 未经用户明确授权，不调用真实 DeepSeek、不提交 Git、不推送远程仓库。
- 不执行破坏性 Git 命令，不清理来源不明的 `.tmp` 历史目录。

## 2. 目标依赖图

箭头表示“左侧可以导入右侧”：

```text
web / worker / main（组合根）
      ├──────────────> application（用例、编排、端口）
      ├──────────────> infrastructure（端口实现）
      └──────────────> agent（LangGraph/模型适配）

infrastructure ──────> application.ports + domain
agent ───────────────> application.ports + domain
application ─────────> domain
domain ──────────────> Python/允许的纯数据依赖
```

### 2.1 最终硬规则

- `domain` 不导入 `application`、`infrastructure`、`agent`、`web`。
- `application` 不导入 `infrastructure`、`web`、SQLAlchemy、Qdrant、psycopg。
- `application` 不依赖 Agent 的具体存储、工具或项目内 Agent 实现；`TaskWorker` 可直接使用 LangGraph/LangChain 的消息、`Command` 与图调用，避免只为隔离框架而新增转发端口。
- `infrastructure` 可以导入 `application.ports`、application 的稳定 DTO 和 `domain`，但不能导入 Web 或组合根。
- `agent` 可以导入 application 的端口和稳定 DTO，但 application 不能反向导入 agent 具体实现。
- `web`、`worker.py`、`main.py` 是组合根，可以同时看到 application、infrastructure、agent。
- `rag_kb.observability` 暂作为允许的横切模块；本计划不额外抽象 telemetry port。

### 2.2 禁止的伪重构

- 仅把 SQLAlchemy 代码移动到名字带 `port` 的文件。
- 创建包含几十个 CRUD 方法的 `RepositoryPort`。
- 为每个 ORM 表创建一套一比一 DTO，只为满足目录规则。
- 通过运行时动态导入、`Any` 或服务定位器绕过边界测试。
- 同时重写状态机、接口返回结构和目录布局。
- 为了“统一”把 Fast/Durable 强行合并成一个巨型 Service。

## 3. 端口设计原则

### 3.1 按用例能力设计，不按数据库表设计

推荐的端口粒度：

```text
application/ports/
  retrieval.py          授权快照、向量搜索、检索核心协议
  evidence.py           证据记录、引用复核、query 生命周期
  documents.py          文档注册、上传、版本与摄取工作项
  indexing.py           解析、索引、投影收敛契约
  tasks.py              task claim、状态转换、恢复命令、fencing DTO
  approvals.py          审批读取与原子业务转换
  object_store.py       最小对象读写/删除与稳定错误
  agent_runtime.py      Agent 调用、结果存储和运行上下文契约
```

这只是建议分组，不要求机械创建所有文件。实施者必须先依据调用链确认一个端口至少有两个价值之一：

- 隔离具体基础设施；
- 表达可测试的业务原子操作。

### 3.2 事务边界属于端口契约

task、approval、tool execution 中最重要的不是 CRUD，而是固定锁序、版本检查和同事务提交。端口方法应表达完整原子意图，例如：

- `approve_and_enqueue_resume(...)`；
- `claim_task_with_resume_command(...)`；
- `record_query_evidence(...)`；
- `finish_query_with_citations(...)`。

不得把一个原子操作拆成多个 application 调用，再期望 application 自己持有 SQLAlchemy Session 拼接事务。

### 3.3 DTO 归属

- 业务事实，如 `RetrievalCandidate`、`RetrievalResult`、授权版本快照，放在 `domain` 或稳定 application contract。
- 用例命令/领取结果，如 `ClaimedIngestionJob`、`ClaimedProjectionRepair`、task lease，放在 application contract/port。
- ORM Row、SQLAlchemy Session、Qdrant Point 不得越过 infrastructure 边界。
- `RankedChunk` 同时被检索核心和 Qdrant adapter 使用，应迁到稳定的 retrieval contract，而不是继续由 infrastructure 导入 `application.retrieval_fusion` 的实现模块。

## 4. Fast Path 与 Durable Path 契约

### 4.1 Fast Path

| 属性 | 契约 |
| --- | --- |
| 入口 | `POST /v1/knowledge/query` |
| 返回 | 同步返回 answer/refusal 与已复核 citations |
| 执行者 | API 进程中的 `KnowledgeQueryService` |
| 模型行为 | 单次受约束回答，不运行工具循环 |
| 状态 | 内部 `knowledge_query` 生命周期只服务于 evidence/fencing，不向用户表现为耐久 Agent task |
| 依赖 | PostgreSQL、Qdrant、embedder、回答模型；不依赖 task-worker capability |
| 适用 | 单问题、低步骤数、无需澄清/审批/产物的知识问答 |

### 4.2 Durable Path

| 属性 | 契约 |
| --- | --- |
| 入口 | `POST /v1/tasks`，后续通过 task/result/continue/approval API 交互 |
| 返回 | 创建与恢复命令返回 `202 Accepted`，结果异步读取 |
| 执行者 | 唯一的 task Worker |
| 模型行为 | 多步 Agent、受控工具、预算、checkpoint、HITL |
| 状态 | PostgreSQL task + resume command + LangGraph checkpoint 的耐久状态 |
| 依赖 | task-worker capability、Agent model、工具与相应持久化 |
| 适用 | 多文档研究、版本比较、用户澄清、审批、受控写入与产物生成 |

### 4.3 必须共享的部分

- `AuthContext` 与 `KNOWLEDGE_READ` 权限门。
- 活动版本、tenant、ACL subject 与 projection generation 过滤。
- dense/sparse 搜索、RRF、rerank、上下文预算和证据充分性策略。
- query fingerprint、候选 DTO、稳定排序和 insufficiency code。
- 相同 retrieval settings 的解析与生产装配。
- 检索阶段的低基数 telemetry 语义。

### 4.4 必须保持独立的部分

- Fast Path 的回答模型调用与同步 refusal 映射。
- Durable Path 的工具 Schema、ToolMessage 裁剪、预算和 Agent 循环。
- Fast Path 创建内部 query task 的生命周期。
- Durable Path 使用既有 Agent task 绑定 evidence。
- 最终响应 Schema、HTTP 时序、checkpoint/HITL 和 artifact 行为。

“统一 Retrieval Core”不等于统一上述独立职责。

## 5. Retrieval Core 目标设计

### 5.1 保留 `EnterpriseRetriever` 作为唯一算法核心

当前 `EnterpriseRetriever` 已负责：

1. 查询规范化与 fingerprint；
2. 权限和授权活动版本快照；
3. Qdrant dense/sparse 检索；
4. 融合、重排、SQL 精确复核；
5. 上下文裁剪与证据充分性判断；
6. 返回 `RetrievalResult`。

它已经是 Retrieval Core，不需要仅为命名再创建一层转发类。可以在文档中称其为“Retrieval Core”，类名暂时保留，降低无价值改名。

### 5.2 清除 Core 对 infrastructure DTO 的依赖

当前 `EnterpriseRetriever` 从 `infrastructure.database.acl_repository` 导入 `AuthorizedVersionSet`。应将该数据契约迁入稳定层，并让：

```text
EnterpriseRetriever -> application/domain retrieval contract
DatabaseAuthorizationPort -> 实现并返回该 contract
```

Qdrant adapter 当前从 `application.retrieval_fusion` 导入 `RankedChunk`。应把候选 contract 与融合算法解耦，使：

```text
QdrantAuthorizedSearch -> retrieval contract
retrieval_fusion       -> retrieval contract
EnterpriseRetriever    -> retrieval contract + retrieval_fusion
```

### 5.3 统一生产装配

当前以下两处重复创建 BGE、Qdrant、授权端口和全部检索参数：

- `src/rag_kb/web/dependencies.py::build_knowledge_query_service()`；
- `src/rag_kb/worker.py::build_task_worker()`。

应新增一个 infrastructure 侧生产工厂，例如：

```text
infrastructure/retrieval/factory.py
  build_enterprise_retriever(settings, database) -> EnterpriseRetriever
```

组合根仍负责：

- 创建/复用 Database；
- 决定生命周期与缓存；
- 把共享 retriever 注入 Fast QueryService 或 Durable tool registry。

工厂负责：

- 解析 candidate/context limit；
- 创建 embedder、Qdrant adapter、authorization adapter；
- 注入 token budget、RRF、hybrid、threshold；
- 对非法配置 fail-fast。

不得让 application factory 反向导入 concrete infrastructure。

### 5.4 共享结果，不共享证据事务

两条路径都消费同一 `RetrievalResult`：

- Fast Path 把候选写入其内部 query task 的 Evidence Ledger，再调用回答模型并构造 citations。
- Durable `search_knowledge` 工具把候选绑定到已有 task，再生成受限 ToolMessage 数据。

证据写入可以复用同一个 evidence port/adapter，但两个调用方的 task 生命周期不同，不应合并成一个巨型 `retrieve_and_answer_or_tool()`。

## 6. 架构边界测试

### 6.1 使用 AST，不增加第三方依赖

新增 `tests/test_architecture_boundaries.py`，使用 Python 标准库 `ast` 扫描 `src/rag_kb` 的 `Import` 和 `ImportFrom`。

测试至少覆盖：

- domain 禁止向上依赖；
- application 禁止导入 infrastructure/web；
- application 最终禁止导入 SQLAlchemy/Qdrant/psycopg；
- infrastructure 和 agent 禁止导入 web/worker/main；
- 组合根是明确例外，不用模糊的路径前缀放行。

### 6.2 先建立棘轮，再清零

初始测试可记录当前 application 违规模块集合：

```text
approval_service
artifact_content_service
document_service
enterprise_retriever
ingestion_worker
query_service
task_query_service
task_result_service
task_retention_service
task_worker
tool_execution_service
upload_reconciler
```

规则：

- 实际违规只能是 allowlist 的子集；出现新模块立即失败。
- 每完成一个批次，立刻从 allowlist 删除已经清理的模块。
- 最终删除 allowlist，改为零容忍。
- 不按行号或具体 import 文本放行，避免通过改写语句绕过。

### 6.3 架构测试不能代替行为测试

每个端口迁移批次仍须运行原有状态机、数据库、API 和集成测试。架构测试只证明依赖方向，不证明事务语义正确。

## 7. 实施节点

状态只允许：`待执行`、`进行中`、`已完成`、`阻塞`。一次只推进一个会改变同一调用链的节点。

### L0：基线与架构棘轮

- 复杂度：中等。
- 工作：
  - 记录当时的 git 状态、验收报告绑定和定向测试结果；
  - 新增 AST 边界测试及精确初始 allowlist；
  - 自动输出违规模块，便于每批收缩；
  - 不移动生产代码。
- 验收：测试能拒绝一个临时新增的 application→infrastructure import；撤销临时反例后通过。

#### L0 实施记录

- 状态：已完成。
- 实际改动文件：`tests/test_architecture_boundaries.py`、`LAYER_AND_RETRIEVAL_REFACTOR_PLAN.md`；未移动生产代码。
- 架构决策：allowlist 以 `(源模块, 目标模块)` 精确匹配而非行号或文本；除既定 infrastructure/持久化限制外，同时冻结现有 application→agent 具体依赖，确保 L6 前不会扩大循环。
- 验证证据：`conda run -n rag-kb --no-capture-output python -m pytest -q tests/test_architecture_boundaries.py tests/test_new_app_factory.py tests/test_worker_runtime.py` 通过（35 passed）；棘轮测试在 `tmp_path` 写入临时 `application→infrastructure` 导入并确认被拒绝，pytest 清理该反例；`tests/test_observability.py` 通过（8 passed）；`python -m compileall -q src tests/test_architecture_boundaries.py`、`docker compose config --quiet`、`git diff --check` 均通过。
- 遗留风险：未连接真实 PostgreSQL/Qdrant，完整回归尚留给涉及对应行为的后续节点；pytest 仅报告已有 `.pytest_cache` 写入权限警告，未产生本轮临时资源。

### L1：稳定检索 DTO 与端口

- 复杂度：中等。
- 工作：
  - 移动 `AuthorizedVersionSet`、`RankedChunk` 等跨边界数据契约；
  - 集中授权快照、向量搜索、reranker、sufficiency、retriever 协议；
  - `EnterpriseRetriever` 不再导入 infrastructure；
  - Qdrant/ACL adapters 改为实现稳定端口；
  - 不改变检索排序、阈值、ACL 或 telemetry。
- 验收：`enterprise_retriever` 从 allowlist 删除；安全、RRF、context packing、PostgreSQL/Qdrant pipeline 测试通过。

#### L1 实施记录

- 状态：已完成。
- 实际改动文件：新增 `src/rag_kb/application/ports/__init__.py`、`src/rag_kb/application/ports/retrieval.py`；修改 `application/enterprise_retriever.py`、`application/retrieval_fusion.py`、`application/query_service.py`、`infrastructure/database/acl_repository.py`、`infrastructure/vectorstore/qdrant_authorized_search.py`、`infrastructure/tools/{factory,read_tools}.py` 及相关测试。
- 架构决策：`AuthorizedVersionSet`、`RankedChunk` 和六个检索 Protocol 归属 `application.ports.retrieval`；`EnterpriseRetriever`、ACL/Qdrant adapter、Fast Query 与 Durable 只读工具均使用该稳定契约，不引入 R1 工厂或改变运行时装配。
- 实际依赖收缩：application→infrastructure 从 44 条降至 43 条；application 对 SQLAlchemy 仍为 11 条，Qdrant/psycopg 仍为 0；`enterprise_retriever` 已从架构 allowlist 删除。
- 验证证据：`conda run -n rag-kb --no-capture-output python -m pytest -q tests/test_architecture_boundaries.py tests/test_enterprise_retriever_security.py tests/test_stage3_retrieval_pipeline.py tests/test_qdrant_authorized_search.py tests/test_evidence_sufficiency.py tests/test_enterprise_query_service.py tests/test_read_tool_adapters.py tests/test_v1_knowledge_api.py tests/test_worker_runtime.py` 通过（80 passed）；其中检索安全测试继续断言 secret 不进入指标或 span。`python -m compileall -q src ...` 与 `git diff --check` 通过。
- 遗留风险：真实 Qdrant 组件、PostgreSQL/Qdrant pipeline、投影收敛和 PostgreSQL ACL 测试均因本机服务未运行或 `RAG_KB_TEST_POSTGRES_URL` 未配置而跳过（共 4 项）；未创建 collection、数据库或对象资源。

### R1：统一 Retrieval Core 生产工厂

- 复杂度：中等。
- 工作：
  - 创建唯一生产工厂；
  - Web Fast Path 和 task Worker Durable Path 都调用该工厂；
  - 删除两处重复的 BGE/Qdrant/参数装配；
  - 确认 Fast Path 不因此依赖 task Worker capability；
  - 确认 task Worker 仍只在 task 进程初始化 Agent 与检索运行时。
- 验收：静态扫描证明 Web/worker 不再直接各自拼装 `EnterpriseRetriever`；非法 retrieval config 两条路径一致 fail-fast。

### R2：双路径契约与一致性测试

- 复杂度：中等。
- 工作：
  - 新增 `tests/test_retrieval_path_conformance.py`；
  - 使用相同 fake authorization/vector ports 和配置；
  - 证明 Fast 与 Durable 消费相同 fingerprint、候选顺序、ACL 结果和 insufficiency code；
  - Durable 工具的 `limit` 只能截取 core 结果前缀，不能重新检索或改变排序；
  - 证明 Fast refusal 与 Durable ToolMessage 可以不同，但不能改变底层候选事实。
- 验收：注入配置漂移或候选重排反例时测试失败。

### L2：Fast Query 的持久化端口

- 复杂度：高。
- 工作：
  - 从 `KnowledgeQueryService` 移除 Database、UnitOfWork、AgentTaskRow、SQLAlchemy update 和 concrete EvidenceLedgerRepository；
  - 提取表达 query 生命周期和证据原子操作的窄端口；
  - infrastructure adapter 保留 ORM、锁、事务和 citations revalidation；
  - 保持同步租约/fencing、错误关闭和响应隐私语义。
- 验收：`query_service` 从 allowlist 删除且不导入 SQLAlchemy；现有 query service/API/撤权复核测试全过。

#### L2 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/knowledge_query.py`、`infrastructure/database/knowledge_query_lifecycle.py`；修改 `application/query_service.py`、`web/dependencies.py`、`scripts/benchmark_observability.py`、`tests/test_enterprise_query_service.py`、`tests/test_architecture_boundaries.py`。
- 架构决策：`KnowledgeQueryLifecyclePort` 只暴露三个完整业务操作：创建 fencing 保护的 query 与 evidence、同事务 citation 复核后完成、按 fencing 关闭终态。`DatabaseKnowledgeQueryLifecycle` 原样保留任务行写入、证据账本、SQLAlchemy compare-and-set 和 transaction 边界；没有将 Session、ORM Row 或 Evidence repository 泄漏到 application。
- 实际依赖收缩：application→infrastructure 从 43 条降至 39 条；application 直接 SQLAlchemy 从 11 条降至 10 条；`query_service` 已从架构 allowlist 删除。无 schema 变化，未新增 Alembic migration。
- 验证证据：`conda run -n rag-kb --no-capture-output python -m pytest -q tests/test_architecture_boundaries.py tests/test_enterprise_query_service.py tests/test_v1_knowledge_api.py tests/test_new_app_factory.py tests/test_benchmark_observability.py tests/test_stage3_acl_evidence.py tests/test_worker_runtime.py` 通过（78 passed），覆盖 SQLite lease/generation fence、撤权后 citation 拒绝、失败关闭、Fast API 身份隔离与 secret 不入日志/span；`python -m compileall -q src scripts ...` 与 `git diff --check` 通过。
- 遗留风险：`tests/test_postgres_stage3_acl_evidence.py`、`tests/test_postgres_task_repository.py`、`tests/test_postgres_qdrant_stage3_pipeline.py` 共 8 项因未配置 `RAG_KB_TEST_POSTGRES_URL` 且本机无服务而跳过；未创建数据库、collection 或对象资源。

### L3：文档、摄取与投影边界

- 复杂度：高。
- 工作：
  - 把 Document、Ingestion、Projection 所需 claim/receipt/guard DTO 与端口移到稳定 application contract；
  - `DocumentService`、`IngestionWorker`、`UploadReservationReconciler` 不再导入 concrete database/object store；
  - infrastructure parser/index/repository 只依赖端口 DTO，不导入具体 Worker 实现模块；
  - 保持 PostgreSQL 单一事实源与 Qdrant 三阶段投影协议。
- 验收：对应四个 application 模块从 allowlist 删除；摄取、投影收敛、世代 fencing、清理与真实 Qdrant 测试通过。

#### L3 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/{documents,indexing,object_store}.py`、`application/lease_heartbeat.py`、`infrastructure/database/document_lifecycle.py`；修改四个 L3 application 用例、document/projection/metadata 数据库 adapter、对象存储、parser、Qdrant adapter/init、Web/Worker 组合根、benchmark 与相关测试。
- 架构决策：文档注册、上传恢复、摄取、投影和对象存储分别通过窄稳定 contract 表达；`DatabaseDocumentLifecycle` 保留 ACL 快照、事务、PostgreSQL advisory lock、类级 SQLite `RLock`、lease/generation fencing 和三阶段投影绑定。`DocumentService`、`IngestionWorker`、`UploadReservationReconciler`、`ProjectionReconciler` 不再导入具体基础设施或 SQLAlchemy。
- 实际依赖收缩：application→infrastructure 从 39 条降至 27 条；application 直接 SQLAlchemy 从 10 条降至 9 条；`document_service`、`ingestion_worker`、`upload_reconciler` 已从架构 allowlist 删除，`projection_reconciler` 保持零直接基础设施依赖。无 schema 变化，未新增 Alembic migration。
- 验证证据：`conda run -n rag-kb --no-capture-output python -m compileall -q src scripts tests/test_architecture_boundaries.py` 通过；文档、摄取、投影、Qdrant、API、Worker 与架构定向测试共 171 passed，4 skipped；`docker compose config --quiet` 与 `git diff --check` 通过。测试继续覆盖上传 reservation 恢复、ACL 隐藏读取、删除顺序、lease/generation fence、投影收敛、临时 PDF 清理与安全错误码。
- 遗留风险：4 个真实 PostgreSQL/Qdrant 集成测试因 `RAG_KB_TEST_POSTGRES_URL` 未配置或本机服务不可用跳过；未创建数据库、collection 或对象资源。

### L4：只读查询、结果与正文服务边界

- 复杂度：中高。
- 工作：
  - 清理 `task_query_service`、`task_result_service`、`artifact_content_service`；
  - 对外只使用 task/result/artifact DTO；
  - ORM Row、LocalObjectStore 和 evidence repository 留在 infrastructure adapter；
  - 保持 owner/tenant 隐藏、hash 校验、409/410/500 映射。
- 验收：三个模块从 allowlist 删除，API 与 retention 相关测试通过。

#### L4 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/task_read.py`、`infrastructure/database/task_read.py`、`tests/test_task_query_service.py`；修改三个 L4 application 用例、`agent/result.py`、Web composition/API、M2 smoke 脚本、任务/产物测试和架构测试。
- 架构决策：`TaskQueryPort`、`ArtifactContentPort` 和 `TaskResultReaderPort` 分别表达完整只读操作；`DatabaseTaskQueryReader` 保留 tenant/owner 隐藏，`DatabaseArtifactContentReader` 保留对象 hash 与二次绑定复核，`DatabaseTaskResultReader` 保留当前 ACL/evidence citation 复核、结果绑定、行锁与 PostgreSQL 时钟。`TaskResult`、`ArtifactSummary` 迁入稳定 contract，`agent.result` 仅 re-export，避免 application 反向依赖 agent。
- 实际依赖收缩：application→infrastructure 从 27 条降至 19 条；application 直接 SQLAlchemy 从 9 条降至 6 条；application→agent 从 5 条降至 4 条。`task_query_service`、`artifact_content_service`、`task_result_service` 已从架构 allowlist 删除。无 schema 变化，未新增 Alembic migration。
- 验证证据：编译通过；任务查询、产物、结果、API、Worker、保留、恢复与架构定向测试共 186 passed，19 skipped；`docker compose config --quiet`、`git diff --check` 与 acceptance-report 测试通过。新测试锁定 `task_read_denied`、owner 隐藏和事件顺序。
- 遗留风险：19 个真实 PostgreSQL 集成测试因 `RAG_KB_TEST_POSTGRES_URL` 未配置或本机服务不可用跳过；L6 仍需清理余下 4 条 application→agent 依赖。

### L5：Task、Approval、Tool Execution 与 Retention 高风险边界

- 复杂度：最高。
- 工作：
  - 分别围绕 task execution、approval transaction、tool side effect、retention 提取原子端口；
  - application 不再持有 SQLAlchemy Session 或 ORM Row；
  - 固定锁序、state version、generation、lease、checkpoint fence、resume command fence 由 infrastructure 原子实现保证；
  - 不为了接口纯净拆散已验证的数据库原子操作；
  - 每个子域单独迁移和验证，不允许四个大文件一次性重写。
- 推荐子顺序：
  1. retention；
  2. task result/evidence helper；
  3. approval；
  4. tool execution；
  5. task worker。
- 验收：剩余 application→infrastructure 与 application→SQLAlchemy 导入清零；M1/M2 崩溃恢复、审批、tool fencing 和副作用测试通过。

#### L5.1 Retention 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/task_retention.py`、`infrastructure/database/task_retention.py`；修改 `application/task_retention_service.py`、Task Worker 组合根、SQLite/PostgreSQL retention 测试和架构测试。
- 架构决策：application 保留孤儿对象扫描、前向游标和有界重试队列；`DatabaseTaskRetentionLifecycle` 原样承接数据库时钟、`skip_locked` 领取、marker compare-and-set、引用/在途保护快照及 waiting task 的原子到期处理。对象删除与 checkpoint 清理继续先于 marker 提交，失败保持可重试。
- 实际依赖收缩：application→infrastructure 从 19 条降至 14 条；application 直接 SQLAlchemy 从 6 条降至 5 条；application→agent 从 4 条降至 3 条；`task_retention_service` 已从架构 allowlist 删除。无 schema 变化，未新增 Alembic migration。
- 验证证据：`compileall` 通过；retention、PostgreSQL retention、Worker maintenance、task repository/resume 与架构定向测试 78 passed，7 skipped；`git diff --check` 通过。SQLite 回归继续覆盖失败删除重试、孤儿游标轮转、引用保护、结果/产物 hash 与终态审计保留。
- 遗留风险：7 项真实 PostgreSQL retention 测试因 `RAG_KB_TEST_POSTGRES_URL` 未配置或本机服务不可用跳过；approval、tool execution 与 task worker 仍在 L5 后续子节点。

#### L5.2 Approval 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/approvals.py`、`infrastructure/database/approval_lifecycle.py`；修改 `application/approval_service.py`、ToolExecution 的 approval 消费桥接、Worker/Web/烟雾脚本组合根、approval router、架构测试及审批/恢复/工具测试。
- 架构决策：application 保留审批权限、注册表白名单、指纹和 telemetry；`DatabaseApprovalLifecycle` 原样承接 plan/evidence/主体复核、到期、reject 审计、task→approval→resume command 固定锁序及同事务命令入队。决定 API 不再注入 concrete repository，adapter 在组合时持有同一 `DatabaseTaskRepository`。
- 原子消费：当时 ToolExecution 通过 adapter 的 `consume_for_execution_in_session(...)` 在同一事务取得 `ApprovalExecutionBinding` 并将 approved→executed；ToolCall 意图插入失败会连同消费回滚，不再把 `ApprovalRow` 泄漏回 application。该过渡 bridge 已由后续 L5.3 移入 tool lifecycle 内部 helper。
- 实际依赖收缩：application→infrastructure 从 14 条降至 9 条；application 直接 SQLAlchemy 从 5 条降至 3 条；总违规从 22 条降至 15 条；`approval_service` 已从架构 allowlist 删除。无 schema 变化，未新增 Alembic migration。
- 验证证据：`compileall src scripts tests` 通过；approval/tool/decision/recovery/worker/repository/retention/架构定向回归 221 passed，29 skipped；最终核心回归 92 passed；`git diff --check` 通过。覆盖同事务 decision+command 回滚、幂等 decision、审批撤权、checkpoint binding、一次性 WRITE 消费、UNKNOWN 与 API 202 时序。
- 遗留风险：29 项真实 PostgreSQL 集成因 `RAG_KB_TEST_POSTGRES_URL` 未配置或本机服务不可用跳过；L5.3 仍须将 ToolExecution 的剩余 ORM/session 持久化逻辑整体移入其 lifecycle，届时移除当前同事务 approval bridge。

#### L5.3 Tool Execution 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/tool_execution.py`、`infrastructure/database/tool_execution_lifecycle.py`；修改 `application/tool_execution_service.py`、approval port/service/lifecycle、Worker 与 M2 smoke 组合根、架构测试和工具相关 SQLite/PostgreSQL/Agent/retention 测试构造点。
- 架构决策：application 保留注册表 schema/权限校验、adapter 超时/输出验证、对象结果序列化与 telemetry；`DatabaseToolExecutionLifecycle` 原样承接数据库时钟、task/主体/lease 双重 fencing、幂等查询、ToolCall intent、计划/证据复核、UNKNOWN 收敛、artifact/task 状态同步、结果 TTL 和 stale sweeper。
- 原子消费：审批执行复核从 `ApprovalLifecyclePort` 移除，成为 infrastructure 内部 helper；WRITE 的 approved→executed 与 `ToolCallRow(status=running)` 在同一 UnitOfWork 提交，失败一并回滚。对象读取前后仍分别复核租约/当前主体，最终绑定失败仅删除已确认未绑定的新对象。
- 实际依赖收缩：application→infrastructure 从 9 条降至 4 条；application 直接 SQLAlchemy 从 3 条降至 1 条；application→agent 仍为 3 条；总违规从 15 条降至 8 条。`tool_execution_service` 已从架构 allowlist 删除，无 schema 变化，未新增 Alembic migration。
- 验证证据：`compileall src scripts tests` 通过；Tool/approval/fencing/retention/Agent/Worker 定向回归 196 passed；全量本地回归 540 passed、45 skipped；`git diff --check` 通过。覆盖 WRITE 审批一次性消费、幂等回放、双重租约/身份 fence、deadline、UNKNOWN、迟到 artifact、对象补偿和 PostgreSQL DB clock 行为。
- 遗留风险：45 项真实 PostgreSQL/Qdrant 集成因 `RAG_KB_TEST_POSTGRES_URL` 未配置或相关服务不可用跳过；L5.4 仍须将 `TaskWorker` 的剩余 Session/ORM/evidence/object-store 依赖收敛为 task lifecycle/runtime ports。

#### L5.4 Task Worker 实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/ports/task_worker.py`（`TaskLease`/`ResumeCommandLease`/`TaskWorkerPersistenceError`/`TaskWorkerLifecyclePort`/`TaskWorkerSettlementPort`）；修改 `application/task_worker.py`、`infrastructure/database/task_repository.py`、`infrastructure/database/task_execution.py`（新增 `DatabaseTaskWorkerSettlement`）、Worker 与 M2 smoke 组合根、架构测试及 task/approval/tool/guards/worker runtime/PostgreSQL M2 恢复测试构造点。
- 架构决策：`DatabaseTaskRepository` 直接实现 `TaskWorkerLifecyclePort`（claim/resume 双 lease/refresh/pause/finish/expire 全部动作原样保留，无转发层）；完成期四段持久化（citation 复核、tool 计数、artifact 归属+SHA-256 校验、结果绑定确认）收敛为 `DatabaseTaskWorkerSettlement`，对象路径不泄漏回 application。`TaskRepositoryError` 保留为旧导入路径的别名（= `TaskWorkerPersistenceError`），Web 层与既有测试不受影响。
- 语义保持：resume 锁序（task→approval→command）、双 lease 同源 expiry、checkpoint marker 只结算不重注入、settle 顺序（身份复核→citation/artifact 复核→结果对象预写→finish_task→未绑定补偿）、绑定确认失败保守保留对象，均未改动。无 schema 变化，未新增 Alembic migration。
- 实际依赖收缩：application→infrastructure 从 4 条降至 0 条；application 直接 SQLAlchemy 从 1 条降至 0 条；application→agent 仍为 3 条（L6 目标）；总违规从 8 条降至 3 条。`task_worker` 的 infrastructure/SQLAlchemy 项已从架构 allowlist 删除，并新增 `test_task_worker_persistence_depends_on_stable_worker_contract` 锁定双向契约。
- 验证证据：`compileall src scripts tests` 通过；task/approval/tool/guards/repository/resume/worker runtime/架构定向回归 139 passed；全量本地回归 541 passed、45 skipped（新增 1 项架构契约测试）；`git diff --check` 通过。
- 遗留风险：45 项真实 PostgreSQL/Qdrant 集成因 `RAG_KB_TEST_POSTGRES_URL` 未配置或相关服务不可用跳过；L6 收口 `task_worker` 剩余的 `agent.context`/`agent.result`/`agent.tools` 三个 application→agent 导入。

#### L6 Agent 小环收口实施记录

- 状态：已完成。
- 实际改动文件：新增 `application/agent_contract.py`（`AgentRunContext`、`CONTROLLED_WRITE_TOOL_NAMES`、`TaskResultCandidate`、`AgentResultStore`、`build_task_result`、`build_controlled_task_result`，唯一变更：结果存储改依赖 `ObjectStorePort` 而非具体 `LocalObjectStore`）；`agent/context.py` 与 `agent/result.py` 改为纯 re-export shim；`agent/tools.py` 从 application 契约导入 WRITE 工具集合；`application/task_worker.py` 的三个 agent 导入替换为单一 application 导入；`agent/checkpointer.py` 的 `_TASK_CHECKPOINT_MSGPACK_TYPES` 增加 `rag_kb.application.agent_contract.TaskResultCandidate`；smoke/验收报告脚本的 strict msgpack allowlist 派生与相关测试同步。
- Checkpoint 兼容：已持久化 MsgPack checkpoint 仍以 `rag_kb.agent.result.TaskResultCandidate` 路径引用类型，shim re-export + serde 白名单双路径（旧路径读、新路径写）保证旧数据可继续反序列化；新增路径仅扩展 serde allowlist，不改 checkpoint 协议、prompt、工具 Schema 或预算。
- 实际依赖收缩：application→agent 从 3 条降至 0 条；总违规从 3 条降至 0 条；`TEMPORARY_ALLOWLIST` 清空，架构棘轮进入零豁免状态。新增 `test_agent_layer_consumes_stable_application_contract` 锁定 worker/shim 与 application 契约的依赖方向。
- 验证证据：`compileall src scripts tests` 通过；checkpoint serde、验收报告、架构、task agent、worker/approval/tool 恢复等定向回归 25+159 passed（含 serde 白名单双路径 round-trip）；全量本地回归 542 passed、45 skipped；`git diff --check` 通过。
- 遗留风险：45 项真实 PostgreSQL/Qdrant 集成因 `RAG_KB_TEST_POSTGRES_URL` 未配置或相关服务不可用跳过；F1 最终回归已完成，结果见 F1 当前执行记录。

### L6：收口 application/agent 小型循环

- 复杂度：高。
- 工作：
  - 将 `AgentRunContext`、TaskResult contract、结果存储、Agent 调用与工具集合表达为 application/domain contract；
  - agent 层实现 LangGraph/模型适配；
  - application TaskWorker 只依赖 Agent runtime/result ports；
  - `agent.tools` 可以依赖 application 的 tool execution port，application 不再反向导入 agent 具体实现；
  - 不重写 Agent prompt、工具 Schema、预算或 checkpoint 协议。
- 验收：application↔agent 环消失；Agent E2E 与序列化 allowlist 测试通过。

### R3：正式命名、文档与可观测性

- 复杂度：简单。
- 工作：
  - README 和 `docs/architecture.md` 明确使用 Fast Path/Durable Path 名称；
  - 添加场景选择表和两条执行时序；
  - 说明 `EnterpriseRetriever` 是共享 Retrieval Core；
  - 如需区分检索消费者，只允许固定低基数 `fast|durable` 属性，禁止 query/task/user 进入指标标签；
  - 不手写未经机器报告支持的测试数字或提交哈希。
- 验收：文档与路由、Worker 和 readiness 实际行为一致。

### F1：严格边界与最终回归

- 复杂度：高。
- 工作：
  - 删除架构测试 allowlist；
  - 全仓扫描依赖图和直接生产装配；
  - 执行单元、API、Agent、PostgreSQL/Qdrant、migration、Compose 和全量回归；
  - 运行隐私扫描、`git diff --check`；
  - 清理本轮新建的数据库、collection、对象目录和临时文件，不处理来源不明的历史 `.tmp`。
- 验收：满足第 10 章完成定义。

#### F1 当前执行记录（2026-09-02）

- 状态：已完成。源码提交为 `ed056ac08970493d9f3fbfcad72db34ec1cae37b`，验收报告仅绑定该 clean worktree。
- 静态门禁：`application/domain` 对 `rag_kb.agent`、`rag_kb.infrastructure`、`rag_kb.worker`、`rag_kb.main` 及 SQLAlchemy/Qdrant/psycopg 的直接导入均为 0；动态加载命中仅为架构测试的 `__import__` 拦截，`TEMPORARY_ALLOWLIST` 为空。`TaskWorker` 对 LangGraph/LangChain 框架 API 的允许依赖保持不变。
- 隐私初审：生产日志、trace、metrics 与 HTTP 错误边界未见正文、prompt、凭据或授权 header 直接泄漏；手工诊断/运维脚本的输出不在本节点扩围修改。
- 本地验证：设置 `RAG_KB_DISABLE_ENV_FILE=1` 后，架构、配置和 observability 定向集 `27 passed`；`compileall`、`git diff --check` 通过。隔离 PostgreSQL/Qdrant 的全部 `integration` 标记集在主库升级至 `20260831_0015`、checkpointer 初始化且独立空迁移库参与 round-trip 后为 `65 passed, 523 deselected`；clean worktree 全量 pytest 为 `588 passed`。
- M2 证据：controlled-write E2E 为 `status=passed`，覆盖 31 个任务、21 个场景、三类 crash recovery，严格 MsgPack 无 pickle，隐私计数与未批准/重复/UNKNOWN replay 副作用均为 0；脚本自证 PostgreSQL、Qdrant collection、对象根清理完成。
- M1 证据：获得用户明确授权后，仅将 `.env` 作为验证容器环境输入，未输出、复制或提交密钥；真实 DeepSeek smoke 为 `status=passed`，验证两版文档、四个固定 READ 工具、3 个引用与严格 MsgPack 无 pickle。
- Retrieval 证据：真实 BGE-M3/PostgreSQL/Qdrant benchmark 的 test split 为 Recall@5=`1.0`、MRR=`1.0`、no-answer accuracy=`1.0`、ACL leak rate=`0.0`；脚本自证随机 tenant、collection 与对象根清理完成。
- 正式报告：`docs/acceptance/latest.json` 为 `passed`，绑定源码提交与 clean-worktree SHA，记录 588 项全量、检索、M1 与 M2 的机器证据。本轮未执行 R1/R2/R3。

## 8. 执行状态表

| 节点 | 状态 | 主要产物 | 验证证据 | 遗留风险 |
| --- | --- | --- | --- | --- |
| L0 基线与棘轮 | 已完成 | `tests/test_architecture_boundaries.py` | AST 反例、35 项静态/API/Worker 测试、8 项 observability 测试、编译与 Compose 均通过 | 真实 PostgreSQL/Qdrant 回归待相关节点 |
| L1 检索 DTO/端口 | 已完成 | `application.ports.retrieval` | 80 项检索/API/Worker/架构测试通过；真实集成 4 项因环境未配置跳过 | 真实 PostgreSQL/Qdrant 回归待可用环境 |
| R1 统一生产工厂 | 待执行 | retrieval factory | - | - |
| R2 双路径一致性 | 待执行 | conformance tests | - | - |
| L2 Fast Query 持久化端口 | 已完成 | `KnowledgeQueryLifecyclePort` 与数据库 adapter | 78 项 Fast Query/API/ACL/Worker/架构测试通过；8 项 PostgreSQL 测试因环境未配置跳过 | 真实 PostgreSQL fencing/撤权回归待可用环境 |
| L3 文档摄取投影边界 | 已完成 | document/indexing/object-store ports 与 `DatabaseDocumentLifecycle` | 171 项定向测试通过，4 项真实集成因环境跳过；编译、Compose、diff 通过 | 真实 PostgreSQL/Qdrant 回归待可用环境 |
| L4 查询结果正文边界 | 已完成 | `application.ports.task_read` 与 `DatabaseTask*Reader` | 186 项定向测试通过，19 项真实 PostgreSQL 集成因环境跳过；编译、Compose、diff 通过 | L6 尚余 4 条 application→agent 依赖 |
| L5 高风险任务事务边界 | 已完成 | retention/approval/tool/task worker ports | L5.4 已清零 application→infrastructure/SQLAlchemy；L6 后全量本地历史结果 542 passed、45 skipped | F1 最终回归已完成；R1/R2/R3 按本轮范围保持待执行 |
| L6 Agent 小环收口 | 已完成 | `application.agent_contract` 与 agent shim/serde 双路径 allowlist | 全量本地 542 passed、45 项真实集成因环境跳过；编译、diff、serde round-trip 通过 | F1 最终回归已完成；R1/R2/R3 按本轮范围保持待执行 |
| R3 文档与观测 | 待执行 | 路径选择说明 | - | - |
| F1 最终回归 | 已完成 | 零 allowlist 与验收证据 | 静态门禁、27 项定向集、编译/diff、65 项 integration、588 项全量、真实 M1/M2 与 BGE-M3 检索 benchmark 均通过；`TaskWorker` 直接使用 LangGraph/LangChain 经决策允许 | R1/R2/R3 按本轮范围保持待执行 |

## 9. 测试矩阵

| 风险 | 必须验证 |
| --- | --- |
| 新增反向依赖 | AST 架构棘轮测试 |
| DTO 迁移导致序列化变化 | Pydantic/model dump、strict msgpack、schema 测试 |
| Retrieval 配置漂移 | Web/Worker 工厂测试、双路径 conformance |
| ACL 或租户泄漏 | enterprise retriever security、真实 PostgreSQL/Qdrant pipeline、ACL leak gate |
| 候选排序变化 | RRF、rerank、context packing、benchmark |
| Fast query 事务退化 | query service、撤权复核、同步 fencing、API 测试 |
| Durable tool 证据退化 | read tool adapter、evidence ledger、task agent 测试 |
| 摄取/投影原子性退化 | ingestion、projection、generation、convergence 集成测试 |
| Task/审批恢复退化 | resume command、approval、checkpoint、crash matrix |
| 写工具重复副作用 | PostgreSQL tool fencing、M2 controlled-write E2E |
| 部署配置漂移 | `docker compose config`、三 Worker 静态验收 |
| 隐私泄漏 | 响应、日志、trace、artifact、resume payload privacy scan |

## 10. 完成定义

### 第 3 项完成条件

- application→infrastructure 直接导入为 0。
- application 对 SQLAlchemy、Qdrant、psycopg 的直接导入为 0。
- application↔agent 的实现级循环消失。
- infrastructure/agent 只通过稳定端口和 DTO 依赖 application/domain。
- AST 架构测试无 allowlist，并能稳定阻止新反向依赖。
- 所有事务、fencing、ACL、证据和副作用安全回归通过。

### 第 4 项完成条件

- Fast Path 与 Durable Path 的入口、时序、适用场景和依赖在 README/architecture 中明确。
- `EnterpriseRetriever` 是唯一 Retrieval Core；没有第二套检索编排。
- Web 与 task Worker 通过同一生产工厂获得相同检索配置。
- 两条路径共享授权、搜索、融合、重排、上下文预算和充分性策略。
- 两条路径保留各自正确的任务生命周期、证据绑定、回答/工具和 HTTP 语义。
- conformance tests 能检测候选顺序、配置、ACL 或 insufficiency code 漂移。

### 总体验收

- 不新增消息中间件或与目标无关的框架。
- 不覆盖实施前用户已有改动。
- 全量与集成测试通过，机器报告绑定当前代码版本。
- 本轮产生的临时 PostgreSQL 数据库、Qdrant collection、对象目录和测试文件残留为 0。

## 11. 推荐实施顺序与停点

推荐顺序：

```text
L0
  → L1 → R1 → R2
  → L2
  → L3
  → L4
  → L5
  → L6
  → R3
  → F1
```

其中有三个适合单独提交、单独验收的安全停点：

1. **停点 A：L0 + L1 + R1 + R2**
   Retrieval Core 已真正统一，架构只能继续变好；尚未触碰高风险 task 事务。
2. **停点 B：L2 + L3 + L4**
   普通查询与文档链路完成端口化，仍可在进入 Agent 状态机前复验。
3. **停点 C：L5 + L6 + R3 + F1**
   最后处理 task/approval/tool/checkpoint 高风险边界并完成零 allowlist。

任何停点出现以下情况，应停止扩展范围并先修复：

- 行为测试需要通过放宽 ACL、fencing 或审批约束才能恢复；
- 为端口化被迫复制业务规则到 application 和 infrastructure 两处；
- 一个端口开始暴露 ORM Row、Session、Qdrant client 或任意字典；
- Fast/Durable 为了“统一”开始共享不相同的任务生命周期；
- 新增抽象数量明显超过被消除的具体依赖且没有测试价值。
