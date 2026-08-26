# M1 二次修复实施计划

> 创建日期：2026-08-26
> 状态：待实施；第一次修复仅作为现状基线，不再视为 M1 已最终收口
> 依据：`AGENT_REFACTOR_PLAN.md`、第一次 `M1_FIX_PLAN.md` 的实施结果、2026-08-26 代码质量/架构/业务逻辑复审
> 范围：只修复 M1 Research Agent 的并发、状态、配置和正文生命周期缺口；不进入 M2
> 发布约束：新增 0012；发布时停止任务 Worker，不允许 0011 旧 Worker 与 0012 新 Worker 滚动混跑

---

## 1. 结论与目标

第一次修复已经完成租约前置校验、致命工具码穿透、身份重解析、停止中间件、TTL 扫描和 `agent_tasks` 0011 物理收缩，但复审确认其仍未满足 M1 最终收口标准。缺口不是单纯测试不足，而是以下业务不变量尚未在代码结构中成立：

1. 旧 generation 在失租后仍可能写 LangGraph checkpoint 或提交已启动工具的迟到结果。
2. 幂等回放没有使用当前服务端身份，撤权后仍可能返回旧正文。
3. 连续失败护栏不符合真实 `AIMessage -> ToolMessage` 交错序列。
4. API 与独立 Worker 的可用性没有共同事实源，Worker 退出后 API 仍可接受任务。
5. Agent 中间件预算与 `agent_tasks.budgets` 可因环境覆盖而分裂。
6. 工具/任务的执行状态与结果正文可用状态混为一体，GC 会改写历史事实或产生假可用。
7. PostgreSQL 与对象存储之间缺少失败补偿和孤儿重扫，敏感正文可能永久失去引用。
8. deadline 只在 `_settle()` 前检查，最终提交没有数据库锁内终检。
9. `waiting_for_user` 无人恢复时不会自动终止，checkpoint 可无限保留。
10. 部分稳定错误码和第一次计划要求的 legacy 清理尚未完成。

本轮目标不是重写 Agent，而是把所有写边界收敛到同一组可验证规则：

```text
任务租约是业务写权限
checkpoint / 工具开始 / 幂等回放 / 工具完成 / 最终结果提交都必须服从该权限

执行终态是不可变审计事实
正文可用性是独立、可过期、可清理的生命周期

PostgreSQL 是事实源
对象存储通过确定性引用、失败补偿和幂等重扫最终收敛
```

完成本文全部批次和验收后，才能重新声明 M1 收口完成。

---

## 2. 范围与决策记录

| 议题 | 本轮决策 | 原因 |
|---|---|---|
| 数据库迁移 | 新增 `20260826_0012`，不改写 0011 | 0011 已成为当前 head，后续修复必须前向演进 |
| checkpoint fencing | 包装当前固定版本 `PostgresSaver`，在同一 psycopg 事务中锁任务行并写 checkpoint | 仅在写前查询租约仍有 TOCTOU；心跳不能阻止 LangGraph 写入 |
| checkpoint thread | 继续保持 `thread_id = task_id` | 新 Worker 必须从旧 Worker 最后一个合法 checkpoint 恢复 |
| 工具 fencing | `_complete()` 接收原请求并执行最终租约/身份门；不新增 tool generation 审计列 | request 已由服务端携带 owner/generation；新增列不是正确性必需 |
| API/Worker 可用性 | 新增 PostgreSQL `agent_capability_leases` 动态能力租约 | API 与 Worker 是独立进程，环境变量或 Compose 静态依赖不能表示运行时存活 |
| 全局 readiness | 不把任务 Worker 缺失升级为整个 API 不 ready | 快速 RAG/文档接口不应因 Research Worker 暂时离线而整体下线；任务创建单独返回 503 |
| 预算事实源 | `agent_tasks.budgets` 是已创建任务的唯一预算；Agent 按 TaskLease 预算构建/缓存 | Worker 重启或环境变化不得覆盖存量任务预算 |
| 执行状态与正文 | 增加正文过期/清理字段，禁止 TTL 改写 `succeeded/refused` | “执行成功”与“正文还在”是两个业务维度 |
| 跨存储一致性 | 不引入分布式事务；采用写后绑定、失败补偿、grace + 有界孤儿扫描 | 本地对象存储不支持数据库事务，补偿协议足够且更简单 |
| GC 归属 | 从 `TaskWorker`/`ToolExecutionService` 拆出单一 `TaskRetentionService` | Worker 只推进任务；正文保留策略是独立周期职责，当前耦合已导致状态错误 |
| deadline | 最终正确性边界放到 `finish_task()` 的数据库锁内 | invoke 后的本机时钟检查只能提前失败，不能保证提交时未过期 |
| 发布方式 | stop-the-world：先停任务 Worker，再迁移/部署，再启动新 Worker | checkpoint 写协议变化，不支持旧/new saver 混跑 |
| 可选 backlog | token/provider 成本统计、`max_context_tokens` 中间件继续不做 | 不阻塞 M1 正确性；避免借修复进入新功能 |

### 2.1 明确不做

- 不实现 M2 WRITE 工具暴露、HITL 审批恢复、MCP、多 Agent 或代码执行。
- 不引入 Redis、消息队列、分布式锁服务、对象存储 SDK或新的 Python 依赖。
- 不把消息、ToolMessage、Todo、用户回答或图节点复制回 `agent_tasks`。
- 不使用 `thread_id = task_id:generation`，不创建 generation 分叉线程。
- 不依赖容器重启策略代替 API 任务接受门。
- 不用“大幅提高 recursion_limit”代替模型/工具/失败/重复调用边界。
- 不为了兼容尚未发布的旧 Runtime 保留双写、回退开关或 legacy approval 入口。
- 不修改 `AGENT_REFACTOR_PLAN.md` 中 M2 及后续阶段设计。

---

## 3. 必须成立的业务不变量

### 3.1 租约与唯一事实源

1. 只有当前 `agent_tasks.lease_owner + lease_generation` 对应的 Worker 可以写任务相关业务事实。
2. `claim_next()` generation 递增与 checkpoint 写必须通过同一任务行锁排序。
3. 若旧 Worker 先持有任务行锁，它可以完成该次合法 checkpoint 写；新 Worker 随后抢占并从该 checkpoint 恢复。
4. 若新 Worker 已提交 generation 递增，旧 generation 此后的 checkpoint、pending writes、工具终态和任务终态必须全部被拒绝。
5. 心跳用于减少租约过期概率，不作为 fencing 正确性边界；不要求强行终止正在运行的 Python 线程或模型 HTTP 请求。
6. 失租 Worker 可以继续完成进程内计算，但不得把该计算提交到 checkpoint、数据库业务状态或结果引用。

### 3.2 checkpoint

1. `thread_id` 始终等于 `task_id`。
2. `put()` 与 `put_writes()` 每次写入都必须验证 tenant、task、worker、generation、lease expiry、cancelled 和可执行状态。
3. 租约验证和 checkpoint SQL 必须位于同一个 PostgreSQL 事务。
4. saver 返回的新 config 必须保留服务端 fencing 字段，不能被原生 `PostgresSaver.put()` 返回值丢弃。
5. `get_tuple()`/`list()` 可保持只读委托；`delete_thread()` 只用于保留期清理，不走执行租约门。

### 3.3 工具执行与身份

1. 新工具调用、幂等回放和工具完成都必须经过当前租约门和当前身份门。
2. 当前身份由 `current_task_auth()` 从数据库重新构建，不能信任 invoke 开始时的 `request.auth` 快照。
3. principal、roles、groups 或 attributes 摘要变化时，不返回旧工具正文或 hash-only 成功。
4. 工具执行期间发生抢占时，旧 Worker 不提交工具终态；`running` 行由新 Worker 回放或过期扫描收敛。
5. 新 generation 可以回放旧 generation 已经合法提交的相同幂等结果；generation 不加入幂等冲突绑定。
6. READ 崩溃可收敛为 `failed`；WRITE 保持 `unknown`。本轮不新增任何 WRITE 暴露。

### 3.4 任务预算与停止

1. 新任务创建时由当前 Settings 生成完整 `TaskBudgets` 并持久化。
2. 任务创建后，以 `agent_tasks.budgets` 为唯一来源；Settings 变化不影响该任务。
3. `ModelCallLimitMiddleware`、`ToolCallLimitMiddleware` 和 recursion limit 必须使用同一个 TaskLease 预算。
4. 连续失败和重复调用只从 checkpoint 消息历史重算，不在 `agent_tasks` 建第二套计数。
5. Todo、结构化输出工具和未知 ToolMessage 不参与业务工具失败/重复统计。

### 3.5 执行终态与正文生命周期

1. `succeeded`、`refused`、`failed`、`cancelled`、`unknown` 是执行结论；正文过期不得改变该结论。
2. `result_available` 必须同时考虑终态、ref/hash、`content_expires_at` 和 `result_purged_at`。
3. 过期但尚未物理删除的正文也不可再通过 API 或幂等工具回放暴露。
4. 工具成功结果清理后仍保持 `tool_calls.status = succeeded`，保留 hash、verified/completed 审计时间。
5. 对象删除失败时不得清空数据库引用或写 purge marker。
6. 无结果对象的终态任务也必须完成 `result_purged_at` 标记，避免永久重复命中。

### 3.6 deadline

1. invoke 前后检查用于快速失败，但不构成最终提交证明。
2. `succeeded/refused` 的最终提交必须在任务行锁内使用数据库时间复查 deadline、cancelled、owner 和 generation。
3. deadline 已过且租约仍合法时，任务在同一事务内收敛为 `failed/task_deadline_expired`，不得绑定结果对象。
4. 若租约已经失效，旧 Worker不得替任务写 deadline 失败终态，只返回 stale。

### 3.7 跨存储一致性

1. 不宣称 PostgreSQL 与文件对象存储具有原子提交。
2. 对象写入后必须在最终绑定事务中再次验证租约、身份和 deadline。
3. 绑定失败立即 best-effort 删除本次对象；真实删除错误必须记录并保留可重扫线索。
4. 进程在“对象写成功、数据库绑定前”崩溃产生的孤儿，由固定前缀、mtime grace 和数据库引用核对进行有界清理。
5. 对象已删除但数据库 purge 事务失败时，下次 `missing_ok=True` 删除后继续清理引用，最终幂等收敛。

---

## 4. 当前缺陷与根因映射

| ID | 缺陷 | 当前根因 | 违反的不变量 |
|---|---|---|---|
| R2-01 | 失租后仍可能写 checkpoint | `PostgresSaver` 未包装；config 只有 thread_id；心跳异常在 invoke 返回后才抛 | 3.1、3.2 |
| R2-02 | 工具执行门不完整 | `_complete(tool_call_id)` 丢失 request/lease；幂等回放不重建当前身份；身份错误又与 stale 混为一类 | 3.1、3.3 |
| R2-03 | API 接受无执行者任务 | fail-fast 只发生在独立 Worker 进程；创建 API 不知道 Worker 存活 | 业务可用性 |
| R2-04 | 预算双源 | Agent 用 Settings，任务行用模块默认常量，recursion limit 又用任务行 | 3.4 |
| R2-05 | 连续失败保护不触发 | 反向扫描遇到中间 AIMessage 即停止 | 3.4 |
| R2-06 | deadline 后仍可能 succeeded | 最后一次检查发生在 citation/对象保存之前；`finish_task()` 不验 deadline | 3.6 |
| R2-07 | 执行终态与正文状态混淆 | 工具 TTL 改写 succeeded；TaskState 不表达 purge/expiry；API 只看 ref/hash | 3.5 |
| R2-08 | 保留策略不能可靠收敛 | 删除失败仍清 ref、无结果任务重复命中、对象无孤儿扫描、waiting 无 deadline reaper | 3.5、3.6、3.7 |
| R2-09 | 错误码仍被压平 | READ 工具兜底异常固定映射为 `tool_execution_unavailable` | 错误契约 |
| R2-10 | 0011 清理不完整 | `approve_for_task_revision()` 无调用方但仍保留 | 方案 A 边界 |

---

## 5. 详细修复方案

## 5.1 R2-01：checkpoint generation fencing

### 目标

让 LangGraph 的每一次持久写都与 `agent_tasks` 当前租约在同一 PostgreSQL 事务中判定，从根本上消除“校验通过后被抢占、随后继续写”的 TOCTOU。

### 文件

| 文件 | 改动 |
|---|---|
| `src/rag_kb/agent/checkpointer.py` | 新增组合式 `FencedPostgresSaver`；包装固定版本 `langgraph-checkpoint-postgres==3.1.2` |
| `src/rag_kb/application/task_worker.py` | config 注入 tenant/task/worker/generation fencing 字段 |
| `src/rag_kb/worker.py` | 构造并注入 fenced saver，保留统一 close/setup 生命周期 |

### config 约定

```python
{
    "configurable": {
        "thread_id": task_id,
        "checkpoint_tenant_id": tenant_id,
        "checkpoint_worker_id": worker_id,
        "checkpoint_lease_generation": lease_generation,
    },
    "recursion_limit": calculated_limit,
}
```

这些字段只能由服务端 `TaskWorker` 构造，不进入模型输入、工具参数或 HTTP 请求。

### 写入算法

`FencedPostgresSaver.put()` 与 `put_writes()` 必须执行：

1. 校验 fencing 字段齐全，且 `thread_id` 是合法 task_id。
2. 使用与内部 `PostgresSaver` 相同的单一 psycopg connection 和共享可重入锁。
3. 在显式 `connection.transaction()` 中执行任务行锁查询：

```sql
SELECT task_id
FROM agent_tasks
WHERE task_id = %(task_id)s
  AND tenant_id = %(tenant_id)s
  AND lease_owner = %(worker_id)s
  AND lease_generation = %(lease_generation)s
  AND lease_expires_at > clock_timestamp()
  AND cancelled = FALSE
  AND status IN ('planning', 'running')
FOR UPDATE;
```

4. 无匹配行时抛稳定错误 `task_checkpoint_lease_stale`。
5. 保持同一 transaction 与行锁，调用内部 saver 对应写方法。
6. `put()` 返回 config 时，把原生 saver 返回的 `checkpoint_id/checkpoint_ns` 与四个 fencing 字段合并，防止后续 superstep 丢失 fence context。
7. `put_writes()` 同样走该事务，不能只保护完整 checkpoint。

### 版本与实现门禁

- 只支持当前锁定的 `langgraph-checkpoint-postgres==3.1.2`。
- 实施前通过小型契约测试确认内部 saver 在外层 transaction 中使用同一 connection，不自行提交独立事务。
- wrapper 使用 composition，不复制第三方 checkpoint SQL。
- `setup()`、`get_tuple()`、`list()`、`delete_thread()` 委托原 saver。
- 若依赖升级改变 connection/transaction 行为，启动时 fail-fast，不允许静默退化为无 fence saver。

### 被否决方案

| 方案 | 否决原因 |
|---|---|
| 仅依赖心跳或工具致命错误 | 模型节点和 middleware 仍会写 checkpoint |
| checkpoint 写前普通查询租约 | 查询与写不原子，存在 TOCTOU |
| generation 拼入 thread_id | 破坏同线程恢复并产生分叉事实源 |
| 修改 LangGraph checkpoint 表或触发器 | 强耦合第三方 schema，升级风险高 |
| 捕获 GraphRecursionError 后删除旧 checkpoint | 事后补偿无法撤销已污染的恢复历史 |

---

## 5.2 R2-02：工具开始、幂等回放和完成共用执行门

### 目标

所有工具路径都复用同一个“任务行锁 + 当前租约 + 当前身份”事务门，任何捷径都不能绕过。

### 文件

| 文件 | 改动 |
|---|---|
| `src/rag_kb/application/tool_execution_service.py` | 合并租约/身份门；重写幂等回放；`_complete()` 接收 request 并最终复核 |
| `src/rag_kb/infrastructure/database/task_execution.py` | 继续复用 `current_task_auth()`，不新建第二套身份解析 |
| `src/rag_kb/agent/tools.py` | 保持 lease/auth 中止码穿透，调整错误分类 |

### 共享门语义

共享私有方法在已有 session/transaction 内执行：

```text
SELECT agent_tasks FOR UPDATE
-> 校验 tenant/task/user
-> 校验 owner/generation/lease expiry
-> 校验 status/cancelled/deadline
-> current_task_auth()
-> 当前身份指纹 == request.auth 指纹 == 历史 tool_call.principal_hash（若存在）
```

该门用于：

1. 首次幂等查询与正文回放。
2. `_persist_running()` 创建工具执行意图。
3. IntegrityError 竞态后的再次幂等查询。
4. adapter 返回后的 `_complete()` 最终绑定。
5. ObjectConflictError 后读取已有对象并确认 hash 的路径。

### 幂等回放算法

1. 在同一事务中锁任务行并重建当前身份。
2. 查询历史 ToolCallRow 并校验 tool/version/args/plan/evidence/principal 绑定。
3. 身份变化立即抛 `task_auth_context_changed`，不得读取对象正文。
4. 结果已过期或 `result_purged_at` 非空时返回不可重试 `tool_result_expired`，不修改历史执行状态。
5. 对象读取可在第一次门检查后进行，但向 Agent 返回前必须再次执行当前租约/身份门，避免读取期间发生撤权或抢占。
6. 新 generation 身份未变时允许回放旧 generation 已成功提交的结果，adapter 执行次数不增加。

### 工具完成算法

`_complete(request, tool_call_id, ...)` 替代只接收 tool_call_id 的接口：

1. 读取 running ToolCallRow，确认其 tenant/task 与 request 一致。
2. 执行第一次当前租约/身份门。
3. 验证 adapter 输出并写确定性对象键 `tool-results/{task_id}/{tool_call_id}.json`。
4. 开启最终绑定事务，重新锁任务行和工具行，再次执行租约/身份/deadline 门。
5. 合法时更新工具终态、hash、ref、size、expiry；不合法时不更新 ToolCallRow。
6. 最终绑定失败时 best-effort 删除刚写对象；进程崩溃遗留由孤儿扫描处理。
7. WRITE unknown 路径仍按现有规则处理，但同样必须通过完成门；本轮不增加 WRITE Agent 工具。

### 错误分类

| 类别 | 错误码 | Worker 行为 |
|---|---|---|
| 失租 | `tool_task_lease_stale`、`task_checkpoint_lease_stale`、repository lease stale/expired | 返回 `stale`，不写任务终态 |
| 身份失效 | `task_auth_context_changed`、当前 principal 不可解析 | 若仍持有租约，收敛 `failed`；若同时失租，返回 stale |
| 普通工具失败 | 权限/参数/版本/瞬时执行错误 | 按受控 ToolMessage 或普通任务失败处理 |

当前 `_FATAL_TOOL_CODES` 必须拆成“stale 不可写”和“auth 可在合法租约下失败关闭”两类，不能继续统一返回 stale。

### 不新增 ToolCall generation 列的理由

ToolExecutionRequest 已由服务端携带 worker_id/generation，`_complete()` 是内部调用且可以校验 ToolCallRow 的 task/tenant。持久化 generation 仅增加审计信息，不是 fencing 正确性所必需；本轮遵循最小改动，不新增该列。

---

## 5.3 R2-03：API 与 Worker 动态能力租约

### 目标

Research Worker 未成功构建、已退出或协议不兼容时，`POST /v1/tasks` 不创建 planning 任务。

### 数据结构

0012 新增 `agent_capability_leases`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `capability_name` | `VARCHAR(64)` | 固定 `research-agent` |
| `protocol_version` | `VARCHAR(64)` | 本轮固定 `m1-fenced-v2` |
| `worker_id` | `VARCHAR(200)` | Worker 实例 ID |
| `lease_expires_at` | timezone datetime | 由数据库时间判断 |
| `updated_at` | timezone datetime | 最近续约时间 |

主键：`(capability_name, protocol_version, worker_id)`。
索引：`(capability_name, protocol_version, lease_expires_at)`。

### 文件

| 文件 | 改动 |
|---|---|
| `src/rag_kb/infrastructure/database/models.py` | 新增 capability row |
| `src/rag_kb/infrastructure/database/task_repository.py` | 注册/续约/注销/查询能力；创建任务事务内检查能力 |
| `src/rag_kb/worker.py` | Agent 完整构建成功后登记，独立后台心跳续约，退出时 best-effort 注销 |
| `src/rag_kb/web/dependencies.py` | 生产 TaskRepository 注入默认预算和 `m1-fenced-v2` |
| `src/rag_kb/web/routers/tasks.py` | 无有效能力租约时返回 503，不创建任务 |
| `src/rag_kb/config.py` | 增加 `agent_capability_lease_seconds`，心跳间隔固定为其不超过三分之一 |

### 行为

1. Worker 完整构建模型、四个 READ 工具、fenced saver 和默认预算 Agent 后才登记能力。
2. 心跳使用独立数据库会话，不依赖任务 invoke 是否正在执行。
3. API 创建 Research 任务的同一事务中查询未过期且版本一致的 capability lease；不存在则抛 `research_agent_unavailable`。
4. API 映射为 HTTP 503，事务不插入 `agent_tasks`。
5. Worker 正常退出 best-effort 删除自身行；异常退出依靠 TTL。
6. capability 只证明“接受请求时存在近期活跃的兼容 Worker”；请求接受后的 Worker 崩溃仍由持久任务和租约恢复处理。
7. capability 缺失不影响 `/ready` 的 Qdrant、PostgreSQL、schema 和 projection 门，也不阻断同步知识查询/文档接口。
8. API `/continue` 按请求临时构建的 resume Worker 不发布 capability；它只执行已存在任务，配置失败继续返回 503。

### 被否决方案

| 方案 | 否决原因 |
|---|---|
| API 只检查 `DEEPSEEK_API_KEY` | 不能证明独立 Worker 存活或协议兼容 |
| Compose `depends_on` 或 restart policy | 只能描述启动顺序，不能发现运行中死亡 |
| API 每次构建完整 Agent | 重复加载连接/模型，仍不能证明后台消费者存活 |
| 全局 readiness 因 Worker 离线失败 | 会无故下线快速 RAG 和文档能力 |
| 引入 Redis/队列心跳 | PostgreSQL 已是任务事实源，无需新依赖 |

---

## 5.4 R2-04：预算改为任务行单一来源

### 目标数据流

```text
Settings（只决定新任务默认值）
-> TaskBudgets
-> agent_tasks.budgets
-> TaskLease.budgets
-> Agent middleware + recursion limit
```

### 文件与改动

| 文件 | 改动 |
|---|---|
| `src/rag_kb/domain/tasks.py` | 移除对 `config.py` 的反向依赖；领域默认常量留在 domain；model/tool 最小值与 Settings 一致为 1 |
| `src/rag_kb/config.py` | Settings 默认引用领域默认值；保留环境覆盖；增加 `agent_max_runtime_seconds` |
| `src/rag_kb/infrastructure/database/task_repository.py` | 创建 Research 任务必须使用构造时注入的 Settings-derived TaskBudgets，不再在方法内静默 `TaskBudgets()` |
| `src/rag_kb/agent/factory.py` | max_model_calls/max_tool_calls 改为必传，无第二默认源 |
| `src/rag_kb/application/task_worker.py` | 注入 `agent_for(TaskBudgets)`，领取后按 lease.budgets 选择 Agent |
| `src/rag_kb/worker.py` | 创建按预算元组缓存的 Agent 工厂；默认 Agent 构建成功后才发布 capability |
| `.env.example` | `AGENT_MAX_STEPS` 改为 `AGENT_MAX_MODEL_CALLS`；示例值与文档一致 |

### 约束

- Agent 工厂缓存 key 至少包含 `max_model_calls/max_tool_calls`。
- recursion limit 使用同一 TaskBudgets，继续给 middleware superstep 留余量。
- 已创建任务在 Worker 重启后仍使用持久预算，不受新环境值覆盖。
- `max_context_tokens` 与 token 成本统计仍不执行，保留为 backlog，不借本轮补做。
- `tool_timeout_seconds` 继续由 ToolDefinition 决定；不把未生效字段伪装为已执行预算。

---

## 5.5 R2-05：修正 ExecutionGuard 的真实轨迹语义

### 文件

`src/rag_kb/agent/middleware.py`

### 业务工具集合

只统计以下四个工具：

- `search_knowledge`
- `fetch_evidence`
- `list_document_versions`
- `compare_document_versions`

排除：

- `write_todos`
- `TaskResultCandidate` 结构化输出工具
- 未知工具
- 无法解析为受控 JSON 的 ToolMessage

### 连续失败算法

1. 先从完整消息历史提取业务 ToolMessage 结果序列，忽略中间 AIMessage。
2. 从最后一个业务结果向前统计 `status == failed`。
3. 遇到最近一个 `succeeded/unknown` 或非失败业务结果即停止。
4. 连续 3 个 failed 结果抛 `agent_consecutive_tool_failures`。
5. checkpoint 恢复后重新扫描相同历史，结果必须一致。

### 重复调用算法

1. 只扫描 AIMessage 中四个业务工具的 tool_calls。
2. 对 `{tool_name, canonical args}` 计算稳定指纹，不包含 tool_call_id。
3. 第二次相同调用已经是允许的一次受控重试；完成两次后，在下一次模型调用前抛 `agent_repeated_tool_call`，阻止第三次。
4. 不把 Todo 或结构化响应计入重复。
5. 不在数据库保存计数。

### 说明

本轮保持 `before_model` 纯函数扫描方案，不引入新的 graph state reducer。若未来需要在第二次调用执行前拦截，应另行设计 `after_model` 提议门；M1 当前四工具均为 READ，允许一次重试符合现有 retryable 语义。

---

## 5.6 R2-06：deadline 数据库锁内终检

### 文件

| 文件 | 改动 |
|---|---|
| `src/rag_kb/infrastructure/database/task_repository.py` | `finish_task()` 增加数据库时间终检和实际终态返回 |
| `src/rag_kb/application/task_worker.py` | 处理 deadline 实际失败结果并补偿未绑定对象 |

### `finish_task()` 算法

1. `SELECT ... FOR UPDATE` 锁任务行。
2. 使用 PostgreSQL `clock_timestamp()` 获取最终判定时间。
3. 校验 owner/generation/lease expiry。
4. 对期望 `succeeded/refused` 的提交复核 cancelled、status 和 deadline。
5. deadline 已过且租约合法时，不抛出导致回滚的异常，而是在同一事务写：
   - `status = failed`
   - `last_error_code = task_deadline_expired`
   - `result_ref/result_kind = NULL`
   - 不绑定新 result_hash
   - 清租约、中断摘要
   - 设置正文到期时间
   - 写 task_failed event
6. 返回数据库实际终态。
7. 若租约已经失效，抛 lease stale，旧 Worker不得写 deadline 失败状态。

### Worker 补偿

1. `_settle()` 保存结果对象后调用原子 `finish_task()`。
2. 若返回实际 `failed/task_deadline_expired`，立即 best-effort 删除未绑定对象并返回 failed。
3. 若 finish 因失租失败，删除未绑定对象并返回 stale。
4. 若进程在补偿前崩溃，交由孤儿扫描。
5. invoke 后现有本机 deadline 检查保留为快速失败，但不再被描述为最终正确性边界。

---

## 5.7 R2-07：执行终态与正文可用性分离

### 0012 新字段

`agent_tasks` 新增：

| 字段 | 类型 | 说明 |
|---|---|---|
| `content_expires_at` | timezone datetime nullable | Research 终态 checkpoint/任务结果正文统一到期时间 |

继续使用 0011 已有：

- `checkpoint_purged_at`
- `result_purged_at`

`tool_calls` 新增：

| 字段 | 类型 | 说明 |
|---|---|---|
| `result_purged_at` | timezone datetime nullable | 工具正文物理清理完成标记 |

### 约束调整

#### `agent_tasks`

重写 research result 约束：

1. `succeeded/refused` 且 `result_purged_at IS NULL`：要求 `result_kind='research'`、ref/hash 完整。
2. `succeeded/refused` 且 `result_purged_at IS NOT NULL`：允许 `result_ref IS NULL`，但要求 result_kind/hash 保留。
3. Research 终态必须有 `content_expires_at`。
4. 非结果终态允许没有 result_kind/ref/hash，但 GC 后仍写 `result_purged_at`。

#### `tool_calls`

重写 succeeded result reference 约束：

1. active body：`result_purged_at IS NULL` 且 ref/size/expires 完整。
2. purged body：`result_purged_at IS NOT NULL`、ref/size 为空、hash 与 verified/completed 保留。
3. 两种情况都保持 `status='succeeded'`、`error_code IS NULL`。

### 所有 Research 终态入口

`finish_task()`、用户取消、waiting deadline reaper 以及任何将 Research 任务转为终态的恢复路径，都必须在同一事务设置 `content_expires_at = db_now + task_result_retention`。不能只在 succeeded/refused 路径设置，否则 cancelled/failed checkpoint 仍会无限保留。

### TaskState 与 API

| 文件 | 改动 |
|---|---|
| `src/rag_kb/domain/tasks.py` | 增加 `content_expires_at`、`result_purged_at`，提供统一可用性判断 |
| `src/rag_kb/infrastructure/database/task_repository.py` | row/state 映射上述字段 |
| `src/rag_kb/web/routers/tasks.py` | 不再自行用 `bool(ref and hash)` 推导 |
| `src/rag_kb/application/task_result_service.py` | 读取对象前先检查 expiry/purge，区分 expired 与 unavailable |

`result_available` 规则：

```text
status in {succeeded, refused}
AND result_ref/result_hash 完整
AND result_purged_at IS NULL
AND content_expires_at > now
```

正文已到期但 GC 尚未运行时也必须返回不可用。`GET /result` 对明确到期映射为 HTTP 410；绑定缺失或状态不允许仍为 409；完整性损坏按服务端错误处理。

---

## 5.8 R2-08：独立保留服务、可靠 GC 与孤儿重扫

### 架构调整

新增 `src/rag_kb/application/task_retention_service.py`，集中负责：

1. `expire_waiting_tasks(limit)`
2. `purge_expired_tool_results(limit)`
3. `purge_finished_task_content(limit)`
4. `purge_orphaned_result_objects(limit)`

从 `TaskWorker` 删除直接 import Settings/ORM/UoW 的清理方法；从 `ToolExecutionService` 删除 TTL 扫描职责。`worker.py` 在现有周期恢复循环中调用 retention service。该拆分不是为未来扩展，而是避免任务推进服务继续直接管理跨存储生命周期。

### 工具结果 GC

1. 查询 `status='succeeded'`、ref 非空、`result_expires_at <= db_now`、`result_purged_at IS NULL`。
2. 先调用 `delete_exact(ref)`。
3. 删除成功或对象本来不存在后，在数据库事务中再次锁行并确认 ref 未变化。
4. 清空 ref/size，写 `result_purged_at`，保留 status/hash/expires/verified/completed。
5. 真实 OSError 时保留 ref 和 marker，记录错误，下一轮重试。
6. 幂等回放依据 expiry/purge 返回 `tool_result_expired`，不修改历史行。

### 任务 checkpoint/result GC

1. 只扫描 Research 终态且 `content_expires_at <= db_now` 的任务。
2. checkpoint 未清理时调用 `delete_thread(task_id)`；成功后写 `checkpoint_purged_at`。
3. 有 result_ref 时先删除对象，成功后清空 ref、写 `result_purged_at`、保留 kind/hash/status。
4. 无 result_ref 的 failed/cancelled/unknown 也写 `result_purged_at`，表示该分支已完成。
5. 单个 checkpoint 或对象失败不得伪造 marker；保留到下轮重扫。
6. 两个 marker 都存在后不再命中。

### waiting deadline reaper

`expire_waiting_tasks(limit)`：

1. 锁定 `waiting_for_user` 且 deadline 已过的 Research 任务。
2. 原子转为 `failed/task_deadline_expired`。
3. 清中断摘要和租约字段，递增 state_version，写 event。
4. 设置 `content_expires_at = db_now + task_result_retention`。
5. `claim_user_resume()` 同样在锁内拒绝过期任务，不能恢复已过 deadline 的 checkpoint。
6. 终态后由普通 checkpoint/result GC 处理正文。

### 对象孤儿扫描

修改 `src/rag_kb/infrastructure/object_store/local.py`，增加只允许固定命名空间的有界枚举：

```python
iter_refs(prefix, *, older_than, limit)
```

安全约束：

- prefix 只允许 `tool-results/` 和 `task-results/`。
- 不接受客户端或模型提供的路径。
- 返回规范化相对 key。
- 不跟随 symlink，不越过 object root。
- 按 mtime、key 稳定排序并严格应用 limit。

扫描算法：

1. 只处理早于 `result_object_orphan_grace_seconds` 的对象。
2. 批量查询数据库中对应 ref 是否仍被引用。
3. 有引用则保留。
4. 工具对象对应 running 行且 execution deadline 未到时保留。
5. 无引用且超过 grace 时删除。
6. 单条删除失败只记录并继续，不阻塞同批其他对象。

新增配置：

- `result_object_orphan_grace_seconds`

启动关系校验：

- orphan grace 必须大于最大工具 timeout 与合理的最终绑定窗口。
- `tool_result_ttl_seconds` 必须大于 Agent 正常单次工具执行窗口。

### 跨存储补偿统一规则

```text
对象写成功 + DB 绑定成功       -> 正常引用
对象写失败                     -> DB 不绑定成功状态
对象写成功 + DB 绑定失败       -> 立即 best-effort 删除
补偿前进程崩溃                 -> orphan grace 后扫描删除
对象删除成功 + purge 提交失败  -> 下轮 missing_ok 删除后重新提交
对象删除真实失败               -> 保留 ref/marker，等待重试
```

---

## 5.9 R2-09/R2-10：错误码透传与 legacy 收尾

### READ 错误码

`src/rag_kb/agent/tools.py` 的兜底分支改为：

```python
except Exception as exc:
    code = getattr(exc, "code", None)
    return render_tool_message(
        tool_name,
        None,
        error_code=code if isinstance(code, str) and code else "tool_execution_unavailable",
    )
```

失租和身份中止码仍直接 raise，不渲染普通 ToolMessage。

### 异常边界

- `ToolExecutionService` 不再用 `BaseException` 吞掉 `KeyboardInterrupt/SystemExit/GeneratorExit`，只捕获 `Exception`。
- 对象存储、数据库和 adapter 错误保留稳定 code；日志记录 task/tool_call 标识但不记录正文和凭据。
- retryable 继续使用显式白名单，权限/参数/版本/身份/过期类保持不可重试。

### legacy 清理

- 删除无调用方 `ApprovalService.approve_for_task_revision()`。
- 不修改 M2 approval 核心绑定、WRITE 适配器或 Artifact 流程。
- 全仓 grep 再次确认 0011 删除字段与 legacy 方法零生产引用。

---

## 6. 数据库迁移 0012

新增：

`migrations/versions/20260826_0012_m1_fencing_and_content_lifecycle.py`

```text
revision = 20260826_0012
down_revision = 20260826_0011
```

### 6.1 upgrade 顺序

1. 创建 `agent_capability_leases` 与过期索引。
2. `agent_tasks` 增加 nullable `content_expires_at`。
3. `tool_calls` 增加 nullable `result_purged_at`。
4. 回填 Research 终态任务：`content_expires_at = updated_at + 7 days`。7 天仅用于迁移历史默认值，迁移后新终态使用运行时配置持久化。
5. 对 0011 已清理任务：`result_purged_at IS NOT NULL` 时清空已失效 result_ref，保留 result_kind/hash/status。
6. 修复第一次 GC 已改写的工具审计行：满足 `error_code='tool_result_expired'`、ref 为空、hash/verified_at 非空的行恢复 `status='succeeded'`、清 error_code、写 result_purged_at。
7. 删除并重建 `agent_tasks` research result/content expiry 约束。
8. 删除并重建 `tool_calls` succeeded result/reference 约束。
9. 增加 GC 查询索引：`agent_tasks(task_mode, status, content_expires_at)`、`tool_calls(result_expires_at, result_purged_at)`。
10. 更新 ORM 与 `EXPECTED_DATABASE_REVISION`。

### 6.2 存量非终态任务

- 发布前停止所有旧任务 Worker。
- 不因迁移批量失败 planning/running/waiting 任务；新 Worker仍从同一 task_id checkpoint 恢复。
- 新 fenced saver 必须在恢复旧 checkpoint 后保留本次 invocation 注入的 fencing config。
- 迁移时已过 deadline 的 waiting 任务由新 reaper 首轮收敛。
- 旧 running ToolCallRow 继续由 execution deadline sweeper 按 READ/WRITE 语义收敛。

### 6.3 downgrade

- 删除 capability 表和新增索引。
- 恢复 0011 的旧 result reference 约束前，只能对已清 ref 的 purged succeeded 行做可逆物理兼容处理；正文不可恢复。
- 删除 `tool_calls.result_purged_at`、`agent_tasks.content_expires_at`。
- downgrade 只恢复物理结构，不恢复已删除正文、已过期 capability 或已修正的历史状态。
- 已启动 0012 Worker 或创建 0012 状态后，不允许直接恢复旧 Worker；正确回滚优先关闭任务入口并前向修复。

---

## 7. 精确文件改动清单

### 7.1 生产代码

| 文件 | 必需改动 |
|---|---|
| `src/rag_kb/agent/checkpointer.py` | `FencedPostgresSaver`、同事务行锁、fence config 保留、稳定 stale code |
| `src/rag_kb/agent/factory.py` | model/tool limits 改为必传 |
| `src/rag_kb/agent/middleware.py` | 真实 ToolMessage 序列连续失败；只统计四业务工具 |
| `src/rag_kb/agent/tools.py` | stale/auth 分类；兜底 `exc.code` 透传 |
| `src/rag_kb/application/tool_execution_service.py` | 共享租约/身份门、幂等回放、完成 fencing、移出 GC、Exception 边界 |
| `src/rag_kb/application/task_worker.py` | fence config、按预算取 Agent、auth/stale 分类、deadline 返回与对象补偿；移除 GC 直连 ORM |
| `src/rag_kb/application/task_retention_service.py` | 新增 waiting reaper、tool/task/checkpoint GC、孤儿扫描 |
| `src/rag_kb/application/task_result_service.py` | expiry/purge 前置检查，expired/unavailable/integrity 分类 |
| `src/rag_kb/application/approval_service.py` | 删除无调用 legacy 适配入口 |
| `src/rag_kb/config.py` | capability TTL、runtime 默认、orphan grace；默认值方向修正和关系校验 |
| `src/rag_kb/domain/tasks.py` | 预算默认归属 domain；TaskState 增加 expiry/purge 表达与可用性判断 |
| `src/rag_kb/infrastructure/database/models.py` | capability 表、content expiry、tool purge marker、约束和索引 |
| `src/rag_kb/infrastructure/database/task_repository.py` | capability lease、创建门、预算注入、waiting reaper 支持、deadline 原子终检、state 映射 |
| `src/rag_kb/infrastructure/object_store/local.py` | 两个固定前缀的安全有界枚举 |
| `src/rag_kb/web/dependencies.py` | 默认 TaskBudgets/capability 注入，revision 更新 |
| `src/rag_kb/web/routers/tasks.py` | create 503；统一 result_available；expired 410 |
| `src/rag_kb/worker.py` | capability 心跳、按预算 Agent 缓存、RetentionService 调度 |
| `.env.example` | 修正 Agent 预算变量名和值 |
| `docker-compose.yml` | API/Worker 显式共享预算、capability TTL、orphan grace 配置 |
| `migrations/versions/20260826_0012_m1_fencing_and_content_lifecycle.py` | 0012 完整 upgrade/downgrade |

### 7.2 明确无需改动

- `src/rag_kb/agent/context.py`：worker_id/generation 已完整表达。
- `src/rag_kb/infrastructure/database/task_execution.py`：复用现有 `current_task_auth()`。
- 文档上传、上传恢复和文档删除链：已有 reservation/reconciler，不属于本缺陷。
- Artifact WRITE 路径：M1 不暴露 WRITE，不借本轮进入 M2。
- LangGraph checkpoint 表结构：不修改第三方 schema。
- `AGENT_REFACTOR_PLAN.md`：本轮只落实其 M1 不变量，不改后续路线。

### 7.3 测试改动计划

| 文件 | 新增/修改覆盖 |
|---|---|
| `tests/test_postgres_checkpoint_fencing.py`（新增） | 双连接 generation 竞争、put/put_writes、合法先写/抢占后拒绝、config 保留 |
| `tests/test_postgres_agent_checkpointer.py` | saver 基础恢复、serializer、delete_thread 委托保持通过 |
| `tests/test_postgres_tool_fencing.py` | adapter 执行中抢占、迟到完成不提交、身份撤销后相同 key 回放拒绝 |
| `tests/test_tool_lease_fencing.py` | stale/auth 分类、IntegrityError 回放重新过门、对象补偿 |
| `tests/test_execution_guards.py` | 真实 AI/Tool 交错、成功重置连续失败、Todo/结构化工具排除、允许一次重试 |
| `tests/test_task_worker_recovery.py` | fence config、按 TaskBudgets Agent、finish 返回实际 deadline 终态、对象补偿 |
| `tests/test_postgres_retention.py` | 状态不变、无结果终态 marker、删除失败保留 ref、waiting reaper、孤儿 grace |
| `tests/test_task_repository.py` | capability 创建门、预算持久化、deadline DB 终检、expired resume |
| `tests/test_v1_tasks_api.py` | 无 capability 503 且不落行、result_available、expired 410 |
| `tests/test_worker_runtime.py` | capability 注册/续约/注销、RetentionService 周期调用、缺配置 fail-fast |
| `tests/test_database_migrations.py` | 0012 SQLite round-trip、约束、历史数据回填 |
| PostgreSQL migration 验证 | 0012 upgrade/downgrade/upgrade；精确列、约束和索引检查 |
| `tests/test_deployment_acceptance_static.py` | `.env.example`/Compose/API/Worker 预算与 capability 配置一致 |

新增 PostgreSQL 测试不得 `drop_all/create_all` 破坏迁移态；每个测试使用唯一 tenant/task/worker 并只清理自身数据。数据库 URL 必须指向显式临时测试库。

---

## 8. 实施批次与门禁

## 批次 0：0012 与协议骨架

### 实施

1. 新增 0012、ORM 字段/约束/索引/capability 表。
2. 更新 readiness revision。
3. 增加 capability repository 基础方法和 TaskState 新字段映射。
4. 完成历史数据回填与旧 GC 状态修复。
5. 明确部署 stop-the-world，不兼容旧任务 Worker 混跑。

### 门禁

- SQLite 与 PostgreSQL 0012 upgrade/downgrade/upgrade 成功。
- `agent_tasks`、`tool_calls` 精确约束支持“执行成功但正文已清理”。
- 迁移后不再需要通过改写 succeeded 状态清空 ref。
- ORM、Alembic head、readiness revision 一致。

## 批次 1：并发事实源闭环

### 实施

1. 实现 fenced saver 的 put/put_writes 同事务行锁。
2. TaskWorker config 注入并保留 fencing 字段。
3. 工具执行共享租约/身份门。
4. 工具完成最终 fencing 与对象失败补偿。
5. 幂等回放重解析当前身份。
6. stale 与 auth failure 分类拆分。
7. capability lease 登记、续约、过期和 API 创建门。

### 门禁

- 抢占后旧 generation 的 checkpoint 与 pending writes 均为零新增。
- adapter 执行中被抢占后，旧 Worker 不提交工具终态/ref/ToolMessage。
- 新 generation 可回放抢占前合法完成的相同工具结果。
- 撤权后相同 idempotency key 不返回正文。
- 无兼容 capability 时 API 返回 503 且任务表无新增。

## 批次 2：停止边界与提交契约

### 实施

1. 修正 ExecutionGuard 真实轨迹算法。
2. 预算改为 task row 单源并按预算构建 Agent。
3. deadline 放入 `finish_task()` 数据库锁内终检。
4. 修正错误码透传和 stale/auth 任务终态。
5. 修正 `.env.example` 与 Compose 配置传播。

### 门禁

- 真实 AI/Tool 交错的三次连续失败触发；Todo/结构化输出不误触发。
- 环境覆盖后的新任务预算、Agent middleware、recursion limit 精确一致。
- 已生成候选但提交前 deadline 过期时最终只能是 failed，无结果引用。
- 身份失效且租约有效时任务失败关闭；失租时不写终态。

## 批次 3：正文生命周期闭环

### 实施

1. 新增 TaskRetentionService 并迁移所有周期清理职责。
2. 工具/任务执行状态与正文状态分离。
3. 修正 result_available 和 expired 读取契约。
4. 增加 waiting deadline reaper。
5. 删除失败保留 ref/marker，支持下一轮重试。
6. 增加两个固定前缀的 orphan grace 扫描。
7. 删除 legacy approval 入口并执行零引用 grep。

### 门禁

- TTL 后 succeeded/refused/tool succeeded 状态均不改变。
- 到期即不可读，物理删除前后 `result_available` 都为 false。
- 真实删除失败不会丢引用；下一轮可重试。
- 无结果终态只清理一次。
- waiting 过 deadline 自动终止并进入正常 retention。
- “对象写成功、DB 绑定前崩溃”的对象在 grace 后删除，有引用对象不误删。

## 批次 4：最终验收与文档同步

1. 运行全部定向、全量和隔离 PostgreSQL 测试。
2. 严格 msgpack 真实 Research E2E。
3. 更新 `docs/acceptance/m1-research-e2e.json` 与 `docs/acceptance/latest.json`。
4. 更新 README、架构、迁移、使用指南和 `TASK_PROGRESS.md`。
5. 清理全部临时数据库、Qdrant collection、对象目录和调试脚本。
6. 停止，不进入 M2，等待用户明确指令。

---

## 9. 详细验收矩阵

### 9.1 checkpoint fencing

1. Worker A 取得 generation N 并在 checkpoint 写前阻塞。
2. 租约过期后 Worker B 取得 N+1。
3. 释放 A，A 的 `put()` 与 `put_writes()` 均抛 `task_checkpoint_lease_stale`。
4. 最新 checkpoint 不包含 A 抢占后的 AIMessage、ToolMessage 或 pending write。
5. B 从抢占前最后一个合法 checkpoint 恢复。
6. 反向竞争中若 A 先持有任务行锁，B 必须等待 A 合法提交后再递增 generation。
7. saver 返回 config 在多个 superstep 后仍携带 tenant/worker/generation。

### 9.2 工具完成 fencing

1. A 创建 running READ 调用并阻塞 adapter。
2. B 抢占后释放 adapter。
3. A 不得把工具行改成 succeeded/failed，不得绑定 ref，不得写 ToolMessage checkpoint。
4. A 返回 stale，不调用 `_fail_safely`。
5. running 行由 sweeper 收敛，或由 B 在安全时机恢复/重试。
6. 若工具在抢占前已合法完成但 ToolMessage 未写，B 使用相同 idempotency key 回放，adapter 总执行次数仍为 1。

### 9.3 身份与幂等

1. 首次工具执行成功并保存正文。
2. 撤销 principal role、group 或修改受控 attributes。
3. 使用相同 idempotency key 回放，返回 `task_auth_context_changed`，不读取/返回正文。
4. 身份未变化的新 generation 可以回放。
5. `_complete()` 前发生身份变化时，不提交工具终态和结果引用。
6. `_settle()` 继续使用最新身份复核 citation。

### 9.4 capability

1. 无 capability lease 时创建任务返回 503，任务行数不增加。
2. Worker 完整构建并登记 `m1-fenced-v2` 后返回 202。
3. 只有其他 protocol version 时仍返回 503。
4. Worker 异常退出并超过 TTL 后停止接受新 Research 任务。
5. 已接受任务不丢失，可由后续同版本 Worker 恢复。
6. capability 缺失不影响同步知识查询、文档接口和全局 readiness。

### 9.5 ExecutionGuard

1. `AI -> failed Tool -> AI -> failed Tool -> AI -> failed Tool` 触发连续失败。
2. 最近一次业务 ToolMessage succeeded 后，之前失败不再计为连续。
3. 同参业务工具允许初次调用和一次受控重试，第三次前停止。
4. `write_todos`、结构化结果和未知工具不参与计数。
5. checkpoint 保存/恢复后重算结果一致。

### 9.6 预算

1. 配置 model=2、tool=3、runtime=120 时，新任务 JSON 精确保存 2/3/120。
2. Worker 为该任务构建的两个 limit middleware 精确使用 2/3。
3. recursion limit 使用同一预算并足够覆盖 middleware superstep。
4. 修改 Worker 默认配置不改变已持久化任务预算。
5. 生产代码不存在 `AGENT_MAX_*` 第二默认源或 factory 隐式默认值。

### 9.7 deadline

1. Agent 已生成合法候选并写对象，但最终任务行锁取得时 deadline 已过。
2. 任务最终为 `failed/task_deadline_expired`。
3. 没有 result_ref 绑定，`result_available=false`。
4. 刚写对象被立即补偿或随后由 orphan scanner 删除。
5. 若同时已失租，旧 Worker 不写 failed，只返回 stale。
6. 不存在先 succeeded 后被后台改 failed 的可见中间状态。

### 9.8 TTL/GC

1. succeeded/refused 任务正文过期后任务状态不变。
2. succeeded 工具正文过期后工具状态不变，hash/verified/completed 保留。
3. 到期后、物理删除前，任务结果和工具幂等正文均不可读取。
4. 任务结果明确过期返回 410，未完成/无结果返回 409。
5. 对象删除真实失败时 ref 与 purge marker 保持原值。
6. 对象不存在视为幂等删除成功。
7. failed/cancelled/unknown 无结果任务写 result_purged_at，第二次扫描不命中。
8. checkpoint/result/tool 三类清理均可重复执行且不改写执行终态。

### 9.9 waiting 与孤儿对象

1. waiting_for_user 超过 deadline 后自动转 failed，清 interrupt 摘要并写 event。
2. deadline 后的 continue 不得恢复 graph。
3. waiting 终止后 checkpoint 按 content expiry 清理。
4. 注入“对象写成功、DB 绑定前崩溃”，grace 内不删除，grace 后删除。
5. 有数据库引用的对象、未过 deadline 的 running 工具对象不误删。
6. 单个坏对象不阻塞同批其他对象。

### 9.10 物理结构与文档

1. 0012 在 SQLite/PostgreSQL upgrade/downgrade/upgrade。
2. ORM 字段、数据库列、约束和索引精确一致。
3. 0011 删除字段生产引用继续为 0。
4. legacy `approve_for_task_revision` 引用与定义均为 0。
5. README/架构/迁移/使用指南/TASK_PROGRESS 与 0012、capability、预算、TTL 行为一致。

---

## 10. 最终 M1 收口标准

必须同时满足：

1. 本文 R2-01 至 R2-10 全部完成，无“仅测试证明但代码仍有反例”的项目。
2. checkpoint、工具完成、幂等回放和最终任务提交四个写边界全部受租约 fencing。
3. 失租 Worker 抢占后的 checkpoint/pending writes/工具终态/任务终态新增数均为 0。
4. 身份变化不能通过新调用或幂等回放读取旧正文。
5. 无兼容 Worker capability 时 API 不创建 Research 任务。
6. 任务预算从创建到 Agent middleware/recursion limit 只有 `agent_tasks.budgets` 一个任务级事实源。
7. deadline 在数据库锁内终检，过期任务不可能提交 succeeded/refused。
8. 执行终态与正文状态分离；GC 不改写历史成功/拒绝结论。
9. checkpoint、工具结果、任务结果和 waiting checkpoint 都有到期与幂等清理路径。
10. 对象写后绑定失败有即时补偿和孤儿重扫，删除失败不丢引用。
11. 0012 双数据库 round-trip、全量测试、全部隔离 PostgreSQL 条件测试通过。
12. 严格真实 E2E（`LANGGRAPH_STRICT_MSGPACK=true`）通过：V1/V2 上传、四 READ 工具精确轨迹、任务 succeeded、citations 服务端复核。
13. acceptance artifacts 和全部相关文档更新为当前代码行为。
14. 临时资源全部清理。
15. 完成后停止，不进入 M2。

---

## 11. 风险、缓解与回滚边界

| 风险 | 缓解 |
|---|---|
| saver wrapper 与第三方 3.1.2 事务行为不一致 | 固定版本；为 put/put_writes 做真实 PostgreSQL 契约测试；启动时 fail-fast，禁止退化 |
| checkpoint 写持有任务行锁增加竞争 | 锁只覆盖短 SQL，不覆盖模型和工具执行；M1 单任务单执行者 |
| API capability 短暂误报 | 使用数据库时钟和 TTL；完整构建后才登记；心跳独立于任务 invoke |
| capability 离线导致任务入口 503 | 只影响 Research 创建，不影响快速 RAG/文档接口；恢复 Worker 后自动恢复 |
| 按预算编译多个 Agent | 进程内按小元组缓存；通常只有默认预算一项 |
| 对象写后最终绑定失败 | 立即删除 + grace 扫描双保险；不引入分布式事务 |
| orphan 扫描误删在途对象 | 固定前缀、mtime grace、running/deadline 检查、有界批次 |
| 0012 约束重写影响历史行 | upgrade 前先回填/修正历史状态，再创建新约束；临时库双向验证 |
| 旧/new Worker 混跑 | 发布前停止任务 Worker；0012 上线后只启动 `m1-fenced-v2` |
| downgrade 无法恢复正文 | 文档明确 downgrade 仅恢复物理结构；正文删除不可逆，优先前向修复 |

### 11.1 发布步骤

1. 禁止新的 Research 任务创建或停止 API 任务入口。
2. 停止全部任务 Worker，确认 capability lease 过期/删除。
3. 备份数据库与对象目录（若环境包含需保留数据）。
4. 执行 0012 upgrade。
5. 部署 API 与 Worker 新代码。
6. 启动 Worker，确认 capability `m1-fenced-v2` 有效。
7. 恢复任务入口。
8. 执行并发故障注入和严格 E2E。

### 11.2 回滚边界

1. 发现 fencing/capability/迁移异常时，先关闭 Research 创建并停止任务 Worker；快速 RAG 继续服务。
2. 不自动切回未 fenced 的旧 Worker。
3. 不删除 LangGraph checkpoint 表、不清空 PostgreSQL、不删除对象根目录。
4. 保留 checkpoint、工具审计、结果 hash 和 orphan 对象用于诊断。
5. 0012 downgrade 不能恢复已删除正文或已终止状态。
6. 已运行 0012 Worker 后优先前向修复；只有确认数据兼容时才执行 downgrade。

---

## 12. 实施纪律

- 顺序固定：批次 0 -> 批次 1 -> 批次 2 -> 批次 3 -> 批次 4。
- 每批只修改表中列出的文件，不顺手重构快速 RAG、摄取、检索或 M2 代码。
- 关键并发路径先写可阻塞的故障注入场景，再实现事务边界。
- PostgreSQL 条件测试必须使用迁移后的独立临时数据库，不得 `drop_all/create_all` 伪造 schema。
- 任何对象清理测试使用独立对象根目录，结束后删除。
- 连续两次同一修复失败时停止，记录根因并重新评审方案，不叠加补丁。
- 不提交、不推送、不修改全局配置，不覆盖用户已有未提交改动。
- 所有敏感连接信息只从环境或容器读取，不写入文档、日志和 artifact。
- 最终只在所有门禁成立后更新 `TASK_PROGRESS.md` 为“已完成”。
