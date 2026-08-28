# 架构说明

## 事实源与边界

PostgreSQL 保存租户、主体、文档版本、ACL、Research 与 controlled_action 任务、审批、产物、工具调用和证据账本；Qdrant 只保存可重建的向量 projection。任何检索候选必须同时通过 SQL 授权集和 Qdrant payload 过滤：tenant、active/current、version generation、allowed principals。

## 请求路径

`/v1/knowledge/query` 由服务端身份解析后调用 `KnowledgeQueryService`，检索器只返回当前授权候选；候选先进入 `evidence_records`，模型只能选择本次账本中的 chunk。拒答判定分两层：确定性证据门只在检索层拦截字面可验证的缺失（问题点名的数量/单位、ASCII 锚点在证据中不存在）；主谓语义是否足以回答由回答模型在受约束生成时以 verdict 判定（上下文不足必须 refusal），最终引用一律经证据账本复核构造，模型无法编造出处。

`POST /v1/documents` 建立首个版本，`POST /v1/documents/{document_id}/versions` 在同一文档下建立新版本；路径和 metadata 的 document_id 必须一致。文档上传先建立 SQL 预留并独占写入对象；摄取 worker 解析并保存 manifest 后分配新投影世代。上传、版本新增、删除和投影变更共享同文档 guard，避免跨存储操作相互越过。

## Qdrant 三阶段投影

1. **构建**：新世代以 `lifecycle_status=indexing`、`current=false` 分批嵌入和 upsert，默认每批最多 64 个块；每批 retrieve 回读，最终核对块数、manifest、payload 和有限向量值。
2. **发布**：新世代改为 `active/current=true`，但 PostgreSQL 仍绑定原来的精确 `(version_id, generation)`。授权检索的 `projection_key` 来自 SQL，因此切换窗口内仍只返回旧绑定。
3. **绑定与退役**：单个 SQL 事务以原绑定为 CAS 条件激活版本并写入新绑定；成功后验证目标世代，再删除同文档的旧版本和旧 generation。旧投影清理失败不回滚已完成的 SQL 绑定，由周期审计继续收敛。

周期审计只核对当前精确世代的块数、manifest、生命周期、ACL 和 payload；健康时不重新解析或嵌入，仅在不一致时分配新 generation 重建。未绑定目标世代会在重试前清理，只保留 SQL 当前绑定；若 SQL 提交响应不确定，worker 先复查精确绑定，已经绑定时按成功处理，不能误删新投影。

Qdrant 的 `allowed_principals` 使用 `user|group|role:<非空标识>`；标识部分允许包含身份提供器命名空间分隔符。

## 状态与恢复

Research 与 controlled_action 任务均以 `thread_id = task_id` 使用 LangGraph 图执行。PostgreSQL checkpointer 保存消息、ToolMessage、Todo 与 interrupt/resume；`agent_tasks` 只保存业务生命周期、所有者、租约、中断摘要和不可变结果引用，不复制消息或用户回答正文。controlled_action 的 WRITE 在同一线程先被 HITL 暂停，再由 `ApprovalService` 绑定 checkpoint、工具、参数和证据摘要；只有服务端可构造固定 resume。worker 重启后从同一 checkpointer 恢复，不能为历史 Runtime 建立双写或回退路径。

任务预算只有一个任务级事实源：创建时由 Settings 派生的 `TaskBudgets` 以 JSON 精确持久化到 `agent_tasks.budgets`，Worker 按租约携带的持久值选择（并按小元组缓存）Agent，模型/工具两个 limit middleware 与 recursion limit 全部来自该值；环境变量只影响新任务的初始派生，不改变已持久化任务。

四个写边界全部受租约 generation fencing，失租一方的任何写入都被拒绝且新增业务数据为 0：

1. **checkpoint 写入**：`FencedPostgresSaver.put()`/`put_writes()` 在同一条 PostgreSQL 事务内对 `agent_tasks` 行 `FOR UPDATE` 复核 `lease_owner`、`lease_generation` 与数据库时钟过期时间；失租抛 `task_checkpoint_lease_stale`，checkpoint 和 pending writes 都不会新增，抢占方从最后一个合法 checkpoint 恢复。
2. **工具完成**：工具调用完成提交在同一事务内绑定租约复核；被抢占的旧 Worker 只返回 stale，不能把工具行改为终态、绑定 ref 或写 ToolMessage checkpoint。
3. **幂等回放**：身份快照（tenant/user/roles/groups/attributes 的指纹）与回放校验绑定；身份变化后相同 idempotency key 返回 `task_auth_context_changed` 且不读取旧正文。
4. **最终任务提交**：`finish_task()` 在任务行锁内以 PostgreSQL `clock_timestamp()` 终检租约、状态、取消和 deadline；deadline 已过时只能原子收敛为 `failed/task_deadline_expired` 并补偿未绑定结果对象，不可能先 succeeded 再被后台改写。

只有构建完成的 Worker 才注册 `task-agent/m2-fenced-v1` capability lease（独立心跳续租，与任务 invoke 解耦）；创建 Research 或 controlled_action 任务的仓储门在插入事务内复核 capability 名称、协议版本和租期，缺失、错版本或过期时返回 503，同步知识查询、文档接口和全局 readiness 不受影响。受控创建还受 feature gate 约束。任务执行期间按租约三分之一间隔续租，失败则本次结果不提交，任务保持可安全恢复。

执行结论与正文可用性分离：独立的 `TaskRetentionService` 周期收敛 waiting deadline reaper、工具正文 GC、任务 checkpoint/结果 GC 与 `artifacts/` 固定前缀孤儿对象扫描；GC 只删除正文对象并写 purge 标记，绝不改写 succeeded/refused 等历史执行终态。artifact 正文读取要求任务 owner 与 `task:read`，并逐项复核 tenant/task/artifact 绑定、ready 状态、TTL/purge 与 SHA-256；对象键不离开服务边界。结果读取在返回前于行锁内二次复核 deadline、授权和证据账本：到期为 410、未完成/无结果为 409、完整性错误 fail-close 为 500。"object 写成功、DB 绑定前崩溃"的对象由即时补偿加固定前缀 grace 重扫双保险兜底，删除失败保留引用供下轮重试。checkpointer 的 msgpack serializer 只 allowlist `TaskResultCandidate`、显式禁用 pickle fallback，并在 `LANGGRAPH_STRICT_MSGPACK=true` 下叠加运行时 schema allowlist。

摄取和投影处理使用后台心跳续租；失去租约后旧 worker 不能提交。已有 receipt 的发布恢复跳过解析和 embedding。达到普通重试上限但仍可能遗留 Qdrant 世代时，任务进入可重复领取的 cleanup 阶段，清理成功后才写入最终失败。

worker 周期扫描“对象已写入、SQL 尚未推进”的上传预留，只在内容哈希匹配时推进；缺失、哈希不符或单条异常记录会延后，不能饿死同批后续记录。活动摄取或未过期投影收敛租约存在时，删除失败关闭。

启用通用任务后，worker 按 `task → ingestion → projection` 轮转，每轮每队列最多处理一项；只有全部队列均空闲才休眠。Agent 调用期间按 lease 的三分之一间隔续租；续租失败时本次结果不提交，任务保持可安全恢复。恢复扫描使用有界批次和时间预算，不计作普通队列任务；模型初始化失败只安全关闭领取到的通用任务，不阻断摄取。

## 任务工具与结果

`research` 只注册 `search_knowledge`、`fetch_evidence`、`list_document_versions`、`compare_document_versions` 四个 READ 工具。`controlled_action` 复用四个 READ 工具，并只增加两个固定 WRITE adapter；每次 WRITE 均须通过 HITL 和服务端审批。工具适配器只能读取服务端注入的 `AgentRunContext`，并委托 `ToolExecutionService` 校验参数、权限和证据；模型不能传递 tenant、权限、lease、批准或存储路径。

每个 ToolMessage 只保留允许字段。任务成功前，服务端重新读取证据账本并验证 `TaskResult` 的 evidence_ids/citations，再原子绑定不可变结果 ref/hash；controlled_action 还逐项核验 artifact 的归属、ready 状态和正文 hash。缺失、越权或过期证据会使任务失败关闭。checkpointer 的 msgpack serializer 只 allowlist `TaskResultCandidate`，并显式禁用 pickle fallback。ExecutionGuard 仅统计四个 READ 业务工具：完整轨迹中连续失败达到阈值、或同参重复调用超限时提前停止；最近一次成功会重置连续计数，Todo、结构化结果和未知工具不参与计数。普通工具异常的稳定业务码原样进入 ToolMessage，`task_checkpoint_lease_stale` 等致命码直接抛出。

严格真实 M2 E2E 已在独立 PostgreSQL/Qdrant/对象根目录上验证 31 个任务、HITL、批准/拒绝、artifact 隔离、恢复窗口与 strict msgpack；故障矩阵、M1 回归、检索质量与最终 acceptance 也已通过。M2 停止于这两个固定 WRITE adapter，不进入多 Agent、MCP、长期记忆或外部写入。
