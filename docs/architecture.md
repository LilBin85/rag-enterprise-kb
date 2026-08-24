# 架构说明

## 事实源与边界

PostgreSQL 保存租户、主体、文档版本、ACL、任务、计划、批准、工具调用、产物和证据账本；Qdrant 只保存可重建的向量 projection。任何检索候选必须同时通过 SQL 授权集和 Qdrant payload 过滤：tenant、active/current、version generation、allowed principals。

## 请求路径

`/v1/knowledge/query` 由服务端身份解析后调用 `KnowledgeQueryService`，检索器只返回当前授权候选；候选先进入 `evidence_records`，模型只能选择本次账本中的 chunk。拒答判定分两层：确定性证据门只在检索层拦截字面可验证的缺失（问题点名的数量/单位、ASCII 锚点在证据中不存在）；主谓语义是否足以回答由回答模型在受约束生成时以 verdict 判定（上下文不足必须 refusal），最终引用一律经证据账本复核构造，模型无法编造出处。

文档上传先建立 SQL 预留并独占写入对象；摄取 worker 解析并保存 manifest 后分配新投影世代。上传、删除和投影变更共享同文档 guard，避免跨存储操作相互越过。

## Qdrant 三阶段投影

1. **构建**：新世代以 `lifecycle_status=indexing`、`current=false` 分批嵌入和 upsert，默认每批最多 64 个块；每批 retrieve 回读，最终核对块数、manifest、payload 和有限向量值。
2. **发布**：新世代改为 `active/current=true`，但 PostgreSQL 仍绑定原来的精确 `(version_id, generation)`。授权检索的 `projection_key` 来自 SQL，因此切换窗口内仍只返回旧绑定。
3. **绑定与退役**：单个 SQL 事务以原绑定为 CAS 条件激活版本并写入新绑定；成功后验证目标世代，再删除同文档的旧版本和旧 generation。旧投影清理失败不回滚已完成的 SQL 绑定，由周期审计继续收敛。

周期审计只核对当前精确世代的块数、manifest、生命周期、ACL 和 payload；健康时不重新解析或嵌入，仅在不一致时分配新 generation 重建。未绑定目标世代会在重试前清理，只保留 SQL 当前绑定；若 SQL 提交响应不确定，worker 先复查精确绑定，已经绑定时按成功处理，不能误删新投影。

Qdrant 的 `allowed_principals` 使用 `user|group|role:<非空标识>`；标识部分允许包含身份提供器命名空间分隔符。

## 状态与恢复

任务节点为 load、classify、clarify、plan、select action、execute read、record evidence、evaluate、await approval、execute approved action、verify、generate result、terminal。每次状态变更带 state version、lease generation 和事件；CAS 失败、过期 lease、重复 action/query/result、预算超限和 UNKNOWN 均失败关闭。worker 重启后依据 checkpoint 与 pending action 恢复，不重放不确定写副作用。

摄取和投影处理使用后台心跳续租；失去租约后旧 worker 不能提交。已有 receipt 的发布恢复跳过解析和 embedding。达到普通重试上限但仍可能遗留 Qdrant 世代时，任务进入可重复领取的 cleanup 阶段，清理成功后才写入最终失败。

worker 周期扫描“对象已写入、SQL 尚未推进”的上传预留，只在内容哈希匹配时推进；缺失、哈希不符或单条异常记录会延后，不能饿死同批后续记录。活动摄取或未过期投影收敛租约存在时，删除失败关闭。

启用通用任务后，worker 按 `task → ingestion → projection` 轮转，每轮每队列最多处理一项；只有全部队列均空闲才休眠。恢复扫描使用有界批次和时间预算，不计作普通队列任务；模型初始化失败只安全关闭领取到的通用任务，不阻断摄取。

同步问答使用 120 秒 API 自租约，回答模型超时为 60 秒且不做模型层重试。通用任务领取排除 `knowledge_query`；API 终态写入必须匹配未过期 owner/generation。API 崩溃后，回收器将过期问答收敛为 `failed/knowledge_query_lease_expired`，不重放检索或模型，迟到响应也不能覆盖终态。

## 三模式与工具

`knowledge_query` 只读；`research` 可暂停/继续并持久化证据；`controlled_action` 的写动作必须有当前批准。六个固定工具通过 schema、权限、风险、超时、验证器和脱敏策略注册。对象产物使用服务端生成 storage key，exclusive 写入和读回 hash 校验；任何正文不写入长期任务/批准/证据记录。

工具意图和执行期限先持久化为 `running` 再调用适配器。超时记录为 `UNKNOWN/tool_timeout`，因为线程超时不能证明副作用未发生；写工具同时把任务和关联产物收敛为 `unknown`，批准保持已消费且不自动重试。迟到适配器不能把产物重新提升为 `ready`；被卡住的线程池会被替换以恢复后续调用，但旧线程仍可能结束，因此超时不等于执行已取消。worker 对过期 `running` 工具只做状态收敛，不重新执行。
