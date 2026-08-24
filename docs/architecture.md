# 架构说明

## 事实源与边界

PostgreSQL 保存租户、主体、文档版本、ACL、任务、计划、批准、工具调用、产物和证据账本；Qdrant 只保存可重建的向量 projection。任何检索候选必须同时通过 SQL 授权集和 Qdrant payload 过滤：tenant、active/current、version generation、allowed principals。

## 请求路径

`/v1/knowledge/query` 由服务端身份解析后调用 `KnowledgeQueryService`，检索器只返回当前授权候选；候选先进入 `evidence_records`，模型只能选择本次账本中的 chunk。拒答判定分两层：确定性证据门只在检索层拦截字面可验证的缺失（问题点名的数量/单位、ASCII 锚点在证据中不存在）；主谓语义是否足以回答由回答模型在受约束生成时以 verdict 判定（上下文不足必须 refusal），最终引用一律经证据账本复核构造，模型无法编造出处。

文档上传由 `DocumentService` 写入对象存储和 SQL 任务，worker 执行解析、索引和 receipt，最后才发布 active。

## 状态与恢复

任务节点为 load、classify、clarify、plan、select action、execute read、record evidence、evaluate、await approval、execute approved action、verify、generate result、terminal。每次状态变更带 state version、lease generation 和事件；CAS 失败、过期 lease、重复 action/query/result、预算超限和 UNKNOWN 均失败关闭。worker 重启后依据 checkpoint 与 pending action 恢复，不重放不确定写副作用。

## 三模式与工具

`knowledge_query` 只读；`research` 可暂停/继续并持久化证据；`controlled_action` 的写动作必须有当前批准。六个固定工具通过 schema、权限、风险、超时、验证器和脱敏策略注册。对象产物使用服务端生成 storage key，exclusive 写入和读回 hash 校验；任何正文不写入长期任务/批准/证据记录。

