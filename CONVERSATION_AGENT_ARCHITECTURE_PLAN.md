# 统一会话与智能体架构方案

## 目标与边界

对外统一为 `POST /v1/conversations` 创建会话、`POST /v1/conversations/{conversation_id}/messages` 提交回合；原有快速问答和任务创建/澄清写入口不再公开。会话是用户可见的长期上下文边界，任务仍是耐久执行单元。项目运行时仍只有快速执行器与耐久智能体：本方案不引入第二个运行时 subagent。

## API 契约

创建会话标题长度为 1–200。新消息与澄清消息是以 `kind` 为判别字段的严格联合，所有请求禁止未知字段：

- `new`：`content` 长度 1–8000，`mode` 为 `auto|answer|research|controlled_action`。
- `clarification`：必须提供 `turn_id`、`task_id`、`expected_task_state_version`，且禁止 `mode`。
- `Idempotency-Key` 是消息提交必填请求头，长度 1–200；相同 key 与不同语义请求返回 `idempotency_conflict`。

提交响应固定包含 `conversation_id`、`conversation_state_version`、`turn_id`、`user_message(message_id,sequence)`、`requested_mode`、`selected_mode`、`route_reason_code`、`turn_status`、`task(task_id,status,state_version,result_available)`、`assistant_message` 与 `idempotent_replay`。快速回答同步返回 200；耐久任务接受返回 202。历史消息固定包含 `message_id`、`turn_id`、`sequence`、`role`、`kind`、`content`、`visibility`、`redaction_reason`、`citations`、`artifacts`、`created_at`。

会话支持创建、列表、读取、读取消息和删除。删除返回 202，进入 deleting 状态；会话不存在、跨租户或非 owner 一律返回 404。旧 `POST /v1/knowledge/query`、`POST /v1/tasks`、`POST /v1/tasks/{task_id}/continue` 必须从路由与 OpenAPI 删除并返回 404。任务的 GET/result/events/artifacts/cancel 与审批接口保留；任务读取响应新增可空 `conversation_id` 和 `turn_id`。

## 路由与执行模型

Web 路由只做身份解析、请求校验、调用应用服务及稳定错误映射，不实现业务状态机。`auto` 由会话路由器选择：可即时、只读且无需澄清/审批的请求走快速执行器；研究、多步骤、澄清、审批或受控写入走耐久智能体。`controlled_action` 必须显式选择，且能力不可用时拒绝；不得由模型或客户端绕过审批。

应用服务负责在同一业务边界内写入会话、turn、用户消息、路由结果和幂等记录。耐久路径创建任务并绑定会话/turn；快速路径写入经引用复核的 assistant 消息。澄清只能恢复绑定的任务、turn 与预期状态版本，不能直接构造内部 checkpoint、命令或工具审批载荷。

## 记忆、可见性与保留

会话历史按稳定 sequence 追加，用户和 assistant 消息均有可见性。模型上下文只由服务端从可见、未删除且未被脱敏的消息构造；秘密、内部工具载荷、对象键和 checkpoint 不进入 API 或模型记忆。`redaction_reason` 表示内容因保留策略、权限或安全策略不可读。引用和 artifacts 仅返回经过 owner/租户授权的展示元数据；正文继续通过现有受控读取接口提供。

## 安全与错误语义

身份、租户、owner、权限、任务绑定、模式和 worker capability 均由服务端验证。稳定错误码包括：`conversation_not_found`(404)、`conversation_deleting`(409)、`conversation_has_active_turn`(409)、`idempotency_conflict`(409)、`clarification_binding_invalid`(409)、`task_state_conflict`(409)、`controlled_action_explicit_mode_required`(409)、`controlled_action_unavailable`(503)、`task_agent_unavailable`(503)、`conversation_router_unavailable`(503) 和 `conversation_content_unavailable`(410 或 409，遵循核心语义)。错误响应不得回显 prompt、令牌、内部引用或执行载荷。

## Codex 开发实施编排（非项目运行时架构）

本次开发可由并行 Codex 子任务分别处理领域/应用/基础设施、Web 契约和迁移/测试；这只是研发协作方式，不是产品架构，不新增运行时双代理或 subagent。合并前必须以核心应用服务的真实 DTO、方法和错误码为唯一依据完成 Web 接线。

## 测试与交付提醒

最小 Web/OpenAPI 测试覆盖：严格请求校验、new/clarification 判别、状态码、跨租户 404、稳定错误映射、幂等重放、旧写入口从 OpenAPI 消失，以及 TaskResponse 的新绑定字段。定向测试不启动 Docker；真实 PostgreSQL/Qdrant、迁移、Worker capability 与 Docker Compose 验证需在后续集成环境运行。不要在 API 进程执行迁移、下载模型或启动 worker。

## 一、数据、状态机与一致性

会话至少保存 tenant、owner、标题、状态、state_version、创建/更新时间和删除请求时间；turn 至少保存会话绑定、请求/选择模式、路由原因、状态、任务绑定和幂等语义；消息保存可展示正文及其可见性/脱敏、引用、artifact 摘要和 sequence。所有读取均以 tenant 与 owner 过滤，避免以 403 暴露资源存在性。

会话状态为 `active → deleting → deleted/purged`；`deleting` 期间拒绝新消息。turn 的终态由快速回答或耐久任务结果决定，未终结的 turn 阻止删除；删除请求只耐久化意图，清理由后台保留流程执行。状态版本是乐观锁，澄清与取消必须携带所绑定任务的当前版本；任何陈旧版本不可覆盖更晚的任务/会话状态。

幂等键的作用域为会话与调用者。首次请求保存规范化请求摘要和完整业务结果；同 key 同摘要直接返回相同结果并标记 `idempotent_replay=true`；同 key 不同摘要返回 409。不得把重放实现成再次调用模型、重新创建任务或重新发送受控动作。

## 二、路由、模式与预算

`answer` 只允许快速执行器，`research` 只允许耐久智能体，`controlled_action` 只能显式请求并经过 feature/capability/审批检查。`auto` 只可在服务端有可用路由器时选择 answer 或 research，不能自动升级为受控动作。路由原因采用稳定枚举码，供客户端解释而不泄露提示词或内部分类分数。

快速执行器维持现有检索、引用校验、超时、候选数和上下文预算；耐久智能体沿用任务预算、工具 allowlist、checkpoint、lease/fencing、deadline 和恢复窗口。会话仅保存经过允许的摘要/消息，不把完整工具调用、模型隐藏推理、认证属性、对象存储键或 worker lease 复制到记忆。一次用户消息至多创建一个绑定任务；澄清不创建新目标，只恢复已绑定且 `waiting_for_user` 的任务。

## 三、记忆、证据、隐私与审计

记忆分为可展示消息、服务端上下文摘要和不可公开执行记录。模型可见上下文必须通过租户、owner、visibility、删除状态与脱敏策略筛选；引用引用知识版本/片段的公开标识，artifact 仅为授权后的展示摘要。正文依旧受既有任务 artifact content 读取策略保护。

安全策略可将消息正文改为不可见并保留 `redaction_reason`，但 sequence、审计时间和最少生命周期信息仍可保留。会话删除要先阻止新写、检查活动 turn/任务、异步撤销可清理内容、保留必要审计事实，再标记完成；过期内容的读取依核心语义返回 410 或 409。日志、指标、trace 不记录正文、问题、回答、token、引用正文或内部 binding。

## 四、错误矩阵与测试矩阵

| 场景 | 预期 |
| --- | --- |
| 不存在、跨租户、非 owner 会话 | `conversation_not_found` / 404 |
| 删除中的会话 | `conversation_deleting` / 409 |
| 删除时存在活动 turn | `conversation_has_active_turn` / 409 |
| 幂等键复用且请求不同 | `idempotency_conflict` / 409 |
| 澄清的 turn/task/version 绑定不符 | `clarification_binding_invalid` 或 `task_state_conflict` / 409 |
| 受控动作未显式选择 | `controlled_action_explicit_mode_required` / 409 |
| 受控动作或任务 worker 不可用 | `controlled_action_unavailable`、`task_agent_unavailable` / 503 |
| 会话路由器不可用 | `conversation_router_unavailable` / 503 |
| 会话内容不可读取 | `conversation_content_unavailable` / 410 或 409 |

单元测试覆盖判别联合、长度边界、未知字段、Idempotency-Key、响应字段、错误码和不泄露内容。Web 测试通过假服务覆盖 create/list/get/messages/delete、200/201/202、跨租户 404、模式路由、重放、澄清绑定与 OpenAPI。应用/基础设施测试覆盖事务、版本冲突、owner 过滤、删除状态、任务绑定和保留；集成测试验证迁移、PostgreSQL、Qdrant 和 Worker capability。

## 五、迁移、Docker 与发布顺序

这是一次明确的破坏性后端发布，不提供旧写入口兼容层。发布顺序为：完成非 Docker 验证；在维护窗口应用 `20260906_0016`；部署新任务 worker；部署新 API；检查数据库 revision、worker capability、统一会话冒烟用例和升级前旧任务恢复。迁移包含明确的 revision/down_revision、索引、唯一约束和拒绝有会话数据时降级的保护；API readiness 的 schema revision 同步更新。发布后观察会话创建、路由、幂等冲突、任务入队、删除积压、4xx/5xx 和 lease/capability 指标。

Docker 集成只在所有非 Docker 测试通过后执行一次。需要完整异步链路时，主代理先暂停并提醒用户运行：

```powershell
docker compose up -d postgres qdrant migrate qdrant-init task-worker
```

如果实际只剩数据库迁移验证，则只提醒运行：

```powershell
docker compose up -d postgres
```

两条命令二选一，不重复执行；无需启动可观测性组件。用户确认容器健康后再验证 PostgreSQL、Qdrant、迁移和任务 worker。集成完成后不自行执行 `docker compose down`，容器是否停止由用户决定。开发机没有 Docker 时不得伪造验证结论。
