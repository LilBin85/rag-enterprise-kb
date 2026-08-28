# 威胁模型

## 资产

租户边界、文档正文与版本、ACL、证据引用、Research 与 controlled_action 任务状态、审批 binding、artifact 正文与密钥。

## 主要威胁与控制

- 越权或跨租户检索：身份由服务端解析；SQL ACL 与 Qdrant tenant/active/version/principal 过滤双重约束；证据账本再次复核。
- 同名文档或旧版本泄漏：所有主键查询带 tenant；撤回、superseded、旧 generation 和非 current projection 不进入授权集合。
- Prompt injection：文档、问题和模型输出均是不可信数据；Research Agent 只能调用四个固定 READ 工具并使用本次证据 ID，不能选择角色、路径、集合或任意工具。
- 未审批 WRITE：controlled_action 仅有两个固定 WRITE adapter；每次执行都先经过 HITL 暂停，再由服务端校验的审批决定恢复，同一 action 的重复恢复只回读既有结果，UNKNOWN 不自动重放。
- 审批越权或自批：审批 discovery/decision 要求 `task:approve`，按 tenant 过滤；发起人自批拒绝，客户端不能提交 checkpoint、resume、工具名、参数或审批 binding。
- artifact 正文越权或泄漏：读取要求任务 owner 与 `task:read`，逐项复核 tenant/task/artifact、ready、TTL/purge 和 SHA-256；跨租户或非 owner 不泄露存在性，响应与日志不返回对象键或正文元数据。
- 重复或不确定上传：idempotency key 唯一；对象 exclusive 写入；文档版本、摄取和投影按同一文档串行化，异常不伪报成功。
- 旧 worker 覆盖新投影：lease generation、projection generation、receipt 和 CAS 绑定；旧世代提交被拒绝。
- 路径、网络、命令注入：固定工具无 shell、SQL、URL、任意路径或外部系统能力；对象键由服务端生成。
- 日志泄漏：长期记录只存 hash、ID、计数、耗时和错误码，不存密钥、正文、提示词或完整模型输出。

## 未解决边界

真实模型行为、外部 Provider 重试、生产身份提供器和部署级网络隔离不由离线测试证明。`docs/acceptance/latest.json` 保留其记录 worktree 的 M1/M2 严格真实 E2E、故障注入、双库回归和检索质量证据；它不验证当前工作树，也不替代生产身份、网络隔离或持续运行监控。
