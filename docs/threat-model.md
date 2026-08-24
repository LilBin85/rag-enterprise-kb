# 威胁模型

## 资产

租户边界、文档正文与版本、ACL、证据引用、批准事实、对象产物、任务状态和密钥。

## 主要威胁与控制

- 越权或跨租户检索：身份由服务端解析；SQL ACL 与 Qdrant tenant/active/version/principal 过滤双重约束；证据账本再次复核。
- 同名文档或旧版本泄漏：所有主键查询带 tenant；撤回、superseded、旧 generation 和非 current projection 不进入授权集合。
- Prompt injection：文档、问题和模型输出均是不可信数据；模型只能选择固定 action/schema 和本次证据 ID，不能选择角色、路径、集合或任意工具。
- 重复或不确定写入：批准绑定不可变摘要；idempotency key 唯一；对象 exclusive 写入；超时/崩溃进入 UNKNOWN，不伪报成功、不自动重试写副作用。
- 旧 worker 覆盖新投影：lease generation、projection generation、receipt 和 CAS 绑定；旧世代提交被拒绝。
- 路径、网络、命令注入：固定工具无 shell、SQL、URL、任意路径或外部系统能力；对象键由服务端生成。
- 日志泄漏：长期记录只存 hash、ID、计数、耗时和错误码，不存密钥、正文、提示词或完整模型输出。

## 未解决边界

真实模型行为、外部 Provider 重试、生产身份提供器和部署级网络隔离不由离线测试证明。请以 `docs/acceptance/latest.json` 的 `unverified` 与质量门状态为准；真实检索质量门当前可能失败。

