# API/Worker 架构重构续跑 Prompt

请在 `D:\PythonProject\rag-enterprise-kb` 中继续实施 API/Worker 架构重构。

唯一执行计划与状态事实源：

`D:\PythonProject\rag-enterprise-kb\ASYNC_WORKER_REFACTOR_PLAN.md`

当前进度基线：

- 0A、A1、A2、A3、A4、A5 已完成；下一节点是 A6。
- A5 已通过真实 PostgreSQL A5 集成测试 36 项、Alembic 真库往返 1 项、SQLite/API 核心回归 104 项。
- A5 临时 `postgres:18.6` 容器及三个测试数据库已删除；需要真库时必须新建隔离资源。
- 运行环境已明确不支持 Luna Max。简单节点按计划由主 Agent 亲自完成并如实记录；不得假称使用 Luna Max。
- 高难度节点仍必须使用 Terra Max，推理强度 `max`；不可用时停止该节点，不得降级模型要求。

强制执行规则：

1. 开始工作前，从头完整读取 `ASYNC_WORKER_REFACTOR_PLAN.md`，不能只读摘要、状态表或局部章节。
2. 检查当前分支、HEAD、工作树和受保护资产；保留用户已有未提交修改，不使用破坏性 Git 命令，不擅自提交或推送。
3. 严格从状态表中的第一个 `待执行` 节点开始，即 A6；一次只能有一个节点处于 `进行中`。
4. A6 是简单节点：由于 Luna Max 已确认不可用，由主 Agent 亲自执行，并在计划文档中保持这一模型事实。不得把 A6 扩大为无关重构。
5. A7、B1、B2、C1 等高难度架构、事务状态机、迁移、并发、fencing、崩溃恢复和跨模块节点，必须逐节点委派给 Terra Max，推理强度设为 `max`。
6. B3、B4 等简单、机械且边界明确的配置、文档、静态检查和测试同步节点，由主 Agent 亲自完成，并继续记录 Luna Max 不可用。
7. 必须开启并实际使用 subagent；每次只能委派一个边界明确的计划节点。委派内容必须包含：TASK、EXPECTED OUTCOME、REQUIRED TOOLS、MUST DO、MUST NOT DO、CONTEXT。
8. 不得并行修改同一文件、同一数据库状态机或同一 Worker 运行链；不得在任务一完成前提前混改任务二共享运行链。
9. 每个节点完成后，主 Agent 必须依次执行：
   - 读取子 Agent 的实际改动；简单节点则逐项读取自己的实际差异；
   - 检查完整差异，不能只相信总结；
   - 使用项目 `rag-kb` 环境运行该节点最相关的最小验证；
   - 更新计划文档中的状态、证据、实际改动文件和遗留风险；
   - 更新完成后，再次从头完整读取整份计划文档；
   - 全文重读完成后，才能决定并启动下一节点。
10. 文件修改使用精确补丁；不做无关重构、格式化、依赖升级或功能扩展。
11. 不读取、输出或提交 `.env`；不把仓库语料、用户答案、审批原因、验收正文或凭据发送给外部服务。
12. 未经用户另行授权，不调用真实 DeepSeek，不执行会发送敏感数据的网络请求。
13. 继续使用 PostgreSQL claim、lease、generation 和 fencing；不得引入 Redis、Celery、Kafka 或其他消息中间件。
14. 需要真实 PostgreSQL 验收时，可使用 Docker Desktop CLI 的绝对路径：
    `C:\Users\26219\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`
    创建临时 `postgres:18.6` 容器，只绑定 `127.0.0.1` 随机端口，并分别创建主测试库、空迁移库和严格 E2E 库。不得复用或修改项目现有业务数据库；验收结束后删除临时容器和测试库。
15. 最终 C1 执行完整回归、集成测试、`docker compose config`、隐私扫描、`git diff --check` 和临时资源清理；在此之前每个节点只做最小相关验证。
16. 遇到非阻塞问题时按计划中的最小安全方案继续；只有涉及架构目标变化、敏感数据外发、破坏性操作或无法安全决定的事务语义时才暂停询问。
17. 工作期间持续提供简短中文进度，不复制整段原始工具输出。
18. 直到计划全部完成或出现真实阻塞前，不要提前结束；如果用户明确要求某节点完成后暂停，则完成“更新文档 → 全文重读”后暂停。

现在开始：

- 完整读取 `ASYNC_WORKER_REFACTOR_PLAN.md`；
- 检查工作树与受保护资产；
- 将 A6 标为 `进行中`；
- 按 Luna Max 不可用的既定规则，由主 Agent 以最小改动完成 A6；
- 完成后严格执行“更新文档 → 主 Agent 全文重读 → 下一节点”的循环。
