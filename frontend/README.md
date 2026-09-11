# 浏览器工作台

这是本地开发和演示用的 React/Vite 前端，直接调用后端的版本化 `/v1` 接口。

## 启动

先按项目根目录的 [README](../README.md) 启动后端、PostgreSQL、Qdrant 和 Worker，并确认 `http://localhost:8000/health/ready` 返回 `200`。

```powershell
cd D:\PythonProject\rag-enterprise-kb\frontend
npm install
npm run dev
```

在浏览器打开 `http://localhost:5173`。Vite 会将 `/v1` 代理到 `http://localhost:8000`，因此前端不需要保存后端地址或 API Key。

## 演示身份

1. 首次部署、修改 `DEMO_ROLE_TOKENS` 或新增角色后，先在项目根目录运行 `conda run -n rag-kb --no-capture-output python backend/scripts/init_demo_principals.py`。
2. 从根目录 `.env` 的 `DEMO_ROLE_TOKENS` 中只复制一个令牌值。
3. 打开工作台，粘贴令牌并连接。
4. 使用 `admin` 演示上传、提问和任务；`approver` 是可选的独立审批身份，受控操作演示需要它，发起者不能自审批。

令牌仅保存于当前标签页的 `sessionStorage`，不会写入源码、构建变量或持久缓存。切换身份会取消旧查询、关闭旧流并清除旧身份数据。轮换令牌后，按根目录使用指南同步主体并重建 `api`，再在工作台断开并使用新令牌连接。

## 业务行为

- 对话首条消息会创建会话。普通问答以 SSE 显示累计 `answer.preview` 未校验草稿；只有服务器发送 `done` 后才用正式答案与来源替换草稿。
- SSE 的 `error` 或网络中断会清除草稿并刷新服务器会话状态，不会用新幂等键重新生成；浏览器关闭流不等于取消服务器任务。
- 研究、受控操作和澄清继续使用同一个会话消息入口，任务状态、结果和产物在任务详情抽屉中读取。
- 上传返回 `202` 仅表示服务器已接收。文档页面会轮询摄取状态，只有版本显示“可用于问答”时才可检索。
- 审批只对拥有 `task:approve` 的身份显示。批准只表示恢复命令已被接受，不代表任务已经执行完成；冲突时页面会重新读取详情并要求再次确认。

## 验证

```powershell
npm run typecheck
npm run test
npm run build
npm run test:e2e
```

`test:e2e` 需要本机已有 Playwright Chromium。Windows 已安装 Edge 时，可临时使用 `$env:PLAYWRIGHT_CHANNEL = "msedge"; npm run test:e2e`，无需下载浏览器。否则，先在获得许可后执行 `npx playwright install chromium`，随后重试。真实端到端演示还需要正在运行的后端与有效演示令牌。`RAG_KB_E2E_ADMIN_TOKEN` 启用真实上传和问答用例，`RAG_KB_E2E_APPROVER_TOKEN` 额外启用独立审批用例；未设置的用例会跳过，测试不会把令牌写入文件。

## 常见问题

- 连接返回 `403`：确认使用的是 `.env` 中单个令牌值，且已经同步演示主体。
- 页面请求失败：确认 Vite 运行在 `5173`，后端运行在 `8000`，并检查 `/health/ready`。
- 文档长时间“排队中”：检查 `ingestion-worker` 是否运行；上传成功不代表索引完成。
- 批准后任务没有变化：检查 `task-worker` 是否运行，并通过会话的任务详情查看后续状态。
