# 迁移说明

项目已从旧的巨型 `rag_kb.api` 与 role-only 检索路径迁移到 `/v1`、PostgreSQL 事实源、tenant/ACL/generation-aware projection 和持久任务运行时。

## 迁移顺序

1. 运行 `alembic upgrade head`，当前目标为 `20260822_0008`。
2. 启动 PostgreSQL、Qdrant，再启动 API 与 `worker --include-tasks`。
3. 使用新版 `/v1` 客户端；上传请求补齐 `Idempotency-Key`，查询使用服务端身份。
4. 旧客户端在兼容期通过显式 `include_legacy_routes=True` 接入；响应带 `Deprecation`、`Sunset`、`Link`，业务委托新版服务。
5. 以机器报告 `docs/acceptance/latest.json` 验证迁移结果，不沿用旧 README 或旧报告数字。

## 旧测试与文档

旧 planner/store、旧 API 内部 builder 和 role-only 检索测试不再作为新架构验收依据；保留的领域测试仅验证仍然独立存在的服务契约。迁移映射见 [stage6-test-migration.md](stage6-test-migration.md)。旧验收报告、中文学习笔记及用户材料仅作历史参考，不删除、不覆盖。

本次实际删除的已替代内容：

| 分类 | 删除内容 | 替代覆盖 |
| --- | --- | --- |
| 已替代代码 | `policy_rollout_planning.py`、`policy_rollout_store.py`、`policy_rollout_execution.py` | `application/task_runtime.py`、`application/task_worker.py`、PostgreSQL task repository、固定工具与审批服务 |
| 已替代脚本 | `benchmark_planner_workflow_agent.py`、`diagnose_planner_decision.py` | `enterprise_task_evaluation.yaml`、新版任务恢复/安全测试与机器验收报告 |
| 已替代测试 | `test_policy_rollout_planning.py`、`test_policy_rollout_execution.py` | `test_task_runtime.py`、`test_task_worker_recovery.py`、`test_approval_tool_execution.py`、`test_v1_tasks_api.py` |
| 已替代测试 | `test_planner_workflow_agent_benchmark.py`、`test_diagnose_planner_decision.py` | `test_enterprise_task_evaluation_dataset.py`、`test_task_planner_model.py`、恢复与安全门测试 |
| 过期评测集 | `planner_workflow_agent.yaml` | `enterprise_task_evaluation.yaml`（development/calibration/test 三分） |
