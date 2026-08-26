# 迁移说明

项目已从旧的巨型 `rag_kb.api` 与 role-only 检索路径迁移到 `/v1`、PostgreSQL 事实源、tenant/ACL/generation-aware projection 和 LangGraph PostgreSQL checkpointer 驱动的 M1 Research Agent。

## 迁移顺序

1. 运行 `alembic upgrade head`，当前目标为 `20260825_0010`，再运行 `python scripts/init_agent_checkpointer.py` 初始化 checkpointer schema。
2. 启动 PostgreSQL、Qdrant，再启动 API 与 `worker --include-tasks`。
3. 使用新版 `/v1` 客户端；首个版本使用 `POST /v1/documents`，同一文档的后续版本使用 `POST /v1/documents/{document_id}/versions`，两者都必须提供 `Idempotency-Key`。
4. 旧接口兼容层已删除，客户端必须使用 `/v1` 接口。
5. 以机器报告 `docs/acceptance/latest.json` 验证迁移结果，不沿用旧 README 或旧报告数字。

## 0010 Research Agent 状态收缩

| 变更 | 语义 |
| --- | --- |
| `agent_tasks` | 删除旧 Runtime 业务状态，保留 Research 生命周期、中断摘要、结果引用和租约字段。 |
| `tool_calls` | 持久化 READ 工具结果 ref/hash，供 checkpointer 恢复后的幂等回读。 |
| 旧任务 | 遇到旧 Runtime 状态或不可解释的 pending action 时失败关闭，不能作为 M1 任务恢复。 |

`20260825_0010` 必须与 M1 应用代码一起部署。Compose 的 `migrate` 服务会先执行 `alembic upgrade head`，再初始化 PostgreSQL checkpointer；不要只升级 SQL 表而遗漏 checkpointer schema。

```powershell
alembic upgrade head
alembic current
conda run -n rag-kb --no-capture-output python -m pytest -q tests/test_database_migrations.py
```

`alembic current` 应显示 `20260825_0010`。

## 旧测试与文档

旧 Planner/Runtime、pending-action 调度和确定性任务模拟不再是 M1 Agent 的验收依据。当前覆盖由 Agent 工厂、PostgreSQL checkpointer interrupt/resume、READ 工具适配器、结果读取、版本上传和隔离真实 Research E2E 组成。旧验收报告、中文学习笔记及用户材料仅作历史参考，不删除、不覆盖。
