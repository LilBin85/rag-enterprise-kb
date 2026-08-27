# 迁移说明

项目已从旧的巨型 `rag_kb.api` 与 role-only 检索路径迁移到 `/v1`、PostgreSQL 事实源、tenant/ACL/generation-aware projection 和 LangGraph PostgreSQL checkpointer 驱动的 M1 Research Agent。

## 迁移顺序

1. 运行 `alembic upgrade head`，当前目标为 `20260826_0012`，再运行 `python scripts/init_agent_checkpointer.py` 初始化 checkpointer schema。
2. 启动 PostgreSQL、Qdrant，再启动 API 与 Worker；只有 Worker 全部构建完成并注册 `m1-fenced-v2` capability lease 后才恢复 Research 任务入口。
3. 使用新版 `/v1` 客户端；首个版本使用 `POST /v1/documents`，同一文档的后续版本使用 `POST /v1/documents/{document_id}/versions`，两者都必须提供 `Idempotency-Key`。
4. 旧接口兼容层已删除，客户端必须使用 `/v1` 接口。
5. 以机器报告 `docs/acceptance/latest.json` 验证迁移结果，不沿用旧 README 或旧报告数字。

## 0010 Research Agent 状态收缩

| 变更 | 语义 |
| --- | --- |
| `agent_tasks` | 删除旧 Runtime 业务状态，保留 Research 生命周期、中断摘要、结果引用和租约字段。 |
| `tool_calls` | 持久化 READ 工具结果 ref/hash，供 checkpointer 恢复后的幂等回读。 |
| 旧任务 | 遇到旧 Runtime 状态或不可解释的 pending action 时失败关闭，不能作为 M1 任务恢复。 |

## 0011 retention 标记与任务物理收缩

| 变更 | 语义 |
| --- | --- |
| `agent_tasks.checkpoint_purged_at` / `result_purged_at` | 幂等重扫依据：保留窗口清理只在首次成功时写入，重复扫描不命中。 |
| 旧 Runtime 执行列 | 删除 `current_node`、`token_usage`、pending action 等全部遗留执行字段与关联约束（方案 A 删列清单）。 |
| 历史行修复 | 升级前回填/修正不可解释的历史状态，再重建约束；无主 succeeded/refused 结果按失败关闭处理。 |

## 0012 capability 租约与正文生命周期

| 变更 | 语义 |
| --- | --- |
| `agent_capability_leases` | `(capability_name, protocol_version, worker_id)` 主键 + 到期索引；创建 Research 任务在插入事务内复核活动租约，缺失/错协议/过期返回 `research_agent_unavailable`（503）。 |
| `agent_tasks.content_expires_at` | 任务结果正文的到期时间；与执行终态分离，GC 只删正文不改写 succeeded/refused。 |
| `tool_calls.result_expires_at` / `result_purged_at` | 工具正文保留窗口与幂等清除标记；hash 与 verified/completed 时间戳永久保留供审计。 |
| 约束重建 | `research_result_complete`、`succeeded_result_reference_complete` 等 CHECK 在历史行修复后重建。 |

## Stop-the-world 发布顺序

1. 关闭 Research 任务创建入口（或停止对外接受新任务）。
2. 停止全部任务 Worker，确认 capability lease 已过期或手动删除。
3. 备份数据库与对象目录（若环境包含需保留数据）。
4. 执行 `alembic upgrade head` 至 `20260826_0012` 并初始化 checkpointer schema。
5. 部署 API 与 Worker 新代码。
6. 启动 Worker，确认 capability 名称 `research-agent`、协议 `m1-fenced-v2` 的租约有效。
7. 恢复任务入口并执行并发故障注入与严格真实 E2E。

**禁止混跑**：已上线 0012 后不允许旧协议 Worker 接入任务队列；capability 门会拒绝一切非 `m1-fenced-v2` Worker 创建/领取语义。降级只能恢复物理结构，不能恢复已删除的任务/工具正文和已终止状态；正文删除不可逆，异常优先前向修复。已在 0012 上运行过 Worker 的环境仅在确认数据兼容时才允许 downgrade，且 downgrade/upgrade 后必须重新验证上述 round-trip 命令。

```powershell
alembic upgrade head
alembic current
conda run -n rag-kb --no-capture-output python -m pytest -q tests/test_database_migrations.py
```

`alembic current` 应显示 `20260826_0012 (head)`。PostgreSQL round-trip 测试需要显式空临时库：设置 `RAG_KB_TEST_POSTGRES_MIGRATION_URL` 后运行同一测试文件，upgrade/downgrade/upgrade 序列不得以 metadata 建表伪造。

## 旧测试与文档

旧 Planner/Runtime、pending-action 调度和确定性任务模拟不再是 M1 Agent 的验收依据。当前覆盖由 Agent 工厂、PostgreSQL checkpointer interrupt/resume、READ 工具适配器、结果读取、版本上传和隔离真实 Research E2E 组成。旧验收报告、中文学习笔记及用户材料仅作历史参考，不删除、不覆盖。
