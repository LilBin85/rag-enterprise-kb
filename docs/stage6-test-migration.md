# 阶段 6 测试迁移说明

本轮将已删除的 `rag_kb.api` 内部构建器依赖迁移到公开 web 应用与依赖注入契约。

| 原测试 | 处理 |
| --- | --- |
| `tests/test_api.py` | 改为验证 legacy 路由默认关闭、显式开启后进入认证；不再测试旧 lifecycle/store。 |
| `tests/test_answering.py` | 保留 `AnswerService` 领域测试；移除旧 `/questions` endpoint 测试，公开路由由 `test_compatibility_routes.py` 覆盖。 |
| `test_policy_compare.py` 的旧 api builder/endpoint/token 测试 | 删除过时内部符号依赖；领域比较、失败关闭和版本生命周期测试保留。 |
| `test_policy_rollout_planning.py` | 在通用 runtime/worker、计划修订、恢复和兼容契约覆盖后删除；不恢复旧 planner/store builder。 |
| `test_policy_rollout_execution.py` | 在统一工具执行、批准、幂等、UNKNOWN 和产物验证覆盖后删除。 |

替代公开契约集中在 `tests/test_compatibility_routes.py`：身份与 body role 校验、薄委托、`Deprecation`/`Sunset`/`Link`、版本布尔值拒绝、任务仓储委托和 legacy 默认关闭。
