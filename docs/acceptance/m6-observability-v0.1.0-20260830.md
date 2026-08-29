# M6 可观测性验收指标

## 快照信息

| 字段 | 值 |
| --- | --- |
| 项目 | `rag-kb` |
| 项目版本 | `0.1.0` |
| 记录时间 | `2026-08-30T02:17:36+08:00` |
| 分支 | `main` |
| 基线提交 | `1163232` |
| 验收对象 | 基线提交之上的当前未提交工作树 |
| 最终结论 | **PASS** |

本文件是本轮 M6 可观测性验收的事实快照。由于验收改动尚未提交，`1163232` 仅表示工作树基线，不表示这些结果已包含在该提交中。

## 自动化验证

| 项目 | 结果 |
| --- | --- |
| 全量 pytest | `473 passed, 42 skipped, 0 failed` |
| 全量 pytest 用时 | `206.03s` |
| Python 编译检查 | `compileall` 通过 |
| Compose 配置 | `docker compose config --quiet` 通过 |
| Prometheus 配置 | 通过 |
| Prometheus recording rules | `18` 条，通过 |
| OpenTelemetry Collector 配置 | `validate` 通过 |
| Git whitespace 检查 | `git diff --check` 通过，仅有 CRLF 转换提示 |

## 性能门禁

测试方法为单并发、AB/BA 成对交替、每臂预热 40 次。P95 开销按 `(enabled_p95 / disabled_p95 - 1) * 100` 计算，置信区间使用固定随机种子的 1000 次 paired bootstrap。

| 场景 | 每臂样本 | 关闭 P50 | 关闭 P95 | 关闭 P99 | 开启 P50 | 开启 P95 | 开启 P99 | P95 开销 | 门限 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 普通 API | 600 | 0.974ms | 1.405ms | 1.786ms | 1.041ms | 1.453ms | 1.833ms | 3.43% | <5% | PASS |
| 确定性 RAG | 120 | 47.894ms | 56.919ms | 390.839ms | 48.312ms | 51.836ms | 54.777ms | -8.93% | <2% | PASS |
| `/metrics` | 600 | - | - | - | - | 1.924ms | - | - | <100ms | PASS |

| 统计项 | 普通 API | 确定性 RAG |
| --- | ---: | ---: |
| P95 开销 95% CI | `[-3.38%, 9.70%]` | `[-79.21%, 6.52%]` |
| 误差半宽 | `6.54%` | `42.87%` |

正式门禁按计划规定的 P95 点估计判定，三项均通过。bootstrap 区间仍较宽，因此这些数据不代表生产规模或高并发性能承诺。

## 运行态指标

| 项目 | 观测值 | 结果 |
| --- | --- | --- |
| HTTP 端点 | `6/6` 返回 200 | PASS |
| Prometheus targets | API、Worker、Collector 共 `3/3` UP | PASS |
| Grafana dashboards | `5` 个 | PASS |
| Grafana alert rules | `10` 条 | PASS |
| 最终活跃告警 | `0` | PASS |
| Collector exporter queue capacity | `1000` spans | PASS |
| Collector exporter queue size（观测时） | `0` spans | PASS |

验证的 6 个端点：API liveness、API readiness、API metrics、Prometheus readiness、Grafana health、Jaeger UI。

## 安全告警闭环

| 信号 | 注入事件数 | Grafana 状态变化 | 结果 |
| --- | ---: | --- | --- |
| UNKNOWN WRITE | 1 | `Normal -> Firing -> Normal` | PASS |
| 安全边界违规 | 2 | `Normal -> Firing -> Normal` | PASS |

安全 Counter 的固定低基数 series 已在进程启动时初始化为 0，避免首个事件因缺少事件前样本而被 `increase(...[5m])` 漏报。标准 Worker 恢复后 target 为 UP。

## 业务与链路证据

| 项目 | 观测值 | 结果 |
| --- | --- | --- |
| 隔离文档摄取 | `succeeded` | PASS |
| 摄取任务计数增量 | `1` | PASS |
| 摄取阶段 histogram count 增量 | `5` | PASS |
| 投影收敛 | 已收敛 | PASS |
| controlled-action 批准前 adapter 调用 | `0` | PASS |
| controlled-action 批准后 adapter 调用 | `1` | PASS |
| duplicate/unapproved/unknown replay | 均为 `0` | PASS |
| 五类隐私扫描 | 均为 `0` | PASS |
| `task.run` spans | `4` | PASS |
| `task.run` producer Links | `4` | PASS |
| `task.run` CHILD_OF | `0` | PASS |
| dangling child references | `0` | PASS |
| task run error spans | `0` | PASS |

## 降级与 Provider 验证

| 场景 | 观测结果 | 结果 |
| --- | --- | --- |
| Jaeger 不可用 | Collector 记录导出重试，20/20 业务请求返回 200 | PASS |
| 完整 observability profile 停止 | 核心 API 5/5 请求返回 200 | PASS |
| Metrics 关闭 | 业务路由正常，`/metrics` 返回 404 | PASS |
| OTLP endpoint 不可用 | 业务请求正常 | PASS |
| 真实 DeepSeek 合成调用 | `answer`，仅引用预期合成 chunk | PASS |
| DeepSeek 成功 Metric | `1` | PASS |
| DeepSeek usage 信号 | 存在 token 或 usage-missing 指标 | PASS |

真实 Provider 冒烟只使用合成问题和合成上下文，不包含业务正文、用户标识或其他生产数据。

## 清理状态

- M6 临时容器：`0`。
- pytest M6 临时目录：`0`。
- 隔离 PostgreSQL 数据库、Qdrant collection、对象目录和 tenant：已清理。
- 标准 API、Worker 和 observability profile：运行中。
- `M2_PLAN.md` 与 `docs/observability-plan.md`：未修改。
