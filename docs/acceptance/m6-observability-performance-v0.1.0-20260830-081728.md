# M6 可观测性最终性能验收

## 快照信息

| 字段 | 值 |
| --- | --- |
| 项目 | `rag-kb` |
| 项目版本 | `0.1.0` |
| 记录时间 | `2026-08-30 08:43:45 +08:00` |
| 分支 | `main` |
| 测试 revision | `966bcfda7b4d55498f1252351af5fe0cf6b41756`（短 revision：`966bcfd`） |
| 验收对象 | revision `966bcfd` 对应的代码 |
| 验收脚本 | `scripts/benchmark_observability.py` |
| Python | `3.13.14` |
| 操作系统 | Windows 11 AMD64 |
| CPU 逻辑核心数 | `20` |
| 最终结论 | **FAIL** |

本文件记录 revision `966bcfd` 对应代码的最终可观测性性能门禁结果。benchmark 与专项测试改动已经包含在该 revision 中；本验收文档自身尚未包含在该 revision 中。本文件不覆盖此前的验收快照。

## 验收口径

- 使用 `GET /v1/documents/{document_id}` 作为普通业务 API 正式门禁。
- `/health/live` 仅作为固定成本诊断，按开启观测后的绝对 P95 判定，不使用相对百分比。
- benchmark 内部顺序运行 3 个独立轮次。
- 每轮 API 和 RAG 使用新的临时 SQLite database；同轮 off/on 共享业务 service 和 database。
- 每轮每臂预热 40 次，正式采集 600 个 pair，单并发，轮内按 AB/BA 交替。
- 相对开销按 `(enabled_p95 / disabled_p95 - 1) * 100` 计算。
- 95% CI 使用固定随机种子、1000 次 hierarchical bootstrap：先有放回重采样轮次，再在所选轮次内分别重采样 AB 与 BA 层。
- 不裁剪、删除或 winsorize 异常值；P99 仅作诊断，不参与正式门禁。
- OTLP endpoint 为空，不把 exporter 网络耗时混入本地 instrumentation benchmark。

## 自动化验证

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| 全量 pytest | `python -m pytest -q` | `483 passed, 42 skipped in 250.24s` |
| 默认三轮 benchmark | `python scripts/benchmark_observability.py` | **FAIL**，退出码 `1` |

所有 Python 命令均使用：

`D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe`

## Document API

端点：`GET /v1/documents/{document_id}`。每轮在计时区间外创建固定 tenant、principal 和 document；off/on app 共享 `DocumentService` 与 `DemoTokenIdentityProvider`。每次请求断言 HTTP 200，off/on JSON 响应一致，并确认开启观测后的 Metrics 包含动态路由模板、`GET` 和 `2xx` 标签。

单位：ms。

| 轮次 | Off P50 | Off P95 | Off P99 | On P50 | On P95 | On P99 | P95 开销 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.702 | 4.710 | 5.282 | 3.716 | 4.657 | 5.298 | -1.125% |
| 2 | 3.675 | 4.628 | 5.090 | 3.742 | 4.799 | 5.075 | 3.682% |
| 3 | 3.829 | 4.916 | 5.523 | 3.922 | 5.119 | 5.576 | 4.126% |
| 汇总 | 3.736 | 4.797 | 5.282 | 3.778 | 4.875 | 5.389 | 1.623% |

| 正式统计项 | 结果 |
| --- | ---: |
| Hierarchical bootstrap 95% CI | `[-1.116%, 4.987%]` |
| CI 半宽 | `3.051%` |
| CI 上界门限 | `<5%` |
| 判定依据 | `hierarchical_p95_overhead_95ci_upper` |
| 正式结论 | **PASS** |

## 确定性 RAG

单位：ms。

| 轮次 | Off P50 | Off P95 | Off P99 | On P50 | On P95 | On P99 | P95 开销 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.045 | 50.746 | 150.285 | 48.306 | 51.919 | 67.882 | 2.311% |
| 2 | 47.909 | 52.084 | 192.683 | 48.107 | 51.726 | 86.142 | -0.687% |
| 3 | 47.906 | 51.435 | 351.978 | 48.012 | 50.543 | 53.371 | -1.735% |
| 汇总 | 47.949 | 51.660 | 286.391 | 48.117 | 51.463 | 72.638 | -0.381% |

| 正式统计项 | 结果 |
| --- | ---: |
| Hierarchical bootstrap 95% CI | `[-1.955%, 2.088%]` |
| CI 半宽 | `2.021%` |
| CI 上界门限 | `<2%` |
| 判定依据 | `hierarchical_p95_overhead_95ci_upper` |
| 正式结论 | **FAIL** |

RAG P99 存在明显长尾，但按既定口径仅作为诊断信息，不参与正式判定，也未对样本进行裁剪。

## Health 与 Metrics

| 场景 | Off P50 | Off P95 | Off P99 | On P50 | On P95 | On P99 | 正式门限 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/health/live` | 0.943ms | 1.367ms | 1.594ms | 0.997ms | 1.406ms | 1.649ms | On P95 `<5ms` | PASS |

`/health/live` P95 absolute delta 为 `0.039ms`。该端点未计算或使用相对 overhead 作为正式门禁。

| 场景 | 样本数 | P95 | 正式门限 | 结果 |
| --- | ---: | ---: | ---: | --- |
| `/metrics` | 1800 | 2.054ms | `<100ms` | PASS |

## 最终判定

| 门禁 | 观测值 | 门限 | 结果 |
| --- | ---: | ---: | --- |
| Document API hierarchical CI upper | 4.987% | `<5%` | PASS |
| RAG hierarchical CI upper | 2.088% | `<2%` | **FAIL** |
| `/health/live` enabled P95 | 1.406ms | `<5ms` | PASS |
| `/metrics` P95 | 2.054ms | `<100ms` | PASS |

全量测试与 benchmark 内业务断言通过。四项正式性能门禁中三项通过；RAG hierarchical CI 上界 `2.088%` 未满足严格门限 `<2%`，因此本次最终结论为 **FAIL**。未修改统计规则、采样率或异常值处理方式来制造 PASS。

## 变更范围

- 性能基准实现：`scripts/benchmark_observability.py`。
- 性能基准专项测试：`tests/test_benchmark_observability.py`。
- 未修改 `src/rag_kb/` 下的生产业务代码。
- 未修改 `M2_PLAN.md`、`docs/observability-plan.md` 或既有验收快照。
- Benchmark 与专项测试已提交为 revision `966bcfd`。
- 本验收文档当前尚未提交。
