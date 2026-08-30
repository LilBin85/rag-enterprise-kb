# M6 可观测性最终性能验收

## 快照信息

| 字段 | 值 |
| --- | --- |
| 项目 | `rag-kb` |
| 项目版本 | `0.1.0` |
| 记录时间 | `2026-08-30 11:35:31 +08:00` |
| 分支 | `main` |
| 测试 revision | `2d26066e0ff416d5d063ff3d578fe7a50e20d641`（短 revision：`2d26066`） |
| 预注册计划 | `M6_OBSERVABILITY_PERFORMANCE_ACCEPTANCE_PLAN.md` |
| 预注册 revision | `562d28eba213100f58439223cfdd102c1213c0f8`（短 revision：`562d28e`） |
| Python | `3.13.14` |
| 操作系统 | Windows 11 AMD64 |
| CPU 逻辑核心数 | `20` |
| 最终结论 | **PASS** |

预注册计划已在新数据采集前以 revision `562d28e` 提交。协议实现和专项测试随后固定为 revision `2d26066`，全量测试与唯一一次正式 benchmark 均基于该 revision。本文档在正式运行后写入，因此本文档自身不包含在被测试 revision 中。

## 验收状态

```text
功能与回归验收：PASS
可复现性验收：PASS
严格性能验收：PASS
观测层最终验收：PASS
```

## 冻结协议

| 参数 | 正式值 |
| --- | --- |
| 比较口径 | `full_observability_off_vs_on` |
| Off | 不激活观测上下文，Trace 关闭，Metrics 关闭 |
| On | 激活观测上下文，Trace 开启，Metrics 开启，采样率 `1.0` |
| 独立轮次 | `10` |
| API / RAG 每轮每臂样本 | `180` |
| API / RAG 每臂总样本 | `1800` |
| 每轮每臂预热 | `40` |
| 并发 | `1` |
| 顺序 | paired alternating AB/BA |
| 正式采样 GC | 关闭，结束后恢复原状态 |
| P95 | 排序样本上按 `(n-1)×q` 线性插值 |
| 相对开销 | `(enabled_p95 / disabled_p95 - 1) × 100` |
| Bootstrap | hierarchical；先重采样轮次，再分别重采样 AB/BA strata |
| Bootstrap 次数 / 种子 | `10000` / `20260830` |
| 置信区间 | percentile 95% interval |
| 异常值 | 全部保留 |
| P99 | 仅诊断，不参与正式判定 |

OTLP endpoint 保持为空。本结果衡量本地 instrumentation 开销，不表述为完整生产 OTLP、Collector 或 exporter 性能承诺。

## 执行前检查

- HEAD 为固定协议实现 revision `2d26066e0ff416d5d063ff3d578fe7a50e20d641`。
- `scripts/benchmark_observability.py`、`tests/test_benchmark_observability.py` 和预注册计划工作树干净。
- 未发现并行的 pytest、Docker build、其他 benchmark 或 Python 工作负载。
- benchmark 前三次 CPU 总利用率采样为 `13.00%`、`6.25%`、`13.32%`。
- 未复用或追加旧三轮原始数据。

## 自动化验证

### 专项验证

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| benchmark 专项测试 | `D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe -m pytest tests/test_benchmark_observability.py -q` | `15 passed in 6.28s`，退出码 `0` |
| 编译检查 | `D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe -m compileall -q scripts/benchmark_observability.py tests/test_benchmark_observability.py` | PASS，退出码 `0` |
| 空白检查 | `git diff --check` | PASS，退出码 `0` |

首次在文件系统沙箱内执行专项测试时，pytest 无权访问用户临时目录，产生 2 个 fixture setup error；这不是代码失败。允许访问 pytest 临时目录后使用同一命令重跑，15 项全部通过。该过程未采集正式性能数据。

### 全量测试

命令：

```powershell
& "D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe" -m pytest -q
```

结果：`485 passed, 43 skipped, 1 warning in 203.27s`，退出码 `0`，命令墙钟 `210.5s`。

相对旧快照，benchmark 专项测试从 12 项增加到 15 项。额外的 1 个 skip 来自本机 `localhost:6333` Qdrant 不可用；`tests/test_stage3_qdrant_component.py` 的既有 fixture 按预定行为调用 `pytest.skip("qdrant_unavailable")`。全量测试没有失败。

## 一次性正式 benchmark

完整命令：

```powershell
& "D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe" scripts/benchmark_observability.py `
  --rounds 10 `
  --api-samples 180 `
  --rag-samples 180 `
  --warmup 40 `
  --output "data/local/benchmarks/m6-observability-performance-v2-20260830-112845.json"
```

结果：`PASS`，退出码 `0`，命令墙钟 `298s`。

本命令只执行一次；未查看中间轮次，未提前停止，未删除或重跑任何轮次，未追加轮次，未更换随机种子，也未启动第二次正式运行。

## Document API

端点：`GET /v1/documents/{document_id}`。单位：ms。

| 轮次 | Off P50 | Off P95 | Off P99 | On P50 | On P95 | On P99 | P95 开销 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.299 | 5.082 | 5.485 | 4.369 | 5.052 | 5.822 | -0.598% |
| 2 | 4.161 | 5.140 | 5.521 | 4.129 | 4.902 | 5.764 | -4.635% |
| 3 | 4.238 | 5.173 | 5.625 | 4.205 | 5.184 | 5.597 | 0.214% |
| 4 | 3.893 | 4.707 | 5.047 | 3.964 | 4.718 | 5.114 | 0.239% |
| 5 | 3.997 | 4.740 | 5.140 | 4.137 | 5.027 | 5.362 | 6.041% |
| 6 | 4.289 | 5.137 | 5.352 | 4.376 | 5.231 | 5.803 | 1.813% |
| 7 | 4.053 | 4.814 | 5.051 | 4.109 | 5.023 | 5.324 | 4.339% |
| 8 | 3.958 | 4.613 | 5.082 | 3.994 | 4.759 | 4.891 | 3.170% |
| 9 | 4.022 | 4.863 | 5.229 | 4.137 | 4.924 | 5.276 | 1.245% |
| 10 | 4.047 | 4.939 | 5.327 | 4.087 | 5.037 | 5.425 | 1.994% |
| 汇总 | 4.092 | 4.963 | 5.343 | 4.138 | 5.036 | 5.531 | 1.464% |

| 正式统计项 | 结果 |
| --- | ---: |
| Hierarchical bootstrap 95% CI | `[-1.180%, 3.588%]` |
| CI 半宽 | `2.384%` |
| CI 上界门限 | `<5%` |
| 判定依据 | `hierarchical_p95_overhead_95ci_upper` |
| 正式结论 | **PASS** |

## 确定性 RAG

单位：ms。

| 轮次 | Off P50 | Off P95 | Off P99 | On P50 | On P95 | On P99 | P95 开销 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.167 | 50.551 | 438.687 | 48.178 | 51.247 | 68.852 | 1.376% |
| 2 | 48.328 | 53.552 | 163.710 | 48.450 | 51.195 | 60.686 | -4.401% |
| 3 | 48.680 | 54.754 | 233.617 | 48.776 | 51.432 | 54.781 | -6.068% |
| 4 | 48.153 | 50.451 | 51.832 | 48.353 | 50.710 | 191.296 | 0.512% |
| 5 | 48.409 | 52.172 | 55.028 | 48.583 | 53.092 | 238.662 | 1.763% |
| 6 | 49.001 | 52.997 | 94.022 | 49.198 | 53.210 | 89.557 | 0.401% |
| 7 | 48.234 | 50.232 | 84.549 | 48.373 | 50.603 | 57.785 | 0.739% |
| 8 | 48.635 | 51.746 | 56.234 | 48.937 | 52.536 | 238.361 | 1.528% |
| 9 | 48.703 | 51.363 | 57.147 | 48.834 | 51.637 | 183.415 | 0.534% |
| 10 | 48.108 | 51.031 | 191.875 | 48.300 | 50.575 | 52.556 | -0.893% |
| 汇总 | 48.422 | 51.755 | 150.399 | 48.613 | 51.802 | 114.857 | 0.090% |

| 正式统计项 | 结果 |
| --- | ---: |
| Hierarchical bootstrap 95% CI | `[-1.769%, 1.991%]` |
| CI 半宽 | `1.880%` |
| CI 上界门限 | `<2%` |
| 判定依据 | `hierarchical_p95_overhead_95ci_upper` |
| 正式结论 | **PASS** |

RAG P99 继续仅作为诊断信息。全部原始样本均被保留，没有裁剪、删除或 winsorize。

## Health 与 Metrics

| 场景 | Off P50 | Off P95 | Off P99 | On P50 | On P95 | On P99 | 正式门限 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/health/live` | 0.988ms | 1.433ms | 1.662ms | 1.049ms | 1.505ms | 1.721ms | On P95 `<5ms` | PASS |

`/health/live` P95 absolute delta 为 `0.072ms`。该端点未使用相对 overhead 作为正式门禁。

| 场景 | 样本数 | P95 | 正式门限 | 结果 |
| --- | ---: | ---: | ---: | --- |
| `/metrics` | 1800 | 2.377ms | `<100ms` | PASS |

## 机器证据

| 字段 | 值 |
| --- | --- |
| 文件 | `data/local/benchmarks/m6-observability-performance-v2-20260830-112845.json` |
| 字节数 | `626932` |
| SHA-256 | `F79AEB2CEE5E321F1574F00E5CEF8A62763A40570AD75B0683F93917B5DA6961` |
| API 原始 pair | `1800` |
| RAG 原始 pair | `1800` |
| Health 原始 pair | `1800` |

哈希在正式进程结束后立即计算；此后未修改机器文件。每条原始 pair 只包含轮次、pair 序号、AB/BA 顺序、Off 秒数和 On 秒数，不包含正文、Prompt、Token、身份、数据库路径或其他敏感内容。

## 最终门禁

| 门禁 | 观测值 | 门限 | 结果 |
| --- | ---: | ---: | --- |
| Document API hierarchical CI upper | 3.588% | `<5%` | PASS |
| RAG hierarchical CI upper | 1.991% | `<2%` | PASS |
| `/health/live` enabled P95 | 1.505ms | `<5ms` | PASS |
| `/metrics` P95 | 2.377ms | `<100ms` | PASS |

四项正式门禁全部严格通过，因此本次观测层最终验收为 **PASS**。

## 与历史 FAIL 的关系

旧快照 `docs/acceptance/m6-observability-performance-v0.1.0-20260830-081728.md` 记录 revision `966bcfd` 的独立三轮结果：RAG hierarchical CI 上界 `2.088%` 未满足 `<2%`，最终为 **FAIL**。旧快照已在 commit `df3616b` 中固化，内容未被覆盖或改写。

本报告使用预先提交的 `10×180`、10000 次 bootstrap 新协议生成独立证据；没有把新样本追加到旧数据，也没有用本结果抹去历史 FAIL。

## 变更与边界声明

- 未放宽 Document API `<5%` 或 RAG `<2%` 门限。
- 除预注册的轮次重分配和 bootstrap 次数提升外，未修改 P95、overhead 公式、层级重采样结构、AB/BA、GC 或异常值规则。
- 未修改 deterministic RAG 的 20ms sleep、Trace 采样率或 Off/On 工作负载。
- 未修改 `src/rag_kb/` 下生产业务或观测代码。
- 未修改依赖声明或锁文件。
- 未修改、暂存或提交 `M2_PLAN.md` 和 `docs/observability-plan.md`。
- 本报告自身未包含在被测试 revision `2d26066`；它是正式运行后写入的独立证据文档。
