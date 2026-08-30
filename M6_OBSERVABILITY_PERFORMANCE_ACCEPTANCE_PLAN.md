# M6 可观测性性能验收落地计划

## 1. 文档定位

| 字段 | 值 |
| --- | --- |
| 状态 | 待实施 |
| 制定日期 | 2026-08-30 |
| 当前分支 | `main` |
| 当前代码 revision | `966bcfda7b4d55498f1252351af5fe0cf6b41756`（短 revision：`966bcfd`） |
| 历史性能快照 | `docs/acceptance/m6-observability-performance-v0.1.0-20260830-081728.md` |
| 历史结论 | **FAIL**，RAG hierarchical 95% CI 上界 `2.088%`，门限 `<2%` |

本文是下一轮 M6 可观测性性能验收的预注册实施计划，不是验收结果。所有统计协议、门限和执行纪律必须在采集新性能数据前冻结并提交；后续不得根据结果临时改变口径。

历史 FAIL 快照必须原样保留。新协议产生的是独立验收证据，不覆盖、不改写旧结果，也不把新样本追加到旧三轮数据中。

## 2. 背景与问题

revision `966bcfd` 已完成全量测试和三轮性能 benchmark：

| 项目 | 结果 |
| --- | --- |
| 全量 pytest | `483 passed, 42 skipped in 250.24s` |
| Document API CI 上界 | `4.987% < 5%`，PASS |
| 确定性 RAG CI 上界 | `2.088% > 2%`，FAIL |
| `/health/live` On P95 | `1.406ms < 5ms`，PASS |
| `/metrics` P95 | `2.054ms < 100ms`，PASS |
| 总结论 | **FAIL** |

只读审计未发现以下实现存在方向性错误：

- P95 线性插值公式。
- Off/On 开销方向。
- AB/BA 成对交替与 pair 保存。
- 先重采样轮次、再重采样 AB/BA strata 的 hierarchical bootstrap。
- 使用 CI 上界作为正式判定依据。
- API、RAG、Health、Metrics 四项逻辑与的最终聚合。

当前协议的主要稳定性限制是顶层只有三个独立轮次。三次重采样全部选中最高开销轮次的概率为 `(1/3)^3 = 1/27 ≈ 3.70%`，高于双侧 95% 区间的 `2.5%` 上尾，因此单轮波动足以显著控制 CI 上界。每轮增加 pair 数不能替代更多独立轮次。

此外，1000 次 bootstrap 的 97.5 分位主要由最高约 25 个估计值决定。固定随机种子保证可复现，但不能消除有限重采样次数的数值误差。

## 3. 目标与非目标

### 3.1 目标

- 保持每臂总正式样本量 1800 不变，将 `3×600` 重分配为 `10×180`。
- 将固定随机种子的 hierarchical bootstrap 从 1000 次提升至 10000 次。
- 保持门限、工作负载、AB/BA、GC 和异常值规则不变。
- 保存可独立审计的机器报告及原始 pair 耗时。
- 在固定 revision 上只运行一次正式 benchmark。
- 无论 PASS 或 FAIL，都生成并保留新的时间戳验收快照。

### 3.2 非目标

- 不放宽 `<5%` 或 `<2%` 门限。
- 不删除、裁剪或 winsorize 异常值。
- 不修改 deterministic RAG 的 `time.sleep(0.02)` 来稀释相对开销。
- 不降低 Trace 采样率来制造 PASS。
- 不修改 `src/rag_kb/` 下的生产业务或观测代码。
- 不新增统计库或其他第三方依赖。
- 不通过多次运行后挑选有利结果。
- 不把本地 instrumentation benchmark 表述为完整生产 OTLP/exporter 性能承诺。

## 4. 冻结的正式验收协议

### 4.1 比较口径

正式门禁继续比较完整观测层的关闭与开启，不切换为只比较 Trace 开关：

| Arm | 定义 |
| --- | --- |
| Off | 不激活观测上下文，Trace 关闭，Metrics 关闭 |
| On | 激活观测上下文，Trace 开启，Metrics 开启，采样率 `1.0` |

OTLP endpoint 保持为空，不把网络、Collector 或 exporter 队列耗时混入本地 instrumentation benchmark。

生产配置默认可在 Trace 关闭时继续开启 Metrics，因此该正式口径比单纯切换生产 Trace 配置更严格。生产等价开关比较只允许作为后续析因实验，不替代本轮正式门禁。

### 4.2 样本协议

| 参数 | 冻结值 |
| --- | ---: |
| 独立轮次 | 10 |
| API 每轮每臂正式样本 | 180 |
| RAG 每轮每臂正式样本 | 180 |
| API 每臂总正式样本 | 1800 |
| RAG 每臂总正式样本 | 1800 |
| 每轮每臂预热 | 40 |
| 并发 | 1 |
| 顺序 | AB/BA 成对交替 |
| 正式采样 GC | 关闭，结束后精确恢复原状态 |

每轮使用新的临时 SQLite database。相同轮次内 Off/On 继续共享业务 service 和 database，避免依赖构造差异污染测量。

### 4.3 统计协议

| 参数 | 冻结值 |
| --- | --- |
| P95 | 在排序样本上按 `(n-1)×q` 线性插值 |
| 相对开销 | `(enabled_p95 / disabled_p95 - 1) × 100` |
| Bootstrap | hierarchical bootstrap |
| 外层 | 有放回重采样轮次 |
| 内层 | 在所选轮次内分别重采样 AB 与 BA strata |
| 重采样次数 | 10000 |
| 随机种子 | `20260830` |
| 置信区间 | percentile 95% interval |
| 异常值 | 全部保留 |
| P99 | 仅作诊断，不参与正式判定 |

### 4.4 门禁

| 门禁 | 正式判定 |
| --- | --- |
| Document API | hierarchical P95 overhead 95% CI 上界 `<5%` |
| 确定性 RAG | hierarchical P95 overhead 95% CI 上界 `<2%` |
| `/health/live` | 开启观测后的绝对 P95 `<5ms` |
| `/metrics` | P95 `<100ms` |

四项必须全部通过。比较运算保持严格小于，等于门限也判定为 FAIL。

## 5. 文件范围

### 5.1 计划修改或新增

| 文件 | 变更 |
| --- | --- |
| `scripts/benchmark_observability.py` | 实施 `10×180`、10000 次 bootstrap、协议元数据和机器输出 |
| `tests/test_benchmark_observability.py` | 更新默认参数并增强协议、重采样和输出测试 |
| `M6_OBSERVABILITY_PERFORMANCE_ACCEPTANCE_PLAN.md` | 本预注册计划 |
| `docs/acceptance/m6-observability-performance-v0.1.0-<timestamp>.md` | 正式运行后新增结果快照 |

### 5.2 只保留、不覆盖

- `docs/acceptance/m6-observability-performance-v0.1.0-20260830-081728.md`
- `docs/acceptance/m6-observability-v0.1.0-20260830.md`

### 5.3 禁止修改或提交

- `M2_PLAN.md`
- `docs/observability-plan.md`

### 5.4 禁止扩大范围

- 不修改 `src/rag_kb/`。
- 不修改依赖声明或锁文件。
- 不做无关重构、格式化或命名调整。

## 6. 实施阶段

### 阶段 0：保存旧失败证据

目标：先把 revision `966bcfd` 的历史 FAIL 固化为不可变证据。

操作：

1. 复核旧快照记录的 revision、命令、全量测试结果、四项门禁和最终 FAIL。
2. 确认旧快照没有把自身错误描述为 `966bcfd` 的组成部分。
3. 仅暂存旧快照，不使用 `git add .`。
4. 单独提交旧快照，建议提交信息：`docs: 记录 M6 性能验收失败快照`。

完成标准：旧 FAIL 快照进入 Git 历史，内容未被改写。

### 阶段 1：提交预注册协议

目标：在采集任何新性能数据前固定规则。

操作：

1. 提交本文档。
2. 提交必须发生在正式 benchmark 之前。
3. 后续报告必须引用包含本文档的准确 revision。

完成标准：Git 历史能够证明协议先于新结果存在。

### 阶段 2：修改 benchmark

在 `scripts/benchmark_observability.py` 中做最小改动。

#### 2.1 协议常量

增加事实常量，避免实现和报告字符串不一致：

```python
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_REPETITIONS = 10_000
```

要求：

- `bootstrap_p95_overhead()` 使用上述常量作为默认协议。
- `_summary()` 显式记录实际重采样次数和随机种子。
- 不改变 percentile 与 overhead 公式。

#### 2.2 默认参数

将命令行默认值调整为：

```text
--rounds 10
--api-samples 180
--rag-samples 180
--warmup 40
```

命令行参数继续保留，便于专项诊断使用；只有上述默认值代表正式协议。

#### 2.3 报告元数据

`report["method"]` 至少明确记录：

```text
comparison = full_observability_off_vs_on
order = paired alternating AB/BA
rounds = 10
samples_per_arm_per_round = 180
samples_per_arm_total = 1800
warmup_per_arm = 40
bootstrap_seed = 20260830
bootstrap_repetitions = 10000
outlier_policy = retained
```

每个 summary 继续记录每轮与汇总的 P50、P95、P99、点估计、CI、CI 半宽、门限、判定依据和结果。

#### 2.4 可审计机器输出

新增可选参数：

```text
--output data/local/benchmarks/<filename>.json
```

行为：

- 未传 `--output` 时保留 stdout JSON 行为。
- 传入 `--output` 时写出完整 JSON，并在 stdout 输出简短状态与路径。
- 自动创建输出文件的父目录。
- 写文件发生在所有计时结束后，不进入测量区间。
- 输出包含环境、协议、汇总、每轮结果和原始 pair。
- 每条原始 pair 只记录轮次、pair 序号、AB/BA 顺序、Off 秒数和 On 秒数。
- 不记录正文、Prompt、Token、身份、数据库路径或其他敏感内容。

原始 pair 是独立复算 CI、检查顺序效应和审计异常值未被删除的必要证据。正式机器报告放入已忽略的 `data/local/benchmarks/`，Markdown 快照记录文件路径和 SHA-256。

#### 2.5 明确不改的代码

- 不修改 `_Model.answer()` 的 20ms sleep。
- 不修改 `_query_service()` 的业务流程。
- 不修改 `_rag_round()` 的正式 Off/On 定义。
- 不修改 Document API 路由与依赖共享方式。
- 不修改观测 Span、Metric、标签或采样率。

### 阶段 3：修改专项测试

在 `tests/test_benchmark_observability.py` 中完成以下验证：

1. 默认参数断言更新为 `(10, 180, 180, 40)`。
2. 默认协议测试名称同步更新，不再描述“三轮和 600 样本”。
3. 验证正式 summary 使用 10000 次 bootstrap 和种子 `20260830`。
4. 保留 P95 插值与 overhead 公式测试。
5. 保留 AB/BA 顺序和完整 pair 测试。
6. 保留 GC 初始开启、初始关闭和异常恢复测试。
7. 保留 CI 上界作为唯一正式相对开销判定依据的测试。
8. 保留 RAG 同轮复用 service/database 的测试。
9. 增加非退化固定数据测试，验证不同轮次、AB strata 和 BA strata 均参与重采样。
10. 增加机器输出测试，验证协议参数、轮次数、样本数、原始 pair 和结果状态。
11. 增加输出父目录不存在时可创建的测试。
12. 不引入新测试框架、fixture 层或统计依赖。

### 阶段 4：代码验证与固定 revision

先运行不产生正式性能结论的验证：

```powershell
& "D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe" -m pytest tests/test_benchmark_observability.py -q
& "D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe" -m compileall -q scripts/benchmark_observability.py tests/test_benchmark_observability.py
git diff --check
```

检查内容：

- 专项测试全部通过。
- `compileall` 无错误。
- 无 whitespace error。
- `git diff` 只包含本计划允许的文件。
- 没有新增临时文件或重复报告。
- `M2_PLAN.md` 和 `docs/observability-plan.md` 未被暂存。

随后只暂存脚本、专项测试和本计划，建议提交信息：

```text
test: 稳定 M6 性能验收统计协议
```

提交后记录完整 revision。正式全量测试与 benchmark 必须基于该 revision，不允许在存在相关未提交代码改动时运行。

## 7. 正式验收执行

### 7.1 执行前检查

1. 确认当前 HEAD 是协议实现 revision。
2. 确认 `scripts/benchmark_observability.py` 和 `tests/test_benchmark_observability.py` 工作树干净。
3. 确认没有 pytest、Docker build、其他 benchmark 或高负载任务并行运行。
4. 记录 Python、Windows、CPU 和 Git revision。
5. 不复用旧三轮原始数据。

### 7.2 全量测试

执行：

```powershell
& "D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe" -m pytest -q
```

成功条件：

- `0 failed`。
- skip 数量没有无法解释的变化。
- 记录 passed、skipped、failed 和总耗时。

如果全量测试失败，停止正式 benchmark，先修复本次改动引入的问题并产生新的固定 revision。不得在测试失败的 revision 上生成 PASS 验收报告。

### 7.3 一次性正式 benchmark

使用独立 Python 进程运行一次：

```powershell
& "D:\DevelopmentTools\Miniconda3\envs\rag-kb\python.exe" scripts/benchmark_observability.py `
  --rounds 10 `
  --api-samples 180 `
  --rag-samples 180 `
  --warmup 40 `
  --output "data/local/benchmarks/m6-observability-performance-v2-<timestamp>.json"
```

即使显式参数与默认值相同，也保留显式参数以便命令证据自描述。

执行纪律：

- 不查看中间轮次后提前停止。
- 不因某一轮异常而删除或重跑该轮。
- 不因最终 FAIL 而追加轮次。
- 不更换随机种子后比较结果。
- 不启动第二次正式运行挑选更有利结果。
- 进程异常退出时，本次结果标记为无效运行；修复明确的执行故障后必须从第 1 轮完整重跑，并在报告中说明无效运行原因。

### 7.4 机器证据哈希

正式进程完成后计算 SHA-256：

```powershell
Get-FileHash -Algorithm SHA256 "data/local/benchmarks/m6-observability-performance-v2-<timestamp>.json"
```

记录机器文件路径、字节数和 SHA-256。不得在计算哈希后修改机器文件。

## 8. 验收快照

新报告命名：

```text
docs/acceptance/m6-observability-performance-v0.1.0-<timestamp>.md
```

报告至少包含：

1. 项目版本、分支、完整测试 revision 和记录时间。
2. Python、操作系统、CPU 逻辑核心数。
3. 本预注册计划的路径与 revision。
4. 全量 pytest 命令、结果、退出码和耗时。
5. 正式 benchmark 完整命令、退出码和一次性执行声明。
6. 冻结的比较口径、样本协议、bootstrap 次数和种子。
7. Document API 十轮明细、汇总结果和 CI。
8. 确定性 RAG 十轮明细、汇总结果和 CI。
9. Health 与 Metrics 结果。
10. 机器 JSON 路径、字节数和 SHA-256。
11. 四项正式门禁矩阵与总判定。
12. 与旧 `966bcfd` FAIL 快照的关系。
13. 报告自身是否包含在被测试 revision 中。
14. 未修改门限、统计公式、异常值规则或生产代码的声明。

无论最终 PASS 或 FAIL，报告都必须保留。失败报告不得删除，成功报告不得覆盖失败报告。

## 9. 判定与发布

### 9.1 PASS

仅当四项门禁全部通过时，状态可以写为：

```text
功能与回归验收：PASS
可复现性验收：PASS
严格性能验收：PASS
观测层最终验收：PASS
```

随后单独提交新验收快照，建议提交信息：

```text
docs: 记录 M6 性能最终验收结果
```

提交后，快照中的“报告自身未包含在测试 revision”仍应保留，以准确区分被测试代码和后写入的证据文档。

### 9.2 FAIL

任意一项失败时，状态必须写为：

```text
功能与回归验收：按实际测试结果记录
可复现性验收：PASS
严格性能验收：FAIL
观测层最终验收：FAIL
```

失败后仍提交验收快照。不得重复正式 benchmark 直到出现 PASS。

## 10. 失败后的析因计划

只有新协议仍然 FAIL 时，才进入独立的非门禁诊断。诊断数据不得替换正式验收结果。

| 实验 | 两臂 | 目的 |
| --- | --- | --- |
| Off/Off A/A | 完全相同的关闭路径 | 测量环境与工作负载噪声底板 |
| On/On A/A | 完全相同的开启路径 | 测量开启路径自身波动 |
| Context-only | 无上下文 vs disabled Observability activate | 测量 ContextVar 激活固定成本 |
| Trace-only | disabled Trace vs enabled Trace，Metrics 都关闭 | 测量 Span 与属性成本 |
| Metrics-only | Metrics 关闭 vs 开启，Trace 都关闭 | 测量 Prometheus 更新成本 |
| Trace+Metrics | 全关闭 vs 全开启 | 对照正式门禁 |
| 生产开关 | Metrics 均开启，只切换 Trace | 量化生产默认配置的实际增量 |

每个诊断实验必须预先固定轮次、样本、顺序和解释规则，并完整报告全部结果。诊断不能用于删除正式样本或事后改门限。

如果证据显示真实观测开销稳定超过门限，才允许评估以下最小优化：

- 将经过安全白名单过滤的 Span 属性批量写入，减少重复锁操作。
- 缓存固定、低基数且已受枚举约束的 Prometheus label child。

任何优化都必须先有析因证据，且不得删除安全校验、Span、Metric、标签边界或错误处理。

## 11. Git 与资产保护

- 禁止使用 `git add .`。
- 每次只暂存计划明确列出的文件。
- 提交前检查 `git status`、`git diff` 和 staged diff。
- 不 amend，不 force push，不执行破坏性 Git 命令。
- 不修改、暂存、删除 `M2_PLAN.md`。
- 不修改、暂存、删除 `docs/observability-plan.md`。
- 不覆盖任何既有验收快照。
- `data/local/benchmarks/` 中正式机器证据在报告和哈希完成前不得清理。
- 任务完成后删除仅用于调试、且未被报告引用的临时文件；正式机器证据和验收快照不是冗余文件。

## 12. 完成检查表

- [ ] 历史 `966bcfd` FAIL 快照已独立提交。
- [ ] 本预注册计划已在新数据采集前提交。
- [ ] 默认协议已改为 `10×180`。
- [ ] Bootstrap 已改为固定种子、10000 次。
- [ ] 门限、工作负载、AB/BA、GC 和异常值规则未改变。
- [ ] 专项测试已覆盖默认参数、分层重采样、GC、CI 判定和机器输出。
- [ ] 专项测试、`compileall` 和 `git diff --check` 已通过。
- [ ] 协议代码已提交并记录完整 revision。
- [ ] 全量 pytest 在固定 revision 上通过。
- [ ] 正式 benchmark 只执行一次且未提前停止或追加轮次。
- [ ] 原始机器报告已保存并计算 SHA-256。
- [ ] 新时间戳验收快照已记录全部四项门禁。
- [ ] 无论 PASS 或 FAIL，新快照均已保留。
- [ ] 只有四项全部严格通过时才声明观测层最终验收通过。
- [ ] `M2_PLAN.md` 和 `docs/observability-plan.md` 未被修改或提交。
- [ ] 没有修改生产代码、依赖或无关文件。

## 13. 最终原则

本轮工作的目标不是获得一次 PASS，而是获得一份协议预先固定、结果不可挑选、原始数据可审计、revision 可复现的性能证据。旧 FAIL 和新结果共同构成完整历史，任何结论都必须以当次固定 revision、机器证据和严格门禁为准。
