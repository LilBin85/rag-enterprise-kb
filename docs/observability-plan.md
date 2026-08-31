# 可观测性实施方案（标准基础栈）

状态：计划中

更新时间：2026-09-01

## 1. 目标与边界

本方案为 `rag-enterprise-kb` 增加可用于本地演示、联调和后续生产部署的三信号可观测性基础：指标（Metrics）、链路追踪（Traces）和结构化日志（Logs）。它服务于以下问题：

1. 一个 HTTP 请求、同步知识问答、异步 Agent 任务、审批恢复或摄取任务为什么慢、为什么失败。
2. 一个 Agent 任务实际经历了哪些模型调用、工具调用、checkpoint、审批和最终复核阶段。
3. Qdrant、PostgreSQL、模型、工具、投影和队列中的哪个边界造成错误或积压。
4. 发生 ACL、内容哈希、租约 fencing 或 WRITE 工具 `unknown` 等安全事件时，能否被及时发现和定位。

首期采用以下最小技术栈：

```text
API / task-worker / ingestion-worker / projection-worker
  |-- JSON 结构化日志 --> stdout
  |-- OTLP Trace ------> OpenTelemetry Collector --> Jaeger
  `-- /metrics --------> Prometheus -------------> Grafana
```

### 1.1 首期范围

| 信号 | 首期方案 | 目标 |
|---|---|---|
| Metrics | `prometheus-client`，API 暴露 `/metrics`，Worker 暴露独立内部端口 | SLO、告警、容量和趋势判断 |
| Traces | OpenTelemetry API/SDK，经 OTLP 导出到 Collector 和 Jaeger | 单次请求或任务的阶段耗时与失败定位 |
| Logs | Python 标准库 `logging` 的 JSON formatter，输出到 stdout | 低成本关联、稳定错误码和审计辅助 |
| Dashboard | Prometheus 数据源 + Grafana 预置 dashboard | 系统、RAG、Agent、摄取、安全四类视图 |
| Alert | Grafana Alerting 基于 Prometheus 规则 | 首期本地/演示和后续通知通道接入 |

### 1.2 明确不在首期范围内

| 不做的内容 | 原因 | 何时再评估 |
|---|---|---|
| Langfuse、LangSmith | 与 OTel Agent/LLM span 重叠，且默认输入输出追踪不适合企业知识正文 | 需要在线评测、Prompt 管理或人工反馈分析时 |
| Loki/Alloy | stdout JSON 已足以完成首期日志关联，先避免增加日志存储和保留治理 | 有多实例、跨容器检索日志或正式值班时 |
| Prompt、回答、文档正文采集 | 这些内容含企业敏感信息，不能进入 Trace、Metrics 或普通日志 | 有独立的合规存储、审批和最短保留方案时 |
| 自动化 SQL/HTTP 全量埋点 | 容易采集 SQL、动态 URL 或高噪声 span，首期业务边界 span 更有诊断价值 | 经过属性脱敏和采样验证后 |
| 模型成本金额 | 现有模型元数据和供应商计费口径不稳定，错误估价比不展示更误导 | 有版本化价格表与账单对账机制后 |
| 前端观测控制台 | 当前项目没有前端；Grafana 与 Jaeger 已满足工程演示和排障 | 新增产品管理台时 |

## 2. 当前基线

当前项目有业务状态、工具记录、审批记录、证据账本和健康检查，但它们不是完整运行期观测系统。

| 已有事实 | 位置 | 当前限制 |
|---|---|---|
| 同步问答异常日志 | `application/query_service.py` | 只有零散文本日志，无统一 `request_id`、时长和结构化字段 |
| Agent 任务、工具、审批和事件表 | `infrastructure/database/models.py` | `task_events.duration_ms`、`token_usage`、`tool_name` 未实际写入 |
| 三类 Worker 结果日志 | `worker.py` | 按固定 `worker_kind` 区分 task、ingestion、projection 的循环、维护和错误 |
| `/health/live`、`/health/ready` | `web/routers/health.py`、`web/dependencies.py` | 不检查 Worker 可用性、队列积压、对象存储和模型 |
| 请求 ID | `web/dependencies.py:resolve_auth_context()` | 未回写响应头，且允许 200 字符，但 `task_events.trace_id` 只有 64 字符 |
| PostgreSQL checkpointer 与租约 fencing | `agent/checkpointer.py`、`application/task_worker.py` | 可恢复执行，但不能在 Trace 中看到每次恢复、失租和 checkpoint 开销 |

### 2.1 先修复的关联 ID 问题

当前 HTTP `X-Request-ID` 允许 1 至 200 个字符，而 `task_events.trace_id` 和 `run_id` 为 `VARCHAR(64)`。创建任务时会将请求 ID 写入事件表，因此 65 至 200 字符的合法请求 ID 有可能导致 PostgreSQL 写入失败。

实施观测前必须完成以下 M0 修复：

1. `X-Request-ID` 最大长度改为 64，只接受 `[A-Za-z0-9._-]`，不符合时服务端生成新的 UUID hex。
2. 所有 HTTP 响应，包括 4xx、5xx 和验证失败，回写 `X-Request-ID`。
3. 现有 `task_events.trace_id` 重命名为 `correlation_id`，保留其历史语义，不能把现有 request/run ID 误称为 OpenTelemetry trace ID。
4. 新增 `otel_trace_id VARCHAR(32)` 和 `otel_span_id VARCHAR(16)`，仅保存有效 OTel 标识。

`request_id` 是应用关联 ID；`trace_id` 和 `span_id` 是 OpenTelemetry 标识；`run_id` 是 Worker 一次实际执行。三者不得混用。

## 3. 架构决策

### 3.1 信号所有权

| 问题 | 首选信号 | 不使用的信号 |
|---|---|---|
| 当前 API 错误率、P95、队列深度 | Metrics | 不从日志全文聚合 |
| 一次 Agent 任务为何失败 | Trace + 关联日志 | 不扫描数据库事件表重建调用树 |
| 某个稳定业务错误码的具体上下文 | JSON 日志 + `trace_id` | 不作为 Prometheus 高基数 label |
| 长期任务、审批和工具的业务审计 | 现有 PostgreSQL 业务表 | 不用 Jaeger 代替审计事实源 |
| RAG 质量回归 | 现有 benchmark/acceptance 数据 | 不把实时 Trace 当质量评测数据集 |

`task_events`、`tool_calls`、`approvals`、`evidence_records` 和 `ingestion_jobs` 继续承担业务事实与审计职责。它们不复制 OTel 的每一个 span，也不作为 Prometheus 时序数据库的替代品。

### 3.2 运行拓扑

首期在 `docker-compose.yml` 中增加 `observability` profile。默认 `docker compose up` 不增加 Collector、Jaeger、Prometheus 或 Grafana；演示和联调使用：

```powershell
docker compose --profile observability up --build -d
```

当前拓扑：

```text
                    +--------------------------+
                    |        Grafana           |
                    | dashboard + alert rules  |
                    +------------+-------------+
                                 ^
                                 | PromQL
                    +------------+-------------+
                    |       Prometheus          |
                    +-------+------------+------+
                            ^            ^
                            | scrape     | scrape
                    +-------+--+     +---------------------------+
                    | API :8000|     | task-worker      :9464    |
                    | /metrics |     | ingestion-worker :9465    |
                    +----------+     | projection-worker:9466    |
                                     | /metrics                  |
                                     +---------------------------+

                    +--------------------------+
                    | OpenTelemetry Collector  |
                    +------------+-------------+
                                 ^ OTLP/gRPC
                    +------------+-------------+
                    | API / three Worker SDKs   |
                    +------------+-------------+
                                 |
                    +------------+-------------+
                    |           Jaeger         |
                    +--------------------------+
```

API 的 `/metrics` 不应经公网暴露。当前 Compose 已将 API 映射到 `127.0.0.1`，生产反向代理也必须拒绝外部访问 `/metrics`，仅允许 Prometheus 网络访问。三个 Worker 的 `:9464`、`:9465`、`:9466` 仅通过 `expose` 暴露在 Compose 网络内，不映射宿主机端口。

当前 Compose 为单 API 进程和三个独立 Worker 进程。若未来增加多个 Uvicorn worker 或水平扩容同类 Worker，必须让 Prometheus 发现每个进程实例；不得让多个进程共享默认内存 registry，也不得把 API 与 Worker 写入同一 Prometheus multiprocess 目录。

聚合 `_run_recovery_window()` 已不存在：`task-worker` 独占 capability heartbeat/availability、task/tool/knowledge-query 超时、task retention 和任务队列快照；`ingestion-worker` 独占上传预留恢复和摄取队列快照；`projection-worker` 通过自己的 `run_once()` 独占投影审计、修复与重建。三类进程都写自身的 last-loop 和 error 指标。

### 3.3 配置与失效策略

新增以下 Settings，所有值均只从环境变量或 `.env` 读取：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `OBSERVABILITY_ENABLED` | `false` | 总开关；关闭时 tracer 使用 no-op provider，业务路径不依赖 exporter |
| `METRICS_ENABLED` | `true` | 是否暴露 Metrics；生产环境仍必须由网络层限制访问 |
| `OTEL_SERVICE_NAME` | API 为 `rag-kb-api`；Worker 为 `rag-kb-worker-task`、`rag-kb-worker-ingestion`、`rag-kb-worker-projection` | Trace 与日志资源名 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 空 | 空值时不初始化远程 OTLP exporter |
| `OTEL_TRACE_SAMPLE_RATIO` | `1.0` | 应用端 head sampling 比率；标准 profile 由 Collector 做 tail sampling |
| `LOG_LEVEL` | `INFO` | JSON 日志级别 |
| `LOG_FORMAT` | `json` | 首期固定 JSON，开发调试可后续增加 console 格式 |
| `WORKER_METRICS_PORT` | task `9464`、ingestion `9465`、projection `9466` | Compose 网络内的 Worker Metrics 端口 |

初始化、exporter 连接或 Collector 不可用时，只写一次安全启动告警，并继续提供业务能力。Exporter 不得阻塞 API 请求、Agent 领取、摄取、审批或数据库事务。

## 4. 数据分类与隐私契约

观测数据与业务数据的安全边界必须先于任何埋点实现确定。

### 4.1 允许进入各信号的数据

| 数据 | Metrics | Trace attribute | JSON 日志 |
|---|---|---|---|
| 路由模板、HTTP 方法、状态类别 | 允许 | 允许 | 允许 |
| 稳定 operation、阶段、模式、工具名 | 允许 | 允许 | 允许 |
| 时长、候选数量、chunk 数、token 数 | 允许 | 允许 | 允许 |
| 稳定错误类别 | 允许 | 允许 | 允许 |
| 原始稳定 error code | 禁止 | 允许 | 允许 |
| `request_id`、`trace_id`、`span_id`、`run_id` | 禁止 | 允许 | 允许 |
| `task_id`、`job_id`、`tool_call_id`、`approval_id` | 禁止 | 允许 | 允许 |
| 经 HMAC 或不可逆哈希处理的 tenant/user 标识 | 禁止 | 可选 | 可选 |

用户、租户和任务标识只用于受控 Trace/日志关联，不用于 Metrics label。首期通常不记录 `tenant_id` 和 `user_id`，确有审计需求时记录服务端加盐 HMAC 摘要，不使用普通 SHA-256。

### 4.2 永久禁止采集的内容

以下内容不得出现在 Metrics label、Trace attribute、Trace event、普通 JSON 日志、Jaeger 或 Grafana annotation：

1. `Authorization`、Bearer token、API key、密码、cookie、身份 attributes、ACL subject。
2. 用户问题、Agent goal、Prompt、模型消息、模型答案、澄清回答、审批原因正文。
3. 文档正文、chunk 文本、文件内容、上传字节、embedding、向量、对象存储正文。
4. 工具完整参数、工具完整结果、checkpoint、interrupt、resume payload、plan 正文。
5. 对象键、绝对路径、source locator、带查询参数 URL、数据库 SQL 文本、第三方原始响应。

异常日志也不能绕过该规则。生产环境写异常时记录异常类型、稳定错误码和安全 operation；默认不序列化 `str(exc)`。开发环境若保留 traceback，也必须先验证异常链不会携带正文或密钥。

### 4.3 属性与 label 白名单

Telemetry 工具模块采用白名单，而不是依赖后置黑名单。允许的公共字段限定为：

```text
service, component, operation, outcome, error_category, duration_ms,
request_id, trace_id, span_id, run_id, task_mode, task_status,
tool_name, tool_mode, worker_kind, queue_kind, phase,
candidate_count, context_token_estimate, input_tokens, output_tokens,
total_tokens, model_name, provider
```

`model_name` 仅可来自服务端 Settings，且应限制为已部署模型集合；不能从用户请求或工具参数获得。Metrics 默认不带 `model_name`，只带 `operation` 和 `provider`；只有确认模型集合稳定后才在专项面板加入模型 label。

## 5. 关联与异步 Trace 传播

### 5.1 标识语义

| 字段 | 生成者 | 生命周期 | 用途 |
|---|---|---|---|
| `request_id` | HTTP middleware | 一次 API 请求 | 用户和日志检索入口 |
| `trace_id` | OTel SDK 或有效 `traceparent` | 一条分布式 Trace | Jaeger 关联 |
| `span_id` | OTel SDK | 一个阶段 | Trace 内部定位 |
| `run_id` | `TaskWorker._run_claimed()` | 一次 Worker 执行 | 重试、恢复和业务事件关联 |
| `task_id` | 任务创建 | 任务全生命周期 | 业务事实和审计关联 |
| `job_id` | 文档上传/摄取创建 | 摄取生命周期 | 业务事实和审计关联 |

HTTP middleware 接受有效 W3C `traceparent` 作为上游上下文；`X-Request-ID` 只用于应用关联，不用于决定 Trace ID。未提供 `traceparent` 时由 SDK 创建根 Trace。

### 5.2 同步请求

```text
HTTP server span
  -> knowledge.query
    -> retrieval.search
      -> authorization.resolve
      -> vector.search_dense
      -> vector.search_sparse
      -> retrieval.fuse
      -> retrieval.rerank
      -> authorization.revalidate
      -> context.pack
    -> evidence.persist
    -> llm.generate
    -> evidence.revalidate
```

所有 span 仅记录阶段名称、数量、时长、结果和安全错误类别。`knowledge.query` 的 business outcome 必须显式区分 `answer`、`refusal` 和 `failure`，因为当前同步查询将部分运行失败转换为业务拒答响应，不能仅根据 HTTP 状态码判断质量或故障率。

### 5.3 异步任务

任务、摄取和审批恢复跨越 HTTP 进程和 Worker 进程，不能依靠 ContextVar 自动传播。

1. `POST /v1/tasks` 在同一数据库事务中把当前有效 `traceparent` 与可选 `tracestate` 保存到 `agent_tasks.pending_traceparent` 和 `pending_tracestate`。
2. `POST /v1/documents` 与版本上传在创建 `ingestion_jobs` 时保存同样字段。
3. Worker 领取后 extract 已保存的上下文，建立新的 `task.run` 或 `ingestion.run` 根 span，并以 producer span 建立 `Link`。
4. 不让数分钟、数小时甚至跨审批的执行成为原 HTTP span 的子 span，避免错误的超长父子 Trace。
5. 用户澄清和审批决定成功落库后，以当前 HTTP 请求的 trace context 覆盖 `pending_traceparent`。下一次 Worker run 链接到最近一次实际触发执行的请求。
6. `baggage` 不写入数据库，也不跨队列传播。

建议的新增数据库字段：

| 表 | 字段 | 说明 |
|---|---|---|
| `agent_tasks` | `pending_traceparent VARCHAR(55)` | 最近一次触发任务执行的 W3C 上下文 |
| `agent_tasks` | `pending_tracestate TEXT NULL` | 可选 W3C tracestate，不保存 baggage |
| `ingestion_jobs` | `pending_traceparent VARCHAR(55)` | 上传请求的 producer context |
| `ingestion_jobs` | `pending_tracestate TEXT NULL` | 可选 W3C tracestate |
| `task_events` | `correlation_id VARCHAR(64)` | 原 `trace_id` 改名，保留历史 request/run 关联语义 |
| `task_events` | `otel_trace_id VARCHAR(32) NULL` | 任务状态变化所在 Trace |
| `task_events` | `otel_span_id VARCHAR(16) NULL` | 任务状态变化所在 Span |

`TaskLease` 可携带 telemetry metadata，但 `TaskState` 继续只表达业务生命周期，不能将 OTel 内部状态扩散为 Agent 业务状态。

### 5.4 线程与心跳

`TaskWorker` lease heartbeat、摄取/投影 heartbeat、capability heartbeat 和 `ToolExecutionService` 的 `ThreadPoolExecutor` 都会跨线程。首期规则如下：

1. 工具 adapter 提交前使用 `contextvars.copy_context()` 包裹 `executor.submit()`，保证工具 span 是当前 `tool.execute` span 的子节点。
2. 成功 heartbeat 只更新 counter/gauge，不逐次建立 Trace span 或日志，避免噪声。
3. heartbeat 失败建立一个 `lease.renew` error span 和一条安全 ERROR 日志，记录 `kind`、`operation` 和稳定错误码。
4. 不将 ContextVar 泄漏到下一个请求、下一次领取或线程池复用任务；HTTP middleware 和 Worker run 均需在 `finally` 清理。

## 6. 指标契约

### 6.1 命名和标签规则

1. Counter 使用 `_total` 后缀；Histogram 使用 `_seconds` 后缀；Gauge 使用语义明确的瞬时名称。
2. 所有 label 必须为有限枚举或服务端受控配置。
3. Metrics 不记录 ID、原始错误码、文件名、URL、租户、用户、内容哈希或文本。
4. 原始 error code 只进入 Trace 和日志；Metrics 使用 `error_category`。
5. `error_category` 固定为 `none`、`validation`、`authorization`、`dependency`、`timeout`、`lease`、`integrity`、`storage`、`agent`、`unknown`。
6. 业务安全边界失败使用单独 Counter，不混入普通 5xx 错误率。

### 6.2 首期 Metrics

| 指标 | 类型 | 标签 | 记录边界 |
|---|---|---|---|
| `rag_kb_http_requests_total` | Counter | `route,method,status_class` | HTTP middleware finally |
| `rag_kb_http_request_duration_seconds` | Histogram | `route,method` | HTTP middleware finally |
| `rag_kb_knowledge_queries_total` | Counter | `outcome,reason` | `KnowledgeQueryService.query()` 退出 |
| `rag_kb_knowledge_query_duration_seconds` | Histogram | `outcome` | 同步问答总耗时 |
| `rag_kb_retrieval_stage_duration_seconds` | Histogram | `stage,outcome` | 授权、dense、sparse、融合、重排、复核、pack |
| `rag_kb_retrieval_candidates` | Histogram | `stage` | dense、sparse、fused、revalidated、packed 候选数量 |
| `rag_kb_retrieval_context_token_estimate` | Histogram | 无 | pack 后上下文估算 token，仅是估算值 |
| `rag_kb_llm_calls_total` | Counter | `operation,provider,outcome` | 同步回答和 Agent 模型调用 |
| `rag_kb_llm_call_duration_seconds` | Histogram | `operation,provider,outcome` | 同上 |
| `rag_kb_llm_tokens_total` | Counter | `operation,provider,direction` | 供应商真实 usage，`direction` 为 input/output |
| `rag_kb_llm_usage_missing_total` | Counter | `operation,provider` | 响应未提供可靠 usage 时 |
| `rag_kb_task_runs_total` | Counter | `mode,outcome,error_category` | 一次 Worker run 结束 |
| `rag_kb_task_run_duration_seconds` | Histogram | `mode,outcome` | `TaskWorker._run_claimed()` |
| `rag_kb_task_claims_total` | Counter | `mode,result` | 任务领取成功、空闲、协议不匹配等 |
| `rag_kb_lease_failures_total` | Counter | `kind,operation` | task、ingestion、projection、capability 续租失败 |
| `rag_kb_checkpoint_writes_total` | Counter | `outcome` | `FencedPostgresSaver.put()` / `put_writes()` |
| `rag_kb_tool_calls_total` | Counter | `tool,mode,outcome,error_category` | `ToolExecutionService.execute()` 结束 |
| `rag_kb_tool_call_duration_seconds` | Histogram | `tool,mode,outcome` | 工具持久化到验证完成 |
| `rag_kb_approvals_total` | Counter | `decision,outcome` | 创建、批准、拒绝、过期 |
| `rag_kb_approval_wait_duration_seconds` | Histogram | `decision` | 创建至决定或过期 |
| `rag_kb_ingestion_jobs_total` | Counter | `phase,outcome,error_category` | 一次 ingestion job 结束 |
| `rag_kb_ingestion_phase_duration_seconds` | Histogram | `phase,outcome` | parse、index、publish、bind、retire |
| `rag_kb_projection_reconciliations_total` | Counter | `phase,outcome` | audit/reconcile 结束 |
| `rag_kb_projection_reconcile_duration_seconds` | Histogram | `phase,outcome` | 一次 reconcile run |
| `rag_kb_recovery_actions_total` | Counter | `worker_kind,kind,outcome` | task/ingestion 各自拥有的恢复与维护动作 |
| `rag_kb_worker_errors_total` | Counter | `worker_kind,operation,error_category` | 三类 Worker 的维护、队列快照与主循环错误 |
| `rag_kb_security_boundary_violations_total` | Counter | `kind` | ACL、tenant、content hash、fencing、未知 WRITE |
| `rag_kb_task_queue_depth` | Gauge | `worker_kind,mode,status` | task Worker 的周期性 PostgreSQL 快照 |
| `rag_kb_task_queue_oldest_age_seconds` | Gauge | `worker_kind,mode,status` | task Worker 的周期性 PostgreSQL 快照 |
| `rag_kb_ingestion_queue_depth` | Gauge | `worker_kind,status` | ingestion Worker 的周期性 PostgreSQL 快照 |
| `rag_kb_ingestion_queue_oldest_age_seconds` | Gauge | `worker_kind,status` | ingestion Worker 的周期性 PostgreSQL 快照 |
| `rag_kb_worker_last_loop_timestamp_seconds` | Gauge | `worker_kind` | Worker 每轮成功完成后更新 |
| `rag_kb_worker_capability_available` | Gauge | `worker_kind,capability,protocol` | task Worker 的活动 capability lease 快照 |

所有 `worker_kind` 均由运行模式收敛到固定低基数集合 `task`、`ingestion`、`projection`，不能来自外部输入或实例 ID。

### 6.3 Histogram bucket

HTTP 使用短请求 bucket：

```text
0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30
```

模型、Agent、工具、摄取使用长操作 bucket：

```text
0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600
```

不要用平均值取代分位数。Grafana 中的 P95/P99 必须由 Prometheus Histogram bucket 用 `histogram_quantile()` 计算。

### 6.4 队列 Metric 的采集方式

队列深度和最老等待年龄是 PostgreSQL 全局事实，不能通过每个 Worker 的进程内计数相加得到。

1. `task-worker` 和 `ingestion-worker` 在各自维护窗口内执行有界队列快照查询；`projection-worker` 不重复采集这两类队列。
2. 任务的可领取定义必须复用 `DatabaseTaskRepository.claim_next()` 的状态和 capability 条件，不手写另一套不一致的“pending”判断。
3. 摄取可领取定义必须复用 `DocumentRepository.claim_next_job()` 的 `pending`、`available_at` 和 attempts 条件。
4. Trace 和 `/metrics` scrape 均不得触发高成本队列全表扫描。

首期只观测现有状态：Agent 的 `planning`、`running`、`waiting_for_user`、`waiting_for_approval`，摄取的 `awaiting_object`、`pending`、`running`。dashboard 明确区分“可领取积压”和“等待用户/审批”，不把等待人工操作误报为 Worker 堵塞。

## 7. Trace 契约

### 7.1 Span 命名

| 链路 | Span 名称 | 入口位置 |
|---|---|---|
| HTTP | `http.request`，完成后改名为 `HTTP {METHOD} {route}` | `web/app.py` middleware |
| 同步问答 | `knowledge.query` | `KnowledgeQueryService.query()` |
| 检索总过程 | `retrieval.search` | `EnterpriseRetriever.search()` |
| 检索阶段 | `retrieval.authorization`、`retrieval.dense`、`retrieval.sparse`、`retrieval.fuse`、`retrieval.rerank`、`retrieval.revalidate`、`retrieval.pack` | 同上 |
| 同步模型 | `llm.generate` | `DeepSeekAnswerModel.answer()` |
| 任务执行 | `task.run` | `TaskWorker._run_claimed()` |
| Agent 推进 | `agent.advance`、`agent.invoke` | `TaskWorker._advance()` |
| checkpoint | `checkpoint.write` | `agent/checkpointer.py` |
| 工具 | `tool.execute`、`tool.adapter`、`tool.verify`、`tool.complete` | `ToolExecutionService.execute()` |
| 审批 | `approval.transition`、`task.resume` | `ApprovalService`、`TaskWorker.resume_for_approval()` |
| 摄取 | `ingestion.run`、`ingestion.parse`、`projection.index`、`projection.publish`、`projection.bind`、`projection.retire` | `IngestionWorker.run_once()` |
| 投影收敛 | `projection.reconcile`、`projection.verify` | `ProjectionReconciler.run_once()` |
| task 维护 | `worker.task.maintenance`、`worker.task.queue_snapshot` | `task-worker` 的独占维护窗口 |
| ingestion 维护 | `worker.ingestion.maintenance`、`worker.ingestion.queue_snapshot` | `ingestion-worker` 的独占维护窗口 |

### 7.2 Span 状态

业务拒答、等待澄清和等待审批通常不是 OTel ERROR：

| 业务结果 | OTel status | 业务 attribute/metric outcome |
|---|---|---|
| 有证据回答 | `UNSET` | `answer` |
| 证据不足/冲突拒答 | `UNSET` | `refusal` |
| 等待用户澄清 | `UNSET` | `waiting_for_user` |
| 等待审批 | `UNSET` | `waiting_for_approval` |
| 用户拒绝写操作 | `UNSET` | `rejected` |
| 依赖超时、模型故障、Qdrant/DB 故障 | `ERROR` | `failure` |
| 失租/租约 fencing | `ERROR` | `stale` 或 `lease_failure` |
| ACL/hash/tenant 边界校验失败 | `ERROR` | `security_violation` |
| WRITE 工具 `unknown` | `ERROR` | `unknown` |

### 7.3 LLM span 与 token 语义

新增一个只采集元数据的 LangChain callback，并同时服务于同步回答和 Agent：

1. 每次模型 client invoke 建立 `llm.generate` 子 span。
2. 记录服务端配置的 `provider`、`model_name`、operation、时长和结果。
3. 从 `usage_metadata` 或 provider response metadata 中读取真实 `input_tokens`、`output_tokens`、`total_tokens`。
4. 响应未提供可靠 usage 时递增 `rag_kb_llm_usage_missing_total`，不填充伪造的 0。
5. `llm_calls_total` 表示项目发起的模型 client 调用次数，不等同于供应商内部 retry 或最终账单调用数。
6. 首期不计算金额。后续如要增加成本，必须建立按 provider/model/version 生效期管理的价格表，并对账验证。

callback 不读取、存储或输出任何 `messages`、Prompt、工具参数或模型文本。

## 8. 结构化日志契约

### 8.1 实现方式

首期新增一个轻量 `rag_kb.observability` 模块，包含 JSON formatter、ContextVar 绑定、Metrics registry、Tracer 初始化和小型 span/metric helper。先保持单模块；只有在 Metrics 定义或日志脱敏规则已经难以维护时才拆分子模块。

API 在 FastAPI lifespan 中初始化和关闭；Worker 在 `main()` 中初始化，在 `finally` 中 flush/shutdown。日志输出 stdout，不写本地 JSONL 文件，避免容器磁盘保留敏感信息和日志轮转问题。

### 8.2 日志字段

每条本项目日志使用下列扁平 JSON 字段：

```json
{
  "timestamp": "2026-08-29T12:34:56.789Z",
  "level": "INFO",
  "event": "agent.run.completed",
  "service": "rag-kb-worker-task",
  "component": "task_worker",
  "operation": "task.run",
  "outcome": "succeeded",
  "error_code": null,
  "duration_ms": 842.1,
  "request_id": "...",
  "trace_id": "...",
  "span_id": "...",
  "run_id": "...",
  "task_id": "...",
  "worker_id": "..."
}
```

字段不存在时省略，不写空字符串。日志 event 使用小写点分命名，例如：

```text
http.request.completed
knowledge.query.completed
retrieval.completed
agent.run.completed
tool.execution.completed
approval.transitioned
ingestion.job.completed
projection.reconcile.completed
worker.maintenance.completed
security.boundary_violated
```

### 8.3 日志级别

| 等级 | 场景 |
|---|---|
| `INFO` | HTTP 完成、任务终态、工具终态、摄取终态、审批决定、恢复批次汇总 |
| `WARNING` | 可重试依赖失败、队列积压阈值、工具超时、投影将重试 |
| `ERROR` | 未处理异常、失租、安全边界失败、WRITE `unknown`、终态写入失败 |
| `DEBUG` | 仅本地排障的安全元数据，仍不包含正文或密钥 |

不要为 Worker idle poll、每个成功 heartbeat、每个 chunk、每条 SQL 或每个向量候选输出日志。

## 9. 精确埋点位置

| 组件 | 文件/方法 | 必须记录 | 不记录 |
|---|---|---|---|
| HTTP | `web/app.py:create_app()` | route、method、status、duration、request/trace ID | 请求体、query string、header、响应正文 |
| 身份 | `web/dependencies.py:resolve_auth_context()` | 认证结果类别、request ID | token、角色、attributes |
| 问答 | `application/query_service.py:KnowledgeQueryService.query()` | outcome、reason、总时长、证据数量 | question、answer、citation 正文 |
| 检索 | `application/enterprise_retriever.py:EnterpriseRetriever.search()` | 每阶段时长、候选数量、context token estimate | query、chunk、ACL principal、score 明细 |
| 模型 | `answerer.py:DeepSeekAnswerModel.answer()` | provider/model、时长、usage、outcome | messages、context、output |
| Agent | `application/task_worker.py:_run_claimed()` | mode、run ID、结果、时长、恢复类型 | goal、interrupt/resume 内容 |
| checkpoint | `agent/checkpointer.py:put()/put_writes()` | outcome、时长、stale 类别 | checkpoint、pending write 内容 |
| 工具 | `application/tool_execution_service.py:execute()` | 工具名、mode、风险级别、时长、结果、timeout | arguments、result、对象引用 |
| 审批 | `application/approval_service.py` | transition、decision、等待时长 | reason、参数、计划、证据摘要 |
| 摄取 | `application/ingestion_worker.py:run_once()` | phase、时长、chunk count、结果 | 原始文档、文件路径、manifest 内容 |
| 投影 | `application/projection_reconciler.py:run_once()` | audit/reconcile phase、时长、结果 | payload、向量 |
| Worker | `worker.py:run()` | 队列循环、恢复汇总、capability、last loop | 任务/文档正文 |

Repository 层不逐个增加业务 span。数据库调用的诊断由上层业务边界 span 和既有错误码提供。若未来需要 SQL 细节，再单独评估 SQLAlchemy instrumentation 与 SQL 文本脱敏。

## 10. 数据库迁移与业务事件

### 10.1 迁移内容

M0 增加一份 Alembic migration，顺序如下：

1. 校验并处理 `task_events.trace_id` 历史值，将列重命名为 `correlation_id`。
2. 为 `task_events` 新增 `otel_trace_id`、`otel_span_id`，保持 nullable。
3. 为 `agent_tasks` 和 `ingestion_jobs` 新增 `pending_traceparent`、`pending_tracestate`，保持 nullable。
4. 为查询路径补充必要索引前，先使用 `EXPLAIN ANALYZE` 验证；首期不因遥测盲目新增索引。
5. 更新 ORM models、repository 映射、domain lease metadata 和迁移测试。

### 10.2 `task_events` 的使用边界

`task_events` 记录业务状态迁移，不是 OTel span 存储：

| 字段 | 首期写入规则 |
|---|---|
| `correlation_id` | 任务创建/恢复所在 request ID，或 Worker run ID |
| `otel_trace_id` / `otel_span_id` | 触发该状态迁移的当前 OTel context，若未启用观测则为 `NULL` |
| `stage` | 保持现有业务状态语义；不要填入细粒度 Agent 节点名 |
| `duration_ms` | 不回填任意 span 时长；仅当事件本身定义明确的业务时长时写入 |
| `token_usage` | 首期保持 `NULL`，token 属于 LLM Metrics/Trace，不伪装为任务事件总量 |
| `tool_name` | 任务状态事件首期保持 `NULL`；工具事实继续位于 `tool_calls` |

这避免将业务审计、指标和 Trace 三种模型混为一体。

## 11. 部署配置

### 11.1 新增依赖

首期只新增以下 Python 依赖：

```toml
prometheus-client
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp-proto-grpc
```

不在首期安装 FastAPI、HTTPX、SQLAlchemy 自动 instrumentation 包。手工业务 span 已覆盖项目关键路径，且更容易保证不输出 SQL、动态 URL 与正文。

### 11.2 Compose profile

新增服务：

| 服务 | 作用 | 对外端口 |
|---|---|---|
| `otel-collector` | 接收 OTLP Trace 并转发到 Jaeger | 仅 Docker 网络 `4317` |
| `jaeger` | Trace 存储和检索 UI | `127.0.0.1:16686` |
| `prometheus` | 拉取并保存 Metrics | `127.0.0.1:9090` |
| `grafana` | dashboard 和 alert rule UI | `127.0.0.1:3000` |

Prometheus 当前 scrape target：

```yaml
scrape_configs:
  - job_name: rag-kb-api
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
  - job_name: rag-kb-task-worker
    static_configs:
      - targets: ["task-worker:9464"]
  - job_name: rag-kb-ingestion-worker
    static_configs:
      - targets: ["ingestion-worker:9465"]
  - job_name: rag-kb-projection-worker
    static_configs:
      - targets: ["projection-worker:9466"]
```

这些目标对应当前三个独立 Worker 服务；扩容前必须改为每个 Worker 实例的服务发现，不能依赖 Docker service DNS 的单一名称。Prometheus target `DOWN` 表示抓取目标不可达，不等同于 API `/health/ready` 失败；readiness 不直接检查 Worker。

### 11.3 采样与保留

| 环境 | Trace sampling | Jaeger 保留 | 日志 |
|---|---:|---:|---|
| 本地开发/演示 | 100% | 24 小时或手动清理 | stdout，Docker 默认日志策略 |
| 预发布 | 100% | 3 至 7 天 | stdout，由平台采集 |
| 生产初期 | 应用端导出 100%；Collector tail sampling 保留错误/安全/慢请求 100%、普通请求 10% | 按容量和合规要求 | stdout，平台侧配置保留与访问控制 |

生产 tail sampling 是 Collector 配置项，不应散落在业务代码中。启用前必须基于真实流量估算 Trace 体积，并确认任何 span 都不含正文。

## 12. Dashboard 与告警

### 12.1 Dashboard

| Dashboard | 面板 |
|---|---|
| 系统总览 | API 与三个 Worker 分别 up、请求率、5xx、P50/P95/P99、Collector 导出失败 |
| RAG | 问答 answer/refusal/failure、拒答原因、检索阶段时长、候选数、context token estimate、LLM usage 缺失率 |
| Agent | 任务领取/终态、模型调用、工具成功率和超时、审批等待、checkpoint/lease 失败、任务队列年龄 |
| 摄取与投影 | 摄取积压、解析/index/publish 时长、重试、投影 audit/reconcile 结果、未收敛数量 |
| 安全 | ACL/tenant/hash/fencing 边界失败、WRITE unknown、审批拒绝/过期、对象 hash mismatch |

首期 Grafana dashboard 以 JSON provisioning 放在 `ops/observability/grafana/`。不在业务目录中存放 Grafana 产物。

### 12.2 初始告警

| 等级 | 条件 | 说明 |
|---|---|---|
| Critical | `increase(rag_kb_security_boundary_violations_total[5m]) > 0` | ACL、tenant、hash 或 fencing 边界失败必须人工确认 |
| Critical | `increase(rag_kb_tool_calls_total{mode="write",outcome="unknown"}[5m]) > 0` | WRITE 副作用结果不确定 |
| Critical | task queue 非空且 capability lease 不可用 | 已有任务等待消费；lease 到期后新建 task 返回 503 |
| High | lease failure 持续 5 分钟 | 可能发生重复执行、恢复失败或 Worker 健康问题 |
| High | projection 连续两个审计周期未收敛 | 检索投影可能持续不一致 |
| High | PostgreSQL、Qdrant 或模型 dependency 错误率超阈值 | 外部依赖可用性下降 |
| Medium | API 5xx 比例或 P95 超过基线 | 用户可见服务退化 |
| Medium | task/ingestion 最老可领取任务年龄超过 SLO | Worker 吞吐不足或卡住 |
| Medium | 工具 timeout、审批过期、保留清理失败率异常 | 需排查业务操作与后台清理 |

示例 P95 查询：

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(rag_kb_knowledge_query_duration_seconds_bucket[5m])
  )
)
```

示例安全边界告警：

```promql
increase(rag_kb_security_boundary_violations_total[5m]) > 0
```

首期用 Grafana Alerting 提醒即可。接入邮件、Webhook、企业 IM 或 PagerDuty 前，应单独确认通知对象、静默策略、升级策略和敏感字段脱敏。

## 13. 实施批次

### M0：关联 ID 与迁移契约

目标：修复现有 ID 长度风险，定义 request/run/trace 三者边界。

| 改动 | 主要文件 |
|---|---|
| 收紧、生成并回写 `X-Request-ID` | `web/app.py`、`web/dependencies.py` |
| `task_events` 字段语义迁移 | 新 Alembic migration、`database/models.py`、`task_repository.py` |
| 保存异步 producer trace context | `agent_tasks`、`ingestion_jobs` ORM/repository、任务与上传创建路径 |
| 覆盖边界值与迁移 | `test_v1_knowledge_api.py`、`test_v1_tasks_api.py`、`test_database_migrations.py` |

完成标准：64 字符以上的无效 request ID 不会进入数据库；所有响应返回 request ID；关闭观测时迁移后的任务与摄取行为不变。

### M1：遥测基础模块与 HTTP

目标：建立安全 no-op、JSON 日志、API Metrics 和 HTTP Trace。

| 改动 | 主要文件 |
|---|---|
| 新增 `rag_kb.observability` 模块 | `src/rag_kb/observability.py` |
| Settings 与生命周期 | `config.py`、`web/app.py` |
| HTTP middleware、`/metrics`、响应 request ID | `web/app.py` |
| JSON stdout formatter 和 ContextVar 清理 | `observability.py` |
| 单元测试 | `tests/test_new_app_factory.py` |

完成标准：`/metrics` 可被独立 registry 测试；同一 HTTP 请求的 JSON 日志包含 request/trace ID；任意动态路径不作为 Metrics label；Exporter 不可达时 API 仍可用。

### M2：同步 RAG 与模型观测

目标：能解释问答失败、拒答和检索慢点。

| 改动 | 主要文件 |
|---|---|
| 问答总 span/metric | `application/query_service.py` |
| 检索阶段 span/metric | `application/enterprise_retriever.py` |
| 模型 callback、usage 与时长 | `answerer.py`、Agent 构造/调用处 |
| 不泄漏测试 | `tests/test_enterprise_query_service.py`、`tests/test_enterprise_retriever_security.py` |

完成标准：一个同步查询在 Jaeger 展示授权、检索、模型和引用复核；Metrics 能区分 answer/refusal/failure；Trace 与日志不含问题、chunk 或回答文本。

### M3：Agent、工具、审批与租约

目标：能解释任务恢复、工具超时、审批和 UNKNOWN WRITE。

| 改动 | 主要文件 |
|---|---|
| task run/advance/invoke Trace | `application/task_worker.py` |
| 工具 thread context 与工具时长 | `application/tool_execution_service.py` |
| checkpoint 和 fencing | `agent/checkpointer.py` |
| 审批 transition/wait duration | `application/approval_service.py` |
| 恢复和线程传播测试 | `test_task_worker_recovery.py`、`test_task_approval_recovery.py`、`test_approval_tool_execution.py` |

完成标准：一次从创建、HITL、审批到恢复完成的任务可以按 `task_id` 在日志中定位，并在 Jaeger 中看到每次 Worker run 与 producer Link；工具线程内的 span 不孤立。

### M4：摄取、投影、恢复和 Worker Metrics

目标：能发现摄取积压、投影不收敛和后台恢复问题。

| 改动 | 主要文件 |
|---|---|
| 摄取 parse/index/publish/bind/retire | `application/ingestion_worker.py` |
| 投影 audit/reconcile | `application/projection_reconciler.py` |
| 队列快照、恢复汇总、Worker exporter | `worker.py`、repository 只读统计入口 |
| 测试 | `test_ingestion_worker.py`、`test_projection_reconciler.py`、`test_worker_runtime.py` |

完成标准：三个 Worker 的 `:9464`、`:9465`、`:9466` 指标可在 Compose 网络内分别抓取；任务和摄取 dashboard 能区分可领取积压、等待人工和运行中；成功 heartbeat 不造成 Trace/日志洪泛。

### M5：本地观测 Stack、Dashboard 与告警

目标：让项目可以一条 Compose 命令完成可演示观测环境。

| 改动 | 主要文件 |
|---|---|
| Collector、Jaeger、Prometheus、Grafana profile | `docker-compose.yml` |
| Prometheus scrape/recording rules | `ops/observability/prometheus/` |
| Grafana datasource/dashboard/alert provisioning | `ops/observability/grafana/` |
| 运维入口文档 | 本文档和 `README.md` 的最小链接 |

完成标准：profile 启动后可在 Grafana 分别看到 API、task-worker、ingestion-worker、projection-worker、RAG、Agent 和摄取信号；Jaeger 能按 service 和 trace ID 查询；关闭 profile 时主服务仍可独立启动。

### M6：质量门与验收

目标：将观测变成可回归验证的工程能力，而不是演示截图。

| 验证 | 标准 |
|---|---|
| 单元测试 | 使用 InMemory span exporter 和独立 Prometheus registry，无外部网络依赖 |
| 敏感信息回归 | 用假 token、Prompt、正文、对象路径和工具参数触发异常，断言 Metrics/Trace/Logs 不出现 |
| 高基数回归 | 以多个 task/document ID 请求，断言只有 route template 和固定枚举 label |
| 线程传播 | 工具 adapter 线程中的 span 与父 `tool.execute` Trace 一致 |
| 异步传播 | 创建/审批恢复/Worker run 使用正确 Trace Link，不生成跨小时父子 span |
| 故障降级 | Collector 不可用、OTLP endpoint 错误、Metrics 关闭时，现有业务测试仍通过 |
| 性能 | 普通 API P95 增幅低于 5%；模型/RAG 请求附加开销低于 2%；`/metrics` 响应低于 100ms |

## 14. 验收演示脚本

完成 M5 后，演示按以下顺序执行：

1. 使用 `docker compose --profile observability up --build -d` 启动服务。
2. 访问 `/health/live`、`/health/ready` 和 API `/metrics`，确认 Prometheus target 为 `UP`。
3. 发起一条可回答的 `/v1/knowledge/query`，用响应 `X-Request-ID` 在 JSON 日志和 Jaeger 中定位同一次调用。
4. 发起一条证据不足查询，确认 HTTP 仍成功但 `rag_kb_knowledge_queries_total{outcome="refusal"}` 增加，Trace 不标记 ERROR。
5. 创建一个 research task，确认 Grafana 出现 queue 变化和 task run；Jaeger 展示 `task.run -> agent.invoke -> tool.execute`。
6. 创建一个 controlled-action task 并经过审批恢复，确认两次 Worker run 通过 Link 关联，审批等待时长进入 Histogram。
7. 上传一个文档，确认 ingestion、projection 和队列 Metrics 更新。
8. 模拟一个安全边界或 WRITE unknown 测试路径，确认安全 Counter 和 Critical alert 被触发，同时不泄露敏感正文。

## 15. GitHub 对标依据

本方案借鉴公开项目的边界设计，不直接复制其实现：

| 来源 | 借鉴点 | 本项目的取舍 |
|---|---|---|
| [AegisRAG](https://github.com/chyinan/AegisRAG) | Prometheus/Grafana、OTLP/Jaeger、JSON 日志和敏感键脱敏 | 保留三信号分工；不采用其全量自动 instrumentation 和 `ALWAYS_ON` 采样 |
| [FastAPI LangGraph Agent Template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) | request ID、HTTP Metrics、LLM 时长、结构化日志 | 不使用原始 URL path 作为 label；不默认记录模型输入输出 |
| [Google agents-cli observability guidance](https://github.com/google/agents-cli) | Agent span 层级、工具 span、内容采集分级、默认无内容 | 使用 `task.run -> agent.invoke -> llm/tool` 层级；不接入 Google Cloud 专有设施 |
| [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit) | 可选 LLM tracing、端到端 smoke 验证 | 将第三方 LLM 追踪延后，优先覆盖本项目特有的租约、审批和摄取恢复 |

## 16. 方案完成定义

以下条件全部满足，才能将“已实现可观测性”写入 README、简历或项目验收报告：

1. M0 至 M6 的代码、迁移、Compose profile、dashboard、告警规则和测试均已合并。
2. 当前 revision 重新运行全量测试与观测专项测试，结果可复现。
3. 本地演示 Trace、Metrics、Logs 可按 request ID、trace ID、task ID 相互关联。
4. 敏感信息和高基数 label 回归测试通过。
5. Collector、Jaeger 或 Prometheus 不可用时，业务路径不会失败或被显著阻塞。
6. 文档中所有“已实现”表述都附有当前 revision 的验收证据，而不是引用历史报告。
