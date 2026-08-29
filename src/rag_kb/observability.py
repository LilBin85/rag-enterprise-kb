"""M0/M1 的最小可观测性基础设施。"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import Callable, Iterator, Mapping, Sequence
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Link, NoOpTracerProvider, SpanKind, Status, StatusCode, use_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

try:  # 保持 HTTP-only 运行环境无需为了遥测导入 LangChain。
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - 仅用于没有模型依赖的最小运行环境。
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_SAFE_EVENT_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_SAFE_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,99}\Z")
_TRACE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_SPAN_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")
_SAFE_LOG_FIELDS = frozenset(
    {
        "component",
        "operation",
        "outcome",
        "kind",
        "decision",
        "reason",
        "error_code",
        "error_category",
        "duration_ms",
        "run_id",
        "task_id",
        "worker_id",
        "task_mode",
        "task_status",
        "tool_name",
        "tool_mode",
        "worker_kind",
        "queue_kind",
        "phase",
        "candidate_count",
        "context_token_estimate",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "model_name",
        "provider",
    }
)
_HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_LONG_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600)
_ERROR_CATEGORIES = frozenset(
    {
        "none",
        "validation",
        "authorization",
        "dependency",
        "timeout",
        "lease",
        "integrity",
        "storage",
        "agent",
        "unknown",
    }
)
_KNOWLEDGE_REASONS = frozenset(
    {
        "none",
        "no_authorized_evidence",
        "insufficient_evidence",
        "conflicting_evidence",
        "retrieval_failure",
    }
)
_RETRIEVAL_STAGES = frozenset(
    {"authorization", "dense", "sparse", "fuse", "rerank", "revalidate", "pack"}
)
_INGESTION_JOB_PHASES = frozenset({"ingest", "publish", "cleanup"})
_INGESTION_PHASES = frozenset({"parse", "index", "publish", "bind", "retire"})
_INGESTION_OUTCOMES = frozenset(
    {"succeeded", "retry_scheduled", "failed", "failure_recording_failed"}
)
_PROJECTION_PHASES = frozenset({"audit", "reconcile"})
_PROJECTION_OUTCOMES = frozenset(
    {"succeeded", "retry_scheduled", "failure_recording_failed"}
)
_PHASE_OUTCOMES = frozenset({"succeeded", "failure"})
_RECOVERY_KINDS = frozenset(
    {
        "stale_tool_call",
        "stale_knowledge_query",
        "waiting_task",
        "tool_result_retention",
        "task_content_retention",
        "orphan_result",
        "upload_reservation",
        "queue_snapshot",
    }
)
_RECOVERY_OUTCOMES = frozenset(
    {"succeeded", "recovered", "missing", "mismatched", "failure"}
)
_TASK_QUEUE_MODES = frozenset({"research", "controlled_action"})
_TASK_QUEUE_STATUSES = frozenset(
    {"planning", "running", "waiting_for_user", "waiting_for_approval"}
)
_INGESTION_QUEUE_STATUSES = frozenset({"awaiting_object", "pending", "running"})
_WORKER_KINDS = frozenset({"task", "ingestion", "projection"})
_SECURITY_BOUNDARY_KINDS = frozenset(
    {"acl", "tenant", "content_hash", "fencing", "write_unknown"}
)
_SAFE_SPAN_FIELDS = _SAFE_LOG_FIELDS

_request_id: ContextVar[str | None] = ContextVar("rag_kb_request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("rag_kb_run_id", default=None)
_telemetry_active: ContextVar[bool] = ContextVar("rag_kb_telemetry_active", default=False)
_current_observability: ContextVar[Observability | None] = ContextVar(
    "rag_kb_observability", default=None
)
_trace_propagator = TraceContextTextMapPropagator()
_logger = logging.getLogger("rag_kb.observability")


def normalize_request_id(value: str | None) -> str:
    """只接受有限应用关联 ID，其他输入由服务端替换。"""
    if value is not None and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid4().hex


def bind_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def clear_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def current_request_id() -> str | None:
    return _request_id.get()


def bind_run_id(value: str) -> Token[str | None]:
    return _run_id.set(value)


def clear_run_id(token: Token[str | None]) -> None:
    _run_id.reset(token)


def current_run_id() -> str | None:
    return _run_id.get()


def current_observability() -> Observability | None:
    return _current_observability.get()


def stable_error_code(exc: BaseException) -> str:
    """仅保留项目稳定 code；未知异常绝不序列化异常文本。"""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _SAFE_ERROR_CODE_PATTERN.fullmatch(code):
        return code
    name = type(exc).__name__.lower()
    return f"exception_{name}" if _SAFE_ERROR_CODE_PATTERN.fullmatch(f"exception_{name}") else "exception_unknown"


def error_category_for(code: str | None) -> str:
    """将稳定错误码收敛为低基数 Metrics label。"""
    value = (code or "").lower()
    if any(part in value for part in ("lease", "fenc", "stale")):
        return "lease"
    if "timeout" in value or "deadline" in value:
        return "timeout"
    if any(part in value for part in ("acl", "auth", "permission", "principal", "tenant", "denied")):
        return "authorization"
    if any(part in value for part in ("hash", "integrity", "binding", "citation", "evidence")):
        return "integrity"
    if any(part in value for part in ("object", "storage", "persist")):
        return "storage"
    if any(part in value for part in ("model", "retrieval", "qdrant", "database", "dependency")):
        return "dependency"
    if any(part in value for part in ("invalid", "missing", "required", "format", "schema")):
        return "validation"
    if "agent" in value or "checkpoint" in value or "tool" in value:
        return "agent"
    return "unknown"


def _metric_error_category(value: str) -> str:
    return value if value in _ERROR_CATEGORIES else "unknown"


def _bounded_metric_value(value: str, allowed: frozenset[str], fallback: str) -> str:
    return value if value in allowed else fallback


@contextmanager
def observe_span(
    name: str,
    *,
    attributes: Mapping[str, object] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Iterator[object]:
    """在激活实例上建业务 span；关闭或未初始化时退化为 no-op。"""
    observability = current_observability()
    if observability is not None:
        with observability.span(name, attributes=attributes, kind=kind) as span:
            yield span
        return
    with trace.get_tracer("rag_kb.business").start_as_current_span(
        name,
        kind=kind,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        yield span


def finish_observed_span(
    span: object,
    *,
    outcome: str,
    error_category: str = "none",
    error_code: str | None = None,
    attributes: Mapping[str, object] | None = None,
) -> None:
    observability = current_observability()
    if observability is not None:
        observability.finish_span(
            span,
            outcome=outcome,
            error_category=error_category,
            error_code=error_code,
            attributes=attributes,
        )
        return
    if error_category != "none":
        span.set_status(Status(StatusCode.ERROR))


class MetadataOnlyLlmCallback(BaseCallbackHandler):
    """只读取模型 usage metadata，绝不访问 messages、prompt 或 output 文本。"""

    def __init__(self, *, provider: str, model_name: str, operation: str) -> None:
        self._provider = provider
        self._model_name = model_name
        self._operation = operation
        self._active: dict[str, tuple[object, float, Observability]] = {}
        self._lock = RLock()

    def on_chat_model_start(self, _serialized, _messages, *, run_id, **_kwargs) -> None:
        self._start(run_id)

    def on_llm_start(self, _serialized, _prompts, *, run_id, **_kwargs) -> None:
        self._start(run_id)

    def on_llm_end(self, response, *, run_id, **_kwargs) -> None:
        input_tokens, output_tokens, total_tokens = _usage_from_response(response)
        self._finish(
            run_id,
            outcome="succeeded",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def on_llm_error(self, error, *, run_id, **_kwargs) -> None:
        self._finish(
            run_id,
            outcome="failure",
            error_code=stable_error_code(error),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )

    def _start(self, run_id: object) -> None:
        observability = current_observability()
        if observability is None:
            return
        key = str(run_id)
        with self._lock:
            if key in self._active:
                return
            observability.start()
            span = observability._tracer.start_span("llm.generate")
            observability.set_span_attributes(
                span,
                {
                    "operation": self._operation,
                    "provider": self._provider,
                    "model_name": self._model_name,
                },
            )
            self._active[key] = (span, perf_counter(), observability)

    def _finish(
        self,
        run_id: object,
        *,
        outcome: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            active = self._active.pop(str(run_id), None)
        if active is None:
            return
        span, started_at, observability = active
        duration_seconds = perf_counter() - started_at
        category = error_category_for(error_code) if error_code is not None else "none"
        observability.finish_span(
            span,
            outcome=outcome,
            error_category=category,
            error_code=error_code,
            attributes={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "duration_ms": duration_seconds * 1_000,
            },
        )
        observability.record_llm(
            operation=self._operation,
            provider=self._provider,
            outcome=outcome,
            duration_seconds=duration_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        with use_span(span, end_on_exit=False):
            log_event(
                "llm.generate.completed",
                level=logging.ERROR if outcome == "failure" else logging.INFO,
                component="llm",
                operation=self._operation,
                outcome=outcome,
                error_category=category,
                error_code=error_code,
                duration_ms=duration_seconds * 1_000,
                provider=self._provider,
                model_name=self._model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        span.end()


def _usage_from_response(response: object) -> tuple[int | None, int | None, int | None]:
    """仅抽取数值 usage 字段；不读取 generation/message 的正文。"""
    candidates: list[Mapping[str, object]] = []
    if isinstance(response, Mapping):
        candidates.append(response)
    for attribute in ("usage_metadata", "response_metadata", "llm_output"):
        value = getattr(response, attribute, None)
        if isinstance(value, Mapping):
            candidates.append(value)
            nested = value.get("token_usage") or value.get("usage")
            if isinstance(nested, Mapping):
                candidates.append(nested)
    generations = getattr(response, "generations", None)
    if isinstance(generations, Sequence) and generations and isinstance(generations[0], Sequence):
        first = generations[0][0] if generations[0] else None
        message = getattr(first, "message", None)
        for attribute in ("usage_metadata", "response_metadata"):
            value = getattr(message, attribute, None)
            if isinstance(value, Mapping):
                candidates.append(value)
    for usage in candidates:
        input_tokens = _usage_number(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_number(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_number(usage, "total_tokens")
        if input_tokens is not None or output_tokens is not None or total_tokens is not None:
            return input_tokens, output_tokens, total_tokens
    return None, None, None


def _usage_number(usage: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def extract_trace_context(
    traceparent: str | None, tracestate: str | None
) -> Context | None:
    """仅提取 W3C Trace Context，不接受或传播 baggage。"""
    carrier = {
        key: value
        for key, value in (("traceparent", traceparent), ("tracestate", tracestate))
        if value is not None
    }
    if "traceparent" not in carrier:
        return None
    context = _trace_propagator.extract(carrier)
    if not trace.get_current_span(context).get_span_context().is_valid:
        return None
    return context


def current_trace_headers() -> tuple[str | None, str | None]:
    """返回当前 span 注入后的规范 W3C 字段，绝不包含 baggage。"""
    if not _telemetry_active.get():
        return None, None
    carrier: dict[str, str] = {}
    _trace_propagator.inject(carrier)
    traceparent = carrier.get("traceparent")
    if extract_trace_context(traceparent, carrier.get("tracestate")) is None:
        return None, None
    return traceparent, carrier.get("tracestate")


def current_span_ids() -> tuple[str | None, str | None]:
    if not _telemetry_active.get():
        return None, None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None, None
    return f"{context.trace_id:032x}", f"{context.span_id:016x}"


def log_event(event: str, *, level: int = logging.INFO, **fields: object) -> None:
    """写入白名单结构化事件，调用方不能借日志传递任意正文。"""
    extra: dict[str, object] = {"event": event}
    for key, value in fields.items():
        if key in _SAFE_LOG_FIELDS and _safe_log_value(key, value):
            extra[key] = value
    _logger.log(level, event, extra=extra)


def configure_json_logging(*, service: str, level: str) -> None:
    """为项目日志器安装唯一的 stdout JSON handler。"""
    logger = logging.getLogger("rag_kb")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in logger.handlers:
        if getattr(handler, "_rag_kb_json_handler", False):
            handler.setFormatter(JsonFormatter(service))
            return
    handler = logging.StreamHandler(sys.stdout)
    handler._rag_kb_json_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter(service))
    logger.addHandler(handler)


class JsonFormatter(logging.Formatter):
    """忽略原始 log message 和异常文本，避免异常链绕过数据分类边界。"""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        event = record.__dict__.get("event")
        payload: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "event": event if isinstance(event, str) and _SAFE_EVENT_PATTERN.fullmatch(event) else "application.log",
            "service": self._service,
        }
        for field in _SAFE_LOG_FIELDS:
            value = record.__dict__.get(field)
            if _safe_log_value(field, value):
                payload[field] = value
        request_id = current_request_id()
        if request_id is not None:
            payload["request_id"] = request_id
        run_id = current_run_id()
        if run_id is not None:
            payload["run_id"] = run_id
        trace_id, span_id = current_span_ids()
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if span_id is not None:
            payload["span_id"] = span_id
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class _SafeSpanExporter(SpanExporter):
    """导出失败不影响业务请求，并将诊断压缩为一次安全事件。"""

    def __init__(self, exporter: SpanExporter, warn: Callable[[], None]) -> None:
        self._exporter = exporter
        self._warn = warn

    def export(self, spans) -> SpanExportResult:
        try:
            result = self._exporter.export(spans)
        except Exception:
            self._warn()
            return SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            self._warn()
        return result

    def shutdown(self) -> None:
        try:
            self._exporter.shutdown()
        except Exception:
            self._warn()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        try:
            return self._exporter.force_flush(timeout_millis=timeout_millis)
        except Exception:
            self._warn()
            return False


class Observability:
    """每个应用实例独立持有 tracer provider 与 Prometheus registry。"""

    def __init__(
        self,
        *,
        enabled: bool,
        metrics_enabled: bool,
        service_name: str,
        otlp_endpoint: str,
        trace_sample_ratio: float,
    ) -> None:
        self.enabled = enabled
        self.metrics_enabled = metrics_enabled
        self.registry = CollectorRegistry()
        self._requests = Counter(
            "rag_kb_http_requests_total",
            "HTTP requests by route, method and status class.",
            ("route", "method", "status_class"),
            registry=self.registry,
        )
        self._duration = Histogram(
            "rag_kb_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("route", "method"),
            buckets=_HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._knowledge_queries = Counter(
            "rag_kb_knowledge_queries_total",
            "Knowledge query outcomes.",
            ("outcome", "reason"),
            registry=self.registry,
        )
        self._knowledge_duration = Histogram(
            "rag_kb_knowledge_query_duration_seconds",
            "Knowledge query duration in seconds.",
            ("outcome",),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._retrieval_duration = Histogram(
            "rag_kb_retrieval_stage_duration_seconds",
            "Retrieval stage duration in seconds.",
            ("stage", "outcome"),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._retrieval_candidates = Histogram(
            "rag_kb_retrieval_candidates",
            "Retrieval candidates by bounded stage.",
            ("stage",),
            registry=self.registry,
        )
        self._retrieval_context_tokens = Histogram(
            "rag_kb_retrieval_context_token_estimate",
            "Estimated packed context tokens.",
            registry=self.registry,
        )
        self._llm_calls = Counter(
            "rag_kb_llm_calls_total",
            "LLM client calls.",
            ("operation", "provider", "outcome"),
            registry=self.registry,
        )
        self._llm_duration = Histogram(
            "rag_kb_llm_call_duration_seconds",
            "LLM client call duration in seconds.",
            ("operation", "provider", "outcome"),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._llm_tokens = Counter(
            "rag_kb_llm_tokens_total",
            "Provider-reported LLM tokens.",
            ("operation", "provider", "direction"),
            registry=self.registry,
        )
        self._llm_usage_missing = Counter(
            "rag_kb_llm_usage_missing_total",
            "LLM responses without reliable usage metadata.",
            ("operation", "provider"),
            registry=self.registry,
        )
        self._task_runs = Counter(
            "rag_kb_task_runs_total",
            "Worker task run outcomes.",
            ("mode", "outcome", "error_category"),
            registry=self.registry,
        )
        self._task_duration = Histogram(
            "rag_kb_task_run_duration_seconds",
            "Worker task run duration in seconds.",
            ("mode", "outcome"),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._lease_failures = Counter(
            "rag_kb_lease_failures_total",
            "Lease renewal failures.",
            ("kind", "operation"),
            registry=self.registry,
        )
        self._checkpoint_writes = Counter(
            "rag_kb_checkpoint_writes_total",
            "Checkpoint write outcomes.",
            ("outcome",),
            registry=self.registry,
        )
        self._tool_calls = Counter(
            "rag_kb_tool_calls_total",
            "Tool execution outcomes.",
            ("tool", "mode", "outcome", "error_category"),
            registry=self.registry,
        )
        self._tool_duration = Histogram(
            "rag_kb_tool_call_duration_seconds",
            "Tool execution duration in seconds.",
            ("tool", "mode", "outcome"),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._approvals = Counter(
            "rag_kb_approvals_total",
            "Approval transitions.",
            ("decision", "outcome"),
            registry=self.registry,
        )
        self._approval_wait = Histogram(
            "rag_kb_approval_wait_duration_seconds",
            "Approval waiting duration in seconds.",
            ("decision",),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._ingestion_jobs = Counter(
            "rag_kb_ingestion_jobs_total",
            "Ingestion job outcomes.",
            ("phase", "outcome", "error_category"),
            registry=self.registry,
        )
        self._ingestion_phase_duration = Histogram(
            "rag_kb_ingestion_phase_duration_seconds",
            "Ingestion phase duration in seconds.",
            ("phase", "outcome"),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._projection_reconciliations = Counter(
            "rag_kb_projection_reconciliations_total",
            "Projection reconciliation outcomes.",
            ("phase", "outcome"),
            registry=self.registry,
        )
        self._projection_reconcile_duration = Histogram(
            "rag_kb_projection_reconcile_duration_seconds",
            "Projection reconciliation duration in seconds.",
            ("phase", "outcome"),
            buckets=_LONG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._recovery_actions = Counter(
            "rag_kb_recovery_actions_total",
            "Worker recovery actions.",
            ("kind", "outcome"),
            registry=self.registry,
        )
        self._security_boundary_violations = Counter(
            "rag_kb_security_boundary_violations_total",
            "Security boundary violations.",
            ("kind",),
            registry=self.registry,
        )
        self._task_queue_depth = Gauge(
            "rag_kb_task_queue_depth",
            "Task queue depth by fixed mode and status.",
            ("mode", "status"),
            registry=self.registry,
        )
        self._task_queue_oldest_age = Gauge(
            "rag_kb_task_queue_oldest_age_seconds",
            "Age of the oldest task queue item by fixed mode and status.",
            ("mode", "status"),
            registry=self.registry,
        )
        self._ingestion_queue_depth = Gauge(
            "rag_kb_ingestion_queue_depth",
            "Ingestion queue depth by fixed status.",
            ("status",),
            registry=self.registry,
        )
        self._ingestion_queue_oldest_age = Gauge(
            "rag_kb_ingestion_queue_oldest_age_seconds",
            "Age of the oldest ingestion queue item by fixed status.",
            ("status",),
            registry=self.registry,
        )
        self._worker_last_loop = Gauge(
            "rag_kb_worker_last_loop_timestamp_seconds",
            "Unix timestamp of the last completed worker loop.",
            ("worker_kind",),
            registry=self.registry,
        )
        self._worker_capability_available = Gauge(
            "rag_kb_worker_capability_available",
            "Whether the fixed task worker capability has an active lease.",
            ("capability", "protocol"),
            registry=self.registry,
        )
        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint
        self._trace_sample_ratio = trace_sample_ratio
        self._provider: TracerProvider | None = None
        self._tracer = NoOpTracerProvider().get_tracer("rag_kb.http")
        self._started = False
        self._closed = False
        self._exporter_warning_sent = False
        self._lock = RLock()

    def start(self) -> None:
        if self._started or self._closed:
            return
        with self._lock:
            if self._started or self._closed:
                return
            if self.enabled:
                provider = TracerProvider(
                    sampler=ParentBased(TraceIdRatioBased(self._trace_sample_ratio)),
                    resource=Resource({"service.name": self._service_name}),
                )
                self._provider = provider
                self._tracer = provider.get_tracer("rag_kb.http")
                if self._otlp_endpoint:
                    self._configure_exporter(provider)
            self._started = True

    def shutdown(self) -> None:
        if self._closed:
            return
        with self._lock:
            if self._closed:
                return
            if self._provider is not None:
                try:
                    self._provider.force_flush(timeout_millis=1_000)
                    self._provider.shutdown()
                except Exception:
                    self._warn_exporter_unavailable()
            self._closed = True

    @contextmanager
    def activate(self) -> Iterator["Observability"]:
        """绑定当前实例；Metrics 可在 Trace 关闭时继续工作。"""
        self.start()
        observability_token = _current_observability.set(self)
        active_token = _telemetry_active.set(self.enabled)
        try:
            yield self
        finally:
            _telemetry_active.reset(active_token)
            _current_observability.reset(observability_token)

    @contextmanager
    def http_span(self, parent_context: Context | None) -> Iterator[object]:
        with self.activate():
            with self._tracer.start_as_current_span(
                "http.request",
                context=parent_context,
                kind=SpanKind.SERVER,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                yield span

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
        context: Context | None = None,
        links: Sequence[Link] | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Iterator[object]:
        """只接受白名单属性的业务边界 span。"""
        self.start()
        with self._tracer.start_as_current_span(
            name,
            context=context,
            links=links,
            kind=kind,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self.set_span_attributes(span, attributes or {})
            try:
                yield span
            except Exception as exc:
                code = stable_error_code(exc)
                self.finish_span(
                    span,
                    outcome="failure",
                    error_category=error_category_for(code),
                    error_code=code,
                )
                raise

    @contextmanager
    def task_run_span(
        self,
        *,
        traceparent: str | None,
        tracestate: str | None,
        attributes: Mapping[str, object],
    ) -> Iterator[object]:
        """Worker 每次领取均新建 root，只以 Link 关联生产者。"""
        producer = extract_trace_context(traceparent, tracestate)
        links: list[Link] = []
        if producer is not None:
            producer_span = trace.get_current_span(producer).get_span_context()
            if producer_span.is_valid:
                links.append(Link(producer_span))
        with self.activate():
            with self.span(
                "task.run",
                attributes=attributes,
                context=Context(),
                links=links,
                kind=SpanKind.CONSUMER,
            ) as span:
                yield span

    @contextmanager
    def ingestion_run_span(
        self,
        *,
        traceparent: str | None,
        tracestate: str | None,
        attributes: Mapping[str, object],
    ) -> Iterator[object]:
        """摄取领取建立新 root，仅以 Link 关联上传请求。"""
        producer = extract_trace_context(traceparent, tracestate)
        links: list[Link] = []
        if producer is not None:
            producer_span = trace.get_current_span(producer).get_span_context()
            if producer_span.is_valid:
                links.append(Link(producer_span))
        with self.activate():
            with self.span(
                "ingestion.run",
                attributes=attributes,
                context=Context(),
                links=links,
                kind=SpanKind.CONSUMER,
            ) as span:
                yield span

    def set_span_attributes(self, span: object, attributes: Mapping[str, object]) -> None:
        for key, value in attributes.items():
            if key in _SAFE_SPAN_FIELDS and _safe_log_value(key, value):
                span.set_attribute(key, value)

    def finish_span(
        self,
        span: object,
        *,
        outcome: str,
        error_category: str = "none",
        error_code: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        values = {"outcome": outcome, "error_category": error_category}
        if error_code is not None:
            values["error_code"] = error_code
        if attributes is not None:
            values.update(attributes)
        self.set_span_attributes(span, values)
        if error_category != "none":
            span.set_status(Status(StatusCode.ERROR))

    def record_knowledge_query(
        self, *, outcome: str, reason: str, duration_seconds: float
    ) -> None:
        if not self.metrics_enabled:
            return
        reason = reason if reason in _KNOWLEDGE_REASONS else "retrieval_failure"
        self._knowledge_queries.labels(outcome=outcome, reason=reason).inc()
        self._knowledge_duration.labels(outcome=outcome).observe(duration_seconds)

    def record_retrieval_stage(
        self, *, stage: str, outcome: str, duration_seconds: float
    ) -> None:
        if not self.metrics_enabled:
            return
        stage = stage if stage in _RETRIEVAL_STAGES else "authorization"
        self._retrieval_duration.labels(stage=stage, outcome=outcome).observe(duration_seconds)

    def record_retrieval_candidates(self, *, stage: str, count: int) -> None:
        if self.metrics_enabled and stage in {"dense", "sparse", "fused", "revalidated", "packed"}:
            self._retrieval_candidates.labels(stage=stage).observe(max(0, count))

    def record_context_token_estimate(self, count: int) -> None:
        if self.metrics_enabled:
            self._retrieval_context_tokens.observe(max(0, count))

    def record_llm(
        self,
        *,
        operation: str,
        provider: str,
        outcome: str,
        duration_seconds: float,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        if not self.metrics_enabled:
            return
        self._llm_calls.labels(operation=operation, provider=provider, outcome=outcome).inc()
        self._llm_duration.labels(
            operation=operation, provider=provider, outcome=outcome
        ).observe(duration_seconds)
        if input_tokens is None and output_tokens is None:
            self._llm_usage_missing.labels(operation=operation, provider=provider).inc()
            return
        if input_tokens is not None:
            self._llm_tokens.labels(
                operation=operation, provider=provider, direction="input"
            ).inc(input_tokens)
        if output_tokens is not None:
            self._llm_tokens.labels(
                operation=operation, provider=provider, direction="output"
            ).inc(output_tokens)

    def record_task_run(
        self,
        *,
        mode: str,
        outcome: str,
        error_category: str,
        duration_seconds: float,
    ) -> None:
        if not self.metrics_enabled:
            return
        self._task_runs.labels(
            mode=mode, outcome=outcome, error_category=_metric_error_category(error_category)
        ).inc()
        self._task_duration.labels(mode=mode, outcome=outcome).observe(duration_seconds)

    def record_checkpoint_write(self, *, outcome: str) -> None:
        if self.metrics_enabled:
            self._checkpoint_writes.labels(outcome=outcome).inc()

    def record_tool(
        self,
        *,
        tool: str,
        mode: str,
        outcome: str,
        error_category: str,
        duration_seconds: float,
    ) -> None:
        if not self.metrics_enabled:
            return
        self._tool_calls.labels(
            tool=tool,
            mode=mode,
            outcome=outcome,
            error_category=_metric_error_category(error_category),
        ).inc()
        self._tool_duration.labels(tool=tool, mode=mode, outcome=outcome).observe(
            duration_seconds
        )

    def record_approval(
        self, *, decision: str, outcome: str, wait_seconds: float | None = None
    ) -> None:
        if not self.metrics_enabled:
            return
        self._approvals.labels(decision=decision, outcome=outcome).inc()
        if wait_seconds is not None:
            self._approval_wait.labels(decision=decision).observe(max(0, wait_seconds))

    def record_ingestion_job(
        self,
        *,
        phase: str,
        outcome: str,
        error_category: str,
    ) -> None:
        if not self.metrics_enabled:
            return
        self._ingestion_jobs.labels(
            phase=_bounded_metric_value(phase, _INGESTION_JOB_PHASES, "ingest"),
            outcome=_bounded_metric_value(outcome, _INGESTION_OUTCOMES, "failed"),
            error_category=_metric_error_category(error_category),
        ).inc()

    def record_ingestion_phase(
        self, *, phase: str, outcome: str, duration_seconds: float
    ) -> None:
        if not self.metrics_enabled:
            return
        self._ingestion_phase_duration.labels(
            phase=_bounded_metric_value(phase, _INGESTION_PHASES, "parse"),
            outcome=_bounded_metric_value(outcome, _PHASE_OUTCOMES, "failure"),
        ).observe(max(0, duration_seconds))

    def record_projection_reconciliation(
        self, *, phase: str, outcome: str, duration_seconds: float
    ) -> None:
        if not self.metrics_enabled:
            return
        bounded_phase = _bounded_metric_value(phase, _PROJECTION_PHASES, "reconcile")
        bounded_outcome = _bounded_metric_value(
            outcome, _PROJECTION_OUTCOMES, "failure_recording_failed"
        )
        self._projection_reconciliations.labels(
            phase=bounded_phase, outcome=bounded_outcome
        ).inc()
        self._projection_reconcile_duration.labels(
            phase=bounded_phase, outcome=bounded_outcome
        ).observe(max(0, duration_seconds))

    def record_recovery_action(self, *, kind: str, outcome: str, count: int = 1) -> None:
        if not self.metrics_enabled or count <= 0:
            return
        bounded_kind = _bounded_metric_value(kind, _RECOVERY_KINDS, "queue_snapshot")
        bounded_outcome = _bounded_metric_value(outcome, _RECOVERY_OUTCOMES, "failure")
        self._recovery_actions.labels(kind=bounded_kind, outcome=bounded_outcome).inc(count)

    def record_security_boundary_violation(self, *, kind: str) -> None:
        if self.metrics_enabled and kind in _SECURITY_BOUNDARY_KINDS:
            self._security_boundary_violations.labels(kind=kind).inc()

    def set_task_queue_snapshot(
        self, *, mode: str, status: str, depth: int, oldest_age_seconds: float
    ) -> None:
        if not self.metrics_enabled:
            return
        bounded_mode = _bounded_metric_value(mode, _TASK_QUEUE_MODES, "research")
        bounded_status = _bounded_metric_value(status, _TASK_QUEUE_STATUSES, "planning")
        self._task_queue_depth.labels(mode=bounded_mode, status=bounded_status).set(max(0, depth))
        self._task_queue_oldest_age.labels(
            mode=bounded_mode, status=bounded_status
        ).set(max(0, oldest_age_seconds))

    def set_ingestion_queue_snapshot(
        self, *, status: str, depth: int, oldest_age_seconds: float
    ) -> None:
        if not self.metrics_enabled:
            return
        bounded_status = _bounded_metric_value(status, _INGESTION_QUEUE_STATUSES, "pending")
        self._ingestion_queue_depth.labels(status=bounded_status).set(max(0, depth))
        self._ingestion_queue_oldest_age.labels(status=bounded_status).set(
            max(0, oldest_age_seconds)
        )

    def set_worker_last_loop(self, *, worker_kind: str, timestamp_seconds: float) -> None:
        if self.metrics_enabled and worker_kind in _WORKER_KINDS:
            self._worker_last_loop.labels(worker_kind=worker_kind).set(
                max(0, timestamp_seconds)
            )

    def set_worker_capability_available(
        self, *, capability: str, protocol: str, available: bool
    ) -> None:
        if (
            self.metrics_enabled
            and capability == "task-agent"
            and protocol == "m2-fenced-v1"
        ):
            self._worker_capability_available.labels(
                capability=capability, protocol=protocol
            ).set(1 if available else 0)

    def record_lease_failure(
        self,
        *,
        kind: str,
        operation: str,
        error_code: str,
        parent_context: Context | None = None,
    ) -> None:
        with self.activate():
            with self.span(
                "lease.renew",
                context=parent_context,
                attributes={"kind": kind, "operation": operation},
            ) as span:
                category = error_category_for(error_code)
                self.finish_span(
                    span,
                    outcome="failure",
                    error_category="lease" if category == "none" else category,
                    error_code=error_code,
                )
                if self.metrics_enabled:
                    self._lease_failures.labels(kind=kind, operation=operation).inc()
                if "stale" in error_code or "fenc" in error_code:
                    self.record_security_boundary_violation(kind="fencing")
                log_event(
                    "lease.renew.failed",
                    level=logging.ERROR,
                    component="worker",
                    kind=kind,
                    operation=operation,
                    outcome="failure",
                    error_category="lease",
                    error_code=error_code,
                )

    def finish_http_span(
        self,
        span: object,
        *,
        route: str,
        method: str,
        status_code: int,
    ) -> None:
        span.update_name(f"HTTP {method} {route}")
        span.set_attribute("http.route", route)
        span.set_attribute("http.request.method", method)
        span.set_attribute("http.response.status_code", status_code)
        if status_code >= 500:
            span.set_status(Status(StatusCode.ERROR))

    def record_http(
        self,
        *,
        route: str,
        method: str,
        status_class: str,
        duration_seconds: float,
    ) -> None:
        if not self.metrics_enabled:
            return
        self._requests.labels(route=route, method=method, status_class=status_class).inc()
        self._duration.labels(route=route, method=method).observe(duration_seconds)

    def metrics_payload(self) -> bytes:
        return generate_latest(self.registry)

    @property
    def metrics_content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    @property
    def provider(self) -> TracerProvider | None:
        return self._provider

    def _configure_exporter(self, provider: TracerProvider) -> None:
        exporter_logger = logging.getLogger("opentelemetry.exporter.otlp")
        if not getattr(exporter_logger, "_rag_kb_suppressed", False):
            exporter_logger.addHandler(logging.NullHandler())
            exporter_logger.propagate = False
            exporter_logger._rag_kb_suppressed = True  # type: ignore[attr-defined]
        try:
            exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint, timeout=1)
            provider.add_span_processor(
                BatchSpanProcessor(
                    _SafeSpanExporter(exporter, self._warn_exporter_unavailable),
                    export_timeout_millis=1_000,
                )
            )
        except Exception:
            self._warn_exporter_unavailable()

    def _warn_exporter_unavailable(self) -> None:
        with self._lock:
            if self._exporter_warning_sent:
                return
            self._exporter_warning_sent = True
        log_event(
            "observability.exporter.unavailable",
            component="observability",
            operation="trace.export",
            outcome="failure",
        )


def _safe_log_value(field: str, value: object) -> bool:
    if value is None:
        return False
    if field == "duration_ms":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    if field in {"candidate_count", "context_token_estimate", "input_tokens", "output_tokens", "total_tokens"}:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if field in {"run_id", "task_id"}:
        return isinstance(value, str) and bool(_SAFE_ID_PATTERN.fullmatch(value))
    if field == "error_code":
        return isinstance(value, str) and bool(_SAFE_ERROR_CODE_PATTERN.fullmatch(value))
    if field == "worker_id":
        return isinstance(value, str) and 0 < len(value) <= 200 and "\n" not in value and "\r" not in value
    return isinstance(value, str) and len(value) <= 128 and "\n" not in value and "\r" not in value
