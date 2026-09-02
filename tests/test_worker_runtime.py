"""build_task_worker 组合根验收：capability 注册时序与关闭顺序。"""

from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_kb import observability as observability_module
from rag_kb import worker as worker_module
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.database.document_lifecycle import DocumentRegistryIngestionJobPort
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.observability import Observability


class _FakeCapability:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def start(self) -> None:
        self._events.append("capability.start")

    def close(self) -> None:
        self._events.append("capability.close")


def _telemetry() -> tuple[Observability, InMemorySpanExporter]:
    observability = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    observability.start()
    exporter = InMemorySpanExporter()
    assert observability.provider is not None
    observability.provider.add_span_processor(SimpleSpanProcessor(exporter))
    return observability, exporter


def _patch_build_chain(monkeypatch, events: list[str], *, agent_build_error: Exception | None = None):
    """把 build_task_worker 的重依赖全部替换为记录事件的桩。"""

    class _StubTools:
        def close(self) -> None:
            events.append("tools.close")

    class _StubConnection:
        def close(self) -> None:
            events.append("connection.close")

    class _StubSaver:
        serde = None
        conn = None
        lock = None

        def setup(self) -> None: ...

    monkeypatch.setattr(worker_module, "build_database", lambda _url: object())
    monkeypatch.setattr(worker_module, "LocalObjectStore", lambda _root: object())
    monkeypatch.setattr(
        worker_module, "build_enterprise_retriever", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        worker_module,
        "build_tool_registry",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(worker_module, "ApprovalService", lambda *_a, **_kw: object())
    monkeypatch.setattr(
        worker_module,
        "ToolExecutionService",
        lambda *_a, **_kw: _StubTools(),
    )
    monkeypatch.setattr(
        worker_module,
        "open_postgres_checkpointer",
        lambda _url: (_StubSaver(), _StubConnection()),
    )

    class _StubAgent:
        def get_state(self, _config):
            return type("Snapshot", (), {"interrupts": (), "values": {}, "next": ()})()

    if agent_build_error is not None:
        def fail_agent(*_args, **_kwargs):
            raise agent_build_error
    else:
        def fail_agent(*_args, **_kwargs):
            return _StubAgent()

    monkeypatch.setattr(
        worker_module,
        "_build_research_model",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "build_research_agent",
        fail_agent,
    )
    monkeypatch.setattr(worker_module, "DatabaseTaskRepository", lambda *_a, **_kw: object())
    monkeypatch.setattr(
        worker_module,
        "DatabaseTaskAuthContextPort",
        lambda _db: object(),
    )
    monkeypatch.setattr(worker_module, "AgentResultStore", lambda _o: object())
    monkeypatch.setattr(
        worker_module, "DatabaseTaskWorkerSettlement", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        worker_module,
        "_CapabilityLeaseHeartbeat",
        lambda repository, **_kwargs: _FakeCapability(events),
    )


def test_capability_starts_only_after_full_agent_build(monkeypatch) -> None:
    """默认预算 Agent 构建成功之前，capability.start 不得被调用。"""
    events: list[str] = []
    _patch_build_chain(monkeypatch, events)

    order: list[str] = []

    original_start = _FakeCapability.start

    def tracking_start(self) -> None:
        # build_task_worker 尾部才允许 start；此时尚无任何 close 事件。
        assert "tools.close" not in order and "connection.close" not in order
        original_start(self)
        order.append("capability.start")

    monkeypatch.setattr(_FakeCapability, "start", tracking_start)
    monkeypatch.setattr(
        worker_module,
        "build_research_tools",
        lambda tools: order.append("research_tools_built") or object(),
    )

    worker = worker_module.build_task_worker(worker_id="worker-a")
    assert worker is not None
    # start 只在完整 Agent 构建（含默认预算编译）成功之后发生一次。
    assert order == ["research_tools_built", "capability.start"]


def test_failed_agent_build_does_not_register_capability(monkeypatch) -> None:
    """Agent/checkpointer 构建失败时 capability 未注册，且资源按序清理。"""
    events: list[str] = []
    _patch_build_chain(
        monkeypatch,
        events,
        agent_build_error=worker_module.TaskAgentConfigurationError("task_agent_api_key_missing"),
    )
    try:
        worker_module.build_task_worker(worker_id="worker-a")
    except worker_module.TaskAgentConfigurationError:
        pass
    else:
        raise AssertionError("expected TaskAgentConfigurationError")

    assert events == ["tools.close", "connection.close"]


def test_close_order_stops_capability_then_tools_then_connection(monkeypatch) -> None:
    """close 顺序：停止接单 -> tools.close -> capability.close -> connection.close。"""
    events: list[str] = []
    _patch_build_chain(monkeypatch, events)

    started: list[bool] = []

    original_start = _FakeCapability.start

    def tracking_start(self) -> None:
        original_start(self)
        started.append(True)

    monkeypatch.setattr(_FakeCapability, "start", tracking_start)

    worker = worker_module.build_task_worker(worker_id="worker-a")
    assert started == [True]
    capability_closed_via_worker_close = events == ["capability.start"]

    worker.close()

    # TaskWorker.close 先关 tools 再执行外部 close（capability + connection）。
    assert capability_closed_via_worker_close
    expected = [
        "capability.start",
        "tools.close",
        "capability.close",
        "connection.close",
    ]
    assert events == expected


@pytest.mark.parametrize(
    ("queue", "expected_builders", "expected_events"),
    (
        (
            "task",
            {"task_worker", "task_retention"},
            ("task.maintenance", "task.snapshot", "task.run_once", "last_loop:task"),
        ),
        (
            "ingestion",
            {"ingestion_worker", "upload_reconciler"},
            (
                "ingestion.recovery",
                "ingestion.snapshot",
                "ingestion.run_once",
                "last_loop:ingestion",
            ),
        ),
        (
            "projection",
            {"projection_reconciler"},
            ("projection.run_once", "last_loop:projection"),
        ),
    ),
)
def test_run_constructs_only_the_selected_queue_runtime(
    monkeypatch, queue: str, expected_builders: set[str], expected_events: tuple[str, ...]
) -> None:
    """每种队列只调用其拥有的维护、快照和运行组合根。"""

    events: list[str] = []

    class Telemetry:
        metrics_enabled = True

        def set_worker_last_loop(self, *, worker_kind: str, timestamp_seconds: float) -> None:
            events.append(f"last_loop:{worker_kind}")

        def record_worker_error(self, **_kwargs) -> None:
            events.append("worker.error")

    class Runtime:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.max_attempts = 3

        def run_once(self):
            events.append(f"{self.kind}.run_once")
            return SimpleNamespace(status="idle", error_code=None)

        def close(self) -> None:
            events.append(f"{self.kind}.close")

    built: set[str] = set()
    snapshot_builders: list[str] = []

    def build(kind: str) -> Runtime:
        built.add(kind)
        return Runtime(kind.removesuffix("_worker").removesuffix("_reconciler"))

    def build_snapshot(kind: str) -> object:
        snapshot_builders.append(kind)
        return object()

    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: SimpleNamespace(
            task_agent_capability_name="task-agent",
            task_agent_protocol_version="m2-fenced-v1",
        ),
    )
    monkeypatch.setattr(worker_module, "current_observability", Telemetry)
    monkeypatch.setattr(
        worker_module,
        "_run_task_maintenance",
        lambda **_kwargs: events.append("task.maintenance"),
    )
    monkeypatch.setattr(
        worker_module,
        "_run_ingestion_recovery",
        lambda **_kwargs: events.append("ingestion.recovery"),
    )
    monkeypatch.setattr(
        worker_module,
        "_record_task_queue_snapshot",
        lambda **_kwargs: events.append("task.snapshot"),
    )
    monkeypatch.setattr(
        worker_module,
        "_record_ingestion_queue_snapshot",
        lambda **_kwargs: events.append("ingestion.snapshot"),
    )
    monkeypatch.setattr(worker_module, "build_task_worker", lambda: build("task_worker"))
    monkeypatch.setattr(
        worker_module,
        "build_task_retention",
        lambda: build("task_retention"),
    )
    monkeypatch.setattr(
        worker_module,
        "build_ingestion_worker",
        lambda: build("ingestion_worker"),
    )
    monkeypatch.setattr(
        worker_module,
        "build_upload_reconciler",
        lambda: build("upload_reconciler"),
    )
    monkeypatch.setattr(
        worker_module,
        "build_projection_reconciler",
        lambda: build("projection_reconciler"),
    )
    monkeypatch.setattr(
        worker_module,
        "_build_task_queue_snapshot_port",
        lambda: build_snapshot("task"),
    )
    monkeypatch.setattr(
        worker_module,
        "_build_ingestion_queue_snapshot_port",
        lambda: build_snapshot("ingestion"),
    )

    assert worker_module.run(
        queue=queue, watch=False, poll_seconds=1.0, max_jobs=1
    ) == 0

    assert built == expected_builders
    assert snapshot_builders == ([] if queue == "projection" else [queue])
    assert events[: len(expected_events)] == list(expected_events)
    assert "worker.error" not in events


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("--queue", "unknown"),
        ("--include-tasks",),
    ),
)
def test_worker_cli_rejects_missing_invalid_and_legacy_queue_options(argv) -> None:
    with pytest.raises(SystemExit) as exc_info:
        worker_module._parse_args(list(argv))

    assert exc_info.value.code == 2


def test_worker_cli_parses_an_explicit_single_queue() -> None:
    args = worker_module._parse_args(
        ["--queue", "projection", "--watch", "--max-jobs", "4"]
    )

    assert args.queue == "projection"
    assert args.watch is True
    assert args.max_jobs == 4


@pytest.mark.parametrize(
    ("queue", "status", "error_code", "expects_error"),
    (
        ("task", "waiting_for_user", None, False),
        ("task", "failed", "task_deadline_expired", True),
        ("ingestion", "succeeded", None, False),
        ("ingestion", "retry_scheduled", "ingestion_retry_exhausted", True),
        ("projection", "succeeded", None, False),
        ("projection", "retry_scheduled", "projection_reconcile_failed", True),
    ),
)
def test_run_records_only_explicit_result_errors_for_the_selected_queue(
    monkeypatch, queue: str, status: str, error_code: str | None, expects_error: bool
) -> None:
    class Telemetry:
        metrics_enabled = False

        def __init__(self) -> None:
            self.last_loops: list[str] = []
            self.errors: list[tuple[str, str, str]] = []

        def set_worker_last_loop(self, *, worker_kind: str, timestamp_seconds: float) -> None:
            self.last_loops.append(worker_kind)

        def record_worker_error(
            self, *, worker_kind: str, operation: str, error_category: str
        ) -> None:
            self.errors.append((worker_kind, operation, error_category))

    class Runtime:
        max_attempts = 3

        def run_once(self):
            return SimpleNamespace(status=status, error_code=error_code)

        def close(self) -> None:
            return None

    telemetry = Telemetry()
    runtime = Runtime()
    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: SimpleNamespace(
            task_agent_capability_name="task-agent",
            task_agent_protocol_version="m2-fenced-v1",
        ),
    )
    monkeypatch.setattr(worker_module, "current_observability", lambda: telemetry)
    monkeypatch.setattr(worker_module, "build_task_worker", lambda: runtime)
    monkeypatch.setattr(worker_module, "build_task_retention", lambda: Runtime())
    monkeypatch.setattr(worker_module, "build_ingestion_worker", lambda: runtime)
    monkeypatch.setattr(worker_module, "build_upload_reconciler", object)
    monkeypatch.setattr(worker_module, "build_projection_reconciler", lambda: runtime)
    monkeypatch.setattr(worker_module, "_run_task_maintenance", lambda **_kwargs: None)
    monkeypatch.setattr(worker_module, "_run_ingestion_recovery", lambda **_kwargs: None)

    assert worker_module.run(
        queue=queue, watch=False, poll_seconds=1.0, max_jobs=1
    ) == 0

    assert telemetry.last_loops == [queue]
    if expects_error:
        assert telemetry.errors == [
            (queue, "run_once", worker_module.error_category_for(error_code))
        ]
    else:
        assert telemetry.errors == []


@pytest.mark.parametrize("queue", ("task", "ingestion", "projection"))
def test_main_uses_a_queue_scoped_service_name(monkeypatch, queue: str) -> None:
    telemetry_services: list[str] = []
    logging_services: list[str] = []
    run_calls: list[dict[str, object]] = []

    class Telemetry:
        def __init__(self, *, service_name: str, **_kwargs) -> None:
            telemetry_services.append(service_name)

        def start(self) -> None:
            return None

        def activate(self):
            return nullcontext(self)

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(
        worker_module,
        "_parse_args",
        lambda: SimpleNamespace(queue=queue, watch=False, poll_seconds=2.0, max_jobs=1),
    )
    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: SimpleNamespace(
            observability_enabled=False,
            metrics_enabled=False,
            worker_metrics_port=9464,
            log_level="INFO",
            otel_exporter_otlp_endpoint="",
            otel_trace_sample_ratio=1.0,
        ),
    )
    monkeypatch.setattr(worker_module, "Observability", Telemetry)
    monkeypatch.setattr(
        observability_module,
        "configure_json_logging",
        lambda *, service, level: logging_services.append(f"{service}:{level}"),
    )
    monkeypatch.setattr(
        worker_module,
        "run",
        lambda **kwargs: run_calls.append(kwargs) or 0,
    )

    assert worker_module.main() == 0

    expected_service = f"rag-kb-worker-{queue}"
    assert telemetry_services == [expected_service]
    assert logging_services == [f"{expected_service}:INFO"]
    assert run_calls == [
        {"queue": queue, "watch": False, "poll_seconds": 2.0, "max_jobs": 1}
    ]


def test_worker_metrics_server_is_idempotent_and_uses_its_own_registry() -> None:
    observability, _exporter = _telemetry()
    server = worker_module._WorkerMetricsServer(observability, port=0)
    try:
        observability.set_worker_last_loop(worker_kind="ingestion", timestamp_seconds=123.0)
        assert server.start() is True
        assert server.start() is True
        with urlopen(f"http://127.0.0.1:{server.port}/metrics", timeout=2) as response:
            payload = response.read().decode()
    finally:
        server.close()
        server.close()
        observability.shutdown()

    assert "rag_kb_worker_last_loop_timestamp_seconds" in payload
    assert 'worker_kind="ingestion"' in payload


def test_owned_maintenance_and_snapshots_record_fixed_worker_kind_metrics() -> None:
    calls: list[str] = []

    class TaskWorker:
        def expire_stale_tool_calls(self, *, limit: int) -> int:
            assert limit == 25
            calls.append("task.stale_tool_call")
            return 2

        def expire_stale_knowledge_queries(self, *, limit: int) -> int:
            assert limit == 25
            calls.append("task.stale_knowledge_query")
            return 1

    class Retention:
        def expire_waiting_tasks(self, *, limit: int) -> int:
            assert limit == 25
            calls.append("task.waiting_task")
            return 1

        def purge_expired_tool_results(self, *, limit: int) -> int:
            assert limit == 25
            calls.append("task.tool_result_retention")
            return 1

        def purge_finished_task_content(self, *, limit: int) -> int:
            assert limit == 25
            calls.append("task.task_content_retention")
            return 1

        def purge_orphaned_result_objects(self, *, limit: int) -> int:
            assert limit == 25
            calls.append("task.orphan_result")
            return 1

    class UploadReconciler:
        def reconcile(self, **kwargs):
            assert kwargs["limit"] == 25
            assert kwargs["time_budget_seconds"] == 1.0
            calls.append("ingestion.upload_reconciliation")
            return SimpleNamespace(
                recovered=1,
                missing_objects=0,
                content_mismatches=1,
                failed=0,
            )

    class TaskQueue:
        def queue_snapshot(self, **kwargs):
            assert kwargs == {
                "capability_name": "task-agent",
                "protocol_version": "m2-fenced-v1",
                "limit": 1_000,
            }
            calls.append("task.queue_snapshot")
            return (
                SimpleNamespace(
                    task_mode="research",
                    status="planning",
                    depth=2,
                    oldest_age_seconds=12.5,
                ),
            )

        def has_active_capability(self, **kwargs) -> bool:
            assert kwargs == {
                "capability_name": "task-agent",
                "protocol_version": "m2-fenced-v1",
            }
            calls.append("task.capability_availability")
            return False

    class IngestionQueue:
        def queue_snapshot(self, **kwargs):
            assert kwargs == {"max_attempts": 3, "limit": 1_000}
            calls.append("ingestion.queue_snapshot")
            return (
                SimpleNamespace(status="pending", depth=3, oldest_age_seconds=7.5),
            )

    observability, exporter = _telemetry()
    try:
        with observability.activate():
            worker_module._run_task_maintenance(
                task_worker=TaskWorker(),
                task_retention=Retention(),
                telemetry=observability,
            )
            worker_module._record_task_queue_snapshot(
                task_queue=TaskQueue(),
                telemetry=observability,
                capability_name="task-agent",
                protocol_version="m2-fenced-v1",
            )
            worker_module._run_ingestion_recovery(
                upload_reconciler=UploadReconciler(),
                telemetry=observability,
            )
            worker_module._record_ingestion_queue_snapshot(
                ingestion_queue=IngestionQueue(),
                telemetry=observability,
                ingestion_max_attempts=3,
            )
        payload = observability.metrics_payload().decode()
        span_names = {span.name for span in exporter.get_finished_spans()}
    finally:
        observability.shutdown()

    assert calls == [
        "task.stale_tool_call",
        "task.stale_knowledge_query",
        "task.waiting_task",
        "task.tool_result_retention",
        "task.task_content_retention",
        "task.orphan_result",
        "task.queue_snapshot",
        "task.capability_availability",
        "ingestion.upload_reconciliation",
        "ingestion.queue_snapshot",
    ]
    assert {
        "worker.task.maintenance",
        "worker.task.queue_snapshot",
        "worker.ingestion.maintenance",
        "worker.ingestion.queue_snapshot",
    } <= span_names
    assert "worker.recovery" not in span_names
    assert 'kind="stale_tool_call",outcome="succeeded",worker_kind="task"' in payload
    assert 'kind="upload_reservation",outcome="recovered",worker_kind="ingestion"' in payload
    assert 'kind="content_hash"' in payload
    assert 'mode="research",status="planning",worker_kind="task"' in payload
    assert 'status="pending",worker_kind="ingestion"' in payload
    assert 'capability="task-agent",protocol="m2-fenced-v1",worker_kind="task"' in payload


def test_owned_maintenance_errors_are_recorded_by_worker_kind() -> None:
    class StorageFailure(RuntimeError):
        code = "object_storage_failed"

    task_calls: list[str] = []

    class TaskWorker:
        def expire_stale_tool_calls(self, *, limit: int) -> int:
            assert limit == 25
            task_calls.append("stale_tool_call")
            raise StorageFailure()

        def expire_stale_knowledge_queries(self, *, limit: int) -> int:
            assert limit == 25
            task_calls.append("stale_knowledge_query")
            return 0

    class Retention:
        def expire_waiting_tasks(self, *, limit: int) -> int:
            assert limit == 25
            task_calls.append("waiting_task")
            return 0

        def purge_expired_tool_results(self, *, limit: int) -> int:
            assert limit == 25
            task_calls.append("tool_result_retention")
            return 0

        def purge_finished_task_content(self, *, limit: int) -> int:
            assert limit == 25
            task_calls.append("task_content_retention")
            return 0

        def purge_orphaned_result_objects(self, *, limit: int) -> int:
            assert limit == 25
            task_calls.append("orphan_result")
            return 0

    class UploadReconciler:
        def reconcile(self, **kwargs):
            assert kwargs["limit"] == 25
            raise StorageFailure()

    observability, _exporter = _telemetry()
    try:
        with observability.activate():
            worker_module._run_task_maintenance(
                task_worker=TaskWorker(),
                task_retention=Retention(),
                telemetry=observability,
            )
            worker_module._run_ingestion_recovery(
                upload_reconciler=UploadReconciler(),
                telemetry=observability,
            )
        payload = observability.metrics_payload().decode()
    finally:
        observability.shutdown()

    assert task_calls == [
        "stale_tool_call",
        "stale_knowledge_query",
        "waiting_task",
        "tool_result_retention",
        "task_content_retention",
        "orphan_result",
    ]
    assert 'kind="stale_tool_call",outcome="failure",worker_kind="task"' in payload
    assert 'kind="upload_reservation",outcome="failure",worker_kind="ingestion"' in payload
    assert 'error_category="storage",operation="maintenance",worker_kind="task"' in payload
    assert 'error_category="storage",operation="maintenance",worker_kind="ingestion"' in payload


def test_worker_metric_labels_are_bounded_to_fixed_values() -> None:
    observability = Observability(
        enabled=False,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    try:
        observability.record_recovery_action(
            worker_kind="tenant-a",
            kind="task-123",
            outcome="private-error",
        )
        observability.record_worker_error(
            worker_kind="tenant-a",
            operation="task-123",
            error_category="private-error",
        )
        observability.set_task_queue_snapshot(
            worker_kind="tenant-a",
            mode="private-mode",
            status="private-status",
            depth=1,
            oldest_age_seconds=1.0,
        )
        observability.set_ingestion_queue_snapshot(
            worker_kind="tenant-a",
            status="private-status",
            depth=1,
            oldest_age_seconds=1.0,
        )
        observability.set_worker_capability_available(
            worker_kind="tenant-a",
            capability="task-agent",
            protocol="m2-fenced-v1",
            available=True,
        )
        payload = observability.metrics_payload().decode()
    finally:
        observability.shutdown()

    for unbounded_value in ("tenant-a", "task-123", "private-error", "private-mode", "private-status"):
        assert unbounded_value not in payload
    assert 'error_category="unknown",operation="run_once",worker_kind="unknown"' in payload
    assert 'kind="queue_snapshot",outcome="failure",worker_kind="unknown"' in payload
    assert 'mode="research",status="planning",worker_kind="unknown"' in payload
    assert 'status="pending",worker_kind="unknown"' in payload


def test_queue_snapshots_keep_claimable_and_waiting_definitions_separate(tmp_path) -> None:
    now = datetime(2026, 8, 29, 8, 0, tzinfo=UTC)
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'queues.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="Tenant A"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="owner-a",
                roles=[],
                attributes={},
            )
        )
        session.flush()
        session.add(
            DocumentRow(
                tenant_id="tenant-a",
                document_id="document-a",
                title="Queue test",
                department="engineering",
                status="pending",
                created_by="owner-a",
            )
        )
        session.flush()
        for number in range(1, 6):
            session.add(
                DocumentVersionRow(
                    version_id=f"version-{number}",
                    tenant_id="tenant-a",
                    document_id="document-a",
                    version=number,
                    lifecycle_status="pending",
                    source_object_key=f"tenant-a/document-a/version-{number}/source.md",
                    source_file_name="source.md",
                    content_hash=f"sha256:{number}",
                    index_version="kb-v1",
                )
            )
        session.flush()
        session.add_all(
            [
                IngestionJobRow(
                    job_id="awaiting",
                    tenant_id="tenant-a",
                    version_id="version-1",
                    status="awaiting_object",
                    available_at=now - timedelta(seconds=30),
                ),
                IngestionJobRow(
                    job_id="claimable",
                    tenant_id="tenant-a",
                    version_id="version-2",
                    status="pending",
                    attempts=0,
                    available_at=now - timedelta(seconds=20),
                ),
                IngestionJobRow(
                    job_id="cleanup",
                    tenant_id="tenant-a",
                    version_id="version-3",
                    status="pending",
                    attempts=3,
                    error_code="ingestion_projection_cleanup_failed",
                    available_at=now - timedelta(seconds=10),
                ),
                IngestionJobRow(
                    job_id="exhausted",
                    tenant_id="tenant-a",
                    version_id="version-4",
                    status="pending",
                    attempts=3,
                    available_at=now - timedelta(seconds=5),
                ),
                IngestionJobRow(
                    job_id="running",
                    tenant_id="tenant-a",
                    version_id="version-5",
                    status="running",
                    available_at=now - timedelta(seconds=15),
                ),
            ]
        )
        session.add_all(
            [
                AgentTaskRow(
                    task_id="planning-ready",
                    tenant_id="tenant-a",
                    user_id="owner-a",
                    task_mode=TaskMode.RESEARCH.value,
                    status=TaskStatus.PLANNING.value,
                    goal="safe",
                    auth_context_ref="auth",
                    budgets={},
                    deadline=now + timedelta(minutes=5),
                    available_at=now - timedelta(seconds=20),
                    required_capability_name="task-agent",
                    required_protocol_version="m2-fenced-v1",
                ),
                AgentTaskRow(
                    task_id="planning-delayed",
                    tenant_id="tenant-a",
                    user_id="owner-a",
                    task_mode=TaskMode.RESEARCH.value,
                    status=TaskStatus.PLANNING.value,
                    goal="safe",
                    auth_context_ref="auth",
                    budgets={},
                    deadline=now + timedelta(minutes=5),
                    available_at=now + timedelta(seconds=20),
                    required_capability_name="task-agent",
                    required_protocol_version="m2-fenced-v1",
                ),
                AgentTaskRow(
                    task_id="running-expired",
                    tenant_id="tenant-a",
                    user_id="owner-a",
                    task_mode=TaskMode.RESEARCH.value,
                    status=TaskStatus.RUNNING.value,
                    goal="safe",
                    auth_context_ref="auth",
                    budgets={},
                    deadline=now + timedelta(minutes=5),
                    available_at=now - timedelta(seconds=15),
                    lease_owner="worker-a",
                    lease_expires_at=now - timedelta(seconds=1),
                    required_capability_name="task-agent",
                    required_protocol_version="m2-fenced-v1",
                ),
                AgentTaskRow(
                    task_id="running-active",
                    tenant_id="tenant-a",
                    user_id="owner-a",
                    task_mode=TaskMode.RESEARCH.value,
                    status=TaskStatus.RUNNING.value,
                    goal="safe",
                    auth_context_ref="auth",
                    budgets={},
                    deadline=now + timedelta(minutes=5),
                    available_at=now - timedelta(seconds=15),
                    lease_owner="worker-a",
                    lease_expires_at=now + timedelta(seconds=30),
                    required_capability_name="task-agent",
                    required_protocol_version="m2-fenced-v1",
                ),
                AgentTaskRow(
                    task_id="waiting",
                    tenant_id="tenant-a",
                    user_id="owner-a",
                    task_mode=TaskMode.RESEARCH.value,
                    status=TaskStatus.WAITING_FOR_USER.value,
                    goal="safe",
                    auth_context_ref="auth",
                    budgets={},
                    deadline=now + timedelta(minutes=5),
                    available_at=now - timedelta(seconds=10),
                    required_capability_name="task-agent",
                    required_protocol_version="m2-fenced-v1",
                    interrupt_kind="user_clarification",
                    interrupt_summary="need input",
                ),
            ]
        )

    task_snapshots = DatabaseTaskRepository(
        database,
        now=lambda: now,
        task_capability_name="task-agent",
        task_capability_protocol_version="m2-fenced-v1",
    ).queue_snapshot(
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
        limit=100,
    )
    ingestion_snapshots = DocumentRegistryIngestionJobPort(
        database,
        now=lambda: now,
    ).queue_snapshot(max_attempts=3, limit=100)

    task_depths = {(item.task_mode, item.status): item.depth for item in task_snapshots}
    ingestion_depths = {item.status: item.depth for item in ingestion_snapshots}
    assert task_depths[("research", "planning")] == 1
    assert task_depths[("research", "running")] == 1
    assert task_depths[("research", "waiting_for_user")] == 1
    assert ingestion_depths == {"awaiting_object": 1, "pending": 2, "running": 1}
