"""build_task_worker 组合根验收：capability 注册时序与关闭顺序。"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.request import urlopen

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.application.ingestion_worker import DocumentRegistryIngestionJobPort
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
    monkeypatch.setattr(worker_module, "BgeM3Embedder", lambda: object())
    monkeypatch.setattr(
        worker_module,
        "QdrantAuthorizedSearch",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(worker_module, "resolve_retrieval_limits", lambda _s: (5, 5))
    monkeypatch.setattr(
        worker_module,
        "EnterpriseRetriever",
        lambda *_args, **_kwargs: object(),
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
    monkeypatch.setattr(worker_module, "EvidenceLedgerRepository", lambda: object())
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


def test_recovery_window_records_bounded_queue_and_safe_summary_metrics() -> None:
    class TaskWorker:
        def expire_stale_tool_calls(self, *, limit: int) -> int:
            assert limit == 25
            return 2

        def expire_stale_knowledge_queries(self, *, limit: int) -> int:
            assert limit == 25
            return 1

    class Retention:
        def expire_waiting_tasks(self, *, limit: int) -> int:
            assert limit == 25
            return 1

        def purge_expired_tool_results(self, *, limit: int) -> int:
            assert limit == 25
            return 1

        def purge_finished_task_content(self, *, limit: int) -> int:
            assert limit == 25
            return 1

        def purge_orphaned_result_objects(self, *, limit: int) -> int:
            assert limit == 25
            return 1

    class UploadReconciler:
        def reconcile(self, **kwargs):
            assert kwargs["limit"] == 25
            assert kwargs["time_budget_seconds"] == 1.0
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
            return False

    class IngestionQueue:
        def queue_snapshot(self, **kwargs):
            assert kwargs == {"max_attempts": 3, "limit": 1_000}
            return (
                SimpleNamespace(status="pending", depth=3, oldest_age_seconds=7.5),
            )

    observability, exporter = _telemetry()
    try:
        with observability.activate():
            worker_module._run_recovery_window(
                task_worker=TaskWorker(),
                task_retention=Retention(),
                upload_reconciler=UploadReconciler(),
                task_queue=TaskQueue(),
                ingestion_queue=IngestionQueue(),
                telemetry=observability,
                capability_name="task-agent",
                protocol_version="m2-fenced-v1",
                ingestion_max_attempts=3,
            )
        payload = observability.metrics_payload().decode()
        spans = exporter.get_finished_spans()
    finally:
        observability.shutdown()

    assert "worker.recovery" in {span.name for span in spans}
    assert 'kind="stale_tool_call",outcome="succeeded"' in payload
    assert 'kind="content_hash"' in payload
    assert 'mode="research",status="planning"' in payload
    assert 'status="pending"' in payload
    assert 'capability="task-agent",protocol="m2-fenced-v1"' in payload


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
