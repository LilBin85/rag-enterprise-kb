"""Stage 5 批准绑定、持久执行、幂等与 UNKNOWN 回归。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import select, text, update

from rag_kb.application.approval_service import (
    ApprovalCheckpointBinding,
    ApprovalService,
    ApprovalServiceError,
    arguments_fingerprint,
    principal_fingerprint,
)
from rag_kb.application.tool_execution_service import ToolExecutionError, ToolExecutionRequest, ToolExecutionService
from rag_kb.application.tool_registry import ToolRegistry, ToolRegistryError, builtin_tool_definitions
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tasks import TaskBudgets, TaskMode
from rag_kb.domain.tools import ApprovalPolicy, ToolMode
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.approval_lifecycle import (
    DatabaseApprovalLifecycle,
)
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentTaskRow,
    ApprovalRow,
    ArtifactRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PlanRevisionRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskLease
from rag_kb.infrastructure.database.tool_execution_lifecycle import (
    DatabaseToolExecutionLifecycle,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.observability import Observability
from rag_kb.infrastructure.tools.artifacts import LocalArtifactToolAdapter


class _VerifiedAdapter:
    def execute(self, arguments, context):
        return {"artifact_type": "report", "artifact_id": "artifact-" + context.tool_call_id[:12], "content_hash": "sha256:" + "a" * 64, "status": "ready"}

    def verify(self, result, context):
        return True


class _CrashingAdapter(_VerifiedAdapter):
    def execute(self, arguments, context):
        raise RuntimeError("side effect outcome unavailable")


class _CountingAdapter(_VerifiedAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, arguments, context):
        self.calls += 1
        return super().execute(arguments, context)


class _FailingObjectStore:
    def put_exclusive(self, storage_key: str, content: bytes) -> None:
        raise RuntimeError("object write unavailable")

    def read(self, storage_key: str) -> bytes:
        raise OSError("object absent")


class _BlockingObjectStore:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.content: bytes | None = None

    def put_exclusive(self, storage_key: str, content: bytes) -> None:
        self.content = content
        self.entered.set()
        assert self.release.wait(timeout=5), "blocking store was not released"

    def read(self, storage_key: str) -> bytes:
        assert self.content is not None
        return self.content


class _SignalingAdapter:
    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.finished = Event()

    def execute(self, arguments, context):
        try:
            return self.adapter.execute(arguments, context)
        finally:
            self.finished.set()

    def verify(self, result, context):
        return self.adapter.verify(result, context)


@pytest.fixture()
def stage5(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'stage5.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    requester = AuthContext(tenant_id="tenant-a", user_id="requester", attributes={"permissions": ["artifact:write", "knowledge:read", "document:read"]}, request_id="request-1")
    approver = AuthContext(tenant_id="tenant-a", user_id="approver", attributes={"permissions": ["task:approve"]}, request_id="approve-1")
    with database.engine.begin() as connection:
        connection.execute(text("INSERT INTO tenants VALUES ('tenant-a', 'Tenant A', 1, CURRENT_TIMESTAMP)"))
        connection.execute(
            text("INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) VALUES ('tenant-a', 'requester', '[]', :attributes, 1, CURRENT_TIMESTAMP)"),
            {"attributes": '{"permissions":["artifact:write","knowledge:read","document:read"]}'},
        )
        connection.execute(
            text("INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) VALUES ('tenant-a', 'approver', '[]', :attributes, 1, CURRENT_TIMESTAMP)"),
            {"attributes": '{"permissions":["task:approve"]}'},
        )
    DatabaseTaskRepository(database).create_task(auth=requester, goal="create report", task_mode=TaskMode.CONTROLLED_ACTION, task_id="task-1")
    snapshot = canonical_evidence_snapshot(("evidence-1",), require_non_empty=True)
    plan = {"steps": ["report"], "evidence_ids": list(snapshot.evidence_ids)}
    plan_digest = canonical_fingerprint("task_plan", plan)
    with UnitOfWork(database) as work:
        assert work.session is not None
        task = work.session.scalar(select(AgentTaskRow).where(AgentTaskRow.task_id == "task-1"))
        assert task is not None
        task.status = "running"
        task.lease_owner = "worker-1"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=60)
        task.lease_generation = 1
        work.session.add_all(
            [
                DocumentRow(tenant_id="tenant-a", document_id="doc-1", title="制度", department="安全", status="active", created_by="requester", projection_generation=1, applied_projection_generation=1, applied_projection_version_id="version-1"),
                ACLGrantRow(grant_id="grant-1", tenant_id="tenant-a", resource_type="document", resource_id="doc-1", subject_type="user", subject_id="requester", permissions=["knowledge:read", "document:read"]),
            ]
        )
        work.session.flush()
        work.session.add(DocumentVersionRow(version_id="version-1", tenant_id="tenant-a", document_id="doc-1", version=1, lifecycle_status="active", source_object_key="doc-1", source_file_name="doc.md", content_hash="sha256:content", index_version="kb-v1", desired_projection_generation=1, applied_projection_generation=1))
        work.session.flush()
        work.session.add_all(
            [
                ChunkMetadataRow(chunk_id="chunk-1", version_id="version-1", tenant_id="tenant-a", document_id="doc-1", document_version=1, content_hash="sha256:chunk", source_locator="doc/1", source_file_name="doc.md", index_version="kb-v1"),
                IngestionJobRow(job_id="job-1", tenant_id="tenant-a", version_id="version-1", status="succeeded", indexed_chunk_count=1, index_manifest_digest="sha256:manifest", index_completed_at=datetime.now(UTC), projection_generation=1, projection_lease_generation=1),
            ]
        )
        work.session.flush()
        work.session.add_all(
            [
                EvidenceRecordRow(evidence_id="evidence-1", task_id="task-1", query_fingerprint="sha256:query", tenant_id="tenant-a", document_id="doc-1", version_id="version-1", document_version=1, chunk_id="chunk-1", content_hash="sha256:chunk", source_locator="doc/1", permission_granted=True, supported_claim_ids=[]),
                PlanRevisionRow(plan_revision_id="plan-1", task_id="task-1", tenant_id="tenant-a", revision=1, schema_version=1, plan=plan, plan_digest=plan_digest, evidence_digest=snapshot.digest, status="ready"),
            ]
        )
    lease = TaskLease(
        DatabaseTaskRepository(database).get_task(auth=requester, task_id="task-1"),
        "worker-1",
        1,
        TaskBudgets(),
    )
    yield database, requester, approver, tmp_path, lease
    database.engine.dispose()


def _arguments(content: str = "body") -> dict[str, object]:
    return {"title": "report", "content": content, "format": "markdown"}


def _approved(database, requester, approver, arguments=None):
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    binding = service.request(auth=requester, task_id="task-1", plan_revision=1, plan_digest=_binding(database)[0], evidence_digest=_binding(database)[1], evidence_ids=("evidence-1",), tool_name="create_report_artifact", tool_version="1.0.0", arguments=arguments or _arguments(), expires_at=datetime.now(UTC) + timedelta(minutes=5))
    return service, service.approve(auth=approver, approval_id=binding.approval_id)


def _request(requester, approval_id, lease, arguments=None, key="key-1"):
    plan_digest, evidence_digest = _binding(lease.database if hasattr(lease, "database") else None)
    return ToolExecutionRequest(auth=requester, task_id="task-1", plan_revision=1, plan_digest=plan_digest, evidence_digest=evidence_digest, evidence_ids=("evidence-1",), tool_name="create_report_artifact", arguments=arguments or _arguments(), worker_id=lease.worker_id, lease_generation=lease.lease_generation, idempotency_key=key, approval_id=approval_id)


def _binding(database):
    if database is None:
        snapshot = canonical_evidence_snapshot(("evidence-1",), require_non_empty=True)
        plan = {"steps": ["report"], "evidence_ids": ["evidence-1"]}
        return canonical_fingerprint("task_plan", plan), snapshot.digest
    with database.sessions() as session:
        row = session.scalar(select(PlanRevisionRow).where(PlanRevisionRow.task_id == "task-1"))
        assert row is not None
        return row.plan_digest, row.evidence_digest


def _executor(database, registry, approvals) -> ToolExecutionService:
    return ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
        registry,
        approvals,
        tool_results=LocalObjectStore(Path(database.engine.url.database).parent / "tool-results"),
    )


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


def test_any_approval_binding_change_invalidates_execution(stage5) -> None:
    database, requester, approver, _, _lease = stage5
    service, binding = _approved(database, requester, approver)
    adapter = _CountingAdapter()
    executor = _executor(
        database,
        ToolRegistry(adapters={"create_report_artifact": adapter}),
        service,
    )
    changes = (
        {"task_id": "other-task"},
        {"plan_revision": 2},
        {"plan_digest": "sha256:other"},
        {"evidence_digest": "sha256:other"},
        {"arguments": _arguments("other")},
    )
    for change in changes:
        plan_digest, evidence_digest = _binding(database)
        request = ToolExecutionRequest(
            auth=requester,
            approval_id=binding.approval_id,
            task_id="task-1",
            plan_revision=1,
            plan_digest=plan_digest,
            evidence_digest=evidence_digest,
            evidence_ids=("evidence-1",),
            tool_name="create_report_artifact",
            arguments=_arguments(),
            worker_id="worker-1",
            lease_generation=1,
            idempotency_key="binding-check",
        )
        with pytest.raises(ToolExecutionError):
            executor.execute(replace(request, **change))
    assert adapter.calls == 0


def test_approval_requires_authorized_approver_and_write_is_idempotent(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    plan_digest, evidence_digest = _binding(database)
    pending = service.request(auth=requester, task_id="task-1", plan_revision=1, plan_digest=plan_digest, evidence_digest=evidence_digest, evidence_ids=("evidence-1",), tool_name="create_report_artifact", tool_version="1.0.0", arguments=_arguments(), expires_at=datetime.now(UTC) + timedelta(minutes=5))
    with pytest.raises(ApprovalServiceError, match="approver_permission_denied"):
        service.approve(auth=requester, approval_id=pending.approval_id)
    binding = service.approve(auth=approver, approval_id=pending.approval_id)
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": _VerifiedAdapter()}), service)
    first = executor.execute(_request(requester, binding.approval_id, lease))
    second = executor.execute(_request(requester, binding.approval_id, lease))
    assert first.status == second.status == "succeeded"
    assert second.duplicate is True and first.result_hash == second.result_hash
    with pytest.raises(ToolExecutionError, match="idempotency_key_conflict"):
        executor.execute(_request(requester, binding.approval_id, lease, _arguments("changed")))


def test_tool_adapter_thread_keeps_execute_trace_and_excludes_arguments(stage5) -> None:
    database, requester, approver, _, lease = stage5
    arguments = _arguments("Bearer fake-token tool-argument C:/private/object")
    approvals, binding = _approved(database, requester, approver, arguments)
    executor = _executor(
        database,
        ToolRegistry(adapters={"create_report_artifact": _VerifiedAdapter()}),
        approvals,
    )
    observability, exporter = _telemetry()
    try:
        with observability.activate():
            result = executor.execute(
                _request(requester, binding.approval_id, lease, arguments, key="trace-key")
            )
        payload = observability.metrics_payload().decode()
        spans = exporter.get_finished_spans()
        rendered_spans = repr(spans)
    finally:
        executor.close()
        observability.shutdown()

    execute_span = next(span for span in spans if span.name == "tool.execute")
    adapter_span = next(span for span in spans if span.name == "tool.adapter")
    assert result.status == "succeeded"
    assert {"tool.execute", "tool.adapter", "tool.verify", "tool.complete"} <= {
        span.name for span in spans
    }
    assert adapter_span.context.trace_id == execute_span.context.trace_id
    assert adapter_span.parent is not None and adapter_span.parent.span_id == execute_span.context.span_id
    tool_metric = next(
        line
        for line in payload.splitlines()
        if line.startswith("rag_kb_tool_calls_total{") and 'outcome="succeeded"' in line
    )
    assert all(
        label in tool_metric
        for label in ('tool="create_report_artifact"', 'mode="write"', 'outcome="succeeded"')
    )
    assert arguments["content"] not in payload
    assert arguments["content"] not in rendered_spans


def test_same_approval_can_be_consumed_once_even_with_a_new_idempotency_key(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service, binding = _approved(database, requester, approver)
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": _VerifiedAdapter()}), service)
    first = executor.execute(_request(requester, binding.approval_id, lease, key="key-1"))
    assert first.status == "succeeded"
    with pytest.raises(ToolExecutionError, match="approval_not_approved"):
        executor.execute(_request(requester, binding.approval_id, lease, key="key-2"))


def test_acl_revocation_after_approval_rejects_write_before_adapter(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service, binding = _approved(database, requester, approver)
    with database.sessions.begin() as session:
        grant = session.scalar(select(ACLGrantRow).where(ACLGrantRow.grant_id == "grant-1"))
        assert grant is not None
        session.delete(grant)
    adapter = _CountingAdapter()
    executor = _executor(
        database,
        ToolRegistry(adapters={"create_report_artifact": adapter}),
        service,
    )
    with pytest.raises(ToolExecutionError, match="evidence_authorization_stale"):
        executor.execute(_request(requester, binding.approval_id, lease))
    assert adapter.calls == 0
    with database.sessions() as session:
        approval = session.get(ApprovalRow, binding.approval_id)
        assert approval is not None and approval.status == "rejected"


def test_non_controlled_task_cannot_request_write_approval(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        task.task_mode = TaskMode.RESEARCH.value
    plan_digest, evidence_digest = _binding(database)
    with pytest.raises(ApprovalServiceError, match="approval_plan_binding_invalid"):
        ApprovalService(DatabaseApprovalLifecycle(database)).request(
            auth=requester,
            task_id="task-1",
            plan_revision=1,
            plan_digest=plan_digest,
            evidence_digest=evidence_digest,
            evidence_ids=("evidence-1",),
            tool_name="create_report_artifact",
            tool_version="1.0.0",
            arguments=_arguments(),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )


def test_stale_worker_is_fenced_before_adapter_call(stage5) -> None:
    database, requester, approver, _, old_lease = stage5
    service, binding = _approved(database, requester, approver)
    with database.engine.begin() as connection:
        connection.execute(
            update(AgentTaskRow)
            .where(AgentTaskRow.task_id == "task-1")
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    with database.engine.begin() as connection:
        connection.execute(
            update(AgentTaskRow)
            .where(AgentTaskRow.task_id == "task-1")
            .values(
                lease_owner="worker-2",
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
                lease_generation=old_lease.lease_generation + 1,
            )
        )
    adapter = _CountingAdapter()
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": adapter}), service)
    with pytest.raises(ToolExecutionError, match="tool_task_lease_stale"):
        executor.execute(_request(requester, binding.approval_id, old_lease))
    assert adapter.calls == 0


def test_expired_approval_is_persisted_and_same_binding_can_be_requested_again(stage5) -> None:
    database, requester, approver, _, _lease = stage5
    clock = [datetime.now(UTC)]
    service = ApprovalService(
        DatabaseApprovalLifecycle(database, now=lambda: clock[0]),
        now=lambda: clock[0],
    )
    binding = service.request(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=_binding(database)[0],
        evidence_digest=_binding(database)[1],
        evidence_ids=("evidence-1",),
        tool_name="create_report_artifact",
        tool_version="1.0.0",
        arguments=_arguments(),
        expires_at=clock[0] + timedelta(minutes=1),
    )
    binding = service.approve(auth=approver, approval_id=binding.approval_id)
    clock[0] += timedelta(minutes=5)
    assert service.expire_due() == 1
    with database.sessions() as session:
        row = session.scalar(select(ApprovalRow).where(ApprovalRow.approval_id == binding.approval_id))
        assert row is not None and row.status == "expired"
    replacement = service.request(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=_binding(database)[0],
        evidence_digest=_binding(database)[1],
        evidence_ids=("evidence-1",),
        tool_name="create_report_artifact",
        tool_version="1.0.0",
        arguments=_arguments(),
        expires_at=clock[0] + timedelta(minutes=1),
    )
    assert replacement.approval_id != binding.approval_id


def test_write_crash_is_unknown_and_never_success(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service, binding = _approved(database, requester, approver)
    result = _executor(
        database,
        ToolRegistry(adapters={"create_report_artifact": _CrashingAdapter()}),
        service,
    ).execute(_request(requester, binding.approval_id, lease))
    assert result.status == "unknown" and result.status != "succeeded"
    with database.sessions() as session:
        row = session.scalar(select(ToolCallRow).where(ToolCallRow.tool_call_id == result.tool_call_id))
        assert row is not None and row.status == "unknown" and row.result_hash is None


def test_unknown_write_records_safe_metric_trace_and_log(stage5, caplog) -> None:
    database, requester, approver, _, lease = stage5
    secret = "Bearer synthetic-token private tool payload C:/private/object"
    arguments = _arguments(secret)
    service, binding = _approved(database, requester, approver, arguments)
    executor = _executor(
        database,
        ToolRegistry(adapters={"create_report_artifact": _CrashingAdapter()}),
        service,
    )
    observability, exporter = _telemetry()
    try:
        caplog.set_level("INFO")
        with observability.activate():
            result = executor.execute(
                _request(requester, binding.approval_id, lease, arguments)
            )
        payload = observability.metrics_payload().decode()
        spans = exporter.get_finished_spans()
    finally:
        executor.close()
        observability.shutdown()

    assert result.status == "unknown"
    assert 'mode="write",outcome="unknown"' in payload
    assert 'kind="write_unknown"' in payload
    execute_span = next(span for span in spans if span.name == "tool.execute")
    assert execute_span.attributes["outcome"] == "unknown"
    assert execute_span.status.status_code.name == "ERROR"
    assert secret not in payload
    assert secret not in repr(spans)
    assert secret not in caplog.text


def test_unknown_isolated_from_artifact_and_task_after_object_write_failure(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service, binding = _approved(database, requester, approver)
    adapter = LocalArtifactToolAdapter(database, _FailingObjectStore(), artifact_type="report")
    result = _executor(
        database, ToolRegistry(adapters={"create_report_artifact": adapter}), service
    ).execute(_request(requester, binding.approval_id, lease))
    assert result.status == "unknown"
    with database.sessions() as session:
        artifact = session.scalar(
            select(ArtifactRow).where(ArtifactRow.task_id == "task-1", ArtifactRow.tool_call_id == result.tool_call_id)
        )
        task = session.scalar(select(AgentTaskRow).where(AgentTaskRow.task_id == "task-1"))
        assert artifact is not None and artifact.status == "unknown"
        assert task is not None and task.status == "unknown"


def test_timed_out_artifact_adapter_cannot_restore_unknown_to_ready(stage5) -> None:
    database, requester, approver, _, lease = stage5
    approval_service, binding = _approved(database, requester, approver)
    store = _BlockingObjectStore()
    adapter = _SignalingAdapter(
        LocalArtifactToolAdapter(database, store, artifact_type="report")
    )
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "create_report_artifact"
    ).model_copy(update={"timeout_seconds": 1})
    executor = _executor(
        database,
        ToolRegistry([definition], adapters={"create_report_artifact": adapter}),
        approval_service,
    )

    try:
        result = executor.execute(_request(requester, binding.approval_id, lease))
        assert result.status == "unknown"
        assert store.entered.is_set()
    finally:
        store.release.set()

    assert adapter.finished.wait(timeout=5), "timed out adapter did not finish"
    with database.sessions() as session:
        artifact = session.scalar(
            select(ArtifactRow).where(ArtifactRow.tool_call_id == result.tool_call_id)
        )
        assert artifact is not None and artifact.status == "unknown"


def test_artifact_created_after_timeout_is_never_ready(stage5) -> None:
    database, requester, approver, _, lease = stage5
    approval_service, binding = _approved(database, requester, approver)
    store = LocalObjectStore(Path(database.engine.url.database).parent / "late-objects")
    inner = LocalArtifactToolAdapter(database, store, artifact_type="report")

    class DelayedAdapter:
        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()
            self.finished = Event()

        def execute(self, arguments, context):
            self.entered.set()
            assert self.release.wait(timeout=5)
            try:
                return inner.execute(arguments, context)
            finally:
                self.finished.set()

        def verify(self, result, context):
            return inner.verify(result, context)

    adapter = DelayedAdapter()
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "create_report_artifact"
    ).model_copy(update={"timeout_seconds": 1})
    executor = _executor(
        database,
        ToolRegistry([definition], adapters={"create_report_artifact": adapter}),
        approval_service,
    )

    try:
        result = executor.execute(_request(requester, binding.approval_id, lease))
        assert result.status == "unknown"
        with database.sessions() as session:
            assert session.scalar(
                select(ArtifactRow).where(ArtifactRow.tool_call_id == result.tool_call_id)
            ) is None
    finally:
        adapter.release.set()

    assert adapter.finished.wait(timeout=5)
    with database.sessions() as session:
        artifact = session.scalar(
            select(ArtifactRow).where(ArtifactRow.tool_call_id == result.tool_call_id)
        )
        assert artifact is not None and artifact.status == "unknown"


def test_timeout_replaces_stuck_executor_for_next_call(stage5) -> None:
    database, requester, _approver, _, lease = stage5

    class FirstCallBlocks(_VerifiedAdapter):
        def __init__(self) -> None:
            self.calls = 0
            self.release = Event()

        def execute(self, arguments, context):
            self.calls += 1
            if self.calls == 1:
                assert self.release.wait(timeout=5)
            return super().execute(arguments, context)

    adapter = FirstCallBlocks()
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "create_report_artifact"
    ).model_copy(
        update={
            "mode": ToolMode.READ,
            "approval_policy": ApprovalPolicy.NEVER,
            "timeout_seconds": 1,
        }
    )
    executor = _executor(
        database,
        ToolRegistry([definition], adapters={"create_report_artifact": adapter}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
    )

    try:
        first = executor.execute(_request(requester, None, lease, key="stuck"))
        second = executor.execute(_request(requester, None, lease, key="next"))
    finally:
        adapter.release.set()

    assert first.status == "failed"
    assert second.status == "succeeded"
    assert adapter.calls == 2


def test_expired_running_sweeper_persists_unknown_without_adapter_execution(stage5) -> None:
    database, requester, approver, _, _lease = stage5
    _approval_service, binding = _approved(database, requester, approver)
    tool_call_id = uuid4().hex
    artifact_id = uuid4().hex
    now = datetime.now(UTC)
    with UnitOfWork(database) as work:
        assert work.session is not None
        work.session.add(
            ToolCallRow(
                tool_call_id=tool_call_id,
                task_id="task-1",
                tenant_id="tenant-a",
                approval_id=binding.approval_id,
                plan_revision=1,
                plan_digest=_binding(database)[0],
                evidence_digest=_binding(database)[1],
                tool_name="create_report_artifact",
                tool_version="1.0.0",
                arguments_hash=arguments_fingerprint(_arguments()),
                principal_hash=principal_fingerprint(requester),
                mode="write",
                idempotency_key="sweeper-key",
                risk_level="low",
                status="running",
                created_at=now - timedelta(minutes=2),
                execution_deadline=now - timedelta(minutes=1),
            )
        )
        work.session.flush()
        work.session.add(
            ArtifactRow(
                artifact_id=artifact_id,
                task_id="task-1",
                tenant_id="tenant-a",
                tool_call_id=tool_call_id,
                artifact_type="report",
                storage_key="artifacts/pending",
                content_hash="sha256:pending",
                status="creating",
                created_at=now,
            )
        )
    adapter = _CountingAdapter()
    service = _executor(
        database,
        ToolRegistry(adapters={"create_report_artifact": adapter}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
    )
    assert service.expire_stale_running() == 1
    with database.sessions() as session:
        call = session.scalar(select(ToolCallRow).where(ToolCallRow.tool_call_id == tool_call_id))
        artifact = session.scalar(select(ArtifactRow).where(ArtifactRow.artifact_id == artifact_id))
        task = session.scalar(select(AgentTaskRow).where(AgentTaskRow.task_id == "task-1"))
        assert call is not None and call.status == "unknown"
        assert artifact is not None and artifact.status == "unknown"
        assert task is not None and task.status == "unknown"
    assert adapter.calls == 0


def test_local_artifact_is_server_keyed_exclusive_and_readback_verified(stage5) -> None:
    database, requester, approver, tmp_path, lease = stage5
    service, binding = _approved(database, requester, approver)
    adapter = LocalArtifactToolAdapter(database, LocalObjectStore(tmp_path / "objects"), artifact_type="report")
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": adapter}), service)
    result = executor.execute(_request(requester, binding.approval_id, lease))
    assert result.status == "succeeded" and result.result is not None, result
    # 模型可见面只含 artifact_id/hash/status；storage_key 不得出现。
    assert isinstance(result.result["artifact_id"], str) and result.result["artifact_id"]
    assert result.result["content_hash"].startswith("sha256:")
    assert "storage_key" not in result.result
    with pytest.raises(Exception, match="tool_arguments_invalid"):
        executor.execute(_request(requester, binding.approval_id, lease, {**_arguments(), "path": "C:/secret"}, "key-2"))


# ---------- M2 批次 1：信任边界收紧与 checkpoint 绑定 ----------


def _checkpoint_binding(
    **overrides: object,
) -> ApprovalCheckpointBinding:
    values: dict[str, object] = {
        "thread_id": "task-1",
        "checkpoint_ns": "",
        "checkpoint_id": "checkpoint-1",
        "interrupt_id": "interrupt-1",
        "tool_call_id": "call_1",
    }
    values.update(overrides)
    return ApprovalCheckpointBinding(**values)  # type: ignore[arg-type]


def _request_with_checkpoint_binding(
    database,
    requester: AuthContext,
    expires_delta: timedelta = timedelta(minutes=5),
) -> dict[str, object]:
    plan_digest, evidence_digest = _binding(database)
    return {
        "auth": requester,
        "task_id": "task-1",
        "plan_revision": 1,
        "plan_digest": plan_digest,
        "evidence_digest": evidence_digest,
        "evidence_ids": ("evidence-1",),
        "tool_name": "create_report_artifact",
        "arguments": _arguments(),
        "expires_at": datetime.now(UTC) + expires_delta,
    }


def test_request_rejects_unknown_tool_and_forged_version(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    base = _request_with_checkpoint_binding(database, requester)

    with pytest.raises(ApprovalServiceError, match="tool_not_registered"):
        service.request(**{**base, "tool_name": "arbitrary_shell"})
    with pytest.raises(ApprovalServiceError, match="approval_tool_mode_invalid"):
        service.request(**{**base, "tool_name": "search_knowledge"})
    forged = {**base, "tool_version": "9.9.9"}
    with pytest.raises(ApprovalServiceError, match="approval_tool_version_invalid"):
        service.request(**forged)

    # 服务端派生版本 + 未传版本两条合法路径等价。
    derived = service.request(**base)
    explicit = service.request(**{**base, "tool_version": "1.0.0"})
    assert derived.approval_id == explicit.approval_id
    assert derived.tool_version == "1.0.0"


def test_initiator_cannot_self_approve(stage5) -> None:
    database, requester, _approver, tmp_path, _lease = stage5
    # 独立场景：发起人本身持有 task:approve（身份重叠）也不能自批。
    privileged = AuthContext(
        tenant_id="tenant-a",
        user_id="self-approver",
        attributes={
            "permissions": [
                "artifact:write",
                "knowledge:read",
                "document:read",
                "task:approve",
            ]
        },
        request_id="request-self",
    )
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) "
                "VALUES ('tenant-a', 'self-approver', '[]', :attributes, 1, CURRENT_TIMESTAMP)"
            ),
            {"attributes": json.dumps({"permissions": privileged.attributes["permissions"]})},
        )
        connection.execute(
            text(
                "INSERT INTO acl_grants (grant_id,tenant_id,resource_type,resource_id,"
                "subject_type,subject_id,permissions,created_at) VALUES "
                "('grant-self','tenant-a','document','doc-1','user','self-approver',"
                "'[\"knowledge:read\",\"document:read\"]', CURRENT_TIMESTAMP)"
            )
        )
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=privileged,
        goal="自批测试",
        task_mode=TaskMode.CONTROLLED_ACTION,
        task_id="task-self",
    )
    # 复用既有 evidence：把 evidence-1 挂到新任务（EvidenceRecordRow 允许同 chunk 多任务引用）。
    with UnitOfWork(database) as work:
        assert work.session is not None
        work.session.add(
            EvidenceRecordRow(
                evidence_id="evidence-self",
                task_id="task-self",
                query_fingerprint="sha256:query",
                tenant_id="tenant-a",
                document_id="doc-1",
                version_id="version-1",
                document_version=1,
                chunk_id="chunk-1",
                content_hash="sha256:chunk",
                source_locator="doc/1",
                permission_granted=True,
                supported_claim_ids=[],
            )
        )
        task = work.session.get(AgentTaskRow, "task-self")
        assert task is not None
        task.status = "running"
        task.lease_owner = "worker-1"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=60)
        task.lease_generation = 1

    service = ApprovalService(DatabaseApprovalLifecycle(database))
    snapshot = canonical_evidence_snapshot(("evidence-self",), require_non_empty=True)
    plan = {"steps": ["report"], "evidence_ids": ["evidence-self"]}
    plan_digest = canonical_fingerprint("task_plan", plan)
    with UnitOfWork(database) as work:
        assert work.session is not None
        work.session.add(
            PlanRevisionRow(
                plan_revision_id="plan-self",
                task_id="task-self",
                tenant_id="tenant-a",
                revision=1,
                schema_version=1,
                plan=plan,
                plan_digest=plan_digest,
                evidence_digest=snapshot.digest,
                status="ready",
            )
        )

    pending = service.request(
        auth=privileged,
        task_id="task-self",
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=snapshot.digest,
        evidence_ids=("evidence-self",),
        tool_name="create_report_artifact",
        arguments=_arguments(),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    with pytest.raises(ApprovalServiceError, match="approval_self_rejected"):
        service.approve(auth=privileged, approval_id=pending.approval_id)
    with database.sessions() as session:
        row = session.get(ApprovalRow, pending.approval_id)
        assert row is not None
        # 自批是决策非法，不是 action 被拒；approval 保持 pending。
        assert row.status == "pending"
        assert row.approved_by is None

    approver = AuthContext(
        tenant_id="tenant-a",
        user_id="approver",
        attributes={"permissions": ["task:approve"]},
        request_id="approve-2",
    )
    approved = service.approve(auth=approver, approval_id=pending.approval_id)
    assert approved.status == "approved"


def test_checkpoint_binding_persists_and_resync_reuses_active_approval(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    base = _request_with_checkpoint_binding(database, requester)

    first = service.request(**base, checkpoint_binding=_checkpoint_binding())
    assert first.status == "pending"
    assert first.thread_id == "task-1"
    assert first.checkpoint_ns == ""
    assert first.checkpoint_id == "checkpoint-1"
    assert first.interrupt_id == "interrupt-1"
    assert first.tool_call_id == "call_1"
    assert first.checkpoint_binding_hash

    # 同一 checkpoint action 重复同步：复用同一条 active approval。
    second = service.request(**base, checkpoint_binding=_checkpoint_binding())
    assert second.approval_id == first.approval_id
    with database.sessions() as session:
        count = len(
            list(
                session.scalars(
                    select(ApprovalRow).where(
                        ApprovalRow.tenant_id == "tenant-a",
                        ApprovalRow.task_id == "task-1",
                    )
                )
            )
        )
    assert count == 1

    # 不同 tool_call（不同 checkpoint action）不得复用同一 active approval 绑定。
    other_action = service.request(
        **base,
        checkpoint_binding=_checkpoint_binding(tool_call_id="call_2", interrupt_id="interrupt-2"),
    )
    assert other_action.approval_id != first.approval_id


def test_checkpoint_binding_conflict_fails_closed(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    base = _request_with_checkpoint_binding(database, requester)
    service.request(**base, checkpoint_binding=_checkpoint_binding())

    # 同一 checkpoint 绑定但执行事实被篡改：fail-closed。
    tampered_args = {**base["arguments"], "content": "被篡改的正文"}
    with pytest.raises(ApprovalServiceError, match="approval_checkpoint_conflict"):
        service.request(
            **{**base, "arguments": tampered_args},
            checkpoint_binding=_checkpoint_binding(),
        )


def test_legacy_active_approval_adopts_checkpoint_binding(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    base = _request_with_checkpoint_binding(database, requester)

    legacy = service.request(**base)
    assert legacy.checkpoint_binding_hash is None
    resumed = service.request(**base, checkpoint_binding=_checkpoint_binding())
    assert resumed.approval_id == legacy.approval_id
    assert resumed.checkpoint_binding_hash is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("thread_id", "other-thread"),
        ("checkpoint_id", "other-checkpoint"),
        ("interrupt_id", "other-interrupt"),
        ("tool_call_id", "other-call"),
    ],
)
def test_checkpoint_binding_change_does_not_consume_same_approval(
    stage5, field: str, value: str
) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    base = _request_with_checkpoint_binding(database, requester)
    first = service.request(**base, checkpoint_binding=_checkpoint_binding())
    # 不同 checkpoint 事实 = 不同 action：新 approval，不消费第一条。
    second = service.request(
        **base, checkpoint_binding=_checkpoint_binding(**{field: value})
    )
    assert second.approval_id != first.approval_id
    with database.sessions() as session:
        statuses = {
            row.approval_id: row.status
            for row in session.scalars(
                select(ApprovalRow).where(ApprovalRow.task_id == "task-1")
            )
        }
    assert statuses[first.approval_id] == "pending"
    assert statuses[second.approval_id] == "pending"


def test_ensure_action_binding_creates_minimal_immutable_plan(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    binding = _checkpoint_binding()

    action = service.ensure_action_binding(
        auth=requester,
        task_id="task-1",
        tool_name="create_report_artifact",
        arguments=_arguments(),
        checkpoint_binding=binding,
    )
    assert action.plan_revision >= 1
    assert action.evidence_ids == ("evidence-1",)
    with database.sessions() as session:
        row = session.scalar(
            select(PlanRevisionRow).where(
                PlanRevisionRow.task_id == "task-1",
                PlanRevisionRow.revision == action.plan_revision,
            )
        )
        assert row is not None
        # plan 只含 hash 引用：原始参数、正文不进入 plan。
        assert set(row.plan) == {
            "tool_name",
            "tool_version",
            "arguments_hash",
            "evidence_ids",
            "checkpoint_binding_hash",
        }
        assert "content" not in row.plan
        assert "正文" not in str(row.plan)
        assert row.plan["tool_version"] == "1.0.0"

    # 重复派生：同一 action 幂等复用同一 revision。
    repeated = service.ensure_action_binding(
        auth=requester,
        task_id="task-1",
        tool_name="create_report_artifact",
        arguments=_arguments(),
        checkpoint_binding=binding,
    )
    assert repeated.plan_revision == action.plan_revision
    assert repeated.plan_digest == action.plan_digest

    # 参数变化 = 不同 action：新 revision。
    changed = service.ensure_action_binding(
        auth=requester,
        task_id="task-1",
        tool_name="create_report_artifact",
        arguments=_arguments("changed"),
        checkpoint_binding=binding,
    )
    assert changed.plan_revision > action.plan_revision


def test_ensure_action_binding_rejects_unknown_tool(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    with pytest.raises(ApprovalServiceError, match="tool_not_registered"):
        service.ensure_action_binding(
            auth=requester,
            task_id="task-1",
            tool_name="arbitrary_shell",
            arguments={"cmd": "rm -rf /"},
            checkpoint_binding=_checkpoint_binding(),
        )


def test_ensure_action_binding_rejects_invalid_checkpoint_binding(stage5) -> None:
    database, requester, _approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    with pytest.raises(ApprovalServiceError, match="approval_checkpoint_binding_invalid"):
        service.ensure_action_binding(
            auth=requester,
            task_id="task-1",
            tool_name="create_report_artifact",
            arguments=_arguments(),
            checkpoint_binding=_checkpoint_binding(tool_call_id=""),
        )


def test_checkpoint_binding_hash_is_action_scoped() -> None:
    base = _checkpoint_binding()
    same = _checkpoint_binding()
    assert base.checkpoint_binding_hash == same.checkpoint_binding_hash
    # thread/checkpoint/interrupt/tool_call 任一事实变化都构成不同 action。
    for field, value in (
        ("thread_id", "task-2"),
        ("checkpoint_ns", "subgraph"),
        ("checkpoint_id", "checkpoint-2"),
        ("interrupt_id", "interrupt-2"),
        ("tool_call_id", "call_2"),
    ):
        assert _checkpoint_binding(**{field: value}).checkpoint_binding_hash != (
            base.checkpoint_binding_hash
        )


_BOTH_WRITE_TOOLS = (
    ("create_report_artifact", {"title": "报告", "content": "正文", "format": "markdown"}),
    ("create_work_item_draft", {"title": "草稿", "description": "工作项描述", "owner_group": "platform"}),
)


@pytest.mark.parametrize("tool_name,arguments", _BOTH_WRITE_TOOLS)
@pytest.mark.parametrize(
    "dimension",
    ("task", "tool", "arguments", "evidence", "principal", "expiry"),
)
def test_binding_tampering_keeps_write_adapter_unexecuted(
    stage5, tool_name: str, arguments: dict, dimension: str
) -> None:
    """两个 WRITE 工具参数化覆盖执行绑定；任一字段被篡改时 adapter 调用为 0。"""
    database, requester, approver, _, lease = stage5
    clock = [datetime.now(UTC)]
    service = ApprovalService(
        DatabaseApprovalLifecycle(database, now=lambda: clock[0]),
        now=lambda: clock[0],
    )
    plan_digest, evidence_digest = _binding(database)
    # evidence 维度会触发 preflight 失败 → _reject_stale_approval 自动拒绝路径；
    # checkpoint 绑定行的自动拒绝审计字段（批次 0 已知缺口，批次 2 修复）尚未接线，
    # 因此该维度使用 legacy 绑定（binding IS NULL 不受审计 CHECK 约束）。
    checkpoint_binding = (
        None
        if dimension == "evidence"
        else _checkpoint_binding(tool_call_id=f"call-{tool_name}")
    )
    pending = service.request(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
        tool_name=tool_name,
        tool_version=None,
        arguments=arguments,
        expires_at=clock[0] + timedelta(minutes=5),
        checkpoint_binding=checkpoint_binding,
    )
    binding = service.approve(auth=approver, approval_id=pending.approval_id)
    adapter = _CountingAdapter()
    executor = _executor(
        database,
        ToolRegistry(
            adapters={
                "create_report_artifact": adapter,
                "create_work_item_draft": adapter,
            }
        ),
        service,
    )
    request = ToolExecutionRequest(
        auth=requester,
        task_id="task-1",
        tool_name=tool_name,
        arguments=dict(arguments),
        worker_id=lease.worker_id,
        lease_generation=lease.lease_generation,
        idempotency_key="tamper-key-1",
        approval_id=binding.approval_id,
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
    )

    if dimension == "expiry":
        clock[0] = clock[0] + timedelta(minutes=6)
    elif dimension == "task":
        request = replace(request, task_id="other-task")
    elif dimension == "tool":
        other = (
            "create_work_item_draft"
            if tool_name == "create_report_artifact"
            else "create_report_artifact"
        )
        request = replace(request, tool_name=other)
    elif dimension == "arguments":
        request = replace(request, arguments={**arguments, "title": "被篡改的标题"})
    elif dimension == "evidence":
        request = replace(request, evidence_ids=("evidence-1", "evidence-2"))
    elif dimension == "principal":
        forged = AuthContext(
            tenant_id="tenant-a",
            user_id="requester",
            attributes={
                "permissions": [
                    "artifact:write",
                    "knowledge:read",
                    "document:read",
                    "extra:permission",
                ]
            },
            request_id="forged-principal",
        )
        request = replace(request, auth=forged)

    with pytest.raises((ToolExecutionError, ApprovalServiceError, ToolRegistryError)):
        executor.execute(request)
    assert adapter.calls == 0


@pytest.mark.parametrize("tool_name,arguments", _BOTH_WRITE_TOOLS)
def test_valid_checkpoint_bound_approval_executes_exactly_once(
    stage5, tool_name: str, arguments: dict
) -> None:
    """对照组：绑定完整未篡改时，两个 WRITE 工具各恰好执行一次。"""
    database, requester, approver, _, lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    plan_digest, evidence_digest = _binding(database)
    binding = service.request(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
        tool_name=tool_name,
        tool_version=None,
        arguments=arguments,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        checkpoint_binding=_checkpoint_binding(tool_call_id=f"call-{tool_name}"),
    )
    service.approve(auth=approver, approval_id=binding.approval_id)
    adapter = _CountingAdapter()
    executor = _executor(
        database,
        ToolRegistry(
            adapters={
                "create_report_artifact": adapter,
                "create_work_item_draft": adapter,
            }
        ),
        service,
    )
    request = ToolExecutionRequest(
        auth=requester,
        task_id="task-1",
        tool_name=tool_name,
        arguments=dict(arguments),
        worker_id=lease.worker_id,
        lease_generation=lease.lease_generation,
        idempotency_key="valid-key-1",
        approval_id=binding.approval_id,
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
    )

    result = executor.execute(request)

    assert result.status == "succeeded"
    assert adapter.calls == 1


class _ArtifactIdAdapter:
    """批次 2 目标形态 adapter：只返回 artifact_id，不暴露 storage_key。"""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _arguments, _context):
        self.calls += 1
        return {
            "artifact_type": "report",
            "artifact_id": "artifact-1",
            "content_hash": "sha256:" + "a" * 64,
            "status": "ready",
        }

    def verify(self, _result, _context) -> bool:
        return True


def test_stale_checkpoint_bound_approval_auto_rejection_writes_audit(stage5) -> None:
    """C-1：checkpoint 绑定行的自动拒绝必须补齐 rejected_by/at/reason 审计字段。

    服务端固定 actor + 稳定 code（M2_PLAN D3）；现有实现缺审计字段，
    对绑定行触发 ck_approvals_rejected_audit_complete IntegrityError。
    """
    database, requester, approver, _, lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    plan_digest, evidence_digest = _binding(database)
    pending = service.request(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
        tool_name="create_report_artifact",
        tool_version=None,
        arguments=_arguments(),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        checkpoint_binding=_checkpoint_binding(tool_call_id="call-audit"),
    )
    service.approve(auth=approver, approval_id=pending.approval_id)
    with database.sessions.begin() as session:
        grant = session.scalar(select(ACLGrantRow).where(ACLGrantRow.grant_id == "grant-1"))
        assert grant is not None
        session.delete(grant)
    adapter = _CountingAdapter()
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": adapter}), service)

    with pytest.raises(ToolExecutionError, match="evidence_authorization_stale"):
        executor.execute(_request(requester, pending.approval_id, lease))
    assert adapter.calls == 0
    with database.sessions() as session:
        approval = session.get(ApprovalRow, pending.approval_id)
        assert approval is not None
        assert approval.status == "rejected"
        assert approval.rejected_by == "system:approval-reaper"
        assert approval.rejected_at is not None
        assert approval.rejection_reason
        assert approval.rejection_reason.startswith("auto:")


def test_approve_time_binding_invalidation_writes_reject_audit(stage5) -> None:
    """approve 时发现绑定失效：同样必须写完整 reject 审计三元组（C-1 姊妹缺口）。

    checkpoint 绑定行受 ck_approvals_rejected_audit_complete 约束；
    审批者是触发拒绝的真实主体，rejected_by 取审批者。
    """
    database, requester, approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    plan_digest, evidence_digest = _binding(database)
    pending = service.request(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
        tool_name="create_report_artifact",
        tool_version=None,
        arguments=_arguments(),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        checkpoint_binding=_checkpoint_binding(tool_call_id="call-approve-audit"),
    )
    with database.sessions.begin() as session:
        grant = session.scalar(select(ACLGrantRow).where(ACLGrantRow.grant_id == "grant-1"))
        assert grant is not None
        session.delete(grant)

    with pytest.raises(ApprovalServiceError, match="evidence_authorization_stale"):
        service.approve(auth=approver, approval_id=pending.approval_id)
    with database.sessions() as session:
        approval = session.get(ApprovalRow, pending.approval_id)
        assert approval is not None
        assert approval.status == "rejected"
        assert approval.rejected_by == "approver"
        assert approval.rejected_at is not None
        assert approval.rejection_reason
        assert approval.rejection_reason.startswith("auto:")


def test_repeat_same_decision_is_idempotent_and_opposite_decision_conflicts(    stage5,
) -> None:
    """相同 decision 幂等返回当前状态；相反 decision 返回稳定冲突错误。"""
    database, requester, approver, _, _lease = stage5
    service = ApprovalService(DatabaseApprovalLifecycle(database))
    plan_digest, evidence_digest = _binding(database)
    common = dict(
        auth=requester,
        task_id="task-1",
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=("evidence-1",),
        tool_name="create_report_artifact",
        tool_version=None,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    first = service.request(**common, arguments=_arguments("first"))
    approved = service.approve(auth=approver, approval_id=first.approval_id)
    assert approved.status == "approved"

    # 相同 decision 重复提交：幂等返回当前状态，不报错。
    repeat = service.approve(auth=approver, approval_id=first.approval_id)
    assert repeat.status == "approved"
    assert repeat.approval_id == first.approval_id

    second = service.request(**common, arguments=_arguments("second"))
    rejected = service.reject(auth=approver, approval_id=second.approval_id)
    assert rejected.status == "rejected"
    repeat_reject = service.reject(auth=approver, approval_id=second.approval_id)
    assert repeat_reject.status == "rejected"

    with database.sessions() as session:
        row = session.get(ApprovalRow, second.approval_id)
        assert row is not None
        assert row.rejected_by == "approver"
        assert row.rejected_at is not None
        assert row.rejection_reason

    # 相反 decision：稳定冲突错误，不改变终态。
    with pytest.raises(ApprovalServiceError, match="approval_decision_conflict"):
        service.reject(auth=approver, approval_id=first.approval_id)
    with pytest.raises(ApprovalServiceError, match="approval_decision_conflict"):
        service.approve(auth=approver, approval_id=second.approval_id)
    with database.sessions() as session:
        assert session.get(ApprovalRow, first.approval_id).status == "approved"
        assert session.get(ApprovalRow, second.approval_id).status == "rejected"


def test_write_result_exposes_artifact_id_and_task_refs_bind_artifact_id(
    stage5,
) -> None:
    """C-3/C-4：结果与 agent_tasks.artifact_refs 只绑定 artifact_id，无 storage_key。"""
    database, requester, approver, _, lease = stage5
    service, binding = _approved(database, requester, approver)
    adapter = _ArtifactIdAdapter()
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": adapter}), service)

    result = executor.execute(_request(requester, binding.approval_id, lease))
    assert result.status == "succeeded"
    assert result.result is not None
    assert result.result["artifact_id"] == "artifact-1"
    assert "storage_key" not in result.result

    duplicate = executor.execute(_request(requester, binding.approval_id, lease))
    assert duplicate.status == "succeeded"
    assert duplicate.duplicate is True
    assert duplicate.result is not None
    assert duplicate.result["artifact_id"] == "artifact-1"
    assert "storage_key" not in duplicate.result
    assert adapter.calls == 1

    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.artifact_refs == ["artifact-1"]
