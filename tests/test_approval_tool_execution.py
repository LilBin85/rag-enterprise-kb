"""Stage 5 批准绑定、持久执行、幂等与 UNKNOWN 回归。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import select, text, update

from rag_kb.application.approval_service import ApprovalService, ApprovalServiceError, arguments_fingerprint, principal_fingerprint
from rag_kb.application.tool_execution_service import ToolExecutionError, ToolExecutionRequest, ToolExecutionService
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tasks import TaskBudgets, TaskMode
from rag_kb.domain.tools import ApprovalPolicy, ToolMode
from rag_kb.infrastructure.database import Base, build_database
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
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.infrastructure.tools.artifacts import LocalArtifactToolAdapter


class _VerifiedAdapter:
    def execute(self, arguments, context):
        return {"artifact_type": "report", "storage_key": "server/generated", "content_hash": "sha256:" + "a" * 64, "status": "ready"}

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
    service = ApprovalService(database)
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
        database,
        registry,
        approvals,
        tool_results=LocalObjectStore(Path(database.engine.url.database).parent / "tool-results"),
    )


def test_any_approval_binding_change_invalidates_execution(stage5) -> None:
    database, requester, approver, _, _lease = stage5
    service, binding = _approved(database, requester, approver)
    changes = ({"task_id": "other-task"}, {"plan_revision": 2}, {"plan_digest": "sha256:other"}, {"evidence_digest": "sha256:other"}, {"tool_version": "2.0"}, {"arguments_hash": "sha256:other"})
    for change in changes:
        plan_digest, evidence_digest = _binding(database)
        values = dict(auth=requester, approval_id=binding.approval_id, task_id="task-1", plan_revision=1, plan_digest=plan_digest, evidence_digest=evidence_digest, evidence_ids=("evidence-1",), tool_name="create_report_artifact", tool_version="1.0.0", arguments_hash=binding.canonical_arguments_hash)
        values.update(change)
        with database.sessions() as session, pytest.raises(
            ApprovalServiceError,
            match="approval_(binding|evidence_exact_set)_invalid",
        ):
            service.validate_in_session(session, **values)


def test_approval_requires_authorized_approver_and_write_is_idempotent(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service = ApprovalService(database)
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


def test_same_approval_can_be_consumed_once_even_with_a_new_idempotency_key(stage5) -> None:
    database, requester, approver, _, lease = stage5
    service, binding = _approved(database, requester, approver)
    executor = _executor(database, ToolRegistry(adapters={"create_report_artifact": _VerifiedAdapter()}), service)
    first = executor.execute(_request(requester, binding.approval_id, lease, key="key-1"))
    assert first.status == "succeeded"
    with pytest.raises(ApprovalServiceError, match="approval_not_approved"):
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
        ApprovalService(database).request(
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
    service = ApprovalService(database, now=lambda: clock[0])
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
        ApprovalService(database),
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
    service = _executor(database, ToolRegistry(adapters={"create_report_artifact": adapter}), ApprovalService(database))
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
    assert str(result.result["storage_key"]).startswith("artifacts/")
    assert result.result["content_hash"].startswith("sha256:")
    with pytest.raises(Exception, match="tool_arguments_invalid"):
        executor.execute(_request(requester, binding.approval_id, lease, {**_arguments(), "path": "C:/secret"}, "key-2"))
