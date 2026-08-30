"""Stage 5 在真实 PostgreSQL 上的批准与写工具协议回归。"""

from __future__ import annotations

import os
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from rag_kb.application.approval_service import ApprovalService, ApprovalServiceError
from rag_kb.application.tool_execution_service import ToolExecutionRequest, ToolExecutionService
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tasks import TaskBudgets, TaskMode
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ACLGrantRow,
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


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


def _prepare_protocol_stage(tmp_path: Path) -> SimpleNamespace:
    """唯一后缀的受控任务 + 计划绑定 + 已批准审批（真实 PG 行）。"""
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"tool-protocol-{suffix}"
    requester_id = f"requester-{suffix}"
    approver_id = f"approver-{suffix}"
    task_id = f"task-{suffix}"
    object_root = tmp_path / f"tool-protocol-objects-{suffix}"
    requester = AuthContext(
        tenant_id=tenant_id,
        user_id=requester_id,
        attributes={
            "permissions": [
                "task:create",
                "artifact:write",
                "knowledge:read",
                "document:read",
            ]
        },
        request_id=f"request-{suffix}",
    )
    approver = AuthContext(
        tenant_id=tenant_id,
        user_id=approver_id,
        attributes={"permissions": ["task:approve"]},
        request_id=f"approve-{suffix}",
    )
    arguments: dict[str, object] = {
        "title": "阶段五真实协议报告",
        "content": "仅用于隔离的 PostgreSQL 组件测试。",
        "format": "markdown",
    }
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (tenant_id, name, active, created_at) "
                "VALUES (:tenant, :name, TRUE, CURRENT_TIMESTAMP)"
            ),
            {"tenant": tenant_id, "name": tenant_id},
        )
        for user_id, attributes in (
            (requester_id, requester.attributes),
            (approver_id, approver.attributes),
        ):
            connection.execute(
                text(
                    "INSERT INTO principals "
                    "(tenant_id, user_id, roles, attributes, active, created_at) "
                    "VALUES (:tenant, :user, CAST('[]' AS jsonb), "
                    "CAST(:attributes AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant": tenant_id, "user": user_id, "attributes": json.dumps(attributes)},
            )

    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=requester,
        goal="创建隔离报告",
        task_mode=TaskMode.CONTROLLED_ACTION,
        task_id=task_id,
    )
    evidence_id = f"evidence-{suffix}"
    evidence_snapshot = canonical_evidence_snapshot((evidence_id,), require_non_empty=True)
    plan = {"steps": ["report"], "evidence_ids": [evidence_id]}
    plan_digest = canonical_fingerprint("task_plan", plan)
    with UnitOfWork(database) as work:
        assert work.session is not None
        # 使用明确的计划绑定，避免执行端从未冻结的计划读取参数。
        work.session.add_all(
            [
                DocumentRow(tenant_id=tenant_id, document_id=f"doc-{suffix}", title="隔离制度", department="安全", status="active", created_by=requester_id, projection_generation=1, applied_projection_generation=1, applied_projection_version_id=f"version-{suffix}"),
                ACLGrantRow(grant_id=f"grant-{suffix}", tenant_id=tenant_id, resource_type="document", resource_id=f"doc-{suffix}", subject_type="user", subject_id=requester_id, permissions=["knowledge:read", "document:read"]),
            ]
        )
        work.session.flush()
        work.session.add(DocumentVersionRow(version_id=f"version-{suffix}", tenant_id=tenant_id, document_id=f"doc-{suffix}", version=1, lifecycle_status="active", source_object_key=f"source-{suffix}", source_file_name="isolated.md", content_hash="sha256:content", index_version="kb-v1", desired_projection_generation=1, applied_projection_generation=1))
        work.session.flush()
        work.session.add_all(
            [
                ChunkMetadataRow(chunk_id=f"chunk-{suffix}", version_id=f"version-{suffix}", tenant_id=tenant_id, document_id=f"doc-{suffix}", document_version=1, content_hash="sha256:chunk", source_locator="isolated/1", source_file_name="isolated.md", index_version="kb-v1"),
                IngestionJobRow(job_id=f"job-{suffix}", tenant_id=tenant_id, version_id=f"version-{suffix}", status="succeeded", indexed_chunk_count=1, index_manifest_digest="sha256:manifest", index_completed_at=datetime.now(UTC), projection_generation=1, projection_lease_generation=1),
            ]
        )
        work.session.flush()
        work.session.add_all(
            [
                EvidenceRecordRow(evidence_id=evidence_id, task_id=task_id, query_fingerprint="sha256:query", tenant_id=tenant_id, document_id=f"doc-{suffix}", version_id=f"version-{suffix}", document_version=1, chunk_id=f"chunk-{suffix}", content_hash="sha256:chunk", source_locator="isolated/1", permission_granted=True, supported_claim_ids=[]),
                PlanRevisionRow(
                plan_revision_id=f"plan-{suffix}",
                task_id=task_id,
                tenant_id=tenant_id,
                revision=1,
                schema_version=1,
                plan=plan,
                plan_digest=plan_digest,
                evidence_digest=evidence_snapshot.digest,
                status="ready",
                ),
            ]
        )
        task = work.session.scalar(
            select(AgentTaskRow).where(
                AgentTaskRow.tenant_id == tenant_id,
                AgentTaskRow.task_id == task_id,
            )
        )
        assert task is not None
        task.status = "running"
        task.lease_owner = f"worker-{suffix}"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=300)
        task.lease_generation = 1

    lease = TaskLease(
        repository.get_task(auth=requester, task_id=task_id),
        f"worker-{suffix}",
        1,
        TaskBudgets(),
    )

    approvals = ApprovalService(database)
    pending = approvals.request(
        auth=requester,
        task_id=task_id,
        plan_revision=1,
        plan_digest=plan_digest,
        evidence_digest=evidence_snapshot.digest,
        evidence_ids=(evidence_id,),
        tool_name="create_report_artifact",
        tool_version="1.0.0",
        arguments=arguments,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    approved = approvals.approve(auth=approver, approval_id=pending.approval_id)
    return SimpleNamespace(
        database=database,
        suffix=suffix,
        tenant_id=tenant_id,
        task_id=task_id,
        object_root=object_root,
        requester=requester,
        approver=approver,
        arguments=arguments,
        evidence_id=evidence_id,
        evidence_digest=evidence_snapshot.digest,
        plan_digest=plan_digest,
        lease=lease,
        approvals=approvals,
        approved=approved,
    )


def _cleanup_protocol_stage(env: SimpleNamespace) -> None:
    try:
        with env.database.engine.begin() as connection:
            for table in ("artifacts", "tool_calls", "approvals"):
                connection.execute(
                    text(f"DELETE FROM {table} WHERE tenant_id = :tenant"),
                    {"tenant": env.tenant_id},
                )
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": env.tenant_id},
            )
    finally:
        env.database.engine.dispose()
        shutil.rmtree(env.object_root, ignore_errors=True)


def _make_executor(env: SimpleNamespace, adapter) -> ToolExecutionService:
    return ToolExecutionService(
        env.database,
        ToolRegistry(adapters={"create_report_artifact": adapter}),
        env.approvals,
        tool_results=LocalObjectStore(env.object_root / "tool-results"),
    )


def _protocol_request(env: SimpleNamespace, key: str | None = None) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        auth=env.requester,
        task_id=env.task_id,
        plan_revision=1,
        plan_digest=env.plan_digest,
        evidence_digest=env.evidence_digest,
        evidence_ids=(env.evidence_id,),
        tool_name="create_report_artifact",
        arguments=dict(env.arguments),
        worker_id=env.lease.worker_id,
        lease_generation=env.lease.lease_generation,
        idempotency_key=key or f"server-key-{env.suffix}",
        approval_id=env.approved.approval_id,
    )


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_approved_write_is_idempotent_and_consumed(tmp_path: Path) -> None:
    """真实事务中，同一批准只能产生一个可验证的写工具结果。"""
    env = _prepare_protocol_stage(tmp_path)
    try:
        executor = _make_executor(
            env,
            LocalArtifactToolAdapter(
                env.database,
                LocalObjectStore(env.object_root),
                artifact_type="report",
            ),
        )
        request = _protocol_request(env)
        first = executor.execute(request)
        duplicate = executor.execute(request)

        assert first.status == "succeeded"
        assert duplicate.status == "succeeded"
        assert duplicate.duplicate is True
        assert duplicate.tool_call_id == first.tool_call_id
        assert first.result is not None
        # 批次 2：结果面只暴露 artifact_id/hash/status，storage_key 不得出现。
        artifact_id = first.result["artifact_id"]
        assert isinstance(artifact_id, str) and artifact_id
        assert "storage_key" not in first.result
        assert duplicate.result is not None
        assert duplicate.result["artifact_id"] == artifact_id
        assert "storage_key" not in duplicate.result

        with pytest.raises(ApprovalServiceError, match="approval_not_approved"):
            executor.execute(_protocol_request(env, key=f"second-key-{env.suffix}"))

        with env.database.sessions() as session:
            tool_call = session.get(ToolCallRow, first.tool_call_id)
            artifact = session.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.tenant_id == env.tenant_id,
                    ArtifactRow.tool_call_id == first.tool_call_id,
                )
            )
            approval = session.get(ApprovalRow, env.approved.approval_id)
            task = session.get(AgentTaskRow, env.task_id)
            assert tool_call is not None
            assert tool_call.status == "succeeded"
            assert tool_call.idempotency_key == request.idempotency_key
            assert tool_call.result_hash is not None and tool_call.completed_at is not None
            assert tool_call.result_ref
            assert artifact is not None and artifact.status == "ready"
            assert artifact.artifact_id == artifact_id
            assert artifact.content_hash == first.result["content_hash"]
            assert artifact.storage_key.startswith("artifacts/")
            assert approval is not None and approval.status == "executed"
            assert approval.checkpoint_binding_hash is None
            # result_ref 与工具/审批/artifact 行绑定一致；任务引用只含 artifact ID。
            assert task is not None
            assert task.artifact_refs == [artifact_id]
    finally:
        _cleanup_protocol_stage(env)


class _CountingArtifactAdapter:
    def __init__(self, inner: LocalArtifactToolAdapter) -> None:
        self._inner = inner
        self.calls = 0

    def execute(self, arguments, context):  # noqa: ANN001
        self.calls += 1
        return self._inner.execute(arguments, context)

    def verify(self, result, context):  # noqa: ANN001
        return self._inner.verify(result, context)


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
@pytest.mark.parametrize("corruption", ("tampered", "missing"))
def test_postgres_result_ref_corruption_fails_closed(
    tmp_path: Path, corruption: str
) -> None:
    """result_ref 损坏：hash 不符/对象缺失 fail-closed；不猜测正文，不执行第二次。"""
    env = _prepare_protocol_stage(tmp_path)
    try:
        adapter = _CountingArtifactAdapter(
            LocalArtifactToolAdapter(
                env.database,
                LocalObjectStore(env.object_root),
                artifact_type="report",
            )
        )
        executor = _make_executor(env, adapter)
        store = LocalObjectStore(env.object_root / "tool-results")
        request = _protocol_request(env)

        first = executor.execute(request)
        assert first.status == "succeeded"
        assert first.result_ref and first.result_hash
        assert adapter.calls == 1

        if corruption == "tampered":
            store.delete_exact(first.result_ref)
            store.put_exclusive(first.result_ref, b'{"tampered": true}')
        else:
            store.delete_exact(first.result_ref)

        replay = executor.execute(request)

        # WRITE：无法证明原正文即 UNKNOWN；不凭 hash 伪装成功，不再调用 adapter。
        assert replay.status == "unknown"
        assert replay.duplicate is True
        assert replay.result is None
        assert replay.error_code == "tool_result_readback_failed"
        assert replay.tool_call_id == first.tool_call_id
        assert adapter.calls == 1

        with env.database.sessions() as session:
            rows = session.scalars(
                select(ToolCallRow).where(ToolCallRow.task_id == env.task_id)
            ).all()
            assert len(rows) == 1
            assert rows[0].result_ref == first.result_ref
    finally:
        _cleanup_protocol_stage(env)
