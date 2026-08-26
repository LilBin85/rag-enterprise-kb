"""Stage 5 在真实 PostgreSQL 上的批准与写工具协议回归。"""

from __future__ import annotations

import os
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
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


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_approved_write_is_idempotent_and_consumed(tmp_path: Path) -> None:
    """真实事务中，同一批准只能产生一个可验证的写工具结果。"""
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
    try:
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
            task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=60)
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
        executor = ToolExecutionService(
            database,
            ToolRegistry(
                adapters={
                    "create_report_artifact": LocalArtifactToolAdapter(
                        database,
                        LocalObjectStore(object_root),
                        artifact_type="report",
                    )
                }
            ),
            approvals,
            tool_results=LocalObjectStore(object_root / "tool-results"),
        )
        request = ToolExecutionRequest(
            auth=requester,
            task_id=task_id,
            plan_revision=1,
            plan_digest=plan_digest,
            evidence_digest=evidence_snapshot.digest,
            evidence_ids=(evidence_id,),
            tool_name="create_report_artifact",
            arguments=arguments,
            worker_id=lease.worker_id,
            lease_generation=lease.lease_generation,
            idempotency_key=f"server-key-{suffix}",
            approval_id=approved.approval_id,
        )
        first = executor.execute(request)
        duplicate = executor.execute(request)

        assert first.status == "succeeded"
        assert duplicate.status == "succeeded"
        assert duplicate.duplicate is True
        assert duplicate.tool_call_id == first.tool_call_id
        assert first.result is not None
        assert str(first.result["storage_key"]).startswith("artifacts/")

        with pytest.raises(ApprovalServiceError, match="approval_not_approved"):
            executor.execute(replace(request, idempotency_key=f"second-key-{suffix}"))

        with database.sessions() as session:
            tool_call = session.get(ToolCallRow, first.tool_call_id)
            artifact = session.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.tenant_id == tenant_id,
                    ArtifactRow.tool_call_id == first.tool_call_id,
                )
            )
            approval = session.get(ApprovalRow, approved.approval_id)
            assert tool_call is not None
            assert tool_call.status == "succeeded"
            assert tool_call.idempotency_key == request.idempotency_key
            assert tool_call.result_hash is not None and tool_call.completed_at is not None
            assert artifact is not None and artifact.status == "ready"
            assert artifact.content_hash == first.result["content_hash"]
            assert approval is not None and approval.status == "executed"
    finally:
        try:
            with database.engine.begin() as connection:
                for table in ("artifacts", "tool_calls", "approvals"):
                    connection.execute(
                        text(f"DELETE FROM {table} WHERE tenant_id = :tenant"),
                        {"tenant": tenant_id},
                    )
                connection.execute(
                    text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                    {"tenant": tenant_id},
                )
        finally:
            database.engine.dispose()
            shutil.rmtree(object_root, ignore_errors=True)
