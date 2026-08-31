"""真实 PostgreSQL 的 A4 审批决定/恢复命令原子回滚验证。"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from rag_kb.application.approval_service import (
    ApprovalService,
    principal_fingerprint,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    PlanRevisionRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TenantRow,
)
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskRepositoryError,
)


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


class _StableEvidenceValidator:
    def __init__(self, owner: AuthContext) -> None:
        self._owner = owner

    def validate_plan_in_session(self, _session, **_kwargs):
        return self._owner, SimpleNamespace()


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_decision_enqueue_failure_rolls_back_both_rows(monkeypatch) -> None:
    """命令插入后故障时，approval 也必须保持 pending。"""
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"approval-enqueue-{suffix}"
    owner = AuthContext(
        tenant_id=tenant_id,
        user_id=f"owner-{suffix}",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"owner-request-{suffix}",
    )
    approver = AuthContext(
        tenant_id=tenant_id,
        user_id=f"approver-{suffix}",
        attributes={"permissions": ["task:approve"]},
        request_id=f"approver-request-{suffix}",
    )
    task_id = f"task-{suffix}"
    approval_id = f"approval-{suffix}"
    now = datetime.now(UTC)
    try:
        with database.sessions.begin() as session:
            session.add(TenantRow(tenant_id=tenant_id, name=tenant_id))
            session.flush()
            session.add_all(
                (
                    PrincipalRow(
                        tenant_id=tenant_id,
                        user_id=owner.user_id,
                        roles=[],
                        attributes=owner.attributes,
                    ),
                    PrincipalRow(
                        tenant_id=tenant_id,
                        user_id=approver.user_id,
                        roles=[],
                        attributes=approver.attributes,
                    ),
                )
            )
            session.flush()
            session.add(
                AgentTaskRow(
                    task_id=task_id,
                    tenant_id=tenant_id,
                    user_id=owner.user_id,
                    task_mode=TaskMode.CONTROLLED_ACTION.value,
                    status=TaskStatus.WAITING_FOR_APPROVAL.value,
                    goal="PostgreSQL 原子回滚",
                    auth_context_ref="approval-enqueue-auth",
                    budgets={},
                    deadline=now + timedelta(hours=1),
                    required_capability_name="task-agent",
                    required_protocol_version="v1",
                    interrupt_kind="tool_approval",
                    interrupt_summary="请审批写操作",
                    interrupt_ref=approval_id,
                    interrupt_checkpoint_ns="approval-ns",
                    interrupt_checkpoint_id="approval-checkpoint",
                    interrupt_id="approval-interrupt",
                    interrupt_approval_id=approval_id,
                    state_version=4,
                )
            )
            session.flush()
            session.add(
                PlanRevisionRow(
                    plan_revision_id=f"plan-{suffix}",
                    tenant_id=tenant_id,
                    task_id=task_id,
                    revision=1,
                    schema_version=1,
                    plan={"evidence_ids": ["evidence-decision"]},
                    plan_digest="sha256:plan-decision",
                    evidence_digest="sha256:evidence-decision",
                    status="ready",
                    created_at=now,
                )
            )
            session.flush()
            session.add(
                ApprovalRow(
                    approval_id=approval_id,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    plan_revision=1,
                    plan_digest="sha256:plan-decision",
                    evidence_digest="sha256:evidence-decision",
                    tool_name="create_report_artifact",
                    tool_version="1.0.0",
                    arguments_hash="sha256:arguments-decision",
                    principal_hash=principal_fingerprint(owner),
                    thread_id=task_id,
                    checkpoint_ns="approval-ns",
                    checkpoint_id="approval-checkpoint",
                    interrupt_id="approval-interrupt",
                    tool_call_id=f"tool-call-{suffix}",
                    checkpoint_binding_hash="sha256:checkpoint-decision",
                    status="pending",
                    expires_at=now + timedelta(minutes=10),
                    created_at=now,
                )
            )

        repository = DatabaseTaskRepository(database)
        service = ApprovalService(
            database,
            evidence_validator=_StableEvidenceValidator(owner),
        )
        original_enqueue = repository.enqueue_resume_command_in_session

        def fail_after_insert(session, **kwargs):
            original_enqueue(session, **kwargs)
            raise TaskRepositoryError("task_resume_conflict")

        monkeypatch.setattr(repository, "enqueue_resume_command_in_session", fail_after_insert)
        with pytest.raises(TaskRepositoryError, match="task_resume_conflict"):
            service.decide_and_enqueue(
                auth=approver,
                approval_id=approval_id,
                expected_task_state_version=4,
                decision="approve",
                reason=None,
                repository=repository,
            )

        with database.sessions() as session:
            approval = session.get(ApprovalRow, approval_id)
            command = session.scalar(
                select(TaskResumeCommandRow).where(
                    TaskResumeCommandRow.tenant_id == tenant_id
                )
            )
        assert approval is not None
        assert approval.status == "pending"
        assert approval.approved_by is None
        assert command is None
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
        database.engine.dispose()
