"""A4：审批决定与恢复命令必须在同一数据库事务提交。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rag_kb.application.approval_service import (
    ApprovalService,
    ApprovalServiceError,
    principal_fingerprint,
)
from rag_kb.domain.identity import AuthContext, Principal
from rag_kb.domain.resume_commands import ApprovalResumeDecision, ResumeCommandStatus
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    PlanRevisionRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TenantRow,
)
from rag_kb.infrastructure.database.session import build_database
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskRepositoryError,
)
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import (
    build_approval_service,
    build_identity_provider,
    build_task_repository,
)


class _StableEvidenceValidator:
    """让本测试聚焦决定/命令事务，而不是重复覆盖 Evidence Ledger。"""

    def __init__(self, owner: AuthContext) -> None:
        self._owner = owner

    def validate_plan_in_session(self, _session, **_kwargs):
        return self._owner, SimpleNamespace()


@pytest.fixture()
def approval_case(tmp_path):
    database = build_database(
        f"sqlite+pysqlite:///{(tmp_path / 'approval-decision.sqlite3').as_posix()}"
    )
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    owner = AuthContext(
        tenant_id="tenant-decision",
        user_id="owner-decision",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id="owner-decision-request",
    )
    approver = AuthContext(
        tenant_id="tenant-decision",
        user_id="approver-decision",
        attributes={"permissions": ["task:approve"]},
        request_id="approver-decision-request",
    )
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id=owner.tenant_id, name="tenant-decision"))
        session.flush()
        session.add_all(
            (
                PrincipalRow(
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    roles=[],
                    attributes=owner.attributes,
                ),
                PrincipalRow(
                    tenant_id=approver.tenant_id,
                    user_id=approver.user_id,
                    roles=[],
                    attributes=approver.attributes,
                ),
            )
        )
        session.flush()
        session.add(
            AgentTaskRow(
                task_id="task-decision",
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                task_mode=TaskMode.CONTROLLED_ACTION.value,
                status=TaskStatus.WAITING_FOR_APPROVAL.value,
                goal="等待审批后继续",
                auth_context_ref="auth-decision",
                budgets={},
                deadline=now + timedelta(hours=1),
                required_capability_name="task-agent",
                required_protocol_version="v1",
                interrupt_kind="tool_approval",
                interrupt_summary="请审批写操作",
                interrupt_ref="approval-decision",
                interrupt_checkpoint_ns="approval-ns",
                interrupt_checkpoint_id="approval-checkpoint",
                interrupt_id="approval-interrupt",
                interrupt_approval_id="approval-decision",
                state_version=4,
            )
        )
        session.flush()
        session.add(
            PlanRevisionRow(
                plan_revision_id="plan-decision",
                tenant_id=owner.tenant_id,
                task_id="task-decision",
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
                approval_id="approval-decision",
                tenant_id=owner.tenant_id,
                task_id="task-decision",
                plan_revision=1,
                plan_digest="sha256:plan-decision",
                evidence_digest="sha256:evidence-decision",
                tool_name="create_report_artifact",
                tool_version="1.0.0",
                arguments_hash="sha256:arguments-decision",
                principal_hash=principal_fingerprint(owner),
                thread_id="task-decision",
                checkpoint_ns="approval-ns",
                checkpoint_id="approval-checkpoint",
                interrupt_id="approval-interrupt",
                tool_call_id="tool-call-decision",
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
    return database, service, repository, approver


def test_decision_enqueue_is_atomic_and_same_decision_reuses_command(approval_case) -> None:
    database, service, repository, approver = approval_case

    command = service.decide_and_enqueue(
        auth=approver,
        approval_id="approval-decision",
        expected_task_state_version=4,
        decision="approve",
        reason=None,
        repository=repository,
    )
    repeated = service.decide_and_enqueue(
        auth=approver,
        approval_id="approval-decision",
        expected_task_state_version=4,
        decision="approve",
        reason="不应新建命令",
        repository=repository,
    )

    assert repeated.command_id == command.command_id
    assert command.status is ResumeCommandStatus.PENDING
    assert command.decision is ApprovalResumeDecision.APPROVED
    with database.sessions() as session:
        approval = session.get(ApprovalRow, "approval-decision")
        rows = tuple(session.scalars(select(TaskResumeCommandRow)))
    assert approval is not None
    assert approval.status == "approved"
    assert approval.approved_by == approver.user_id
    assert [row.command_id for row in rows] == [command.command_id]

    with pytest.raises(ApprovalServiceError, match="approval_decision_conflict"):
        service.decide_and_enqueue(
            auth=approver,
            approval_id="approval-decision",
            expected_task_state_version=4,
            decision="reject",
            reason=None,
            repository=repository,
        )


def test_decision_api_returns_accepted_only_after_durable_enqueue(approval_case) -> None:
    database, service, repository, approver = approval_case
    app = create_app()
    provider = DemoTokenIdentityProvider(
        {
            "approver-token": Principal(
                tenant_id=approver.tenant_id,
                user_id=approver.user_id,
                attributes=approver.attributes,
            )
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_approval_service] = lambda: service
    app.dependency_overrides[build_task_repository] = lambda: repository
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/approvals/approval-decision/decision",
                headers={"Authorization": "Bearer approver-token"},
                json={"decision": "approve", "expected_task_state_version": 4},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "approved"
    assert payload["resume_command"]["status"] == "accepted"
    with database.sessions() as session:
        command = session.scalar(select(TaskResumeCommandRow))
    assert command is not None
    assert command.command_id == payload["resume_command"]["command_id"]


def test_decided_history_without_command_is_rechecked_before_supplement(approval_case) -> None:
    database, service, repository, approver = approval_case
    with database.sessions.begin() as session:
        approval = session.get(ApprovalRow, "approval-decision")
        assert approval is not None
        approval.status = "approved"
        approval.approved_by = approver.user_id
        approval.approved_at = datetime.now(UTC)

    command = service.decide_and_enqueue(
        auth=approver,
        approval_id="approval-decision",
        expected_task_state_version=4,
        decision="approve",
        reason=None,
        repository=repository,
    )

    assert command.status is ResumeCommandStatus.PENDING
    assert command.decision is ApprovalResumeDecision.APPROVED
    with database.sessions() as session:
        approval = session.get(ApprovalRow, "approval-decision")
        rows = tuple(session.scalars(select(TaskResumeCommandRow)))
    assert approval is not None
    assert approval.status == "approved"
    assert [row.command_id for row in rows] == [command.command_id]


def test_decision_enqueue_failure_rolls_back_decision_and_command(
    approval_case, monkeypatch
) -> None:
    database, service, repository, approver = approval_case
    original_enqueue = repository.enqueue_resume_command_in_session

    def fail_after_insert(session, **kwargs):
        original_enqueue(session, **kwargs)
        raise TaskRepositoryError("task_resume_conflict")

    monkeypatch.setattr(repository, "enqueue_resume_command_in_session", fail_after_insert)

    with pytest.raises(TaskRepositoryError, match="task_resume_conflict"):
        service.decide_and_enqueue(
            auth=approver,
            approval_id="approval-decision",
            expected_task_state_version=4,
            decision="reject",
            reason="故障注入原因",
            repository=repository,
        )

    with database.sessions() as session:
        approval = session.get(ApprovalRow, "approval-decision")
        command = session.scalar(select(TaskResumeCommandRow))
    assert approval is not None
    assert approval.status == "pending"
    assert approval.approved_by is None
    assert approval.rejected_by is None
    assert command is None


def test_decision_cross_checks_approval_binding_against_locked_task(approval_case) -> None:
    database, service, repository, approver = approval_case
    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, "task-decision")
        assert task is not None
        task.interrupt_checkpoint_id = "different-checkpoint"

    with pytest.raises(TaskRepositoryError, match="task_resume_kind_invalid"):
        service.decide_and_enqueue(
            auth=approver,
            approval_id="approval-decision",
            expected_task_state_version=4,
            decision="approve",
            reason=None,
            repository=repository,
        )

    with database.sessions() as session:
        approval = session.get(ApprovalRow, "approval-decision")
        command = session.scalar(select(TaskResumeCommandRow))
    assert approval is not None
    assert approval.status == "pending"
    assert command is None
