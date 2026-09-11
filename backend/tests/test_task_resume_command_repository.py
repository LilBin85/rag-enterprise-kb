"""A2 恢复命令 inbox 的 SQLite 事务与隐私边界。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_kb.application.approval_service import ApprovalService, principal_fingerprint
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.resume_commands import ApprovalResumeDecision, ResumeCommandStatus
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    PlanRevisionRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TenantRow,
)
from rag_kb.infrastructure.database.approval_lifecycle import DatabaseApprovalLifecycle
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    RESEARCH_AGENT_CAPABILITY_NAME,
    RESEARCH_AGENT_PROTOCOL_VERSION,
    TaskRepositoryError,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 31, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


class _StableEvidenceValidator:
    def validate_plan_in_session(self, _session, **_kwargs):
        return auth(), SimpleNamespace()


def approval_service(
    database, repository: DatabaseTaskRepository, clock: Clock
) -> ApprovalService:
    return ApprovalService(
        DatabaseApprovalLifecycle(
            database,
            evidence_validator=_StableEvidenceValidator(),
            resume_repository=repository,
            now=clock,
        ),
        now=clock,
    )


@pytest.fixture()
def database(tmp_path: Path):
    database = build_database(
        f"sqlite+pysqlite:///{(tmp_path / 'resume-commands.sqlite3').as_posix()}"
    )
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-resume", name="tenant-resume"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-resume",
                user_id="owner-resume",
                roles=[],
                attributes={"permissions": ["task:create", "task:read"]},
            )
        )
        session.add(
            PrincipalRow(
                tenant_id="tenant-resume",
                user_id="approver-resume",
                roles=[],
                attributes={"permissions": ["task:approve"]},
            )
        )
    return database


def auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-resume",
        user_id="owner-resume",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id="request-resume",
    )


def approver_auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-resume",
        user_id="approver-resume",
        attributes={"permissions": ["task:approve"]},
        request_id="request-approver-resume",
    )


def paused_user_task(repository: DatabaseTaskRepository, database) -> None:
    repository.create_task(
        auth=auth(),
        goal="需要用户确认范围",
        task_mode=TaskMode.RESEARCH,
        task_id="task-resume",
    )
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "task-resume")
        assert row is not None
        row.status = TaskStatus.WAITING_FOR_USER.value
        row.interrupt_kind = "user_clarification"
        row.interrupt_summary = "请选择部门"
        row.interrupt_ref = None
        row.interrupt_checkpoint_ns = "interrupt-ns"
        row.interrupt_checkpoint_id = "checkpoint-1"
        row.interrupt_id = "interrupt-1"
        row.interrupt_approval_id = None
        row.lease_owner = None
        row.lease_expires_at = None


def paused_approval_task(repository: DatabaseTaskRepository, database, clock: Clock) -> None:
    repository.create_task(
        auth=auth(),
        goal="需要审批后继续",
        task_mode=TaskMode.CONTROLLED_ACTION,
        task_id="task-approval-resume",
    )
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "task-approval-resume")
        assert row is not None
        row.status = TaskStatus.WAITING_FOR_APPROVAL.value
        row.interrupt_kind = "tool_approval"
        row.interrupt_summary = "请审批此操作"
        row.interrupt_ref = "approval-resume"
        row.interrupt_checkpoint_ns = "approval-ns"
        row.interrupt_checkpoint_id = "approval-checkpoint"
        row.interrupt_id = "approval-interrupt"
        row.interrupt_approval_id = "approval-resume"
        row.lease_owner = None
        row.lease_expires_at = None
        session.add(
            PlanRevisionRow(
                plan_revision_id="plan-approval-resume",
                tenant_id="tenant-resume",
                task_id=row.task_id,
                revision=1,
                schema_version=1,
                plan={},
                plan_digest="sha256:plan-approval-resume",
                evidence_digest="sha256:evidence-approval-resume",
                status="active",
                created_at=clock.value - timedelta(minutes=1),
            )
        )
        session.flush()
        session.add(
            ApprovalRow(
                approval_id="approval-resume",
                tenant_id="tenant-resume",
                task_id=row.task_id,
                plan_revision=1,
                plan_digest="sha256:plan-approval-resume",
                evidence_digest="sha256:evidence-approval-resume",
                tool_name="create_report_artifact",
                tool_version="v1",
                arguments_hash="sha256:arguments-approval-resume",
                principal_hash=principal_fingerprint(auth()),
                thread_id=row.task_id,
                checkpoint_ns="approval-ns",
                checkpoint_id="approval-checkpoint",
                interrupt_id="approval-interrupt",
                tool_call_id="tool-call-approval-resume",
                checkpoint_binding_hash="sha256:checkpoint-approval-resume",
                status="pending",
                expires_at=clock.value + timedelta(hours=1),
                created_at=clock.value - timedelta(minutes=1),
            )
        )


def test_enqueue_is_idempotent_conflict_safe_and_hides_body(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_user_task(repository, database)

    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="财务部门",
    )
    repeated = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer=" 财务部门 ",
    )

    assert repeated.command_id == command.command_id
    assert "财务部门" not in repr(command)
    assert command.status is ResumeCommandStatus.PENDING
    assert repository.resume_command_candidates(limit=1)[0].command_id == command.command_id
    with pytest.raises(TaskRepositoryError, match="task_resume_conflict"):
        repository.enqueue_user_resume_command(
            auth=auth(),
            task_id="task-resume",
            expected_state_version=1,
            answer="法务部门",
        )
    with database.sessions() as session:
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert row is not None
        assert row.payload_body == "财务部门"
        task = session.get(AgentTaskRow, "task-resume")
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_USER.value
        assert task.lease_owner is None
        assert task.active_resume_command_id is None


def test_claim_reclaim_and_renew_keep_task_and_command_fences_together(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="财务部门",
    )

    first = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id="worker-first",
        lease_seconds=10,
    )
    assert first is not None
    assert first.command.status is ResumeCommandStatus.CLAIMED
    assert first.command.claimed_task_state_version == 2
    assert first.task_lease.state.status is TaskStatus.RUNNING
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-resume")
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert task is not None and row is not None
        assert task.active_resume_command_id == command.command_id
        assert task.lease_owner == row.lease_owner == "worker-first"
        assert task.lease_expires_at == row.lease_expires_at
        assert task.lease_generation == row.claimed_task_lease_generation
        assert task.interrupt_checkpoint_id is None

    renewed = repository.refresh_resume_command_lease(first, lease_seconds=20)
    assert renewed.command.lease_generation == first.command_lease_generation
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-resume")
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert task is not None and row is not None
        assert task.lease_expires_at == row.lease_expires_at

    clock.value += timedelta(seconds=21)
    recovered = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id="worker-recovery",
        lease_seconds=30,
    )
    assert recovered is not None
    assert recovered.task_lease.state.state_version == first.task_lease.state.state_version
    assert recovered.task_lease.lease_generation == first.task_lease.lease_generation + 1
    assert recovered.command_lease_generation == first.command_lease_generation + 1
    with pytest.raises(TaskRepositoryError, match="task_resume_lease_stale"):
        repository.refresh_resume_command_lease(first, lease_seconds=30)


def test_explicit_capability_claim_requires_matching_active_worker_lease(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="仅供能力校验的答案",
    )
    worker_id = "worker-capability"

    repository.register_capability(
        capability_name="other-task-agent",
        protocol_version="v1",
        worker_id=worker_id,
        lease_seconds=30,
    )
    assert repository.claim_resume_command(
        command_id=command.command_id,
        worker_id=worker_id,
        lease_seconds=30,
        capability_name="other-task-agent",
        protocol_version="v1",
    ) is None
    repository.register_capability(
        capability_name=RESEARCH_AGENT_CAPABILITY_NAME,
        protocol_version=RESEARCH_AGENT_PROTOCOL_VERSION,
        worker_id=worker_id,
        lease_seconds=30,
    )
    lease = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id=worker_id,
        lease_seconds=30,
        capability_name=RESEARCH_AGENT_CAPABILITY_NAME,
        protocol_version=RESEARCH_AGENT_PROTOCOL_VERSION,
    )
    assert lease is not None


def test_claimed_resume_revalidates_the_actor_at_execution_time(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="撤权后的答案",
    )
    lease = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id="worker-auth-recheck",
        lease_seconds=30,
    )
    assert lease is not None
    with database.sessions.begin() as session:
        principal = session.get(PrincipalRow, ("tenant-resume", "owner-resume"))
        assert principal is not None
        principal.active = False

    with pytest.raises(TaskRepositoryError, match="task_resume_actor_unavailable"):
        repository.validate_resume_command_execution(lease)


def test_consume_purges_sensitive_body_after_committed_checkpoint_marker(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="仅此一次的答案",
    )
    lease = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id="worker-consume",
        lease_seconds=30,
    )
    assert lease is not None

    # 真实 marker 与 FencedPostgresSaver 同事务写入的语义由集成测试覆盖；
    # 此处只布置其已提交结果，验证 consume 的正文清理。
    with database.sessions.begin() as session:
        row = session.get(TaskResumeCommandRow, command.command_id)
        task = session.get(AgentTaskRow, "task-resume")
        assert row is not None and task is not None
        now = datetime.now(UTC)
        row.applied_checkpoint_id = "checkpoint-2"
        row.applied_at = now
        row.updated_at = now
        task.active_resume_command_id = None
        task.updated_at = now
    consumed = repository.consume_resume_command(lease)
    assert consumed.status is ResumeCommandStatus.CONSUMED
    assert consumed.payload_body is None
    assert "仅此一次的答案" not in repr(consumed)
    with database.sessions() as session:
        row = session.get(TaskResumeCommandRow, command.command_id)
        task = session.get(AgentTaskRow, "task-resume")
        assert row is not None and task is not None
        assert row.payload_body is None
        assert row.body_purged_at is not None
        assert row.applied_checkpoint_id == "checkpoint-2"
    assert task.active_resume_command_id is None


def test_cancel_terminalizes_an_active_resume_command_without_leaking_body(
    database, monkeypatch
) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    monkeypatch.setattr(repository, "_task_result_retention_days", lambda: 7)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="取消前的答案",
    )
    lease = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id="worker-cancel",
        lease_seconds=30,
    )
    assert lease is not None

    cancelled = repository.cancel(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=lease.task_lease.state.state_version,
    )

    assert cancelled.status is TaskStatus.CANCELLED
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-resume")
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert task is not None and row is not None
        assert task.active_resume_command_id is None
        assert row.status == ResumeCommandStatus.FAILED.value
        assert row.failure_code == "task_cancelled"
        assert row.payload_body is None


def test_cancel_purges_a_pending_resume_command(database, monkeypatch) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    monkeypatch.setattr(repository, "_task_result_retention_days", lambda: 7)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="待取消的答案",
    )

    repository.cancel(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
    )

    with database.sessions() as session:
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert row is not None
        assert row.status == ResumeCommandStatus.FAILED.value
        assert row.payload_body is None
        assert row.body_purged_at is not None


def test_waiting_task_expiry_purges_its_pending_resume_command(database, monkeypatch) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    monkeypatch.setattr(repository, "_task_result_retention_days", lambda: 7)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="待过期的答案",
    )
    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, "task-resume")
        assert task is not None
        task.deadline = clock.value

    assert repository.expire_waiting_tasks() == 1
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-resume")
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert task is not None and row is not None
        assert task.status == TaskStatus.FAILED.value
        assert row.status == ResumeCommandStatus.EXPIRED.value
        assert row.payload_body is None


def test_approval_enqueue_uses_server_side_binding_and_hides_reason(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_approval_task(repository, database, clock)

    command = approval_service(database, repository, clock).decide_and_enqueue(
        auth=approver_auth(),
        approval_id="approval-resume",
        expected_task_state_version=1,
        decision="approve",
        reason="审批原因",
    )

    assert command.status is ResumeCommandStatus.PENDING
    assert command.kind.value == "tool_approval"
    assert command.approval_id == "approval-resume"
    assert command.decision is ApprovalResumeDecision.APPROVED
    assert "审批原因" not in repr(command)


@pytest.mark.parametrize("parent_kind", ("task", "tenant"))
def test_approval_resume_command_cascades_with_its_parent_chain(database, parent_kind) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    paused_approval_task(repository, database, clock)
    command = approval_service(database, repository, clock).decide_and_enqueue(
        auth=approver_auth(),
        approval_id="approval-resume",
        expected_task_state_version=1,
        decision="approve",
        reason=None,
    )

    with database.sessions.begin() as session:
        if parent_kind == "task":
            parent = session.get(AgentTaskRow, "task-approval-resume")
        else:
            parent = session.get(TenantRow, "tenant-resume")
        assert parent is not None
        session.delete(parent)

    with database.sessions() as session:
        assert session.get(TaskResumeCommandRow, command.command_id) is None


def test_expired_claim_is_settled_without_replaying_sensitive_body(database, monkeypatch) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    monkeypatch.setattr(repository, "_task_result_retention_days", lambda: 7)
    paused_user_task(repository, database)
    command = repository.enqueue_user_resume_command(
        auth=auth(),
        task_id="task-resume",
        expected_state_version=1,
        answer="等待重试的答案",
    )
    lease = repository.claim_resume_command(
        command_id=command.command_id,
        worker_id="worker-expiry",
        lease_seconds=30,
    )
    assert lease is not None
    expired = repository.expire_claimed_resume_command(lease)
    assert expired.status is ResumeCommandStatus.EXPIRED
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-resume")
        row = session.get(TaskResumeCommandRow, command.command_id)
        assert task is not None and row is not None
        assert task.status == TaskStatus.FAILED.value
        assert task.active_resume_command_id is None
        assert row.status == ResumeCommandStatus.EXPIRED.value
        assert row.payload_body is None
