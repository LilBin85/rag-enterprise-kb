"""M1 agent_tasks 业务生命周期、租约和公开状态契约。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    AgentCapabilityLeaseRow,
    AgentTaskRow,
    PrincipalRow,
    TaskEventRow,
    TenantRow,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


@pytest.fixture()
def database(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'tasks.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="user-a", roles=[], attributes={}))
    return database


def auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="user-a",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id="request-a",
    )


def created(repository: DatabaseTaskRepository):
    return repository.create_task(
        auth=auth(),
        goal="研究采购审批制度",
        task_mode=TaskMode.RESEARCH,
        task_id="task-a",
    )


def test_create_claim_and_expired_lease_fence_old_worker(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    initial = created(repository)
    assert initial.deadline == clock.value + timedelta(seconds=600)

    first = repository.claim_next(worker_id="worker-a", lease_seconds=1)
    assert first is not None and first.state.status is TaskStatus.RUNNING
    clock.value += timedelta(seconds=2)
    second = repository.claim_next(worker_id="worker-b", lease_seconds=30)
    assert second is not None
    assert second.lease_generation == first.lease_generation + 1
    with pytest.raises(TaskRepositoryError, match="task_lease_generation_stale"):
        repository.refresh_lease(first)


def test_generic_worker_only_claims_research_tasks(database) -> None:
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=auth(),
        goal="同步知识问答",
        task_mode=TaskMode.KNOWLEDGE_QUERY,
        task_id="query-task",
    )

    assert repository.claim_next(worker_id="worker-a", lease_seconds=30) is None
    assert repository.get_task(auth=auth(), task_id="query-task").task_mode is TaskMode.KNOWLEDGE_QUERY


def test_pause_and_resume_store_only_interrupt_summary(database) -> None:
    repository = DatabaseTaskRepository(database)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None

    paused = repository.pause_for_user(
        lease,
        question="需要哪个部门？",
        trace_id="trace-a",
    )
    assert paused.status is TaskStatus.WAITING_FOR_USER
    assert paused.interrupt_kind == "user_clarification"
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        assert row.interrupt_summary == "需要哪个部门？"

    resumed = repository.claim_user_resume(
        auth=auth(),
        task_id="task-a",
        expected_state_version=paused.state_version,
        worker_id="worker-b",
        lease_seconds=30,
    )
    assert resumed.state.status is TaskStatus.RUNNING
    assert resumed.state.interrupt_kind is None
    assert resumed.state.state_version == paused.state_version + 1


def test_finished_research_task_requires_immutable_result_and_is_not_reclaimed(database) -> None:
    repository = DatabaseTaskRepository(database)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None

    with pytest.raises(ValueError, match="task_result_binding_required"):
        repository.finish_task(lease, status=TaskStatus.SUCCEEDED, trace_id="trace-a")
    with pytest.raises(ValueError, match="task_result_binding_required"):
        repository.finish_task(lease, status=TaskStatus.REFUSED, trace_id="trace-a")
    finished = repository.finish_task(
        lease,
        status=TaskStatus.SUCCEEDED,
        result_kind="research",
        result_ref="task-results/task-a/result.json",
        result_hash="sha256:" + "a" * 64,
        trace_id="trace-a",
    )
    assert finished.status is TaskStatus.SUCCEEDED
    assert finished.content_expires_at is not None
    assert repository.claim_next(worker_id="worker-b", lease_seconds=30) is None

    with database.sessions() as session:
        events = tuple(
            session.scalars(
                select(TaskEventRow.event_type)
                .where(TaskEventRow.task_id == "task-a")
                .order_by(TaskEventRow.sequence)
            )
        )
    assert events == ("task_created", "task_succeeded")


def test_finished_research_task_uses_database_clock_for_content_expiry(database) -> None:
    repository = DatabaseTaskRepository(
        database,
        now=lambda: datetime(2000, 1, 1, tzinfo=UTC),
    )
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None

    finished = repository.finish_task(
        lease,
        status=TaskStatus.SUCCEEDED,
        result_kind="research",
        result_ref="task-results/task-a/result.json",
        result_hash="sha256:" + "a" * 64,
        trace_id="trace-a",
    )
    with database.sessions() as session:
        database_now = repository._database_now(session)

    assert finished.content_expires_at is not None
    assert finished.content_expires_at > database_now


def test_cancel_is_terminal_and_state_version_guarded(database) -> None:
    repository = DatabaseTaskRepository(database)
    state = created(repository)
    cancelled = repository.cancel(
        auth=auth(), task_id="task-a", expected_state_version=state.state_version
    )
    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.cancelled is True
    assert cancelled.content_expires_at is not None
    with pytest.raises(TaskRepositoryError, match="task_state_conflict"):
        repository.cancel(
            auth=auth(), task_id="task-a", expected_state_version=state.state_version
        )


def test_finish_task_keeps_cancelled_status_binding_consistent(database) -> None:
    repository = DatabaseTaskRepository(database)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None

    finished = repository.finish_task(
        lease,
        status=TaskStatus.CANCELLED,
        trace_id="trace-a",
    )

    assert finished.status is TaskStatus.CANCELLED
    assert finished.cancelled is True
    assert finished.content_expires_at is not None


def test_capability_lease_registration_renewal_and_unregister(database) -> None:
    repository = DatabaseTaskRepository(
        database,
        now=lambda: datetime(2000, 1, 1, tzinfo=UTC),
    )
    arguments = {
        "capability_name": "research-agent",
        "protocol_version": "m1-fenced-v2",
        "worker_id": "worker-a",
    }

    repository.register_capability(**arguments, lease_seconds=300)
    assert repository.has_active_capability(
        capability_name="research-agent", protocol_version="m1-fenced-v2"
    )
    with database.sessions() as session:
        assert repository._has_active_capability_in_session(
            session,
            capability_name="research-agent",
            protocol_version="m1-fenced-v2",
        )
        first_expiry = session.scalar(select(AgentCapabilityLeaseRow.lease_expires_at))
    assert first_expiry is not None

    repository.register_capability(**arguments, lease_seconds=60)
    with database.sessions() as session:
        rows = tuple(session.scalars(select(AgentCapabilityLeaseRow)))
    assert len(rows) == 1
    assert rows[0].lease_expires_at >= first_expiry

    repository.renew_capability(**arguments, lease_seconds=60)
    with database.sessions() as session:
        renewed_expiry = session.scalar(select(AgentCapabilityLeaseRow.lease_expires_at))
    assert renewed_expiry is not None and renewed_expiry >= first_expiry
    repository.unregister_capability(**arguments)
    assert not repository.has_active_capability(
        capability_name="research-agent", protocol_version="m1-fenced-v2"
    )
    with pytest.raises(TaskRepositoryError, match="agent_capability_not_found"):
        repository.renew_capability(**arguments, lease_seconds=600)
