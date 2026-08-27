"""M1 agent_tasks 业务生命周期、租约和公开状态契约。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from rag_kb.config import Settings, task_budgets_from_settings
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
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
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=[],
                attributes=auth().attributes,
            )
        )
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


def test_settings_derived_budgets_are_persisted_for_new_research_tasks(
    database, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "2")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("AGENT_MAX_RUNTIME_SECONDS", "120")
    clock = Clock()
    budgets = task_budgets_from_settings(Settings(_env_file=None))
    repository = DatabaseTaskRepository(database, default_budgets=budgets, now=clock)

    state = created(repository)

    assert state.deadline == clock.value + timedelta(seconds=120)
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        assert TaskBudgets.model_validate(row.budgets) == budgets


def test_retention_settings_require_safe_object_windows(monkeypatch) -> None:
    monkeypatch.setenv("RESULT_OBJECT_ORPHAN_GRACE_SECONDS", "360")
    with pytest.raises(ValueError, match="result_object_orphan_grace_too_short"):
        Settings(_env_file=None)

    monkeypatch.setenv("RESULT_OBJECT_ORPHAN_GRACE_SECONDS", "361")
    monkeypatch.setenv("TOOL_RESULT_TTL_SECONDS", "300")
    with pytest.raises(ValueError, match="tool_result_ttl_too_short"):
        Settings(_env_file=None)

    monkeypatch.setenv("TOOL_RESULT_TTL_SECONDS", "301")
    assert Settings(_env_file=None).tool_result_ttl_seconds == 301


def test_finish_task_turns_expired_result_submission_into_failed(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=1_000)
    assert lease is not None
    clock.value += timedelta(seconds=601)

    finished = repository.finish_task(
        lease,
        status=TaskStatus.SUCCEEDED,
        result_kind="research",
        result_ref="task-results/task-a/result.json",
        result_hash="sha256:" + "a" * 64,
        trace_id="trace-a",
    )

    assert finished.status is TaskStatus.FAILED
    assert finished.result_kind is None
    assert finished.result_ref is None
    assert finished.result_hash is None
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        assert row.last_error_code == "task_deadline_expired"
        events = tuple(
            session.scalars(
                select(TaskEventRow.event_type)
                .where(TaskEventRow.task_id == "task-a")
                .order_by(TaskEventRow.sequence)
            )
        )
    assert events == ("task_created", "task_failed")


def test_finish_task_does_not_write_deadline_failure_after_lease_loss(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=1)
    assert lease is not None
    clock.value += timedelta(seconds=601)

    with pytest.raises(TaskRepositoryError, match="task_lease_expired"):
        repository.finish_task(
            lease,
            status=TaskStatus.REFUSED,
            result_kind="research",
            result_ref="task-results/task-a/result.json",
            result_hash="sha256:" + "a" * 64,
            trace_id="trace-a",
        )

    assert repository.get_task(auth=auth(), task_id="task-a").status is TaskStatus.RUNNING


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


def test_waiting_deadline_reaper_fails_task_once_and_blocks_resume(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    paused = repository.pause_for_user(
        lease,
        question="需要哪个部门？",
        trace_id="trace-a",
    )
    clock.value += timedelta(seconds=601)

    with pytest.raises(TaskRepositoryError, match="task_deadline_expired"):
        repository.claim_user_resume(
            auth=auth(),
            task_id="task-a",
            expected_state_version=paused.state_version,
            worker_id="worker-b",
            lease_seconds=30,
        )

    assert repository.expire_waiting_tasks(limit=10) == 1
    assert repository.expire_waiting_tasks(limit=10) == 0
    failed = repository.get_task(auth=auth(), task_id="task-a")
    assert failed.status is TaskStatus.FAILED
    assert failed.interrupt_kind is None
    assert failed.interrupt_summary is None
    assert failed.content_expires_at is not None
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        assert row.last_error_code == "task_deadline_expired"
        assert row.lease_owner is None and row.lease_expires_at is None
        events = tuple(
            session.scalars(
                select(TaskEventRow.event_type)
                .where(TaskEventRow.task_id == "task-a")
                .order_by(TaskEventRow.sequence)
            )
        )
    assert events == ("task_created", "task_waiting_for_user", "task_failed")


def test_resume_deadline_is_conditionally_checked_at_state_transition(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    paused = repository.pause_for_user(
        lease,
        question="需要哪个部门？",
        trace_id="trace-a",
    )
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        row.deadline = clock.value + timedelta(seconds=1)

    class AdvancingClock:
        def __init__(self, start: datetime) -> None:
            self._start = start
            self._calls = 0

        def __call__(self) -> datetime:
            self._calls += 1
            return self._start if self._calls == 1 else self._start + timedelta(seconds=2)

    repository._now = AdvancingClock(clock.value)
    with pytest.raises(TaskRepositoryError, match="task_deadline_expired"):
        repository.claim_user_resume(
            auth=auth(),
            task_id="task-a",
            expected_state_version=paused.state_version,
            worker_id="worker-b",
            lease_seconds=30,
        )
    assert repository.get_task(auth=auth(), task_id="task-a").status is TaskStatus.WAITING_FOR_USER


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


def test_finished_research_task_uses_injected_sqlite_clock_for_content_expiry(database) -> None:
    started_at = datetime(2000, 1, 1, tzinfo=UTC)
    repository = DatabaseTaskRepository(
        database,
        now=lambda: started_at,
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
    assert finished.content_expires_at is not None
    assert finished.content_expires_at == started_at + timedelta(days=7)


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


def test_research_creation_uses_same_transaction_capability_gate(database) -> None:
    """缺少兼容 Worker 时不落 planning 行；其他协议版本不能绕过创建门。"""
    repository = DatabaseTaskRepository(
        database,
        research_capability_name="research-agent",
        research_capability_protocol_version="m1-fenced-v2",
    )
    with pytest.raises(TaskRepositoryError, match="research_agent_unavailable"):
        repository.create_task(
            auth=auth(), goal="无 Worker 不能创建", task_mode=TaskMode.RESEARCH, task_id="task-a"
        )
    with database.sessions() as session:
        assert session.scalar(select(AgentTaskRow)) is None

    repository.register_capability(
        capability_name="research-agent",
        protocol_version="m1-fenced-v1",
        worker_id="worker-old",
        lease_seconds=60,
    )
    with pytest.raises(TaskRepositoryError, match="research_agent_unavailable"):
        repository.create_task(
            auth=auth(), goal="旧协议不能创建", task_mode=TaskMode.RESEARCH, task_id="task-b"
        )

    repository.register_capability(
        capability_name="research-agent",
        protocol_version="m1-fenced-v2",
        worker_id="worker-current",
        lease_seconds=60,
    )
    created = repository.create_task(
        auth=auth(), goal="兼容 Worker 可以创建", task_mode=TaskMode.RESEARCH, task_id="task-c"
    )
    assert created.task_id == "task-c"
    with database.sessions.begin() as session:
        current = session.get(
            AgentCapabilityLeaseRow,
            ("research-agent", "m1-fenced-v2", "worker-current"),
        )
        assert current is not None
        current.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(TaskRepositoryError, match="research_agent_unavailable"):
        repository.create_task(
            auth=auth(), goal="过期 Worker 不能创建", task_mode=TaskMode.RESEARCH, task_id="task-d"
        )
    with database.sessions() as session:
        assert len(tuple(session.scalars(select(AgentTaskRow)))) == 1
