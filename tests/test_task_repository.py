"""Stage 4 DatabaseTaskRepository 的 SQLite 事务与恢复契约。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskNode, TaskStatus
from rag_kb.application.task_runtime import TaskRuntime
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import PrincipalRow, TaskEventRow, TenantRow
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskRepositoryError,
)
from rag_kb.infrastructure.database.task_runtime_adapter import LeasedTaskRuntimeRepository


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


@pytest.fixture()
def database(tmp_path: Path):
    db = build_database(f"sqlite+pysqlite:///{(tmp_path / 'tasks.sqlite3').as_posix()}")
    Base.metadata.create_all(db.engine)
    with db.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="user-a", roles=[], attributes={}))
    return db


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


def test_state_version_compare_and_swap_rejects_stale_writer(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    initial = created(repository)
    assert initial.deadline == clock.value + timedelta(seconds=600)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    next_state = lease.state.model_copy(update={"current_node": TaskNode.CLASSIFY_GOAL})
    saved = repository.checkpoint(
        lease,
        next_state,
        event_type="checkpoint",
        trace_id="trace-a",
        run_id="run-a",
    )
    assert saved.state_version == 2
    with pytest.raises(TaskRepositoryError, match="task_state_conflict"):
        repository.checkpoint(
            lease,
            next_state,
            event_type="stale_checkpoint",
            trace_id="trace-a",
            run_id="run-a",
        )


def test_lease_generation_expiry_reclaim_fences_old_worker(database) -> None:
    clock = Clock()
    repository = DatabaseTaskRepository(database, now=clock)
    created(repository)
    first = repository.claim_next(worker_id="worker-a", lease_seconds=1)
    assert first is not None
    clock.value += timedelta(seconds=2)
    reclaimed = repository.claim_next(worker_id="worker-b", lease_seconds=30)
    assert reclaimed is not None
    assert reclaimed.lease_generation == first.lease_generation + 1
    with pytest.raises(TaskRepositoryError, match="task_lease_generation_stale"):
        repository.checkpoint(
            first,
            first.state,
            event_type="stale_checkpoint",
            trace_id="trace-a",
            run_id="run-a",
        )


def test_generic_claim_skips_sync_query_but_explicit_recovery_remains_available(database) -> None:
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=auth(),
        goal="同步知识问答",
        task_mode=TaskMode.KNOWLEDGE_QUERY,
        task_id="query-task",
    )

    assert repository.claim_next(worker_id="generic-worker", lease_seconds=30) is None
    recovered = repository.claim_task(
        "query-task",
        worker_id="recovery-worker",
        lease_seconds=30,
    )
    assert recovered is not None
    assert recovered.state.task_mode is TaskMode.KNOWLEDGE_QUERY


def test_checkpoint_and_task_event_commit_atomically(database) -> None:
    repository = DatabaseTaskRepository(database)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    repository.checkpoint(
        lease,
        lease.state.model_copy(update={"current_node": TaskNode.CLASSIFY_GOAL}),
        event_type="checkpoint",
        trace_id="trace-a",
        run_id="run-a",
    )
    with database.sessions() as session:
        events = tuple(
            session.scalars(
                select(TaskEventRow)
                .where(TaskEventRow.task_id == "task-a")
                .order_by(TaskEventRow.sequence)
            )
        )
        assert [event.event_type for event in events] == ["task_created", "checkpoint"]
        assert all("正文" not in str(event.counts) for event in events)


def test_expired_checkpoint_recovery_does_not_replay_terminal_task(database) -> None:
    repository = DatabaseTaskRepository(database)
    created(repository)
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    terminal = lease.state.model_copy(
        update={"status": TaskStatus.SUCCEEDED, "current_node": TaskNode.TERMINAL}
    )
    saved = repository.checkpoint(
        lease,
        terminal,
        event_type="task_succeeded",
        trace_id="trace-a",
        run_id="run-a",
    )
    assert saved.status is TaskStatus.SUCCEEDED
    assert repository.claim_next(worker_id="worker-recovery", lease_seconds=30) is None


def test_runtime_pause_resume_persists_across_new_lease(database) -> None:
    repository = DatabaseTaskRepository(database)
    created(repository)
    first = repository.claim_task("task-a", worker_id="worker-a", lease_seconds=30)
    assert first is not None
    selection = first.state.model_copy(update={"current_node": TaskNode.SELECT_ACTION})
    selected = repository.checkpoint(
        first,
        selection,
        event_type="ready_for_action",
        trace_id="trace-a",
        run_id="run-a",
    )
    session = LeasedTaskRuntimeRepository(
        repository,
        type(first)(selected, first.worker_id, first.lease_generation, first.budgets),
        trace_id="trace-a",
        run_id="run-a",
    )

    class Planner:
        def next_action(self, _state, observations=()):
            return {
                "kind": "ask_user",
                "question": "需要哪个部门？",
                "reason_code": "missing_department",
            }

    paused = TaskRuntime(session, Planner(), budgets=session.budgets).step("task-a")
    assert paused.status is TaskStatus.WAITING_FOR_USER
    resumed = repository.resume_user(
        auth=auth(),
        task_id="task-a",
        expected_state_version=paused.state_version,
        answer_ref="sha256:answer-reference",
    )
    assert resumed.status is TaskStatus.PLANNING
    second = repository.claim_task("task-a", worker_id="worker-b", lease_seconds=30)
    assert second is not None
    assert second.lease_generation == first.lease_generation + 1
    assert second.state.current_node is TaskNode.UPDATE_PLAN
    assert second.state.model_calls == 1
    assert second.state.observations[-1] == {"ref": "sha256:answer-reference"}


def test_cancel_is_persistent_terminal_and_state_version_guarded(database) -> None:
    repository = DatabaseTaskRepository(database)
    state = created(repository)
    cancelled = repository.cancel(
        auth=auth(),
        task_id="task-a",
        expected_state_version=state.state_version,
    )
    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.cancelled is True
    assert cancelled.current_node is TaskNode.TERMINAL
    assert repository.claim_task("task-a", worker_id="worker-a", lease_seconds=30) is None
    with pytest.raises(TaskRepositoryError, match="task_state_conflict"):
        repository.cancel(
            auth=auth(),
            task_id="task-a",
            expected_state_version=state.state_version,
        )
