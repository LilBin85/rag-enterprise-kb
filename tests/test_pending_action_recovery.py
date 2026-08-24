"""Pending action 对象先写、数据库 CAS 与租约栅栏回归。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import pytest
from sqlalchemy import select

from rag_kb.application.pending_action_store import (
    PendingActionStore,
    PendingActionStoreError,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import SearchAction, TaskMode, TaskNode
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import AgentTaskRow, PrincipalRow, TenantRow
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskRepositoryError,
)
from rag_kb.infrastructure.database.task_runtime_adapter import (
    LeasedTaskRuntimeRepository,
)
from rag_kb.infrastructure.object_store import LocalObjectStore


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 22, 15, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="user-a",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id="request-a",
    )


def _database(tmp_path: Path):
    database = build_database(
        f"sqlite+pysqlite:///{(tmp_path / 'pending-actions.sqlite3').as_posix()}"
    )
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=[],
                attributes={"permissions": ["task:create", "task:read"]},
            )
        )
    return database


def _claimed(tmp_path: Path, clock: Clock):
    database = _database(tmp_path)
    repository = DatabaseTaskRepository(database, now=clock)
    repository.create_task(
        auth=_auth(),
        goal="检索采购审批制度",
        task_mode=TaskMode.RESEARCH,
        task_id="task-a",
    )
    lease = repository.claim_task("task-a", worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    actions = PendingActionStore(
        LocalObjectStore(tmp_path / "objects"),
        now=clock,
    )
    return database, repository, lease, actions


def test_pending_action_round_trip_is_exclusive_and_detects_tampering(
    tmp_path: Path,
) -> None:
    clock = Clock()
    objects = LocalObjectStore(tmp_path / "objects")
    store = PendingActionStore(objects, now=clock)
    action = SearchAction(query="采购审批", reason_code="need_evidence")
    values = {
        "task_id": "task-a",
        "auth_context_ref": "sha256:auth",
        "plan_revision": 2,
        "source_state_version": 4,
        "action": action,
        "expires_at": clock.value + timedelta(minutes=5),
    }

    binding = store.stage(**values)
    assert store.stage(**values) == binding
    assert store.load(
        binding,
        task_id="task-a",
        auth_context_ref="sha256:auth",
        plan_revision=2,
    ) == action

    target = (tmp_path / "objects").joinpath(
        *PurePosixPath(binding.storage_ref).parts
    )
    target.write_bytes(target.read_bytes().replace("采购".encode(), "财务".encode()))
    with pytest.raises(PendingActionStoreError, match="pending_action_hash_mismatch"):
        store.load(
            binding,
            task_id="task-a",
            auth_context_ref="sha256:auth",
            plan_revision=2,
        )


def test_adapter_persists_only_binding_and_recovers_exact_action(tmp_path: Path) -> None:
    clock = Clock()
    database, repository, lease, actions = _claimed(tmp_path, clock)
    adapter = LeasedTaskRuntimeRepository(
        repository,
        lease,
        trace_id="trace-a",
        run_id="run-a",
        pending_actions=actions,
    )
    action = SearchAction(query="采购审批正文", reason_code="need_evidence")
    selected = lease.state.model_copy(
        update={
            "state_version": lease.state.state_version + 1,
            "current_node": TaskNode.EXECUTE_READ_TOOL,
        }
    )

    persisted = adapter.save_pending_action(
        selected,
        expected_state_version=lease.state.state_version,
        action=action,
        expires_at=clock.value + timedelta(minutes=5),
        event={"event_type": "next_action_selected"},
    )

    assert persisted.pending_action is not None
    assert adapter.load_pending_action() == action
    with database.sessions() as session:
        row = session.scalar(select(AgentTaskRow).where(AgentTaskRow.task_id == "task-a"))
        assert row is not None
        assert row.pending_action_ref == persisted.pending_action.storage_ref
        assert row.pending_action_hash == persisted.pending_action.content_hash
        assert row.pending_action_kind == "search"
        assert "采购审批正文" not in "|".join(
            (row.pending_action_ref, row.pending_action_hash, row.pending_action_kind)
        )


def test_crash_after_object_publish_leaves_database_unreferenced(tmp_path: Path) -> None:
    clock = Clock()
    database, _repository, lease, actions = _claimed(tmp_path, clock)
    binding = actions.stage(
        task_id=lease.state.task_id,
        auth_context_ref=lease.state.auth_context_ref,
        plan_revision=lease.state.plan_revision,
        source_state_version=lease.state.state_version,
        action=SearchAction(query="采购审批", reason_code="need_evidence"),
        expires_at=clock.value + timedelta(minutes=5),
    )

    assert binding.storage_ref
    with database.sessions() as session:
        row = session.scalar(select(AgentTaskRow).where(AgentTaskRow.task_id == "task-a"))
        assert row is not None
        assert row.pending_action_ref is None
        assert row.pending_action_hash is None


def test_lease_rollover_fences_old_worker_from_binding_staged_object(
    tmp_path: Path,
) -> None:
    clock = Clock()
    database, repository, first, actions = _claimed(tmp_path, clock)
    binding = actions.stage(
        task_id=first.state.task_id,
        auth_context_ref=first.state.auth_context_ref,
        plan_revision=first.state.plan_revision,
        source_state_version=first.state.state_version,
        action=SearchAction(query="采购审批", reason_code="need_evidence"),
        expires_at=clock.value + timedelta(minutes=5),
    )
    stale_state = first.state.model_copy(
        update={
            "current_node": TaskNode.EXECUTE_READ_TOOL,
            "pending_action": binding,
        }
    )
    clock.value += timedelta(seconds=31)
    second = repository.claim_task("task-a", worker_id="worker-b", lease_seconds=30)
    assert second is not None

    with pytest.raises(TaskRepositoryError, match="task_lease_generation_stale"):
        repository.checkpoint(
            first,
            stale_state,
            event_type="stale_action_selected",
            trace_id="trace-a",
            run_id="run-a",
        )
    with database.sessions() as session:
        row = session.scalar(select(AgentTaskRow).where(AgentTaskRow.task_id == "task-a"))
        assert row is not None
        assert row.pending_action_ref is None
        assert row.lease_generation == second.lease_generation

