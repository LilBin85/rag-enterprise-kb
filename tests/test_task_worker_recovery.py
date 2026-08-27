"""M1 worker 只恢复同一 checkpoint 线程并同步业务摘要。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from rag_kb.agent.result import AgentResultStore
from rag_kb.application.task_worker import TaskWorker
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import AgentTaskRow, PrincipalRow, TenantRow
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskRepositoryError,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore


class ScriptedAgent:
    def __init__(self, snapshots: tuple[object, ...], result: dict[str, object] | None = None) -> None:
        self._snapshots = iter(snapshots)
        self._result = result
        self.invocations: list[tuple[object, dict[str, object], object]] = []
        self.state_configs: list[dict[str, object]] = []

    def get_state(self, config: dict[str, object]) -> object:
        self.state_configs.append(config)
        return next(self._snapshots)

    def invoke(self, input_value: object, config: dict[str, object], *, context: object) -> dict[str, object]:
        self.invocations.append((input_value, config, context))
        return self._result or {}


class AuthContexts:
    def __init__(self, auth: AuthContext) -> None:
        self._auth = auth

    def resolve(self, **_kwargs: object) -> AuthContext:
        return self._auth


class Tools:
    def close(self) -> None:
        return None


def _snapshot(*, values: dict[str, object] | None = None, interrupts=(), next=()):
    return SimpleNamespace(values=values or {}, interrupts=interrupts, next=next)


@pytest.fixture()
def setup(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'worker.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    auth = AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id="request-a",
    )
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=auth.attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=auth,
        goal="验证恢复",
        task_mode=TaskMode.RESEARCH,
        task_id="task-a",
    )
    return database, repository, auth, AgentResultStore(LocalObjectStore(tmp_path / "objects"))


def _worker(
    database,
    repository,
    auth,
    results,
    agent: ScriptedAgent,
    *,
    lease_seconds: int = 300,
    agent_for=None,
) -> TaskWorker:
    return TaskWorker(
        repository=repository,
        auth_contexts=AuthContexts(auth),
        agent_for=agent_for or (lambda _budgets: agent),
        results=results,
        evidence=EvidenceLedgerRepository(),
        tools=Tools(),
        database=database,
        worker_id="worker-a",
        lease_seconds=lease_seconds,
    )


def _refusal() -> dict[str, object]:
    return {
        "structured_response": {
            "status": "refused",
            "refusal_code": "insufficient_evidence",
            "stop_reason": "no_evidence",
        }
    }


def _fence_config(*, generation: int) -> dict[str, object]:
    return {
        "thread_id": "task-a",
        "checkpoint_tenant_id": "tenant-a",
        "checkpoint_worker_id": "worker-a",
        "checkpoint_lease_generation": generation,
    }


def test_worker_persists_verified_result_after_initial_agent_run(setup) -> None:
    database, repository, auth, results = setup
    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)

    outcome = _worker(database, repository, auth, results, agent).run_once()
    state = repository.get_task(auth=auth, task_id="task-a")

    assert outcome.status == "finished"
    assert state.status is TaskStatus.REFUSED
    assert state.result_ref is not None and state.result_hash is not None
    assert agent.invocations[0][1]["configurable"] == _fence_config(generation=1)
    assert all(
        config["configurable"] == _fence_config(generation=1)
        for config in (*agent.state_configs, *(item[1] for item in agent.invocations))
    )
    assert results.read_task_result(state.result_ref, state.result_hash).refusal_code == "insufficient_evidence"


def test_worker_uses_persisted_budgets_for_agent_and_recursion_limit(setup) -> None:
    database, repository, auth, results = setup
    budgets = TaskBudgets(
        max_model_calls=11,
        max_tool_calls=9,
        max_runtime_seconds=120,
    )
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        row.budgets = budgets.model_dump(mode="json")
    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)
    selected: list[TaskBudgets] = []

    def agent_for(task_budgets: TaskBudgets) -> ScriptedAgent:
        selected.append(task_budgets)
        return agent

    outcome = _worker(
        database,
        repository,
        auth,
        results,
        agent,
        agent_for=agent_for,
    ).run_once()

    assert outcome.status == "finished"
    assert selected == [budgets]
    assert agent.invocations[0][1]["recursion_limit"] == 80


def test_settle_compensates_result_when_deadline_expires_before_finish(setup) -> None:
    database, _repository, auth, results = setup

    class DeadlineFinishRepository(DatabaseTaskRepository):
        def finish_task(self, lease, *, status, **kwargs):
            if status in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED}:
                with database.sessions.begin() as session:
                    row = session.get(AgentTaskRow, lease.state.task_id)
                    assert row is not None
                    row.deadline = datetime.now(UTC) - timedelta(seconds=1)
            return super().finish_task(lease, status=status, **kwargs)

    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)
    repository = DeadlineFinishRepository(database)

    outcome = _worker(
        database,
        repository,
        auth,
        results,
        agent,
        lease_seconds=1_000,
    ).run_once()

    state = repository.get_task(auth=auth, task_id="task-a")
    assert outcome.status == "failed"
    assert outcome.error_code == "task_deadline_expired"
    assert state.status is TaskStatus.FAILED
    assert state.result_ref is None and state.result_hash is None
    assert not tuple((results._objects.root / "task-results").rglob("*.json"))


def test_settle_compensates_result_when_auth_changes_before_finish(setup) -> None:
    database, repository, auth, results = setup

    class RevokingStore(LocalObjectStore):
        def put_exclusive(self, storage_key: str, content: bytes) -> None:
            super().put_exclusive(storage_key, content)
            with database.sessions.begin() as session:
                principal = session.get(
                    PrincipalRow,
                    {"tenant_id": "tenant-a", "user_id": "owner-a"},
                )
                assert principal is not None
                principal.roles = ["revoked-away"]

    store = RevokingStore(results._objects.root)
    results._objects = store
    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)

    outcome = _worker(database, repository, auth, results, agent).run_once()

    state = repository.get_task(auth=auth, task_id="task-a")
    assert outcome.status == "failed"
    assert outcome.error_code == "task_auth_context_changed"
    assert state.status is TaskStatus.FAILED
    assert state.result_ref is None and state.result_hash is None
    assert not tuple(store.root.rglob("*.json"))


def test_worker_preserves_result_when_finish_commit_acknowledgement_is_lost(setup) -> None:
    database, _repository, auth, results = setup

    class CommitThenAcknowledgementLostRepository(DatabaseTaskRepository):
        def __init__(self, database) -> None:
            super().__init__(database)
            self._lost = False

        def finish_task(self, lease, *, status, **kwargs):
            finished = super().finish_task(lease, status=status, **kwargs)
            if status in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED} and not self._lost:
                self._lost = True
                raise RuntimeError("commit_acknowledgement_lost")
            return finished

    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)
    repository = CommitThenAcknowledgementLostRepository(database)

    outcome = _worker(database, repository, auth, results, agent).run_once()

    state = repository.get_task(auth=auth, task_id="task-a")
    assert outcome.status == "stale"
    assert outcome.error_code == "task_lease_generation_stale"
    assert state.status is TaskStatus.REFUSED
    assert state.result_ref is not None and state.result_hash is not None
    assert results.read_task_result(state.result_ref, state.result_hash).status == "refused"


def test_user_clarification_resumes_same_thread_without_persisting_answer(setup) -> None:
    database, repository, auth, results = setup
    interrupt = SimpleNamespace(value={"kind": "user_clarification", "question": "需要哪个部门？"})
    waiting_agent = ScriptedAgent((_snapshot(interrupts=(interrupt,)),))
    waiting = _worker(database, repository, auth, results, waiting_agent).run_once()
    paused = repository.get_task(auth=auth, task_id="task-a")

    assert waiting.status == "waiting_for_user"
    assert paused.status is TaskStatus.WAITING_FOR_USER
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        assert row.interrupt_summary == "需要哪个部门？"

    final = _refusal()
    resume_agent = ScriptedAgent(
        (_snapshot(interrupts=(interrupt,)), _snapshot(values=final)), result=final
    )
    outcome = _worker(database, repository, auth, results, resume_agent).resume_for_user(
        auth=auth,
        task_id="task-a",
        expected_state_version=paused.state_version,
        answer="人力资源部",
    )

    command = resume_agent.invocations[0][0]
    assert outcome.status == "finished"
    assert getattr(command, "resume") == {"answer": "人力资源部"}
    assert resume_agent.invocations[0][1]["configurable"] == _fence_config(generation=2)
    assert all(
        config["configurable"] == _fence_config(generation=2)
        for config in (
            *resume_agent.state_configs,
            *(item[1] for item in resume_agent.invocations),
        )
    )


def test_worker_syncs_completed_checkpoint_without_reinvoking_agent(setup) -> None:
    database, repository, auth, results = setup
    final = _refusal()
    agent = ScriptedAgent((_snapshot(values=final), _snapshot(values=final)))

    outcome = _worker(database, repository, auth, results, agent).run_once()

    assert outcome.status == "finished"
    assert agent.invocations == []


def test_long_agent_invoke_renews_task_lease_in_background(setup) -> None:
    database, _repository, auth, results = setup
    renewed_during_invoke = Event()

    class RenewingRepository(DatabaseTaskRepository):
        def __init__(self, database) -> None:
            super().__init__(database)
            self.renewals = 0

        def refresh_lease(self, *args, **kwargs):
            refreshed = super().refresh_lease(*args, **kwargs)
            self.renewals += 1
            if self.renewals >= 3:
                renewed_during_invoke.set()
            return refreshed

    class LongAgent(ScriptedAgent):
        def invoke(self, input_value, config, *, context):
            assert renewed_during_invoke.wait(timeout=2)
            return super().invoke(input_value, config, context=context)

    final = _refusal()
    repository = RenewingRepository(database)
    agent = LongAgent((_snapshot(), _snapshot(values=final)), result=final)

    outcome = _worker(
        database, repository, auth, results, agent, lease_seconds=1
    ).run_once()

    assert outcome.status == "finished"
    assert repository.renewals >= 3


def test_background_lease_failure_returns_stale_without_finishing_task(setup) -> None:
    database, _repository, auth, results = setup
    renewal_failed = Event()

    class FailingRepository(DatabaseTaskRepository):
        def __init__(self, database) -> None:
            super().__init__(database)
            self.renewals = 0

        def refresh_lease(self, *args, **kwargs):
            self.renewals += 1
            if self.renewals >= 3:
                renewal_failed.set()
                raise TaskRepositoryError("task_lease_generation_stale")
            return super().refresh_lease(*args, **kwargs)

    class LongAgent(ScriptedAgent):
        def invoke(self, input_value, config, *, context):
            assert renewal_failed.wait(timeout=2)
            return super().invoke(input_value, config, context=context)

    final = _refusal()
    repository = FailingRepository(database)
    agent = LongAgent((_snapshot(), _snapshot(values=final)), result=final)

    outcome = _worker(
        database, repository, auth, results, agent, lease_seconds=1
    ).run_once()

    assert outcome.status == "stale"
    assert outcome.error_code == "task_lease_generation_stale"
    assert repository.get_task(auth=auth, task_id="task-a").status is TaskStatus.RUNNING


def test_close_runs_injected_cleanup_after_tool_close_fails(setup) -> None:
    database, repository, auth, results = setup
    cleaned: list[str] = []

    class FailingTools:
        def close(self) -> None:
            raise RuntimeError("tool_close_failed")

    worker = TaskWorker(
        repository=repository,
        auth_contexts=AuthContexts(auth),
        agent_for=lambda _budgets: ScriptedAgent(()),
        results=results,
        evidence=EvidenceLedgerRepository(),
        tools=FailingTools(),
        database=database,
        worker_id="worker-a",
        close=lambda: cleaned.append("closed"),
    )

    with pytest.raises(RuntimeError, match="tool_close_failed"):
        worker.close()
    assert cleaned == ["closed"]
