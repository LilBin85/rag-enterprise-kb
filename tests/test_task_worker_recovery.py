"""M1 worker 只恢复同一 checkpoint 线程并同步业务摘要。"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from rag_kb.agent.result import AgentResultStore
from rag_kb.application.task_worker import TaskWorker
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
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

    def get_state(self, _config: dict[str, object]) -> object:
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
) -> TaskWorker:
    return TaskWorker(
        repository=repository,
        auth_contexts=AuthContexts(auth),
        agent=agent,
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


def test_worker_persists_verified_result_after_initial_agent_run(setup) -> None:
    database, repository, auth, results = setup
    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)

    outcome = _worker(database, repository, auth, results, agent).run_once()
    state = repository.get_task(auth=auth, task_id="task-a")

    assert outcome.status == "finished"
    assert state.status is TaskStatus.REFUSED
    assert state.result_ref is not None and state.result_hash is not None
    assert agent.invocations[0][1]["configurable"] == {"thread_id": "task-a"}
    assert results.read_task_result(state.result_ref, state.result_hash).refusal_code == "insufficient_evidence"


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
    assert resume_agent.invocations[0][1]["configurable"] == {"thread_id": "task-a"}


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
