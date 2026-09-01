"""M1 worker 只恢复同一 checkpoint 线程并同步业务摘要。"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Callable

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_kb.agent.result import AgentResultStore
from rag_kb.application.task_worker import TaskWorker, TaskWorkerResult
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    AgentCapabilityLeaseRow,
    AgentTaskRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TenantRow,
)
from rag_kb.infrastructure.database.task_execution import DatabaseTaskWorkerSettlement
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    ResumeCommandLease,
    TaskRepositoryError,
    _command_from_row,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.observability import Observability, current_run_id, current_trace_headers


class ScriptedAgent:
    def __init__(
        self,
        snapshots: tuple[object, ...],
        result: dict[str, object] | None = None,
        *,
        before_invoke: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._snapshots = iter(snapshots)
        self._result = result
        self._before_invoke = before_invoke
        self.invocations: list[tuple[object, dict[str, object], object]] = []
        self.state_configs: list[dict[str, object]] = []

    def get_state(self, config: dict[str, object]) -> object:
        self.state_configs.append(deepcopy(config))
        return next(self._snapshots)

    def invoke(self, input_value: object, config: dict[str, object], *, context: object) -> dict[str, object]:
        if self._before_invoke is not None:
            self._before_invoke(config)
        self.invocations.append((input_value, deepcopy(config), context))
        return self._result or {}


class AuthContexts:
    def __init__(self, auth: AuthContext) -> None:
        self._auth = auth

    def resolve(self, **_kwargs: object) -> AuthContext:
        return self._auth


class Tools:
    def close(self) -> None:
        return None


def _settlement(database) -> DatabaseTaskWorkerSettlement:
    root = Path(database.engine.url.database).parent / "objects"
    return DatabaseTaskWorkerSettlement(database, LocalObjectStore(root))


def _snapshot(
    *,
    values: dict[str, object] | None = None,
    interrupts=(),
    next=(),
    checkpoint_ns: str = "",
    checkpoint_id: str = "checkpoint-a",
):
    return SimpleNamespace(
        values=values or {},
        interrupts=interrupts,
        next=next,
        config={
            "configurable": {
                "thread_id": "task-a",
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        },
    )


def _mark_resume_checkpoint(database, repository, config: dict[str, object]) -> None:
    """测试替身模拟 FencedPostgresSaver 的首次 checkpoint 同事务 marker。"""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return
    command_id = configurable.get("checkpoint_resume_command_id")
    command_generation = configurable.get("checkpoint_resume_command_lease_generation")
    if command_id is None and command_generation is None:
        return
    assert isinstance(command_id, str)
    assert isinstance(command_generation, int)
    worker_id = configurable["checkpoint_worker_id"]
    assert isinstance(worker_id, str)
    with database.sessions.begin() as session:
        command = session.get(TaskResumeCommandRow, command_id)
        assert command is not None
        task = session.get(AgentTaskRow, command.task_id)
        assert task is not None
        repository.mark_resume_command_applied_in_session(
            session,
            lease=ResumeCommandLease(
                command=_command_from_row(command),
                task_lease=repository._resume_task_lease(task, worker_id=worker_id),
                worker_id=worker_id,
                command_lease_generation=command_generation,
            ),
            checkpoint_id="resume-checkpoint-a",
        )


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
        settlement=_settlement(database),
        tools=Tools(),
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


def _telemetry() -> tuple[Observability, InMemorySpanExporter]:
    observability = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    observability.start()
    exporter = InMemorySpanExporter()
    assert observability.provider is not None
    observability.provider.add_span_processor(SimpleSpanProcessor(exporter))
    return observability, exporter


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


def test_worker_run_links_producer_without_becoming_its_child(setup) -> None:
    database, repository, auth, results = setup
    observability, exporter = _telemetry()
    final = _refusal()
    agent = ScriptedAgent((_snapshot(), _snapshot(values=final)), result=final)
    try:
        with observability.activate():
            with observability.span("producer.request") as producer:
                traceparent, tracestate = current_trace_headers()
                producer_context = producer.get_span_context()
                claimed = repository.claim_next(worker_id="worker-a", lease_seconds=300)
                assert claimed is not None
                lease = type(claimed)(
                    claimed.state,
                    claimed.worker_id,
                    claimed.lease_generation,
                    claimed.budgets,
                    traceparent,
                    tracestate,
                )
                result = _worker(database, repository, auth, results, agent)._run_claimed(lease)
                assert current_run_id() is None
        payload = observability.metrics_payload().decode()
        run_span = next(span for span in exporter.get_finished_spans() if span.name == "task.run")
    finally:
        observability.shutdown()

    assert result.status == "finished"
    assert {"task.run", "agent.advance", "agent.invoke"} <= {
        span.name for span in exporter.get_finished_spans()
    }
    assert run_span.parent is None or not run_span.parent.is_valid
    assert len(run_span.links) == 1
    assert run_span.links[0].context.trace_id == producer_context.trace_id
    assert run_span.links[0].context.span_id == producer_context.span_id
    assert agent.invocations[0][2].auth.request_id != run_span.attributes["run_id"]
    task_run_metric = next(
        line for line in payload.splitlines() if line.startswith("rag_kb_task_runs_total{")
    )
    assert all(label in task_run_metric for label in ('mode="research"', 'outcome="succeeded"', 'error_category="none"'))


def test_run_once_records_bounded_claim_results_without_changing_errors(
    setup, monkeypatch
) -> None:
    database, repository, auth, results = setup
    leases = [
        SimpleNamespace(state=SimpleNamespace(task_mode=TaskMode.RESEARCH)),
        SimpleNamespace(state=SimpleNamespace(task_mode=TaskMode.CONTROLLED_ACTION)),
        None,
        TaskRepositoryError("task_protocol_mismatch"),
        RuntimeError("private claim failure"),
    ]

    def claim_next(**_kwargs):
        value = leases.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(repository, "claim_next", claim_next)
    worker = _worker(database, repository, auth, results, ScriptedAgent(()))
    monkeypatch.setattr(
        worker,
        "_run_claimed",
        lambda lease: TaskWorkerResult(status="finished", task_id=lease.state.task_mode.value),
    )
    observability, _exporter = _telemetry()
    try:
        with observability.activate():
            assert worker.run_once().task_id == "research"
            assert worker.run_once().task_id == "controlled_action"
            assert worker.run_once().status == "idle"
            with pytest.raises(TaskRepositoryError, match="task_protocol_mismatch"):
                worker.run_once()
            with pytest.raises(RuntimeError, match="private claim failure"):
                worker.run_once()
            observability.record_task_claim(mode="private-id", result="https://private")
        payload = observability.metrics_payload().decode()
    finally:
        observability.shutdown()

    assert 'mode="research",result="claimed"' in payload
    assert 'mode="controlled_action",result="claimed"' in payload
    assert 'mode="none",result="idle"' in payload
    assert 'mode="none",result="protocol_mismatch"' in payload
    assert 'mode="none",result="failure"' in payload
    assert "private-id" not in payload
    assert "https://private" not in payload


def test_equal_eligible_work_rotates_between_resume_and_initial_task(setup, monkeypatch) -> None:
    database, repository, auth, results = setup
    worker = _worker(database, repository, auth, results, ScriptedAgent(()))
    eligible_at = datetime.now(UTC)
    task_lease = SimpleNamespace(state=SimpleNamespace(task_mode=TaskMode.RESEARCH))
    resume_lease = SimpleNamespace(task_lease=task_lease)
    calls: list[str] = []

    monkeypatch.setattr(
        repository,
        "resume_command_candidates",
        lambda *, limit: (SimpleNamespace(command_id="command-a", eligible_at=eligible_at),),
    )
    monkeypatch.setattr(
        repository,
        "next_claimable_task_eligible_at",
        lambda **_kwargs: eligible_at,
    )

    def claim_resume_command(**_kwargs):
        calls.append("resume")
        return resume_lease

    def claim_next(**_kwargs):
        calls.append("task")
        return task_lease

    monkeypatch.setattr(repository, "claim_resume_command", claim_resume_command)
    monkeypatch.setattr(repository, "claim_next", claim_next)

    first = worker._claim_next_work()
    second = worker._claim_next_work()

    assert first == (task_lease, resume_lease)
    assert second == (task_lease, None)
    assert calls == ["resume", "task"]


def test_task_claim_metric_is_noop_when_metrics_are_disabled() -> None:
    observability = Observability(
        enabled=False,
        metrics_enabled=False,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )

    observability.record_task_claim(mode="research", result="claimed")

    assert "rag_kb_task_claims_total{" not in observability.metrics_payload().decode()


def test_lease_failure_emits_one_safe_error_span_and_metric() -> None:
    from rag_kb.application.task_worker import _TaskLeaseHeartbeat
    from rag_kb.infrastructure.database.task_repository import TaskRepositoryError

    observability, exporter = _telemetry()
    try:
        with observability.activate():
            heartbeat = _TaskLeaseHeartbeat(
                lambda: (_ for _ in ()).throw(TaskRepositoryError("task_lease_generation_stale")),
                1,
                on_failure=lambda exc: observability.record_lease_failure(
                    kind="task",
                    operation="task.lease.renew",
                    error_code=getattr(exc, "code", "unknown"),
                ),
            )
            with pytest.raises(TaskRepositoryError):
                heartbeat.__enter__()
        payload = observability.metrics_payload().decode()
        spans = [span for span in exporter.get_finished_spans() if span.name == "lease.renew"]
    finally:
        observability.shutdown()

    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
    assert 'kind="task",operation="task.lease.renew"' in payload


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
    interrupt = SimpleNamespace(
        value={"kind": "user_clarification", "question": "需要哪个部门？"},
        id="interrupt-a",
    )
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
        (_snapshot(interrupts=(interrupt,)), _snapshot(values=final)),
        result=final,
        before_invoke=lambda config: _mark_resume_checkpoint(database, repository, config),
    )
    command = repository.enqueue_user_resume_command(
        auth=auth,
        task_id="task-a",
        expected_state_version=paused.state_version,
        answer="人力资源部",
    )
    outcome = _worker(database, repository, auth, results, resume_agent).run_once()

    resume = resume_agent.invocations[0][0]
    assert outcome.status == "finished"
    assert getattr(resume, "resume") == {"answer": "人力资源部"}
    invoke_config = resume_agent.invocations[0][1]["configurable"]
    assert invoke_config == {
        **_fence_config(generation=2),
        "checkpoint_resume_command_id": command.command_id,
        "checkpoint_resume_command_lease_generation": 1,
    }
    assert resume_agent.state_configs[-1]["configurable"] == _fence_config(generation=2)


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


def test_long_resume_invoke_renews_both_leases_after_marker(setup) -> None:
    database, _repository, auth, results = setup
    renewed_after_marker = Event()

    class RenewingRepository(DatabaseTaskRepository):
        def __init__(self, database) -> None:
            super().__init__(database)
            self.post_marker_renewals = 0

        def refresh_resume_command_lease(self, *args, **kwargs):
            refreshed = super().refresh_resume_command_lease(*args, **kwargs)
            if refreshed.command.applied_checkpoint_id is not None:
                self.post_marker_renewals += 1
                renewed_after_marker.set()
            return refreshed

    class LongResumeAgent(ScriptedAgent):
        def invoke(self, input_value, config, *, context):
            if self._before_invoke is not None:
                self._before_invoke(config)
            assert renewed_after_marker.wait(timeout=2)
            self.invocations.append((input_value, deepcopy(config), context))
            return self._result or {}

    repository = RenewingRepository(database)
    interrupt = SimpleNamespace(
        value={"kind": "user_clarification", "question": "需要哪个部门？"},
        id="interrupt-long",
    )
    waiting = _worker(
        database,
        repository,
        auth,
        results,
        ScriptedAgent((_snapshot(interrupts=(interrupt,)),)),
        lease_seconds=1,
    ).run_once()
    assert waiting.status == "waiting_for_user"
    paused = repository.get_task(auth=auth, task_id="task-a")
    repository.enqueue_user_resume_command(
        auth=auth,
        task_id="task-a",
        expected_state_version=paused.state_version,
        answer="人力资源部",
    )
    final = _refusal()
    agent = LongResumeAgent(
        (_snapshot(interrupts=(interrupt,)), _snapshot(values=final)),
        result=final,
        before_invoke=lambda config: _mark_resume_checkpoint(database, repository, config),
    )

    outcome = _worker(
        database, repository, auth, results, agent, lease_seconds=1
    ).run_once()

    assert outcome.status == "finished"
    assert repository.post_marker_renewals >= 1


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
        settlement=_settlement(database),
        tools=FailingTools(),
        worker_id="worker-a",
        close=lambda: cleaned.append("closed"),
    )

    with pytest.raises(RuntimeError, match="tool_close_failed"):
        worker.close()
    assert cleaned == ["closed"]


def test_controlled_finish_binds_result_and_repeat_sync_cannot_change_terminal(
    setup,
) -> None:
    """受控任务终态按 task mode 放宽 result_kind；重复业务同步不改变终态。"""
    from uuid import uuid4

    from sqlalchemy import select

    from rag_kb.infrastructure.database.task_repository import TaskLease
    from rag_kb.infrastructure.database.unit_of_work import UnitOfWork

    database, _repository, auth, results = setup
    controlled_repository = DatabaseTaskRepository(
        database,
        task_capability_name="task-agent",
        task_capability_protocol_version="m2-fenced-v1",
    )
    with database.sessions.begin() as session:
        session.add(
            AgentCapabilityLeaseRow(
                capability_name="task-agent",
                protocol_version="m2-fenced-v1",
                worker_id="worker-a",
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=300),
                updated_at=datetime.now(UTC),
            )
        )
    controlled_repository.create_task(
        auth=auth,
        goal="受控报告",
        task_mode=TaskMode.CONTROLLED_ACTION,
        task_id="task-controlled",
    )
    with UnitOfWork(database) as work:
        assert work.session is not None
        task = work.session.scalar(
            select(AgentTaskRow).where(AgentTaskRow.task_id == "task-controlled")
        )
        assert task is not None
        task.status = "running"
        task.lease_owner = "worker-a"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=60)
        task.lease_generation = 1
    lease = TaskLease(
        controlled_repository.get_task(auth=auth, task_id="task-controlled"),
        "worker-a",
        1,
        TaskBudgets(),
    )

    result_ref, result_hash = results.save_task_result(
        "task-controlled",
        _controlled_result(),
    )
    finished = controlled_repository.finish_task(
        lease,
        status=TaskStatus.SUCCEEDED,
        result_kind="controlled_action",
        result_ref=result_ref,
        result_hash=result_hash,
        run_id="trace-controlled",
    )
    assert finished.status is TaskStatus.SUCCEEDED
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-controlled")
        assert row is not None
        assert row.result_kind == "controlled_action"
        assert row.result_ref == result_ref
        terminal_version = row.state_version

    with pytest.raises(TaskRepositoryError):
        controlled_repository.finish_task(
            lease,
            status=TaskStatus.SUCCEEDED,
            result_kind="controlled_action",
            result_ref=f"task-results/task-controlled/{uuid4().hex}.json",
            result_hash="sha256:" + "c" * 64,
            run_id="trace-controlled-2",
        )
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "task-controlled")
        assert row is not None
        assert row.status == "succeeded"
        assert row.result_ref == result_ref
        assert row.state_version == terminal_version


def _controlled_result():
    from rag_kb.agent.result import TaskResult

    return TaskResult(
        task_id="task-controlled",
        result_kind="controlled_action",
        status="succeeded",
        answer="报告已完成",
        evidence_ids=("evidence-1",),
        stop_reason="completed",
        created_at=datetime.now(UTC),
    )
