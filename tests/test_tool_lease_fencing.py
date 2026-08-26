"""M1 批 1 定向单测：租约 fencing、致命码穿透与逐工具身份重解析。"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.result import AgentResultStore
from rag_kb.agent.tools import _execute_read_tool
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_worker import TaskWorker
from rag_kb.application.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    PrincipalRow,
    TenantRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskAuthContextPort,
    TaskExecutionContextError,
)
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.models import Citation


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )


class ReadAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _arguments, _context):
        self.calls += 1
        return {"result_reference": "server-only", "evidence_ids": [], "items": []}

    def verify(self, _result, _context) -> bool:
        return True


@pytest.fixture()
def research(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'fence.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="owner-a",
                roles=[],
                attributes=_auth().attributes,
            )
        )
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    adapter = ReadAdapter()
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "search_knowledge"
    )
    service = ToolExecutionService(
        database,
        ToolRegistry([definition], adapters={"search_knowledge": adapter}),
        ApprovalService(database),
        tool_results=LocalObjectStore(tmp_path / "tool-results"),
    )
    yield database, repository, service, adapter
    service.close()


def _request(lease, key: str) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        auth=_auth(),
        task_id="task-a",
        tool_name="search_knowledge",
        arguments={"query": "采购制度"},
        worker_id=lease.worker_id,
        lease_generation=lease.lease_generation,
        idempotency_key=key,
    )


def _preempt(database, repository, old_lease) -> None:
    """模拟租约过期后由第二个 Worker 抢占（generation +1）。"""
    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, "task-a")
        assert task is not None
        task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    preempted = repository.claim_next(worker_id="worker-b", lease_seconds=60)
    assert preempted is not None
    assert preempted.lease_generation == old_lease.lease_generation + 1


def test_preempted_worker_cannot_replay_idempotent_result(research) -> None:
    """修复 1：失租 Worker 连旧的幂等成功结果都不得回读。"""
    database, repository, service, adapter = research
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    first = service.execute(_request(lease, "call-1"))
    assert first.status == "succeeded"

    _preempt(database, repository, lease)

    with pytest.raises(ToolExecutionError, match="tool_task_lease_stale"):
        service.execute(_request(lease, "call-1"))
    with pytest.raises(ToolExecutionError, match="tool_task_lease_stale"):
        service.execute(_request(lease, "call-2"))
    assert adapter.calls == 1
    with database.sessions() as session:
        rows = tuple(session.scalars(select(ToolCallRow)))
        assert len(rows) == 1


def test_new_worker_proceeds_and_old_results_stay_replayable(research) -> None:
    """租约门只拦失租者：新 Worker 可新执行，也可回放既有结果。"""
    database, repository, service, adapter = research
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    assert service.execute(_request(lease, "call-1")).status == "succeeded"
    _preempt(database, repository, lease)

    assert repository.claim_next(worker_id="worker-c", lease_seconds=60) is None, (
        "worker-b 仍持有有效租约"
    )
    replay = service.execute(_request(_lease_of(database), "call-1"))
    assert replay.duplicate is True and replay.status == "succeeded"
    fresh = service.execute(_request(_lease_of(database), "call-2"))
    assert fresh.status == "succeeded"
    assert adapter.calls == 2


def _lease_of(database) -> TaskLease:
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-a")
        assert task is not None
        return TaskLease(
            DatabaseTaskRepository(database).get_task(auth=_auth(), task_id="task-a"),
            task.lease_owner,
            task.lease_generation,
            TaskBudgets(),
        )


def test_role_revocation_fails_next_tool_call_closed(research) -> None:
    """修复 3：任务执行中撤销主体身份，下一次工具调用以致命码失败关闭。"""
    database, repository, service, adapter = research
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    with database.sessions.begin() as session:
        principal = session.get(PrincipalRow, {"tenant_id": "tenant-a", "user_id": "owner-a"})
        assert principal is not None
        principal.roles = ["revoked-away"]

    with pytest.raises(ToolExecutionError, match="task_auth_context_changed"):
        service.execute(_request(lease, "call-1"))
    assert adapter.calls == 0
    with database.sessions() as session:
        assert tuple(session.scalars(select(ToolCallRow))) == ()

    port = DatabaseTaskAuthContextPort(database)
    with pytest.raises(TaskExecutionContextError, match="task_auth_context_changed"):
        port.resolve(
            task_id="task-a",
            auth_context_ref=lease.state.auth_context_ref,
            request_id="reclaim",
        )


def test_fatal_code_propagates_through_read_tool_adapter() -> None:
    """致命码直接 raise 穿透 middleware；非致命码仍渲染为失败 ToolMessage。"""

    class FencingExecutor:
        def __init__(self, code: str) -> None:
            self.code = code

        def execute(self, _request):
            raise ToolExecutionError(self.code)

    runtime = SimpleNamespace(context=_run_context(), tool_call_id="model-call-1")

    with pytest.raises(ToolExecutionError, match="tool_task_lease_stale"):
        _execute_read_tool(
            "search_knowledge",
            {"query": "q", "limit": 5},
            runtime,
            FencingExecutor("tool_task_lease_stale"),
        )
    with pytest.raises(ToolExecutionError, match="task_auth_context_changed"):
        _execute_read_tool(
            "search_knowledge",
            {"query": "q", "limit": 5},
            runtime,
            FencingExecutor("task_auth_context_changed"),
        )
    message = json.loads(
        _execute_read_tool(
            "search_knowledge",
            {"query": "q", "limit": 5},
            runtime,
            FencingExecutor("tool_timeout"),
        )
    )
    assert message["error_code"] == "tool_timeout"
    assert message["status"] == "failed"


def _run_context() -> AgentRunContext:
    return AgentRunContext(
        task_id="task-a",
        tenant_id="tenant-a",
        user_id="owner-a",
        task_mode=TaskMode.RESEARCH,
        worker_id="worker-a",
        lease_generation=1,
        run_id="run-a",
        auth=_auth(),
    )


class _RaisingAgent:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def get_state(self, _config):
        return SimpleNamespace(values={}, interrupts=(), next=())

    def invoke(self, _input, _config, *, context):
        raise self._exc


class _FixedAuthContexts:
    def resolve(self, **_kwargs) -> AuthContext:
        return _auth()


def _fencing_worker(database, agent) -> TaskWorker:
    return TaskWorker(
        repository=DatabaseTaskRepository(database),
        auth_contexts=_FixedAuthContexts(),
        agent=agent,
        results=AgentResultStore(LocalObjectStore(Path(database.engine.url.database).parent / "objects")),
        evidence=EvidenceLedgerRepository(),
        tools=SimpleNamespace(close=lambda: None),
        database=database,
        worker_id="worker-a",
        lease_seconds=60,
    )


def test_worker_returns_stale_on_fatal_code_without_touching_business_state(tmp_path: Path) -> None:
    """修复 1 验收②：旧 Worker 收到致命码返回 stale，不调用 _fail_safely。"""
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'stale.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    created = repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")

    outcome = _fencing_worker(
        database, _RaisingAgent(ToolExecutionError("tool_task_lease_stale"))
    ).run_once()

    assert outcome.status == "stale"
    assert outcome.error_code == "tool_task_lease_stale"
    with database.sessions() as session:
        task = session.get(AgentTaskRow, "task-a")
        assert task is not None
        assert task.status == TaskStatus.RUNNING.value
        assert task.last_error_code is None
        assert task.state_version == created.state_version


def test_non_fatal_tool_error_still_fails_task_safely(tmp_path: Path) -> None:
    """非致命工具异常不适用 stale 规则；resolve 路径的身份失效必须收敛终态。"""
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'safe.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")

    outcome = _fencing_worker(
        database, _RaisingAgent(ToolExecutionError("tool_execution_unavailable"))
    ).run_once()
    assert outcome.status == "failed"
    assert outcome.error_code == "tool_execution_unavailable"

    repository.create_task(auth=_auth(), goal="研究二", task_mode=TaskMode.RESEARCH, task_id="task-b")
    outcome = _fencing_worker(
        database,
        _RaisingAgent(TaskExecutionContextError("task_auth_context_changed")),
    ).run_once()
    assert outcome.status == "failed"
    assert outcome.error_code == "task_auth_context_changed"


class _RecordingEvidence(EvidenceLedgerRepository):
    def __init__(self) -> None:
        super().__init__()
        self.seen_auth: list[AuthContext] = []

    def citations_from_ledger(self, _session, *, auth, **_kwargs):
        self.seen_auth.append(auth)
        return (
            Citation(
                document_id="doc-1",
                document_version=1,
                source_file_name="doc.md",
                source_locator="doc/1",
                chunk_id="chunk-1",
            ),
        )


class _RotatingAuthContexts:
    """第一次解析返回旧身份（角色含 reader），之后返回撤销后的最新身份。"""

    def __init__(self) -> None:
        self.resolves = 0

    def resolve(self, **_kwargs) -> AuthContext:
        self.resolves += 1
        if self.resolves == 1:
            return AuthContext(
                tenant_id="tenant-a",
                user_id="owner-a",
                roles=frozenset({"reader"}),
                groups=frozenset(),
                attributes={"permissions": ["knowledge:read", "document:read"]},
                request_id="run-a",
            )
        return AuthContext(
            tenant_id="tenant-a",
            user_id="owner-a",
            roles=frozenset(),
            groups=frozenset(),
            attributes={"permissions": ["knowledge:read", "document:read"]},
            request_id="run-a",
        )


def test_settle_revalidates_citations_with_latest_identity(tmp_path: Path) -> None:
    """修复 3 验收③：_settle 的 citation 复核使用重新 resolve 的最新身份。"""
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'settle.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    evidence = _RecordingEvidence()

    class SucceededAgent(_RaisingAgent):
        def __init__(self) -> None:
            super().__init__(RuntimeError("unused"))

        def invoke(self, _input, _config, *, context):
            return {
                "structured_response": {
                    "status": "succeeded",
                    "answer": "结论",
                    "evidence_ids": ["evidence-1"],
                    "stop_reason": "completed",
                }
            }

    worker = TaskWorker(
        repository=repository,
        auth_contexts=_RotatingAuthContexts(),
        agent=SucceededAgent(),
        results=AgentResultStore(LocalObjectStore(tmp_path / "objects")),
        evidence=evidence,
        tools=SimpleNamespace(close=lambda: None),
        database=database,
        worker_id="worker-a",
        lease_seconds=60,
    )

    outcome = worker.run_once()

    assert outcome.status == "finished"
    assert evidence.seen_auth, "citation 复核未执行"
    assert evidence.seen_auth[-1].roles == frozenset(), "复核必须使用最新身份而非 invoke 前快照"
