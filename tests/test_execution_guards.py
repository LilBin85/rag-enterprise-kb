"""M1 批 2 定向单测：停止边界、refused 契约与 READ 失败语义。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from rag_kb.agent.factory import build_research_agent
from rag_kb.agent.middleware import (
    AgentGuardError,
    ExecutionGuardMiddleware,
)
from rag_kb.agent.result import TaskResultCandidate, build_task_result
from rag_kb.agent.tools import render_tool_message
from rag_kb.application.approval_service import ApprovalService, arguments_fingerprint, principal_fingerprint
from rag_kb.application.tool_execution_service import (
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.config import Settings
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import (
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_MAX_TOOL_CALLS,
    TaskBudgets,
    TaskMode,
    TaskStatus,
)
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.approval_lifecycle import (
    DatabaseApprovalLifecycle,
)
from rag_kb.infrastructure.database.models import (
    PrincipalRow,
    TenantRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_execution import DatabaseTaskWorkerSettlement
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.tool_execution_lifecycle import (
    DatabaseToolExecutionLifecycle,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore


# ---------- 修复 4：ExecutionGuardMiddleware 停止边界 ----------


def _tool_message(
    call_id: str,
    *,
    failed: bool,
    tool_name: str = "search_knowledge",
) -> ToolMessage:
    payload = {
        "tool_name": tool_name,
        "status": "failed" if failed else "succeeded",
        "error_code": "tool_timeout" if failed else None,
    }
    return ToolMessage(json.dumps(payload), tool_call_id=call_id, name=tool_name)


def _state(*messages: object) -> dict[str, object]:
    return {"messages": list(messages)}


def _call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_three_consecutive_failed_tool_messages_stop_the_agent() -> None:
    guard = ExecutionGuardMiddleware()
    messages = (
        _call("search_knowledge", {"query": "a"}, "a"),
        _tool_message("a", failed=True),
        _call("search_knowledge", {"query": "b"}, "b"),
        _tool_message("b", failed=True),
        _call("search_knowledge", {"query": "c"}, "c"),
        _tool_message("c", failed=True),
    )

    with pytest.raises(AgentGuardError, match="agent_consecutive_tool_failures"):
        guard.before_model(_state(*messages), runtime=None)

    # 少于 3 条连续失败不触发；非连续失败也不累计。
    guard.before_model(_state(*messages[:4]), runtime=None)
    guard.before_model(
        _state(
            _tool_message("a", failed=True),
            _tool_message("b", failed=False),
            _tool_message("c", failed=True),
            _tool_message("d", failed=True),
        ),
        runtime=None,
    )


def test_repeated_same_arguments_tool_call_stops_the_agent() -> None:
    guard = ExecutionGuardMiddleware()
    repeated = (
        _call("search_knowledge", {"query": "q"}, "1"),
        _call("search_knowledge", {"query": "q"}, "2"),
    )
    guard.before_model(_state(repeated[0]), runtime=None)
    with pytest.raises(AgentGuardError, match="agent_repeated_tool_call"):
        guard.before_model(_state(*repeated), runtime=None)

    different_args = (
        _call("search_knowledge", {"query": "q"}, "1"),
        _call("search_knowledge", {"query": "其他"}, "2"),
    )
    guard.before_model(_state(*different_args), runtime=None)


def test_guard_ignores_todo_structured_unknown_and_uncontrolled_tool_messages() -> None:
    guard = ExecutionGuardMiddleware()
    ignored_failures = (
        *(
            _tool_message(str(index), failed=True, tool_name="write_todos")
            for index in range(3)
        ),
        *(
            _tool_message(str(index), failed=True, tool_name="TaskResultCandidate")
            for index in range(3, 6)
        ),
        ToolMessage("not-json", tool_call_id="ignored", name="search_knowledge"),
    )
    guard.before_model(_state(*ignored_failures), runtime=None)

    ignored_calls = (
        _call("write_todos", {"todos": []}, "todo-1"),
        _call("write_todos", {"todos": []}, "todo-2"),
        _call("TaskResultCandidate", {}, "result-1"),
        _call("TaskResultCandidate", {}, "result-2"),
        _call("unknown_tool", {}, "unknown-1"),
        _call("unknown_tool", {}, "unknown-2"),
    )
    guard.before_model(_state(*ignored_calls), runtime=None)


def test_guard_rescans_full_history_after_checkpoint_restore() -> None:
    """验收④：中间件为纯函数扫描，恢复后从 checkpoint 重算仍然触发。"""
    guard = ExecutionGuardMiddleware()
    restored_history = (
        _call("fetch_evidence", {"evidence_ids": ["e-1"]}, "old-1"),
        _tool_message("old-1", failed=True),
        _tool_message("old-2", failed=True),
        _tool_message("old-3", failed=True),
    )
    with pytest.raises(AgentGuardError, match="agent_consecutive_tool_failures"):
        guard.before_model(_state(*restored_history), runtime=None)


def test_agent_factory_registers_execution_guard_and_budget_defaults_match_config() -> None:
    captured: dict[str, object] = {}
    import rag_kb.agent.factory as factory_module

    original = factory_module.create_agent
    factory_module.create_agent = lambda **kwargs: captured.update(kwargs) or "agent"
    try:
        build_research_agent(
            "model",
            ["tool"],
            "checkpointer",
            max_model_calls=DEFAULT_MAX_MODEL_CALLS,
            max_tool_calls=DEFAULT_MAX_TOOL_CALLS,
        )
    finally:
        factory_module.create_agent = original

    names = {type(middleware).__name__ for middleware in captured["middleware"]}
    assert "ExecutionGuardMiddleware" in names

    settings = Settings()
    budgets = TaskBudgets()
    assert settings.agent_max_model_calls == DEFAULT_MAX_MODEL_CALLS == budgets.max_model_calls
    assert settings.agent_max_tool_calls == DEFAULT_MAX_TOOL_CALLS == budgets.max_tool_calls


# ---------- 修复 4 P0：deadline 在 invoke 期间过期的终检 ----------


class _AuthContexts:
    def resolve(self, **_kwargs) -> AuthContext:
        return _auth()


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )


class _SnapshotAgent:
    def __init__(self) -> None:
        self.invoked = False

    def get_state(self, _config):
        if not self.invoked:
            return SimpleNamespace(values={}, interrupts=(), next=())
        return SimpleNamespace(
            values={
                "structured_response": {
                    "status": "refused",
                    "refusal_code": "insufficient_evidence",
                    "stop_reason": "no_evidence",
                }
            },
            interrupts=(),
            next=(),
        )

    def invoke(self, _input, _config, *, context):
        self.invoked = True
        return {}


class _DeadlineExpiresAfterInvokeRepository(DatabaseTaskRepository):
    def __init__(self, database, agent: _SnapshotAgent) -> None:
        super().__init__(database)
        self._agent = agent
        self.current_time_calls = 0

    def current_time(self) -> datetime:
        self.current_time_calls += 1
        return super().current_time()

    def refresh_lease(self, lease, *, lease_seconds=None):
        refreshed = super().refresh_lease(lease, lease_seconds=lease_seconds)
        if self._agent.invoked and refreshed.state.deadline is not None:
            expired = datetime.now(UTC) - timedelta(seconds=1)
            return replace(
                refreshed,
                state=refreshed.state.model_copy(update={"deadline": expired}),
            )
        return refreshed


def test_deadline_expiring_during_invoke_fails_task_without_succeeding(tmp_path: Path) -> None:
    from rag_kb.agent.result import AgentResultStore
    from rag_kb.application.task_worker import TaskWorker

    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'deadline.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    agent = _SnapshotAgent()

    repository = _DeadlineExpiresAfterInvokeRepository(database, agent)
    objects = LocalObjectStore(tmp_path / "objects")
    worker = TaskWorker(
        repository=repository,
        auth_contexts=_AuthContexts(),
        agent_for=lambda _budgets: agent,
        results=AgentResultStore(objects),
        settlement=DatabaseTaskWorkerSettlement(database, objects),
        tools=SimpleNamespace(close=lambda: None),
        worker_id="worker-a",
        lease_seconds=60,
    )
    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "task_deadline_expired"
    assert repository.get_task(auth=_auth(), task_id="task-a").status is TaskStatus.FAILED
    assert repository.current_time_calls >= 2


# ---------- 修复 6：refused 结果契约 ----------


def test_refused_candidate_with_evidence_ids_is_invalid() -> None:
    candidate = TaskResultCandidate(
        status="refused",
        refusal_code="insufficient_evidence",
        stop_reason="no_evidence",
        evidence_ids=("evidence-1",),
    )
    with pytest.raises(ValueError, match="task_result_candidate_invalid"):
        candidate.require_valid()


def test_build_task_result_strips_evidence_from_refused_results() -> None:
    candidate = TaskResultCandidate(
        status="refused",
        refusal_code="insufficient_evidence",
        stop_reason="no_evidence",
        evidence_ids=("evidence-1",),
    )
    result = build_task_result(candidate, task_id="task-a", citations=("sentinel",))

    assert result.status == "refused"
    assert result.evidence_ids == ()
    assert result.citations == ()


# ---------- 修复 7：READ 失败语义与重试分类 ----------


class _ExplodingReadAdapter:
    def execute(self, _arguments, _context):
        from rag_kb.infrastructure.tools.read_tools import ReadToolError

        raise ReadToolError("document_version_not_found")

    def verify(self, _result, _context) -> bool:
        return False


def test_read_tool_error_code_survives_into_execution_result(tmp_path: Path) -> None:
    """验收②：ReadToolError 错误码原样保留，不被压成类型名拼接码。"""
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'code.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "search_knowledge"
    )
    service = ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
        ToolRegistry([definition], adapters={"search_knowledge": _ExplodingReadAdapter()}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
        tool_results=LocalObjectStore(tmp_path / "tool-results"),
    )
    try:
        result = service.execute(
            ToolExecutionRequest(
                auth=_auth(),
                task_id="task-a",
                tool_name="search_knowledge",
                arguments={"query": "q"},
                worker_id=lease.worker_id,
                lease_generation=lease.lease_generation,
                idempotency_key="k-1",
            )
        )
    finally:
        service.close()

    assert result.status == "failed"
    assert result.error_code == "document_version_not_found"


def test_tool_execution_does_not_swallow_keyboard_interrupt(tmp_path: Path) -> None:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'interrupt.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None

    class InterruptingAdapter:
        def execute(self, _arguments, _context):
            raise KeyboardInterrupt

        def verify(self, _result, _context) -> bool:
            return False

    definition = next(
        item for item in builtin_tool_definitions() if item.name == "search_knowledge"
    )
    service = ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
        ToolRegistry([definition], adapters={"search_knowledge": InterruptingAdapter()}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
        tool_results=LocalObjectStore(tmp_path / "tool-results"),
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            service.execute(
                ToolExecutionRequest(
                    auth=_auth(),
                    task_id="task-a",
                    tool_name="search_knowledge",
                    arguments={"query": "q"},
                    worker_id=lease.worker_id,
                    lease_generation=lease.lease_generation,
                    idempotency_key="interrupt-key",
                )
            )
    finally:
        service.close()


def test_retryable_classification_follows_whitelist() -> None:
    """验收③：瞬时类可重试；权限/参数/版本类永久错误不可重试。"""
    transient = ("tool_timeout", "tool_execution_unavailable", "previous_execution_in_progress", "execution_deadline_expired")
    permanent = (
        "document_read_denied",
        "document_version_not_found",
        "document_version_order_invalid",
        "comparison_version_not_indexed",
        "tool_arguments_invalid",
        "idempotency_key_conflict",
        "tool_plan_binding_invalid",
        "approval_not_approved",
        "tool_current_binding_invalid",
    )
    for code in transient:
        payload = json.loads(render_tool_message("search_knowledge", None, error_code=code))
        assert payload["retryable"] is True, code
    for code in permanent:
        payload = json.loads(render_tool_message("search_knowledge", None, error_code=code))
        assert payload["retryable"] is False, code


def test_sweeper_converges_read_to_failed_and_keeps_write_unknown(tmp_path: Path) -> None:
    """验收①：READ 过期收敛为 failed，WRITE 保持 unknown。"""
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'sweep.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="写任务", task_mode=TaskMode.CONTROLLED_ACTION, task_id="task-w")
    repository.create_task(auth=_auth(), goal="读任务", task_mode=TaskMode.RESEARCH, task_id="task-r")
    now = datetime.now(UTC)
    read_row = uuid4().hex
    write_row = uuid4().hex
    with database.sessions.begin() as session:
        session.add_all(
            [
                ToolCallRow(
                    tool_call_id=read_row,
                    task_id="task-r",
                    tenant_id="tenant-a",
                    tool_name="search_knowledge",
                    tool_version="1.0.0",
                    arguments_hash=arguments_fingerprint({"query": "q"}),
                    principal_hash=principal_fingerprint(_auth()),
                    mode="read",
                    risk_level="low",
                    status="running",
                    created_at=now - timedelta(minutes=2),
                    execution_deadline=now - timedelta(minutes=1),
                ),
                ToolCallRow(
                    tool_call_id=write_row,
                    task_id="task-w",
                    tenant_id="tenant-a",
                    approval_id="approval-dummy",
                    idempotency_key="write-key-1",
                    tool_name="create_report_artifact",
                    tool_version="1.0.0",
                    arguments_hash=arguments_fingerprint({}),
                    principal_hash=principal_fingerprint(_auth()),
                    mode="write",
                    risk_level="low",
                    status="running",
                    created_at=now - timedelta(minutes=2),
                    execution_deadline=now - timedelta(minutes=1),
                ),
            ]
        )
    service = ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
        ToolRegistry(adapters={}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
        tool_results=LocalObjectStore(tmp_path / "objects"),
    )
    try:
        assert service.expire_stale_running() == 2
    finally:
        service.close()
    with database.sessions() as session:
        read = session.get(ToolCallRow, read_row)
        write = session.get(ToolCallRow, write_row)
        assert read is not None and read.status == "failed"
        assert write is not None and write.status == "unknown"
        assert read.error_code == write.error_code == "execution_deadline_expired"


def test_expired_tool_result_readback_returns_non_retryable_failure(
    tmp_path: Path,
) -> None:
    """修复 5 验收①：过期结果回读返回 tool_result_expired 且不可重试。"""
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'ttl.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None

    class _OkAdapter:
        def execute(self, _arguments, _context):
            return {"result_reference": "ref", "evidence_ids": [], "items": []}

        def verify(self, _result, _context) -> bool:
            return True

    definition = next(
        item for item in builtin_tool_definitions() if item.name == "search_knowledge"
    )
    service = ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
        ToolRegistry([definition], adapters={"search_knowledge": _OkAdapter()}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
        tool_results=LocalObjectStore(tmp_path / "tool-results"),
    )
    request = ToolExecutionRequest(
        auth=_auth(),
        task_id="task-a",
        tool_name="search_knowledge",
        arguments={"query": "q"},
        worker_id=lease.worker_id,
        lease_generation=lease.lease_generation,
        idempotency_key="k-ttl",
    )
    try:
        first = service.execute(request)
        assert first.status == "succeeded"
        with database.sessions.begin() as session:
            row = session.get(ToolCallRow, first.tool_call_id)
            assert row is not None
            row.result_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        replay = service.execute(request)
    finally:
        service.close()

    assert replay.duplicate is True
    assert replay.status == "failed"
    assert replay.error_code == "tool_result_expired"
    payload = json.loads(render_tool_message("search_knowledge", None, error_code="tool_result_expired"))
    assert payload["retryable"] is False
