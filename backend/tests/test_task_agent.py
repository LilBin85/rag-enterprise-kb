"""M1 create_agent 工厂、工具适配器与可恢复结果单元契约。"""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from rag_kb.agent import factory
from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.result import TaskResult, TaskResultCandidate, build_task_result
from rag_kb.agent.tools import (
    CONTROLLED_WRITE_TOOL_NAMES,
    RESEARCH_TOOL_NAMES,
    _execute_read_tool,
    build_controlled_action_tools,
    build_research_tools,
    render_tool_message,
    render_write_tool_message,
)
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_result_service import TaskResultError, TaskResultService
from rag_kb.application.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskState, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.approval_lifecycle import (
    DatabaseApprovalLifecycle,
)
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerConflict
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    PrincipalRow,
    TenantRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.task_read import DatabaseTaskResultReader
from rag_kb.infrastructure.database.tool_execution_lifecycle import (
    DatabaseToolExecutionLifecycle,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.models import Citation


class ReadAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _arguments, _context):
        self.calls += 1
        return {"result_reference": "server-only", "evidence_ids": [], "items": []}

    def verify(self, _result, _context) -> bool:
        return True


class RecordingExecutor:
    def __init__(self) -> None:
        self.request: ToolExecutionRequest | None = None

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        self.request = request
        return ToolExecutionResult(
            tool_call_id="server-tool-call",
            status="succeeded",
            result_hash="sha256:" + "a" * 64,
            result={
                "evidence_ids": ["evidence-1"],
                "items": [{"text": "可见正文", "tenant_id": "不得泄露"}],
            },
        )


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )


def _context() -> AgentRunContext:
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


def test_agent_factory_configures_only_research_tools_and_m1_guards(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(factory, "create_agent", lambda **kwargs: captured.update(kwargs) or "agent")

    result = factory.build_research_agent(
        "model",
        ["tool"],
        "checkpointer",
        max_model_calls=2,
        max_tool_calls=3,
    )

    assert result == "agent"
    assert captured["tools"] == ["tool"]
    assert isinstance(captured["response_format"], ToolStrategy)
    assert captured["response_format"].schema is TaskResultCandidate
    assert captured["context_schema"] is AgentRunContext
    assert captured["checkpointer"] == "checkpointer"
    assert [type(item).__name__ for item in captured["middleware"]] == [
        "TodoListMiddleware",
        "SingleBusinessToolCallMiddleware",
        "ClarificationInterruptMiddleware",
        "ExecutionGuardMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    ]
    assert captured["middleware"][-2].thread_limit == 2
    assert captured["middleware"][-1].thread_limit == 3


def test_research_tools_are_exactly_four_and_use_trusted_context() -> None:
    executor = RecordingExecutor()
    tools = build_research_tools(executor)
    assert tuple(tool.name for tool in tools) == RESEARCH_TOOL_NAMES

    message = _execute_read_tool(
        "search_knowledge",
        {"query": "采购制度", "limit": 5},
        SimpleNamespace(context=_context(), tool_call_id="model-call-1"),
        executor,
    )
    payload = json.loads(message)
    assert executor.request is not None
    assert executor.request.auth == _context().auth
    assert executor.request.plan_revision is None
    assert executor.request.evidence_digest is None
    assert payload["content"] == [{"text": "可见正文"}]
    assert payload["result_ref"] == "server-tool-call"
    assert "tenant_id" not in message


def test_research_tool_preserves_code_from_unexpected_exception() -> None:
    class CodedError(RuntimeError):
        code = "adapter_backend_unavailable"

    class CodedExecutor:
        def execute(self, _request):
            raise CodedError()

    message = _execute_read_tool(
        "search_knowledge",
        {"query": "采购制度", "limit": 5},
        SimpleNamespace(context=_context(), tool_call_id="model-call-1"),
        CodedExecutor(),
    )

    assert json.loads(message)["error_code"] == "adapter_backend_unavailable"


def test_research_tool_hides_and_accepts_injected_runtime() -> None:
    tool = next(item for item in build_research_tools(RecordingExecutor()) if item.name == "list_document_versions")

    assert "runtime" not in tool.tool_call_schema.model_fields
    message = tool.invoke(
        {
            "type": "tool_call",
            "name": tool.name,
            "id": "model-call-1",
            "args": {
                "document_id": "document-a",
                "runtime": ToolRuntime(
                    state={},
                    context=_context(),
                    config={},
                    stream_writer=lambda _value: None,
                    tool_call_id="model-call-1",
                    store=None,
                ),
            },
        }
    )

    assert json.loads(message.content)["tool_name"] == "list_document_versions"


def test_tool_message_crops_content_and_preserves_failure_semantics() -> None:
    result = ToolExecutionResult(
        tool_call_id="tool-1",
        status="succeeded",
        result_hash="sha256:" + "a" * 64,
        result={
            "evidence_ids": ["evidence-1"],
            "items": [{"text": "x" * 2_100, "storage_key": "secret/path"}],
        },
    )
    success = json.loads(render_tool_message("search_knowledge", result))
    failed = json.loads(render_tool_message("search_knowledge", None, error_code="tool_timeout"))

    assert success["content"] == [{"text": "x" * 2_000}]
    assert success["result_ref"] == "tool-1"
    assert "storage_key" not in json.dumps(success)
    assert failed == {
        "error_code": "tool_timeout",
        "retryable": True,
        "status": "failed",
        "tool_call_id": None,
        "tool_name": "search_knowledge",
    }


def test_compare_tool_message_preserves_only_bounded_change_content() -> None:
    result = ToolExecutionResult(
        tool_call_id="tool-compare",
        status="succeeded",
        result_hash="sha256:" + "a" * 64,
        result={
            "evidence_ids": ["evidence-1", "evidence-2"],
            "items": [
                {
                    "document_id": "procurement",
                    "left_version": 1,
                    "right_version": 2,
                    "changes": [
                        {
                            "kind": "replace",
                            "left": [
                                {
                                    "chunk_id": "v1:chunk-1",
                                    "text": "旧审批规则",
                                    "evidence_id": "evidence-1",
                                    "storage_key": "hidden",
                                }
                            ],
                            "right": [
                                {
                                    "chunk_id": "v2:chunk-1",
                                    "text": "新审批规则",
                                    "evidence_id": "evidence-2",
                                    "storage_key": "hidden",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    payload = json.loads(render_tool_message("compare_document_versions", result))

    assert payload["content"][0]["changes"][0]["left"] == [
        {"chunk_id": "v1:chunk-1", "evidence_id": "evidence-1", "text": "旧审批规则"}
    ]
    assert "storage_key" not in json.dumps(payload)


def test_read_tool_success_is_persisted_and_idempotent_readback(tmp_path: Path) -> None:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'tool.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=_auth().attributes))
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    adapter = ReadAdapter()
    definition = next(item for item in builtin_tool_definitions() if item.name == "search_knowledge")
    service = ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
        ToolRegistry([definition], adapters={"search_knowledge": adapter}),
        ApprovalService(DatabaseApprovalLifecycle(database)),
        tool_results=LocalObjectStore(tmp_path / "tool-results"),
    )
    request = ToolExecutionRequest(
        auth=_auth(),
        task_id="task-a",
        tool_name="search_knowledge",
        arguments={"query": "采购制度"},
        worker_id=lease.worker_id,
        lease_generation=lease.lease_generation,
        idempotency_key="agent-call-1",
    )
    try:
        first = service.execute(request)
        duplicate = service.execute(request)
    finally:
        service.close()

    assert first.status == duplicate.status == "succeeded"
    assert first.result_ref is not None
    assert duplicate.duplicate is True
    assert duplicate.result == first.result
    assert adapter.calls == 1
    with database.sessions() as session:
        row = session.get(ToolCallRow, first.tool_call_id)
        assert row is not None
        assert row.result_ref == first.result_ref
        assert row.result_size is not None and row.result_expires_at is not None


def test_candidate_result_requires_server_citations() -> None:
    candidate = TaskResultCandidate(
        status="succeeded",
        answer="V2 增加了育儿假。",
        evidence_ids=("evidence-1",),
        stop_reason="completed",
    ).require_valid()
    citation = Citation(
        document_id="leave-policy",
        document_version=2,
        source_file_name="leave-v2.md",
        source_locator="育儿假",
        chunk_id="chunk-1",
    )
    result = build_task_result(candidate, task_id="task-a", citations=(citation,))
    assert result.citations == (citation,)
    assert result.evidence_ids == ("evidence-1",)

    with pytest.raises(ValueError, match="task_result_candidate_invalid"):
        TaskResultCandidate(status="succeeded").require_valid()


def test_result_read_revalidates_current_evidence_and_citation_binding() -> None:
    citation = Citation(
        document_id="procurement",
        document_version=2,
        source_file_name="v2.md",
        source_locator="审批",
        chunk_id="v2:chunk-1",
    )
    result = TaskResult(
        task_id="task-a",
        status="succeeded",
        answer="V2 增加财务审批。",
        evidence_ids=("evidence-1",),
        citations=(citation,),
        created_at=datetime.now(UTC),
    )
    state = TaskState(
        task_id="task-a",
        auth_context_ref="auth",
        goal="比较制度",
        task_mode=TaskMode.RESEARCH,
        status=TaskStatus.SUCCEEDED,
        result_kind="research",
        result_ref="task-results/task-a/result.json",
        result_hash="sha256:" + "a" * 64,
        content_expires_at=datetime.now(UTC) + timedelta(days=7),
    )

    class ResultStore:
        def read_task_result(self, _ref, _hash):
            return result

    class Evidence:
        def __init__(self, citations):
            self.citations = citations

        def citations_from_ledger(self, _session, **_kwargs):
            if isinstance(self.citations, Exception):
                raise self.citations
            return self.citations

    class Database:
        engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def sessions(self):
            return nullcontext(
                SimpleNamespace(
                    scalar=lambda _statement: SimpleNamespace(
                        status=TaskStatus.SUCCEEDED.value,
                        result_purged_at=None,
                        content_expires_at=state.content_expires_at,
                        result_ref=state.result_ref,
                        result_hash=state.result_hash,
                    )
                )
            )

    service = TaskResultService(
        DatabaseTaskResultReader(Database(), ResultStore(), Evidence((citation,)))
    )
    assert service.read(auth=_auth(), state=state) == result

    revoked = TaskResultService(DatabaseTaskResultReader(Database(), ResultStore(), Evidence(())))
    with pytest.raises(TaskResultError, match="task_result_binding_invalid"):
        revoked.read(auth=_auth(), state=state)

    for conflict_code, expected_code in (
        ("duplicate_evidence_id", "task_result_binding_invalid"),
        ("evidence_authorization_stale", "task_result_evidence_unavailable"),
    ):
        conflicted = TaskResultService(DatabaseTaskResultReader(
            Database(),
            ResultStore(),
            Evidence(EvidenceLedgerConflict(conflict_code)),
        ))
        with pytest.raises(TaskResultError, match=expected_code):
            conflicted.read(auth=_auth(), state=state)


def test_result_read_rejects_expired_body_before_object_read() -> None:
    now = datetime.now(UTC)
    state = TaskState(
        task_id="task-a",
        auth_context_ref="auth",
        goal="比较制度",
        task_mode=TaskMode.RESEARCH,
        status=TaskStatus.REFUSED,
        result_kind="research",
        result_ref="task-results/task-a/result.json",
        result_hash="sha256:" + "a" * 64,
        content_expires_at=now - timedelta(seconds=1),
    )

    class ResultStore:
        def read_task_result(self, _ref, _hash):
            raise AssertionError("expired result must not be read")

    database = SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))
    service = TaskResultService(
        DatabaseTaskResultReader(database, ResultStore(), now=lambda: now)
    )
    with pytest.raises(TaskResultError, match="task_result_expired"):
        service.read(auth=_auth(), state=state)


@pytest.mark.parametrize(
    ("change", "error_code"),
    [
        ("expired", "task_result_expired"),
        ("purged", "task_result_expired"),
        ("rebound", "task_result_binding_invalid"),
    ],
)
def test_result_read_rechecks_task_binding_after_object_read(
    tmp_path: Path, change: str, error_code: str
) -> None:
    now = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'result.sqlite3').as_posix()}")
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
    repository.create_task(
        auth=_auth(),
        goal="读取结果竞态",
        task_mode=TaskMode.RESEARCH,
        task_id="task-a",
    )
    result_ref = "task-results/task-a/result.json"
    result_hash = "sha256:" + "a" * 64
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "task-a")
        assert row is not None
        row.status = TaskStatus.REFUSED.value
        row.result_kind = "research"
        row.result_ref = result_ref
        row.result_hash = result_hash
        row.content_expires_at = now + timedelta(days=1)
    state = repository.get_task(auth=_auth(), task_id="task-a")
    result = TaskResult(
        task_id="task-a",
        status="refused",
        refusal_code="insufficient_evidence",
        stop_reason="no_evidence",
        created_at=now,
    )

    class ResultStore:
        def read_task_result(self, _ref, _hash):
            with database.sessions.begin() as session:
                row = session.get(AgentTaskRow, "task-a")
                assert row is not None
                if change == "expired":
                    row.content_expires_at = now - timedelta(seconds=1)
                elif change == "purged":
                    row.result_purged_at = now
                    row.result_ref = None
                else:
                    row.result_ref = "task-results/task-a/rebound.json"
            return result

    service = TaskResultService(
        DatabaseTaskResultReader(database, ResultStore(), now=lambda: now)
    )
    with pytest.raises(TaskResultError, match=error_code):
        service.read(auth=_auth(), state=state)


def test_result_read_uses_postgres_clock_for_expiry() -> None:
    database_now = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)
    state = TaskState(
        task_id="task-a",
        auth_context_ref="auth",
        goal="比较制度",
        task_mode=TaskMode.RESEARCH,
        status=TaskStatus.REFUSED,
        result_kind="research",
        result_ref="task-results/task-a/result.json",
        result_hash="sha256:" + "a" * 64,
        content_expires_at=database_now - timedelta(seconds=1),
    )

    class Session:
        def scalar(self, _statement):
            return database_now

    class Database:
        engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        def sessions(self):
            return nullcontext(Session())

    class ResultStore:
        def read_task_result(self, _ref, _hash):
            raise AssertionError("expired result must not be read")

    service = TaskResultService(DatabaseTaskResultReader(
        Database(),
        ResultStore(),
        now=lambda: database_now - timedelta(days=1),
    ))
    with pytest.raises(TaskResultError, match="task_result_expired"):
        service.read(auth=_auth(), state=state)


# ---------- M2 批次 1：受控动作 Agent 与 WRITE 薄 adapter ----------


class RecordingWriteExecutor:
    def __init__(self) -> None:
        self.requests: list[ToolExecutionRequest] = []

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        self.requests.append(request)
        raise ToolExecutionError("approval_required")


def test_controlled_agent_factory_exposes_hitl_on_exactly_two_writes(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(factory, "create_agent", lambda **kwargs: captured.update(kwargs) or "agent")

    result = factory.build_controlled_action_agent(
        "model",
        ["tool"],
        "checkpointer",
        max_model_calls=2,
        max_tool_calls=3,
    )

    assert result == "agent"
    assert captured["tools"] == ["tool"]
    assert isinstance(captured["response_format"], ToolStrategy)
    assert captured["response_format"].schema is TaskResultCandidate
    assert captured["context_schema"] is AgentRunContext
    assert captured["checkpointer"] == "checkpointer"
    middlewares = captured["middleware"]
    names = [type(item).__name__ for item in middlewares]
    assert names == [
        "TodoListMiddleware",
        "HumanInTheLoopMiddleware",
        "SingleBusinessToolCallMiddleware",
        "ClarificationInterruptMiddleware",
        "ExecutionGuardMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    ]
    hitl = middlewares[1]
    assert set(hitl.interrupt_on) == {
        "create_report_artifact",
        "create_work_item_draft",
    }
    for config in hitl.interrupt_on.values():
        assert set(config["allowed_decisions"]) == {"approve", "reject"}
        assert "when" not in config
    assert middlewares[-2].thread_limit == 2
    assert middlewares[-1].thread_limit == 3


def test_controlled_tools_are_four_reads_plus_exactly_two_writes() -> None:
    executor = RecordingExecutor()
    write_executor = RecordingWriteExecutor()
    tools = build_controlled_action_tools(write_executor)

    names = tuple(tool.name for tool in tools)
    assert names == RESEARCH_TOOL_NAMES + CONTROLLED_WRITE_TOOL_NAMES
    write_names = [name for name in names if name.startswith("create_")]
    assert write_names == ["create_report_artifact", "create_work_item_draft"]


def test_write_tool_uses_server_context_and_rejects_injected_control_fields() -> None:
    executor = RecordingWriteExecutor()
    tools = build_controlled_action_tools(executor)
    report_tool = next(tool for tool in tools if tool.name == "create_report_artifact")

    message = report_tool.invoke(
        {
            "type": "tool_call",
            "name": report_tool.name,
            "id": "model-call-1",
            "args": {
                "title": "报告",
                "content": "正文",
                "format": "markdown",
                "runtime": ToolRuntime(
                    state={},
                    context=_context(),
                    config={},
                    stream_writer=lambda _value: None,
                    tool_call_id="model-call-1",
                    store=None,
                ),
            },
        }
    )
    payload = json.loads(message.content)
    # 控制字段全部来自服务端 AgentRunContext（批次 1 恒无审批事实），
    # 请求在授权门 fail-closed，业务副作用为 0。
    assert payload["error_code"] == "approval_required"
    assert payload["status"] == "failed"
    assert executor.requests, "请求必须经服务端 executor 派发"
    request = executor.requests[0]
    assert request.auth == _context().auth
    assert request.task_id == "task-a"
    assert request.worker_id == "worker-a"
    assert request.lease_generation == 1
    assert request.approval_id is None
    assert request.plan_revision is None
    assert request.idempotency_key is None
    assert request.arguments == {"title": "报告", "content": "正文", "format": "markdown"}

    # 模型注入控制字段：schema additionalProperties=False 在工具边界拒绝。
    injected = {
        "type": "tool_call",
        "name": report_tool.name,
        "id": "model-call-2",
        "args": {
            "title": "报告",
            "content": "正文",
            "format": "markdown",
            "tenant_id": "tenant-evil",
            "roles": ["admin"],
            "approval_id": "forged",
            "idempotency_key": "forged",
            "thread_id": "evil-thread",
            "checkpoint_id": "evil-checkpoint",
            "storage_key": "secret/path",
        },
    }
    with pytest.raises(Exception):
        report_tool.invoke(injected)
    assert len(executor.requests) == 1


def test_write_tool_runtime_context_is_not_model_controllable() -> None:
    executor = RecordingWriteExecutor()
    tools = build_controlled_action_tools(executor)
    draft_tool = next(tool for tool in tools if tool.name == "create_work_item_draft")

    assert "runtime" not in draft_tool.tool_call_schema.model_fields
    for field in draft_tool.tool_call_schema.model_fields:
        assert field not in {
            "tenant_id",
            "roles",
            "approval_id",
            "idempotency_key",
            "thread_id",
            "checkpoint_id",
            "storage_key",
            "worker_id",
            "lease_generation",
        }

    # runtime 由框架注入；模型无法通过参数覆盖（extra 参数在 schema 层被拒）。
    message = draft_tool.invoke(
        {
            "type": "tool_call",
            "name": draft_tool.name,
            "id": "model-call-3",
            "args": {
                "title": "草稿",
                "description": "描述",
                "runtime": ToolRuntime(
                    state={},
                    context=_context(),
                    config={},
                    stream_writer=lambda _value: None,
                    tool_call_id="model-call-3",
                    store=None,
                ),
            },
        }
    )
    payload = json.loads(message.content)
    assert payload["error_code"] == "approval_required"
    request = executor.requests[0]
    assert request.arguments == {"title": "草稿", "description": "描述"}


class _ToolCallingFakeModel(GenericFakeChatModel):
    """锁定版本契约测试用的假模型：支持 bind_tools 并回放预设消息。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def test_controlled_agent_hitl_interrupt_pauses_before_write_execution() -> None:
    """锁定版本（langchain 1.3.14）HITL 契约：interrupt 在 ToolNode 前触发。"""
    executor = RecordingWriteExecutor()
    tools = build_controlled_action_tools(executor)
    write_arguments = {"title": "报告", "content": "正文", "format": "markdown"}
    model = _ToolCallingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_report_artifact",
                            "args": dict(write_arguments),
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
    )
    agent = factory.build_controlled_action_agent(
        model,
        tools,
        InMemorySaver(),
        max_model_calls=4,
        max_tool_calls=4,
    )
    context = AgentRunContext(
        task_id="task-hitl",
        tenant_id="tenant-a",
        user_id="owner-a",
        task_mode=TaskMode.CONTROLLED_ACTION,
        worker_id="worker-a",
        lease_generation=1,
        run_id="run-a",
        auth=_auth(),
    )
    config = {"configurable": {"thread_id": "task-hitl"}}

    agent.invoke(
        {"messages": [HumanMessage(content="基于证据生成报告")]},
        config=config,
        context=context,
    )

    snapshot = agent.get_state(config)
    interrupts = tuple(snapshot.interrupts or ())
    assert len(interrupts) == 1
    value = interrupts[0].value
    assert set(value) == {"action_requests", "review_configs"}
    requests = value["action_requests"]
    assert len(requests) == 1
    assert requests[0]["name"] == "create_report_artifact"
    assert requests[0]["args"] == write_arguments
    assert set(requests[0]) == {"name", "args", "description"}
    assert value["review_configs"] == [
        {"action_name": "create_report_artifact", "allowed_decisions": ["approve", "reject"]}
    ]
    # HITL 未恢复：WRITE adapter 零调用（请求未到达 ToolExecutionService）。
    assert executor.requests == []
    # checkpoint 坐标可用于派生 action 绑定。
    configurable = snapshot.config["configurable"]
    assert configurable["thread_id"] == "task-hitl"
    assert configurable["checkpoint_id"]
    last_call = snapshot.values["messages"][-1].tool_calls[-1]
    assert last_call["id"] == "call_1"


def _write_result(**extra: object) -> ToolExecutionResult:
    body = {
        "artifact_type": "report",
        "artifact_id": "artifact-1",
        "content_hash": "sha256:" + "a" * 64,
        "status": "ready",
    }
    body.update(extra)
    return ToolExecutionResult(
        tool_call_id="call_1",
        status="succeeded",
        result_hash="sha256:" + "b" * 64,
        result=body,
    )


def test_write_tool_message_success_render_exposes_artifact_id_only() -> None:
    """WRITE 成功渲染只允许 artifact_id/hash/status；storage_key 必须缺席（C-2）。"""
    payload = json.loads(
        render_write_tool_message(
            "create_report_artifact",
            _write_result(storage_key="artifacts/leaked/path.md"),
        )
    )
    assert payload["status"] == "succeeded"
    assert payload["tool_call_id"] == "call_1"
    assert payload["artifact_id"] == "artifact-1"
    assert payload["artifact_type"] == "report"
    assert payload["content_hash"] == "sha256:" + "a" * 64
    assert "storage_key" not in payload


def test_write_tool_message_work_item_draft_render_exposes_artifact_id_only() -> None:
    payload = json.loads(
        render_write_tool_message(
            "create_work_item_draft",
            _write_result(
                artifact_type="work_item_draft",
                artifact_id="artifact-2",
            ),
        )
    )
    assert payload["artifact_id"] == "artifact-2"
    assert payload["artifact_type"] == "work_item_draft"
    assert "storage_key" not in payload
