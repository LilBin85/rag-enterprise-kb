"""M1 create_agent 工厂、工具适配器与可恢复结果单元契约。"""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.structured_output import ToolStrategy
from langgraph.prebuilt import ToolRuntime

from rag_kb.agent import factory
from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.result import TaskResult, TaskResultCandidate, build_task_result
from rag_kb.agent.tools import (
    RESEARCH_TOOL_NAMES,
    _execute_read_tool,
    build_research_tools,
    render_tool_message,
)
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_result_service import TaskResultError, TaskResultService
from rag_kb.application.tool_execution_service import ToolExecutionRequest, ToolExecutionResult, ToolExecutionService
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskState, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import PrincipalRow, TenantRow, ToolCallRow
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
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

    result = factory.build_research_agent("model", ["tool"], "checkpointer")

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
        database,
        ToolRegistry([definition], adapters={"search_knowledge": adapter}),
        ApprovalService(database),
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
            return self.citations

    class Database:
        def sessions(self):
            return nullcontext(object())

    service = TaskResultService(Database(), ResultStore(), Evidence((citation,)))
    assert service.read(auth=_auth(), state=state) == result

    revoked = TaskResultService(Database(), ResultStore(), Evidence(()))
    with pytest.raises(TaskResultError, match="task_result_binding_invalid"):
        revoked.read(auth=_auth(), state=state)
