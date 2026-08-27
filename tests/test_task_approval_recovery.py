"""M2 批次 1：HITL 暂停、interrupt 严格分流与审批事实故障注入。

双层审批门（批次 1 切面）：HITL 未恢复或业务审批未创建时，两个 WRITE
业务 adapter 调用次数必须为 0；clarification 与 approval 两类 interrupt
互斥分流，错误形状一律 fail-closed。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select, text

from rag_kb.agent.result import AgentResultStore
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_worker import TaskWorker
from rag_kb.application.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, ToolRegistryError
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentTaskRow,
    ApprovalRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PlanRevisionRow,
)
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
    TaskRepositoryError,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore

_REPORT_ARGUMENTS: dict[str, object] = {
    "title": "差旅制度摘要报告",
    "content": "基于证据的报告正文",
    "format": "markdown",
}


class CountingWriteAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _arguments, _context):
        self.calls += 1
        return {
            "artifact_type": "report",
            "storage_key": "server/generated",
            "content_hash": "sha256:" + "a" * 64,
            "status": "ready",
        }

    def verify(self, _result, _context) -> bool:
        return True


class ScriptedAgent:
    def __init__(self, snapshots: tuple[object, ...]) -> None:
        self._snapshots = iter(snapshots)

    def get_state(self, _config: dict[str, object]) -> object:
        return next(self._snapshots)

    def invoke(self, _input: object, _config: dict[str, object], *, context: object) -> dict[str, object]:
        return {}


class AuthContexts:
    def __init__(self, auth: AuthContext) -> None:
        self._auth = auth

    def resolve(self, **_kwargs: object) -> AuthContext:
        return self._auth


def _empty_snapshot() -> SimpleNamespace:
    return SimpleNamespace(values={}, interrupts=(), next=())


def _hitl_snapshot(
    *,
    tool_name: str = "create_report_artifact",
    arguments: dict[str, object] | None = None,
    action_requests: list | None = None,
    review_configs: list | None = None,
    interrupt_id: str = "interrupt-1",
    checkpoint_id: str = "checkpoint-1",
    thread_id: str = "task-1",
    tool_call_id: str = "call_1",
    values_messages: list | None = None,
) -> SimpleNamespace:
    args = dict(_REPORT_ARGUMENTS) if arguments is None else arguments
    if action_requests is None:
        action_requests = [{"name": tool_name, "args": args, "description": "pending"}]
    if review_configs is None:
        review_configs = [
            {"action_name": tool_name, "allowed_decisions": ["approve", "reject"]}
        ]
    if values_messages is None:
        values_messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": args, "id": tool_call_id, "type": "tool_call"}
                ],
            )
        ]
    interrupt = SimpleNamespace(
        value={"action_requests": action_requests, "review_configs": review_configs},
        id=interrupt_id,
    )
    return SimpleNamespace(
        values={"messages": values_messages},
        interrupts=(interrupt,),
        next=("tools",),
        config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        },
    )


def _clarification_snapshot(question: str = "请确认报告格式") -> SimpleNamespace:
    interrupt = SimpleNamespace(
        value={"kind": "user_clarification", "question": question}, id="interrupt-q"
    )
    return SimpleNamespace(
        values={},
        interrupts=(interrupt,),
        next=("model",),
        config={
            "configurable": {
                "thread_id": "task-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-q",
            }
        },
    )


def _build_stage(tmp_path: Path, *, task_mode: TaskMode) -> SimpleNamespace:
    database = build_database(
        f"sqlite+pysqlite:///{(tmp_path / 'approval.sqlite3').as_posix()}"
    )
    Base.metadata.create_all(database.engine)
    requester = AuthContext(
        tenant_id="tenant-a",
        user_id="requester",
        attributes={
            "permissions": ["artifact:write", "knowledge:read", "document:read"]
        },
        request_id="request-1",
    )
    with database.engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants VALUES ('tenant-a', 'Tenant A', 1, CURRENT_TIMESTAMP)")
        )
        connection.execute(
            text(
                "INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) "
                "VALUES ('tenant-a', 'requester', '[]', :attributes, 1, CURRENT_TIMESTAMP)"
            ),
            {"attributes": '{"permissions":["artifact:write","knowledge:read","document:read"]}'},
        )
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=requester,
        goal="创建制度摘要报告",
        task_mode=task_mode,
        task_id="task-1",
    )
    with UnitOfWork(database) as work:
        assert work.session is not None
        task = work.session.scalar(
            select(AgentTaskRow).where(AgentTaskRow.task_id == "task-1")
        )
        assert task is not None
        task.status = "running"
        task.lease_owner = "worker-1"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=60)
        task.lease_generation = 1
        work.session.add_all(
            [
                DocumentRow(
                    tenant_id="tenant-a",
                    document_id="doc-1",
                    title="差旅制度",
                    department="安全",
                    status="active",
                    created_by="requester",
                    projection_generation=1,
                    applied_projection_generation=1,
                    applied_projection_version_id="version-1",
                ),
                ACLGrantRow(
                    grant_id="grant-1",
                    tenant_id="tenant-a",
                    resource_type="document",
                    resource_id="doc-1",
                    subject_type="user",
                    subject_id="requester",
                    permissions=["knowledge:read", "document:read"],
                ),
            ]
        )
        work.session.flush()
        work.session.add(
            DocumentVersionRow(
                version_id="version-1",
                tenant_id="tenant-a",
                document_id="doc-1",
                version=1,
                lifecycle_status="active",
                source_object_key="doc-1",
                source_file_name="doc.md",
                content_hash="sha256:content",
                index_version="kb-v1",
                desired_projection_generation=1,
                applied_projection_generation=1,
            )
        )
        work.session.flush()
        work.session.add_all(
            [
                ChunkMetadataRow(
                    chunk_id="chunk-1",
                    version_id="version-1",
                    tenant_id="tenant-a",
                    document_id="doc-1",
                    document_version=1,
                    content_hash="sha256:chunk",
                    source_locator="doc/1",
                    source_file_name="doc.md",
                    index_version="kb-v1",
                ),
                IngestionJobRow(
                    job_id="job-1",
                    tenant_id="tenant-a",
                    version_id="version-1",
                    status="succeeded",
                    indexed_chunk_count=1,
                    index_manifest_digest="sha256:manifest",
                    index_completed_at=datetime.now(UTC),
                    projection_generation=1,
                    projection_lease_generation=1,
                ),
            ]
        )
        work.session.flush()
        work.session.add(
            EvidenceRecordRow(
                evidence_id="evidence-1",
                task_id="task-1",
                query_fingerprint="sha256:query",
                tenant_id="tenant-a",
                document_id="doc-1",
                version_id="version-1",
                document_version=1,
                chunk_id="chunk-1",
                content_hash="sha256:chunk",
                source_locator="doc/1",
                permission_granted=True,
                supported_claim_ids=[],
            )
        )
    report_adapter = CountingWriteAdapter()
    draft_adapter = CountingWriteAdapter()
    registry = ToolRegistry(
        adapters={
            "create_report_artifact": report_adapter,
            "create_work_item_draft": draft_adapter,
        }
    )
    approvals = ApprovalService(database, registry=registry)
    tools = ToolExecutionService(
        database,
        registry,
        approvals,
        tool_results=LocalObjectStore(tmp_path / "tool-results"),
    )
    lease = TaskLease(
        repository.get_task(auth=requester, task_id="task-1"),
        "worker-1",
        1,
        TaskBudgets(),
    )
    return SimpleNamespace(
        database=database,
        repository=repository,
        requester=requester,
        report_adapter=report_adapter,
        draft_adapter=draft_adapter,
        approvals=approvals,
        tools=tools,
        lease=lease,
        tmp_path=tmp_path,
    )


@pytest.fixture()
def stage(tmp_path: Path) -> SimpleNamespace:
    stage = _build_stage(tmp_path, task_mode=TaskMode.CONTROLLED_ACTION)
    yield stage
    stage.tools.close()
    stage.database.engine.dispose()


def _worker(stage: SimpleNamespace, agent: ScriptedAgent) -> TaskWorker:
    return TaskWorker(
        repository=stage.repository,
        auth_contexts=AuthContexts(stage.requester),
        agent_for=lambda _budgets: agent,
        controlled_agent_for=lambda _budgets: agent,
        results=AgentResultStore(LocalObjectStore(stage.tmp_path / "objects")),
        evidence=EvidenceLedgerRepository(),
        tools=stage.tools,
        database=stage.database,
        worker_id="worker-1",
        approvals=stage.approvals,
    )


def _approval_rows(stage: SimpleNamespace) -> list[ApprovalRow]:
    with stage.database.sessions() as session:
        return list(
            session.scalars(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == "tenant-a",
                    ApprovalRow.task_id == "task-1",
                )
            )
        )


def _plan_rows(stage: SimpleNamespace) -> list[PlanRevisionRow]:
    with stage.database.sessions() as session:
        return list(
            session.scalars(
                select(PlanRevisionRow).where(
                    PlanRevisionRow.tenant_id == "tenant-a",
                    PlanRevisionRow.task_id == "task-1",
                )
            )
        )


def _assert_no_side_effects(stage: SimpleNamespace) -> None:
    assert _approval_rows(stage) == []
    assert _plan_rows(stage) == []
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_hitl_interrupt_pauses_for_approval_without_execution(stage) -> None:
    agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))

    outcome = _worker(stage, agent)._run_claimed(stage.lease)

    assert outcome.status == "waiting_for_approval"
    assert outcome.task_id == "task-1"
    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_APPROVAL.value
        assert task.interrupt_kind == "tool_approval"
        assert task.interrupt_summary
        assert task.interrupt_ref
        assert task.lease_owner is None
    approvals = _approval_rows(stage)
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval.status == "pending"
    assert approval.thread_id == "task-1"
    assert approval.checkpoint_ns == ""
    assert approval.checkpoint_id == "checkpoint-1"
    assert approval.interrupt_id == "interrupt-1"
    assert approval.tool_call_id == "call_1"
    assert approval.checkpoint_binding_hash
    assert approval.tool_name == "create_report_artifact"
    # 版本由服务端从 ToolRegistry 派生，不接受模型或调用方自报版本。
    assert approval.tool_version == "1.0.0"
    assert approval.arguments_hash
    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.interrupt_ref == approval.approval_id
    plans = _plan_rows(stage)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.revision >= 1
    assert set(plan.plan) == {
        "tool_name",
        "tool_version",
        "arguments_hash",
        "evidence_ids",
        "checkpoint_binding_hash",
    }
    assert plan.plan["evidence_ids"] == ["evidence-1"]
    # 原始参数与 HITL payload 不进入任何业务表，只保存 hash。
    assert "title" not in plan.plan
    assert "正文" not in (task.interrupt_summary or "")
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_resynced_checkpoint_reuses_single_active_approval(stage) -> None:
    first_agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
    first = _worker(stage, first_agent)._run_claimed(stage.lease)
    assert first.status == "waiting_for_approval"
    approval_id = _approval_rows(stage)[0].approval_id

    # 模拟 D6 一致性窗口：approval 已写、业务摘要同步丢失（任务回退 running、新租约），
    # 新 Worker 从同一 checkpoint 读取同一 interrupt 重放同步。
    with UnitOfWork(stage.database) as work:
        assert work.session is not None
        task = work.session.scalar(
            select(AgentTaskRow).where(AgentTaskRow.task_id == "task-1")
        )
        assert task is not None
        task.status = "running"
        task.interrupt_kind = None
        task.interrupt_summary = None
        task.interrupt_ref = None
        task.lease_owner = "worker-1"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=60)
        task.lease_generation = 2
    second_lease = TaskLease(
        stage.repository.get_task(auth=stage.requester, task_id="task-1"),
        "worker-1",
        2,
        TaskBudgets(),
    )
    second_agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
    second = _worker(stage, second_agent)._run_claimed(second_lease)

    assert second.status == "waiting_for_approval"
    approvals = _approval_rows(stage)
    assert len(approvals) == 1
    assert approvals[0].approval_id == approval_id
    assert approvals[0].status == "pending"
    assert len(_plan_rows(stage)) == 1
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_clarification_interrupt_still_routes_to_waiting_for_user(stage) -> None:
    agent = ScriptedAgent((_empty_snapshot(), _clarification_snapshot()))

    outcome = _worker(stage, agent)._run_claimed(stage.lease)

    assert outcome.status == "waiting_for_user"
    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_USER.value
        assert task.interrupt_kind == "user_clarification"
        assert task.interrupt_ref is None
    _assert_no_side_effects(stage)


def test_approval_interrupt_on_research_task_fails_closed(tmp_path: Path) -> None:
    stage = _build_stage(tmp_path, task_mode=TaskMode.RESEARCH)
    try:
        agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
        outcome = _worker(stage, agent)._run_claimed(stage.lease)
        assert outcome.status == "failed"
        with stage.database.sessions() as session:
            task = session.get(AgentTaskRow, "task-1")
            assert task is not None
            assert task.status == TaskStatus.FAILED.value
        _assert_no_side_effects(stage)
    finally:
        stage.tools.close()
        stage.database.engine.dispose()


def test_unknown_interrupt_kind_fails_closed(stage) -> None:
    interrupt = SimpleNamespace(
        value={"kind": "tool_approval", "summary": "非法形状"}, id="interrupt-x"
    )
    snapshot = SimpleNamespace(
        values={},
        interrupts=(interrupt,),
        next=(),
        config={"configurable": {"thread_id": "task-1", "checkpoint_id": "checkpoint-x"}},
    )
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.FAILED.value
    _assert_no_side_effects(stage)


def test_multiple_interrupts_fail_closed(stage) -> None:
    base = _hitl_snapshot()
    snapshot = SimpleNamespace(
        values=base.values,
        interrupts=(base.interrupts[0], base.interrupts[0]),
        next=base.next,
        config=base.config,
    )
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_multiple_action_requests_fail_closed(stage) -> None:
    snapshot = _hitl_snapshot(
        action_requests=[
            {
                "name": "create_report_artifact",
                "args": dict(_REPORT_ARGUMENTS),
                "description": "pending",
            },
            {
                "name": "create_work_item_draft",
                "args": {"title": "草稿", "description": "正文"},
                "description": "pending",
            },
        ],
        review_configs=[
            {"action_name": "create_report_artifact", "allowed_decisions": ["approve", "reject"]},
            {"action_name": "create_work_item_draft", "allowed_decisions": ["approve", "reject"]},
        ],
    )
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_unknown_tool_in_action_request_fails_closed(stage) -> None:
    snapshot = _hitl_snapshot(tool_name="arbitrary_shell")
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_review_config_with_open_decisions_fails_closed(stage) -> None:
    snapshot = _hitl_snapshot(
        review_configs=[
            {
                "action_name": "create_report_artifact",
                "allowed_decisions": ["approve", "reject", "edit"],
            }
        ]
    )
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_review_config_name_mismatch_fails_closed(stage) -> None:
    snapshot = _hitl_snapshot(
        review_configs=[
            {"action_name": "create_work_item_draft", "allowed_decisions": ["approve", "reject"]}
        ]
    )
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_action_args_not_matching_checkpoint_tool_call_fails_closed(stage) -> None:
    tampered = dict(_REPORT_ARGUMENTS)
    tampered["title"] = "被篡改的标题"
    # 只篡改 HITL payload 的 args；checkpoint 内 AI tool call 保持原始参数。
    snapshot = _hitl_snapshot(arguments=tampered, values_messages=[
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_report_artifact",
                    "args": dict(_REPORT_ARGUMENTS),
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
    ])
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_interrupt_payload_with_unknown_fields_fails_closed(stage) -> None:
    base = _hitl_snapshot()
    interrupt = SimpleNamespace(
        value={
            "action_requests": base.interrupts[0].value["action_requests"],
            "review_configs": base.interrupts[0].value["review_configs"],
            "approval_id": "forged-approval",
        },
        id="interrupt-1",
    )
    snapshot = SimpleNamespace(
        values=base.values,
        interrupts=(interrupt,),
        next=base.next,
        config=base.config,
    )
    outcome = _worker(stage, ScriptedAgent((_empty_snapshot(), snapshot)))._run_claimed(
        stage.lease
    )
    assert outcome.status == "failed"
    _assert_no_side_effects(stage)


def test_waiting_for_approval_rejects_user_continue(stage) -> None:
    agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
    worker = _worker(stage, agent)
    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    with stage.database.sessions() as session:
        state_version = (
            session.get(AgentTaskRow, "task-1").state_version
        )

    with pytest.raises(TaskRepositoryError):
        worker.resume_for_user(
            auth=stage.requester,
            task_id="task-1",
            expected_state_version=state_version,
            answer="直接放行",
        )
    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_APPROVAL.value
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_pending_approval_and_released_lease_never_execute_adapter(stage) -> None:
    agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
    outcome = _worker(stage, agent)._run_claimed(stage.lease)
    assert outcome.status == "waiting_for_approval"
    approvals = _approval_rows(stage)
    assert len(approvals) == 1
    plan = _plan_rows(stage)[0]

    # HITL 从未恢复：即使把 pending approval 直接塞进执行请求，
    # 租约已释放 + approval 未批准，执行必须在 adapter 前被拒绝。
    request = ToolExecutionRequest(
        auth=stage.requester,
        task_id="task-1",
        tool_name="create_report_artifact",
        arguments=dict(_REPORT_ARGUMENTS),
        worker_id="worker-1",
        lease_generation=1,
        idempotency_key="key-1",
        approval_id=approvals[0].approval_id,
        plan_revision=plan.revision,
        plan_digest=plan.plan_digest,
        evidence_digest=plan.evidence_digest,
        evidence_ids=("evidence-1",),
    )
    with pytest.raises(ToolExecutionError):
        stage.tools.execute(request)
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_write_adapter_without_approval_fails_closed(stage) -> None:
    # HITL 未恢复（无审批事实）：WRITE 请求在授权门前被拒绝，业务 adapter 为 0。
    request = ToolExecutionRequest(
        auth=stage.requester,
        task_id="task-1",
        tool_name="create_report_artifact",
        arguments=dict(_REPORT_ARGUMENTS),
        worker_id="worker-1",
        lease_generation=1,
        idempotency_key="key-1",
    )
    with pytest.raises((ToolExecutionError, ToolRegistryError)) as exc_info:
        stage.tools.execute(request)
    assert getattr(exc_info.value, "code", None) == "approval_required"
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0
