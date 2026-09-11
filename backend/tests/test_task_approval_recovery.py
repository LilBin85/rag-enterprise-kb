"""M2 批次 1：HITL 暂停、interrupt 严格分流与审批事实故障注入。

双层审批门（批次 1 切面）：HITL 未恢复或业务审批未创建时，两个 WRITE
业务 adapter 调用次数必须为 0；clarification 与 approval 两类 interrupt
互斥分流，错误形状一律 fail-closed。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import select, text

from rag_kb.agent.result import AgentResultStore, TaskResultCandidate
from rag_kb.application.approval_service import ApprovalService, ApprovalServiceError
from rag_kb.application.ports.task_worker import TaskWorkerPersistenceError
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
from rag_kb.infrastructure.database.approval_lifecycle import (
    DatabaseApprovalLifecycle,
)
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentCapabilityLeaseRow,
    AgentTaskRow,
    ApprovalRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PlanRevisionRow,
    PrincipalRow,
    TaskResumeCommandRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_execution import DatabaseTaskWorkerSettlement
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
    TaskRepositoryError,
)
from rag_kb.infrastructure.database.tool_execution_lifecycle import (
    DatabaseToolExecutionLifecycle,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.observability import Observability, current_trace_headers

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
            "artifact_id": "artifact-1",
            "content_hash": "sha256:" + "a" * 64,
            "status": "ready",
        }

    def verify(self, _result, _context) -> bool:
        return True


class ScriptedAgent:
    def __init__(self, snapshots: tuple[object, ...]) -> None:
        self._snapshots = iter(snapshots)
        self.inputs: list[object] = []
        self.contexts: list[object] = []

    def get_state(self, _config: dict[str, object]) -> object:
        return next(self._snapshots)

    def invoke(self, _input: object, _config: dict[str, object], *, context: object) -> dict[str, object]:
        self.inputs.append(_input)
        self.contexts.append(context)
        return {}


class _CheckpointMarkingAgent:
    """测试替身：模拟真实 fenced saver 在 ToolNode 前的首个 resume checkpoint。"""

    def __init__(self, stage: SimpleNamespace, agent: ScriptedAgent) -> None:
        self._stage = stage
        self._agent = agent

    def get_state(self, config: dict[str, object]) -> object:
        return self._agent.get_state(config)

    def invoke(self, input_value: object, config: dict[str, object], *, context: object) -> dict[str, object]:
        _simulate_resume_checkpoint_marker(self._stage, config)
        return self._agent.invoke(input_value, config, context=context)


class ExecutorAgent(ScriptedAgent):
    """恢复路径契约代理：invoke 用服务端 context 驱动真实 ToolExecutionService，
    模拟 ToolNode 在 approve 后执行 WRITE 的行为（真图执行由 PG 恢复测试证明）。"""

    def __init__(self, stage: SimpleNamespace, snapshots: tuple[object, ...]) -> None:
        super().__init__(snapshots)
        self._stage = stage

    def invoke(self, _input: object, _config: dict[str, object], *, context: object) -> dict[str, object]:
        self.inputs.append(_input)
        self.contexts.append(context)
        request = ToolExecutionRequest(
            auth=context.auth,
            task_id=context.task_id,
            tool_name="create_report_artifact",
            arguments=dict(_REPORT_ARGUMENTS),
            worker_id=context.worker_id,
            lease_generation=context.lease_generation,
            idempotency_key=context.idempotency_key,
            approval_id=context.approval_id,
            plan_revision=context.plan_revision,
            plan_digest=context.plan_digest,
            evidence_digest=context.evidence_digest,
            evidence_ids=context.evidence_ids,
        )
        self._stage.tools.execute(request)
        return {}


class CrashingInvokeAgent(ScriptedAgent):
    """模拟 resume 提交后在 ToolNode 前崩溃的 Worker。"""

    def invoke(self, _input: object, _config: dict[str, object], *, context: object) -> dict[str, object]:
        self.inputs.append(_input)
        self.contexts.append(context)
        raise RuntimeError("injected worker crash before tool node")


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
    approver = AuthContext(
        tenant_id="tenant-a",
        user_id="approver",
        attributes={"permissions": ["task:approve"]},
        request_id="approve-1",
    )
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (tenant_id, name, active, created_at) "
                "VALUES ('tenant-a', 'Tenant A', 1, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) "
                "VALUES ('tenant-a', 'requester', '[]', :attributes, 1, CURRENT_TIMESTAMP)"
            ),
            {"attributes": '{"permissions":["artifact:write","knowledge:read","document:read"]}'},
        )
        connection.execute(
            text(
                "INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) "
                "VALUES ('tenant-a', 'approver', '[]', :attributes, 1, CURRENT_TIMESTAMP)"
            ),
            {"attributes": '{"permissions":["task:approve"]}'},
        )
    repository = DatabaseTaskRepository(
        database,
        task_capability_name="task-agent",
        task_capability_protocol_version="m2-fenced-v1",
    ) if task_mode is TaskMode.CONTROLLED_ACTION else DatabaseTaskRepository(database)
    if task_mode is TaskMode.CONTROLLED_ACTION:
        with database.sessions.begin() as session:
            # controlled 创建门要求兼容 worker 已在线。
            session.add(
                AgentCapabilityLeaseRow(
                    capability_name="task-agent",
                    protocol_version="m2-fenced-v1",
                    worker_id="worker-1",
                    lease_expires_at=datetime.now(UTC) + timedelta(seconds=300),
                    updated_at=datetime.now(UTC),
                )
            )
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
    approvals = ApprovalService(
        DatabaseApprovalLifecycle(database, resume_repository=repository),
        registry=registry,
    )
    tools = ToolExecutionService(
        DatabaseToolExecutionLifecycle(database),
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
        approver=approver,
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


class CrashingWorker(TaskWorker):
    """模拟 resume 提交后进程死亡：异常直接外抛，不收敛业务状态。"""

    def _run_claimed(self, lease: TaskLease, **kwargs: object) -> TaskWorkerResult:
        return self._advance(lease, **kwargs)


def _simulate_resume_checkpoint_marker(stage: SimpleNamespace, config: dict[str, object]) -> None:
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return
    command_id = configurable.get("checkpoint_resume_command_id")
    command_generation = configurable.get("checkpoint_resume_command_lease_generation")
    if command_id is None and command_generation is None:
        return
    assert isinstance(command_id, str)
    assert isinstance(command_generation, int)
    worker_id = configurable.get("checkpoint_worker_id")
    assert isinstance(worker_id, str)
    with stage.database.sessions.begin() as session:
        command = session.get(TaskResumeCommandRow, command_id)
        assert command is not None
        task = session.get(AgentTaskRow, command.task_id)
        assert task is not None
        now = datetime.now(UTC)
        command.applied_checkpoint_id = "resume-checkpoint-1"
        command.applied_at = now
        command.updated_at = now
        task.active_resume_command_id = None
        task.updated_at = now


def _settlement(stage: SimpleNamespace) -> DatabaseTaskWorkerSettlement:
    return DatabaseTaskWorkerSettlement(
        stage.database,
        LocalObjectStore(stage.tmp_path / "objects"),
    )


def _worker(stage: SimpleNamespace, agent: ScriptedAgent) -> TaskWorker:
    fenced_agent = _CheckpointMarkingAgent(stage, agent)
    return TaskWorker(
        repository=stage.repository,
        auth_contexts=AuthContexts(stage.requester),
        agent_for=lambda _budgets: fenced_agent,
        controlled_agent_for=lambda _budgets: fenced_agent,
        results=AgentResultStore(LocalObjectStore(stage.tmp_path / "objects")),
        settlement=_settlement(stage),
        tools=stage.tools,
        worker_id="worker-1",
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
        approvals=stage.approvals,
    )


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
        task.interrupt_checkpoint_ns = None
        task.interrupt_checkpoint_id = None
        task.interrupt_id = None
        task.interrupt_approval_id = None
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
        stage.repository.enqueue_user_resume_command(
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


# ---------------------------------------------------------------------------
# 批次 2：四格真值表与受控恢复链路（先红后绿）
# ---------------------------------------------------------------------------


def _final_snapshot(candidate: TaskResultCandidate) -> SimpleNamespace:
    return SimpleNamespace(
        values={"structured_response": candidate, "messages": []},
        interrupts=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "task-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-final",
            }
        },
    )


def _succeeded_candidate() -> TaskResultCandidate:
    return TaskResultCandidate(
        status="succeeded",
        answer="报告已完成",
        evidence_ids=("evidence-1",),
        stop_reason="completed",
    )


def _paused_stage(stage: SimpleNamespace) -> tuple[str, int]:
    agent = ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
    worker = _worker(stage, agent)
    outcome = worker._run_claimed(stage.lease)
    assert outcome.status == "waiting_for_approval"
    approval_id = _approval_rows(stage)[0].approval_id
    with stage.database.sessions() as session:
        state_version = session.get(AgentTaskRow, "task-1").state_version
    return approval_id, state_version


def _task_row(stage: SimpleNamespace) -> AgentTaskRow:
    with stage.database.sessions() as session:
        return session.get(AgentTaskRow, "task-1")


def test_truth_table_hitl_no_business_no_executes_zero(stage) -> None:
    """HITL=否/业务=否：无暂停、无审批，恢复入口 fail-closed，adapter 为 0。"""
    worker = _worker(stage, ScriptedAgent(()))
    state_version = _task_row(stage).state_version

    with pytest.raises(ApprovalServiceError) as exc_info:
        stage.approvals.decide_and_enqueue(
            auth=stage.approver,
            approval_id="missing-approval",
            expected_task_state_version=state_version,
            decision="approve",
            reason=None,
        )
    assert exc_info.value.code == "approval_not_found"
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_truth_table_hitl_yes_business_no_executes_zero(stage) -> None:
    """HITL=是/业务=否：审批仍 pending（自批被拒），恢复必须中止，adapter 为 0。"""
    approval_id, state_version = _paused_stage(stage)
    with stage.database.sessions() as session:
        approval = session.get(ApprovalRow, approval_id)
        assert approval is not None and approval.status == "pending"

    # 发起人自批：授予 task:approve 以穿过权限门，命中自批拒绝。
    elevated_requester = AuthContext(
        tenant_id="tenant-a",
        user_id="requester",
        attributes={
            "permissions": [
                "artifact:write",
                "knowledge:read",
                "document:read",
                "task:approve",
            ]
        },
        request_id="self-approve-1",
    )
    with pytest.raises(ApprovalServiceError) as exc_info:
        stage.approvals.decide_and_enqueue(
            auth=elevated_requester,
            approval_id=approval_id,
            expected_task_state_version=state_version,
            decision="approve",
            reason=None,
        )
    assert exc_info.value.code == "approval_self_rejected"
    with stage.database.sessions() as session:
        approval = session.get(ApprovalRow, approval_id)
        assert approval is not None and approval.status == "pending"
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_APPROVAL.value
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_truth_table_hitl_no_business_yes_executes_zero(stage) -> None:
    """HITL=否/业务=是：审批已批准但服务端从未驱动 resume，adapter 为 0。"""
    approval_id, _state_version = _paused_stage(stage)
    command = stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=_state_version,
        decision="approve",
        reason=None,
    )
    assert command.command_id

    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_APPROVAL.value
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_truth_table_both_yes_executes_exactly_once(stage) -> None:
    """HITL=是/业务=是且绑定有效：恰好执行 1 次，终态由服务端受控构造。"""
    approval_id, state_version = _paused_stage(stage)
    stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )

    resume_agent = ExecutorAgent(
        stage, (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
    )
    outcome = _worker(stage, resume_agent).run_once()

    assert outcome.status == "finished"
    assert stage.report_adapter.calls == 1
    assert stage.draft_adapter.calls == 0
    with stage.database.sessions() as session:
        approval = session.get(ApprovalRow, approval_id)
        assert approval is not None and approval.status == "executed"
        task = session.get(AgentTaskRow, "task-1")
        assert task is not None
        assert task.status == TaskStatus.SUCCEEDED.value
        assert task.result_kind == "controlled_action"
        tool_calls = (
            session.scalars(
                select(ToolCallRow).where(ToolCallRow.task_id == "task-1")
            ).all()
        )
        assert len(tool_calls) == 1


def test_repeat_decision_does_not_reexecute_after_consumption(stage) -> None:
    """重复 approve decision：approval 已 executed，重复恢复不产生第二次副作用。"""
    approval_id, state_version = _paused_stage(stage)
    command = stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )
    resume_agent = ExecutorAgent(
        stage, (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
    )
    first = _worker(stage, resume_agent).run_once()
    assert first.status == "finished"
    assert stage.report_adapter.calls == 1

    second_agent = ExecutorAgent(stage, ())
    repeated = stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )
    assert repeated.command_id == command.command_id
    assert _worker(stage, second_agent).run_once().status == "idle"
    assert stage.report_adapter.calls == 1
    assert stage.draft_adapter.calls == 0


def test_queued_approval_command_fails_closed_after_approver_revocation(stage) -> None:
    """审批命令入队后撤权，Worker 不得注入 resume 或调用 WRITE adapter。"""
    approval_id, state_version = _paused_stage(stage)
    command = stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )
    with stage.database.sessions.begin() as session:
        approver = session.get(PrincipalRow, ("tenant-a", "approver"))
        assert approver is not None
        approver.active = False

    resume_agent = ExecutorAgent(
        stage, (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
    )
    outcome = _worker(stage, resume_agent).run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "task_resume_actor_unavailable"
    assert resume_agent.inputs == []
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0
    with stage.database.sessions() as session:
        stored = session.get(TaskResumeCommandRow, command.command_id)
        tool_calls = tuple(
            session.scalars(select(ToolCallRow).where(ToolCallRow.task_id == "task-1"))
        )
    assert stored is not None
    assert stored.status == "failed"
    assert stored.payload_body is None
    assert tool_calls == ()


def test_reject_decision_resumes_thread_with_fixed_payload_and_zero_execution(
    stage,
) -> None:
    """reject decision：固定服务端 reject payload 同 thread 恢复，adapter 为 0。"""
    approval_id, state_version = _paused_stage(stage)
    resume_agent = ScriptedAgent(
        (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
    )
    stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="reject",
        reason=None,
    )
    outcome = _worker(stage, resume_agent).run_once()

    assert outcome.status == "finished"
    command = resume_agent.inputs[0]
    resume = command.resume
    assert set(resume) == {"decisions"}
    assert len(resume["decisions"]) == 1
    decision = resume["decisions"][0]
    assert decision["type"] == "reject"
    assert isinstance(decision.get("message"), str) and decision["message"].strip()
    with stage.database.sessions() as session:
        approval = session.get(ApprovalRow, approval_id)
        assert approval is not None and approval.status == "rejected"
        assert approval.rejected_by == "approver"
        assert approval.rejected_at is not None
        assert approval.rejection_reason
    assert stage.report_adapter.calls == 0
    assert stage.draft_adapter.calls == 0


def test_approval_transition_and_resume_are_safe_and_measured(stage) -> None:
    observability, exporter = _telemetry()
    secret = "Bearer fake-token approval reason tool payload C:/private/object"
    try:
        with observability.activate():
            approval_id, state_version = _paused_stage(stage)
            stage.approvals.decide_and_enqueue(
                auth=stage.approver,
                approval_id=approval_id,
                expected_task_state_version=state_version,
                decision="reject",
                reason=secret,
            )
            result = _worker(
                stage,
                ScriptedAgent((_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))),
            ).run_once()
        payload = observability.metrics_payload().decode()
        spans = exporter.get_finished_spans()
        rendered_spans = repr(spans)
    finally:
        observability.shutdown()

    assert result.status == "finished"
    assert any(span.name == "approval.transition" for span in spans)
    assert any(span.name == "task.run" for span in spans)
    assert 'decision="requested",outcome="succeeded"' in payload
    assert 'decision="rejected",outcome="succeeded"' in payload
    assert secret not in payload
    assert secret not in rendered_spans


def test_initial_and_approval_resumed_runs_link_their_producers_as_new_roots(stage) -> None:
    observability, exporter = _telemetry()
    try:
        with observability.activate():
            with observability.span("task.create.request") as create_span:
                traceparent, tracestate = current_trace_headers()
                create_context = create_span.get_span_context()
            initial_lease = TaskLease(
                stage.lease.state,
                stage.lease.worker_id,
                stage.lease.lease_generation,
                stage.lease.budgets,
                traceparent,
                tracestate,
            )
            first = _worker(
                stage, ScriptedAgent((_empty_snapshot(), _hitl_snapshot()))
            )._run_claimed(initial_lease)
            approval_id = _approval_rows(stage)[0].approval_id
            with stage.database.sessions() as session:
                state_version = session.get(AgentTaskRow, "task-1").state_version

            with observability.span("approval.request") as approval_span:
                approval_context = approval_span.get_span_context()
                stage.approvals.decide_and_enqueue(
                    auth=stage.approver,
                    approval_id=approval_id,
                    expected_task_state_version=state_version,
                    decision="reject",
                    reason=None,
                )
                second = _worker(
                    stage,
                    ScriptedAgent(
                        (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
                    ),
                ).run_once()
        run_spans = [span for span in exporter.get_finished_spans() if span.name == "task.run"]
        transition_span = next(
            span
            for span in exporter.get_finished_spans()
            if span.name == "approval.transition"
            and span.attributes.get("decision") == "rejected"
        )
    finally:
        observability.shutdown()

    assert first.status == "waiting_for_approval"
    assert second.status == "finished"
    assert len(run_spans) == 2
    assert all(span.parent is None or not span.parent.is_valid for span in run_spans)
    assert len(run_spans[0].links) == 1
    # A4 只耐久化审批决定与恢复命令；后台消费是独立的新 root。
    assert len(run_spans[1].links) == 0
    assert run_spans[0].links[0].context.trace_id == create_context.trace_id
    assert run_spans[0].links[0].context.span_id == create_context.span_id
    assert transition_span.context.trace_id == approval_context.trace_id
    assert transition_span.parent is not None
    assert transition_span.parent.span_id == approval_context.span_id
    assert run_spans[1].context.trace_id != transition_span.context.trace_id


def test_resume_crash_before_checkpoint_reclaims_the_same_command_once(stage) -> None:
    """首个 checkpoint 前崩溃只能双 lease 重领同一命令，不能降级为普通 task。"""
    approval_id, state_version = _paused_stage(stage)
    command = stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )

    crashing = CrashingInvokeAgent((_hitl_snapshot(),))
    with pytest.raises(RuntimeError):
        CrashingWorker(
            repository=stage.repository,
            auth_contexts=AuthContexts(stage.requester),
            agent_for=lambda _budgets: crashing,
            controlled_agent_for=lambda _budgets: crashing,
            results=AgentResultStore(LocalObjectStore(stage.tmp_path / "objects")),
            settlement=_settlement(stage),
            tools=stage.tools,
            worker_id="worker-1",
            capability_name="task-agent",
            protocol_version="m2-fenced-v1",
            approvals=stage.approvals,
        ).run_once()

    # 不得单边释放 task lease：active command 仍保护正文。两边同时过期后
    # 只能由 resume command 路径重领，task state_version 不会再次递增。
    with UnitOfWork(stage.database) as work:
        assert work.session is not None
        task = work.session.scalar(
            select(AgentTaskRow).where(AgentTaskRow.task_id == "task-1")
        )
        command = work.session.scalar(
            select(TaskResumeCommandRow).where(
                TaskResumeCommandRow.tenant_id == "tenant-a",
                TaskResumeCommandRow.task_id == "task-1",
            )
        )
        assert task is not None
        assert command is not None
        expired = datetime.now(UTC) - timedelta(seconds=1)
        task.lease_expires_at = expired
        command.lease_expires_at = expired

    recovered_agent = ExecutorAgent(
        stage, (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
    )
    outcome = _worker(stage, recovered_agent).run_once()
    assert outcome.status == "finished"
    assert stage.report_adapter.calls == 1
    assert stage.draft_adapter.calls == 0
    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        command = session.scalar(
            select(TaskResumeCommandRow).where(
                TaskResumeCommandRow.tenant_id == "tenant-a",
                TaskResumeCommandRow.task_id == "task-1",
            )
        )
        assert task is not None and task.active_resume_command_id is None
        assert command is not None and command.status == "consumed"
        assert command.payload_body is None


def test_resume_crash_after_checkpoint_settles_without_reinjecting(stage) -> None:
    """checkpoint 已提交、consume 前崩溃时仅清正文结算，绝不再调用 Agent。"""
    approval_id, state_version = _paused_stage(stage)
    stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )

    crashing = CrashingInvokeAgent((_hitl_snapshot(),))
    fenced_crashing = _CheckpointMarkingAgent(stage, crashing)
    with pytest.raises(RuntimeError):
        CrashingWorker(
            repository=stage.repository,
            auth_contexts=AuthContexts(stage.requester),
            agent_for=lambda _budgets: fenced_crashing,
            controlled_agent_for=lambda _budgets: fenced_crashing,
            results=AgentResultStore(LocalObjectStore(stage.tmp_path / "objects")),
            settlement=_settlement(stage),
            tools=stage.tools,
            worker_id="worker-1",
            capability_name="task-agent",
            protocol_version="m2-fenced-v1",
            approvals=stage.approvals,
        ).run_once()

    with stage.database.sessions() as session:
        task = session.get(AgentTaskRow, "task-1")
        command = session.scalar(
            select(TaskResumeCommandRow).where(
                TaskResumeCommandRow.tenant_id == "tenant-a",
                TaskResumeCommandRow.task_id == "task-1",
            )
        )
        assert task is not None and task.active_resume_command_id is None
        assert command is not None and command.status == "claimed"
        assert command.applied_checkpoint_id == "resume-checkpoint-1"

    idle_agent = ScriptedAgent(())
    outcome = _worker(stage, idle_agent).run_once()
    assert outcome.status == "idle"
    assert idle_agent.inputs == []
    with stage.database.sessions() as session:
        command = session.scalar(
            select(TaskResumeCommandRow).where(
                TaskResumeCommandRow.tenant_id == "tenant-a",
                TaskResumeCommandRow.task_id == "task-1",
            )
        )
        assert command is not None and command.status == "consumed"
        assert command.payload_body is None


def test_approval_resume_context_carries_stable_action_key(stage) -> None:
    """恢复路径构造的 AgentRunContext 必须携带审批事实与 D7 稳定 action key。"""
    from langgraph.types import Command

    from rag_kb.application.approval_service import stable_write_action_key

    approval_id, state_version = _paused_stage(stage)
    stage.approvals.decide_and_enqueue(
        auth=stage.approver,
        approval_id=approval_id,
        expected_task_state_version=state_version,
        decision="approve",
        reason=None,
    )

    resume_agent = ExecutorAgent(
        stage, (_hitl_snapshot(), _final_snapshot(_succeeded_candidate()))
    )
    _worker(stage, resume_agent).run_once()

    command = resume_agent.inputs[0]
    assert isinstance(command, Command)
    assert command.resume == {"decisions": [{"type": "approve"}]}
    context = resume_agent.contexts[0]
    assert context.approval_id == approval_id
    assert context.plan_revision >= 1
    assert context.plan_digest
    assert context.evidence_digest
    assert context.evidence_ids == ("evidence-1",)
    approval = _approval_rows(stage)[0]
    expected_key = stable_write_action_key(
        task_id="task-1",
        thread_id="task-1",
        checkpoint_ns="",
        checkpoint_id="checkpoint-1",
        interrupt_id="interrupt-1",
        tool_call_id="call_1",
        tool_name="create_report_artifact",
        tool_version="1.0.0",
        arguments_hash=approval.arguments_hash,
        evidence_digest=approval.evidence_digest,
    )
    assert context.idempotency_key == expected_key


def test_settle_verifies_declared_artifacts_belonging_and_hash(stage) -> None:
    """settle 前逐个校验产物：归属 + ready + hash 与正文一致；任一失败 fail-closed。"""
    import hashlib

    from rag_kb.infrastructure.database.models import ArtifactRow

    worker = _worker(stage, ScriptedAgent(()))
    approval_id, _state_version = _paused_stage(stage)
    approval = _approval_rows(stage)[0]
    assert approval.approval_id == approval_id
    store = LocalObjectStore(stage.tmp_path / "objects")
    content = b"report-body"
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    store.put_exclusive("artifacts/tenant/task/report/k.md", content)
    approval = _approval_rows(stage)[0]
    with UnitOfWork(stage.database) as work:
        assert work.session is not None
        # artifacts→tool_calls→approvals 十列复合 FK：先落与审批绑定一致的执行行。
        work.session.add(
            ToolCallRow(
                tool_call_id="call_1",
                task_id="task-1",
                tenant_id="tenant-a",
                approval_id=approval.approval_id,
                plan_revision=approval.plan_revision,
                plan_digest=approval.plan_digest,
                evidence_digest=approval.evidence_digest,
                tool_name=approval.tool_name,
                tool_version=approval.tool_version,
                arguments_hash=approval.arguments_hash,
                principal_hash=approval.principal_hash,
                mode="write",
                idempotency_key="key-artifact",
                risk_level="low",
                status="succeeded",
                result_ref="tool-results/task-1/call_1.json",
                result_hash=digest,
                result_size=len(content),
                result_expires_at=datetime.now(UTC) + timedelta(days=7),
                completed_at=datetime.now(UTC),
                verified_at=datetime.now(UTC),
                created_at=datetime.now(UTC),
            )
        )
        work.session.flush()
        work.session.add(
            ArtifactRow(
                artifact_id="artifact-1",
                task_id="task-1",
                tenant_id="tenant-a",
                tool_call_id="call_1",
                artifact_type="report",
                storage_key="artifacts/tenant/task/report/k.md",
                content_hash=digest,
                status="ready",
            )
        )

    settlement = DatabaseTaskWorkerSettlement(
        stage.database, LocalObjectStore(stage.tmp_path / "objects")
    )
    summaries = settlement.verify_artifacts(
        tenant_id="tenant-a", task_id="task-1", artifact_ids=("artifact-1",)
    )
    assert len(summaries) == 1
    assert summaries[0].artifact_id == "artifact-1"
    assert summaries[0].content_hash == digest
    assert summaries[0].status == "ready"

    # hash 与正文不一致 → fail-closed，不进入最终结果。
    with UnitOfWork(stage.database) as work:
        assert work.session is not None
        row = work.session.get(ArtifactRow, "artifact-1")
        assert row is not None
        row.content_hash = "sha256:" + "f" * 64
    with pytest.raises(TaskWorkerPersistenceError) as tampered:
        settlement.verify_artifacts(
            tenant_id="tenant-a", task_id="task-1", artifact_ids=("artifact-1",)
        )
    assert tampered.value.code == "task_artifact_binding_invalid"

    # 未知/不属于任务的 artifact id → fail-closed。
    with pytest.raises(TaskWorkerPersistenceError) as missing:
        settlement.verify_artifacts(
            tenant_id="tenant-a", task_id="task-1", artifact_ids=("artifact-missing",)
        )
    assert missing.value.code == "task_artifact_binding_invalid"
