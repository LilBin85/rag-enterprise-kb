"""M2 批次 2：真实 PostgreSQL + FencedPostgresSaver 上的双层恢复与崩溃窗口。

禁止 fake checkpointer：全部场景使用真实 Alembic 迁移库 + FencedPostgresSaver +
真实 create_agent（锁定版本契约假模型 + 真实 ToolExecutionService/adapter）。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import select, text

from rag_kb.agent import factory
from rag_kb.agent.checkpointer import open_postgres_checkpointer
from rag_kb.agent.result import AgentResultStore
from rag_kb.agent.tools import build_controlled_action_tools
from rag_kb.application.approval_service import (
    ApprovalService,
    ApprovalServiceError,
    stable_write_action_key,
)
from rag_kb.application.task_worker import TaskWorker, TaskWorkerError
from rag_kb.application.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import (
    ToolRegistry,
    builtin_tool_definitions,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentCapabilityLeaseRow,
    AgentTaskRow,
    ApprovalRow,
    ArtifactRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PlanRevisionRow,
    TaskEventRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_execution import DatabaseTaskAuthContextPort
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
    TaskRepositoryError,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.infrastructure.tools.artifacts import LocalArtifactToolAdapter

POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured"
)

_CANDIDATE_ARGS = {
    "status": "succeeded",
    "answer": "报告已完成",
    "evidence_ids": ["evidence-1"],
    "stop_reason": "completed",
}


def _write_arguments() -> dict[str, object]:
    return {"title": "差旅报告", "content": "基于证据的正文", "format": "markdown"}


def _model_messages(evidence_id: str) -> list[AIMessage]:
    """锁定版本契约消息序列：①WRITE tool_call → interrupt；②结构化候选。"""
    candidate_args = dict(_CANDIDATE_ARGS, evidence_ids=[evidence_id])
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_report_artifact",
                    "args": _write_arguments(),
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "TaskResultCandidate",
                    "args": candidate_args,
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
    ]


class FakeModel(GenericFakeChatModel):
    """锁定版本契约假模型：bind_tools 恒返回自身，按序回放消息。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


class CountingLocalAdapter:
    """真实 LocalArtifactToolAdapter 的计数包装（verify 委托真实 readback）。"""

    def __init__(self, inner: LocalArtifactToolAdapter) -> None:
        self._inner = inner
        self.calls = 0

    def execute(self, arguments, context):  # noqa: ANN001
        self.calls += 1
        return self._inner.execute(arguments, context)

    def verify(self, result, context):  # noqa: ANN001
        return self._inner.verify(result, context)


class LateArtifactAdapter:
    """超时后才真正落地的 adapter：迟到完成不得把 UNKNOWN 改回 ready。"""

    def __init__(self, inner: LocalArtifactToolAdapter, *, delay_seconds: float) -> None:
        self._inner = inner
        self._delay = delay_seconds
        self.calls = 0
        from threading import Event

        self.done = Event()

    def execute(self, arguments, context):  # noqa: ANN001
        self.calls += 1
        time.sleep(self._delay)
        try:
            return self._inner.execute(arguments, context)
        finally:
            self.done.set()

    def verify(self, result, context):  # noqa: ANN001
        return self._inner.verify(result, context)


class ToolServiceProxy:
    """ToolExecutionService 代理：注入一次性故障钩子，其余委托。"""

    def __init__(self, inner: ToolExecutionService, hook=None) -> None:
        self._inner = inner
        self._hook = hook

    def execute(self, request: ToolExecutionRequest):
        if self._hook is not None:
            self._hook(request)
        return self._inner.execute(request)

    def close(self) -> None:
        self._inner.close()

    def expire_stale_running(self, *, limit: int = 100) -> int:
        return self._inner.expire_stale_running(limit=limit)


class CrashingWorker(TaskWorker):
    """模拟 resume 提交后进程死亡：异常直接外抛，不收敛业务状态。"""

    def _run_claimed(self, lease: TaskLease, **kwargs: object):
        return self._advance(lease, **kwargs)


class FailOnceFinishRepository(DatabaseTaskRepository):
    """第一次 finish_task 抛异常：模拟 checkpoint 完成后、摘要提交前崩溃。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finish_attempts = 0

    def finish_task(self, lease, **kwargs):  # noqa: ANN001, ANN003
        self.finish_attempts += 1
        if self.finish_attempts == 1:
            raise RuntimeError("injected finish crash")
        return super().finish_task(lease, **kwargs)


_CHECKPOINT_STATE: dict[str, object] = {}


def _checkpointer():
    saver = _CHECKPOINT_STATE.get("saver")
    if saver is None:
        saver, connection = open_postgres_checkpointer(POSTGRES_TEST_URL)
        saver.setup()
        _CHECKPOINT_STATE["saver"] = saver
        _CHECKPOINT_STATE["connection"] = connection
    return saver


@dataclass
class _Stage:
    database: object
    repository: DatabaseTaskRepository
    requester: AuthContext
    approver: AuthContext
    tenant_id: str
    task_id: str
    evidence_id: str
    object_root: Path
    lease: TaskLease
    suffix: str


def _select_task(task_id: str):
    return select(AgentTaskRow).where(AgentTaskRow.task_id == task_id)


def _build_stage(tmp_path: Path) -> _Stage:
    """唯一后缀的受控任务环境：真实 PG 行 + 手工 running 租约。"""
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"m2-recovery-{suffix}"
    task_id = f"task-{suffix}"
    object_root = tmp_path / f"objects-{suffix}"
    requester = AuthContext(
        tenant_id=tenant_id,
        user_id="requester",
        attributes={
            "permissions": ["artifact:write", "knowledge:read", "document:read"]
        },
        request_id=f"req-{suffix}",
    )
    approver = AuthContext(
        tenant_id=tenant_id,
        user_id="approver",
        attributes={"permissions": ["task:approve"]},
        request_id=f"appr-{suffix}",
    )
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (tenant_id,name,active,created_at) "
                "VALUES (:t,'M2 Recovery',TRUE,CURRENT_TIMESTAMP)"
            ),
            {"t": tenant_id},
        )
        for user, attributes in (
            ("requester", requester.attributes),
            ("approver", approver.attributes),
        ):
            connection.execute(
                text(
                    "INSERT INTO principals (tenant_id,user_id,roles,attributes,active,created_at) "
                    "VALUES (:t,:u,'[]',:a,TRUE,CURRENT_TIMESTAMP)"
                ),
                {"t": tenant_id, "u": user, "a": json.dumps(attributes)},
            )
    repository = DatabaseTaskRepository(
        database,
        task_capability_name="task-agent",
        task_capability_protocol_version="m2-fenced-v1",
    )
    with UnitOfWork(database) as work:
        assert work.session is not None
        # capability 租约是全局行（无 tenant）：先清理遗留，再为创建门注册。
        work.session.execute(
            text(
                "DELETE FROM agent_capability_leases "
                "WHERE capability_name = 'task-agent' AND worker_id = 'worker-1'"
            )
        )
        work.session.add(
            AgentCapabilityLeaseRow(
                capability_name="task-agent",
                protocol_version="m2-fenced-v1",
                worker_id="worker-1",
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=600),
                updated_at=datetime.now(UTC),
            )
        )
    repository.create_task(
        auth=requester,
        goal="受控恢复验证",
        task_mode=TaskMode.CONTROLLED_ACTION,
        task_id=task_id,
    )
    evidence_id = f"evidence-{suffix}"
    with UnitOfWork(database) as work:
        assert work.session is not None
        task = work.session.scalar(_select_task(task_id))
        assert task is not None
        task.status = "running"
        task.lease_owner = "worker-1"
        task.lease_expires_at = datetime.now(UTC) + timedelta(seconds=300)
        task.lease_generation = 1
        work.session.flush()
        work.session.add_all(
            [
                DocumentRow(
                    tenant_id=tenant_id,
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
                    tenant_id=tenant_id,
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
                tenant_id=tenant_id,
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
                    tenant_id=tenant_id,
                    document_id="doc-1",
                    document_version=1,
                    content_hash="sha256:chunk",
                    source_locator="doc/1",
                    source_file_name="doc.md",
                    index_version="kb-v1",
                ),
                IngestionJobRow(
                    job_id="job-1",
                    tenant_id=tenant_id,
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
                evidence_id=evidence_id,
                task_id=task_id,
                query_fingerprint="sha256:query",
                tenant_id=tenant_id,
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
    lease = TaskLease(
        repository.get_task(auth=requester, task_id=task_id),
        "worker-1",
        1,
        TaskBudgets(max_model_calls=8, max_tool_calls=8),
    )
    return _Stage(
        database=database,
        repository=repository,
        requester=requester,
        approver=approver,
        tenant_id=tenant_id,
        task_id=task_id,
        evidence_id=evidence_id,
        object_root=object_root,
        lease=lease,
        suffix=suffix,
    )


def _build_chain(stage: _Stage, messages, hook=None, *, timeout_seconds: int | None = None, artifact_adapter=None):
    """真实 create_agent + FencedPostgresSaver + 真实 adapter/executor/approvals。

    hook：ToolServiceProxy 的一次性故障注入钩子（崩溃窗口用）；
    timeout_seconds：覆盖 create_report_artifact 的执行超时（UNKNOWN 窗口用）；
    artifact_adapter：替换计数 adapter（迟到完成用）。
    """
    store = LocalObjectStore(stage.object_root)
    adapter = artifact_adapter or CountingLocalAdapter(
        LocalArtifactToolAdapter(stage.database, store, artifact_type="report")
    )
    definitions = builtin_tool_definitions()
    if timeout_seconds is not None:
        definitions = tuple(
            definition.model_copy(update={"timeout_seconds": timeout_seconds})
            if definition.name == "create_report_artifact"
            else definition
            for definition in definitions
        )
    registry = ToolRegistry(
        definitions=definitions,
        adapters={"create_report_artifact": adapter},
    )
    approvals = ApprovalService(stage.database, registry=registry)
    tools = ToolExecutionService(
        stage.database,
        registry,
        approvals,
        tool_results=LocalObjectStore(stage.object_root / "tool-results"),
    )
    proxy = ToolServiceProxy(tools, hook=hook)
    model = FakeModel(messages=iter(messages))
    agent = factory.build_controlled_action_agent(
        model,
        build_controlled_action_tools(proxy),
        _checkpointer(),
        max_model_calls=8,
        max_tool_calls=8,
    )
    return agent, adapter, approvals, proxy


def _worker(stage: _Stage, agent, *, approvals, tools, repository=None) -> TaskWorker:
    return TaskWorker(
        repository=repository or stage.repository,
        auth_contexts=DatabaseTaskAuthContextPort(stage.database),
        agent_for=lambda _budgets: agent,
        controlled_agent_for=lambda _budgets: agent,
        results=AgentResultStore(LocalObjectStore(stage.object_root / "task-results")),
        evidence=EvidenceLedgerRepository(),
        tools=tools,
        database=stage.database,
        worker_id="worker-1",
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
        approvals=approvals,
        objects=LocalObjectStore(stage.object_root),
    )


def _release_lease(stage: _Stage) -> None:
    """模拟进程死亡后的失租收敛：释放租约，等待同协议 Worker 重领取。"""
    with UnitOfWork(stage.database) as work:
        assert work.session is not None
        task = work.session.scalar(_select_task(stage.task_id))
        assert task is not None
        task.lease_owner = None
        task.lease_expires_at = None


def _reclaim(stage: _Stage):
    return stage.repository.claim_next(
        worker_id="worker-1",
        lease_seconds=300,
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
    )


def _state_version(stage: _Stage) -> int:
    with stage.database.sessions() as session:
        row = session.scalar(_select_task(stage.task_id))
        assert row is not None
        return row.state_version


def _approvals_rows(stage: _Stage) -> list[ApprovalRow]:
    with stage.database.sessions() as session:
        return list(
            session.scalars(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == stage.tenant_id,
                    ApprovalRow.task_id == stage.task_id,
                )
            )
        )


def _plan_rows(stage: _Stage) -> list[PlanRevisionRow]:
    with stage.database.sessions() as session:
        return list(
            session.scalars(
                select(PlanRevisionRow).where(
                    PlanRevisionRow.tenant_id == stage.tenant_id,
                    PlanRevisionRow.task_id == stage.task_id,
                )
            )
        )


def _task_row(stage: _Stage) -> AgentTaskRow:
    with stage.database.sessions() as session:
        row = session.scalar(_select_task(stage.task_id))
        assert row is not None
        return row


def _tool_call_rows(stage: _Stage) -> list[ToolCallRow]:
    with stage.database.sessions() as session:
        return list(
            session.scalars(
                select(ToolCallRow).where(ToolCallRow.task_id == stage.task_id)
            )
        )


def _artifact_rows(stage: _Stage) -> list[ArtifactRow]:
    with stage.database.sessions() as session:
        return list(
            session.scalars(
                select(ArtifactRow).where(
                    ArtifactRow.tenant_id == stage.tenant_id,
                    ArtifactRow.task_id == stage.task_id,
                )
            )
        )


@pytest.fixture()
def stage(tmp_path: Path):
    built = _build_stage(tmp_path)
    yield built
    # 线程 checkpoint 清理 + 业务行清理（按 FK 依赖倒序）。
    _checkpointer().delete_thread(built.task_id)
    try:
        with built.database.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM agent_capability_leases "
                    "WHERE capability_name = 'task-agent' AND worker_id = 'worker-1'"
                )
            )
            for table in (
                "artifacts",
                "tool_calls",
                "approvals",
                "plan_revisions",
                "evidence_records",
                "acl_grants",
                "ingestion_jobs",
                "chunk_metadata",
                "document_versions",
                "documents",
                "agent_tasks",
                "principals",
                "tenants",
            ):
                connection.execute(
                    text(f"DELETE FROM {table} WHERE tenant_id = :t"),
                    {"t": built.tenant_id},
                )
    finally:
        built.database.engine.dispose()


# ---------------------------------------------------------------------------
# 基础链路：真实 PG checkpoint 上 pause → approve → resume 恰好执行一次
# ---------------------------------------------------------------------------


def test_full_approval_resume_executes_write_once_on_postgres(stage, caplog) -> None:
    agent, adapter, approvals, proxy = _build_chain(stage, _model_messages(stage.evidence_id))
    worker = _worker(stage, agent, approvals=approvals, tools=proxy)

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    assert adapter.calls == 0
    approval_id = _approvals_rows(stage)[0].approval_id
    approvals.approve(auth=stage.approver, approval_id=approval_id)

    with caplog.at_level("INFO"):
        outcome = worker.resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )

    assert outcome.status == "finished"
    assert adapter.calls == 1
    tool_calls = _tool_call_rows(stage)
    assert len(tool_calls) == 1
    assert tool_calls[0].status == "succeeded"
    assert tool_calls[0].idempotency_key
    assert tool_calls[0].result_ref and tool_calls[0].result_hash
    artifacts = _artifact_rows(stage)
    assert len(artifacts) == 1
    assert artifacts[0].status == "ready"
    assert artifacts[0].content_hash
    approval = _approvals_rows(stage)[0]
    assert approval.status == "executed"
    task = _task_row(stage)
    assert task.status == TaskStatus.SUCCEEDED.value
    assert task.result_kind == "controlled_action"
    assert task.artifact_refs == [artifacts[0].artifact_id]
    with stage.database.sessions() as session:
        terminal_event = session.scalar(
            select(TaskEventRow)
            .where(
                TaskEventRow.task_id == stage.task_id,
                TaskEventRow.status == TaskStatus.SUCCEEDED.value,
            )
            .order_by(TaskEventRow.sequence.desc())
        )
    assert terminal_event is not None
    assert terminal_event.counts == {"model_calls": 2, "tool_calls": 1}
    # 敏感信息门禁：工具结果正文、任务结果正文与日志不出现 storage_key。
    tool_result_payload = LocalObjectStore(stage.object_root / "tool-results").read(
        tool_calls[0].result_ref
    )
    assert "storage_key" not in tool_result_payload.decode("utf-8")
    task_result_payload = LocalObjectStore(stage.object_root / "task-results").read(
        task.result_ref
    )
    assert "storage_key" not in task_result_payload.decode("utf-8")
    assert "storage_key" not in caplog.text


# ---------------------------------------------------------------------------
# 崩溃窗口 A：resume 已提交、ToolNode/adapter 前崩溃 → 恢复后恰好执行一次
# ---------------------------------------------------------------------------


def test_window_a_crash_before_tool_node_recovers_executes_once(stage) -> None:
    calls = {"count": 0}

    def hook(_request: ToolExecutionRequest) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            # FATAL code：工具边界 re-raise，ToolMessage 不 checkpoint；
            # adapter 尚未调用，副作用为 0。
            raise ToolExecutionError("tool_task_lease_stale")

    agent, adapter, approvals, proxy = _build_chain(stage, _model_messages(stage.evidence_id), hook)
    crashing = CrashingWorker(
        repository=stage.repository,
        auth_contexts=DatabaseTaskAuthContextPort(stage.database),
        agent_for=lambda _budgets: agent,
        controlled_agent_for=lambda _budgets: agent,
        results=AgentResultStore(LocalObjectStore(stage.object_root / "task-results")),
        evidence=EvidenceLedgerRepository(),
        tools=proxy,
        database=stage.database,
        worker_id="worker-1",
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
        approvals=approvals,
        objects=LocalObjectStore(stage.object_root),
    )
    worker = _worker(stage, agent, approvals=approvals, tools=proxy)

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    approval_id = _approvals_rows(stage)[0].approval_id
    approvals.approve(auth=stage.approver, approval_id=approval_id)

    with pytest.raises(ToolExecutionError):
        crashing.resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )
    assert adapter.calls == 0
    assert len(_tool_call_rows(stage)) == 0

    # Worker 已死：释放租约，同协议 Worker 经 claim_next 重领取；
    # 同一 checkpoint 的 interrupt 幂等重建 waiting_for_approval 与审批。
    _release_lease(stage)
    recovered = _reclaim(stage)
    assert recovered is not None
    # 两条合法收敛路径：resume write 已存 checkpoint 时 invoke(None) 直接重放；
    # 否则 interrupt 仍在，幂等重建 waiting_for_approval 后重新 decision/resume。
    resync = _worker(stage, agent, approvals=approvals, tools=proxy)._run_claimed(
        recovered
    )
    if resync.status == "waiting_for_approval":
        assert len(_approvals_rows(stage)) == 1
        assert len(_plan_rows(stage)) == 1
        outcome = _worker(
            stage, agent, approvals=approvals, tools=proxy
        ).resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )
    else:
        outcome = resync
    assert outcome.status == "finished"
    assert adapter.calls == 1
    assert len(_tool_call_rows(stage)) == 1
    assert len(_artifact_rows(stage)) == 1
    assert _task_row(stage).status == TaskStatus.SUCCEEDED.value


# ---------------------------------------------------------------------------
# 崩溃窗口 B：adapter 成功、ToolMessage checkpoint 前崩溃 → result_ref 回读
# ---------------------------------------------------------------------------


def test_window_b_crash_after_adapter_reads_result_ref_single_execution(stage) -> None:
    calls = {"count": 0}

    agent, adapter, approvals, proxy = _build_chain(stage, _model_messages(stage.evidence_id))

    # 后置崩溃注入：真实 execute 完整落库后抛 FATAL——副作用已发生，
    # 但 ToolMessage 尚未 checkpoint。
    real_execute = proxy._inner.execute

    def execute_then_crash(request):
        result = real_execute(request)
        calls["count"] += 1
        if calls["count"] == 1:
            raise ToolExecutionError("tool_task_lease_stale")
        return result

    proxy._inner.execute = execute_then_crash  # type: ignore[method-assign]

    crashing = CrashingWorker(
        repository=stage.repository,
        auth_contexts=DatabaseTaskAuthContextPort(stage.database),
        agent_for=lambda _budgets: agent,
        controlled_agent_for=lambda _budgets: agent,
        results=AgentResultStore(LocalObjectStore(stage.object_root / "task-results")),
        evidence=EvidenceLedgerRepository(),
        tools=proxy,
        database=stage.database,
        worker_id="worker-1",
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
        approvals=approvals,
        objects=LocalObjectStore(stage.object_root),
    )
    worker = _worker(stage, agent, approvals=approvals, tools=proxy)

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    approval_id = _approvals_rows(stage)[0].approval_id
    approvals.approve(auth=stage.approver, approval_id=approval_id)

    with pytest.raises(ToolExecutionError):
        crashing.resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )

    # 副作用已发生且完整落库：恰一次执行、result_ref 绑定在案。
    assert adapter.calls == 1
    first_rows = _tool_call_rows(stage)
    assert len(first_rows) == 1
    assert first_rows[0].status == "succeeded"
    assert first_rows[0].result_ref and first_rows[0].result_hash
    assert len(_artifact_rows(stage)) == 1

    # 恢复：ToolNode 重放以同一 D7 key 命中 tool_calls 唯一约束，
    # 从 result_ref 回读相同结果，adapter 不再调用。
    _release_lease(stage)
    recovered = _reclaim(stage)
    assert recovered is not None
    outcome = _worker(stage, agent, approvals=approvals, tools=proxy)._run_claimed(
        recovered
    )
    if outcome.status == "waiting_for_approval":
        outcome = _worker(
            stage, agent, approvals=approvals, tools=proxy
        ).resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )
    assert outcome.status == "finished"
    assert adapter.calls == 1
    rows = _tool_call_rows(stage)
    assert len(rows) == 1
    assert rows[0].tool_call_id == first_rows[0].tool_call_id
    assert rows[0].result_ref == first_rows[0].result_ref
    assert len(_artifact_rows(stage)) == 1
    assert _task_row(stage).status == TaskStatus.SUCCEEDED.value


# ---------------------------------------------------------------------------
# 崩溃窗口 C：checkpoint 完成、业务摘要未同步 → 只幂等修正 agent_tasks
# ---------------------------------------------------------------------------


def test_window_c_checkpoint_done_summary_sync_idempotent(stage) -> None:
    failing_repository = FailOnceFinishRepository(
        stage.database,
        task_capability_name="task-agent",
        task_capability_protocol_version="m2-fenced-v1",
    )
    agent, adapter, approvals, proxy = _build_chain(stage, _model_messages(stage.evidence_id))
    worker = _worker(
        stage, agent, approvals=approvals, tools=proxy, repository=failing_repository
    )

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    approval_id = _approvals_rows(stage)[0].approval_id
    approvals.approve(auth=stage.approver, approval_id=approval_id)

    with pytest.raises(RuntimeError, match="injected finish crash"):
        CrashingWorker(
            repository=failing_repository,
            auth_contexts=DatabaseTaskAuthContextPort(stage.database),
            agent_for=lambda _budgets: agent,
            controlled_agent_for=lambda _budgets: agent,
            results=AgentResultStore(
                LocalObjectStore(stage.object_root / "task-results")
            ),
            evidence=EvidenceLedgerRepository(),
            tools=proxy,
            database=stage.database,
            worker_id="worker-1",
            capability_name="task-agent",
            protocol_version="m2-fenced-v1",
            approvals=approvals,
            objects=LocalObjectStore(stage.object_root),
        ).resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )

    # 副作用恰好一次；checkpoint 已含结构化候选；任务业务侧仍 RUNNING。
    assert adapter.calls == 1
    task = _task_row(stage)
    assert task.status == TaskStatus.RUNNING.value

    # 恢复：claim_next 领取后 graph state 为事实源，只做幂等摘要同步，
    # 不重跑任何图步骤（消息迭代器已耗尽，再 invoke 即异常）。
    _release_lease(stage)
    recovered = failing_repository.claim_next(
        worker_id="worker-1",
        lease_seconds=300,
        capability_name="task-agent",
        protocol_version="m2-fenced-v1",
    )
    assert recovered is not None
    outcome = _worker(
        stage, agent, approvals=approvals, tools=proxy, repository=failing_repository
    )._run_claimed(recovered)
    assert outcome.status == "finished"
    assert failing_repository.finish_attempts == 2
    assert adapter.calls == 1
    assert len(_tool_call_rows(stage)) == 1
    assert _task_row(stage).status == TaskStatus.SUCCEEDED.value
    assert _task_row(stage).result_kind == "controlled_action"


# ---------------------------------------------------------------------------
# 批准后失效矩阵：恢复可被接受/拒绝，但 adapter 调用次数必须为 0
# ---------------------------------------------------------------------------


def _mutate(stage: _Stage, mutation: str) -> None:
    with UnitOfWork(stage.database) as work:
        assert work.session is not None
        if mutation == "expiry":
            approval = work.session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == stage.tenant_id,
                    ApprovalRow.task_id == stage.task_id,
                )
            )
            assert approval is not None
            # 保持 expires_at > created_at（CHECK），同时相对当前时刻已过期。
            approval.created_at = datetime.now(UTC) - timedelta(minutes=10)
            approval.expires_at = datetime.now(UTC) - timedelta(minutes=5)
        elif mutation == "principal_changed":
            work.session.execute(
                text(
                    "UPDATE principals SET attributes = :a "
                    "WHERE tenant_id = :t AND user_id = 'requester'"
                ),
                {
                    "t": stage.tenant_id,
                    "a": json.dumps(
                        {
                            "permissions": [
                                "artifact:write",
                                "knowledge:read",
                                "document:read",
                                "extra:permission",
                            ]
                        }
                    ),
                },
            )
        elif mutation == "acl_revoked":
            work.session.execute(
                text("DELETE FROM acl_grants WHERE tenant_id = :t"),
                {"t": stage.tenant_id},
            )
        elif mutation == "evidence_withdrawn":
            work.session.execute(
                text("DELETE FROM evidence_records WHERE tenant_id = :t"),
                {"t": stage.tenant_id},
            )
        elif mutation == "arguments_drift":
            approval = work.session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == stage.tenant_id,
                    ApprovalRow.task_id == stage.task_id,
                )
            )
            assert approval is not None
            approval.arguments_hash = "sha256:" + "d" * 64
        elif mutation == "version_drift":
            approval = work.session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == stage.tenant_id,
                    ApprovalRow.task_id == stage.task_id,
                )
            )
            assert approval is not None
            approval.tool_version = "9.9.9"
        else:
            raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        "expiry",
        "principal_changed",
        "acl_revoked",
        "evidence_withdrawn",
        "arguments_drift",
        "version_drift",
    ),
)
def test_invalidated_approval_never_executes_adapter(stage, mutation: str) -> None:
    agent, adapter, approvals, proxy = _build_chain(stage, _model_messages(stage.evidence_id))
    worker = _worker(stage, agent, approvals=approvals, tools=proxy)

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    approval_id = _approvals_rows(stage)[0].approval_id
    if mutation != "expiry":
        approvals.approve(auth=stage.approver, approval_id=approval_id)
    _mutate(stage, mutation)

    # 审批期失效在 decision 阶段（首次提交即失效）或幂等重放+最终门拒绝；
    # 参数/版本漂移由 ToolExecutionService 最终门拒绝。
    # 无论哪种路径：副作用必须为 0（D6 行 6）。
    if mutation == "expiry":
        with pytest.raises(ApprovalServiceError):
            worker.resume_for_approval(
                auth=stage.approver,
                task_id=stage.task_id,
                expected_state_version=_state_version(stage),
                approval_id=approval_id,
                decision="approve",
            )
    else:
        outcome = worker.resume_for_approval(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            decision="approve",
        )
        assert outcome.status in {"finished", "failed"}

    assert adapter.calls == 0
    assert len(_tool_call_rows(stage)) == 0
    assert len(_artifact_rows(stage)) == 0


# ---------------------------------------------------------------------------
# WRITE timeout → UNKNOWN（真实 PG）：迟到完成不能恢复 ready，自动重放为 0
# ---------------------------------------------------------------------------


def test_write_timeout_unknown_never_replays_on_postgres(stage) -> None:
    real_adapter = LocalArtifactToolAdapter(
        stage.database, LocalObjectStore(stage.object_root), artifact_type="report"
    )
    late = LateArtifactAdapter(real_adapter, delay_seconds=2.5)
    agent, adapter, approvals, proxy = _build_chain(
        stage,
        _model_messages(stage.evidence_id),
        timeout_seconds=1,
        artifact_adapter=late,
    )
    worker = _worker(stage, agent, approvals=approvals, tools=proxy)

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    approval_id = _approvals_rows(stage)[0].approval_id
    approvals.approve(auth=stage.approver, approval_id=approval_id)

    outcome = worker.resume_for_approval(
        auth=stage.approver,
        task_id=stage.task_id,
        expected_state_version=_state_version(stage),
        approval_id=approval_id,
        decision="approve",
    )
    # 超时收敛 UNKNOWN 后任务已不可执行，settle 停在 stale；副作用归属未知。
    assert outcome.status == "stale"

    # 迟到线程真正落地：artifact 行已建但必须保持 unknown，不得恢复 ready。
    assert late.done.wait(timeout=15), "late adapter did not finish"
    assert late.calls == 1
    rows = _tool_call_rows(stage)
    assert len(rows) == 1
    assert rows[0].status == "unknown"
    assert rows[0].error_code == "tool_timeout"
    assert rows[0].result_ref is None
    task = _task_row(stage)
    assert task.status == TaskStatus.UNKNOWN.value
    assert task.lease_owner is None
    artifacts = _artifact_rows(stage)
    assert len(artifacts) == 1
    assert artifacts[0].status == "unknown"

    # 自动重放为 0：sweeper 只收敛状态、绝不重放；行已终态，无可扫。
    assert proxy._inner.expire_stale_running() == 0
    assert late.calls == 1

    # 结构性不可重放：UNKNOWN 任务既不能被 claim_next 领取，
    # 也不能走审批恢复领取（task 不再 waiting_for_approval）。
    assert _reclaim(stage) is None
    with pytest.raises(TaskRepositoryError):
        stage.repository.claim_approval_resume(
            auth=stage.approver,
            task_id=stage.task_id,
            expected_state_version=_state_version(stage),
            approval_id=approval_id,
            worker_id="worker-1",
            lease_seconds=300,
            capability_name="task-agent",
            protocol_version="m2-fenced-v1",
        )

    # 同 key 重复执行：任务 UNKNOWN 使最终门拒绝，adapter 调用数不增。
    facts, evidence_ids = approvals.load_action_facts(
        auth=stage.requester, approval_id=approval_id
    )
    duplicate_request = ToolExecutionRequest(
        auth=stage.requester,
        task_id=stage.task_id,
        tool_name="create_report_artifact",
        arguments=_write_arguments(),
        worker_id="worker-1",
        lease_generation=task.lease_generation,
        idempotency_key=stable_write_action_key(
            task_id=stage.task_id,
            thread_id=stage.task_id,
            checkpoint_ns=facts.checkpoint_ns,
            checkpoint_id=facts.checkpoint_id,
            interrupt_id=facts.interrupt_id,
            tool_call_id=facts.tool_call_id,
            tool_name=facts.tool_name,
            tool_version=facts.tool_version,
            arguments_hash=facts.canonical_arguments_hash,
            evidence_digest=facts.evidence_hash,
        ),
        approval_id=approval_id,
        plan_revision=facts.plan_revision,
        plan_digest=facts.plan_digest,
        evidence_digest=facts.evidence_hash,
        evidence_ids=evidence_ids,
    )
    with pytest.raises(ToolExecutionError):
        proxy._inner.execute(duplicate_request)
    assert late.calls == 1
    assert len(_tool_call_rows(stage)) == 1


# ---------------------------------------------------------------------------
# 并发 decision：两个并发恢复驱动只产生一条有效 action、一个 tool call
# ---------------------------------------------------------------------------


def test_concurrent_decisions_single_action_on_postgres(stage) -> None:
    agent, adapter, approvals, proxy = _build_chain(
        stage, _model_messages(stage.evidence_id)
    )
    worker = _worker(stage, agent, approvals=approvals, tools=proxy)

    paused = worker._run_claimed(stage.lease)
    assert paused.status == "waiting_for_approval"
    approval_id = _approvals_rows(stage)[0].approval_id
    approvals.approve(auth=stage.approver, approval_id=approval_id)

    def drive(_index: int):
        try:
            return _worker(
                stage, agent, approvals=approvals, tools=proxy
            ).resume_for_approval(
                auth=stage.approver,
                task_id=stage.task_id,
                expected_state_version=_state_version(stage),
                approval_id=approval_id,
                decision="approve",
            )
        except (TaskRepositoryError, ApprovalServiceError, TaskWorkerError) as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(drive, range(2)))

    finished = [
        outcome
        for outcome in outcomes
        if getattr(outcome, "status", None) == "finished"
    ]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(finished) == 1
    assert len(conflicts) == 1
    # 稳定冲突码：state_version CAS 或终态拒绝（后者由 winner 进度决定）。
    assert getattr(conflicts[0], "code", "") in {
        "task_state_conflict",
        "task_not_waiting_for_approval",
    }

    assert adapter.calls == 1
    assert len(_tool_call_rows(stage)) == 1
    assert len(_artifact_rows(stage)) == 1
    approvals_rows = _approvals_rows(stage)
    assert len(approvals_rows) == 1
    assert approvals_rows[0].status == "executed"
    assert _task_row(stage).status == TaskStatus.SUCCEEDED.value


