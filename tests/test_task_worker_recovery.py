"""Stage 6 pending action 恢复：不重调 planner、不重放写动作。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update

from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.pending_action_store import PendingActionStore
from rag_kb.application.task_runtime import TaskRuntime
from rag_kb.application.task_worker import TaskWorker
from rag_kb.application.tool_execution_service import ToolExecutionResult
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import (
    AskUserAction,
    FinishAction,
    ProposeAction,
    SearchAction,
    TaskMode,
    TaskNode,
    UpdatePlanAction,
)
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentTaskRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PlanRevisionRow,
    PrincipalRow,
    TaskEventRow,
    TenantRow,
)
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskAuthContextPort,
    DatabaseTaskEvidencePort,
    DatabaseTaskPlanBindingPort,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.task_runtime_adapter import LeasedTaskRuntimeRepository
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore


class Planner:
    def __init__(self, action) -> None:
        self.action = action
        self.calls = 0

    def next_action(self, state, observations=()):
        self.calls += 1
        if self.action is None:
            raise AssertionError("恢复 pending action 时不得调用 planner")
        return self.action


class CountingExecutor:
    """模拟执行层的原子工具计数/state-version 更新，不写任何正文。"""

    def __init__(self, database) -> None:
        self.database = database
        self.calls = 0
        self.requests = []

    def execute(self, request):
        self.calls += 1
        self.requests.append(request)
        with UnitOfWork(self.database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(AgentTaskRow.task_id == request.task_id)
                .with_for_update()
            )
            assert row is not None
            row.tool_calls += 1
            row.state_version += 1
        return ToolExecutionResult(
            tool_call_id=uuid4().hex,
            status="succeeded",
            result_hash=f"sha256:{self.calls:064x}",
            result={
                "result_reference": "safe-ref",
                "evidence_ids": [f"evidence-{self.calls}"],
                "items": [],
            },
        )


def _setup(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'worker.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    auth = AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["task:create", "task:read", "knowledge:read", "artifact:write"]},
        request_id="request-a",
    )
    approver = AuthContext(
        tenant_id="tenant-a",
        user_id="approver-a",
        attributes={"permissions": ["task:approve"]},
        request_id="approve-a",
    )
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add_all(
            [
                PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=auth.attributes),
                PrincipalRow(tenant_id="tenant-a", user_id="approver-a", roles=[], attributes=approver.attributes),
            ]
        )
        session.flush()
        session.add_all(
            [
                DocumentRow(tenant_id="tenant-a", document_id="doc-1", title="制度", department="安全", status="active", created_by="owner-a", projection_generation=1, applied_projection_generation=1, applied_projection_version_id="version-1"),
                ACLGrantRow(grant_id="grant-1", tenant_id="tenant-a", resource_type="document", resource_id="doc-1", subject_type="user", subject_id="owner-a", permissions=["knowledge:read", "document:read"]),
            ]
        )
        session.flush()
        session.add(DocumentVersionRow(version_id="version-1", tenant_id="tenant-a", document_id="doc-1", version=1, lifecycle_status="active", source_object_key="doc-1", source_file_name="doc.md", content_hash="sha256:content", index_version="kb-v1", desired_projection_generation=1, applied_projection_generation=1))
        session.flush()
        session.add_all(
            [
                ChunkMetadataRow(chunk_id="chunk-1", version_id="version-1", tenant_id="tenant-a", document_id="doc-1", document_version=1, content_hash="sha256:chunk", source_locator="doc/1", source_file_name="doc.md", index_version="kb-v1"),
                IngestionJobRow(job_id="job-1", tenant_id="tenant-a", version_id="version-1", status="succeeded", indexed_chunk_count=1, index_manifest_digest="sha256:manifest", index_completed_at=datetime.now(UTC), projection_generation=1, projection_lease_generation=1),
            ]
        )
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=auth, goal="验证恢复", task_mode=TaskMode.CONTROLLED_ACTION, task_id="task-a")
    with database.sessions.begin() as session:
        session.add(EvidenceRecordRow(evidence_id="evidence-1", task_id="task-a", query_fingerprint="sha256:query", tenant_id="tenant-a", document_id="doc-1", version_id="version-1", document_version=1, chunk_id="chunk-1", content_hash="sha256:chunk", source_locator="doc/1", permission_granted=True, supported_claim_ids=[]))
    return database, repository, auth, approver


def _worker(database, repository, planner, executor, tmp_path: Path) -> TaskWorker:
    registry = ToolRegistry()
    return TaskWorker(
        repository,
        planner,
        executor,
        registry,
        ApprovalService(database),
        DatabaseTaskAuthContextPort(database),
        DatabaseTaskPlanBindingPort(database),
        DatabaseTaskEvidencePort(database),
        PendingActionStore(LocalObjectStore(tmp_path / "objects")),
        worker_id="worker-a",
    )


def _expire_lease(database) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            update(AgentTaskRow)
            .where(AgentTaskRow.task_id == "task-a")
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )


def test_restart_executes_staged_read_once_without_second_planner_or_tool_count(tmp_path: Path) -> None:
    database, repository, _auth, _approver = _setup(tmp_path)
    initial = repository.claim_task("task-a", worker_id="seed", lease_seconds=60)
    assert initial is not None
    bootstrapped = repository.bootstrap_initial_plan(initial, trace_id="seed", run_id="seed")
    staged_planner = Planner(SearchAction(query="发布检查", reason_code="need_evidence"))
    runtime = TaskRuntime(
        LeasedTaskRuntimeRepository(
            repository,
            bootstrapped,
            trace_id="seed",
            run_id="seed",
            pending_actions=PendingActionStore(LocalObjectStore(tmp_path / "objects")),
        ),
        staged_planner,
    )
    staged = runtime.step("task-a")
    assert staged.pending_action is not None and staged.current_node is TaskNode.EXECUTE_READ_TOOL
    _expire_lease(database)

    executor = CountingExecutor(database)
    recovered = _worker(database, repository, Planner(None), executor, tmp_path).run_once()
    state = repository.get_task(auth=_auth, task_id="task-a")

    assert recovered.status == "progressed"
    assert executor.calls == 1
    assert state.tool_calls == 1
    assert state.pending_action is None
    assert state.current_node is TaskNode.RECORD_EVIDENCE

    # 下一次领取只做确定性 RECORD→EVALUATE→SELECT 推进，随后可安全选择不同检索。
    continued = _worker(
        database,
        repository,
        Planner(SearchAction(query="发布审批责任", reason_code="need_more_evidence")),
        executor,
        tmp_path,
    ).run_once()
    state = repository.get_task(auth=_auth, task_id="task-a")
    assert continued.status == "progressed", continued
    assert executor.calls == 2
    assert state.current_node is TaskNode.RECORD_EVIDENCE


def test_new_task_bootstrap_and_update_plan_create_real_revisions(tmp_path: Path) -> None:
    """公开 create_task 后，worker 必须先落初始计划，再让模型重规划。"""

    database, repository, auth, _approver = _setup(tmp_path)
    executor = CountingExecutor(database)
    updated = _worker(
        database,
        repository,
        Planner(UpdatePlanAction(steps=("检索证据", "生成答复"), reason_code="plan_refresh")),
        executor,
        tmp_path,
    ).run_once()
    assert updated.status == "progressed"
    after_update = repository.get_task(auth=auth, task_id="task-a")
    assert after_update.plan_revision == 2
    assert after_update.current_node is TaskNode.SELECT_ACTION

    with database.sessions() as session:
        revisions = list(
            session.scalars(
                select(PlanRevisionRow)
                .where(PlanRevisionRow.task_id == "task-a")
                .order_by(PlanRevisionRow.revision)
            )
        )
        event_types = list(
            session.scalars(
                select(TaskEventRow.event_type)
                .where(TaskEventRow.task_id == "task-a")
                .order_by(TaskEventRow.sequence)
            )
        )
    assert [(row.revision, row.status) for row in revisions] == [(1, "superseded"), (2, "active")]
    assert all(row.plan_digest.startswith("sha256:") for row in revisions)
    assert all(row.evidence_digest.startswith("sha256:") for row in revisions)
    assert "plan_created" in event_types and "plan_revised" in event_types

    # 下一轮通过真实 revision=2 计划绑定执行；没有手工预置任何内部状态。
    continued = _worker(
        database,
        repository,
        Planner(SearchAction(query="发布检查", reason_code="need_evidence")),
        executor,
        tmp_path,
    ).run_once()
    assert continued.status == "progressed"
    assert executor.requests[-1].plan_revision == 2


def test_new_task_bootstrap_allows_first_ask_user_action(tmp_path: Path) -> None:
    database, repository, auth, _approver = _setup(tmp_path)
    result = _worker(
        database,
        repository,
        Planner(AskUserAction(question="需要哪个部门？", reason_code="need_scope")),
        CountingExecutor(database),
        tmp_path,
    ).run_once()
    state = repository.get_task(auth=auth, task_id="task-a")
    assert result.status == "progressed"
    assert state.current_node is TaskNode.CHECK_CLARIFICATION
    assert state.plan_revision == 1


def test_approved_write_wakes_and_restarts_exact_pending_action_once(tmp_path: Path) -> None:
    database, repository, auth, approver = _setup(tmp_path)
    executor = CountingExecutor(database)
    searched = _worker(
        database,
        repository,
        Planner(SearchAction(query="发布检查", reason_code="need_evidence")),
        executor,
        tmp_path,
    ).run_once()
    assert searched.status == "progressed"
    replanned = _worker(
        database,
        repository,
        Planner(UpdatePlanAction(steps=("基于证据生成报告",), reason_code="bind_evidence")),
        executor,
        tmp_path,
    ).run_once()
    assert replanned.status == "progressed"
    proposal = ProposeAction(
        tool_name="create_report_artifact",
        arguments={"title": "报告", "content": "受控", "format": "markdown"},
        evidence_ids=("evidence-1",),
        reason_code="approved_output",
    )
    waiting = _worker(database, repository, Planner(proposal), executor, tmp_path).run_once()
    assert waiting.status == "waiting_for_approval", waiting
    pending = repository.get_task(auth=auth, task_id="task-a")
    assert pending.pending_approval is not None

    ApprovalService(database).approve(
        auth=approver,
        approval_id=pending.pending_approval,
        expected_task_id="task-a",
        expected_plan_revision=2,
    )
    woken = repository.get_task(auth=auth, task_id="task-a")
    assert woken.current_node is TaskNode.EXECUTE_APPROVED_ACTION

    recovered = _worker(database, repository, Planner(None), executor, tmp_path).run_once()
    finished = repository.get_task(auth=auth, task_id="task-a")

    assert recovered.status == "progressed"
    assert executor.calls == 2
    assert finished.tool_calls == 2
    assert finished.pending_action is None and finished.pending_approval is None
    assert finished.current_node is TaskNode.VERIFY_ACTION

    # 验证后的下一轮固定进入 GENERATE_RESULT；Finish 因而是合法终结动作。
    terminal = _worker(
        database,
        repository,
        Planner(FinishAction(supported_claim_ids=("evidence-1",), reason_code="write_completed")),
        executor,
        tmp_path,
    ).run_once()
    completed = repository.get_task(auth=auth, task_id="task-a")
    assert terminal.status == "progressed", terminal
    assert completed.current_node is TaskNode.TERMINAL
