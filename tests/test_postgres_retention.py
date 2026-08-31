"""M1 批 3 隔离 PostgreSQL 集成：工具结果 TTL 清理与终态任务线程清理。

运行方式：临时数据库经 docker compose createdb/dropdb 创建删除，
RAG_KB_TEST_POSTGRES_URL 指向该库；不触碰持久 rag_kb 数据库。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import NotRequired, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from sqlalchemy import func, select, text, update as sa_update

from rag_kb.agent.checkpointer import open_postgres_checkpointer
from rag_kb.agent.result import (
    AgentResultStore,
    TaskResult,
    TaskResultCandidate,
    build_task_result,
)
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_retention_service import TaskRetentionService
from rag_kb.application.tool_execution_service import (
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    ArtifactRow,
    PlanRevisionRow,
    PrincipalRow,
    TaskEventRow,
    TenantRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.object_store.local import LocalObjectStore


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


def _auth(*, tenant_id: str, user_id: str) -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )


class _OkAdapter:
    def execute(self, _arguments, _context):
        return {"result_reference": "ref", "evidence_ids": [], "items": []}

    def verify(self, _result, _context) -> bool:
        return True


@dataclass(frozen=True)
class _Ids:
    tenant_id: str
    user_id: str
    task_id: str


@pytest.fixture()
def postgres_db(tmp_path: Path):
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    from uuid import uuid4

    suffix = uuid4().hex
    ids = _Ids(
        tenant_id=f"retention-tenant-{suffix}",
        user_id=f"retention-owner-{suffix}",
        task_id=f"retention-task-{suffix}",
    )
    auth = _auth(tenant_id=ids.tenant_id, user_id=ids.user_id)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id=ids.tenant_id, name=ids.tenant_id))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id=ids.tenant_id,
                user_id=ids.user_id,
                roles=[],
                attributes=auth.attributes,
            )
        )
    try:
        yield database, LocalObjectStore(tmp_path / "objects"), auth, ids
    finally:
        from sqlalchemy import text

        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM approvals WHERE tenant_id = :tenant"),
                {"tenant": ids.tenant_id},
            )
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": ids.tenant_id},
            )
        database.engine.dispose()


def _object_exists(objects: LocalObjectStore, ref: str) -> bool:
    try:
        objects.read(ref)
        return True
    except OSError:
        return False


class _RecordingCheckpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_thread(self, task_id: str) -> None:
        self.deleted.append(task_id)


def _claim_research_task(
    repository: DatabaseTaskRepository,
    auth: AuthContext,
    task_id: str,
):
    repository.create_task(
        auth=auth,
        goal="研究",
        task_mode=TaskMode.RESEARCH,
        task_id=task_id,
    )
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    return lease


def _expire_task_content(database, *, tenant_id: str, task_id: str) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE agent_tasks "
                "SET content_expires_at = clock_timestamp() - INTERVAL '1 second' "
                "WHERE tenant_id = :tenant_id AND task_id = :task_id"
            ),
            {"tenant_id": tenant_id, "task_id": task_id},
        )


def _set_mtime(objects: LocalObjectStore, ref: str, when: datetime) -> None:
    path = objects.root.joinpath(*PurePosixPath(ref).parts)
    os.utime(path, (when.timestamp(), when.timestamp()))


def _controlled_artifact(database, objects, auth, ids):
    task_id = f"controlled-{ids.task_id.rsplit('-', 1)[-1]}"
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=auth,
        goal="生成报告",
        task_mode=TaskMode.CONTROLLED_ACTION,
        task_id=task_id,
    )
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    ref = f"artifacts/tenant-hash/{task_id}/report/body.md"
    content = b"controlled artifact"
    objects.put_exclusive(ref, content)
    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        session.add(
            PlanRevisionRow(
                plan_revision_id=f"plan-{task_id}",
                task_id=task_id,
                tenant_id=ids.tenant_id,
                revision=1,
                schema_version=1,
                plan={},
                plan_digest="sha256:" + "p" * 64,
                evidence_digest="sha256:" + "e" * 64,
                status="ready",
            )
        )
        session.flush()
        session.add(
            ApprovalRow(
                approval_id=f"approval-{task_id}",
                task_id=task_id,
                tenant_id=ids.tenant_id,
                plan_revision=1,
                plan_digest="sha256:" + "p" * 64,
                evidence_digest="sha256:" + "e" * 64,
                tool_name="create_report_artifact",
                tool_version="1.0.0",
                arguments_hash="sha256:" + "a" * 64,
                principal_hash="sha256:" + "u" * 64,
                status="approved",
                approved_by=ids.user_id,
                approved_at=now,
                expires_at=now + timedelta(minutes=10),
                created_at=now,
                thread_id=task_id,
                checkpoint_ns="",
                checkpoint_id="checkpoint-1",
                interrupt_id="interrupt-1",
                tool_call_id="action-1",
                checkpoint_binding_hash="sha256:" + "c" * 64,
            )
        )
        session.add(
            ToolCallRow(
                tool_call_id=f"call-{task_id}",
                task_id=task_id,
                tenant_id=ids.tenant_id,
                approval_id=None,
                plan_revision=None,
                plan_digest=None,
                evidence_digest=None,
                tool_name="search_knowledge",
                tool_version="1.0.0",
                arguments_hash="arguments",
                principal_hash="principal",
                mode="read",
                idempotency_key=None,
                risk_level="low",
                status="failed",
                error_code="fixture_artifact",
                completed_at=now,
                execution_deadline=now + timedelta(minutes=1),
            )
        )
        session.flush()
        session.add(
            ArtifactRow(
                artifact_id=f"artifact-{task_id}",
                task_id=task_id,
                tenant_id=ids.tenant_id,
                tool_call_id=f"call-{task_id}",
                artifact_type="report",
                storage_key=ref,
                content_hash="sha256:" + "a" * 64,
                status="ready",
            )
        )
    repository.finish_task(
        lease,
        status=TaskStatus.FAILED,
        error_code="task_agent_failed",
        run_id="trace",
    )
    return repository, task_id, ref


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_controlled_artifact_purge_uses_postgres_clock_and_preserves_approval(postgres_db) -> None:
    database, objects, auth, ids = postgres_db
    repository, task_id, ref = _controlled_artifact(database, objects, auth, ids)
    with database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE artifacts SET content_expires_at = clock_timestamp() - INTERVAL '1 second' "
                "WHERE tenant_id = :tenant_id AND task_id = :task_id"
            ),
            {"tenant_id": ids.tenant_id, "task_id": task_id},
        )
    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=AgentResultStore(objects),
        orphan_grace_seconds=60,
    )

    assert retention.purge_expired_artifact_content(limit=10) == 1
    assert not _object_exists(objects, ref)
    with database.sessions() as session:
        artifact = session.get(ArtifactRow, f"artifact-{task_id}")
        approval = session.get(ApprovalRow, f"approval-{task_id}")
        task = session.get(AgentTaskRow, task_id)
        assert artifact is not None and artifact.storage_key is None
        assert artifact.content_purged_at is not None
        assert approval is not None
        assert approval.status == "approved"
        assert approval.checkpoint_binding_hash == "sha256:" + "c" * 64
        assert approval.arguments_hash == "sha256:" + "a" * 64
        assert task is not None and task.status == TaskStatus.FAILED.value
    assert retention.purge_expired_artifact_content(limit=10) == 0


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_expired_tool_results_keeps_succeeded_audit(postgres_db) -> None:
    """验收：TTL 仅清正文，工具执行成功与审计事实保持不变。"""
    database, objects, auth, ids = postgres_db
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=auth, goal="研究", task_mode=TaskMode.RESEARCH, task_id=ids.task_id
    )
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "search_knowledge"
    )
    service = ToolExecutionService(
        database,
        ToolRegistry([definition], adapters={"search_knowledge": _OkAdapter()}),
        ApprovalService(database),
        tool_results=objects,
    )
    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=AgentResultStore(objects),
        orphan_grace_seconds=60,
    )
    try:
        first = service.execute(
            ToolExecutionRequest(
                auth=auth,
                task_id=ids.task_id,
                tool_name="search_knowledge",
                arguments={"query": "q"},
                worker_id=lease.worker_id,
                lease_generation=lease.lease_generation,
                idempotency_key="k-1",
            )
        )
        assert first.status == "succeeded" and first.result_ref
        stored_ref: str = first.result_ref
        assert _object_exists(objects, stored_ref)

        with database.sessions.begin() as session:
            session.execute(
                sa_update(ToolCallRow)
                .where(ToolCallRow.tool_call_id == first.tool_call_id)
                .values(result_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )

        assert retention.purge_expired_tool_results(limit=10) == 1
        assert not _object_exists(objects, stored_ref)
        with database.sessions() as session:
            row = session.get(ToolCallRow, first.tool_call_id)
            assert row is not None
            assert row.status == "succeeded"
            assert row.error_code is None
            assert row.result_ref is None
            assert row.result_size is None
            assert row.result_hash == first.result_hash
            assert row.result_expires_at is not None
            assert row.result_purged_at is not None
            assert row.verified_at is not None and row.completed_at is not None

        # 幂等重扫：已清理行不再命中。
        assert retention.purge_expired_tool_results(limit=10) == 0
    finally:
        service.close()


class _CheckpointState(TypedDict):
    value: NotRequired[int]


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_finished_task_content_removes_checkpoint_and_result(postgres_db) -> None:
    """验收：终态超期后删正文与 checkpoint，但 refused 结论保持不变。"""
    database, objects, auth, ids = postgres_db
    checkpointer, connection = open_postgres_checkpointer(POSTGRES_TEST_URL)
    try:
        checkpointer.setup()

        repository = DatabaseTaskRepository(database)
        repository.create_task(
            auth=auth, goal="研究", task_mode=TaskMode.RESEARCH, task_id=ids.task_id
        )
        lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
        assert lease is not None

        graph = StateGraph(_CheckpointState)
        graph.add_node("noop", lambda state: {"value": 1})
        graph.add_edge(START, "noop")
        agent = graph.compile(checkpointer=checkpointer)
        config = {
            "configurable": {
                "thread_id": ids.task_id,
                "checkpoint_tenant_id": ids.tenant_id,
                "checkpoint_worker_id": lease.worker_id,
                "checkpoint_lease_generation": lease.lease_generation,
            }
        }
        agent.invoke({"messages": [HumanMessage("x")]}, config)
        assert agent.get_state(config).values.get("value") == 1

        result: TaskResult = build_task_result(
            TaskResultCandidate(
                status="refused",
                refusal_code="insufficient_evidence",
                stop_reason="no_evidence",
            ),
            task_id=ids.task_id,
            citations=(),
        )
        result_ref, result_hash = AgentResultStore(objects).save_task_result(
            ids.task_id, result
        )
        repository.finish_task(
            lease,
            status=TaskStatus.REFUSED,
            result_kind=result.result_kind,
            result_ref=result_ref,
            result_hash=result_hash,
            run_id="trace",
        )
        assert _object_exists(objects, result_ref)

        # 以持久正文到期时间而非 updated_at 驱动清理。
        with database.sessions.begin() as session:
            task = session.get(AgentTaskRow, ids.task_id)
            assert task is not None
            task.content_expires_at = datetime.now(UTC) - timedelta(seconds=1)

        retention = TaskRetentionService(
            database=database,
            repository=repository,
            objects=objects,
            results=AgentResultStore(objects),
            checkpointer=checkpointer,
            orphan_grace_seconds=60,
        )
        assert retention.purge_finished_task_content(limit=10) == 1

        assert not agent.get_state(config).values
        assert not _object_exists(objects, result_ref)
        with database.sessions() as session:
            row = session.get(AgentTaskRow, ids.task_id)
            assert row is not None
            assert row.checkpoint_purged_at is not None
            assert row.result_purged_at is not None
            assert row.result_ref is None
            assert row.status == TaskStatus.REFUSED.value
            assert row.result_hash == result_hash

        # 幂等重扫不重复清理。
        assert retention.purge_finished_task_content(limit=10) == 0
    finally:
        connection.close()


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_finished_task_content_retries_failed_result_delete(
    postgres_db, monkeypatch
) -> None:
    database, objects, auth, ids = postgres_db
    repository = DatabaseTaskRepository(database)
    lease = _claim_research_task(repository, auth, ids.task_id)
    result = build_task_result(
        TaskResultCandidate(
            status="refused",
            refusal_code="insufficient_evidence",
            stop_reason="no_evidence",
        ),
        task_id=ids.task_id,
        citations=(),
    )
    results = AgentResultStore(objects)
    result_ref, result_hash = results.save_task_result(ids.task_id, result)
    repository.finish_task(
        lease,
        status=TaskStatus.REFUSED,
        result_kind=result.result_kind,
        result_ref=result_ref,
        result_hash=result_hash,
            run_id="trace",
    )
    _expire_task_content(database, tenant_id=ids.tenant_id, task_id=ids.task_id)
    checkpointer = _RecordingCheckpointer()
    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=results,
        checkpointer=checkpointer,
        orphan_grace_seconds=60,
    )
    original_delete = objects.delete_exact

    def fail_delete(_ref: str) -> None:
        raise OSError("object_store_unavailable")

    monkeypatch.setattr(objects, "delete_exact", fail_delete)
    assert retention.purge_finished_task_content(limit=10) == 1
    with database.sessions() as session:
        row = session.get(AgentTaskRow, ids.task_id)
        assert row is not None
        assert row.status == TaskStatus.REFUSED.value
        assert row.result_ref == result_ref
        assert row.result_hash == result_hash
        assert row.result_purged_at is None
        assert row.checkpoint_purged_at is not None
    assert _object_exists(objects, result_ref)
    assert checkpointer.deleted == [ids.task_id]

    monkeypatch.setattr(objects, "delete_exact", original_delete)
    assert retention.purge_finished_task_content(limit=10) == 1
    with database.sessions() as session:
        row = session.get(AgentTaskRow, ids.task_id)
        assert row is not None
        assert row.status == TaskStatus.REFUSED.value
        assert row.result_ref is None
        assert row.result_hash == result_hash
        assert row.result_purged_at is not None
    assert not _object_exists(objects, result_ref)
    assert retention.purge_finished_task_content(limit=10) == 0


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_finished_task_content_marks_failed_without_result_once(postgres_db) -> None:
    database, objects, auth, ids = postgres_db
    repository = DatabaseTaskRepository(database)
    lease = _claim_research_task(repository, auth, ids.task_id)
    repository.finish_task(
        lease,
        status=TaskStatus.FAILED,
        error_code="task_agent_failed",
            run_id="trace",
    )
    _expire_task_content(database, tenant_id=ids.tenant_id, task_id=ids.task_id)
    checkpointer = _RecordingCheckpointer()
    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=AgentResultStore(objects),
        checkpointer=checkpointer,
        orphan_grace_seconds=60,
    )

    assert retention.purge_finished_task_content(limit=10) == 1
    with database.sessions() as session:
        row = session.get(AgentTaskRow, ids.task_id)
        assert row is not None
        assert row.status == TaskStatus.FAILED.value
        assert row.last_error_code == "task_agent_failed"
        assert row.result_ref is None
        assert row.checkpoint_purged_at is not None
        assert row.result_purged_at is not None
        first_purged_at = row.result_purged_at

    assert retention.purge_finished_task_content(limit=10) == 0
    with database.sessions() as session:
        row = session.get(AgentTaskRow, ids.task_id)
        assert row is not None
        assert row.result_purged_at == first_purged_at
    assert checkpointer.deleted == [ids.task_id]


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_orphaned_result_objects_respects_grace_references_and_live_deadlines(
    postgres_db,
) -> None:
    database, objects, auth, ids = postgres_db
    now = datetime(2100, 1, 1, tzinfo=UTC)
    repository = DatabaseTaskRepository(database)
    referenced_task_id = f"{ids.task_id}-referenced"
    referenced_lease = _claim_research_task(repository, auth, referenced_task_id)
    referenced_result = build_task_result(
        TaskResultCandidate(
            status="refused",
            refusal_code="insufficient_evidence",
            stop_reason="no_evidence",
        ),
        task_id=referenced_task_id,
        citations=(),
    )
    referenced_ref, referenced_hash = AgentResultStore(objects).save_task_result(
        referenced_task_id,
        referenced_result,
    )
    repository.finish_task(
        referenced_lease,
        status=TaskStatus.REFUSED,
        result_kind=referenced_result.result_kind,
        result_ref=referenced_ref,
        result_hash=referenced_hash,
            run_id="trace",
    )
    running_task_id = f"{ids.task_id}-running"
    running_lease = _claim_research_task(repository, auth, running_task_id)
    running_tool_call_id = f"running-{ids.task_id}"
    orphan_ref = "task-results/orphan-task/orphan.json"
    fresh_ref = "task-results/fresh-task/fresh.json"
    running_tool_ref = f"tool-results/{running_task_id}/{running_tool_call_id}.json"
    pending_task_ref = f"task-results/{running_task_id}/pending.json"
    for ref in (orphan_ref, fresh_ref, running_tool_ref, pending_task_ref):
        objects.put_exclusive(ref, b"body")
    for ref in (orphan_ref, referenced_ref, running_tool_ref, pending_task_ref):
        _set_mtime(objects, ref, now - timedelta(seconds=61))
    _set_mtime(objects, fresh_ref, now - timedelta(seconds=30))
    with database.sessions.begin() as session:
        database_now = session.scalar(select(func.clock_timestamp()))
        assert isinstance(database_now, datetime)
        session.add(
            ToolCallRow(
                tool_call_id=running_tool_call_id,
                task_id=running_task_id,
                tenant_id=ids.tenant_id,
                approval_id=None,
                plan_revision=None,
                plan_digest=None,
                evidence_digest=None,
                tool_name="search_knowledge",
                tool_version="1.0.0",
                arguments_hash="arguments",
                principal_hash="principal",
                mode="read",
                idempotency_key=None,
                risk_level="low",
                status="running",
                created_at=database_now,
                execution_deadline=database_now + timedelta(seconds=120),
            )
        )
    assert running_lease.state.status is TaskStatus.RUNNING
    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=AgentResultStore(objects),
        orphan_grace_seconds=60,
        now=lambda: now,
    )

    assert retention.purge_orphaned_result_objects(limit=10) == 1
    assert not _object_exists(objects, orphan_ref)
    assert _object_exists(objects, fresh_ref)
    assert _object_exists(objects, referenced_ref)
    assert _object_exists(objects, running_tool_ref)
    assert _object_exists(objects, pending_task_ref)


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_expire_waiting_task_uses_postgres_clock_and_writes_event(postgres_db) -> None:
    database, objects, auth, ids = postgres_db
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=auth, goal="研究", task_mode=TaskMode.RESEARCH, task_id=ids.task_id
    )
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease is not None
    paused = repository.pause_for_user(
        lease,
        question="需要部门",
        checkpoint_ns="",
        checkpoint_id="checkpoint-retention",
        interrupt_id="interrupt-retention",
        run_id="trace",
    )
    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, ids.task_id)
        assert task is not None
        task.deadline = datetime.now(UTC) - timedelta(seconds=1)

    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=AgentResultStore(objects),
        orphan_grace_seconds=60,
    )
    assert retention.expire_waiting_tasks(limit=10) == 1
    assert retention.expire_waiting_tasks(limit=10) == 0
    with database.sessions() as session:
        row = session.get(AgentTaskRow, ids.task_id)
        assert row is not None
        assert row.status == TaskStatus.FAILED.value
        assert row.last_error_code == "task_deadline_expired"
        assert row.interrupt_kind is None and row.interrupt_summary is None
        assert row.lease_owner is None and row.lease_expires_at is None
        assert row.content_expires_at is not None
        events = tuple(
            session.scalars(
                select(TaskEventRow.event_type)
                .where(TaskEventRow.task_id == ids.task_id)
                .order_by(TaskEventRow.sequence)
            )
        )
    assert paused.status is TaskStatus.WAITING_FOR_USER
    assert events == ("task_created", "task_waiting_for_user", "task_failed")
