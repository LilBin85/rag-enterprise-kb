"""M1 批 3 隔离 PostgreSQL 集成：工具结果 TTL 清理与终态任务线程清理。

运行方式：临时数据库经 docker compose createdb/dropdb 创建删除，
RAG_KB_TEST_POSTGRES_URL 指向该库；不触碰持久 rag_kb 数据库。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import NotRequired, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from sqlalchemy import update as sa_update

from rag_kb.agent.checkpointer import open_postgres_checkpointer
from rag_kb.agent.result import (
    AgentResultStore,
    TaskResult,
    TaskResultCandidate,
    build_task_result,
)
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_worker import TaskWorker
from rag_kb.application.tool_execution_service import (
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import AgentTaskRow, PrincipalRow, TenantRow, ToolCallRow
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.object_store.local import LocalObjectStore


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )


class _OkAdapter:
    def execute(self, _arguments, _context):
        return {"result_reference": "ref", "evidence_ids": [], "items": []}

    def verify(self, _result, _context) -> bool:
        return True


@pytest.fixture()
def postgres_db(tmp_path: Path):
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    Base.metadata.drop_all(database.engine)
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
    yield database, LocalObjectStore(tmp_path / "objects")
    database.engine.dispose()


def _object_exists(objects: LocalObjectStore, ref: str) -> bool:
    try:
        objects.read(ref)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_expired_tool_results_deletes_objects_and_nulls_refs(postgres_db) -> None:
    """验收②④：删对象、置 result_ref=NULL、保留 hash，且重扫不重复清理。"""
    database, objects = postgres_db
    repository = DatabaseTaskRepository(database)
    repository.create_task(auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a")
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
    try:
        first = service.execute(
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
        assert first.status == "succeeded" and first.result_ref
        stored_ref: str = first.result_ref
        assert _object_exists(objects, stored_ref)

        with database.sessions.begin() as session:
            session.execute(
                sa_update(ToolCallRow)
                .where(ToolCallRow.tool_call_id == first.tool_call_id)
                .values(result_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )

        assert service.purge_expired_tool_results(limit=10) == 1
        assert not _object_exists(objects, stored_ref)
        with database.sessions() as session:
            row = session.get(ToolCallRow, first.tool_call_id)
            assert row is not None
            assert row.status == "failed"
            assert row.error_code == "tool_result_expired"
            assert row.result_ref is None
            assert row.result_hash == first.result_hash

        # 幂等重扫：已清理行不再命中。
        assert service.purge_expired_tool_results(limit=10) == 0
    finally:
        service.close()


class _CheckpointState(TypedDict):
    value: NotRequired[int]


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_purge_finished_task_threads_removes_checkpoint_and_result(postgres_db) -> None:
    """验收③⑤：终态超期任务删除 thread checkpoint 与结果对象并写入标记。"""
    database, objects = postgres_db
    checkpointer, connection = open_postgres_checkpointer(POSTGRES_TEST_URL)
    try:
        checkpointer.setup()

        repository = DatabaseTaskRepository(database)
        repository.create_task(
            auth=_auth(), goal="研究", task_mode=TaskMode.RESEARCH, task_id="task-a"
        )
        lease = repository.claim_next(worker_id="worker-a", lease_seconds=60)
        assert lease is not None

        graph = StateGraph(_CheckpointState)
        graph.add_node("noop", lambda state: {"value": 1})
        graph.add_edge(START, "noop")
        agent = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "task-a"}}
        agent.invoke({"messages": [HumanMessage("x")]}, config)
        assert agent.get_state(config).values.get("value") == 1

        result: TaskResult = build_task_result(
            TaskResultCandidate(
                status="refused",
                refusal_code="insufficient_evidence",
                stop_reason="no_evidence",
            ),
            task_id="task-a",
            citations=(),
        )
        result_ref, result_hash = AgentResultStore(objects).save_task_result(
            "task-a", result
        )
        repository.finish_task(
            lease,
            status=TaskStatus.REFUSED,
            result_kind=result.result_kind,
            result_ref=result_ref,
            result_hash=result_hash,
            trace_id="trace",
        )
        assert _object_exists(objects, result_ref)

        # 超过保留期（默认 7 天）。
        with database.sessions.begin() as session:
            task = session.get(AgentTaskRow, "task-a")
            assert task is not None
            task.updated_at = datetime.now(UTC) - timedelta(days=30)

        worker = TaskWorker(
            repository=repository,
            auth_contexts=SimpleNamespace(resolve=lambda **k: _auth()),
            agent=None,
            results=AgentResultStore(objects),
            evidence=None,
            tools=None,
            database=database,
            worker_id="worker-a",
            checkpointer=checkpointer,
        )
        assert worker.purge_finished_task_threads(limit=10) == 1

        assert not agent.get_state(config).values
        assert not _object_exists(objects, result_ref)
        with database.sessions() as session:
            row = session.get(AgentTaskRow, "task-a")
            assert row is not None
            assert row.checkpoint_purged_at is not None
            assert row.result_purged_at is not None
            assert row.result_ref is None
            assert row.status == TaskStatus.REFUSED.value
            assert row.result_hash == result_hash

        # 幂等重扫不重复清理。
        assert worker.purge_finished_task_threads(limit=10) == 0
    finally:
        connection.close()
