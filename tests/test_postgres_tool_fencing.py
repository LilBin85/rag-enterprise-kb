"""M1 批 1 隔离 PostgreSQL 集成：双 Worker 租约抢占与执行中撤权。

运行方式：先经 docker compose 创建临时数据库（用后即删），再以
RAG_KB_TEST_POSTGRES_URL 指向该库执行；不触碰持久 rag_kb 数据库。
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

import pytest
from sqlalchemy import select

from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.database import Base, build_database
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
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.object_store.local import LocalObjectStore


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


class ReadAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _arguments, _context):
        self.calls += 1
        return {"result_reference": "server-only", "evidence_ids": [], "items": []}

    def verify(self, _result, _context) -> bool:
        return True


@dataclass(frozen=True)
class _Ids:
    tenant_id: str
    user_id: str
    task_id: str


def _auth(ids: _Ids) -> AuthContext:
    return AuthContext(
        tenant_id=ids.tenant_id,
        user_id=ids.user_id,
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )


@pytest.fixture()
def postgres_research(tmp_path: Path):
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    # 临时库可能残留先前运行的过期任务；每次全量重建，保证领取确定性。
    Base.metadata.drop_all(database.engine)
    Base.metadata.create_all(database.engine)
    ids = _Ids(
        tenant_id=f"tenant-{uuid.uuid4().hex[:12]}",
        user_id=f"owner-{uuid.uuid4().hex[:12]}",
        task_id=f"task-{uuid.uuid4().hex[:12]}",
    )
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id=ids.tenant_id, name=ids.tenant_id))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id=ids.tenant_id,
                user_id=ids.user_id,
                roles=[],
                attributes=_auth(ids).attributes,
            )
        )
    repository = DatabaseTaskRepository(database)
    repository.create_task(
        auth=_auth(ids), goal="研究", task_mode=TaskMode.RESEARCH, task_id=ids.task_id
    )
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
    yield database, repository, service, adapter, ids
    service.close()
    database.engine.dispose()


def _request(ids: _Ids, lease, key: str) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        auth=_auth(ids),
        task_id=ids.task_id,
        tool_name="search_knowledge",
        arguments={"query": "采购制度"},
        worker_id=lease.worker_id,
        lease_generation=lease.lease_generation,
        idempotency_key=key,
    )


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_dual_worker_preemption_fences_stale_tool_calls(postgres_research) -> None:
    """验收①：新 Worker 抢占租约后，旧 Worker 的工具调用（含幂等回读）被拒。"""
    database, repository, service, adapter, ids = postgres_research
    lease_a = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease_a is not None
    first = service.execute(_request(ids, lease_a, "call-1"))
    assert first.status == "succeeded"

    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, ids.task_id)
        assert task is not None
        task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    lease_b = repository.claim_next(worker_id="worker-b", lease_seconds=60)
    assert lease_b is not None
    assert lease_b.lease_generation == lease_a.lease_generation + 1

    with pytest.raises(ToolExecutionError, match="tool_task_lease_stale"):
        service.execute(_request(ids, lease_a, "call-2"))
    with pytest.raises(ToolExecutionError, match="tool_task_lease_stale"):
        service.execute(_request(ids, lease_a, "call-1"))

    assert adapter.calls == 1
    with database.sessions() as session:
        rows = tuple(
            session.scalars(select(ToolCallRow).where(ToolCallRow.task_id == ids.task_id))
        )
        assert len(rows) == 1

    replay = service.execute(_request(ids, lease_b, "call-1"))
    assert replay.duplicate is True and replay.status == "succeeded"
    assert service.execute(_request(ids, lease_b, "call-2")).status == "succeeded"
    assert adapter.calls == 2


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_role_revocation_mid_task_fails_next_tool_call_closed(postgres_research) -> None:
    """修复 3 验收①②：身份未变时正常执行；撤销角色后下一次调用致命失败关闭。"""
    database, repository, service, adapter, ids = postgres_research
    lease_a = repository.claim_next(worker_id="worker-a", lease_seconds=60)
    assert lease_a is not None
    assert service.execute(_request(ids, lease_a, "call-1")).status == "succeeded"

    with database.sessions.begin() as session:
        principal = session.get(
            PrincipalRow, {"tenant_id": ids.tenant_id, "user_id": ids.user_id}
        )
        assert principal is not None
        principal.roles = ["revoked-away"]

    with pytest.raises(ToolExecutionError, match="task_auth_context_changed"):
        service.execute(_request(ids, lease_a, "call-2"))
    assert adapter.calls == 1
    with database.sessions() as session:
        rows = tuple(
            session.scalars(select(ToolCallRow).where(ToolCallRow.task_id == ids.task_id))
        )
        assert len(rows) == 1

    with pytest.raises(TaskExecutionContextError, match="task_auth_context_changed"):
        DatabaseTaskAuthContextPort(database).resolve(
            task_id=ids.task_id,
            auth_context_ref=lease_a.state.auth_context_ref,
            request_id="reclaim",
        )
