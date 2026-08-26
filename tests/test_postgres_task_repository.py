"""真实 PostgreSQL 的 M1 Research 领取与 lease generation 栅栏。"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import text

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_concurrent_claim_and_expired_lease_fencing() -> None:
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"task-agent-{suffix}"
    user_id = f"owner-{suffix}"
    task_id = f"task-{suffix}"
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"request-{suffix}",
    )
    repository = DatabaseTaskRepository(database)
    try:
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO tenants (tenant_id, name, active, created_at) "
                    "VALUES (:tenant, :name, TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant": tenant_id, "name": tenant_id},
            )
            connection.execute(
                text(
                    "INSERT INTO principals "
                    "(tenant_id, user_id, roles, attributes, active, created_at) "
                    "VALUES (:tenant, :user, CAST('[]' AS jsonb), "
                    "CAST('{}' AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant": tenant_id, "user": user_id},
            )
        repository.create_task(
            auth=auth,
            goal="并发恢复测试",
            task_mode=TaskMode.RESEARCH,
            task_id=task_id,
        )

        def claim(index: int):
            return DatabaseTaskRepository(database).claim_next(
                worker_id=f"worker-{index}-{suffix}", lease_seconds=30
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            claims = tuple(executor.map(claim, range(4)))
        winners = tuple(item for item in claims if item is not None)
        assert len(winners) == 1
        first = winners[0]

        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE agent_tasks SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' "
                    "WHERE tenant_id = :tenant AND task_id = :task"
                ),
                {"tenant": tenant_id, "task": task_id},
            )
        second = repository.claim_next(worker_id=f"recovery-{suffix}", lease_seconds=30)
        assert second is not None
        assert second.lease_generation == first.lease_generation + 1
        with pytest.raises(TaskRepositoryError, match="task_lease_generation_stale"):
            repository.refresh_lease(first)
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
        database.engine.dispose()
