"""真实 PostgreSQL 的 M1 Research 领取与 lease generation 栅栏。"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    RESEARCH_AGENT_CAPABILITY_NAME,
    RESEARCH_AGENT_PROTOCOL_VERSION,
    TASK_AGENT_CAPABILITY_NAME,
    TASK_AGENT_PROTOCOL_VERSION,
    TaskRepositoryError,
)


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
                    "CAST(:attributes AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant": tenant_id,
                    "user": user_id,
                    "attributes": json.dumps(auth.attributes),
                },
            )
        repository.create_task(
            auth=auth,
            goal="并发恢复测试",
            task_mode=TaskMode.RESEARCH,
            task_id=task_id,
        )

        def claim(index: int):
            return DatabaseTaskRepository(
                database, now=lambda: datetime(2000, 1, 1, tzinfo=UTC)
            ).claim_next(
                worker_id=f"worker-{index}-{suffix}", lease_seconds=30
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            claims = tuple(executor.map(claim, range(4)))
        winners = tuple(item for item in claims if item is not None)
        assert len(winners) == 1
        first = winners[0]
        with database.engine.connect() as connection:
            lease_expires_at, database_now = connection.execute(
                text(
                    "SELECT lease_expires_at, clock_timestamp() FROM agent_tasks "
                    "WHERE tenant_id = :tenant AND task_id = :task"
                ),
                {"tenant": tenant_id, "task": task_id},
            ).one()
        assert lease_expires_at > database_now

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


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_create_task_default_deadline_uses_database_clock() -> None:
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"create-deadline-{suffix}"
    user_id = f"owner-{suffix}"
    task_id = f"task-{suffix}"
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"request-{suffix}",
    )
    repository = DatabaseTaskRepository(
        database,
        now=lambda: datetime(2100, 1, 1, tzinfo=UTC),
    )
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
                    "CAST(:attributes AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant": tenant_id,
                    "user": user_id,
                    "attributes": json.dumps(auth.attributes),
                },
            )
        repository.create_task(
            auth=auth,
            goal="创建 deadline 使用数据库时钟",
            task_mode=TaskMode.RESEARCH,
            task_id=task_id,
        )
        with database.engine.connect() as connection:
            deadline, database_now = connection.execute(
                text(
                    "SELECT deadline, clock_timestamp() FROM agent_tasks "
                    "WHERE tenant_id = :tenant AND task_id = :task"
                ),
                {"tenant": tenant_id, "task": task_id},
            ).one()
        assert database_now < deadline <= database_now + timedelta(seconds=600)
        assert deadline < datetime(2100, 1, 1, tzinfo=UTC)
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
        database.engine.dispose()


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_finish_task_uses_database_clock_for_deadline_terminal_check() -> None:
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"deadline-{suffix}"
    user_id = f"owner-{suffix}"
    task_id = f"task-{suffix}"
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"request-{suffix}",
    )
    repository = DatabaseTaskRepository(
        database,
        now=lambda: datetime(2000, 1, 1, tzinfo=UTC),
    )
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
                    "CAST(:attributes AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant": tenant_id,
                    "user": user_id,
                    "attributes": json.dumps(auth.attributes),
                },
            )
        repository.create_task(
            auth=auth,
            goal="数据库时钟 deadline 终检",
            task_mode=TaskMode.RESEARCH,
            task_id=task_id,
        )
        lease = repository.claim_next(worker_id=f"worker-{suffix}", lease_seconds=60)
        assert lease is not None
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE agent_tasks SET deadline = clock_timestamp() - INTERVAL '1 second' "
                    "WHERE tenant_id = :tenant AND task_id = :task"
                ),
                {"tenant": tenant_id, "task": task_id},
            )

        finished = repository.finish_task(
            lease,
            status=TaskStatus.REFUSED,
            result_kind="research",
            result_ref=f"task-results/{task_id}/result.json",
            result_hash="sha256:" + "a" * 64,
            trace_id="trace",
        )

        assert finished.status is TaskStatus.FAILED
        assert finished.result_ref is None and finished.result_hash is None
        with database.engine.connect() as connection:
            error_code, result_ref = connection.execute(
                text(
                    "SELECT last_error_code, result_ref FROM agent_tasks "
                    "WHERE tenant_id = :tenant AND task_id = :task"
                ),
                {"tenant": tenant_id, "task": task_id},
            ).one()
        assert error_code == "task_deadline_expired"
        assert result_ref is None
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
        database.engine.dispose()


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_research_creation_requires_live_compatible_capability() -> None:
    """同一 PostgreSQL 事务内的 capability 门拒绝缺失、错协议和过期 Worker。"""
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"capability-{suffix}"
    user_id = f"owner-{suffix}"
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"request-{suffix}",
    )
    repository = DatabaseTaskRepository(
        database,
        research_capability_name="research-agent",
        research_capability_protocol_version="m1-fenced-v2",
    )
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
                    "CAST(:attributes AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant": tenant_id,
                    "user": user_id,
                    "attributes": json.dumps(auth.attributes),
                },
            )
        with pytest.raises(TaskRepositoryError, match="research_agent_unavailable"):
            repository.create_task(
                auth=auth,
                goal="无 Worker 不能创建",
                task_mode=TaskMode.RESEARCH,
                task_id=f"task-missing-{suffix}",
            )
        with database.engine.connect() as connection:
            assert connection.execute(
                text("SELECT COUNT(*) FROM agent_tasks WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            ).scalar_one() == 0

        repository.register_capability(
            capability_name="research-agent",
            protocol_version="m1-fenced-v1",
            worker_id=f"worker-old-{suffix}",
            lease_seconds=60,
        )
        with pytest.raises(TaskRepositoryError, match="research_agent_unavailable"):
            repository.create_task(
                auth=auth,
                goal="旧协议不能创建",
                task_mode=TaskMode.RESEARCH,
                task_id=f"task-old-{suffix}",
            )
        repository.register_capability(
            capability_name="research-agent",
            protocol_version="m1-fenced-v2",
            worker_id=f"worker-current-{suffix}",
            lease_seconds=60,
        )
        created = repository.create_task(
            auth=auth,
            goal="兼容 Worker 可以创建",
            task_mode=TaskMode.RESEARCH,
            task_id=f"task-current-{suffix}",
        )
        assert created.task_id == f"task-current-{suffix}"
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE agent_capability_leases "
                    "SET lease_expires_at = clock_timestamp() - INTERVAL '1 second' "
                    "WHERE capability_name = 'research-agent' "
                    "AND protocol_version = 'm1-fenced-v2' AND worker_id = :worker"
                ),
                {"worker": f"worker-current-{suffix}"},
            )
        with pytest.raises(TaskRepositoryError, match="research_agent_unavailable"):
            repository.create_task(
                auth=auth,
                goal="过期 Worker 不能创建",
                task_mode=TaskMode.RESEARCH,
                task_id=f"task-expired-{suffix}",
            )
        with database.engine.connect() as connection:
            assert connection.execute(
                text("SELECT COUNT(*) FROM agent_tasks WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            ).scalar_one() == 1
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
        database.engine.dispose()


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_m2_task_claim_requires_exact_live_worker_protocol() -> None:
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    tenant_id = f"m2-capability-{suffix}"
    user_id = f"owner-{suffix}"
    task_id = f"task-{suffix}"
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"request-{suffix}",
    )
    repository = DatabaseTaskRepository(
        database,
        task_capability_name=TASK_AGENT_CAPABILITY_NAME,
        task_capability_protocol_version=TASK_AGENT_PROTOCOL_VERSION,
    )
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
                    "CAST(:attributes AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant": tenant_id,
                    "user": user_id,
                    "attributes": json.dumps(auth.attributes),
                },
            )
        repository.register_capability(
            capability_name=TASK_AGENT_CAPABILITY_NAME,
            protocol_version=TASK_AGENT_PROTOCOL_VERSION,
            worker_id=f"worker-m2-{suffix}",
            lease_seconds=60,
        )
        state = repository.create_task(
            auth=auth,
            goal="M2 协议领取",
            task_mode=TaskMode.RESEARCH,
            task_id=task_id,
        )
        assert state.required_capability_name == TASK_AGENT_CAPABILITY_NAME
        assert state.required_protocol_version == TASK_AGENT_PROTOCOL_VERSION

        repository.register_capability(
            capability_name=RESEARCH_AGENT_CAPABILITY_NAME,
            protocol_version=RESEARCH_AGENT_PROTOCOL_VERSION,
            worker_id=f"worker-m1-{suffix}",
            lease_seconds=60,
        )
        repository.register_capability(
            capability_name="wrong-task-agent",
            protocol_version=TASK_AGENT_PROTOCOL_VERSION,
            worker_id=f"worker-wrong-{suffix}",
            lease_seconds=60,
        )
        assert repository.claim_next(
            worker_id=f"worker-m1-{suffix}",
            lease_seconds=30,
            capability_name=RESEARCH_AGENT_CAPABILITY_NAME,
            protocol_version=RESEARCH_AGENT_PROTOCOL_VERSION,
        ) is None
        assert repository.claim_next(
            worker_id=f"worker-wrong-{suffix}",
            lease_seconds=30,
            capability_name="wrong-task-agent",
            protocol_version=TASK_AGENT_PROTOCOL_VERSION,
        ) is None

        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE agent_capability_leases "
                    "SET lease_expires_at = clock_timestamp() - INTERVAL '1 second' "
                    "WHERE capability_name = :capability AND protocol_version = :protocol "
                    "AND worker_id = :worker"
                ),
                {
                    "capability": TASK_AGENT_CAPABILITY_NAME,
                    "protocol": TASK_AGENT_PROTOCOL_VERSION,
                    "worker": f"worker-m2-{suffix}",
                },
            )
        assert repository.claim_next(
            worker_id=f"worker-m2-{suffix}",
            lease_seconds=30,
            capability_name=TASK_AGENT_CAPABILITY_NAME,
            protocol_version=TASK_AGENT_PROTOCOL_VERSION,
        ) is None

        repository.register_capability(
            capability_name=TASK_AGENT_CAPABILITY_NAME,
            protocol_version=TASK_AGENT_PROTOCOL_VERSION,
            worker_id=f"worker-current-{suffix}",
            lease_seconds=60,
        )
        lease = repository.claim_next(
            worker_id=f"worker-current-{suffix}",
            lease_seconds=30,
            capability_name=TASK_AGENT_CAPABILITY_NAME,
            protocol_version=TASK_AGENT_PROTOCOL_VERSION,
        )
        assert lease is not None
        assert lease.state.required_capability_name == TASK_AGENT_CAPABILITY_NAME
        assert lease.state.required_protocol_version == TASK_AGENT_PROTOCOL_VERSION
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
        database.engine.dispose()
