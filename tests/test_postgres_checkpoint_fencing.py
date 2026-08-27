"""M1 批次 1：真实 PostgreSQL checkpoint generation fencing。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Thread
import time
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import text

from rag_kb.agent.checkpointer import TaskCheckpointError, open_postgres_checkpointer
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import AgentTaskRow, PrincipalRow, TenantRow
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


@dataclass(frozen=True)
class _Ids:
    tenant_id: str
    user_id: str
    task_id: str


def _auth(ids: _Ids) -> AuthContext:
    return AuthContext(
        tenant_id=ids.tenant_id,
        user_id=ids.user_id,
        attributes={"permissions": ["task:create", "task:read"]},
        request_id=f"checkpoint-{ids.task_id}",
    )


@pytest.fixture()
def fenced_postgres():
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    suffix = uuid4().hex
    ids = _Ids(
        tenant_id=f"checkpoint-tenant-{suffix}",
        user_id=f"checkpoint-owner-{suffix}",
        task_id=f"checkpoint-task-{suffix}",
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
        auth=_auth(ids),
        goal="验证 checkpoint generation fencing",
        task_mode=TaskMode.RESEARCH,
        task_id=ids.task_id,
    )
    saver, connection = open_postgres_checkpointer(POSTGRES_TEST_URL)
    saver.setup()
    try:
        yield database, repository, saver, ids
    finally:
        connection.close()
        with database.engine.begin() as engine_connection:
            for table_name in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                engine_connection.execute(
                    text(f"DELETE FROM {table_name} WHERE thread_id = :task"),
                    {"task": ids.task_id},
                )
            engine_connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                {"tenant": ids.tenant_id},
            )
        database.engine.dispose()


def _config(ids: _Ids, lease) -> dict[str, object]:
    return {
        "configurable": {
            "thread_id": ids.task_id,
            "checkpoint_ns": "",
            "checkpoint_tenant_id": ids.tenant_id,
            "checkpoint_worker_id": lease.worker_id,
            "checkpoint_lease_generation": lease.lease_generation,
        }
    }


def _checkpoint(message: str):
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["message"] = message
    return checkpoint


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_preempted_generation_cannot_write_checkpoint_or_pending_writes(fenced_postgres) -> None:
    """抢占后旧 generation 的 put/put_writes 都是零新增，B 从合法 checkpoint 恢复。"""
    database, repository, saver, ids = fenced_postgres
    lease_a = repository.claim_next(worker_id="checkpoint-worker-a", lease_seconds=60)
    assert lease_a is not None
    config_a = _config(ids, lease_a)
    legal_config = saver.put(
        config_a,
        _checkpoint("legal-before-preemption"),
        {"source": "input", "step": -1, "writes": {}},
        {},
    )
    assert legal_config["configurable"] == {
        **config_a["configurable"],
        "checkpoint_id": legal_config["configurable"]["checkpoint_id"],
    }
    # 第二次 put 模拟跨 superstep，返回 config 仍含所有服务端 fencing 字段。
    legal_config = saver.put(
        legal_config,
        _checkpoint("legal-before-preemption"),
        {"source": "loop", "step": 0, "writes": {}},
        {},
    )
    saver.put_writes(legal_config, [("message", "legal-write")], "legal-task")

    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, ids.task_id)
        assert task is not None
        task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    lease_b = repository.claim_next(worker_id="checkpoint-worker-b", lease_seconds=60)
    assert lease_b is not None
    assert lease_b.lease_generation == lease_a.lease_generation + 1

    with database.engine.connect() as engine_connection:
        before_checkpoints = engine_connection.execute(
            text("SELECT COUNT(*) FROM checkpoints WHERE thread_id = :task"),
            {"task": ids.task_id},
        ).scalar_one()
        before_writes = engine_connection.execute(
            text("SELECT COUNT(*) FROM checkpoint_writes WHERE thread_id = :task"),
            {"task": ids.task_id},
        ).scalar_one()
    with pytest.raises(TaskCheckpointError, match="task_checkpoint_lease_stale"):
        saver.put(
            legal_config,
            _checkpoint("stale-after-preemption"),
            {"source": "loop", "step": 1, "writes": {}},
            {},
        )
    with pytest.raises(TaskCheckpointError, match="task_checkpoint_lease_stale"):
        saver.put_writes(legal_config, [("message", "stale-write")], "stale-task")
    with database.engine.connect() as engine_connection:
        assert engine_connection.execute(
            text("SELECT COUNT(*) FROM checkpoints WHERE thread_id = :task"),
            {"task": ids.task_id},
        ).scalar_one() == before_checkpoints
        assert engine_connection.execute(
            text("SELECT COUNT(*) FROM checkpoint_writes WHERE thread_id = :task"),
            {"task": ids.task_id},
        ).scalar_one() == before_writes

    recovered = saver.get_tuple(_config(ids, lease_b))
    assert recovered is not None
    assert recovered.checkpoint["channel_values"]["message"] == "legal-before-preemption"


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_task_row_lock_orders_legal_checkpoint_before_next_generation(
    fenced_postgres, monkeypatch
) -> None:
    """A 持锁时 B 不得递增 generation；skip-locked 重试只能发生在 A 合法提交后。"""
    database, repository, saver, ids = fenced_postgres
    lease_a = repository.claim_next(worker_id="checkpoint-worker-a", lease_seconds=1)
    assert lease_a is not None
    entered_write = Event()
    release_write = Event()
    claimed_b = Event()
    output: dict[str, object] = {}
    original_put = saver._saver.put

    def delayed_put(*args, **kwargs):
        entered_write.set()
        assert release_write.wait(timeout=5)
        return original_put(*args, **kwargs)

    monkeypatch.setattr(saver._saver, "put", delayed_put)

    def write_a() -> None:
        output["saved"] = saver.put(
            _config(ids, lease_a),
            _checkpoint("legal-before-claim-b"),
            {"source": "input", "step": -1, "writes": {}},
            {},
        )

    def claim_b() -> None:
        output["lease_b"] = repository.claim_next(
            worker_id="checkpoint-worker-b", lease_seconds=60
        )
        claimed_b.set()

    writer = Thread(target=write_a)
    writer.start()
    assert entered_write.wait(timeout=5)
    time.sleep(1.1)
    claimer = Thread(target=claim_b)
    claimer.start()
    assert claimed_b.wait(timeout=5)
    assert output["lease_b"] is None
    release_write.set()
    writer.join(timeout=5)
    claimer.join(timeout=5)
    assert not writer.is_alive()
    assert not claimer.is_alive()
    lease_b = repository.claim_next(worker_id="checkpoint-worker-b", lease_seconds=60)
    assert lease_b is not None
    assert lease_b.lease_generation == lease_a.lease_generation + 1
    recovered = saver.get_tuple(_config(ids, lease_b))
    assert recovered is not None
    assert recovered.checkpoint["channel_values"]["message"] == "legal-before-claim-b"
