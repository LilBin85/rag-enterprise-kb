from datetime import UTC, datetime
import hashlib
from threading import Event
from types import SimpleNamespace

import pytest

from rag_kb import worker as worker_module
from rag_kb.application.upload_reconciler import ReconciliationResult, UploadReservationReconciler
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.object_store import LocalObjectStore


class QueueWorker:
    def __init__(self, name: str, events: list[str], *, status: str) -> None:
        self.name = name
        self.events = events
        self.status = status
        self.sweeps: list[int] = []
        self.close_calls = 0

    def run_once(self):
        self.events.append(self.name)
        return SimpleNamespace(
            status=self.status,
            task_id=None,
            job_id=None,
            document_id=None,
            error_code=None,
        )

    def expire_stale_tool_calls(self, *, limit: int) -> int:
        self.sweeps.append(limit)
        return 1

    def expire_stale_knowledge_queries(self, *, limit: int) -> int:
        self.sweeps.append(limit)
        return 1

    def close(self) -> None:
        self.close_calls += 1


class RetentionWorker:
    def __init__(self) -> None:
        self.sweeps: list[int] = []
        self.close_calls = 0

    def expire_waiting_tasks(self, *, limit: int) -> int:
        self.sweeps.append(limit)
        return 0

    def purge_expired_tool_results(self, *, limit: int) -> int:
        self.sweeps.append(limit)
        return 0

    def purge_finished_task_content(self, *, limit: int) -> int:
        self.sweeps.append(limit)
        return 0

    def purge_orphaned_result_objects(self, *, limit: int) -> int:
        self.sweeps.append(limit)
        return 0

    def close(self) -> None:
        self.close_calls += 1


class UploadReconciler:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reconcile(self, **kwargs: object) -> ReconciliationResult:
        self.calls.append(kwargs)
        return ReconciliationResult(1, 1, 0, 0)


def _patch_runtime(monkeypatch, ingestion, projection, upload, task=None, retention=None):
    monkeypatch.setattr(worker_module, "build_workers", lambda: (ingestion, projection))
    monkeypatch.setattr(worker_module, "build_upload_reconciler", lambda: upload)
    if task is not None:
        monkeypatch.setattr(worker_module, "build_task_worker", lambda **_kwargs: task)
        retention = retention or RetentionWorker()
        monkeypatch.setattr(
            worker_module,
            "build_task_retention",
            lambda: retention,
        )
    return retention


def test_task_backlog_cannot_starve_ingestion_and_max_jobs_still_stops(monkeypatch) -> None:
    events: list[str] = []
    task = QueueWorker("task", events, status="progressed")
    ingestion = QueueWorker("ingestion", events, status="succeeded")
    projection = QueueWorker("projection", events, status="idle")
    retention = RetentionWorker()
    _patch_runtime(monkeypatch, ingestion, projection, UploadReconciler(), task, retention)

    assert worker_module.run(watch=False, poll_seconds=0.01, max_jobs=2, include_tasks=True) == 0
    assert events == ["task", "ingestion"]
    assert task.close_calls == 1
    assert retention.close_calls == 1


def test_task_worker_closes_when_queue_is_interrupted(monkeypatch) -> None:
    events: list[str] = []

    class InterruptedTaskWorker(QueueWorker):
        def run_once(self):
            super().run_once()
            raise KeyboardInterrupt

    task = InterruptedTaskWorker("task", events, status="progressed")
    ingestion = QueueWorker("ingestion", events, status="idle")
    projection = QueueWorker("projection", events, status="idle")
    retention = RetentionWorker()
    _patch_runtime(monkeypatch, ingestion, projection, UploadReconciler(), task, retention)

    with pytest.raises(KeyboardInterrupt):
        worker_module.run(watch=False, poll_seconds=0.01, max_jobs=2, include_tasks=True)
    assert task.close_calls == 1
    assert retention.close_calls == 1


def test_production_loop_runs_bounded_upload_and_tool_recovery(monkeypatch) -> None:
    events: list[str] = []
    task = QueueWorker("task", events, status="idle")
    ingestion = QueueWorker("ingestion", events, status="idle")
    projection = QueueWorker("projection", events, status="idle")
    upload = UploadReconciler()
    retention = RetentionWorker()
    _patch_runtime(monkeypatch, ingestion, projection, upload, task, retention)

    assert worker_module.run(watch=False, poll_seconds=0.01, max_jobs=10, include_tasks=True) == 0
    assert task.sweeps == [worker_module._RECOVERY_BATCH_SIZE] * 2
    assert retention.sweeps == [worker_module._RECOVERY_BATCH_SIZE] * 4
    assert upload.calls[0]["limit"] == worker_module._RECOVERY_BATCH_SIZE
    assert upload.calls[0]["time_budget_seconds"] == worker_module._RECOVERY_TIME_BUDGET_SECONDS
    assert events == ["task", "ingestion", "projection"]


def test_missing_agent_configuration_fails_fast(monkeypatch) -> None:
    """修复 2：Agent 未配置时 --include-tasks 直接失败，不再静默降级。"""
    events: list[str] = []
    ingestion = QueueWorker("ingestion", events, status="idle")
    projection = QueueWorker("projection", events, status="idle")
    _patch_runtime(monkeypatch, ingestion, projection, UploadReconciler())

    def unconfigured(**_kwargs):
        raise worker_module.TaskAgentConfigurationError("task_agent_api_key_missing")

    monkeypatch.setattr(worker_module, "build_task_worker", unconfigured)
    with pytest.raises(worker_module.TaskAgentConfigurationError):
        worker_module.run(watch=False, poll_seconds=0.01, max_jobs=1, include_tasks=True)
    assert events == []


def test_research_capability_heartbeat_registers_renews_and_unregisters() -> None:
    renewed = Event()

    class CapabilityRepository:
        def __init__(self) -> None:
            self.registered: list[dict[str, object]] = []
            self.renewed: list[dict[str, object]] = []
            self.unregistered: list[dict[str, object]] = []

        def register_capability(self, **kwargs: object) -> None:
            self.registered.append(kwargs)

        def renew_capability(self, **kwargs: object) -> None:
            self.renewed.append(kwargs)
            renewed.set()

        def unregister_capability(self, **kwargs: object) -> None:
            self.unregistered.append(kwargs)

    repository = CapabilityRepository()
    heartbeat = worker_module._CapabilityLeaseHeartbeat(
        repository, worker_id="worker-a", lease_seconds=1
    )
    heartbeat.start()
    assert renewed.wait(timeout=2)
    heartbeat.close()

    assert repository.registered == [
        {
            "capability_name": "research-agent",
            "protocol_version": "m1-fenced-v2",
            "worker_id": "worker-a",
            "lease_seconds": 1,
        }
    ]
    assert repository.renewed
    assert repository.unregistered == [
        {
            "capability_name": "research-agent",
            "protocol_version": "m1-fenced-v2",
            "worker_id": "worker-a",
        }
    ]


def test_watch_sleeps_only_after_every_queue_is_idle(monkeypatch) -> None:
    events: list[str] = []
    ingestion = QueueWorker("ingestion", events, status="idle")
    projection = QueueWorker("projection", events, status="idle")
    _patch_runtime(monkeypatch, ingestion, projection, UploadReconciler())

    def stop_after_sleep(seconds: float) -> None:
        assert seconds == 0.25
        raise RuntimeError("stop_watch")

    monkeypatch.setattr(worker_module.time, "sleep", stop_after_sleep)
    with pytest.raises(RuntimeError, match="stop_watch"):
        worker_module.run(watch=True, poll_seconds=0.25, max_jobs=1)
    assert events == ["ingestion", "projection"]


def test_upload_recovery_skips_one_bad_record_and_continues(tmp_path) -> None:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'worker.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    store = LocalObjectStore(tmp_path / "objects")
    content = b"safe-content"
    content_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    for key in ("bad", "good"):
        store.put_exclusive(key, content)

    class Repository:
        def __init__(self) -> None:
            self.marked: list[str] = []

        def list_pending_object_reservations(self, _session, **_kwargs):
            return tuple(
                SimpleNamespace(
                    tenant_id="tenant-a",
                    user_id="user-a",
                    idempotency_key=key,
                    request_hash="sha256:request",
                    source_object_key=key,
                    content_hash=content_hash,
                )
                for key in ("bad", "good")
            )

        def mark_object_ready(self, _session, *, idempotency_key: str, **_kwargs) -> None:
            if idempotency_key == "bad":
                raise RuntimeError("broken_record")
            self.marked.append(idempotency_key)

    repository = Repository()
    result = UploadReservationReconciler(database, store, repository).reconcile(
        older_than=datetime.now(UTC),
        limit=2,
        time_budget_seconds=1,
    )

    assert result.scanned == 2
    assert result.recovered == 1
    assert result.failed == 1
    assert repository.marked == ["good"]


def test_upload_recovery_rotates_bad_full_batch(tmp_path) -> None:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'rotation.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    store = LocalObjectStore(tmp_path / "rotation-objects")
    content = b"safe-content"
    content_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    store.put_exclusive("good", content)

    class Repository:
        def __init__(self) -> None:
            self.keys = ["bad-1", "bad-2", "good"]
            self.marked: list[str] = []

        def list_pending_object_reservations(self, _session, *, limit, **_kwargs):
            return tuple(
                SimpleNamespace(
                    tenant_id="tenant-a",
                    user_id="user-a",
                    idempotency_key=key,
                    request_hash="sha256:request",
                    source_object_key=key,
                    content_hash=content_hash,
                )
                for key in self.keys[:limit]
            )

        def defer_pending_object_reservation(self, _session, *, idempotency_key, **_kwargs):
            self.keys.remove(idempotency_key)
            self.keys.append(idempotency_key)

        def mark_object_ready(self, _session, *, idempotency_key, **_kwargs):
            self.marked.append(idempotency_key)
            self.keys.remove(idempotency_key)

    repository = Repository()
    reconciler = UploadReservationReconciler(database, store, repository)
    first = reconciler.reconcile(older_than=datetime.now(UTC), limit=2)
    second = reconciler.reconcile(older_than=datetime.now(UTC), limit=2)

    assert first.missing_objects == 2
    assert second.recovered == 1
    assert repository.marked == ["good"]
