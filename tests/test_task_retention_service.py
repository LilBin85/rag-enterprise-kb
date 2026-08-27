"""批次 3 正文生命周期的 SQLite 回归。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path, PurePosixPath

import pytest

import rag_kb.application.task_retention_service as task_retention_module
from rag_kb.agent.result import AgentResultStore, TaskResultCandidate, build_task_result
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_retention_service import TaskRetentionService
from rag_kb.application.tool_execution_service import (
    ToolExecutionRequest,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry, builtin_tool_definitions
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    PrincipalRow,
    TenantRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.object_store.local import LocalObjectStore


@dataclass
class Clock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


class Checkpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_thread(self, task_id: str) -> None:
        self.deleted.append(task_id)


class OkAdapter:
    def execute(self, _arguments, _context):
        return {"result_reference": "ref", "evidence_ids": [], "items": []}

    def verify(self, _result, _context) -> bool:
        return True


@pytest.fixture()
def runtime(tmp_path: Path):
    database = build_database(
        f"sqlite+pysqlite:///{(tmp_path / 'retention.sqlite3').as_posix()}"
    )
    Base.metadata.create_all(database.engine)
    clock = Clock(datetime(2026, 8, 25, 10, 0, tzinfo=UTC))
    auth = AuthContext(
        tenant_id="tenant-a",
        user_id="user-a",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="request-a",
    )
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id=auth.tenant_id, name="tenant-a"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
                roles=[],
                attributes=auth.attributes,
            )
        )
    objects = LocalObjectStore(tmp_path / "objects")
    repository = DatabaseTaskRepository(database, now=clock)
    checkpointer = Checkpointer()
    retention = TaskRetentionService(
        database=database,
        repository=repository,
        objects=objects,
        results=AgentResultStore(objects),
        checkpointer=checkpointer,
        orphan_grace_seconds=60,
        now=clock,
    )
    return database, clock, auth, objects, repository, checkpointer, retention


def _claim(runtime, task_id: str):
    _database, _clock, auth, _objects, repository, _checkpointer, _retention = runtime
    repository.create_task(
        auth=auth,
        goal="研究制度",
        task_mode=TaskMode.RESEARCH,
        task_id=task_id,
    )
    lease = repository.claim_next(worker_id="worker-a", lease_seconds=120)
    assert lease is not None
    return lease


def _execute_tool(runtime, task_id: str):
    database, clock, auth, objects, _repository, _checkpointer, _retention = runtime
    lease = _claim(runtime, task_id)
    definition = next(
        item for item in builtin_tool_definitions() if item.name == "search_knowledge"
    )
    service = ToolExecutionService(
        database,
        ToolRegistry([definition], adapters={"search_knowledge": OkAdapter()}),
        ApprovalService(database),
        tool_results=objects,
        now=clock,
    )
    result = service.execute(
        ToolExecutionRequest(
            auth=auth,
            task_id=task_id,
            tool_name="search_knowledge",
            arguments={"query": "q"},
            worker_id=lease.worker_id,
            lease_generation=lease.lease_generation,
            idempotency_key=f"key-{task_id}",
        )
    )
    assert result.status == "succeeded" and result.result_ref is not None
    return service, result, lease


def _set_mtime(objects: LocalObjectStore, ref: str, when: datetime) -> None:
    path = objects.root.joinpath(*PurePosixPath(ref).parts)
    os.utime(path, (when.timestamp(), when.timestamp()))


def test_tool_result_gc_preserves_success_audit_and_retries_delete_failure(runtime, monkeypatch) -> None:
    database, clock, _auth, objects, _repository, _checkpointer, retention = runtime
    executor, result, lease = _execute_tool(runtime, "tool-task")
    try:
        assert result.result_ref is not None
        expired_at = clock.value - timedelta(seconds=1)
        with database.sessions.begin() as session:
            row = session.get(ToolCallRow, result.tool_call_id)
            assert row is not None
            row.result_expires_at = expired_at

        replayed = executor.execute(
            ToolExecutionRequest(
                auth=_auth,
                task_id="tool-task",
                tool_name="search_knowledge",
                arguments={"query": "q"},
                worker_id=lease.worker_id,
                lease_generation=lease.lease_generation,
                idempotency_key="key-tool-task",
            )
        )
        assert replayed.duplicate is True
        assert replayed.error_code == "tool_result_expired"
        assert objects.read(result.result_ref)

        original_delete = objects.delete_exact

        def fail_delete(_ref: str) -> None:
            raise OSError("object_store_unavailable")

        monkeypatch.setattr(objects, "delete_exact", fail_delete)
        assert retention.purge_expired_tool_results(limit=10) == 0
        with database.sessions() as session:
            retained = session.get(ToolCallRow, result.tool_call_id)
            assert retained is not None
            assert retained.status == "succeeded"
            assert retained.error_code is None
            assert retained.result_ref == result.result_ref
            assert retained.result_purged_at is None

        monkeypatch.setattr(objects, "delete_exact", original_delete)
        assert retention.purge_expired_tool_results(limit=10) == 1
        with database.sessions() as session:
            purged = session.get(ToolCallRow, result.tool_call_id)
            assert purged is not None
            assert purged.status == "succeeded"
            assert purged.error_code is None
            assert purged.result_ref is None and purged.result_size is None
            assert purged.result_hash == result.result_hash
            assert purged.result_expires_at is not None
            assert purged.result_expires_at.replace(tzinfo=UTC) == expired_at
            assert purged.result_purged_at is not None
            assert purged.verified_at is not None and purged.completed_at is not None
        with pytest.raises(OSError):
            objects.read(result.result_ref)
        assert retention.purge_expired_tool_results(limit=10) == 0
    finally:
        executor.close()


def test_finished_task_gc_marks_no_result_once_without_changing_status(runtime) -> None:
    database, clock, _auth, _objects, repository, checkpointer, retention = runtime
    lease = _claim(runtime, "failed-task")
    repository.finish_task(
        lease,
        status=TaskStatus.FAILED,
        error_code="task_agent_failed",
        trace_id="trace-a",
    )
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "failed-task")
        assert row is not None
        row.content_expires_at = clock.value - timedelta(seconds=1)

    assert retention.purge_finished_task_content(limit=10) == 1
    assert retention.purge_finished_task_content(limit=10) == 0
    with database.sessions() as session:
        row = session.get(AgentTaskRow, "failed-task")
        assert row is not None
        assert row.status == TaskStatus.FAILED.value
        assert row.last_error_code == "task_agent_failed"
        assert row.result_ref is None
        assert row.checkpoint_purged_at is not None
        assert row.result_purged_at is not None
    assert checkpointer.deleted == ["failed-task"]


def test_finished_task_gc_retains_result_ref_after_real_delete_failure(runtime, monkeypatch) -> None:
    database, clock, _auth, objects, repository, checkpointer, retention = runtime
    lease = _claim(runtime, "refused-task")
    result = build_task_result(
        TaskResultCandidate(
            status="refused",
            refusal_code="insufficient_evidence",
            stop_reason="no_evidence",
        ),
        task_id="refused-task",
        citations=(),
    )
    result_ref, result_hash = AgentResultStore(objects).save_task_result("refused-task", result)
    repository.finish_task(
        lease,
        status=TaskStatus.REFUSED,
        result_kind="research",
        result_ref=result_ref,
        result_hash=result_hash,
        trace_id="trace-a",
    )
    with database.sessions.begin() as session:
        row = session.get(AgentTaskRow, "refused-task")
        assert row is not None
        row.content_expires_at = clock.value - timedelta(seconds=1)

    original_delete = objects.delete_exact

    def fail_delete(_ref: str) -> None:
        raise OSError("object_store_unavailable")

    monkeypatch.setattr(objects, "delete_exact", fail_delete)
    assert retention.purge_finished_task_content(limit=10) == 1
    with database.sessions() as session:
        retained = session.get(AgentTaskRow, "refused-task")
        assert retained is not None
        assert retained.status == TaskStatus.REFUSED.value
        assert retained.result_ref == result_ref
        assert retained.result_purged_at is None
        assert retained.checkpoint_purged_at is not None
    assert checkpointer.deleted == ["refused-task"]

    monkeypatch.setattr(objects, "delete_exact", original_delete)
    assert retention.purge_finished_task_content(limit=10) == 1
    with database.sessions() as session:
        purged = session.get(AgentTaskRow, "refused-task")
        assert purged is not None
        assert purged.status == TaskStatus.REFUSED.value
        assert purged.result_ref is None
        assert purged.result_hash == result_hash
        assert purged.result_purged_at is not None


def test_orphan_scanner_respects_grace_references_and_running_deadlines(runtime) -> None:
    database, clock, _auth, objects, repository, _checkpointer, retention = runtime
    orphan_ref = "task-results/orphan-task/orphan.json"
    fresh_ref = "task-results/fresh-task/fresh.json"
    objects.put_exclusive(orphan_ref, b"orphan")
    objects.put_exclusive(fresh_ref, b"fresh")
    _set_mtime(objects, orphan_ref, clock.value - timedelta(seconds=61))
    _set_mtime(objects, fresh_ref, clock.value - timedelta(seconds=30))

    reference_lease = _claim(runtime, "referenced-task")
    referenced_result = build_task_result(
        TaskResultCandidate(
            status="refused",
            refusal_code="insufficient_evidence",
            stop_reason="no_evidence",
        ),
        task_id="referenced-task",
        citations=(),
    )
    referenced_ref, referenced_hash = AgentResultStore(objects).save_task_result(
        "referenced-task", referenced_result
    )
    repository.finish_task(
        reference_lease,
        status=TaskStatus.REFUSED,
        result_kind="research",
        result_ref=referenced_ref,
        result_hash=referenced_hash,
        trace_id="trace-a",
    )
    _set_mtime(objects, referenced_ref, clock.value - timedelta(seconds=61))

    running_lease = _claim(runtime, "running-task")
    running_ref = "tool-results/running-task/running-call.json"
    pending_task_ref = "task-results/running-task/pending.json"
    objects.put_exclusive(running_ref, b"running")
    objects.put_exclusive(pending_task_ref, b"pending")
    _set_mtime(objects, running_ref, clock.value - timedelta(seconds=61))
    _set_mtime(objects, pending_task_ref, clock.value - timedelta(seconds=61))
    with database.sessions.begin() as session:
        session.add(
            ToolCallRow(
                tool_call_id="running-call",
                task_id="running-task",
                tenant_id="tenant-a",
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
                created_at=clock.value,
                execution_deadline=clock.value + timedelta(seconds=120),
            )
        )
    assert running_lease.state.status is TaskStatus.RUNNING

    assert retention.purge_orphaned_result_objects(limit=10) == 1
    with pytest.raises(OSError):
        objects.read(orphan_ref)
    assert objects.read(fresh_ref) == b"fresh"
    assert objects.read(referenced_ref)
    assert objects.read(running_ref) == b"running"
    assert objects.read(pending_task_ref) == b"pending"


def test_orphan_scanner_continues_after_one_delete_failure(runtime, monkeypatch) -> None:
    _database, clock, _auth, objects, _repository, _checkpointer, retention = runtime
    bad_ref = "tool-results/orphan-task/bad.json"
    good_ref = "tool-results/orphan-task/good.json"
    for ref in (bad_ref, good_ref):
        objects.put_exclusive(ref, ref.encode())
        _set_mtime(objects, ref, clock.value - timedelta(seconds=61))

    original_delete = objects.delete_exact

    def fail_one(ref: str) -> None:
        if ref == bad_ref:
            raise OSError("object_store_unavailable")
        original_delete(ref)

    monkeypatch.setattr(objects, "delete_exact", fail_one)
    assert retention.purge_orphaned_result_objects(limit=10) == 1
    assert objects.read(bad_ref) == bad_ref.encode()
    with pytest.raises(OSError):
        objects.read(good_ref)


def test_orphan_scanner_retries_delete_failures_before_advancing_cursor(runtime, monkeypatch) -> None:
    _database, clock, _auth, objects, _repository, _checkpointer, retention = runtime
    failed_ref = "tool-results/orphan-task/failed.json"
    later_ref = "tool-results/orphan-task/later.json"
    objects.put_exclusive(failed_ref, b"failed")
    _set_mtime(objects, failed_ref, clock.value - timedelta(seconds=62))
    original_delete = objects.delete_exact

    def fail_once(ref: str) -> None:
        if ref == failed_ref:
            raise OSError("object_store_unavailable")
        original_delete(ref)

    monkeypatch.setattr(objects, "delete_exact", fail_once)
    assert retention.purge_orphaned_result_objects(limit=1) == 0
    objects.put_exclusive(later_ref, b"later")
    _set_mtime(objects, later_ref, clock.value - timedelta(seconds=61))
    # tool-results 在 limit=1 下隔轮被处理；失败重试不能永久挡住新孤儿。
    assert retention.purge_orphaned_result_objects(limit=1) == 0
    assert retention.purge_orphaned_result_objects(limit=1) == 0
    assert retention.purge_orphaned_result_objects(limit=1) == 0
    assert retention.purge_orphaned_result_objects(limit=1) == 1
    assert objects.read(failed_ref) == b"failed"
    with pytest.raises(OSError):
        objects.read(later_ref)

    monkeypatch.setattr(objects, "delete_exact", original_delete)
    assert retention.purge_orphaned_result_objects(limit=1) == 0
    assert retention.purge_orphaned_result_objects(limit=1) == 1
    with pytest.raises(OSError):
        objects.read(failed_ref)


def test_orphan_retry_overflow_defers_cursor_reset_until_queue_drains(runtime) -> None:
    _database, _clock, _auth, _objects, _repository, _checkpointer, retention = runtime
    prefix = "tool-results"
    retries = [
        f"{prefix}/task-{index}/result.json"
        for index in range(task_retention_module._ORPHAN_RETRY_CAPACITY)
    ]
    retention._orphan_retries[prefix] = retries.copy()
    retention._orphan_cursors[prefix] = f"{prefix}/cursor/result.json"

    retention._queue_orphan_retry(f"{prefix}/overflow/result.json")

    assert retention._orphan_retry_overflowed[prefix] is True
    assert retention._orphan_cursors[prefix] == f"{prefix}/cursor/result.json"
    for ref in retries:
        retention._discard_orphan_retry(ref)
    assert retention._orphan_retry_overflowed[prefix] is False
    assert retention._orphan_cursors[prefix] is None


def test_orphan_scanner_rotates_prefixes_when_limit_is_one(runtime) -> None:
    _database, clock, _auth, objects, _repository, _checkpointer, retention = runtime
    tool_ref = "tool-results/orphan-tool/result.json"
    task_ref = "task-results/orphan-task/result.json"
    for ref in (tool_ref, task_ref):
        objects.put_exclusive(ref, b"orphan")
        _set_mtime(objects, ref, clock.value - timedelta(seconds=61))

    assert retention.purge_orphaned_result_objects(limit=1) == 1
    with pytest.raises(OSError):
        objects.read(tool_ref)
    assert objects.read(task_ref) == b"orphan"

    assert retention.purge_orphaned_result_objects(limit=1) == 1
    with pytest.raises(OSError):
        objects.read(task_ref)


def test_orphan_scanner_skips_referenced_objects_after_restart(runtime) -> None:
    database, clock, _auth, objects, repository, checkpointer, _retention = runtime
    executors = []
    referenced_refs: list[str] = []
    try:
        for index in range(8):
            executor, result, _lease = _execute_tool(
                runtime, f"protected-tool-task-{index}"
            )
            executors.append(executor)
            assert result.result_ref is not None
            referenced_refs.append(result.result_ref)
            _set_mtime(objects, result.result_ref, clock.value - timedelta(seconds=62))
        orphan_ref = "tool-results/orphan-tool-task/orphan.json"
        objects.put_exclusive(orphan_ref, b"orphan")
        _set_mtime(objects, orphan_ref, clock.value - timedelta(seconds=61))

        restarted = TaskRetentionService(
            database=database,
            repository=repository,
            objects=objects,
            results=AgentResultStore(objects),
            checkpointer=checkpointer,
            orphan_grace_seconds=60,
            now=clock,
        )
        assert all(cursor is None for cursor in restarted._orphan_cursors.values())
        assert restarted.purge_orphaned_result_objects(limit=1) == 1
        assert all(objects.read(ref) for ref in referenced_refs)
        with pytest.raises(OSError):
            objects.read(orphan_ref)
    finally:
        for executor in executors:
            executor.close()


def test_orphan_scanner_keeps_cursor_when_scan_budget_exhausted(
    runtime, monkeypatch
) -> None:
    """预算截断时位置游标前进到本轮最后访问的文件，保证跨轮推进。"""
    _database, clock, _auth, objects, _repository, _checkpointer, retention = runtime
    for name in ("a.json", "b.json"):
        ref = f"tool-results/budget-dir/{name}"
        objects.put_exclusive(ref, b"body")
        # 未过 cutoff，不会成为删除候选，仅用于触发预算截断。
        _set_mtime(objects, ref, clock.value - timedelta(seconds=30))

    original_entries = objects.iter_ref_entries

    def limited_entries(*args, **kwargs):
        return original_entries(*args, max_scanned=1, **kwargs)

    monkeypatch.setattr(objects, "iter_ref_entries", limited_entries)

    assert retention.purge_orphaned_result_objects(limit=1) == 0
    # 溢出点未检验不作游标；游标停在最后一个已检验文件 a.json 上，
    # b.json 下轮重检，不产生跳过夹缝。
    assert (
        retention._orphan_cursors["tool-results"]
        == "tool-results/budget-dir/a.json"
    )


def test_object_store_retention_enumeration_is_fixed_prefix_stably_bounded(runtime) -> None:
    _database, clock, _auth, objects, _repository, _checkpointer, _retention = runtime
    refs = (
        "tool-results/task-b/b.json",
        "tool-results/task-a/a.json",
    )
    for ref in refs:
        objects.put_exclusive(ref, b"body")
        _set_mtime(objects, ref, clock.value - timedelta(seconds=61))

    assert objects.iter_refs(
        "tool-results/",
        older_than=clock.value - timedelta(seconds=60),
        limit=1,
    ) == ("tool-results/task-a/a.json",)
    entries, exhausted, _resume = objects.iter_ref_entries(
        "tool-results/",
        older_than=clock.value - timedelta(seconds=60),
        limit=1,
        exclude=frozenset({"tool-results/task-a/a.json"}),
    )
    assert tuple(ref for _mtime, ref in entries) == ("tool-results/task-b/b.json",)
    assert exhausted is True
    entries, exhausted, _resume = objects.iter_ref_entries(
        "tool-results/",
        older_than=clock.value - timedelta(seconds=60),
        limit=1,
        max_scanned=1,
    )
    assert len(entries) == 1
    assert exhausted is False
    with pytest.raises(ValueError, match="invalid_object_scan_prefix"):
        objects.iter_refs(
            "artifacts/",
            older_than=clock.value,
            limit=1,
        )


def test_iter_ref_entries_resumes_across_windows_until_exhausted(runtime) -> None:
    """越窗可达：N 个旧对象、max_scanned < N，连续枚举数轮后全部可达。"""
    _database, clock, _auth, objects, _repository, _checkpointer, _retention = runtime
    refs = sorted(
        f"tool-results/task-{index:02d}/result.json" for index in range(6)
    )
    for ref in refs:
        objects.put_exclusive(ref, b"body")
        _set_mtime(objects, ref, clock.value - timedelta(seconds=61))

    seen: list[str] = []
    after = None
    turns = 0
    for _turn in range(10):
        turns += 1
        entries, exhausted, resume = objects.iter_ref_entries(
            "tool-results/",
            older_than=clock.value - timedelta(seconds=60),
            limit=2,
            after=after,
            max_scanned=4,
        )
        seen.extend(ref for _mtime, ref in entries)
        if exhausted:
            assert resume is None
            break
        assert resume is not None
        after = resume
    else:
        pytest.fail("orphan enumeration did not exhaust within bounded turns")

    # 位置游标按规范序前进：预算窗口之间的对象被精确跳过，扫尽即收敛。
    # R=6、B=4 → ceil(6/4)+1=3 轮内全部可达。
    assert turns <= 3
    assert sorted(seen) == refs


def test_orphan_scanner_converges_after_restart_with_referenced_prefix_larger_than_budget(
    runtime, monkeypatch
) -> None:
    """重启收敛：全新 service + 引用对象数 > max_scanned + 真孤儿排后，有限轮内删除。"""
    database, clock, _auth, objects, repository, checkpointer, _retention = runtime
    executors = []
    referenced_refs: list[str] = []
    try:
        for index in range(6):
            executor, result, _lease = _execute_tool(
                runtime, f"padded-tool-task-{index}"
            )
            executors.append(executor)
            assert result.result_ref is not None
            referenced_refs.append(result.result_ref)
            _set_mtime(objects, result.result_ref, clock.value - timedelta(seconds=62))
        orphan_ref = "tool-results/zz-orphan-task/orphan.json"
        objects.put_exclusive(orphan_ref, b"orphan")
        _set_mtime(objects, orphan_ref, clock.value - timedelta(seconds=61))

        original_entries = objects.iter_ref_entries

        def limited_entries(*args, **kwargs):
            return original_entries(*args, max_scanned=3, **kwargs)

        monkeypatch.setattr(objects, "iter_ref_entries", limited_entries)

        restarted = TaskRetentionService(
            database=database,
            repository=repository,
            objects=objects,
            results=AgentResultStore(objects),
            checkpointer=checkpointer,
            orphan_grace_seconds=60,
            now=clock,
        )
        removed = False
        for turn in range(10):
            restarted.purge_orphaned_result_objects(limit=1)
            try:
                objects.read(orphan_ref)
            except OSError:
                removed = True
                break
        assert removed, f"orphan unreachable after {turn + 1} turns"
        assert all(objects.read(ref) for ref in referenced_refs)
    finally:
        for executor in executors:
            executor.close()


def test_orphan_scanner_removes_expired_empty_task_dirs_during_scan(runtime) -> None:
    """空目录收敛：过期空目录在扫描中被 rmdir，后续轮次 opendir 数下降。"""
    _database, clock, _auth, objects, _repository, _checkpointer, retention = runtime
    empty_dirs = [f"tool-results/empty-{index:02d}" for index in range(8)]
    for dir_name in empty_dirs:
        (objects.root / dir_name).mkdir(parents=True)
        expired = clock.value - timedelta(days=30)
        os.utime(
            objects.root / dir_name,
            (expired.timestamp(), expired.timestamp()),
        )

    retention.purge_orphaned_result_objects(limit=10)

    remaining = [name for name in empty_dirs if (objects.root / name).exists()]
    assert remaining == []


def test_put_exclusive_retries_once_when_parent_dir_removed_concurrently(
    tmp_path: Path,
) -> None:
    """写入竞态：目录被机会式 rmdir 后，put_exclusive 重试一次即成功。"""
    store = LocalObjectStore(tmp_path / "objects")

    class RemovingParent(LocalObjectStore):
        """模拟真实竞态：首次发布时目录已被并发 rmdir。"""

        def _put_exclusive_once(self, target, content):
            if not getattr(self, "_removed", False):
                self._removed = True
                target.parent.mkdir(parents=True, exist_ok=True)
                target.parent.rmdir()
            return super()._put_exclusive_once(target, content)

    racer = RemovingParent(tmp_path / "objects")
    ref = "tool-results/racy-task/result.json"
    racer.put_exclusive(ref, b"retried")

    assert store.read(ref) == b"retried"
