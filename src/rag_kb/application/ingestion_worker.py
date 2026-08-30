"""可恢复的异步文档摄取工作器。

工作器只持久化安全错误码，绝不把正文、解析异常文本或对象内容写入任务错误。
索引端口必须按 ``(tenant_id, version_id)`` 实现幂等；进程在索引完成但激活前
终止时，租约到期后的再次执行因此仍是安全的。
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Callable, Protocol

from opentelemetry import context as otel_context
from sqlalchemy import text

from rag_kb.domain.indexing import IndexReceipt, ParsedDocument
from rag_kb.infrastructure.database.document_repository import (
    DocumentRegistryConflict,
    DocumentRegistryRepository,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


@dataclass(frozen=True)
class ClaimedIngestionJob:
    """传给解析与索引端口的最小任务元数据，不包含原文正文。"""

    tenant_id: str
    job_id: str
    version_id: str
    document_id: str
    version: int
    source_object_key: str
    source_file_name: str
    content_hash: str
    index_version: str
    attempts: int
    lease_owner: str
    lease_generation: int
    projection_generation: int | None = None
    previous_projection_generation: int | None = None
    previous_projection_version_id: str | None = None
    phase: str = "ingest"
    lifecycle_status: str = "indexing"
    index_receipt: IndexReceipt | None = None
    pending_traceparent: str | None = None
    pending_tracestate: str | None = None


@dataclass(frozen=True)
class WorkerRunResult:
    status: str
    job_id: str | None = None
    error_code: str | None = None


class RetryableIngestionError(RuntimeError):
    """端口可显式标记为可重试的失败；其消息不会被持久化。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ProjectionCleanupError(RetryableIngestionError):
    pass


class PermanentIngestionError(RuntimeError):
    """端口可显式标记为不可重试的失败；其消息不会被持久化。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DocumentParserPort(Protocol):
    def parse(self, job: ClaimedIngestionJob) -> ParsedDocument: ...


class DocumentIndexPort(Protocol):
    def index(self, job: ClaimedIngestionJob, parsed_document: ParsedDocument) -> IndexReceipt: ...

    def publish(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        lifecycle_status: str,
    ) -> None: ...

    def retire(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> None: ...

    def discard(self, job: ClaimedIngestionJob) -> None: ...

    def discard_unbound(self, job: ClaimedIngestionJob) -> None: ...


class IngestionJobPort(Protocol):
    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> ClaimedIngestionJob | None: ...

    def persist_manifest(self, job: ClaimedIngestionJob, parsed: ParsedDocument) -> None: ...

    def allocate_projection(self, job: ClaimedIngestionJob) -> ClaimedIngestionJob: ...

    def renew_lease(self, job: ClaimedIngestionJob, *, lease_seconds: int) -> None: ...

    def load_index_receipt(self, job: ClaimedIngestionJob) -> IndexReceipt: ...

    def activate(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str: ...

    def bind_projection(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str: ...

    def current_lifecycle(self, job: ClaimedIngestionJob) -> str: ...

    def projection_guard(self, job: ClaimedIngestionJob): ...

    def projection_is_bound(self, job: ClaimedIngestionJob) -> bool: ...

    def refresh_projection_binding(self, job: ClaimedIngestionJob) -> ClaimedIngestionJob: ...

    def complete_projection(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        expected_lifecycle_status: str | None = None,
    ) -> None: ...

    def record_failure(
        self,
        job: ClaimedIngestionJob,
        *,
        error_code: str,
        retry: bool,
        retry_after: timedelta,
    ) -> None: ...


class DocumentRegistryIngestionJobPort:
    """为旧文档仓储补齐租约回收和延迟重试，不修改其共享接口。

    现有仓储仅领取 ``pending`` 任务、终态标记失败。适配层将可重试任务重新
    排入 ``pending``，并在每次领取前回收已过期的 ``running`` 租约。
    """

    def __init__(
        self,
        database: Database,
        repository: DocumentRegistryRepository | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._repository = repository or DocumentRegistryRepository()
        self._now = now or (lambda: datetime.now(UTC))

    _sqlite_lock_registry_guard = threading.Lock()
    _sqlite_locks: dict[str, threading.RLock] = {}

    @contextmanager
    def document_projection_guard(self, tenant_id: str, document_id: str):
        """串行化同一文档的跨存储投影变更和删除。"""
        lock_name = f"{tenant_id}|{document_id}"
        if self._database.engine.dialect.name == "postgresql":
            lock_id = int.from_bytes(
                hashlib.sha256(lock_name.encode("utf-8")).digest()[:8],
                "big",
                signed=True,
            )
            with self._database.engine.connect() as connection:
                connection.execute(text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id})
                try:
                    yield
                finally:
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": lock_id}
                    )
            return
        with self._sqlite_lock_registry_guard:
            lock = self._sqlite_locks.setdefault(lock_name, threading.RLock())
        with lock:
            yield

    def projection_guard(self, job: ClaimedIngestionJob):
        return self.document_projection_guard(job.tenant_id, job.document_id)

    def projection_is_bound(self, job: ClaimedIngestionJob) -> bool:
        if job.projection_generation is None:
            return False
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            generation, version_id = self._repository.get_projection_binding(
                work.session,
                tenant_id=job.tenant_id,
                document_id=job.document_id,
            )
        return generation == job.projection_generation and version_id == job.version_id

    def refresh_projection_binding(self, job: ClaimedIngestionJob) -> ClaimedIngestionJob:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            generation, version_id = self._repository.get_projection_binding(
                work.session,
                tenant_id=job.tenant_id,
                document_id=job.document_id,
            )
        return replace(
            job,
            previous_projection_generation=generation,
            previous_projection_version_id=version_id,
        )

    def current_lifecycle(self, job: ClaimedIngestionJob) -> str:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            version = self._repository.get_version(work.session, job.tenant_id, job.version_id)
            if version is None or version.document_id != job.document_id:
                raise DocumentRegistryConflict("projection_lifecycle_binding_invalid")
            if version.lifecycle_status not in {"active", "superseded"}:
                raise DocumentRegistryConflict("projection_lifecycle_not_publishable")
            return version.lifecycle_status

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> ClaimedIngestionJob | None:
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            # 进程终止后不依赖内存状态；过期租约再次进入既有 pending 领取路径。
            self._repository.recover_expired_jobs(
                work.session,
                now=now,
                max_attempts=max_attempts,
            )
            row = self._repository.claim_next_job(
                work.session,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
            )
            if row is None:
                return None
            version = self._repository.get_version(work.session, row.tenant_id, row.version_id)
            if version is None:
                raise DocumentRegistryConflict("ingestion_version_missing")
            phase = (
                "cleanup"
                if row.error_code == "ingestion_projection_cleanup_failed"
                and row.projection_generation is not None
                else "ingest"
            )
            if phase != "cleanup" and version.lifecycle_status in {"active", "superseded"}:
                if (
                    row.indexed_chunk_count is None
                    or row.index_manifest_digest is None
                    or row.index_completed_at is None
                ):
                    raise DocumentRegistryConflict("index_receipt_missing")
                phase = "publish"
            if phase == "ingest" and version.lifecycle_status == "pending":
                version.lifecycle_status = "indexing"
            applied_generation, applied_version_id = self._repository.get_projection_binding(
                work.session,
                tenant_id=row.tenant_id,
                document_id=version.document_id,
            )
            return ClaimedIngestionJob(
                tenant_id=row.tenant_id,
                job_id=row.job_id,
                version_id=version.version_id,
                document_id=version.document_id,
                version=version.version,
                source_object_key=version.source_object_key,
                source_file_name=version.source_file_name,
                content_hash=version.content_hash,
                index_version=version.index_version,
                attempts=row.attempts,
                lease_owner=worker_id,
                lease_generation=row.lease_generation,
                projection_generation=row.projection_generation,
                previous_projection_generation=applied_generation,
                previous_projection_version_id=applied_version_id,
                phase=phase,
                lifecycle_status=version.lifecycle_status,
                pending_traceparent=row.pending_traceparent,
                pending_tracestate=row.pending_tracestate,
            )

    def queue_snapshot(self, *, max_attempts: int, limit: int):
        if max_attempts <= 0 or limit <= 0:
            raise ValueError("invalid_ingestion_queue_snapshot_limits")
        now = self._now()
        with self._database.sessions() as session:
            return self._repository.queue_snapshot(
                session,
                max_attempts=max_attempts,
                now=now,
                limit=limit,
            )

    def allocate_projection(self, job: ClaimedIngestionJob) -> ClaimedIngestionJob:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            generation = self._repository.allocate_projection_generation(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
            )
            version = self._repository.get_version(work.session, job.tenant_id, job.version_id)
            if version is None:
                raise DocumentRegistryConflict("projection_version_missing")
            applied_generation, applied_version_id = self._repository.get_projection_binding(
                work.session,
                tenant_id=job.tenant_id,
                document_id=version.document_id,
            )
        return replace(
            job,
            projection_generation=generation,
            previous_projection_generation=applied_generation,
            previous_projection_version_id=applied_version_id,
        )

    def renew_lease(self, job: ClaimedIngestionJob, *, lease_seconds: int) -> None:
        if lease_seconds <= 0:
            raise ValueError("invalid_ingestion_lease_duration")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._repository.renew_job_lease(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                lease_seconds=lease_seconds,
            )

    def load_index_receipt(self, job: ClaimedIngestionJob) -> IndexReceipt:
        if job.projection_generation is None:
            raise DocumentRegistryConflict("projection_generation_missing")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = self._repository.lock_current_lease(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
            )
            version = self._repository.get_version(work.session, job.tenant_id, job.version_id)
            if (
                version is None
                or row.indexed_chunk_count is None
                or row.index_manifest_digest is None
                or row.index_completed_at is None
            ):
                raise DocumentRegistryConflict("index_receipt_missing")
            verified_at = row.index_completed_at
            if verified_at.tzinfo is None:
                verified_at = verified_at.replace(tzinfo=UTC)
            return IndexReceipt(
                tenant_id=row.tenant_id,
                job_id=row.job_id,
                version_id=version.version_id,
                source_content_hash=version.content_hash,
                index_version=version.index_version,
                projection_generation=job.projection_generation,
                chunk_count=row.indexed_chunk_count,
                manifest_digest=row.index_manifest_digest,
                verified_at=verified_at,
            )

    def persist_manifest(self, job: ClaimedIngestionJob, parsed: ParsedDocument) -> None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._repository.replace_chunk_manifest(
                work.session,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                parsed=parsed,
            )

    def activate(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            activated = self._repository.activate_version(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                receipt=receipt,
            )
            return activated.lifecycle_status

    def bind_projection(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str:
        """在一个 SQL 事务中激活版本并 CAS 绑定已可见的 Qdrant 世代。"""
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            activated = self._repository.activate_version(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                receipt=receipt,
            )
            self._repository.complete_projection(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                receipt=receipt,
                expected_lifecycle_status=activated.lifecycle_status,
                expected_applied_generation=job.previous_projection_generation,
                expected_applied_version_id=job.previous_projection_version_id,
            )
            return activated.lifecycle_status

    def complete_projection(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        expected_lifecycle_status: str | None = None,
    ) -> None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._repository.complete_projection(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                receipt=receipt,
                expected_lifecycle_status=expected_lifecycle_status,
                expected_applied_generation=job.previous_projection_generation,
                expected_applied_version_id=job.previous_projection_version_id,
            )

    def record_failure(
        self,
        job: ClaimedIngestionJob,
        *,
        error_code: str,
        retry: bool,
        retry_after: timedelta,
    ) -> None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = self._repository.lock_current_lease(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
            )
            if retry:
                now = self._now()
                row.status = "pending"
                row.error_code = error_code
                row.available_at = now + retry_after
                row.lease_owner = None
                row.lease_expires_at = None
                row.updated_at = now
                return
            self._repository.fail_job(
                work.session,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease_owner,
                lease_generation=job.lease_generation,
                error_code=error_code,
            )


class _LeaseHeartbeat:
    def __init__(
        self,
        renew: Callable[[], None] | None,
        lease_seconds: int,
        *,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> None:
        self._renew = renew
        self._on_failure = on_failure
        self._interval = max(0.1, lease_seconds / 3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def __enter__(self) -> "_LeaseHeartbeat":
        if self._renew is not None:
            self.renew_now()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._renew_once()
            except Exception as exc:
                self._set_error(exc)
                self._stop.set()

    def _renew_once(self) -> None:
        try:
            assert self._renew is not None
            self._renew()
        except Exception as exc:
            raise RetryableIngestionError("ingestion_lease_renewal_failed") from exc

    def check(self) -> None:
        if self._error is not None:
            raise RetryableIngestionError("ingestion_lease_renewal_failed") from self._error

    def renew_now(self) -> None:
        self.check()
        if self._renew is not None:
            try:
                self._renew_once()
            except Exception as exc:
                self._set_error(exc)
                raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
            self._thread = None
        self.check()

    def _set_error(self, exc: Exception) -> None:
        if self._error is not None:
            return
        self._error = exc
        if self._on_failure is not None:
            try:
                self._on_failure(exc)
            except Exception:
                pass


@contextmanager
def _ingestion_phase(phase: str, span_name: str):
    """统一结束摄取阶段 span 和低基数时长指标。"""
    started_at = perf_counter()
    outcome = "succeeded"
    error_code: str | None = None
    with observe_span(
        span_name,
        attributes={
            "component": "ingestion_worker",
            "operation": "ingestion.run",
            "phase": phase,
        },
    ) as span:
        try:
            yield
        except Exception as exc:
            outcome = "failure"
            error_code = stable_error_code(exc)
            raise
        finally:
            duration_seconds = perf_counter() - started_at
            finish_observed_span(
                span,
                outcome=outcome,
                error_category=error_category_for(error_code)
                if error_code is not None
                else "none",
                error_code=error_code,
                attributes={"duration_ms": duration_seconds * 1_000},
            )
            telemetry = current_observability()
            if telemetry is not None:
                telemetry.record_ingestion_phase(
                    phase=phase,
                    outcome=outcome,
                    duration_seconds=duration_seconds,
                )


class IngestionWorker:
    """一次领取一个任务的工作器；重复运行只会处理尚未成功的任务。"""

    def __init__(
        self,
        jobs: IngestionJobPort,
        parser: DocumentParserPort,
        indexer: DocumentIndexPort,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        max_attempts: int = 3,
        retry_delay: timedelta = timedelta(seconds=30),
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id_required")
        if lease_seconds <= 0 or max_attempts <= 0 or retry_delay.total_seconds() < 0:
            raise ValueError("invalid_ingestion_worker_limits")
        self._jobs = jobs
        self._parser = parser
        self._indexer = indexer
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def run_once(self) -> WorkerRunResult:
        job = self._jobs.claim_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
        )
        if job is None:
            return WorkerRunResult(status="idle")
        started_at = perf_counter()
        run_phase = job.phase
        telemetry = current_observability()
        attributes = {
            "component": "ingestion_worker",
            "operation": "ingestion.run",
            "phase": run_phase,
        }
        scope = (
            telemetry.ingestion_run_span(
                traceparent=job.pending_traceparent,
                tracestate=job.pending_tracestate,
                attributes=attributes,
            )
            if telemetry is not None
            else observe_span("ingestion.run", attributes=attributes)
        )
        with scope as span:
            result = WorkerRunResult(status="failed", job_id=job.job_id)
            try:
                renew = getattr(self._jobs, "renew_lease", None)
                heartbeat_context = otel_context.get_current()

                def record_lease_failure(exc: Exception) -> None:
                    if telemetry is not None:
                        telemetry.record_lease_failure(
                            kind="ingestion",
                            operation="ingestion.lease.renew",
                            error_code=stable_error_code(exc),
                            parent_context=heartbeat_context,
                        )

                heartbeat = _LeaseHeartbeat(
                    (lambda: renew(job, lease_seconds=self._lease_seconds))
                    if renew is not None
                    else None,
                    self._lease_seconds,
                    on_failure=record_lease_failure,
                )
                with heartbeat:
                    if job.phase == "cleanup":
                        guard_factory = getattr(self._jobs, "projection_guard", None)
                        guard = guard_factory(job) if guard_factory is not None else nullcontext()
                        with guard:
                            heartbeat.renew_now()
                            refresh = getattr(self._jobs, "refresh_projection_binding", None)
                            if refresh is not None:
                                job = refresh(job)
                            discard_unbound = getattr(self._indexer, "discard_unbound", None)
                            if discard_unbound is None:
                                raise ProjectionCleanupError("ingestion_projection_cleanup_failed")
                            try:
                                discard_unbound(job)
                            except Exception as exc:
                                raise ProjectionCleanupError(
                                    "ingestion_projection_cleanup_failed"
                                ) from exc
                        heartbeat.stop()
                        self._jobs.record_failure(
                            job,
                            error_code="ingestion_retry_exhausted",
                            retry=False,
                            retry_after=self._retry_delay,
                        )
                        result = WorkerRunResult(
                            status="failed",
                            job_id=job.job_id,
                            error_code="ingestion_retry_exhausted",
                        )
                        return result
                    if job.phase == "ingest" and job.projection_generation is not None:
                        guard_factory = getattr(self._jobs, "projection_guard", None)
                        guard = guard_factory(job) if guard_factory is not None else nullcontext()
                        with guard:
                            heartbeat.renew_now()
                            refresh = getattr(self._jobs, "refresh_projection_binding", None)
                            if refresh is not None:
                                job = refresh(job)
                            discard_unbound = getattr(self._indexer, "discard_unbound", None)
                            if discard_unbound is not None:
                                try:
                                    discard_unbound(job)
                                except Exception as exc:
                                    raise ProjectionCleanupError(
                                        "ingestion_projection_cleanup_failed"
                                    ) from exc
                    receipt = job.index_receipt
                    lifecycle_status = job.lifecycle_status
                    parsed_document = None
                    if job.phase == "ingest":
                        with _ingestion_phase("parse", "ingestion.parse"):
                            parsed_document = self._parser.parse(job)
                            heartbeat.check()
                            self._jobs.persist_manifest(job, parsed_document)
                            heartbeat.check()
                    guard_factory = getattr(self._jobs, "projection_guard", None)
                    guard = guard_factory(job) if guard_factory is not None else nullcontext()
                    with guard:
                        heartbeat.renew_now()
                        if job.phase == "ingest":
                            allocator = getattr(self._jobs, "allocate_projection", None)
                            if allocator is not None:
                                job = allocator(job)
                            discard_unbound = getattr(self._indexer, "discard_unbound", None)
                            if discard_unbound is not None:
                                try:
                                    discard_unbound(job)
                                except Exception as exc:
                                    raise ProjectionCleanupError(
                                        "ingestion_projection_cleanup_failed"
                                    ) from exc
                        if job.projection_generation is None:
                            raise PermanentIngestionError("projection_generation_missing")
                        if job.phase == "ingest":
                            if parsed_document is None:
                                raise PermanentIngestionError("parsed_document_missing")
                            with _ingestion_phase("index", "projection.index"):
                                receipt = self._indexer.index(job, parsed_document)
                            heartbeat.renew_now()
                            # 新世代先可见；SQL 精确世代仍指向旧投影，因此切换期间两者均可安全存在。
                            with _ingestion_phase("publish", "projection.publish"):
                                self._indexer.publish(job, receipt, lifecycle_status="active")
                            heartbeat.stop()
                            with _ingestion_phase("bind", "projection.bind"):
                                binder = getattr(self._jobs, "bind_projection", None)
                                if binder is not None:
                                    lifecycle_status = binder(job, receipt)
                                    if lifecycle_status != "active":
                                        try:
                                            with _ingestion_phase(
                                                "publish", "projection.publish"
                                            ):
                                                self._indexer.publish(
                                                    job,
                                                    receipt,
                                                    lifecycle_status=lifecycle_status,
                                                )
                                        except Exception:
                                            # 精确 SQL 绑定不会授权 superseded 世代；活动投影审计会清理它。
                                            pass
                                else:
                                    lifecycle_status = self._jobs.activate(job, receipt)
                                    if lifecycle_status != "active":
                                        with _ingestion_phase("publish", "projection.publish"):
                                            self._indexer.publish(
                                                job,
                                                receipt,
                                                lifecycle_status=lifecycle_status,
                                            )
                                    self._jobs.complete_projection(
                                        job,
                                        receipt,
                                        expected_lifecycle_status=lifecycle_status,
                                    )
                        else:
                            receipt_loader = getattr(self._jobs, "load_index_receipt", None)
                            if receipt_loader is not None:
                                receipt = receipt_loader(job)
                            current_lifecycle = getattr(self._jobs, "current_lifecycle", None)
                            if current_lifecycle is not None:
                                lifecycle_status = current_lifecycle(job)
                            if receipt is None:
                                raise PermanentIngestionError("index_receipt_missing")
                            with _ingestion_phase("publish", "projection.publish"):
                                self._indexer.publish(
                                    job,
                                    receipt,
                                    lifecycle_status=lifecycle_status,
                                )
                            heartbeat.stop()
                            with _ingestion_phase("bind", "projection.bind"):
                                self._jobs.complete_projection(
                                    job,
                                    receipt,
                                    expected_lifecycle_status=lifecycle_status,
                                )
                        if receipt is None:
                            raise PermanentIngestionError("index_receipt_missing")
                        if lifecycle_status == "active":
                            retire = getattr(self._indexer, "retire", None)
                            if retire is not None:
                                try:
                                    with _ingestion_phase("retire", "projection.retire"):
                                        retire(job, receipt)
                                except Exception:
                                    # SQL 已精确绑定新世代；周期审计会安全重试旧投影清理。
                                    pass
                result = WorkerRunResult(status="succeeded", job_id=job.job_id)
                return result
            except Exception as exc:
                error_code, retry = self._failure_disposition(exc, attempts=job.attempts)
                cleanup_failed = job.phase == "cleanup" or isinstance(exc, ProjectionCleanupError)
                if (
                    job.projection_generation is not None
                    and job.projection_generation != job.previous_projection_generation
                ):
                    projection_is_bound = getattr(self._jobs, "projection_is_bound", None)
                    try:
                        bound = projection_is_bound(job) if projection_is_bound is not None else False
                    except Exception:
                        bound = True
                    if bound and not cleanup_failed:
                        result = WorkerRunResult(status="succeeded", job_id=job.job_id)
                        return result
                    discard = getattr(self._indexer, "discard", None)
                    if discard is not None:
                        try:
                            discard(job)
                        except Exception:
                            # 数据保留清理优先于终态；清理恢复前持续保持可领取。
                            cleanup_failed = True
                if cleanup_failed:
                    error_code, retry = "ingestion_projection_cleanup_failed", True
                try:
                    self._jobs.record_failure(
                        job,
                        error_code=error_code,
                        retry=retry,
                        retry_after=self._retry_delay,
                    )
                except Exception:
                    # 错误持久化失败时不能暴露原始异常；租约到期后由适配层恢复。
                    result = WorkerRunResult(
                        status="failure_recording_failed",
                        job_id=job.job_id,
                        error_code="ingestion_failure_recording_failed",
                    )
                    return result
                result = WorkerRunResult(
                    status="retry_scheduled" if retry else "failed",
                    job_id=job.job_id,
                    error_code=error_code,
                )
                return result
            finally:
                duration_seconds = perf_counter() - started_at
                category = (
                    error_category_for(result.error_code)
                    if result.error_code is not None
                    else "none"
                )
                finish_observed_span(
                    span,
                    outcome=result.status,
                    error_category=category if result.status != "succeeded" else "none",
                    error_code=result.error_code,
                    attributes={"duration_ms": duration_seconds * 1_000},
                )
                if telemetry is not None:
                    telemetry.record_ingestion_job(
                        phase=run_phase,
                        outcome=result.status,
                        error_category=category if result.status != "succeeded" else "none",
                    )
                log_event(
                    "ingestion.job.completed",
                    level=40 if result.status != "succeeded" else 20,
                    component="ingestion_worker",
                    operation="ingestion.run",
                    phase=run_phase,
                    outcome=result.status,
                    error_category=category if result.status != "succeeded" else "none",
                    error_code=result.error_code,
                    duration_ms=duration_seconds * 1_000,
                )

    def _failure_disposition(self, exc: Exception, *, attempts: int) -> tuple[str, bool]:
        if isinstance(exc, PermanentIngestionError):
            return "ingestion_permanent_failure", False
        if attempts >= self._max_attempts:
            return "ingestion_retry_exhausted", False
        if isinstance(exc, RetryableIngestionError):
            return "ingestion_retryable_failure", True
        return "ingestion_processing_failed", True
