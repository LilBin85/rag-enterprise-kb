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
from typing import Any, Callable, Protocol

from sqlalchemy import text

from rag_kb.domain.indexing import IndexReceipt, ParsedDocument
from rag_kb.infrastructure.database.document_repository import (
    DocumentRegistryConflict,
    DocumentRegistryRepository,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


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
    phase: str = "ingest"
    lifecycle_status: str = "indexing"
    index_receipt: IndexReceipt | None = None


@dataclass(frozen=True)
class WorkerRunResult:
    status: str
    job_id: str | None = None
    error_code: str | None = None


class RetryableIngestionError(RuntimeError):
    """端口可显式标记为可重试的失败；其消息不会被持久化。"""


class PermanentIngestionError(RuntimeError):
    """端口可显式标记为不可重试的失败；其消息不会被持久化。"""


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

    def load_index_receipt(self, job: ClaimedIngestionJob) -> IndexReceipt: ...

    def activate(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str: ...

    def current_lifecycle(self, job: ClaimedIngestionJob) -> str: ...

    def projection_guard(self, job: ClaimedIngestionJob): ...

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
    def projection_guard(self, job: ClaimedIngestionJob):
        """串行化同一文档的 DB 激活与 Qdrant 发布，防止旧 worker 倒退 current。"""
        lock_name = f"{job.tenant_id}|{job.document_id}"
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
            receipt = None
            phase = "ingest"
            if version.lifecycle_status in {"active", "superseded"}:
                if (
                    row.indexed_chunk_count is None
                    or row.index_manifest_digest is None
                    or row.index_completed_at is None
                ):
                    raise DocumentRegistryConflict("index_receipt_missing")
                phase = "publish"
            if version.lifecycle_status == "pending":
                version.lifecycle_status = "indexing"
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
                projection_generation=(
                    row.projection_generation
                    if phase == "publish"
                    or row.projection_lease_generation == row.lease_generation
                    else None
                ),
                phase=phase,
                lifecycle_status=version.lifecycle_status,
                index_receipt=receipt,
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
        return replace(job, projection_generation=generation)

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

    def run_once(self) -> WorkerRunResult:
        job = self._jobs.claim_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
        )
        if job is None:
            return WorkerRunResult(status="idle")
        try:
            receipt = job.index_receipt
            lifecycle_status = job.lifecycle_status
            parsed_document = None
            if job.phase == "ingest":
                parsed_document = self._parser.parse(job)
                self._jobs.persist_manifest(job, parsed_document)
            guard_factory = getattr(self._jobs, "projection_guard", None)
            guard = guard_factory(job) if guard_factory is not None else nullcontext()
            with guard:
                if job.phase == "ingest":
                    allocator = getattr(self._jobs, "allocate_projection", None)
                    if allocator is not None:
                        job = allocator(job)
                if job.projection_generation is None:
                    raise PermanentIngestionError("projection_generation_missing")
                if job.phase == "ingest":
                    if parsed_document is None:
                        raise PermanentIngestionError("parsed_document_missing")
                    receipt = self._indexer.index(job, parsed_document)
                    lifecycle_status = self._jobs.activate(job, receipt)
                else:
                    receipt_loader = getattr(self._jobs, "load_index_receipt", None)
                    if receipt_loader is not None:
                        receipt = receipt_loader(job)
                    current_lifecycle = getattr(self._jobs, "current_lifecycle", None)
                    if current_lifecycle is not None:
                        lifecycle_status = current_lifecycle(job)
                if receipt is None:
                    raise PermanentIngestionError("index_receipt_missing")
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
        except Exception as exc:
            error_code, retry = self._failure_disposition(exc, attempts=job.attempts)
            try:
                self._jobs.record_failure(
                    job,
                    error_code=error_code,
                    retry=retry,
                    retry_after=self._retry_delay,
                )
            except Exception:
                # 错误持久化失败时不能暴露原始异常；租约到期后由适配层恢复。
                return WorkerRunResult(
                    status="failure_recording_failed",
                    job_id=job.job_id,
                    error_code="ingestion_failure_recording_failed",
                )
            return WorkerRunResult(
                status="retry_scheduled" if retry else "failed",
                job_id=job.job_id,
                error_code=error_code,
            )
        return WorkerRunResult(status="succeeded", job_id=job.job_id)

    def run_until_idle(self, *, max_jobs: int) -> tuple[WorkerRunResult, ...]:
        if max_jobs <= 0:
            raise ValueError("max_jobs_must_be_positive")
        results: list[WorkerRunResult] = []
        for _ in range(max_jobs):
            result = self.run_once()
            results.append(result)
            if result.status == "idle":
                break
        return tuple(results)

    def _failure_disposition(self, exc: Exception, *, attempts: int) -> tuple[str, bool]:
        if isinstance(exc, PermanentIngestionError):
            return "ingestion_permanent_failure", False
        if attempts >= self._max_attempts:
            return "ingestion_retry_exhausted", False
        if isinstance(exc, RetryableIngestionError):
            return "ingestion_retryable_failure", True
        return "ingestion_processing_failed", True
