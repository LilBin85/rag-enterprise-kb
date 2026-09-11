"""文档注册、摄取与投影生命周期的数据库适配器。"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Callable

from sqlalchemy import text

from rag_kb.application.ports.documents import (
    DocumentAccessDenied,
    DocumentDeletionReservation,
    DocumentListSnapshot,
    DocumentRegistryConflict,
    DocumentSnapshot,
    IngestionJobSnapshot,
    PendingObjectReservation,
    RegisteredUpload,
)
from rag_kb.application.ports.indexing import (
    ClaimedIngestionJob,
    IngestionQueueSnapshot,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.indexing import IndexReceipt, ParsedDocument
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.document_repository import DocumentRegistryRepository
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class DatabaseDocumentLifecycle:
    """将文档用例的事务、ACL 快照、租约与投影锁保持在数据库边界内。"""

    _sqlite_lock_registry_guard = threading.Lock()
    _sqlite_locks: dict[str, threading.RLock] = {}

    def __init__(
        self,
        database: Database,
        repository: DocumentRegistryRepository | None = None,
        acl_repository: ACLRepository | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._repository = repository or DocumentRegistryRepository()
        self._acl_repository = acl_repository or ACLRepository()
        self._now = now or (lambda: datetime.now(UTC))

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

    def reserve_pending_upload(
        self,
        *,
        auth: AuthContext,
        document_id: str,
        title: str,
        department: str,
        source_file_name: str,
        content_hash: str,
        idempotency_key: str,
        request_hash: str,
        expect_existing: bool,
    ) -> tuple[RegisteredUpload, bool]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            if not self._repository.principal_exists(
                work.session,
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
            ):
                raise DocumentAccessDenied("principal_not_registered")
            if expect_existing and not self._acl_repository.document_is_allowed(
                work.session,
                auth=auth,
                document_id=document_id,
                permission=Permission.DOCUMENT_WRITE,
            ):
                raise DocumentAccessDenied("document_write_denied")
            allow_existing_recovery = self._acl_repository.document_is_allowed(
                work.session,
                auth=auth,
                document_id=document_id,
                permission=Permission.DOCUMENT_READ,
            )
            return self._repository.reserve_pending_upload(
                work.session,
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
                document_id=document_id,
                title=title,
                department=department,
                source_file_name=source_file_name,
                content_hash=content_hash,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                expect_existing=expect_existing,
                allow_existing_recovery=allow_existing_recovery,
            )

    def mark_object_ready(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> RegisteredUpload:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self._repository.mark_object_ready(
                work.session,
                tenant_id=tenant_id,
                user_id=user_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )

    def get_ingestion_job(
        self,
        *,
        auth: AuthContext,
        job_id: str,
    ) -> IngestionJobSnapshot | None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            job = self._repository.get_job(work.session, auth.tenant_id, job_id)
            if job is None:
                return None
            version = self._repository.get_version(work.session, auth.tenant_id, job.version_id)
            if version is None or not self._acl_repository.document_is_allowed(
                work.session,
                auth=auth,
                document_id=version.document_id,
                permission=Permission.DOCUMENT_READ,
            ):
                return None
            return IngestionJobSnapshot(
                job_id=job.job_id,
                version_id=job.version_id,
                status=job.status,
                attempts=job.attempts,
                error_code=job.error_code,
            )

    def get_document(
        self,
        *,
        auth: AuthContext,
        document_id: str,
    ) -> DocumentSnapshot | None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            snapshot = self._repository.get_document(work.session, auth.tenant_id, document_id)
            if snapshot is None:
                return None
            if not self._acl_repository.document_is_allowed(
                work.session,
                auth=auth,
                document_id=document_id,
                permission=Permission.DOCUMENT_READ,
            ):
                return None
            return snapshot

    def list_documents(
        self,
        *,
        auth: AuthContext,
        limit: int,
        cursor: str | None,
    ) -> DocumentListSnapshot:
        if not 1 <= limit <= 100:
            raise ValueError("document_limit_invalid")
        after = self._repository.decode_list_cursor(cursor) if cursor else None
        scan_after = after
        items = []
        batch_size = max(limit + 1, 100)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            while True:
                candidates = self._repository.list_document_candidates(
                    work.session,
                    tenant_id=auth.tenant_id,
                    after=scan_after,
                    limit=batch_size,
                )
                if not candidates:
                    return DocumentListSnapshot(tuple(items), None)
                allowed_ids = self._acl_repository.authorized_document_ids(
                    work.session,
                    auth=auth,
                    document_ids=frozenset(item.document_id for item in candidates),
                    permission=Permission.DOCUMENT_READ,
                )
                for item in candidates:
                    scan_after = (item.created_at, item.document_id)
                    if item.document_id not in allowed_ids:
                        continue
                    items.append(item)
                    if len(items) > limit:
                        page = tuple(items[:limit])
                        return DocumentListSnapshot(
                            page,
                            self._repository.encode_list_cursor(
                                page[-1].created_at,
                                page[-1].document_id,
                            ),
                        )
                if len(candidates) < batch_size:
                    return DocumentListSnapshot(tuple(items), None)

    def list_ingestion_jobs(
        self,
        *,
        auth: AuthContext,
        document_id: str,
        limit: int,
    ) -> tuple[IngestionJobSnapshot, ...] | None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            snapshot = self._repository.get_document(work.session, auth.tenant_id, document_id)
            if snapshot is None:
                return None
            if not self._acl_repository.document_is_allowed(
                work.session,
                auth=auth,
                document_id=document_id,
                permission=Permission.DOCUMENT_READ,
            ):
                return None
            return tuple(
                IngestionJobSnapshot(
                    job_id=job.job_id,
                    version_id=job.version_id,
                    status=job.status,
                    attempts=job.attempts,
                    error_code=job.error_code,
                )
                for job in self._repository.list_jobs_for_document(
                    work.session,
                    tenant_id=auth.tenant_id,
                    document_id=document_id,
                    limit=limit,
                )
            )

    def prepare_document_deletion(
        self,
        *,
        auth: AuthContext,
        document_id: str,
    ) -> DocumentDeletionReservation | None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            if not self._repository.principal_exists(
                work.session,
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
            ):
                raise DocumentAccessDenied("principal_not_registered")
            snapshot = self._repository.get_document(work.session, auth.tenant_id, document_id)
            if snapshot is None or not self._acl_repository.document_is_allowed(
                work.session,
                auth=auth,
                document_id=document_id,
                permission=Permission.DOCUMENT_WRITE,
            ):
                return None
            return self._repository.prepare_document_deletion(
                work.session,
                tenant_id=auth.tenant_id,
                document_id=document_id,
            )

    def finalize_document_deletion(self, *, tenant_id: str, document_id: str) -> bool:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self._repository.finalize_document_deletion(
                work.session,
                tenant_id=tenant_id,
                document_id=document_id,
            )

    def list_pending_object_reservations(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> tuple[PendingObjectReservation, ...]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self._repository.list_pending_object_reservations(
                work.session,
                older_than=older_than,
                limit=limit,
            )

    def defer_pending_object_reservation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._repository.defer_pending_object_reservation(
                work.session,
                tenant_id=tenant_id,
                user_id=user_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )

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

    def queue_snapshot(
        self,
        *,
        max_attempts: int,
        limit: int,
    ) -> tuple[IngestionQueueSnapshot, ...]:
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


# 保留既有适配器名，调用方仍从 infrastructure 边界显式装配它。
DocumentRegistryIngestionJobPort = DatabaseDocumentLifecycle
