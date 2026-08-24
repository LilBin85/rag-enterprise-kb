"""文档投影周期收敛任务的 PostgreSQL/SQLite 持久适配器。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, or_, select

from rag_kb.application.ingestion_worker import (
    ClaimedIngestionJob,
    DocumentRegistryIngestionJobPort,
)
from rag_kb.application.projection_reconciler import ClaimedProjectionRepair
from rag_kb.domain.indexing import IndexReceipt, compute_chunk_manifest_digest
from rag_kb.infrastructure.database.document_repository import DocumentRegistryConflict
from rag_kb.infrastructure.database.models import (
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class DatabaseProjectionReconcilePort:
    def __init__(self, database: Database, *, now=None) -> None:
        self._database = database
        self._now = now or (lambda: datetime.now(UTC))
        self._guard = DocumentRegistryIngestionJobPort(database, now=self._now)

    def claim_next(
        self, *, worker_id: str, lease_seconds: int
    ) -> ClaimedProjectionRepair | None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_projection_reconcile_claim")
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.execute(
                select(DocumentRow, DocumentVersionRow, IngestionJobRow)
                .join(
                    DocumentVersionRow,
                    and_(
                        DocumentVersionRow.tenant_id == DocumentRow.tenant_id,
                        DocumentVersionRow.document_id == DocumentRow.document_id,
                    ),
                )
                .join(
                    IngestionJobRow,
                    and_(
                        IngestionJobRow.tenant_id == DocumentVersionRow.tenant_id,
                        IngestionJobRow.version_id == DocumentVersionRow.version_id,
                    ),
                )
                .where(
                    DocumentRow.status == "active",
                    DocumentRow.projection_reconcile_after <= now,
                    or_(
                        DocumentRow.projection_reconcile_owner.is_(None),
                        DocumentRow.projection_reconcile_expires_at <= now,
                    ),
                    DocumentVersionRow.lifecycle_status == "active",
                    IngestionJobRow.status.in_(("succeeded", "failed")),
                    IngestionJobRow.indexed_chunk_count.is_not(None),
                    IngestionJobRow.index_manifest_digest.is_not(None),
                    IngestionJobRow.index_completed_at.is_not(None),
                )
                .order_by(DocumentRow.projection_reconcile_after, DocumentRow.document_id)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).one_or_none()
            if row is None:
                return None
            document, version, ingestion = row
            document.projection_generation += 1
            document.projection_reconcile_owner = worker_id
            document.projection_reconcile_expires_at = now + timedelta(seconds=lease_seconds)
            document.projection_reconcile_lease_generation += 1
            version.desired_projection_generation = document.projection_generation
            work.session.flush()
            job = ClaimedIngestionJob(
                tenant_id=document.tenant_id,
                job_id=ingestion.job_id,
                version_id=version.version_id,
                document_id=document.document_id,
                version=version.version,
                source_object_key=version.source_object_key,
                source_file_name=version.source_file_name,
                content_hash=version.content_hash,
                index_version=version.index_version,
                attempts=ingestion.attempts,
                lease_owner=worker_id,
                lease_generation=ingestion.lease_generation,
                projection_generation=document.projection_generation,
                phase="reconcile",
                lifecycle_status="active",
            )
            return ClaimedProjectionRepair(
                job=job,
                reconcile_lease_generation=document.projection_reconcile_lease_generation,
            )

    def projection_guard(self, claim: ClaimedProjectionRepair):
        return self._guard.projection_guard(claim.job)

    def complete(
        self,
        claim: ClaimedProjectionRepair,
        receipt: IndexReceipt,
        *,
        audit_after: timedelta,
    ) -> None:
        if audit_after.total_seconds() <= 0:
            raise ValueError("invalid_projection_audit_interval")
        job = claim.job
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            document = work.session.scalar(
                select(DocumentRow)
                .where(
                    DocumentRow.tenant_id == job.tenant_id,
                    DocumentRow.document_id == job.document_id,
                )
                .with_for_update()
            )
            version = work.session.scalar(
                select(DocumentVersionRow).where(
                    DocumentVersionRow.tenant_id == job.tenant_id,
                    DocumentVersionRow.version_id == job.version_id,
                )
            )
            ingestion = work.session.scalar(
                select(IngestionJobRow)
                .where(
                    IngestionJobRow.tenant_id == job.tenant_id,
                    IngestionJobRow.job_id == job.job_id,
                    IngestionJobRow.version_id == job.version_id,
                )
                .with_for_update()
            )
            chunks = tuple(
                work.session.scalars(
                    select(ChunkMetadataRow)
                    .where(
                        ChunkMetadataRow.tenant_id == job.tenant_id,
                        ChunkMetadataRow.version_id == job.version_id,
                    )
                    .order_by(ChunkMetadataRow.chunk_id)
                )
            )
            if (
                document is None
                or version is None
                or ingestion is None
                or version.lifecycle_status != "active"
                or document.projection_reconcile_owner != job.lease_owner
                or document.projection_reconcile_lease_generation
                != claim.reconcile_lease_generation
                or document.projection_generation != job.projection_generation
                or version.desired_projection_generation != job.projection_generation
                or ingestion.projection_generation is None
                or ingestion.projection_generation >= job.projection_generation
                or ingestion.indexed_chunk_count is None
                or ingestion.index_manifest_digest is None
                or ingestion.index_completed_at is None
                or receipt.tenant_id != job.tenant_id
                or receipt.job_id != job.job_id
                or receipt.version_id != job.version_id
                or receipt.source_content_hash != job.content_hash
                or receipt.index_version != job.index_version
                or receipt.projection_generation != job.projection_generation
                or receipt.chunk_count != len(chunks)
                or receipt.manifest_digest != compute_chunk_manifest_digest(chunks)
                or not chunks
            ):
                raise DocumentRegistryConflict("projection_reconcile_binding_invalid")
            now = self._now()
            version.applied_projection_generation = receipt.projection_generation
            document.applied_projection_generation = receipt.projection_generation
            document.applied_projection_version_id = version.version_id
            document.projection_reconcile_after = now + audit_after
            document.projection_verified_at = now
            document.projection_reconcile_error_code = None
            document.projection_reconcile_owner = None
            document.projection_reconcile_expires_at = None
            document.updated_at = now
            ingestion.projection_generation = receipt.projection_generation
            ingestion.projection_lease_generation = max(
                1,
                ingestion.projection_lease_generation or 0,
                ingestion.lease_generation,
                claim.reconcile_lease_generation,
            )
            ingestion.indexed_chunk_count = receipt.chunk_count
            ingestion.index_manifest_digest = receipt.manifest_digest
            ingestion.index_completed_at = receipt.verified_at
            ingestion.status = "succeeded"
            ingestion.error_code = None
            ingestion.lease_owner = None
            ingestion.lease_expires_at = None
            ingestion.updated_at = now

    def fail(self, claim: ClaimedProjectionRepair, *, retry_after: timedelta) -> None:
        if retry_after.total_seconds() < 0:
            raise ValueError("invalid_projection_reconcile_retry")
        job = claim.job
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            document = work.session.scalar(
                select(DocumentRow)
                .where(
                    DocumentRow.tenant_id == job.tenant_id,
                    DocumentRow.document_id == job.document_id,
                )
                .with_for_update()
            )
            if (
                document is None
                or document.projection_reconcile_owner != job.lease_owner
                or document.projection_reconcile_lease_generation
                != claim.reconcile_lease_generation
            ):
                raise DocumentRegistryConflict("projection_reconcile_lease_stale")
            document.projection_reconcile_after = self._now() + retry_after
            document.projection_reconcile_error_code = "projection_reconcile_failed"
            document.projection_reconcile_owner = None
            document.projection_reconcile_expires_at = None
