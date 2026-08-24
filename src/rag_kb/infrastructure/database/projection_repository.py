"""文档投影周期收敛任务的 PostgreSQL/SQLite 持久适配器。"""

from __future__ import annotations

from dataclasses import replace
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
            document.projection_reconcile_owner = worker_id
            document.projection_reconcile_expires_at = now + timedelta(seconds=lease_seconds)
            document.projection_reconcile_lease_generation += 1
            binding_valid = (
                document.applied_projection_generation is not None
                and document.applied_projection_version_id == version.version_id
                and document.projection_generation
                == document.applied_projection_generation
                and version.desired_projection_generation
                == document.applied_projection_generation
                and version.applied_projection_generation
                == document.applied_projection_generation
                and ingestion.projection_generation
                == document.applied_projection_generation
            )
            previous_generation = document.applied_projection_generation
            previous_version_id = document.applied_projection_version_id
            if binding_valid:
                generation = document.applied_projection_generation
                phase = "audit"
            else:
                generation = document.projection_generation + 1
                document.projection_generation = generation
                version.desired_projection_generation = generation
                phase = "reconcile"
            work.session.flush()
            verified_at = ingestion.index_completed_at
            if verified_at.tzinfo is None:
                verified_at = verified_at.replace(tzinfo=UTC)
            index_receipt = (
                IndexReceipt(
                    tenant_id=document.tenant_id,
                    job_id=ingestion.job_id,
                    version_id=version.version_id,
                    source_content_hash=version.content_hash,
                    index_version=version.index_version,
                    projection_generation=generation,
                    chunk_count=ingestion.indexed_chunk_count,
                    manifest_digest=ingestion.index_manifest_digest,
                    verified_at=verified_at,
                )
                if binding_valid
                else None
            )
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
                projection_generation=generation,
                previous_projection_generation=previous_generation,
                previous_projection_version_id=previous_version_id,
                phase=phase,
                lifecycle_status="active",
                index_receipt=index_receipt,
            )
            return ClaimedProjectionRepair(
                job=job,
                reconcile_lease_generation=document.projection_reconcile_lease_generation,
            )

    def projection_guard(self, claim: ClaimedProjectionRepair):
        return self._guard.projection_guard(claim.job)

    def projection_is_bound(self, claim: ClaimedProjectionRepair) -> bool:
        job = claim.job
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            document = work.session.scalar(
                select(DocumentRow).where(
                    DocumentRow.tenant_id == job.tenant_id,
                    DocumentRow.document_id == job.document_id,
                )
            )
        return (
            document is not None
            and document.applied_projection_generation == job.projection_generation
            and document.applied_projection_version_id == job.version_id
        )

    def renew(self, claim: ClaimedProjectionRepair, *, lease_seconds: int) -> None:
        if lease_seconds <= 0:
            raise ValueError("invalid_projection_reconcile_lease_duration")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            document, _version, _ingestion = self._lock_claim(work.session, claim)
            document.projection_reconcile_expires_at = self._now() + timedelta(
                seconds=lease_seconds
            )

    def prepare_rebuild(self, claim: ClaimedProjectionRepair) -> ClaimedProjectionRepair:
        if claim.job.phase != "audit":
            raise DocumentRegistryConflict("projection_reconcile_binding_invalid")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            document, version, _ingestion = self._lock_claim(work.session, claim)
            generation = document.projection_generation + 1
            document.projection_generation = generation
            version.desired_projection_generation = generation
            work.session.flush()
        return replace(
            claim,
            job=replace(
                claim.job,
                projection_generation=generation,
                previous_projection_generation=claim.job.projection_generation,
                previous_projection_version_id=claim.job.version_id,
                phase="reconcile",
                index_receipt=None,
            ),
        )

    def _lock_claim(self, session, claim: ClaimedProjectionRepair):
        job = claim.job
        document = session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.tenant_id == job.tenant_id,
                DocumentRow.document_id == job.document_id,
            )
            .with_for_update()
        )
        version = session.scalar(
            select(DocumentVersionRow).where(
                DocumentVersionRow.tenant_id == job.tenant_id,
                DocumentVersionRow.version_id == job.version_id,
            )
        )
        ingestion = session.scalar(
            select(IngestionJobRow).where(
                IngestionJobRow.tenant_id == job.tenant_id,
                IngestionJobRow.job_id == job.job_id,
                IngestionJobRow.version_id == job.version_id,
            )
        )
        expires_at = document.projection_reconcile_expires_at if document is not None else None
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        valid = (
            document is not None
            and version is not None
            and ingestion is not None
            and document.status == "active"
            and version.lifecycle_status == "active"
            and document.projection_reconcile_owner == job.lease_owner
            and document.projection_reconcile_lease_generation
            == claim.reconcile_lease_generation
            and expires_at is not None
            and expires_at > self._now()
            and ingestion.indexed_chunk_count is not None
            and ingestion.index_manifest_digest is not None
            and ingestion.index_completed_at is not None
        )
        if valid and job.phase == "audit":
            valid = (
                document.applied_projection_generation == job.projection_generation
                and document.applied_projection_version_id == job.version_id
                and version.applied_projection_generation == job.projection_generation
                and ingestion.projection_generation == job.projection_generation
            )
        elif valid and job.phase == "reconcile":
            valid = (
                document.projection_generation == job.projection_generation
                and version.desired_projection_generation == job.projection_generation
                and document.applied_projection_generation
                == job.previous_projection_generation
                and document.applied_projection_version_id
                == job.previous_projection_version_id
            )
        else:
            valid = False
        if not valid:
            raise DocumentRegistryConflict("projection_reconcile_lease_stale")
        return document, version, ingestion

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
            document, version, ingestion = self._lock_claim(work.session, claim)
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
                receipt.tenant_id != job.tenant_id
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
            if job.phase == "reconcile":
                if (
                    ingestion.projection_generation is not None
                    and ingestion.projection_generation >= job.projection_generation
                ):
                    raise DocumentRegistryConflict("projection_reconcile_binding_invalid")
                version.applied_projection_generation = receipt.projection_generation
                document.applied_projection_generation = receipt.projection_generation
                document.applied_projection_version_id = version.version_id
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
            document.projection_reconcile_after = now + audit_after
            document.projection_verified_at = now
            document.projection_reconcile_error_code = None
            document.projection_reconcile_owner = None
            document.projection_reconcile_expires_at = None
            document.updated_at = now

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
