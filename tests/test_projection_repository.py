from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rag_kb.domain.indexing import IndexReceipt, compute_chunk_manifest_digest
from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.document_repository import DocumentRegistryConflict
from rag_kb.infrastructure.database.models import (
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.database.projection_repository import (
    DatabaseProjectionReconcilePort,
)
from rag_kb.infrastructure.database.session import build_database


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 22, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def seeded_database(tmp_path, clock: Clock):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'projection.db').as_posix()}")
    Base.metadata.create_all(database.engine)
    chunk = ChunkMetadataRow(
        chunk_id="chunk-a",
        version_id="version-a",
        tenant_id="tenant-a",
        document_id="document-a",
        document_version=1,
        content_hash="sha256:chunk",
        source_locator="section/1",
        source_file_name="policy.md",
        index_version="kb-v2",
    )
    digest = compute_chunk_manifest_digest((chunk,))
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="Tenant A"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="owner",
                roles=["admin"],
                attributes={},
            )
        )
        session.flush()
        session.add(
            DocumentRow(
                tenant_id="tenant-a",
                document_id="document-a",
                title="Policy",
                department="HR",
                status="active",
                created_by="owner",
                projection_generation=1,
                applied_projection_generation=1,
                applied_projection_version_id="version-a",
                projection_reconcile_after=clock.value,
            )
        )
        session.add(
            DocumentVersionRow(
                version_id="version-a",
                tenant_id="tenant-a",
                document_id="document-a",
                version=1,
                lifecycle_status="active",
                source_object_key="tenant-a/document-a/version-a/policy.md",
                source_file_name="policy.md",
                content_hash="sha256:source",
                index_version="kb-v2",
                desired_projection_generation=1,
                applied_projection_generation=1,
            )
        )
        session.flush()
        session.add(chunk)
        session.add(
            IngestionJobRow(
                job_id="job-a",
                tenant_id="tenant-a",
                version_id="version-a",
                status="succeeded",
                attempts=1,
                lease_generation=1,
                projection_generation=1,
                projection_lease_generation=1,
                indexed_chunk_count=1,
                index_manifest_digest=digest,
                index_completed_at=clock.value,
            )
        )
    return database, digest


def receipt(claim, digest: str) -> IndexReceipt:
    job = claim.job
    return IndexReceipt(
        tenant_id=job.tenant_id,
        job_id=job.job_id,
        version_id=job.version_id,
        source_content_hash=job.content_hash,
        index_version=job.index_version,
        projection_generation=job.projection_generation,
        chunk_count=1,
        manifest_digest=digest,
        verified_at=datetime.now(UTC),
    )


def test_expired_reconcile_lease_fences_old_worker_and_new_generation_wins(tmp_path) -> None:
    clock = Clock()
    database, digest = seeded_database(tmp_path, clock)
    port = DatabaseProjectionReconcilePort(database, now=clock)

    first = port.claim_next(worker_id="repair-a", lease_seconds=1)
    assert first is not None and first.job.projection_generation == 2
    with database.sessions() as session:
        job = session.get(IngestionJobRow, "job-a")
        assert job is not None and job.projection_generation == 1
    clock.value += timedelta(seconds=2)
    second = port.claim_next(worker_id="repair-b", lease_seconds=30)
    assert second is not None and second.job.projection_generation == 3
    with database.sessions() as session:
        job = session.get(IngestionJobRow, "job-a")
        assert job is not None and job.projection_generation == 1

    with pytest.raises(DocumentRegistryConflict, match="binding_invalid"):
        port.complete(first, receipt(first, digest), audit_after=timedelta(minutes=5))
    port.complete(second, receipt(second, digest), audit_after=timedelta(minutes=5))

    with database.sessions() as session:
        document = session.get(DocumentRow, ("tenant-a", "document-a"))
        version = session.get(DocumentVersionRow, "version-a")
        job = session.get(IngestionJobRow, "job-a")
        assert document is not None and document.applied_projection_generation == 3
        assert document.applied_projection_version_id == "version-a"
        assert document.projection_reconcile_owner is None
        assert version is not None and version.applied_projection_generation == 3
        assert job is not None and job.status == "succeeded" and job.projection_generation == 3
