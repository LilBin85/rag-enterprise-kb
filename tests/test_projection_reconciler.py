from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rag_kb.application.ingestion_worker import ClaimedIngestionJob
from rag_kb.application.projection_reconciler import (
    ClaimedProjectionRepair,
    ProjectionReconciler,
)
from rag_kb.domain.indexing import (
    IndexReceipt,
    ParsedChunk,
    ParsedDocument,
    compute_chunk_manifest_digest,
)


def claim() -> ClaimedProjectionRepair:
    job = ClaimedIngestionJob(
        tenant_id="tenant-a",
        job_id="job-a",
        version_id="version-a",
        document_id="document-a",
        version=2,
        source_object_key="tenant-a/document-a/version-a/policy.md",
        source_file_name="policy.md",
        content_hash="sha256:source",
        index_version="kb-v2",
        attempts=1,
        lease_owner="reconciler-a",
        lease_generation=3,
        projection_generation=7,
        phase="reconcile",
        lifecycle_status="active",
    )
    return ClaimedProjectionRepair(job=job, reconcile_lease_generation=4)


def parsed(job: ClaimedIngestionJob) -> ParsedDocument:
    chunk = ParsedChunk(
        chunk_id="chunk-a",
        ordinal=1,
        text="当前有效正文",
        content_hash="sha256:chunk",
        source_locator="section/1",
    )
    return ParsedDocument(
        tenant_id=job.tenant_id,
        document_id=job.document_id,
        version_id=job.version_id,
        document_version=job.version,
        source_file_name=job.source_file_name,
        chunks=(chunk,),
        manifest_digest=compute_chunk_manifest_digest((chunk,)),
    )


class Repairs:
    def __init__(self, item: ClaimedProjectionRepair | None) -> None:
        self.item = item
        self.completed = []
        self.failed = []

    def claim_next(self, **_kwargs):
        item, self.item = self.item, None
        return item

    def complete(self, item, receipt, *, audit_after):
        self.completed.append((item, receipt, audit_after))

    def fail(self, item, *, retry_after):
        self.failed.append((item, retry_after))


class Parser:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def parse(self, job):
        if self.fail:
            raise RuntimeError("正文不得进入错误码")
        return parsed(job)


class Index:
    def __init__(self) -> None:
        self.published = []

    def index(self, job, document):
        return IndexReceipt(
            tenant_id=job.tenant_id,
            job_id=job.job_id,
            version_id=job.version_id,
            source_content_hash=job.content_hash,
            index_version=job.index_version,
            projection_generation=job.projection_generation,
            chunk_count=len(document.chunks),
            manifest_digest=document.manifest_digest,
            verified_at=datetime.now(UTC),
        )

    def publish(self, job, receipt, *, lifecycle_status):
        self.published.append((job.projection_generation, receipt.projection_generation, lifecycle_status))


def test_projection_reconciler_rebuilds_and_checkpoints_exact_generation() -> None:
    repairs = Repairs(claim())
    index = Index()
    result = ProjectionReconciler(
        repairs,
        Parser(),
        index,
        worker_id="reconciler-a",
        audit_interval=timedelta(minutes=10),
    ).run_once()

    assert result.status == "succeeded"
    assert index.published == [(7, 7, "active")]
    assert repairs.completed[0][1].projection_generation == 7
    assert repairs.completed[0][2] == timedelta(minutes=10)
    assert repairs.failed == []


def test_projection_reconciler_failure_is_retryable_and_redacted() -> None:
    repairs = Repairs(claim())
    result = ProjectionReconciler(
        repairs,
        Parser(fail=True),
        Index(),
        worker_id="reconciler-a",
        retry_delay=timedelta(seconds=9),
    ).run_once()

    assert result.status == "retry_scheduled"
    assert result.error_code == "projection_reconcile_failed"
    assert "正文" not in result.error_code
    assert repairs.failed[0][1] == timedelta(seconds=9)
