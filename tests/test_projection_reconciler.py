from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
from rag_kb.observability import Observability


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
        self.prepared = []
        self.bound = False

    def claim_next(self, **_kwargs):
        item, self.item = self.item, None
        return item

    def complete(self, item, receipt, *, audit_after):
        self.completed.append((item, receipt, audit_after))

    def fail(self, item, *, retry_after):
        self.failed.append((item, retry_after))

    def prepare_rebuild(self, item):
        rebuilt = replace(
            item,
            job=replace(
                item.job,
                projection_generation=(item.job.projection_generation or 0) + 1,
                previous_projection_generation=item.job.projection_generation,
                previous_projection_version_id=item.job.version_id,
                phase="reconcile",
                index_receipt=None,
            ),
        )
        self.prepared.append(rebuilt)
        return rebuilt

    def projection_is_bound(self, _item):
        return self.bound


class Parser:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def parse(self, job):
        self.calls.append(job.projection_generation)
        if self.fail:
            raise RuntimeError("正文不得进入错误码")
        return parsed(job)


class Index:
    def __init__(self) -> None:
        self.published = []
        self.indexed = []
        self.verified = []
        self.retired = []
        self.discarded = []
        self.discarded_unbound = []
        self.healthy = True

    def index(self, job, document):
        self.indexed.append(job.projection_generation)
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

    def verify(self, job, receipt):
        self.verified.append((job.projection_generation, receipt.projection_generation))
        return self.healthy

    def retire(self, job, receipt):
        self.retired.append(receipt.projection_generation)

    def discard(self, job):
        self.discarded.append(job.projection_generation)

    def discard_unbound(self, job):
        self.discarded_unbound.append(job.projection_generation)


def telemetry() -> tuple[Observability, InMemorySpanExporter]:
    observability = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    observability.start()
    exporter = InMemorySpanExporter()
    assert observability.provider is not None
    observability.provider.add_span_processor(SimpleSpanProcessor(exporter))
    return observability, exporter


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
    index = Index()
    result = ProjectionReconciler(
        repairs,
        Parser(fail=True),
        index,
        worker_id="reconciler-a",
        retry_delay=timedelta(seconds=9),
    ).run_once()

    assert result.status == "retry_scheduled"
    assert result.error_code == "projection_reconcile_failed"
    assert "正文" not in result.error_code
    assert repairs.failed[0][1] == timedelta(seconds=9)
    assert index.discarded_unbound == [7]
    assert index.discarded == [7]


def test_projection_reconciler_does_not_discard_uncertain_bound_commit() -> None:
    class CommitUnknownRepairs(Repairs):
        def complete(self, item, receipt, *, audit_after):
            self.bound = True
            raise RuntimeError("commit acknowledgement lost")

    repairs = CommitUnknownRepairs(claim())
    index = Index()

    result = ProjectionReconciler(
        repairs,
        Parser(),
        index,
        worker_id="reconciler-a",
    ).run_once()

    assert result.status == "succeeded"
    assert index.discarded == []
    assert repairs.failed == []


def test_projection_reconciler_revalidates_after_entering_guard() -> None:
    class StaleAfterWaitRepairs(Repairs):
        def __init__(self, item):
            super().__init__(item)
            self.renewals = 0

        def renew(self, _item, *, lease_seconds):
            assert lease_seconds > 0
            self.renewals += 1
            if self.renewals == 2:
                raise RuntimeError("projection_reconcile_lease_stale")

    repairs = StaleAfterWaitRepairs(claim())
    index = Index()

    result = ProjectionReconciler(
        repairs,
        Parser(),
        index,
        worker_id="reconciler-a",
    ).run_once()

    assert result.status == "retry_scheduled"
    assert repairs.renewals == 2
    assert index.discarded_unbound == []
    assert index.indexed == []


def test_healthy_periodic_audit_skips_parser_and_embedder() -> None:
    base = claim()
    document = parsed(base.job)
    audit_receipt = IndexReceipt(
        tenant_id=base.job.tenant_id,
        job_id=base.job.job_id,
        version_id=base.job.version_id,
        source_content_hash=base.job.content_hash,
        index_version=base.job.index_version,
        projection_generation=base.job.projection_generation,
        chunk_count=len(document.chunks),
        manifest_digest=document.manifest_digest,
        verified_at=datetime.now(UTC),
    )
    audit = replace(
        base,
        job=replace(base.job, phase="audit", index_receipt=audit_receipt),
    )
    repairs = Repairs(audit)
    parser = Parser()
    index = Index()

    result = ProjectionReconciler(
        repairs,
        parser,
        index,
        worker_id="reconciler-a",
    ).run_once()

    assert result.status == "succeeded"
    assert parser.calls == []
    assert index.indexed == []
    assert index.verified == [(7, 7)]
    assert index.retired == [7]
    assert repairs.completed[0][1] == audit_receipt


def test_audit_rebuilds_only_after_projection_mismatch() -> None:
    base = claim()
    document = parsed(base.job)
    audit_receipt = IndexReceipt(
        tenant_id=base.job.tenant_id,
        job_id=base.job.job_id,
        version_id=base.job.version_id,
        source_content_hash=base.job.content_hash,
        index_version=base.job.index_version,
        projection_generation=base.job.projection_generation,
        chunk_count=len(document.chunks),
        manifest_digest=document.manifest_digest,
        verified_at=datetime.now(UTC),
    )
    audit = replace(
        base,
        job=replace(base.job, phase="audit", index_receipt=audit_receipt),
    )
    repairs = Repairs(audit)
    parser = Parser()
    index = Index()
    index.healthy = False

    result = ProjectionReconciler(
        repairs,
        parser,
        index,
        worker_id="reconciler-a",
    ).run_once()

    assert result.status == "succeeded"
    assert index.verified == [(7, 7)]
    assert repairs.prepared[0].job.projection_generation == 8
    assert parser.calls == [8]
    assert index.indexed == [8]
    assert index.published == [(8, 8, "active")]
    assert index.retired == [8]


def test_projection_telemetry_records_verify_and_reconcile_without_document_content() -> None:
    base = claim()
    document = parsed(base.job)
    audit = replace(
        base,
        job=replace(
            base.job,
            phase="audit",
            index_receipt=IndexReceipt(
                tenant_id=base.job.tenant_id,
                job_id=base.job.job_id,
                version_id=base.job.version_id,
                source_content_hash=base.job.content_hash,
                index_version=base.job.index_version,
                projection_generation=base.job.projection_generation,
                chunk_count=len(document.chunks),
                manifest_digest=document.manifest_digest,
                verified_at=datetime.now(UTC),
            ),
        ),
    )
    observability, exporter = telemetry()
    secret = "当前有效正文"
    try:
        with observability.activate():
            result = ProjectionReconciler(
                Repairs(audit),
                Parser(),
                Index(),
                worker_id="reconciler-a",
            ).run_once()
        payload = observability.metrics_payload().decode()
        spans = exporter.get_finished_spans()
        rendered_spans = repr(spans)
    finally:
        observability.shutdown()

    assert result.status == "succeeded"
    assert {"projection.reconcile", "projection.verify"} <= {
        span.name for span in spans
    }
    assert "rag_kb_projection_reconciliations_total" in payload
    assert "rag_kb_projection_reconcile_duration_seconds" in payload
    assert 'phase="audit"' in payload
    assert secret not in payload
    assert secret not in rendered_spans
