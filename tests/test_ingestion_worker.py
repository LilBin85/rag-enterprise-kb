"""可恢复摄取工作器的端口级测试，不连接真实存储或索引服务。"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from rag_kb.application.ingestion_worker import (
    ClaimedIngestionJob,
    IngestionWorker,
    PermanentIngestionError,
)
from rag_kb.domain.indexing import (
    IndexReceipt,
    ParsedChunk,
    ParsedDocument,
    compute_chunk_manifest_digest,
)


def claimed_job(*, attempts: int = 1) -> ClaimedIngestionJob:
    return ClaimedIngestionJob(
        tenant_id="tenant-a",
        job_id="job-a",
        version_id="version-a",
        document_id="document-a",
        version=1,
        source_object_key="tenant-a/document-a/version-a/document.md",
        source_file_name="document.md",
        content_hash="sha256:content",
        index_version="kb-v1",
        attempts=attempts,
        lease_owner="worker-a",
        lease_generation=attempts,
        projection_generation=1,
    )


def parsed_document(job: ClaimedIngestionJob) -> ParsedDocument:
    chunks = (
        ParsedChunk(
            chunk_id=f"{job.version_id}:chunk-000001",
            ordinal=1,
            text="正文",
            content_hash="sha256:chunk",
            source_locator="正文",
        ),
    )
    return ParsedDocument(
        tenant_id=job.tenant_id,
        document_id=job.document_id,
        version_id=job.version_id,
        document_version=job.version,
        source_file_name=job.source_file_name,
        chunks=chunks,
        manifest_digest=compute_chunk_manifest_digest(chunks),
    )


def index_receipt(job: ClaimedIngestionJob, parsed: ParsedDocument) -> IndexReceipt:
    return IndexReceipt(
        tenant_id=job.tenant_id,
        job_id=job.job_id,
        version_id=job.version_id,
        source_content_hash=job.content_hash,
        index_version=job.index_version,
        projection_generation=job.projection_generation or 1,
        chunk_count=len(parsed.chunks),
        manifest_digest=parsed.manifest_digest,
        verified_at=datetime.now(UTC),
    )


class FakeJobs:
    def __init__(self, jobs: list[ClaimedIngestionJob]) -> None:
        self.pending = jobs
        self.activated: list[str] = []
        self.persisted: list[str] = []
        self.completed: list[str] = []
        self.failures: list[tuple[str, str, bool, timedelta]] = []

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> ClaimedIngestionJob | None:
        assert worker_id == "worker-a"
        assert lease_seconds == 60
        assert max_attempts > 0
        return self.pending.pop(0) if self.pending else None

    def persist_manifest(self, job: ClaimedIngestionJob, parsed: ParsedDocument) -> None:
        assert parsed.version_id == job.version_id
        self.persisted.append(job.job_id)

    def activate(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str:
        assert receipt.job_id == job.job_id
        self.activated.append(job.job_id)
        return "active"

    def complete_projection(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        expected_lifecycle_status: str | None = None,
    ) -> None:
        assert receipt.job_id == job.job_id
        assert expected_lifecycle_status in {None, "active", "superseded"}
        self.completed.append(job.job_id)

    def record_failure(
        self,
        job: ClaimedIngestionJob,
        *,
        error_code: str,
        retry: bool,
        retry_after: timedelta,
    ) -> None:
        self.failures.append((job.job_id, error_code, retry, retry_after))
        if retry:
            self.pending.append(replace(job, attempts=job.attempts + 1))


class FakeParser:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def parse(self, job: ClaimedIngestionJob) -> ParsedDocument:
        self.calls.append(job.job_id)
        if self.error is not None:
            raise self.error
        return parsed_document(job)


class FakeIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ParsedDocument]] = []
        self.published: list[tuple[str, str]] = []

    def index(self, job: ClaimedIngestionJob, parsed_document: ParsedDocument) -> IndexReceipt:
        self.calls.append((job.job_id, parsed_document))
        return index_receipt(job, parsed_document)

    def publish(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        lifecycle_status: str,
    ) -> None:
        assert receipt.job_id == job.job_id
        self.published.append((job.job_id, lifecycle_status))


def worker(jobs: FakeJobs, parser: FakeParser, indexer: FakeIndexer, *, max_attempts: int = 3) -> IngestionWorker:
    return IngestionWorker(
        jobs,
        parser,
        indexer,
        worker_id="worker-a",
        lease_seconds=60,
        max_attempts=max_attempts,
        retry_delay=timedelta(seconds=5),
    )


def test_success_activates_once_and_duplicate_run_is_idle() -> None:
    jobs = FakeJobs([claimed_job()])
    parser = FakeParser()
    indexer = FakeIndexer()
    service = worker(jobs, parser, indexer)

    assert service.run_once().status == "succeeded"
    assert service.run_once().status == "idle"
    assert jobs.activated == ["job-a"]
    assert jobs.persisted == ["job-a"]
    assert jobs.completed == ["job-a"]
    assert parser.calls == ["job-a"]
    assert [job_id for job_id, _parsed in indexer.calls] == ["job-a"]
    assert indexer.published == [("job-a", "active")]


def test_retry_is_safe_and_a_restarted_worker_can_continue() -> None:
    jobs = FakeJobs([claimed_job()])
    failed = worker(jobs, FakeParser(RuntimeError("正文不得进入错误码")), FakeIndexer())

    first = failed.run_once()

    assert first.status == "retry_scheduled"
    assert first.error_code == "ingestion_processing_failed"
    assert "正文" not in first.error_code
    assert jobs.failures == [("job-a", "ingestion_processing_failed", True, timedelta(seconds=5))]

    restarted_indexer = FakeIndexer()
    restarted = worker(jobs, FakeParser(), restarted_indexer)
    assert restarted.run_once().status == "succeeded"
    assert jobs.activated == ["job-a"]
    assert jobs.completed == ["job-a"]
    assert [job_id for job_id, _parsed in restarted_indexer.calls] == ["job-a"]


def test_terminal_failure_never_requeues_and_keeps_only_safe_code() -> None:
    jobs = FakeJobs([claimed_job(attempts=3)])
    service = worker(
        jobs,
        FakeParser(PermanentIngestionError("敏感原文")),
        FakeIndexer(),
        max_attempts=3,
    )

    result = service.run_once()

    assert result.status == "failed"
    assert result.error_code == "ingestion_permanent_failure"
    assert jobs.failures == [("job-a", "ingestion_permanent_failure", False, timedelta(seconds=5))]
    assert service.run_once().status == "idle"


def test_publish_recovery_skips_parser_and_embedding() -> None:
    original = claimed_job(attempts=2)
    parsed = parsed_document(original)
    receipt = index_receipt(original, parsed)
    recovered = replace(
        original,
        phase="publish",
        lifecycle_status="active",
        index_receipt=receipt,
    )
    jobs = FakeJobs([recovered])
    parser = FakeParser()
    indexer = FakeIndexer()

    assert worker(jobs, parser, indexer).run_once().status == "succeeded"
    assert parser.calls == []
    assert indexer.calls == []
    assert indexer.published == [("job-a", "active")]
    assert jobs.activated == []
    assert jobs.completed == ["job-a"]
