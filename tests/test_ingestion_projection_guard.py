"""同文档版本投影串行化回归测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest
from sqlalchemy import select

from rag_kb.application.document_service import DocumentService
from rag_kb.application.ingestion_worker import (
    ClaimedIngestionJob,
    DocumentRegistryIngestionJobPort,
    IngestionWorker,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.indexing import (
    IndexReceipt,
    ParsedChunk,
    ParsedDocument,
    compute_chunk_manifest_digest,
)
from rag_kb.infrastructure.database import Base, Database, build_database
from rag_kb.infrastructure.database.models import (
    DocumentVersionRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.object_store import LocalObjectStore


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="user-a",
        attributes={"permissions": ["document:write", "document:read"]},
        request_id="projection-guard-test",
    )


def _build_service(tmp_path) -> tuple[DocumentService, Database]:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'guard.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=[],
                attributes={},
            )
        )
    return DocumentService(database, LocalObjectStore(tmp_path / "objects")), database


def _register(service: DocumentService, *, key: str, content: bytes, expect_existing: bool = False):
    return service.register_upload(
        document_id="policy-a",
        title="Policy",
        department="engineering",
        source_file_name="policy.md",
        content=content,
        idempotency_key=key,
        auth=_auth(),
        expect_existing=expect_existing,
    )


def _parsed(job: ClaimedIngestionJob) -> ParsedDocument:
    chunks = (
        ParsedChunk(
            chunk_id=f"{job.version_id}:chunk-000001",
            ordinal=1,
            text="safe chunk",
            content_hash="sha256:chunk",
            source_locator="section",
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


class Parser:
    def parse(self, job: ClaimedIngestionJob) -> ParsedDocument:
        return _parsed(job)


class ControlledProjection:
    """受控 fake Qdrant：旧版本在 guard 内发布前暂停。"""

    def __init__(self, *, old_version_id: str) -> None:
        self._old_version_id = old_version_id
        self.old_publish_entered = Event()
        self.release_old_publish = Event()
        self.projection: dict[str, tuple[str, bool]] = {}

    def index(self, job: ClaimedIngestionJob, parsed: ParsedDocument) -> IndexReceipt:
        self.projection[job.version_id] = ("indexing", False)
        receipt = IndexReceipt(
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
        return receipt

    def publish(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        lifecycle_status: str,
    ) -> None:
        assert receipt.version_id == job.version_id
        if job.version_id == self._old_version_id:
            self.old_publish_entered.set()
            assert self.release_old_publish.wait(timeout=5), "old_worker_not_released"
        if lifecycle_status == "active":
            for version_id, (_status, current) in tuple(self.projection.items()):
                if version_id != job.version_id and current:
                    self.projection[version_id] = ("superseded", False)
        self.projection[job.version_id] = (lifecycle_status, lifecycle_status == "active")


class ObservedProjectionPort(DocumentRegistryIngestionJobPort):
    """测试用适配层：仅观察领取、guard 等待和栅栏调用，不改变生产语义。"""

    def __init__(self, *args, observed_job_id: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._observed_job_id = observed_job_id
        self.claimed = Event()
        self.guard_attempted = Event()
        self.claimed_generations: list[int] = []
        self.activate_generations: list[int] = []
        self.complete_generations: list[int] = []
        self.bind_generations: list[int] = []

    def claim_next(self, **kwargs):
        job = super().claim_next(**kwargs)
        if job is not None and job.job_id == self._observed_job_id:
            self.claimed_generations.append(job.lease_generation)
            self.claimed.set()
        return job

    @contextmanager
    def projection_guard(self, job: ClaimedIngestionJob):
        if job.job_id == self._observed_job_id:
            self.guard_attempted.set()
        with super().projection_guard(job):
            yield

    def activate(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str:
        self.activate_generations.append(job.lease_generation)
        return super().activate(job, receipt)

    def complete_projection(self, job: ClaimedIngestionJob, receipt: IndexReceipt, **kwargs) -> None:
        self.complete_generations.append(job.lease_generation)
        super().complete_projection(job, receipt, **kwargs)

    def bind_projection(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str:
        self.bind_generations.append(job.lease_generation)
        return super().bind_projection(job, receipt)


class LeaseFencedProjection:
    """让 generation A 停在 index 内，验证 generation B 最终覆盖其半成品投影。"""

    def __init__(self) -> None:
        self.a_index_entered = Event()
        self.release_a_index = Event()
        self.index_generations: list[int] = []
        self.publish_generations: list[int] = []
        self.payload: dict[str, object] | None = None

    def index(self, job: ClaimedIngestionJob, parsed: ParsedDocument) -> IndexReceipt:
        self.index_generations.append(job.lease_generation)
        if job.lease_generation == 1:
            self.a_index_entered.set()
            assert self.release_a_index.wait(timeout=5), "generation_a_not_released"
        self.payload = {
            "lifecycle_status": "indexing",
            "current": False,
            "lease_generation": job.lease_generation,
        }
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

    def publish(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        lifecycle_status: str,
    ) -> None:
        assert receipt.version_id == job.version_id
        self.publish_generations.append(job.lease_generation)
        self.payload = {
            "lifecycle_status": lifecycle_status,
            "current": lifecycle_status == "active",
            "lease_generation": job.lease_generation,
        }


def _worker(port, indexer, *, worker_id: str) -> IngestionWorker:
    return IngestionWorker(
        port,
        Parser(),
        indexer,
        worker_id=worker_id,
        lease_seconds=300,
        max_attempts=3,
        retry_delay=timedelta(seconds=1),
    )


def test_v2_late_projection_cannot_regress_v3_current_state(tmp_path) -> None:
    """时序：v2 guard 内激活后暂停发布，v3 只能等待其 guard 释放。"""
    service, database = _build_service(tmp_path)
    v1 = _register(service, key="v1", content=b"# v1")
    # 只为构造 v2/v3；v1 不参与该投影时序。
    with database.sessions.begin() as session:
        version = session.get(DocumentVersionRow, v1.version_id)
        job = session.get(IngestionJobRow, v1.ingestion_job_id)
        assert version is not None and job is not None
        version.desired_projection_generation = 1
        version.applied_projection_generation = 1
        version.applied_projection_version_id = version.version_id
        version.lifecycle_status = "superseded"
        job.projection_generation = 1
        job.projection_lease_generation = 1
        job.status = "succeeded"

    v2 = _register(service, key="v2", content=b"# v2", expect_existing=True)
    v3 = _register(service, key="v3", content=b"# v3", expect_existing=True)
    with database.sessions.begin() as session:
        v3_job = session.get(IngestionJobRow, v3.ingestion_job_id)
        assert v3_job is not None
        # 确保旧 worker 先领取 v2，随后才让新 worker 领取 v3。
        v3_job.available_at = datetime.now(UTC) + timedelta(minutes=10)

    old_port = DocumentRegistryIngestionJobPort(database)
    new_port = ObservedProjectionPort(database, observed_job_id=v3.ingestion_job_id)
    projection = ControlledProjection(old_version_id=v2.version_id)
    old_worker = _worker(old_port, projection, worker_id="worker-old")
    new_worker = _worker(new_port, projection, worker_id="worker-new")

    with ThreadPoolExecutor(max_workers=2) as executor:
        old_future = executor.submit(old_worker.run_once)
        assert projection.old_publish_entered.wait(timeout=5), "v2 did not reach guarded publish point"
        with database.sessions.begin() as session:
            v3_job = session.get(IngestionJobRow, v3.ingestion_job_id)
            assert v3_job is not None
            v3_job.available_at = datetime.now(UTC)

        new_future = executor.submit(new_worker.run_once)
        assert new_port.guard_attempted.wait(timeout=5), "v3 did not reach projection guard"
        assert not new_future.done(), "v3 must wait while v2 holds the same document guard"
        with database.sessions() as session:
            v2_before_publish = session.get(DocumentVersionRow, v2.version_id)
            v3_before_guard = session.get(DocumentVersionRow, v3.version_id)
            assert v2_before_publish is not None and v2_before_publish.lifecycle_status == "indexing"
            assert v3_before_guard is not None and v3_before_guard.lifecycle_status == "indexing"

        projection.release_old_publish.set()
        assert old_future.result(timeout=5).status == "succeeded"
        new_result = new_future.result(timeout=5)
        assert new_result.status == "succeeded", new_result.error_code

    with database.sessions() as session:
        v2_version = session.get(DocumentVersionRow, v2.version_id)
        v3_version = session.get(DocumentVersionRow, v3.version_id)
        v2_job = session.get(IngestionJobRow, v2.ingestion_job_id)
        v3_job = session.get(IngestionJobRow, v3.ingestion_job_id)
        active_versions = tuple(
            session.scalars(
                select(DocumentVersionRow.version_id).where(
                    DocumentVersionRow.tenant_id == "tenant-a",
                    DocumentVersionRow.document_id == "policy-a",
                    DocumentVersionRow.lifecycle_status == "active",
                )
            )
        )
        assert v2_version is not None and v2_version.lifecycle_status == "superseded"
        assert v3_version is not None and v3_version.lifecycle_status == "active"
        assert v2_job is not None and v2_job.status == "succeeded"
        assert v3_job is not None and v3_job.status == "succeeded"
        assert active_versions == (v3.version_id,)

    assert projection.projection[v2.version_id] == ("superseded", False)
    assert projection.projection[v3.version_id] == ("active", True)
    assert {version_id for version_id, (_status, current) in projection.projection.items() if current} == {
        v3.version_id
    }


def test_sqlite_projection_guard_releases_after_exception(tmp_path) -> None:
    service, database = _build_service(tmp_path)
    result = _register(service, key="guard", content=b"# guard")
    port = DocumentRegistryIngestionJobPort(database)
    claim = port.claim_next(worker_id="worker-a", lease_seconds=300, max_attempts=3)
    assert claim is not None and claim.job_id == result.ingestion_job_id
    with pytest.raises(RuntimeError, match="projection_guard_test"):
        with port.projection_guard(claim):
            raise RuntimeError("projection_guard_test")

    acquired = Event()

    def acquire_after_exception() -> None:
        with port.projection_guard(claim):
            acquired.set()

    thread = Thread(target=acquire_after_exception)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert acquired.is_set()


def test_stale_lease_generation_cannot_leave_projection_indexing(tmp_path) -> None:
    """A 在 guard 内 index 时失租；B 重领后等待并完成最终 active 投影。"""
    service, database = _build_service(tmp_path)
    upload = _register(service, key="lease-fencing", content=b"# lease fencing")
    clock = [datetime.now(UTC) - timedelta(minutes=1)]
    projection = LeaseFencedProjection()
    old_port = ObservedProjectionPort(
        database,
        observed_job_id=upload.ingestion_job_id,
        now=lambda: clock[0],
    )
    new_port = ObservedProjectionPort(
        database,
        observed_job_id=upload.ingestion_job_id,
        now=lambda: clock[0],
    )
    old_worker = _worker(old_port, projection, worker_id="worker-generation-a")
    new_worker = _worker(new_port, projection, worker_id="worker-generation-b")

    with ThreadPoolExecutor(max_workers=2) as executor:
        old_future = executor.submit(old_worker.run_once)
        assert projection.a_index_entered.wait(timeout=5), "generation_a_did_not_enter_index"
        assert old_port.claimed_generations == [1]

        # 受控推进时钟并回收租约；A 此时仍持有同文档的进程内投影 guard。
        with database.sessions.begin() as session:
            job = session.get(IngestionJobRow, upload.ingestion_job_id)
            assert job is not None and job.status == "running"
            job.lease_expires_at = clock[0]
        clock[0] += timedelta(seconds=1)

        new_future = executor.submit(new_worker.run_once)
        assert new_port.claimed.wait(timeout=5), "generation_b_did_not_reclaim_expired_lease"
        assert new_port.claimed_generations == [2]
        assert new_port.guard_attempted.wait(timeout=5), "generation_b_did_not_wait_for_guard"
        assert not new_future.done(), "generation_b_must_wait_for_generation_a_guard"

        projection.release_a_index.set()
        old_result = old_future.result(timeout=5)
        new_result = new_future.result(timeout=5)

    # A 已失去 fencing，不能激活或完成；B 才能发布最终状态。
    assert old_result.status == "failure_recording_failed"
    assert old_result.error_code == "ingestion_failure_recording_failed"
    assert new_result.status == "succeeded"
    assert old_port.activate_generations == []
    assert old_port.complete_generations == []
    assert old_port.bind_generations == []
    assert new_port.activate_generations == []
    assert new_port.complete_generations == []
    assert new_port.bind_generations == [2]
    assert projection.index_generations == [1, 2]
    assert projection.publish_generations == [2]
    assert projection.payload == {
        "lifecycle_status": "active",
        "current": True,
        "lease_generation": 2,
    }

    with database.sessions() as session:
        job = session.get(IngestionJobRow, upload.ingestion_job_id)
        version = session.get(DocumentVersionRow, upload.version_id)
        assert job is not None and job.status == "succeeded"
        assert job.lease_generation == 2
        assert version is not None and version.lifecycle_status == "active"
