from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import func, select

from rag_kb.application.document_service import DocumentAccessDenied, DocumentService
from rag_kb.application.ingestion_worker import DocumentRegistryIngestionJobPort
from rag_kb.application.upload_reconciler import UploadReservationReconciler
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.indexing import (
    IndexReceipt,
    ParsedChunk,
    ParsedDocument,
    compute_chunk_manifest_digest,
)
from rag_kb.infrastructure.database import Base, Database, build_database
from rag_kb.infrastructure.database.document_repository import (
    DocumentRegistryConflict,
    DocumentRegistryRepository,
    IdempotencyConflict,
)
from rag_kb.infrastructure.database.models import (
    DocumentRow,
    DocumentVersionRow,
    IdempotencyRecordRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store import LocalObjectStore


def build_service(tmp_path: Path) -> tuple[DocumentService, Database, LocalObjectStore]:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'registry.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="租户甲"))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=[],
                attributes={},
            )
        )
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="user-b",
                roles=[],
                attributes={},
            )
        )
    store = LocalObjectStore(tmp_path / "objects")
    return DocumentService(database, store), database, store


def auth(
    *permissions: str,
    tenant_id: str = "tenant-a",
    user_id: str = "user-a",
) -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": list(permissions)},
        request_id="request-a",
    )


def register(
    service: DocumentService,
    *,
    key: str,
    content: bytes = b"# title",
    document_id: str = "policy-a",
    user_id: str = "user-a",
    expect_existing: bool = False,
):
    return service.register_upload(
        document_id=document_id,
        title="制度甲",
        department="研发部",
        source_file_name="policy.md",
        content=content,
        idempotency_key=key,
        auth=auth("document:write", "document:read", user_id=user_id),
        expect_existing=expect_existing,
    )


def parsed_for_claim(job: object) -> ParsedDocument:
    chunk = ParsedChunk(
        chunk_id=f"{job.version_id}:chunk-000001",
        ordinal=1,
        text="正文",
        content_hash="sha256:chunk",
        source_locator="正文",
    )
    chunks = (chunk,)
    return ParsedDocument(
        tenant_id=job.tenant_id,
        document_id=job.document_id,
        version_id=job.version_id,
        document_version=job.version,
        source_file_name=job.source_file_name,
        chunks=chunks,
        manifest_digest=compute_chunk_manifest_digest(chunks),
    )


def receipt_for_claim(job: object, parsed: ParsedDocument) -> IndexReceipt:
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


def allocate_after_manifest(port: DocumentRegistryIngestionJobPort, job, parsed):
    port.persist_manifest(job, parsed)
    return port.allocate_projection(job)


class FailFirstPutObjectStore(LocalObjectStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._should_fail = True

    def put_exclusive(self, storage_key: str, content: bytes) -> None:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("simulated_process_exit_before_object_write")
        super().put_exclusive(storage_key, content)


class FailFirstMarkRepository(DocumentRegistryRepository):
    def __init__(self) -> None:
        self._should_fail = True

    def mark_object_ready(self, *args: object, **kwargs: object):
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("simulated_process_exit_after_object_write")
        return super().mark_object_ready(*args, **kwargs)


class RecordingVectorDeleter:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def delete_document(self, *, tenant_id: str, document_id: str) -> None:
        self.deleted.append((tenant_id, document_id))


def test_register_upload_is_idempotent_and_persists_object(tmp_path: Path) -> None:
    service, database, store = build_service(tmp_path)

    first = register(service, key="idem-a")
    repeated = register(service, key="idem-a")

    assert repeated == first
    with database.sessions() as session:
        version = session.get(DocumentVersionRow, first.version_id)
        assert version is not None
        assert store.read(version.source_object_key) == b"# title"
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionJobRow)) == 1
        assert session.scalar(select(func.count()).select_from(IdempotencyRecordRow)) == 1


def test_new_key_recovers_existing_semantically_identical_upload(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)

    first = register(service, key="idem-first")
    recovered = register(service, key="idem-second")

    assert recovered.document_id == first.document_id
    assert recovered.version_id == first.version_id
    assert recovered.ingestion_job_id == first.ingestion_job_id
    assert recovered.outcome == "existing_recovered"
    with database.sessions() as session:
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionJobRow)) == 1
        assert session.scalar(select(func.count()).select_from(IdempotencyRecordRow)) == 2


def test_new_key_conflict_exposes_recovery_only_to_authorized_reader(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    first = register(service, key="idem-first")

    with pytest.raises(DocumentRegistryConflict) as error:
        register(service, key="idem-second", content=b"changed")
    assert str(error.value) == "document_already_exists"
    assert error.value.recovery is not None
    assert error.value.recovery.job_id == first.ingestion_job_id

    with pytest.raises(DocumentRegistryConflict) as denied:
        register(
            service,
            key="idem-other-user",
            content=b"changed",
            user_id="user-b",
        )
    assert denied.value.recovery is None
    with database.sessions() as session:
        assert session.scalar(select(func.count()).select_from(IdempotencyRecordRow)) == 1


def test_delete_document_removes_database_object_and_vectors(tmp_path: Path) -> None:
    service, database, store = build_service(tmp_path)
    created = register(service, key="delete-key")
    with database.sessions() as session:
        version = session.get(DocumentVersionRow, created.version_id)
        assert version is not None
        source_object_key = version.source_object_key
    vector_deleter = RecordingVectorDeleter()
    deletion_service = DocumentService(database, store, vector_deleter=vector_deleter)

    deleted = deletion_service.delete_document(
        "policy-a",
        auth("document:write", "document:read"),
    )

    assert deleted is not None
    assert deleted.deleted_versions == 1
    assert deleted.deleted_ingestion_jobs == 1
    assert vector_deleter.deleted == [("tenant-a", "policy-a")]
    assert not store._resolve(source_object_key).exists()
    with database.sessions() as session:
        assert session.get(DocumentRow, ("tenant-a", "policy-a")) is None
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 0
        assert session.scalar(select(func.count()).select_from(IngestionJobRow)) == 0
        assert session.scalar(select(func.count()).select_from(IdempotencyRecordRow)) == 0


def test_delete_document_rejects_running_ingestion_job(tmp_path: Path) -> None:
    service, database, store = build_service(tmp_path)
    created = register(service, key="delete-running")
    with database.sessions.begin() as session:
        job = session.get(IngestionJobRow, created.ingestion_job_id)
        assert job is not None
        job.status = "running"
    deletion_service = DocumentService(database, store, vector_deleter=RecordingVectorDeleter())

    with pytest.raises(DocumentRegistryConflict, match="document_ingestion_running"):
        deletion_service.delete_document(
            "policy-a",
            auth("document:write", "document:read"),
        )


def test_reusing_idempotency_key_with_changed_request_fails_closed(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    register(service, key="idem-a")

    with pytest.raises(IdempotencyConflict, match="different_request"):
        register(service, key="idem-a", content=b"changed")

    with database.sessions() as session:
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 1


def test_document_reads_are_tenant_scoped_and_permission_checked(tmp_path: Path) -> None:
    service, _, _ = build_service(tmp_path)
    register(service, key="idem-a")

    assert service.get_document("policy-a", auth("document:read")) is not None
    assert service.get_document("policy-a", auth("document:read", tenant_id="tenant-b")) is None
    with pytest.raises(DocumentAccessDenied):
        service.get_document("policy-a", auth())


def test_claim_and_activation_transition_registry_atomically(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    result = register(service, key="idem-a")
    port = DocumentRegistryIngestionJobPort(database)
    claimed = port.claim_next(worker_id="worker-a", lease_seconds=60, max_attempts=3)
    assert claimed is not None
    assert claimed.job_id == result.ingestion_job_id
    parsed = parsed_for_claim(claimed)
    claimed = allocate_after_manifest(port, claimed, parsed)
    receipt = receipt_for_claim(claimed, parsed)
    assert port.activate(claimed, receipt) == "active"
    port.complete_projection(claimed, receipt)

    document = service.get_document("policy-a", auth("document:read"))
    job = service.get_ingestion_job(result.ingestion_job_id, auth("document:read"))
    assert document is not None
    assert document.status == "active"
    assert document.versions[0].lifecycle_status == "active"
    assert job is not None
    assert job.status == "succeeded"


def test_database_worker_port_reclaims_expired_lease_and_retries(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    result = register(service, key="idem-a")
    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        assert job is not None
        job.status = "running"
        job.attempts = 1
        job.lease_owner = "dead-worker"
        job.lease_expires_at = now - timedelta(seconds=1)

    port = DocumentRegistryIngestionJobPort(database, now=lambda: now)
    reclaimed = port.claim_next(worker_id="worker-b", lease_seconds=60, max_attempts=3)
    assert reclaimed is not None
    assert reclaimed.job_id == result.ingestion_job_id
    assert reclaimed.attempts == 2

    port.record_failure(
        reclaimed,
        error_code="ingestion_retryable_failure",
        retry=True,
        retry_after=timedelta(0),
    )
    retried = port.claim_next(worker_id="worker-c", lease_seconds=60, max_attempts=3)
    assert retried is not None
    assert retried.attempts == 3
    parsed = parsed_for_claim(retried)
    retried = allocate_after_manifest(port, retried, parsed)
    receipt = receipt_for_claim(retried, parsed)
    port.activate(retried, receipt)
    port.complete_projection(retried, receipt)

    job_result = service.get_ingestion_job(result.ingestion_job_id, auth("document:read"))
    assert job_result is not None
    assert job_result.status == "succeeded"
    assert job_result.attempts == 3


def test_reserved_upload_recovers_after_object_write_crash(tmp_path: Path) -> None:
    _service, database, store = build_service(tmp_path)
    flaky_store = FailFirstPutObjectStore(store.root)
    service = DocumentService(database, flaky_store)

    with pytest.raises(RuntimeError, match="simulated_process_exit"):
        register(service, key="idem-crash")

    with database.sessions() as session:
        version = session.scalar(select(DocumentVersionRow))
        job = session.scalar(select(IngestionJobRow))
        idempotency = session.scalar(select(IdempotencyRecordRow))
        assert version is not None and version.lifecycle_status == "awaiting_object"
        assert job is not None and job.status == "awaiting_object"
        assert idempotency is not None and idempotency.status == "in_progress"
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 1
        expected_version_id = version.version_id
        expected_job_id = job.job_id

    recovered = register(service, key="idem-crash")

    assert recovered.version_id == expected_version_id
    assert recovered.ingestion_job_id == expected_job_id
    with database.sessions() as session:
        version = session.get(DocumentVersionRow, recovered.version_id)
        job = session.get(IngestionJobRow, recovered.ingestion_job_id)
        idempotency = session.scalar(select(IdempotencyRecordRow))
        assert version is not None
        assert flaky_store.read(version.source_object_key) == b"# title"
        assert job is not None and job.status == "pending"
        assert idempotency is not None and idempotency.status == "succeeded"


def test_reconciler_recovers_committed_object_orphan(tmp_path: Path) -> None:
    _service, database, store = build_service(tmp_path)
    repository = FailFirstMarkRepository()
    service = DocumentService(database, store, repository)

    with pytest.raises(RuntimeError, match="simulated_process_exit_after_object_write"):
        register(service, key="idem-orphan")

    result = UploadReservationReconciler(database, store, repository).reconcile(
        older_than=datetime.now(UTC) + timedelta(seconds=1)
    )

    assert result.scanned == 1
    assert result.recovered == 1
    assert result.missing_objects == 0
    assert result.content_mismatches == 0
    with database.sessions() as session:
        version = session.scalar(select(DocumentVersionRow))
        job = session.scalar(select(IngestionJobRow))
        idempotency = session.scalar(select(IdempotencyRecordRow))
        assert version is not None and version.lifecycle_status == "pending"
        assert job is not None and job.status == "pending"
        assert idempotency is not None and idempotency.status == "succeeded"


def test_stale_lease_generation_cannot_activate_or_fail(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    result = register(service, key="idem-lease")
    repository = DocumentRegistryRepository()

    with UnitOfWork(database) as work:
        assert work.session is not None
        first = repository.claim_next_job(work.session, worker_id="worker-a")
        assert first is not None
        first_generation = first.lease_generation

    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        assert job is not None
        job.lease_expires_at = now - timedelta(seconds=1)

    port = DocumentRegistryIngestionJobPort(database, now=lambda: now)
    reclaimed = port.claim_next(worker_id="worker-b", lease_seconds=60, max_attempts=3)
    assert reclaimed is not None
    assert reclaimed.lease_generation > first_generation
    stale_receipt = receipt_for_claim(reclaimed, parsed_for_claim(reclaimed))

    with pytest.raises(DocumentRegistryConflict, match="ingestion_lease_stale"):
        with UnitOfWork(database) as work:
            assert work.session is not None
            repository.activate_version(
                work.session,
                tenant_id="tenant-a",
                job_id=result.ingestion_job_id,
                worker_id="worker-a",
                lease_generation=first_generation,
                receipt=stale_receipt,
            )
    with pytest.raises(DocumentRegistryConflict, match="ingestion_lease_stale"):
        with UnitOfWork(database) as work:
            assert work.session is not None
            repository.fail_job(
                work.session,
                tenant_id="tenant-a",
                job_id=result.ingestion_job_id,
                worker_id="worker-a",
                lease_generation=first_generation,
                error_code="stale_worker_failure",
            )

    with database.sessions() as session:
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        assert job is not None
        assert job.status == "running"
        assert job.lease_owner == "worker-b"
        assert job.lease_generation == reclaimed.lease_generation


def test_late_old_version_activation_cannot_overwrite_new_active_version(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    old_result = register(service, key="idem-v1")
    repository = DocumentRegistryRepository()

    old_port = DocumentRegistryIngestionJobPort(database)
    old_claim = old_port.claim_next(worker_id="worker-old", lease_seconds=300, max_attempts=3)
    assert old_claim is not None
    assert old_claim.job_id == old_result.ingestion_job_id
    old_parsed = parsed_for_claim(old_claim)
    old_claim = allocate_after_manifest(old_port, old_claim, old_parsed)
    old_receipt = receipt_for_claim(old_claim, old_parsed)

    new_result = register(
        service,
        key="idem-v2",
        content=b"# version two",
        expect_existing=True,
    )
    new_port = DocumentRegistryIngestionJobPort(database)
    new_claim = new_port.claim_next(worker_id="worker-new", lease_seconds=300, max_attempts=3)
    assert new_claim is not None
    assert new_claim.job_id == new_result.ingestion_job_id
    new_parsed = parsed_for_claim(new_claim)
    new_claim = allocate_after_manifest(new_port, new_claim, new_parsed)
    new_receipt = receipt_for_claim(new_claim, new_parsed)
    assert new_port.activate(new_claim, new_receipt) == "active"
    new_port.complete_projection(new_claim, new_receipt)
    assert old_port.activate(old_claim, old_receipt) == "superseded"
    old_port.complete_projection(old_claim, old_receipt)

    with database.sessions() as session:
        old_version = session.get(DocumentVersionRow, old_result.version_id)
        new_version = session.get(DocumentVersionRow, new_result.version_id)
        old_job = session.get(IngestionJobRow, old_result.ingestion_job_id)
        active_ids = tuple(
            session.scalars(
                select(DocumentVersionRow.version_id).where(
                    DocumentVersionRow.tenant_id == "tenant-a",
                    DocumentVersionRow.document_id == "policy-a",
                    DocumentVersionRow.lifecycle_status == "active",
                )
            )
        )
        assert old_version is not None and old_version.lifecycle_status == "superseded"
        assert new_version is not None and new_version.lifecycle_status == "active"
        assert old_job is not None and old_job.status == "succeeded"
        assert active_ids == (new_result.version_id,)


def test_expired_job_at_max_attempts_is_failed_not_requeued(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    result = register(service, key="idem-exhausted")
    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        assert job is not None
        job.status = "running"
        job.attempts = 3
        job.lease_owner = "dead-worker"
        job.lease_generation = 3
        job.lease_expires_at = now - timedelta(seconds=1)

    port = DocumentRegistryIngestionJobPort(database, now=lambda: now)
    assert port.claim_next(worker_id="worker-next", lease_seconds=60, max_attempts=3) is None

    with database.sessions() as session:
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        version = session.get(DocumentVersionRow, result.version_id)
        document = session.get(DocumentRow, ("tenant-a", "policy-a"))
        assert job is not None
        assert job.status == "failed"
        assert job.error_code == "ingestion_retry_exhausted"
        assert job.lease_owner is None and job.lease_expires_at is None
        assert version is not None and version.lifecycle_status == "failed"
        assert document is not None and document.status == "failed"


def test_same_idempotency_key_is_isolated_by_user(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)

    user_a_result = register(
        service,
        key="shared-key",
        document_id="policy-user-a",
        user_id="user-a",
    )
    user_b_result = register(
        service,
        key="shared-key",
        document_id="policy-user-b",
        user_id="user-b",
    )

    assert user_a_result.version_id != user_b_result.version_id
    assert user_a_result.ingestion_job_id != user_b_result.ingestion_job_id
    assert register(
        service,
        key="shared-key",
        document_id="policy-user-a",
        user_id="user-a",
    ) == user_a_result
    assert register(
        service,
        key="shared-key",
        document_id="policy-user-b",
        user_id="user-b",
    ) == user_b_result

    with database.sessions() as session:
        records = tuple(
            session.scalars(
                select(IdempotencyRecordRow).order_by(IdempotencyRecordRow.user_id)
            )
        )
        assert [record.user_id for record in records] == ["user-a", "user-b"]
        assert {record.idempotency_key for record in records} == {"shared-key"}
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 2


def test_activation_requires_persisted_matching_index_receipt(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    register(service, key="receipt-required")
    port = DocumentRegistryIngestionJobPort(database)
    claimed = port.claim_next(worker_id="worker-receipt", lease_seconds=60, max_attempts=3)
    assert claimed is not None
    parsed = parsed_for_claim(claimed)
    receipt = receipt_for_claim(claimed, parsed)

    with pytest.raises(DocumentRegistryConflict, match="index_receipt_binding_invalid"):
        port.activate(claimed, receipt)

    with database.sessions() as session:
        version = session.get(DocumentVersionRow, claimed.version_id)
        job = session.get(IngestionJobRow, claimed.job_id)
        assert version is not None and version.lifecycle_status == "indexing"
        assert job is not None and job.status == "running"
        assert job.index_manifest_digest is None


def test_projection_publish_crash_reclaims_receipt_without_reactivation(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    register(service, key="projection-crash")
    port = DocumentRegistryIngestionJobPort(database)
    first = port.claim_next(worker_id="worker-first", lease_seconds=60, max_attempts=3)
    assert first is not None
    parsed = parsed_for_claim(first)
    first = allocate_after_manifest(port, first, parsed)
    receipt = receipt_for_claim(first, parsed)
    assert port.activate(first, receipt) == "active"
    with database.sessions.begin() as session:
        job = session.get(IngestionJobRow, first.job_id)
        assert job is not None
        job.indexed_chunk_count = receipt.chunk_count
        job.index_manifest_digest = receipt.manifest_digest
        job.index_completed_at = receipt.verified_at

    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        version = session.get(DocumentVersionRow, first.version_id)
        job = session.get(IngestionJobRow, first.job_id)
        assert version is not None and version.lifecycle_status == "active"
        assert job is not None and job.status == "running"
        job.lease_expires_at = now - timedelta(seconds=1)

    recovered_port = DocumentRegistryIngestionJobPort(database, now=lambda: now)
    recovered = recovered_port.claim_next(
        worker_id="worker-recovered",
        lease_seconds=60,
        max_attempts=3,
    )
    assert recovered is not None
    assert recovered.phase == "publish"
    recovered_receipt = recovered_port.load_index_receipt(recovered)
    assert recovered_receipt.manifest_digest == receipt.manifest_digest
    recovered_port.complete_projection(recovered, recovered_receipt)

    job_result = service.get_ingestion_job(first.job_id, auth("document:read"))
    assert job_result is not None and job_result.status == "succeeded"


def test_projection_retry_exhaustion_does_not_downgrade_active_version(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    result = register(service, key="projection-exhausted")
    port = DocumentRegistryIngestionJobPort(database)
    claimed = port.claim_next(worker_id="worker-first", lease_seconds=60, max_attempts=3)
    assert claimed is not None
    parsed = parsed_for_claim(claimed)
    claimed = allocate_after_manifest(port, claimed, parsed)
    receipt = receipt_for_claim(claimed, parsed)
    assert port.activate(claimed, receipt) == "active"

    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        assert job is not None
        job.attempts = 3
        job.lease_expires_at = now - timedelta(seconds=1)

    exhausted_port = DocumentRegistryIngestionJobPort(database, now=lambda: now)
    assert exhausted_port.claim_next(
        worker_id="worker-next", lease_seconds=60, max_attempts=3
    ) is None

    with database.sessions() as session:
        version = session.get(DocumentVersionRow, result.version_id)
        job = session.get(IngestionJobRow, result.ingestion_job_id)
        document = session.get(DocumentRow, ("tenant-a", "policy-a"))
        assert version is not None and version.lifecycle_status == "active"
        assert document is not None and document.status == "active"
        assert job is not None and job.status == "failed"
        assert job.error_code == "ingestion_retry_exhausted"


def test_concurrent_same_idempotency_key_converges_on_sqlite(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    start = Barrier(4)

    def submit() -> object:
        start.wait(timeout=5)
        return register(service, key="concurrent-key")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(lambda _index: submit(), range(4)))

    assert all(result == results[0] for result in results)
    with database.sessions() as session:
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionJobRow)) == 1
        assert session.scalar(select(func.count()).select_from(IdempotencyRecordRow)) == 1


def test_concurrent_distinct_keys_for_new_document_return_stable_conflict(tmp_path: Path) -> None:
    service, database, _ = build_service(tmp_path)
    start = Barrier(2)

    def submit(key: str) -> object:
        start.wait(timeout=5)
        return register(service, key=key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit, key) for key in ("key-a", "key-b")]
        outcomes: list[object] = []
        failures: list[BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except BaseException as exc:
                failures.append(exc)

    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], DocumentRegistryConflict)
    assert str(failures[0]) == "document_already_exists"
    with database.sessions() as session:
        assert session.scalar(select(func.count()).select_from(DocumentVersionRow)) == 1
        assert session.scalar(select(func.count()).select_from(IdempotencyRecordRow)) == 1
