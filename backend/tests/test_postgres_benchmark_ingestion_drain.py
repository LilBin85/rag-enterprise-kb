"""真实 PostgreSQL 上验证检索基准的多任务摄取排空。"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, text

from rag_kb.application.document_service import DocumentService
from rag_kb.application.ingestion_worker import IngestionWorker, WorkerRunResult
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.indexing import IndexReceipt, ParsedDocument
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.document_lifecycle import DatabaseDocumentLifecycle
from rag_kb.infrastructure.database.models import DocumentVersionRow, IngestionJobRow, TenantRow
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
from rag_kb.infrastructure.object_store import LocalObjectStore
from scripts.benchmark_enterprise_retrieval import _drain_ingestion_jobs, _ingestion_diagnostics


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


class _DeterministicIndexer:
    def index(self, job, parsed: ParsedDocument) -> IndexReceipt:
        return IndexReceipt(
            tenant_id=job.tenant_id,
            job_id=job.job_id,
            version_id=job.version_id,
            source_content_hash=job.content_hash,
            index_version=job.index_version,
            projection_generation=job.projection_generation,
            chunk_count=len(parsed.chunks),
            manifest_digest=parsed.manifest_digest,
            verified_at=datetime.now(UTC),
        )

    def publish(self, job, receipt, *, lifecycle_status: str) -> None:
        return None

    def retire(self, job, receipt) -> None:
        return None

    def discard_unbound(self, job) -> None:
        return None


class _InitiallyIdleWorker:
    def __init__(self, worker: IngestionWorker) -> None:
        self.worker = worker
        self.first = True

    def run_once(self) -> WorkerRunResult:
        if self.first:
            self.first = False
            return WorkerRunResult("idle")
        return self.worker.run_once()


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_drain_four_registered_jobs_and_supersede_old_version(tmp_path: Path) -> None:
    assert POSTGRES_TEST_URL is not None
    suffix = uuid4().hex
    tenant_id = f"benchmark-drain-{suffix}"
    user_id = f"owner-{suffix}"
    database = build_database(POSTGRES_TEST_URL)
    lifecycle = DatabaseDocumentLifecycle(database)
    objects = LocalObjectStore(tmp_path / "objects")
    service = DocumentService(lifecycle, objects)
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["document:write", "document:read"]},
        request_id=f"benchmark-drain-{suffix}",
    )

    try:
        with database.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (tenant_id, name, active, created_at) VALUES (:id, :id, TRUE, CURRENT_TIMESTAMP)"),
                {"id": tenant_id},
            )
            connection.execute(
                text("INSERT INTO principals (tenant_id, user_id, roles, attributes, active, created_at) VALUES (:tenant, :user, CAST('[]' AS jsonb), CAST('{}' AS jsonb), TRUE, CURRENT_TIMESTAMP)"),
                {"tenant": tenant_id, "user": user_id},
            )

        uploads = []
        for document_id, version, expect_existing in (
            (f"doc-a-{suffix}", 1, False),
            (f"doc-b-{suffix}", 1, False),
            (f"doc-c-{suffix}", 1, False),
            (f"doc-a-{suffix}", 2, True),
        ):
            uploads.append(
                service.register_upload(
                    document_id=document_id,
                    title=document_id,
                    department="test",
                    source_file_name=f"v{version}.md",
                    content=f"# {document_id} v{version}\n\ncontent".encode(),
                    idempotency_key=f"{document_id}-v{version}",
                    auth=auth,
                    expect_existing=expect_existing,
                )
            )

        worker = _InitiallyIdleWorker(
            IngestionWorker(
                lifecycle,
                LocalDocumentParser(objects),
                _DeterministicIndexer(),
                worker_id=f"worker-{suffix}",
                lease_seconds=60,
            )
        )
        job_ids = [item.ingestion_job_id for item in uploads]
        outcomes, diagnostics = _drain_ingestion_jobs(
            worker,
            service,
            auth,
            job_ids,
            diagnostics=lambda ids: _ingestion_diagnostics(database, lifecycle, ids),
            timeout_seconds=30,
            poll_seconds=0,
        )

        assert outcomes[0]["status"] == "idle"
        assert diagnostics["error_code"] is None
        assert diagnostics["completed_job_ids"] == sorted(job_ids)
        with database.sessions() as session:
            jobs = tuple(session.scalars(select(IngestionJobRow).where(IngestionJobRow.job_id.in_(job_ids))))
            versions = tuple(
                session.scalars(
                    select(DocumentVersionRow)
                    .where(DocumentVersionRow.tenant_id == tenant_id)
                    .order_by(DocumentVersionRow.document_id, DocumentVersionRow.version)
                )
            )
        assert len(jobs) == 4 and {job.status for job in jobs} == {"succeeded"}
        assert not {job.status for job in jobs} & {"pending", "running"}
        doc_a = [row for row in versions if row.document_id == f"doc-a-{suffix}"]
        assert [row.lifecycle_status for row in doc_a] == ["superseded", "active"]
    finally:
        with database.sessions() as session:
            session.execute(delete(TenantRow).where(TenantRow.tenant_id == tenant_id))
            session.commit()
        database.engine.dispose()
