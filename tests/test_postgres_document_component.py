"""真实 PostgreSQL 文档注册表组件测试；默认不要求开发机已启动服务。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import os
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text

from rag_kb.application.document_service import DocumentService
from rag_kb.application.ingestion_worker import DocumentRegistryIngestionJobPort
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.indexing import IndexReceipt
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import (
    DocumentVersionRow,
    IdempotencyRecordRow,
    IngestionJobRow,
)
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="未配置 RAG_KB_TEST_POSTGRES_URL")
def test_postgres_concurrent_upload_and_activation(tmp_path: Path) -> None:
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    assert database.engine.dialect.name == "postgresql"
    with database.engine.connect() as connection:
        assert connection.get_isolation_level() == "READ COMMITTED"
    suffix = uuid4().hex
    tenant_id = f"component-{suffix}"
    user_id = f"user-{suffix}"
    document_id = f"document-{suffix}"
    auth = AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": ["document:write", "document:read"]},
        request_id=f"request-{suffix}",
    )
    object_store = LocalObjectStore(tmp_path / "objects")
    service = DocumentService(database, object_store)

    try:
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO tenants (tenant_id, name, active, created_at) "
                    "VALUES (:tenant_id, :name, TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant_id": tenant_id, "name": tenant_id},
            )
            connection.execute(
                text(
                    "INSERT INTO principals "
                    "(tenant_id, user_id, roles, attributes, active, created_at) "
                    "VALUES (:tenant_id, :user_id, '[]', '{}', TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant_id": tenant_id, "user_id": user_id},
            )

        start = Barrier(4)

        def submit() -> object:
            start.wait(timeout=10)
            return service.register_upload(
                document_id=document_id,
                title="PostgreSQL 组件测试",
                department="研发部",
                source_file_name="component.md",
                content=b"# component\n## section\ncontent",
                idempotency_key="concurrent-key",
                auth=auth,
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = tuple(executor.map(lambda _index: submit(), range(4)))

        assert all(result == results[0] for result in results)
        result = results[0]
        with database.sessions() as session:
            assert session.scalar(
                select(func.count()).select_from(DocumentVersionRow).where(
                    DocumentVersionRow.tenant_id == tenant_id
                )
            ) == 1
            assert session.scalar(
                select(func.count()).select_from(IngestionJobRow).where(
                    IngestionJobRow.tenant_id == tenant_id
                )
            ) == 1
            assert session.scalar(
                select(func.count()).select_from(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.tenant_id == tenant_id
                )
            ) == 1

        port = DocumentRegistryIngestionJobPort(database)
        claimed = port.claim_next(worker_id="postgres-worker", lease_seconds=60, max_attempts=3)
        assert claimed is not None
        assert claimed.job_id == result.ingestion_job_id
        parsed = LocalDocumentParser(object_store).parse(claimed)
        port.persist_manifest(claimed, parsed)
        claimed = port.allocate_projection(claimed)
        receipt = IndexReceipt(
            tenant_id=claimed.tenant_id,
            job_id=claimed.job_id,
            version_id=claimed.version_id,
            source_content_hash=claimed.content_hash,
            index_version=claimed.index_version,
            projection_generation=claimed.projection_generation or 1,
            chunk_count=len(parsed.chunks),
            manifest_digest=parsed.manifest_digest,
            verified_at=datetime.now(UTC),
        )
        assert port.activate(claimed, receipt) == "active"
        port.complete_projection(claimed, receipt)
        snapshot = service.get_document(document_id, auth)
        assert snapshot is not None and snapshot.status == "active"
    finally:
        with database.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM idempotency_records WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text("DELETE FROM ingestion_jobs WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text("DELETE FROM chunk_metadata WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text("DELETE FROM document_versions WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text("DELETE FROM documents WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text("DELETE FROM principals WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            connection.execute(
                text("DELETE FROM tenants WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )
        database.engine.dispose()
