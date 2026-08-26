"""PostgreSQL + Qdrant 的 Stage 3 摄取、投影与授权检索组件测试。"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.ingestion_worker import IngestionWorker
from rag_kb.application.document_service import DocumentService
from rag_kb.domain.identity import AuthContext
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.database.index_projection_metadata import DatabaseDocumentIndexMetadataPort
from rag_kb.infrastructure.database.models import DocumentVersionRow, IngestionJobRow
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import QdrantAuthorizedSearch
from rag_kb.infrastructure.vectorstore.qdrant_document_index import QdrantDocumentIndex
from rag_kb.application.ingestion_worker import DocumentRegistryIngestionJobPort


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


class DeterministicEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _auth(tenant_id: str, user_id: str, permissions: tuple[str, ...]) -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={"permissions": list(permissions)},
        request_id=f"request-{user_id}",
    )


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_qdrant_ingestion_and_authorized_retrieval(tmp_path: Path) -> None:
    assert POSTGRES_TEST_URL is not None
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
    except Exception:
        pytest.skip("qdrant_unavailable")

    database = build_database(POSTGRES_TEST_URL)
    assert database.engine.dialect.name == "postgresql"
    suffix = uuid4().hex
    tenant_id = f"pipeline-{suffix}"
    other_tenant = f"pipeline-other-{suffix}"
    user_id = f"user-{suffix}"
    unauthorized_id = f"unauthorized-{suffix}"
    other_user_id = f"other-user-{suffix}"
    document_id = f"document-{suffix}"
    collection = f"stage3_pipeline_{suffix}"
    object_store = LocalObjectStore(tmp_path / "objects")
    auth = _auth(tenant_id, user_id, ("document:write", "document:read", "knowledge:read"))
    unauthorized = _auth(tenant_id, unauthorized_id, ("document:read", "knowledge:read"))
    cross_tenant = _auth(other_tenant, other_user_id, ("document:read", "knowledge:read"))
    try:
        with database.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (tenant_id, name, active, created_at) VALUES (:id, :name, TRUE, CURRENT_TIMESTAMP)"),
                [{"id": tenant_id, "name": tenant_id}, {"id": other_tenant, "name": other_tenant}],
            )
            connection.execute(
                text(
                    "INSERT INTO principals (tenant_id, user_id, roles, attributes, active, created_at) "
                    "VALUES (:tenant, :user, CAST('[]' AS jsonb), CAST('{}' AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                [
                    {"tenant": tenant_id, "user": user_id},
                    {"tenant": tenant_id, "user": unauthorized_id},
                    {"tenant": other_tenant, "user": other_user_id},
                ],
            )

        service = DocumentService(database, object_store)
        registered = service.register_upload(
            document_id=document_id,
            title="Pipeline policy",
            department="finance",
            source_file_name="policy.md",
            content=b"# Travel policy\n\nTravel reimbursement requires receipts.",
            idempotency_key=f"upload-{suffix}",
            auth=auth,
        )
        index = QdrantDocumentIndex(
            client,
            DeterministicEmbedder(),
            DatabaseDocumentIndexMetadataPort(database),
            collection=collection,
        )
        worker = IngestionWorker(
            DocumentRegistryIngestionJobPort(database),
            LocalDocumentParser(object_store),
            index,
            worker_id=f"pipeline-worker-{suffix}",
            lease_seconds=60,
        )
        outcome = worker.run_once()
        assert outcome.status == "succeeded"
        assert outcome.job_id == registered.ingestion_job_id

        with database.sessions() as session:
            version = session.scalar(select(DocumentVersionRow).where(DocumentVersionRow.tenant_id == tenant_id))
            job = session.scalar(select(IngestionJobRow).where(IngestionJobRow.tenant_id == tenant_id))
            assert version is not None and version.lifecycle_status == "active"
            assert job is not None and job.status == "succeeded"
            assert job.index_completed_at is not None

        authorization = DatabaseAuthorizationPort(database)
        vectors = QdrantAuthorizedSearch(client, DeterministicEmbedder(), collection=collection)
        retriever = EnterpriseRetriever(
            authorization,
            vectors,
            candidate_k=5,
            context_k=5,
            context_token_budget=1000,
        )
        result = retriever.search("travel reimbursement", auth)
        assert result.sufficient is True
        assert len(result.candidates) == 1
        assert result.candidates[0].tenant_id == tenant_id
        assert retriever.search("travel reimbursement", unauthorized).candidates == ()
        assert retriever.search("travel reimbursement", cross_tenant).candidates == ()
    finally:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
        except Exception:
            pass
        with database.engine.begin() as connection:
            connection.execute(text("DELETE FROM tenants WHERE tenant_id IN (:tenant, :other)"), {"tenant": tenant_id, "other": other_tenant})
        database.engine.dispose()
