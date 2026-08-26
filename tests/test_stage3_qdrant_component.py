"""真实 Qdrant Stage 3 检索组件测试；只使用本测试创建的临时 collection。"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

import pytest

from rag_kb.application.ingestion_worker import ClaimedIngestionJob
from rag_kb.domain.indexing import ParsedChunk, ParsedDocument, compute_chunk_manifest_digest
from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import QdrantAuthorizedSearch
from rag_kb.infrastructure.vectorstore.qdrant_document_index import (
    DocumentIndexMetadata,
    QdrantDocumentIndex,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class DeterministicEmbedder:
    """固定二维向量，确保组件测试不依赖真实模型。"""

    def embed(self, texts):
        return [[1.0, 0.0] if "travel" in text.lower() else [0.0, 1.0] for text in texts]


class Metadata:
    def get_metadata(self, job: ClaimedIngestionJob) -> DocumentIndexMetadata:
        return DocumentIndexMetadata(
            title="Travel policy",
            department="finance",
            allowed_principals=("user:alice", "group:finance"),
        )


def _job(*, tenant_id: str, version_id: str, document_id: str, version: int) -> ClaimedIngestionJob:
    return ClaimedIngestionJob(
        tenant_id=tenant_id,
        job_id=f"job-{tenant_id}-{version_id}",
        version_id=version_id,
        document_id=document_id,
        version=version,
        source_object_key=f"objects/{tenant_id}/{version_id}",
        source_file_name="travel.md",
        content_hash=_hash(f"source-{tenant_id}-{version_id}"),
        index_version="index-v3",
        attempts=1,
        lease_owner="test-worker",
        lease_generation=1,
        projection_generation=version,
    )


def _parsed(job: ClaimedIngestionJob, label: str) -> ParsedDocument:
    text = f"travel policy {label}"
    chunks = (
        ParsedChunk(
            chunk_id=f"{job.version_id}:chunk-1",
            ordinal=1,
            text=text,
            content_hash=_hash(text),
            source_locator=f"section/{label}",
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


@pytest.fixture()
def qdrant_collection():
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
    except Exception:
        pytest.skip("qdrant_unavailable")

    collection = f"stage3_component_{uuid.uuid4().hex}"
    try:
        yield client, collection
    finally:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
        except Exception:
            pass


def test_real_qdrant_enforces_tenant_version_lifecycle_and_acl_on_dense_and_sparse(qdrant_collection) -> None:
    client, collection = qdrant_collection
    embedder = DeterministicEmbedder()
    index = QdrantDocumentIndex(
        client, embedder, Metadata(), collection=collection,
        now=lambda: datetime(2026, 8, 22, tzinfo=UTC),
    )

    current_a = _job(tenant_id="tenant-a", version_id="version-shared", document_id="same-name", version=1)
    old_a = _job(tenant_id="tenant-a", version_id="version-old", document_id="same-name", version=2)
    current_b = _job(tenant_id="tenant-b", version_id="version-shared", document_id="same-name", version=1)
    for job, label, lifecycle in (
        (current_a, "current-a", "active"),
        (old_a, "old-a", "superseded"),
        (current_b, "current-b", "active"),
    ):
        receipt = index.index(job, _parsed(job, label))
        index.publish(job, receipt, lifecycle_status=lifecycle)

    search = QdrantAuthorizedSearch(client, embedder, collection=collection)
    scope = {
        "tenant_id": "tenant-a",
        "version_ids": frozenset({"version-shared", "version-old"}),
        "projection_generations": frozenset({("version-shared", 1), ("version-old", 2)}),
        "limit": 10,
    }
    for route in (search.search_dense, search.search_sparse):
        authorized = route("travel policy", allowed_principals=frozenset({"user:alice"}), **scope)
        assert len(authorized) == 1
        assert authorized[0].tenant_id == "tenant-a"
        assert authorized[0].version_id == "version-shared"

        group_authorized = route(
            "travel policy", allowed_principals=frozenset({"group:finance"}), **scope
        )
        assert [item.version_id for item in group_authorized] == ["version-shared"]

        unauthorized = route(
            "travel policy", allowed_principals=frozenset({"user:bob"}), **scope
        )
        assert unauthorized == ()

        cross_tenant = route(
            "travel policy",
            tenant_id="tenant-b",
            version_ids=frozenset({"version-shared"}),
            projection_generations=frozenset({("version-shared", 1)}),
            allowed_principals=frozenset({"user:alice"}),
            limit=10,
        )
        assert len(cross_tenant) == 1
        assert cross_tenant[0].tenant_id == "tenant-b"
