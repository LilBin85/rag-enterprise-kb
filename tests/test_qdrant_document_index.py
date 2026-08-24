from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from rag_kb.application.ingestion_worker import ClaimedIngestionJob
from rag_kb.domain.indexing import ParsedChunk, ParsedDocument, compute_chunk_manifest_digest
from rag_kb.infrastructure.vectorstore.qdrant_document_index import (
    DocumentIndexError,
    DocumentIndexMetadata,
    QdrantDocumentDeletion,
    QdrantDocumentIndex,
    keyword_sparse_vector,
    projection_key,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _job(*, tenant_id: str = "tenant-a", version_id: str = "version-a") -> ClaimedIngestionJob:
    return ClaimedIngestionJob(
        tenant_id=tenant_id,
        job_id="job-a",
        version_id=version_id,
        document_id="document-a",
        version=1,
        source_object_key="object-a",
        source_file_name="source.md",
        content_hash=_hash("source"),
        index_version="index-v1",
        attempts=1,
        lease_owner="worker-a",
        lease_generation=1,
        projection_generation=1,
    )


def _parsed(job: ClaimedIngestionJob) -> ParsedDocument:
    first_text = "English travel policy 中文差旅"
    second_text = "second reimbursement paragraph"
    chunks = (
        ParsedChunk(f"{job.version_id}:chunk-000001", 1, first_text, _hash(first_text), "First"),
        ParsedChunk(f"{job.version_id}:chunk-000002", 2, second_text, _hash(second_text), "Second"),
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


class FakeEmbedder:
    def embed(self, texts):
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


class FakeMetadata:
    def __init__(self, principals=("role:employee", "user:alice")):
        self.principals = principals

    def get_metadata(self, job):
        return DocumentIndexMetadata("Travel Policy", "finance", self.principals)


@dataclass
class FakeRecord:
    id: str
    payload: dict
    vector: dict | None = None


class FakeQdrantClient:
    def __init__(self):
        self.exists = False
        self.created = None
        self.payload_indexes = []
        self.payload_schema = {}
        self.points: dict[str, FakeRecord] = {}

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.exists = True
        self.created = kwargs
        return True

    def get_collection(self, collection_name):
        return SimpleNamespace(
            payload_schema=self.payload_schema,
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
                    },
                    sparse_vectors={
                        "keyword": models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
            ),
        )

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)
        self.payload_schema[kwargs["field_name"]] = SimpleNamespace(
            data_type=kwargs["field_schema"]
        )

    def upsert(self, *, points, **kwargs):
        for point in points:
            self.points[str(point.id)] = FakeRecord(
                str(point.id), dict(point.payload or {}), dict(point.vector or {})
            )

    def delete(self, *, points_selector, **kwargs):
        expected_ids = {str(value) for condition in points_selector.must_not for value in condition.has_id}
        for point_id, record in list(self.points.items()):
            if self._matches(record.payload, points_selector.must) and point_id not in expected_ids:
                del self.points[point_id]

    def retrieve(self, *, ids, **kwargs):
        return [self.points[str(point_id)] for point_id in ids if str(point_id) in self.points]

    def scroll(self, *, scroll_filter, **kwargs):
        return ([record for record in self.points.values() if self._matches(record.payload, scroll_filter.must)], None)

    def set_payload(self, *, payload, points, **kwargs):
        for record in self.points.values():
            if self._matches(record.payload, points.must) and not self._matches_any(
                record.payload, points.must_not
            ):
                record.payload.update(payload)

    @staticmethod
    def _matches(payload, conditions):
        return all(payload.get(condition.key) == condition.match.value for condition in conditions or ())

    @staticmethod
    def _matches_any(payload, conditions):
        return any(payload.get(condition.key) == condition.match.value for condition in conditions or ())


def test_collection_is_named_dense_sparse_with_required_payload_indexes() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(client, FakeEmbedder(), FakeMetadata(), now=lambda: datetime(2026, 8, 22, tzinfo=UTC))
    job = _job()

    receipt = adapter.index(job, _parsed(job))

    assert client.created["vectors_config"]["dense"].size == 2
    assert client.created["sparse_vectors_config"]["keyword"].modifier == models.Modifier.IDF
    assert {call["field_name"] for call in client.payload_indexes} >= {
        "tenant_id", "version_id", "document_id", "lifecycle_status", "current", "allowed_principals", "index_version"
    }
    assert receipt.chunk_count == 2


def test_document_deletion_filters_by_tenant_and_document() -> None:
    class DeleteOnlyClient:
        def __init__(self) -> None:
            self.deleted = None

        def collection_exists(self, collection_name):
            return collection_name == "kb_chunks_v2"

        def delete(self, **kwargs):
            self.deleted = kwargs

    client = DeleteOnlyClient()

    QdrantDocumentDeletion(client).delete_document(
        tenant_id="tenant-a",
        document_id="document-a",
    )

    assert client.deleted["collection_name"] == "kb_chunks_v2"
    assert client.deleted["wait"] is True
    conditions = client.deleted["points_selector"].must
    assert [(item.key, item.match.value) for item in conditions] == [
        ("tenant_id", "tenant-a"),
        ("document_id", "document-a"),
    ]


def test_index_uses_tenant_version_chunk_uuid_and_removes_stale_points() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(client, FakeEmbedder(), FakeMetadata(), now=lambda: datetime(2026, 8, 22, tzinfo=UTC))
    job = _job()
    stale_id = "stale"
    client.points[stale_id] = FakeRecord(
        stale_id,
        {
            "tenant_id": "tenant-a",
            "version_id": "version-a",
            "projection_generation": 1,
            "projection_key": projection_key("tenant-a", "version-a", 1),
        },
    )

    adapter.index(job, _parsed(job))

    assert stale_id not in client.points
    expected_id = adapter.point_id(
        tenant_id="tenant-a", version_id="version-a", projection_generation=1,
        chunk_id="version-a:chunk-000001",
    )
    assert expected_id in client.points
    assert expected_id != adapter.point_id(
        tenant_id="tenant-b", version_id="version-a", projection_generation=1,
        chunk_id="version-a:chunk-000001",
    )
    payload = client.points[expected_id].payload
    assert payload["lifecycle_status"] == "indexing" and payload["current"] is False
    assert payload["allowed_principals"] == ["role:employee", "user:alice"]
    assert payload["projection_generation"] == 1
    assert payload["projection_key"] == projection_key("tenant-a", "version-a", 1)
    assert payload["text"] == "English travel policy 中文差旅"


def test_publish_is_idempotent_and_readback_verified() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(client, FakeEmbedder(), FakeMetadata(), now=lambda: datetime(2026, 8, 22, tzinfo=UTC))
    job = _job()
    receipt = adapter.index(job, _parsed(job))

    adapter.publish(job, receipt, lifecycle_status="active")
    adapter.publish(job, receipt, lifecycle_status="active")
    assert {record.payload["current"] for record in client.points.values()} == {True}
    adapter.publish(job, receipt, lifecycle_status="superseded")
    assert {record.payload["lifecycle_status"] for record in client.points.values()} == {"superseded"}
    assert {record.payload["current"] for record in client.points.values()} == {False}


def test_publishing_new_active_version_supersedes_old_projection() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(
        client,
        FakeEmbedder(),
        FakeMetadata(),
        now=lambda: datetime(2026, 8, 22, tzinfo=UTC),
    )
    old_job = _job(version_id="version-old")
    new_job = ClaimedIngestionJob(
        **{
            **old_job.__dict__,
            "job_id": "job-new",
            "version_id": "version-new",
            "version": 2,
        }
    )
    old_receipt = adapter.index(old_job, _parsed(old_job))
    adapter.publish(old_job, old_receipt, lifecycle_status="active")
    new_receipt = adapter.index(new_job, _parsed(new_job))
    adapter.publish(new_job, new_receipt, lifecycle_status="active")

    old_payloads = [
        record.payload for record in client.points.values()
        if record.payload["version_id"] == "version-old"
    ]
    new_payloads = [
        record.payload for record in client.points.values()
        if record.payload["version_id"] == "version-new"
    ]
    assert old_payloads and {item["lifecycle_status"] for item in old_payloads} == {"superseded"}
    assert {item["current"] for item in old_payloads} == {False}
    assert new_payloads and {item["lifecycle_status"] for item in new_payloads} == {"active"}
    assert {item["current"] for item in new_payloads} == {True}


def test_sparse_keyword_vector_is_deterministic_for_chinese_and_english() -> None:
    first = keyword_sparse_vector("Travel 差旅报销 travel")
    second = keyword_sparse_vector("Travel 差旅报销 travel")

    assert first == second
    assert first.indices and first.values
    assert first.indices == sorted(first.indices)


def test_invalid_principals_fail_closed_with_safe_code() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(client, FakeEmbedder(), FakeMetadata(("employee",)), now=lambda: datetime(2026, 8, 22, tzinfo=UTC))
    job = _job()

    with pytest.raises(DocumentIndexError, match="document_index_metadata_invalid"):
        adapter.index(job, _parsed(job))


def test_existing_collection_schema_mismatch_blocks_receipt() -> None:
    class BadSchemaClient(FakeQdrantClient):
        def __init__(self):
            super().__init__()
            self.exists = True

        def get_collection(self, collection_name):
            info = super().get_collection(collection_name)
            info.config.params.vectors["dense"] = models.VectorParams(
                size=3, distance=models.Distance.DOT
            )
            return info

    with pytest.raises(DocumentIndexError, match="collection_schema_invalid"):
        QdrantDocumentIndex(
            BadSchemaClient(), FakeEmbedder(), FakeMetadata()
        ).index(_job(), _parsed(_job()))


def test_missing_named_vector_in_readback_blocks_receipt() -> None:
    class MissingVectorClient(FakeQdrantClient):
        def retrieve(self, *, ids, **kwargs):
            records = super().retrieve(ids=ids, **kwargs)
            return [FakeRecord(item.id, item.payload, None) for item in records]

    with pytest.raises(DocumentIndexError, match="vector_readback_mismatch"):
        QdrantDocumentIndex(
            MissingVectorClient(), FakeEmbedder(), FakeMetadata()
        ).index(_job(), _parsed(_job()))
