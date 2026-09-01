from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from rag_kb.application.ports.indexing import (
    ClaimedIngestionJob,
    DocumentIndexError,
    DocumentIndexMetadata,
)
from rag_kb.domain.indexing import ParsedChunk, ParsedDocument, compute_chunk_manifest_digest
from rag_kb.infrastructure.vectorstore.qdrant_document_index import (
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
        self.upsert_sizes = []
        self.retrieve_sizes = []
        self.scroll_limits = []

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
        self.upsert_sizes.append(len(points))
        for point in points:
            self.points[str(point.id)] = FakeRecord(
                str(point.id), dict(point.payload or {}), dict(point.vector or {})
            )

    def delete(self, *, points_selector, **kwargs):
        for point_id, record in list(self.points.items()):
            if self._matches(record.payload, points_selector.must) and not self._matches_any(
                record.payload, points_selector.must_not
            ):
                del self.points[point_id]

    def retrieve(self, *, ids, **kwargs):
        self.retrieve_sizes.append(len(ids))
        return [self.points[str(point_id)] for point_id in ids if str(point_id) in self.points]

    def scroll(self, *, scroll_filter, limit, offset=None, **kwargs):
        self.scroll_limits.append(limit)
        records = [
            record
            for record in self.points.values()
            if self._matches(record.payload, scroll_filter.must)
            and not self._matches_any(record.payload, scroll_filter.must_not)
        ]
        start = int(offset or 0)
        page = records[start : start + limit]
        next_offset = start + limit if start + limit < len(records) else None
        return page, next_offset

    def set_payload(self, *, payload, points, **kwargs):
        for record in self.points.values():
            if self._matches(record.payload, points.must) and not self._matches_any(
                record.payload, points.must_not
            ):
                record.payload.update(payload)

    @staticmethod
    def _matches(payload, conditions):
        return all(
            FakeQdrantClient._matches_filter(payload, condition)
            for condition in conditions or ()
        )

    @staticmethod
    def _matches_any(payload, conditions):
        return any(
            FakeQdrantClient._matches_filter(payload, condition)
            for condition in conditions or ()
        )

    @staticmethod
    def _matches_filter(payload, condition):
        if isinstance(condition, models.Filter):
            return (
                FakeQdrantClient._matches(payload, condition.must)
                and not FakeQdrantClient._matches_any(payload, condition.must_not)
            )
        return payload.get(condition.key) == condition.match.value


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


def test_discard_unbound_preserves_only_applied_generation() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(
        client,
        FakeEmbedder(),
        FakeMetadata(),
        now=lambda: datetime(2026, 8, 22, tzinfo=UTC),
    )
    active = _job()
    orphan = replace(active, job_id="job-orphan", projection_generation=2)
    cleanup = replace(
        active,
        job_id="job-cleanup",
        projection_generation=3,
        previous_projection_generation=1,
        previous_projection_version_id=active.version_id,
    )
    adapter.index(active, _parsed(active))
    adapter.index(orphan, _parsed(orphan))

    adapter.discard_unbound(cleanup)

    assert {
        record.payload["projection_generation"] for record in client.points.values()
    } == {1}


def test_audit_rejects_acl_payload_drift_and_retirement_fails_closed() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(
        client,
        FakeEmbedder(),
        FakeMetadata(),
        now=lambda: datetime(2026, 8, 22, tzinfo=UTC),
    )
    job = _job()
    receipt = adapter.index(job, _parsed(job))
    adapter.publish(job, receipt, lifecycle_status="active")
    for record in client.points.values():
        record.payload["allowed_principals"] = ["role:admin"]

    assert adapter.verify(job, receipt) is False
    with pytest.raises(DocumentIndexError, match="retire_target_invalid"):
        adapter.retire(job, receipt)
    assert len(client.points) == receipt.chunk_count


def test_generation_switch_keeps_bound_projection_visible_until_old_is_retired() -> None:
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
            "projection_generation": 2,
        }
    )
    old_receipt = adapter.index(old_job, _parsed(old_job))
    adapter.publish(old_job, old_receipt, lifecycle_status="active")
    new_receipt = adapter.index(new_job, _parsed(new_job))
    adapter.publish(new_job, new_receipt, lifecycle_status="active")

    # 新世代可见但 SQL 尚绑定时，精确旧世代仍可检索；CAS 后精确新世代立即可检索。
    current_by_generation = {
        generation: [
            record
            for record in client.points.values()
            if record.payload["projection_generation"] == generation
            and record.payload["current"] is True
        ]
        for generation in (old_receipt.projection_generation, new_receipt.projection_generation)
    }
    assert current_by_generation[old_receipt.projection_generation]
    assert current_by_generation[new_receipt.projection_generation]

    old_payloads = [
        record.payload for record in client.points.values()
        if record.payload["version_id"] == "version-old"
    ]
    new_payloads = [
        record.payload for record in client.points.values()
        if record.payload["version_id"] == "version-new"
    ]
    assert old_payloads and {item["lifecycle_status"] for item in old_payloads} == {"active"}
    assert {item["current"] for item in old_payloads} == {True}
    assert new_payloads and {item["lifecycle_status"] for item in new_payloads} == {"active"}
    assert {item["current"] for item in new_payloads} == {True}

    adapter.retire(new_job, new_receipt)

    assert {
        record.payload["projection_generation"] for record in client.points.values()
    } == {new_receipt.projection_generation}


def test_large_document_embedding_upsert_and_readback_are_bounded() -> None:
    class RecordingEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            self.batch_sizes = []

        def embed(self, texts):
            self.batch_sizes.append(len(texts))
            return super().embed(texts)

    client = FakeQdrantClient()
    embedder = RecordingEmbedder()
    adapter = QdrantDocumentIndex(
        client,
        embedder,
        FakeMetadata(),
        batch_size=2,
        now=lambda: datetime(2026, 8, 22, tzinfo=UTC),
    )
    job = _job()
    chunks = tuple(
        ParsedChunk(
            f"{job.version_id}:chunk-{index:06d}",
            index,
            f"chunk text {index}",
            _hash(f"chunk text {index}"),
            f"section/{index}",
        )
        for index in range(1, 8)
    )
    document = ParsedDocument(
        tenant_id=job.tenant_id,
        document_id=job.document_id,
        version_id=job.version_id,
        document_version=job.version,
        source_file_name=job.source_file_name,
        chunks=chunks,
        manifest_digest=compute_chunk_manifest_digest(chunks),
    )

    receipt = adapter.index(job, document)

    assert receipt.chunk_count == 7
    assert max(embedder.batch_sizes) <= 2
    assert max(client.upsert_sizes) <= 2
    assert max(client.retrieve_sizes) <= 2
    assert max(client.scroll_limits) <= 2


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


def test_principal_identifier_may_contain_identity_provider_namespace() -> None:
    client = FakeQdrantClient()
    adapter = QdrantDocumentIndex(
        client,
        FakeEmbedder(),
        FakeMetadata(("user:idp:alice",)),
        now=lambda: datetime(2026, 8, 22, tzinfo=UTC),
    )
    job = _job()

    assert adapter.index(job, _parsed(job)).chunk_count == 2


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
