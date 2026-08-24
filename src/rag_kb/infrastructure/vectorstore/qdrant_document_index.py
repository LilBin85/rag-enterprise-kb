"""Stage 3 独立 Qdrant 文档索引适配器。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Protocol

from qdrant_client.http import models

from rag_kb.application.ingestion_worker import ClaimedIngestionJob, DocumentIndexPort
from rag_kb.domain.indexing import (
    IndexReceipt,
    ParsedChunk,
    ParsedDocument,
    compute_chunk_manifest_digest,
)
from rag_kb.infrastructure.vectorstore.qdrant_collection import (
    DocumentIndexError,
    ensure_document_collection,
)


_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-enterprise-kb/document-index-v2")
_ASCII_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")
_PRINCIPAL_TYPES = frozenset({"user", "group", "role"})


@dataclass(frozen=True)
class DocumentIndexMetadata:
    title: str
    department: str
    allowed_principals: tuple[str, ...]


class DocumentIndexMetadataPort(Protocol):
    def get_metadata(self, job: ClaimedIngestionJob) -> DocumentIndexMetadata: ...


class DenseEmbedderPort(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class QdrantDocumentDeletion:
    """按租户和文档标识删除新版 collection 中的全部向量。"""

    def __init__(self, client, *, collection: str = "kb_chunks_v2") -> None:
        if not collection:
            raise ValueError("document_index_collection_required")
        self._client = client
        self._collection = collection

    def delete_document(self, *, tenant_id: str, document_id: str) -> None:
        try:
            if not self._client.collection_exists(self._collection):
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=tenant_id)
                        ),
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        ),
                    ]
                ),
                wait=True,
            )
        except Exception as exc:
            raise DocumentIndexError("document_index_delete_failed") from exc


@dataclass(frozen=True)
class PreparedDocumentIndex:
    points: tuple[models.PointStruct, ...]
    point_ids: tuple[str, ...]
    payloads_by_id: dict[str, dict]


def _normalize_principals(principals: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(principals)))
    if not normalized or any(
        not separator or subject_type not in _PRINCIPAL_TYPES or not subject_id
        for item in normalized
        for subject_type, separator, subject_id in (item.partition(":"),)
    ):
        raise DocumentIndexError("document_index_metadata_invalid")
    return normalized


def _keyword_tokens(text: str) -> tuple[str, ...]:
    normalized = text.lower()
    tokens = list(_ASCII_TOKEN_PATTERN.findall(normalized))
    for run in _CJK_PATTERN.findall(normalized):
        tokens.extend(run[index : index + 1] for index in range(len(run)))
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(tokens)


def keyword_sparse_vector(text: str) -> models.SparseVector:
    """用稳定 SHA-256 哈希构造可复现的中英文 keyword sparse vector。"""
    counts = Counter(_keyword_tokens(text))
    hashed_counts: dict[int, float] = {}
    for token, count in counts.items():
        index = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % (2**31 - 1)
        hashed_counts[index] = hashed_counts.get(index, 0.0) + float(count)
    indices = sorted(hashed_counts)
    return models.SparseVector(indices=indices, values=[hashed_counts[index] for index in indices])


def projection_key(tenant_id: str, version_id: str, generation: int) -> str:
    if not tenant_id or not version_id or generation <= 0:
        raise DocumentIndexError("projection_generation_invalid")
    encoded = json.dumps(
        [tenant_id, version_id, generation],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class QdrantDocumentIndex(DocumentIndexPort):
    """写入新 collection 的三阶段文档索引；不依赖旧 ``ChunkStore``。"""

    def __init__(
        self,
        client,
        embedder: DenseEmbedderPort,
        metadata_port: DocumentIndexMetadataPort,
        *,
        collection: str = "kb_chunks_v2",
        now: Callable[[], datetime] | None = None,
        batch_size: int = 64,
    ) -> None:
        if not collection:
            raise ValueError("document_index_collection_required")
        if batch_size <= 0 or batch_size > 1024:
            raise ValueError("invalid_document_index_batch_size")
        self._client = client
        self._embedder = embedder
        self._metadata_port = metadata_port
        self._collection = collection
        self._now = now or (lambda: datetime.now(UTC))
        self._batch_size = batch_size

    @staticmethod
    def point_id(
        *, tenant_id: str, version_id: str, projection_generation: int, chunk_id: str
    ) -> str:
        if projection_generation <= 0:
            raise DocumentIndexError("projection_generation_invalid")
        encoded = json.dumps(
            [tenant_id, version_id, projection_generation, chunk_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return str(uuid.uuid5(_POINT_NAMESPACE, encoded))

    def _load_metadata(
        self, job: ClaimedIngestionJob
    ) -> tuple[DocumentIndexMetadata, tuple[str, ...]]:
        try:
            metadata = self._metadata_port.get_metadata(job)
        except DocumentIndexError:
            raise
        except Exception as exc:
            raise DocumentIndexError("document_index_metadata_unavailable") from exc
        try:
            if not metadata.title or not metadata.department:
                raise DocumentIndexError("document_index_metadata_invalid")
            principals = _normalize_principals(metadata.allowed_principals)
        except DocumentIndexError:
            raise
        except Exception as exc:
            raise DocumentIndexError("document_index_metadata_invalid") from exc
        return metadata, principals

    def _embed_chunks(self, chunks: Sequence[ParsedChunk]) -> list[Sequence[float]]:
        try:
            vectors = list(self._embedder.embed([chunk.text for chunk in chunks]))
        except Exception as exc:
            raise DocumentIndexError("document_embedding_failed") from exc
        if len(vectors) != len(chunks) or not vectors:
            raise DocumentIndexError("dense_embedding_invalid")
        dimension = len(vectors[0])
        if dimension <= 0 or any(
            len(vector) != dimension
            or any(not math.isfinite(float(value)) for value in vector)
            for vector in vectors
        ):
            raise DocumentIndexError("dense_embedding_invalid")
        return vectors

    def _prepare_chunks(
        self,
        job: ClaimedIngestionJob,
        chunks: Sequence[ParsedChunk],
        vectors: Sequence[Sequence[float]],
        metadata: DocumentIndexMetadata,
        principals: tuple[str, ...],
    ) -> PreparedDocumentIndex:
        try:
            points: list[models.PointStruct] = []
            payloads_by_id: dict[str, dict] = {}
            for chunk, vector in zip(chunks, vectors, strict=True):
                point_id = self.point_id(
                    tenant_id=job.tenant_id,
                    version_id=job.version_id,
                    projection_generation=job.projection_generation or 0,
                    chunk_id=chunk.chunk_id,
                )
                payload = {
                    "tenant_id": job.tenant_id,
                    "version_id": job.version_id,
                    "document_id": job.document_id,
                    "document_version": job.version,
                    "projection_generation": job.projection_generation,
                    "projection_key": projection_key(
                        job.tenant_id, job.version_id, job.projection_generation or 0
                    ),
                    "chunk_id": chunk.chunk_id,
                    "content_hash": chunk.content_hash,
                    "source_locator": chunk.source_locator,
                    "source_file_name": job.source_file_name,
                    "text": chunk.text,
                    "title": metadata.title,
                    "department": metadata.department,
                    "allowed_principals": list(principals),
                    "index_version": job.index_version,
                    "lifecycle_status": "indexing",
                    "current": False,
                }
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"dense": list(vector), "keyword": keyword_sparse_vector(chunk.text)},
                        payload=payload,
                    )
                )
                payloads_by_id[point_id] = payload
        except Exception as exc:
            raise DocumentIndexError("document_index_prepare_failed") from exc
        return PreparedDocumentIndex(tuple(points), tuple(payloads_by_id), payloads_by_id)

    def index(self, job: ClaimedIngestionJob, parsed_document: ParsedDocument) -> IndexReceipt:
        try:
            self._validate_parsed_document(job, parsed_document)
            metadata, principals = self._load_metadata(job)
            dimension: int | None = None
            for start in range(0, len(parsed_document.chunks), self._batch_size):
                chunks = parsed_document.chunks[start : start + self._batch_size]
                vectors = self._embed_chunks(chunks)
                batch_dimension = len(vectors[0])
                if dimension is None:
                    dimension = batch_dimension
                    self._ensure_collection(dimension)
                    self._delete_generation(job)
                elif batch_dimension != dimension:
                    raise DocumentIndexError("dense_embedding_invalid")
                prepared = self._prepare_chunks(job, chunks, vectors, metadata, principals)
                self._client.upsert(
                    collection_name=self._collection,
                    points=list(prepared.points),
                    wait=True,
                )
                self._validate_indexed_batch(prepared)
            count, manifest, payloads_valid = self._projection_summary(
                job,
                lifecycle_status="indexing",
                current=False,
            )
            if (
                count != len(parsed_document.chunks)
                or manifest != parsed_document.manifest_digest
                or not payloads_valid
            ):
                raise DocumentIndexError("document_index_readback_mismatch")
        except DocumentIndexError:
            raise
        except Exception as exc:
            raise DocumentIndexError("document_index_write_failed") from exc
        verified_at = self._now()
        if verified_at.tzinfo is None or verified_at.utcoffset() is None:
            raise DocumentIndexError("document_index_clock_invalid")
        return IndexReceipt(
            tenant_id=job.tenant_id,
            job_id=job.job_id,
            version_id=job.version_id,
            source_content_hash=job.content_hash,
            index_version=job.index_version,
            projection_generation=job.projection_generation or 0,
            chunk_count=len(parsed_document.chunks),
            manifest_digest=parsed_document.manifest_digest,
            verified_at=verified_at,
        )

    def publish(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        lifecycle_status: str,
    ) -> None:
        if lifecycle_status not in {"active", "superseded"}:
            raise DocumentIndexError("document_index_lifecycle_invalid")
        self._validate_receipt(job, receipt)
        expected_current = lifecycle_status == "active"
        version_filter = self._version_filter(job)
        try:
            metadata, principals = self._load_metadata(job)
            self._client.set_payload(
                collection_name=self._collection,
                payload={"lifecycle_status": lifecycle_status, "current": expected_current},
                points=version_filter,
                wait=True,
            )
            count, manifest, payloads_valid = self._projection_summary(
                job,
                lifecycle_status=lifecycle_status,
                current=expected_current,
                metadata=metadata,
                principals=principals,
            )
        except Exception as exc:
            raise DocumentIndexError("document_index_publish_failed") from exc
        if (
            count != receipt.chunk_count
            or manifest != receipt.manifest_digest
            or not payloads_valid
        ):
            raise DocumentIndexError("document_index_publish_verification_failed")

    def verify(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> bool:
        """轻量核对 SQL 已绑定的精确世代，不读取向量或重新嵌入。"""
        self._validate_receipt(job, receipt)
        try:
            if not self._client.collection_exists(self._collection):
                return False
            metadata, principals = self._load_metadata(job)
            count, manifest, payloads_valid = self._projection_summary(
                job,
                lifecycle_status="active",
                current=True,
                metadata=metadata,
                principals=principals,
            )
        except Exception as exc:
            raise DocumentIndexError("document_index_audit_failed") from exc
        return (
            count == receipt.chunk_count
            and manifest == receipt.manifest_digest
            and payloads_valid
        )

    def retire(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> None:
        """SQL 绑定成功后删除同文档除精确当前世代外的所有旧投影。"""
        self._validate_receipt(job, receipt)
        try:
            if not self.verify(job, receipt):
                raise DocumentIndexError("document_index_retire_target_invalid")
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=job.tenant_id)
                        ),
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=job.document_id)
                        ),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="version_id", match=models.MatchValue(value=job.version_id)
                        )
                    ],
                ),
                wait=True,
            )
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=job.tenant_id)
                        ),
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=job.document_id)
                        ),
                        models.FieldCondition(
                            key="version_id", match=models.MatchValue(value=job.version_id)
                        ),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="projection_generation",
                            match=models.MatchValue(value=receipt.projection_generation),
                        )
                    ],
                ),
                wait=True,
            )
            if self._has_old_document_points(job, receipt):
                raise DocumentIndexError("document_index_retire_verification_failed")
        except DocumentIndexError:
            raise
        except Exception as exc:
            raise DocumentIndexError("document_index_retire_failed") from exc

    def discard(self, job: ClaimedIngestionJob) -> None:
        """删除一次未绑定成功的目标世代。"""
        try:
            if self._client.collection_exists(self._collection):
                self._delete_generation(job)
        except Exception as exc:
            raise DocumentIndexError("document_index_discard_failed") from exc

    def discard_unbound(self, job: ClaimedIngestionJob) -> None:
        """重试前清除硬崩溃遗留，仅保留 SQL 当前精确绑定。"""
        must_not = []
        if (
            job.previous_projection_version_id is not None
            and job.previous_projection_generation is not None
        ):
            must_not.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="version_id",
                            match=models.MatchValue(value=job.previous_projection_version_id),
                        ),
                        models.FieldCondition(
                            key="projection_generation",
                            match=models.MatchValue(value=job.previous_projection_generation),
                        ),
                    ]
                )
            )
        try:
            if not self._client.collection_exists(self._collection):
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=job.tenant_id)
                        ),
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=job.document_id)
                        ),
                    ],
                    must_not=must_not or None,
                ),
                wait=True,
            )
        except Exception as exc:
            raise DocumentIndexError("document_index_discard_failed") from exc

    def _ensure_collection(self, dimension: int) -> None:
        ensure_document_collection(
            self._client, collection=self._collection, dimension=dimension
        )

    def _delete_generation(self, job: ClaimedIngestionJob) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._version_filter(job),
            wait=True,
        )

    def _iter_point_pages(self, scroll_filter: models.Filter) -> Iterator[Sequence]:
        offset = None
        while True:
            page, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                limit=self._batch_size,
                offset=offset,
            )
            yield page
            if offset is None:
                return

    def _projection_summary(
        self,
        job: ClaimedIngestionJob,
        *,
        lifecycle_status: str,
        current: bool,
        metadata: DocumentIndexMetadata | None = None,
        principals: tuple[str, ...] | None = None,
    ) -> tuple[int, str, bool]:
        entries = []
        payloads_valid = True
        expected_key = projection_key(
            job.tenant_id,
            job.version_id,
            job.projection_generation or 0,
        )
        for page in self._iter_point_pages(self._version_filter(job)):
            for point in page:
                payload = point.payload or {}
                payloads_valid = payloads_valid and (
                    payload.get("tenant_id") == job.tenant_id
                    and payload.get("document_id") == job.document_id
                    and payload.get("version_id") == job.version_id
                    and payload.get("projection_generation") == job.projection_generation
                    and payload.get("projection_key") == expected_key
                    and payload.get("lifecycle_status") == lifecycle_status
                    and payload.get("current") is current
                )
                if metadata is not None and principals is not None:
                    text = payload.get("text")
                    payloads_valid = payloads_valid and (
                        payload.get("document_version") == job.version
                        and payload.get("source_file_name") == job.source_file_name
                        and payload.get("index_version") == job.index_version
                        and payload.get("title") == metadata.title
                        and payload.get("department") == metadata.department
                        and payload.get("allowed_principals") == list(principals)
                        and isinstance(text, str)
                        and payload.get("content_hash")
                        == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
                    )
                entries.append(
                    SimpleNamespace(
                        chunk_id=str(payload.get("chunk_id", "")),
                        content_hash=str(payload.get("content_hash", "")),
                        source_locator=str(payload.get("source_locator", "")),
                    )
                )
        return len(entries), compute_chunk_manifest_digest(tuple(entries)), payloads_valid

    def _has_old_document_points(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
    ) -> bool:
        document_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id", match=models.MatchValue(value=job.tenant_id)
                ),
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=job.document_id)
                ),
            ]
        )
        return any(
            (point.payload or {}).get("version_id") != job.version_id
            or (point.payload or {}).get("projection_generation")
            != receipt.projection_generation
            for page in self._iter_point_pages(document_filter)
            for point in page
        )

    @staticmethod
    def _version_filter(job: ClaimedIngestionJob) -> models.Filter:
        if job.projection_generation is None or job.projection_generation <= 0:
            raise DocumentIndexError("projection_generation_invalid")
        return models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=job.tenant_id)),
                models.FieldCondition(key="version_id", match=models.MatchValue(value=job.version_id)),
                models.FieldCondition(
                    key="projection_generation",
                    match=models.MatchValue(value=job.projection_generation),
                ),
            ]
        )

    @staticmethod
    def _validate_parsed_document(job: ClaimedIngestionJob, parsed: ParsedDocument) -> None:
        if (
            job.projection_generation is None
            or job.projection_generation <= 0
            or
            parsed.tenant_id != job.tenant_id
            or parsed.document_id != job.document_id
            or parsed.version_id != job.version_id
            or parsed.document_version != job.version
            or parsed.source_file_name != job.source_file_name
        ):
            raise DocumentIndexError("parsed_document_mismatch")
        if any(
            "sha256:" + hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            != chunk.content_hash
            for chunk in parsed.chunks
        ):
            raise DocumentIndexError("parsed_chunk_content_hash_mismatch")

    @staticmethod
    def _validate_receipt(job: ClaimedIngestionJob, receipt: IndexReceipt) -> None:
        if (
            receipt.tenant_id != job.tenant_id
            or receipt.job_id != job.job_id
            or receipt.version_id != job.version_id
            or receipt.source_content_hash != job.content_hash
            or receipt.index_version != job.index_version
            or receipt.projection_generation != job.projection_generation
        ):
            raise DocumentIndexError("invalid_index_receipt")

    def _validate_indexed_batch(self, prepared: PreparedDocumentIndex) -> None:
        retrieved = self._client.retrieve(
            collection_name=self._collection,
            ids=list(prepared.point_ids),
            with_payload=True,
            with_vectors=True,
        )
        if {str(point.id) for point in retrieved} != set(prepared.point_ids):
            raise DocumentIndexError("document_index_readback_mismatch")
        for point in retrieved:
            expected = prepared.payloads_by_id[str(point.id)]
            payload = point.payload or {}
            if any(payload.get(key) != value for key, value in expected.items()):
                raise DocumentIndexError("document_index_readback_mismatch")
        expected_points = {str(point.id): point for point in prepared.points}
        for point in retrieved:
            expected_vector = expected_points[str(point.id)].vector
            actual_vector = point.vector
            if not isinstance(actual_vector, dict) or not isinstance(expected_vector, dict):
                raise DocumentIndexError("document_index_vector_readback_mismatch")
            actual_dense = actual_vector.get("dense")
            expected_dense = expected_vector.get("dense")
            if (
                not isinstance(actual_dense, list)
                or not isinstance(expected_dense, list)
                or len(actual_dense) != len(expected_dense)
                or any(
                    not math.isfinite(float(value))
                    or not math.isclose(
                        float(value), float(expected), rel_tol=1e-5, abs_tol=1e-6
                    )
                    for value, expected in zip(actual_dense, expected_dense, strict=True)
                )
            ):
                raise DocumentIndexError("document_index_vector_readback_mismatch")
            actual_sparse = QdrantDocumentIndex._sparse_components(actual_vector.get("keyword"))
            expected_sparse = QdrantDocumentIndex._sparse_components(expected_vector.get("keyword"))
            if actual_sparse != expected_sparse or not actual_sparse[0]:
                raise DocumentIndexError("document_index_vector_readback_mismatch")

    @staticmethod
    def _sparse_components(value) -> tuple[tuple[int, ...], tuple[float, ...]]:
        if isinstance(value, models.SparseVector):
            indices, values = value.indices, value.values
        elif isinstance(value, dict):
            indices, values = value.get("indices"), value.get("values")
        else:
            return (), ()
        if (
            not isinstance(indices, list)
            or not isinstance(values, list)
            or len(indices) != len(values)
            or any(not math.isfinite(float(item)) for item in values)
        ):
            return (), ()
        return tuple(int(item) for item in indices), tuple(round(float(item), 6) for item in values)
