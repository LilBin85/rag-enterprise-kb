"""带租户、SQL 当前版本和 ACL 主体三重过滤的 Qdrant 检索适配器。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from qdrant_client.http import models

from rag_kb.application.ports.retrieval import RankedChunk
from rag_kb.infrastructure.vectorstore.qdrant_document_index import (
    DenseEmbedderPort,
    _normalize_principals,
    keyword_sparse_vector,
    projection_key,
)


class AuthorizedSearchError(RuntimeError):
    """对外只传播稳定错误码，不包含查询正文或底层响应。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class QdrantAuthorizedSearch:
    def __init__(
        self,
        client,
        embedder: DenseEmbedderPort,
        *,
        collection: str = "kb_chunks_v2",
    ) -> None:
        if not collection:
            raise ValueError("authorized_search_collection_required")
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def search_dense(
        self,
        query: str,
        *,
        tenant_id: str,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]] = frozenset(),
        allowed_principals: frozenset[str],
        limit: int,
        principal_filter_required: bool = True,
        score_threshold: float | None = None,
    ) -> tuple[RankedChunk, ...]:
        try:
            vectors = list(self._embedder.embed([query]))
        except Exception as exc:
            raise AuthorizedSearchError("dense_query_embedding_failed") from exc
        if (
            len(vectors) != 1
            or not vectors[0]
            or any(not math.isfinite(float(value)) for value in vectors[0])
        ):
            raise AuthorizedSearchError("dense_query_embedding_invalid")
        return self._search(
            query_vector=list(vectors[0]),
            using="dense",
            tenant_id=tenant_id,
            version_ids=version_ids,
            projection_generations=projection_generations,
            allowed_principals=allowed_principals,
            limit=limit,
            principal_filter_required=principal_filter_required,
            score_threshold=score_threshold,
        )

    def search_sparse(
        self,
        query: str,
        *,
        tenant_id: str,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]] = frozenset(),
        allowed_principals: frozenset[str],
        limit: int,
        principal_filter_required: bool = True,
        score_threshold: float | None = None,
    ) -> tuple[RankedChunk, ...]:
        vector = keyword_sparse_vector(query)
        if not vector.indices:
            return ()
        return self._search(
            query_vector=vector,
            using="keyword",
            tenant_id=tenant_id,
            version_ids=version_ids,
            projection_generations=projection_generations,
            allowed_principals=allowed_principals,
            limit=limit,
            principal_filter_required=principal_filter_required,
            score_threshold=score_threshold,
        )

    def _search(
        self,
        *,
        query_vector: Sequence[float] | models.SparseVector,
        using: str,
        tenant_id: str,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]],
        allowed_principals: frozenset[str],
        limit: int,
        principal_filter_required: bool,
        score_threshold: float | None,
    ) -> tuple[RankedChunk, ...]:
        query_filter = self._build_filter(
            tenant_id=tenant_id,
            version_ids=version_ids,
            projection_generations=projection_generations,
            allowed_principals=allowed_principals,
            principal_filter_required=principal_filter_required,
        )
        if limit <= 0 or limit > 100:
            raise ValueError("authorized_search_limit_invalid")
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                using=using,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise AuthorizedSearchError("authorized_vector_search_failed") from exc
        try:
            return tuple(
                self._to_ranked(
                    point,
                    route=using,
                    projection_generations=projection_generations,
                )
                for point in response.points
            )
        except AuthorizedSearchError:
            raise
        except Exception as exc:
            raise AuthorizedSearchError("authorized_vector_payload_invalid") from exc

    @staticmethod
    def _build_filter(
        *,
        tenant_id: str,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]],
        allowed_principals: frozenset[str],
        principal_filter_required: bool,
    ) -> models.Filter:
        if not tenant_id or not version_ids or not projection_generations:
            raise AuthorizedSearchError("authorized_search_scope_empty")
        if (
            {version_id for version_id, _generation in projection_generations} != set(version_ids)
            or any(generation <= 0 for _version_id, generation in projection_generations)
        ):
            raise AuthorizedSearchError("authorized_projection_scope_invalid")
        must: list[models.Condition] = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(
                key="version_id", match=models.MatchAny(any=sorted(version_ids))
            ),
            models.FieldCondition(
                key="projection_key",
                match=models.MatchAny(
                    any=[
                        projection_key(tenant_id, version_id, generation)
                        for version_id, generation in sorted(projection_generations)
                    ]
                ),
            ),
            models.FieldCondition(
                key="lifecycle_status", match=models.MatchValue(value="active")
            ),
            models.FieldCondition(key="current", match=models.MatchValue(value=True)),
        ]
        if principal_filter_required:
            try:
                principals = _normalize_principals(tuple(allowed_principals))
            except Exception as exc:
                raise AuthorizedSearchError("authorized_search_principals_invalid") from exc
            must.append(
                models.FieldCondition(
                    key="allowed_principals",
                    match=models.MatchAny(any=list(principals)),
                )
            )
        return models.Filter(must=must)

    @staticmethod
    def _to_ranked(
        point,
        *,
        route: str,
        projection_generations: frozenset[tuple[str, int]],
    ) -> RankedChunk:
        payload = point.payload
        if not isinstance(payload, dict):
            raise AuthorizedSearchError("authorized_vector_payload_invalid")
        required_strings = (
            "tenant_id",
            "document_id",
            "version_id",
            "projection_key",
            "chunk_id",
            "content_hash",
            "source_locator",
            "source_file_name",
            "text",
        )
        if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
            raise AuthorizedSearchError("authorized_vector_payload_invalid")
        document_version = payload.get("document_version")
        projection_generation = payload.get("projection_generation")
        score = float(point.score)
        if (
            isinstance(document_version, bool)
            or not isinstance(document_version, int)
            or document_version <= 0
            or isinstance(projection_generation, bool)
            or not isinstance(projection_generation, int)
            or (payload["version_id"], projection_generation) not in projection_generations
            or payload["projection_key"]
            != projection_key(payload["tenant_id"], payload["version_id"], projection_generation)
            or not math.isfinite(score)
        ):
            raise AuthorizedSearchError("authorized_vector_payload_invalid")
        actual_content_hash = "sha256:" + hashlib.sha256(
            payload["text"].encode("utf-8")
        ).hexdigest()
        if actual_content_hash != payload["content_hash"]:
            raise AuthorizedSearchError("authorized_vector_content_hash_mismatch")
        return RankedChunk(
            tenant_id=payload["tenant_id"],
            document_id=payload["document_id"],
            version_id=payload["version_id"],
            document_version=document_version,
            projection_generation=projection_generation,
            chunk_id=payload["chunk_id"],
            content_hash=payload["content_hash"],
            source_locator=payload["source_locator"],
            source_file_name=payload["source_file_name"],
            text=payload["text"],
            dense_score=score if route == "dense" else None,
            sparse_score=score if route == "keyword" else None,
        )
