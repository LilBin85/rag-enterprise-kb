"""检索核心与基础设施适配器共享的稳定契约。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class AuthorizedVersionSet:
    tenant_id: str
    version_ids: frozenset[str]
    projection_generations: frozenset[tuple[str, int]] = frozenset()


@dataclass(frozen=True)
class RankedChunk:
    """已通过调用方授权过滤的候选块及两路检索分数。"""

    tenant_id: str
    document_id: str
    version_id: str
    document_version: int
    chunk_id: str
    content_hash: str
    source_locator: str
    source_file_name: str
    text: str
    projection_generation: int = 1
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    fused_score: float = 0.0


class AuthorizationSnapshotPort(Protocol):
    def authorized_active_versions(self, auth: AuthContext) -> AuthorizedVersionSet: ...

    def revalidate_versions(
        self,
        auth: AuthContext,
        version_ids: frozenset[str],
    ) -> frozenset[str]: ...

    def revalidate_candidates(
        self,
        auth: AuthContext,
        candidates: Sequence[RankedChunk],
    ) -> frozenset[tuple[str, str]]: ...


class AuthorizedVectorSearchPort(Protocol):
    def search_dense(
        self,
        query: str,
        *,
        tenant_id: str,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]],
        allowed_principals: frozenset[str],
        limit: int,
        principal_filter_required: bool = True,
        score_threshold: float | None = None,
    ) -> tuple[RankedChunk, ...]: ...

    def search_sparse(
        self,
        query: str,
        *,
        tenant_id: str,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]],
        allowed_principals: frozenset[str],
        limit: int,
        principal_filter_required: bool = True,
        score_threshold: float | None = None,
    ) -> tuple[RankedChunk, ...]: ...


class RerankerPort(Protocol):
    def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> tuple[RankedChunk, ...]: ...


class EvidenceSufficiencyPort(Protocol):
    def select_candidates(self, query: str, candidates: Sequence[object]) -> tuple[object, ...]: ...

    def insufficiency_code(self, query: str, candidates: Sequence[object]) -> str | None: ...


class RetrieverPort(Protocol):
    def search(self, query: str, auth: AuthContext) -> RetrievalResult: ...
