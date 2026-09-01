"""直接消费完整 AuthContext 的企业检索编排。"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Sequence

from rag_kb.application.evidence_sufficiency import DeterministicEvidenceSufficiencyPolicy
from rag_kb.application.ports.retrieval import (
    AuthorizedVectorSearchPort,
    AuthorizationSnapshotPort,
    EvidenceSufficiencyPort,
    RankedChunk,
    RerankerPort,
    RetrieverPort,
)
from rag_kb.application.retrieval_fusion import (
    pack_context,
    reciprocal_rank_fusion,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.retrieval import RetrievalResult
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


class RetrievalSecurityError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@contextmanager
def _retrieval_stage(stage: str):
    started_at = perf_counter()
    error_code: str | None = None
    outcome = "succeeded"
    with observe_span(
        f"retrieval.{stage}",
        attributes={"component": "retrieval", "operation": "retrieval.search", "phase": stage},
    ) as span:
        try:
            yield span
        except Exception as exc:
            outcome = "failure"
            error_code = stable_error_code(exc)
            raise
        finally:
            duration_seconds = perf_counter() - started_at
            category = error_category_for(error_code) if error_code is not None else "none"
            finish_observed_span(
                span,
                outcome=outcome,
                error_category=category,
                error_code=error_code,
                attributes={"duration_ms": duration_seconds * 1_000},
            )
            telemetry = current_observability()
            if telemetry is not None:
                telemetry.record_retrieval_stage(
                    stage=stage,
                    outcome=outcome,
                    duration_seconds=duration_seconds,
                )


def _record_candidate_count(stage: str, count: int) -> None:
    telemetry = current_observability()
    if telemetry is not None:
        telemetry.record_retrieval_candidates(stage=stage, count=count)


class EnterpriseRetriever(RetrieverPort):
    def __init__(
        self,
        authorization: AuthorizationSnapshotPort,
        vectors: AuthorizedVectorSearchPort,
        *,
        candidate_k: int = 20,
        context_k: int = 5,
        context_token_budget: int = 4000,
        rrf_k: int = 60,
        hybrid_enabled: bool = False,
        dense_score_threshold: float | None = None,
        reranker: RerankerPort | None = None,
        sufficiency_policy: EvidenceSufficiencyPort | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        if candidate_k <= 0 or context_k <= 0 or context_k > candidate_k:
            raise ValueError("invalid_retrieval_limits")
        if context_token_budget <= 0:
            raise ValueError("invalid_context_token_budget")
        if dense_score_threshold is not None and (
            isinstance(dense_score_threshold, bool)
            or not math.isfinite(float(dense_score_threshold))
        ):
            raise ValueError("invalid_dense_score_threshold")
        self._authorization = authorization
        self._vectors = vectors
        self._candidate_k = candidate_k
        self._context_k = context_k
        self._context_token_budget = context_token_budget
        self._rrf_k = rrf_k
        self._hybrid_enabled = hybrid_enabled
        self._dense_score_threshold = dense_score_threshold
        self._reranker = reranker
        self._sufficiency_policy = sufficiency_policy or DeterministicEvidenceSufficiencyPolicy()
        self._token_estimator = token_estimator or self._estimate_tokens

    def search(self, query: str, auth: AuthContext) -> RetrievalResult:
        started_at = perf_counter()
        error_code: str | None = None
        outcome = "failure"
        with observe_span(
            "retrieval.search",
            attributes={"component": "retrieval", "operation": "retrieval.search"},
        ) as span:
            try:
                result = self._search(query, auth)
                outcome = "succeeded" if result.candidates else "empty"
                span.set_attribute("candidate_count", len(result.candidates))
                span.set_attribute("context_token_estimate", result.context_token_estimate)
                return result
            except Exception as exc:
                outcome = "failure"
                error_code = stable_error_code(exc)
                if isinstance(exc, RetrievalSecurityError):
                    kind = {
                        "authorization_tenant_binding_invalid": "tenant",
                        "vector_acl_filter_contract_violated": "acl",
                        "vector_content_hash_contract_violated": "content_hash",
                    }.get(error_code)
                    telemetry = current_observability()
                    if telemetry is not None and kind is not None:
                        telemetry.record_security_boundary_violation(kind=kind)
                raise
            finally:
                duration_seconds = perf_counter() - started_at
                category = error_category_for(error_code) if error_code is not None else "none"
                finish_observed_span(
                    span,
                    outcome=outcome,
                    error_category=category,
                    error_code=error_code,
                    attributes={"duration_ms": duration_seconds * 1_000},
                )
                log_event(
                    "retrieval.completed",
                    level=40 if error_code is not None else 20,
                    component="retrieval",
                    operation="retrieval.search",
                    outcome=outcome,
                    error_category=category,
                    error_code=error_code,
                    duration_ms=duration_seconds * 1_000,
                )

    def _search(self, query: str, auth: AuthContext) -> RetrievalResult:
        normalized = " ".join(query.split())
        fingerprint = "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if not normalized:
            return self._empty(fingerprint, "invalid_query")
        with _retrieval_stage("authorization"):
            if not auth.has_permission(Permission.KNOWLEDGE_READ):
                return self._empty(fingerprint, "knowledge_read_denied")
            authorized = self._authorization.authorized_active_versions(auth)
            if authorized.tenant_id != auth.tenant_id:
                raise RetrievalSecurityError("authorization_tenant_binding_invalid")
            if not authorized.version_ids:
                return self._empty(fingerprint, "no_authorized_evidence")

        with _retrieval_stage("dense") as dense_span:
            dense = self._vectors.search_dense(
                normalized,
                tenant_id=auth.tenant_id,
                version_ids=authorized.version_ids,
                projection_generations=authorized.projection_generations,
                allowed_principals=auth.acl_subjects,
                limit=self._candidate_k,
                principal_filter_required=not auth.has_permission(Permission.ADMIN),
                score_threshold=self._dense_score_threshold,
            )
            dense_span.set_attribute("candidate_count", len(dense))
        _record_candidate_count("dense", len(dense))

        with _retrieval_stage("sparse") as sparse_span:
            sparse = (
                self._vectors.search_sparse(
                    normalized,
                    tenant_id=auth.tenant_id,
                    version_ids=authorized.version_ids,
                    projection_generations=authorized.projection_generations,
                    allowed_principals=auth.acl_subjects,
                    limit=self._candidate_k,
                    principal_filter_required=not auth.has_permission(Permission.ADMIN),
                    score_threshold=None,
                )
                if self._hybrid_enabled
                else ()
            )
            sparse_span.set_attribute("candidate_count", len(sparse))
        _record_candidate_count("sparse", len(sparse))

        with _retrieval_stage("fuse") as fuse_span:
            self._validate_vector_boundary(
                auth,
                authorized.version_ids,
                authorized.projection_generations,
                dense,
                sparse,
            )
            fused = reciprocal_rank_fusion(dense, sparse, rrf_k=self._rrf_k)
            fuse_span.set_attribute("candidate_count", len(fused))
        _record_candidate_count("fused", len(fused))

        with _retrieval_stage("rerank"):
            if self._reranker is not None and fused:
                reranked = self._reranker.rerank(normalized, fused)
                self._validate_reranker_boundary(fused, reranked)
                fused = reranked

        with _retrieval_stage("revalidate") as revalidate_span:
            returned_versions = frozenset(candidate.version_id for candidate in fused)
            still_authorized = self._authorization.revalidate_versions(auth, returned_versions)
            revalidated = tuple(
                candidate for candidate in fused if candidate.version_id in still_authorized
            )
            exact_revalidator = getattr(self._authorization, "revalidate_candidates", None)
            if exact_revalidator is not None and revalidated:
                exact = exact_revalidator(auth, revalidated)
                revalidated = tuple(
                    candidate
                    for candidate in revalidated
                    if (candidate.version_id, candidate.chunk_id) in exact
                )
            scoped_revalidated = self._sufficiency_policy.select_candidates(
                normalized,
                revalidated,
            )
            revalidate_span.set_attribute("candidate_count", len(revalidated))
        _record_candidate_count("revalidated", len(revalidated))

        with _retrieval_stage("pack") as pack_span:
            packed = pack_context(
                scoped_revalidated,
                context_k=self._context_k,
                token_budget=self._context_token_budget,
                token_estimator=self._token_estimator,
            )
            if packed.candidates:
                insufficiency_code = self._sufficiency_policy.insufficiency_code(
                    normalized,
                    packed.candidates,
                )
                if insufficiency_code is not None:
                    packed = packed.model_copy(
                        update={
                            "candidates": (),
                            "context_token_estimate": 0,
                            "sufficient": False,
                            "insufficiency_code": insufficiency_code,
                        }
                    )
            pack_span.set_attribute("candidate_count", len(packed.candidates))
            pack_span.set_attribute("context_token_estimate", packed.context_token_estimate)
        _record_candidate_count("packed", len(packed.candidates))
        telemetry = current_observability()
        if telemetry is not None:
            telemetry.record_context_token_estimate(packed.context_token_estimate)
        return packed.model_copy(update={"query_fingerprint": fingerprint})

    @staticmethod
    def _validate_vector_boundary(
        auth: AuthContext,
        version_ids: frozenset[str],
        projection_generations: frozenset[tuple[str, int]],
        *rankings: Sequence[RankedChunk],
    ) -> None:
        for candidate in (item for ranking in rankings for item in ranking):
            if (
                candidate.tenant_id != auth.tenant_id
                or candidate.version_id not in version_ids
                or (candidate.version_id, candidate.projection_generation)
                not in projection_generations
            ):
                raise RetrievalSecurityError("vector_acl_filter_contract_violated")
            content_hash = "sha256:" + hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
            if candidate.content_hash != content_hash:
                raise RetrievalSecurityError("vector_content_hash_contract_violated")

    @staticmethod
    def _validate_reranker_boundary(
        original: Sequence[RankedChunk],
        reranked: Sequence[RankedChunk],
    ) -> None:
        original_by_id = {item.chunk_id: item for item in original}
        reranked_by_id = {item.chunk_id: item for item in reranked}
        if len(original_by_id) != len(original) or len(reranked_by_id) != len(reranked):
            raise RetrievalSecurityError("reranker_duplicate_candidate")
        if set(original_by_id) != set(reranked_by_id):
            raise RetrievalSecurityError("reranker_candidate_set_changed")
        for chunk_id, item in reranked_by_id.items():
            expected = original_by_id[chunk_id]
            metadata = (
                "tenant_id",
                "document_id",
                "version_id",
                "document_version",
                "projection_generation",
                "chunk_id",
                "content_hash",
                "source_locator",
                "source_file_name",
                "text",
            )
            if any(getattr(item, field) != getattr(expected, field) for field in metadata):
                raise RetrievalSecurityError("reranker_candidate_binding_changed")
            if item.rerank_score is not None and not math.isfinite(float(item.rerank_score)):
                raise RetrievalSecurityError("reranker_score_invalid")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # 保守、确定性的无模型估算；实际 tokenizer 可通过构造参数注入。
        return max(1, (len(text.encode("utf-8")) + 2) // 3)

    @staticmethod
    def _empty(fingerprint: str, code: str) -> RetrievalResult:
        return RetrievalResult(
            query_fingerprint=fingerprint,
            candidates=(),
            context_token_estimate=0,
            sufficient=False,
            insufficiency_code=code,
        )
