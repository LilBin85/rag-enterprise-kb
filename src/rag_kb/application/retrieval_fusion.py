"""Stage 3 的纯逻辑检索融合与上下文打包。

本模块不执行授权过滤，也不读取数据库或 Qdrant。调用方必须保证 dense
与 sparse 两路输入都已经过同一套 tenant、active、allowed-principals
服务端过滤；这里仅处理已授权候选的排序、去重和预算裁切。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from rag_kb.application.ports.retrieval import RankedChunk
from rag_kb.domain.retrieval import RetrievalCandidate, RetrievalResult


def reciprocal_rank_fusion(
    dense: Sequence[RankedChunk],
    sparse: Sequence[RankedChunk],
    *,
    rrf_k: int = 60,
) -> tuple[RankedChunk, ...]:
    """按 chunk_id 合并两路排名，使用 ``1 / (rrf_k + rank)`` 累计得分。

    同一路中同一 chunk 只计首次出现的排名。相同 chunk 的来源元数据必须
    完全一致；分数字段属于路由观测值，不参与元数据冲突判断。结果按融合
    分数降序、chunk_id 升序稳定排序。
    """

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k_must_be_positive_integer")

    merged: dict[str, RankedChunk] = {}
    fused_scores: dict[str, float] = {}
    route_scores: dict[str, dict[str, float | None]] = {}

    for route, candidates in (("dense", dense), ("sparse", sparse)):
        seen: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            _validate_score(candidate, route)
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            existing = merged.get(candidate.chunk_id)
            if existing is not None:
                _ensure_metadata_match(existing, candidate)
            else:
                merged[candidate.chunk_id] = candidate
                route_scores[candidate.chunk_id] = {"dense": None, "sparse": None}
            route_scores[candidate.chunk_id][route] = (
                candidate.dense_score if route == "dense" else candidate.sparse_score
            )
            fused_scores[candidate.chunk_id] = fused_scores.get(candidate.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )

    output = []
    for chunk_id, candidate in merged.items():
        scores = route_scores[chunk_id]
        output.append(
            RankedChunk(
                **{
                    **_metadata_values(candidate),
                    "dense_score": scores["dense"],
                    "sparse_score": scores["sparse"],
                    "rerank_score": candidate.rerank_score,
                    "fused_score": fused_scores[chunk_id],
                }
            )
        )
    return tuple(sorted(output, key=lambda item: (-item.fused_score, item.chunk_id)))


def pack_context(
    ranked: Sequence[RankedChunk],
    *,
    context_k: int,
    token_budget: int,
    token_estimator: Callable[[str], int],
) -> RetrievalResult:
    """按融合顺序去重并裁切上下文，保证 token 总量不超过预算。"""

    if isinstance(context_k, bool) or not isinstance(context_k, int) or context_k <= 0:
        raise ValueError("context_k_must_be_positive_integer")
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0:
        raise ValueError("token_budget_must_be_nonnegative_integer")

    candidates: list[RetrievalCandidate] = []
    seen_chunks: set[str] = set()
    seen_content: set[str] = set()
    total_tokens = 0
    for item in ranked:
        if item.chunk_id in seen_chunks or item.content_hash in seen_content:
            continue
        seen_chunks.add(item.chunk_id)
        seen_content.add(item.content_hash)
        try:
            estimated = token_estimator(item.text)
        except Exception as exc:
            raise ValueError("token_estimation_failed") from exc
        if isinstance(estimated, bool) or not isinstance(estimated, int) or estimated <= 0:
            raise ValueError("token_estimator_returned_nonpositive_value")
        if len(candidates) >= context_k or total_tokens + estimated > token_budget:
            continue
        candidates.append(
            RetrievalCandidate(
                tenant_id=item.tenant_id,
                document_id=item.document_id,
                version_id=item.version_id,
                document_version=item.document_version,
                projection_generation=item.projection_generation,
                chunk_id=item.chunk_id,
                content_hash=item.content_hash,
                source_locator=item.source_locator,
                source_file_name=item.source_file_name,
                dense_score=item.dense_score,
                sparse_score=item.sparse_score,
                fused_score=item.fused_score,
                rerank_score=item.rerank_score,
                text=item.text,
            )
        )
        total_tokens += estimated

    if not candidates:
        insufficiency = "no_candidates" if not ranked else "context_token_budget_exceeded"
        return RetrievalResult(
            query_fingerprint="",
            candidates=(),
            context_token_estimate=0,
            sufficient=False,
            insufficiency_code=insufficiency,
        )
    return RetrievalResult(
        query_fingerprint="",
        candidates=tuple(candidates),
        context_token_estimate=total_tokens,
        sufficient=True,
        insufficiency_code=None,
    )


def _metadata_values(candidate: RankedChunk) -> dict[str, object]:
    return {
        "tenant_id": candidate.tenant_id,
        "document_id": candidate.document_id,
        "version_id": candidate.version_id,
        "document_version": candidate.document_version,
        "projection_generation": candidate.projection_generation,
        "chunk_id": candidate.chunk_id,
        "content_hash": candidate.content_hash,
        "source_locator": candidate.source_locator,
        "source_file_name": candidate.source_file_name,
        "text": candidate.text,
    }


def _ensure_metadata_match(left: RankedChunk, right: RankedChunk) -> None:
    if _metadata_values(left) != _metadata_values(right):
        raise ValueError("ranked_chunk_metadata_conflict")


def _validate_score(candidate: RankedChunk, route: str) -> None:
    score = candidate.dense_score if route == "dense" else candidate.sparse_score
    if score is not None and (isinstance(score, bool) or not math.isfinite(float(score))):
        raise ValueError("ranked_chunk_score_must_be_finite")
