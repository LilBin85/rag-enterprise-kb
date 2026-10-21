"""BgeReranker 适配器行为：按交叉编码分数重排，失败回落融合顺序。"""

from __future__ import annotations

import hashlib

from rag_kb.application.ports.retrieval import RankedChunk
from rag_kb.infrastructure.retrieval.reranker import BgeReranker


def _chunk(chunk_id: str, text: str) -> RankedChunk:
    return RankedChunk(
        tenant_id="tenant-a",
        document_id="policy",
        version_id="version-1",
        document_version=1,
        chunk_id=chunk_id,
        content_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_locator=f"section/{chunk_id}",
        source_file_name="policy.md",
        text=text,
        dense_score=0.5,
        fused_score=0.1,
    )


def test_reranks_by_cross_encoder_score_without_changing_bindings() -> None:
    reranker = BgeReranker("test/model")
    reranker._model = object()  # type: ignore[assignment]
    reranker._scores = lambda _query, _candidates: [0.1, 0.9]  # type: ignore[method-assign]
    first, second = _chunk("chunk-1", "甲"), _chunk("chunk-2", "乙")

    reranked = reranker.rerank("q", (first, second))

    assert [item.chunk_id for item in reranked] == ["chunk-2", "chunk-1"]
    assert [item.rerank_score for item in reranked] == [0.9, 0.1]
    assert reranked[1].dense_score == 0.5 and reranked[1].text == "甲"


def test_unavailable_or_invalid_scores_fall_back_to_fusion_order() -> None:
    first, second = _chunk("chunk-1", "甲"), _chunk("chunk-2", "乙")

    def _offline(self):
        raise RuntimeError("offline")

    reranker = BgeReranker("test/model")
    reranker._load = _offline.__get__(reranker)  # type: ignore[method-assign]
    assert reranker.rerank("q", (first, second)) == (first, second)
    # 失败后本进程内不再重试加载
    reranker._load = _offline.__get__(reranker)  # type: ignore[method-assign]
    assert reranker.rerank("q", (first, second)) == (first, second)

    nan_reranker = BgeReranker("test/model")
    nan_reranker._model = object()  # type: ignore[assignment]
    nan_reranker._scores = lambda _query, _candidates: [float("nan"), 0.5]  # type: ignore[method-assign]
    assert nan_reranker.rerank("q", (first, second)) == (first, second)
