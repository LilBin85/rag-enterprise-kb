"""Stage 3 RRF 与上下文预算的纯逻辑契约测试。"""

from dataclasses import replace

import pytest

from rag_kb.application.retrieval_fusion import RankedChunk, pack_context, reciprocal_rank_fusion


def chunk(chunk_id: str, *, content_hash: str | None = None, text: str | None = None) -> RankedChunk:
    return RankedChunk(
        tenant_id="tenant-a",
        document_id="doc-a",
        version_id="version-a",
        document_version=1,
        chunk_id=chunk_id,
        content_hash=content_hash or f"hash-{chunk_id}",
        source_locator=f"policy/{chunk_id}",
        source_file_name="policy.md",
        text=text or chunk_id,
        dense_score=0.9,
        sparse_score=0.8,
    )


def test_rrf_rewards_two_route_overlap_and_deduplicates() -> None:
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("b"), chunk("c"), chunk("d")]

    result = reciprocal_rank_fusion(dense, sparse)

    assert [item.chunk_id for item in result] == ["b", "c", "a", "d"]
    assert result[0].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert result[0].dense_score == 0.9
    assert result[0].sparse_score == 0.8


def test_rrf_rejects_metadata_conflict_and_invalid_rrf_k() -> None:
    with pytest.raises(ValueError, match="metadata_conflict"):
        reciprocal_rank_fusion([chunk("a")], [replace(chunk("a"), tenant_id="tenant-b")])
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], [], rrf_k=0)


def test_rrf_is_stably_tie_broken_by_chunk_id() -> None:
    dense = [chunk("b"), chunk("a")]
    sparse = [chunk("a"), chunk("b")]
    assert [item.chunk_id for item in reciprocal_rank_fusion(dense, sparse)] == ["a", "b"]


def test_pack_context_deduplicates_chunk_and_content_hash_and_respects_budget() -> None:
    ranked = reciprocal_rank_fusion(
        [chunk("a", text="aa"), chunk("b", text="bbbb")],
        [chunk("a", text="aa"), chunk("c", content_hash="hash-b", text="duplicate")],
    )

    result = pack_context(
        ranked,
        context_k=5,
        token_budget=6,
        token_estimator=len,
    )

    assert [item.chunk_id for item in result.candidates] == ["a", "b"]
    assert result.context_token_estimate == 6
    assert result.context_token_estimate <= 6
    assert result.sufficient is True


def test_pack_context_skips_over_budget_items_and_returns_safe_empty_result() -> None:
    ranked = reciprocal_rank_fusion([chunk("a", text="long")], [])
    result = pack_context(ranked, context_k=2, token_budget=2, token_estimator=len)
    assert result.candidates == ()
    assert result.sufficient is False
    assert result.insufficiency_code == "context_token_budget_exceeded"

    empty = pack_context([], context_k=2, token_budget=2, token_estimator=len)
    assert empty.sufficient is False
    assert empty.insufficiency_code == "no_candidates"


@pytest.mark.parametrize("bad", [0, -1, False, "2"])
def test_pack_context_rejects_invalid_budget_or_context_limit(bad) -> None:
    with pytest.raises(ValueError):
        pack_context([chunk("a")], context_k=bad, token_budget=10, token_estimator=len)
    if not (isinstance(bad, int) and not isinstance(bad, bool) and bad == 0):
        with pytest.raises(ValueError):
            pack_context([chunk("a")], context_k=1, token_budget=bad, token_estimator=len)


def test_pack_context_fails_closed_for_invalid_token_estimator() -> None:
    with pytest.raises(ValueError, match="token_estimator"):
        pack_context([chunk("a")], context_k=1, token_budget=10, token_estimator=lambda _text: 0)
    with pytest.raises(ValueError, match="token_estimation"):
        pack_context([chunk("a")], context_k=1, token_budget=10, token_estimator=lambda _text: 1 / 0)
