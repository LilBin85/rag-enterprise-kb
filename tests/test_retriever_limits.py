from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_kb.config import Settings, resolve_retrieval_limits
from rag_kb.models import Role
from rag_kb.retriever import Retriever


def _point(chunk_id: str, score: float):
    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "document_id": "policy",
            "document_version": 1,
            "title": "制度",
            "source_file_name": "policy.md",
            "source_locator": "第一章",
            "text": "不会写入报告的正文",
        },
    )


class _Client:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points[: kwargs["limit"]])


class _Embedder:
    def embed(self, texts):
        return [[0.1] for _text in texts]


def test_candidate_threshold_context_order_and_server_acl_are_explicit() -> None:
    client = _Client(
        [_point("a", 0.9), _point("b", 0.8), _point("c", 0.7), _point("d", 0.4)]
    )
    store = SimpleNamespace(client=client, collection="chunks")
    retriever = Retriever(
        store,
        _Embedder(),
        candidate_k=4,
        context_k=2,
        score_threshold=0.5,
    )

    result = retriever.search("问题", Role.HR)

    assert client.calls[0]["limit"] == 4
    conditions = {
        condition.key: condition.match.value
        for condition in client.calls[0]["query_filter"].must
    }
    assert conditions == {"status": "active", "allowed_roles": "hr"}
    assert [candidate.chunk_id for candidate in result.candidates] == ["a", "b"]
    assert result.candidate_count == 4
    assert result.suppressed_count == 1
    assert result.context_truncated_count == 1


def test_legacy_top_k_and_new_limit_resolution_are_backward_compatible() -> None:
    legacy = SimpleNamespace(retrieval_top_k=5)
    assert resolve_retrieval_limits(legacy) == (5, 5)
    gray = SimpleNamespace(
        retrieval_top_k=5, retrieval_candidate_k=10, retrieval_context_k=3
    )
    assert resolve_retrieval_limits(gray) == (10, 3)

    client = _Client([_point("a", 0.9), _point("b", 0.8), _point("c", 0.7)])
    retriever = Retriever(
        SimpleNamespace(client=client, collection="chunks"), _Embedder(), top_k=2
    )
    assert len(retriever.search("问题", Role.ADMIN).candidates) == 2
    assert client.calls[0]["limit"] == 2


def test_invalid_limit_relationship_fails_closed() -> None:
    with pytest.raises(ValueError, match="context_k_cannot_exceed_candidate_k"):
        resolve_retrieval_limits(
            SimpleNamespace(
                retrieval_top_k=5, retrieval_candidate_k=2, retrieval_context_k=3
            )
        )
    with pytest.raises(ValueError, match="context_k_cannot_exceed_candidate_k"):
        Retriever(
            SimpleNamespace(client=_Client([]), collection="chunks"),
            _Embedder(),
            candidate_k=2,
            context_k=3,
        )


@pytest.mark.parametrize("invalid", [0, False, 2.5])
def test_explicit_invalid_limits_do_not_fall_back_to_legacy(invalid) -> None:
    with pytest.raises(ValueError, match="retrieval_limits_must_be"):
        resolve_retrieval_limits(
            SimpleNamespace(
                retrieval_top_k=5,
                retrieval_candidate_k=invalid,
                retrieval_context_k=2,
            )
        )


@pytest.mark.parametrize("invalid", [False, 2.5])
def test_direct_retriever_rejects_non_integer_limits(invalid) -> None:
    with pytest.raises(ValueError, match="retrieval_limits_must_be_integers"):
        Retriever(
            SimpleNamespace(client=_Client([]), collection="chunks"),
            _Embedder(),
            candidate_k=invalid,
            context_k=1,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), False])
def test_non_finite_or_boolean_threshold_fails_closed(invalid) -> None:
    with pytest.raises(ValueError, match="retrieval_score_threshold_must_be_finite"):
        Retriever(
            SimpleNamespace(client=_Client([]), collection="chunks"),
            _Embedder(),
            score_threshold=invalid,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_non_finite_threshold(invalid) -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_score_threshold=invalid)
