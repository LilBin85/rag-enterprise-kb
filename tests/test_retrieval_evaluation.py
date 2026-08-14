import math

import pytest

from rag_kb.models import Role
from rag_kb.retriever import (
    RetrievalCandidate,
    RetrievalEvaluationCase,
    RetrievalResult,
    evaluate_retriever,
)


def _candidate(chunk_id: str, *, document_id: str = "policy", version: int = 2):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=version,
        title="制度",
        source_file_name="policy.md",
        source_locator="条款",
        score=0.9,
        text="完整制度正文。",
    )


class FrozenRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question, role):
        self.calls.append((question, role))
        return RetrievalResult(self.results[question], 0)


def test_evaluate_retriever_computes_quality_acl_refusal_and_version_metrics() -> None:
    retriever = FrozenRetriever(
        {
            "普通正样本": [_candidate("noise"), _candidate("a"), _candidate("c")],
            "ACL 正样本": [_candidate("d", document_id="leave", version=3)],
            "无答案": [],
        }
    )
    cases = [
        RetrievalEvaluationCase(
            "positive",
            "普通正样本",
            Role.HR,
            frozenset({"a", "c"}),
            expected_document_versions=frozenset({("policy", 2)}),
        ),
        RetrievalEvaluationCase(
            "acl-positive",
            "ACL 正样本",
            Role.HR,
            frozenset({"d"}),
            is_acl_filtered=True,
            expected_document_versions=frozenset({("leave", 3)}),
        ),
        RetrievalEvaluationCase(
            "acl-denied",
            "无答案",
            Role.EMPLOYEE,
            expected_no_answer=True,
            is_acl_filtered=True,
        ),
    ]

    metrics = evaluate_retriever(retriever, cases, k=2)

    assert metrics.case_count == 3
    assert metrics.positive_case_count == 2
    assert metrics.no_answer_case_count == 1
    assert metrics.recall_at_k == pytest.approx(2 / 3)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.75)
    expected_first_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k == pytest.approx((expected_first_ndcg + 1) / 2)
    assert metrics.acl_filtered_recall_at_k == 1.0
    assert metrics.no_answer_accuracy == 1.0
    assert metrics.correct_version_recall_at_k == 1.0
    assert retriever.calls == [
        ("普通正样本", Role.HR),
        ("ACL 正样本", Role.HR),
        ("无答案", Role.EMPLOYEE),
    ]


def test_evaluate_retriever_deduplicates_ranked_chunk_ids_and_marks_missing_metrics() -> None:
    retriever = FrozenRetriever({"重复": [_candidate("a"), _candidate("a"), _candidate("b")]})
    metrics = evaluate_retriever(
        retriever,
        [RetrievalEvaluationCase("duplicates", "重复", Role.ADMIN, frozenset({"a", "b"}))],
        k=2,
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.no_answer_accuracy is None
    assert metrics.acl_filtered_recall_at_k is None
    assert metrics.correct_version_recall_at_k is None


def test_retrieval_evaluation_rejects_ambiguous_or_empty_configuration() -> None:
    with pytest.raises(ValueError, match="no_answer_case"):
        RetrievalEvaluationCase(
            "bad", "问题", Role.HR, frozenset({"chunk"}), expected_no_answer=True
        )
    with pytest.raises(ValueError, match="cases_required"):
        evaluate_retriever(FrozenRetriever({}), [], k=5)
    with pytest.raises(ValueError, match="must_be_positive"):
        evaluate_retriever(
            FrozenRetriever({"x": []}),
            [RetrievalEvaluationCase("x", "x", Role.HR)],
            k=0,
        )
