from copy import deepcopy
import json
from pathlib import Path

from jsonschema import ValidationError
import pytest

from rag_kb.models import Role
from rag_kb.retrieval_acceptance import (
    AcceptanceCase,
    RetrievalAcceptanceMetrics,
    _configuration_objective,
    calibrate_retrieval_configuration,
    calibrate_score_threshold,
    evaluate_ranked_results,
    load_acceptance_dataset,
    make_report_header,
    quality_gates,
    serialize_evaluation,
    summarize_retrieval_costs,
    validate_retrieval_acceptance_report,
    write_report,
)
from rag_kb.retriever import RetrievalCandidate, RetrievalResult


def _candidate(chunk_id: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="policy",
        document_version=1,
        title="制度",
        source_file_name="policy.md",
        source_locator="第一章",
        score=score,
        text="冻结正文",
    )


def _passing_metrics(*, recall: float = 1.0, ndcg: float = 1.0) -> RetrievalAcceptanceMetrics:
    return RetrievalAcceptanceMetrics(
        case_count=20,
        positive_case_count=16,
        no_answer_case_count=4,
        exact_term_case_count=4,
        acl_positive_case_count=4,
        acl_denied_case_count=2,
        recall_at_k=recall,
        mrr=recall,
        ndcg_at_k=ndcg,
        correct_version_recall_at_k=recall,
        exact_term_recall_at_k=recall,
        acl_filtered_recall_at_k=recall,
        no_answer_accuracy=1.0,
        acl_leak_rate=0.0,
        acl_leak_count=0,
        acl_leak_case_count=0,
    )


def test_frozen_retrieval_acceptance_dataset_loads_with_required_strata() -> None:
    dataset = load_acceptance_dataset(
        Path("data/benchmarks/retrieval_acceptance.yaml"),
        Path("schemas/retrieval-acceptance-dataset.schema.json"),
    )

    assert len(dataset.cases) == 62
    assert dataset.category_counts == {
        "acl": 10,
        "exact_term": 12,
        "multi_chunk": 5,
        "no_answer": 10,
        "semantic": 25,
    }
    assert dataset.split_counts == {"calibration": 15, "test": 47}


def test_calibration_uses_only_real_ranked_scores_to_cut_off_no_answer_noise() -> None:
    positive = AcceptanceCase(
        case_id="positive",
        split="calibration",
        category="semantic",
        question="可回答问题",
        role=Role.HR,
        answerable=True,
        gold_chunk_ids=frozenset({"policy:v1:chunk-01"}),
        gold_versions=frozenset({("policy", 1)}),
        exact_terms=(),
        acl_filtered=False,
        forbidden_document_ids=frozenset(),
    )
    no_answer = AcceptanceCase(
        case_id="no-answer",
        split="calibration",
        category="no_answer",
        question="无答案问题",
        role=Role.HR,
        answerable=False,
        gold_chunk_ids=frozenset(),
        gold_versions=frozenset(),
        exact_terms=(),
        acl_filtered=False,
        forbidden_document_ids=frozenset(),
        no_answer_kind="out_of_scope",
    )
    results = {
        "positive": RetrievalResult([_candidate("policy:v1:chunk-01", 0.9)], 0),
        "no-answer": RetrievalResult([_candidate("noise:v1:chunk-01", 0.3)], 0),
    }

    calibration = calibrate_score_threshold(
        [positive, no_answer], results, top_k=5
    )
    evaluated = evaluate_ranked_results(
        [positive, no_answer],
        results,
        top_k=5,
        score_threshold=calibration.selected_threshold,
    )

    assert calibration.selected_threshold is not None
    assert 0.3 < calibration.selected_threshold <= 0.9
    assert evaluated.metrics.recall_at_k == 1.0
    assert evaluated.metrics.no_answer_accuracy == 1.0


def test_joint_calibration_selects_smallest_equal_quality_k_without_test_labels() -> None:
    multi = AcceptanceCase(
        case_id="multi",
        split="calibration",
        category="multi_chunk",
        question="需要两块证据",
        role=Role.HR,
        answerable=True,
        gold_chunk_ids=frozenset({"policy:v1:chunk-01", "policy:v1:chunk-02"}),
        gold_versions=frozenset({("policy", 1)}),
        exact_terms=(),
        acl_filtered=False,
        forbidden_document_ids=frozenset(),
    )
    no_answer = AcceptanceCase(
        case_id="none",
        split="calibration",
        category="no_answer",
        question="没有依据",
        role=Role.HR,
        answerable=False,
        gold_chunk_ids=frozenset(),
        gold_versions=frozenset(),
        exact_terms=(),
        acl_filtered=False,
        forbidden_document_ids=frozenset(),
        no_answer_kind="out_of_scope",
    )
    # Mapping 中可以包含 test 原始结果，但选参函数只索引传入的 calibration case。
    results = {
        "multi": RetrievalResult(
            [_candidate("policy:v1:chunk-01", 0.95), _candidate("policy:v1:chunk-02", 0.90)],
            0,
        ),
        "none": RetrievalResult([_candidate("noise:v1:chunk-01", 0.20)], 0),
        "unseen-test": RetrievalResult([_candidate("policy:v1:chunk-99", 1.0)], 0),
    }

    calibration = calibrate_retrieval_configuration(
        [multi, no_answer],
        results,
        candidate_k_values=(2, 5),
        context_k_values=(1, 2),
    )

    assert calibration.selected_candidate_k == 2
    assert calibration.selected_context_k == 2
    assert calibration.selected_threshold is not None
    assert calibration.selected_metrics.recall_at_k == 1.0
    assert calibration.selected_metrics.no_answer_accuracy == 1.0


def test_joint_calibration_rejects_test_split_and_cost_report_has_no_text() -> None:
    test_case = AcceptanceCase(
        case_id="held-out",
        split="test",
        category="semantic",
        question="留出标签",
        role=Role.HR,
        answerable=True,
        gold_chunk_ids=frozenset({"policy:v1:chunk-01"}),
        gold_versions=frozenset({("policy", 1)}),
        exact_terms=(),
        acl_filtered=False,
        forbidden_document_ids=frozenset(),
    )
    results = {
        "held-out": RetrievalResult(
            [_candidate("policy:v1:chunk-01", 0.9), _candidate("noise:v1:chunk-01", 0.2)],
            0,
            candidate_count=2,
        )
    }

    with pytest.raises(ValueError, match="calibration_split"):
        calibrate_retrieval_configuration(
            [test_case], results, candidate_k_values=(2,), context_k_values=(1,)
        )

    evaluated = evaluate_ranked_results(
        [test_case],
        results,
        candidate_k=2,
        context_k=1,
        score_threshold=0.5,
    )
    costs = summarize_retrieval_costs(
        results,
        evaluated,
        raw_candidate_k=2,
        selected_candidate_k=2,
        selected_context_k=1,
        query_latency_ms={"held-out": 12.5},
        selected_query_latency_ms={"held-out": 9.5},
    )
    assert costs["raw_candidate_pool"]["total"] == 2
    assert costs["selected_context"]["total"] == 1
    assert costs["query_latency_ms"]["p50"] == 12.5
    assert costs["query_latency_ms"]["configured_candidate_k"] == 2
    assert costs["selected_configuration_query_latency_ms"]["p50"] == 9.5
    assert costs["selected_context"]["character_count"]["total"] == len("冻结正文")
    assert "冻结正文" not in str(costs)


def _valid_v2_retrieval_report() -> dict:
    metrics = {
        "case_count": 1, "positive_case_count": 1, "no_answer_case_count": 0,
        "exact_term_case_count": 1, "acl_positive_case_count": 1,
        "acl_denied_case_count": 0, "recall_at_k": 1.0, "mrr": 1.0,
        "ndcg_at_k": 1.0, "correct_version_recall_at_k": 1.0,
        "exact_term_recall_at_k": 1.0, "acl_filtered_recall_at_k": 1.0,
        "no_answer_accuracy": None, "acl_leak_rate": None,
        "acl_leak_count": 0, "acl_leak_case_count": 0,
    }
    gates = {
        "recall_at_2": True, "mrr": True, "ndcg_at_2": True,
        "correct_version_recall": True,
        "exact_term_recall": True, "acl_filtered_recall": True,
        "no_answer_accuracy": False, "acl_leak_rate": False, "passed": False,
    }
    case = {
        "case_id": "case-01", "split": "test", "category": "semantic",
        "answerable": True, "returned_chunk_ids": ["policy:v1:chunk-01"],
        "returned_scores": [0.9], "first_gold_rank": 1, "gold_hit_count": 1,
        "no_answer_correct": None, "acl_leak_document_ids": [],
        "candidate_count": 1, "threshold_suppressed_count": 0,
        "context_truncated_count": 0,
    }
    count_summary = {
        "configured_k": 2, "sample_count": 1, "total": 1,
        "mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1,
    }
    latency = {
        "configured_candidate_k": 2, "sample_count": 1, "total": 1.0,
        "mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0,
    }
    return {
        "schema_version": 2, "created_at": "2026-08-14T00:00:00+00:00",
        "status": "test_quality_gates_failed",
        "backend": {
            "mode": "real_bge_m3_qdrant", "model": "BAAI/bge-m3",
            "qdrant_url": "http://localhost:6333",
            "isolated_collection": "retrieval_acceptance_test",
            "embedding_batch_size": "per_document_chunk_batch",
            "acl_filter": "status=active AND allowed_roles contains role",
            "retrieval_order": "qdrant_server_acl -> candidate_k -> score_threshold -> context_k",
            "query_execution": "sequential_production_retriever_search",
            "raw_candidate_k": 5,
        },
        "dataset": {
            "path": "data/benchmarks/retrieval_acceptance.yaml",
            "name": "frozen-retrieval", "total_cases": 1,
            "split_counts": {"test": 1}, "category_counts": {"semantic": 1},
            "required_top_k": 5, "frozen_baseline_context_k": 5,
        },
        "thresholds": {
            "recall_at_k": 0.9, "mrr": 0.8, "ndcg_at_k": 0.8,
            "correct_version_recall_at_k": 0.95,
            "exact_term_recall_at_k": 0.9,
            "acl_filtered_recall_at_k": 0.9,
            "no_answer_accuracy": 0.95, "acl_leak_rate": 0.0,
        },
        "parameter_search": {
            "selection_split": "calibration", "test_labels_used_for_selection": False,
            "candidate_k_values": [5, 10], "context_k_values": [2, 3, 5],
            "raw_candidate_pool_executions_per_case": 1,
            "selected_configuration_replay_executions_per_case": 1,
            "production_configuration_changed": False,
        },
        "collection_cleanup": {"attempted": True, "succeeded": True},
        "ingestion": {"documents": [{
            "document_id": "policy", "version": 1,
            "chunk_count": 1, "previous_version": None,
        }]},
        "gold_label_validation": {
            "passed": True, "active_chunk_count": 1, "ingested_chunk_count": 1,
            "error_count": 0, "errors": [],
        },
        "threshold_calibration": {
            "split": "calibration", "selected_candidate_k": 2,
            "selected_context_k": 2, "selected_score_threshold": 0.5,
            "k_pair_count": 1, "score_threshold_candidate_count": 1,
            "evaluated_configuration_count": 1,
            "quality_gates": deepcopy(gates),
            "all_quality_gates_passed": False, "metrics": deepcopy(metrics),
        },
        "calibration": {
            "metrics": deepcopy(metrics), "cases": [deepcopy(case)],
            "quality_gates": deepcopy(gates),
        },
        "test": {
            "metrics": deepcopy(metrics), "cases": [deepcopy(case)],
            "quality_gates": deepcopy(gates),
        },
        "all_cases": {"metrics": deepcopy(metrics), "cases": [deepcopy(case)]},
        "selected_configuration_replay": {
            "passed": True, "mismatched_case_ids": [],
            "labels_used_for_selection": False,
        },
        "costs": {
            "scope": "raw_calibration_pool_plus_optional_selected_configuration_replay",
            "raw_candidate_pool": {**count_summary, "configured_k": 5},
            "selected_candidate_pool": deepcopy(count_summary),
            "selected_context": {
                **count_summary,
                "character_count": {
                    "sample_count": 1, "total": 10, "mean": 10.0,
                    "p50": 10.0, "p95": 10.0, "max": 10.0,
                },
                "threshold_suppressed_total": 0, "context_truncated_total": 0,
            },
            "query_latency_ms": {
                **latency,
                "scope": "raw_candidate_k_embedding_plus_acl_filtered_qdrant_search",
                "configured_candidate_k": 5,
            },
            "selected_configuration_query_latency_ms": {
                **latency,
                "scope": "selected_candidate_k_embedding_plus_acl_filtered_qdrant_search",
            },
        },
    }


def test_retrieval_v2_report_schema_validates_before_write(tmp_path: Path) -> None:
    report = _valid_v2_retrieval_report()
    schema_path = Path("schemas/retrieval-acceptance-report.schema.json")
    validate_retrieval_acceptance_report(report, schema_path)

    output = tmp_path / "report.json"
    write_report(output, report, schema_path=schema_path)
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 2


def test_retrieval_running_report_uses_real_header_shape() -> None:
    dataset = load_acceptance_dataset(
        Path("data/benchmarks/retrieval_acceptance.yaml"),
        Path("schemas/retrieval-acceptance-dataset.schema.json"),
    )
    report = make_report_header(
        dataset,
        collection="retrieval_acceptance_test",
        qdrant_url="http://localhost:6333",
        model_name="BAAI/bge-m3",
        batch_size=0,
    )
    report["backend"].update({
        "embedding_batch_size": "per_document_chunk_batch",
        "query_execution": "sequential_production_retriever_search",
        "raw_candidate_k": 10,
    })
    report["parameter_search"] = {
        "selection_split": "calibration",
        "test_labels_used_for_selection": False,
        "candidate_k_values": [5, 10],
        "context_k_values": [2, 3, 5],
        "raw_candidate_pool_executions_per_case": 1,
        "selected_configuration_replay_executions_per_case": 1,
        "production_configuration_changed": False,
    }
    report["collection_cleanup"] = {"attempted": False, "succeeded": False}

    validate_retrieval_acceptance_report(
        report, Path("schemas/retrieval-acceptance-report.schema.json")
    )


def test_retrieval_v2_report_rejects_missing_gate_and_sensitive_fields(
    tmp_path: Path,
) -> None:
    schema_path = Path("schemas/retrieval-acceptance-report.schema.json")
    missing_gate = _valid_v2_retrieval_report()
    del missing_gate["test"]["quality_gates"]
    with pytest.raises(ValidationError):
        validate_retrieval_acceptance_report(missing_gate, schema_path)

    contains_query = _valid_v2_retrieval_report()
    contains_query["test"]["cases"][0]["query"] = "不应持久化的问题"
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValidationError):
        write_report(output, contains_query, schema_path=schema_path)
    assert not output.exists()


def test_configuration_objective_prefers_smaller_k_after_both_configs_pass() -> None:
    smaller = _passing_metrics(recall=0.95, ndcg=0.95)
    larger = _passing_metrics()
    smaller_gates = quality_gates(smaller, k=2)
    larger_gates = quality_gates(larger, k=5)

    assert smaller_gates["passed"] is True
    assert larger_gates["passed"] is True
    assert _configuration_objective(
        smaller,
        smaller_gates,
        candidate_k=5,
        context_k=2,
        threshold=0.5,
    ) > _configuration_objective(
        larger,
        larger_gates,
        candidate_k=10,
        context_k=5,
        threshold=0.5,
    )


def test_ndcg_is_a_hard_quality_gate() -> None:
    gates = quality_gates(_passing_metrics(ndcg=0.79), k=2)

    assert gates["ndcg_at_2"] is False
    assert gates["passed"] is False


def test_acceptance_outcome_requires_calibration_and_test_gates() -> None:
    from scripts.benchmark_retrieval_acceptance import _acceptance_outcome

    assert _acceptance_outcome(calibration_passed=True, test_passed=True) == ("passed", 0)
    assert _acceptance_outcome(calibration_passed=False, test_passed=True) == (
        "calibration_quality_gates_failed",
        1,
    )
    assert _acceptance_outcome(calibration_passed=True, test_passed=False) == (
        "test_quality_gates_failed",
        1,
    )
    assert _acceptance_outcome(
        calibration_passed=True, test_passed=True, replay_passed=False
    ) == ("selected_configuration_replay_mismatch", 1)


def test_collection_cleanup_failure_is_recorded_without_backend_details() -> None:
    from scripts.benchmark_retrieval_acceptance import _record_collection_cleanup

    class Client:
        def delete_collection(self, _collection):
            raise RuntimeError("SECRET_BACKEND_DETAIL")

    report = {"collection_cleanup": {"attempted": False, "succeeded": False}}
    succeeded = _record_collection_cleanup(
        type("Store", (), {"client": Client()})(), "isolated", report,
    )
    assert not succeeded
    assert report["collection_cleanup"] == {
        "attempted": True, "succeeded": False, "error_code": "RuntimeError",
    }
    assert "SECRET" not in json.dumps(report)


def test_evaluation_serialization_uses_json_arrays_for_tuple_fields() -> None:
    case = AcceptanceCase(
        case_id="json-array", split="test", category="semantic",
        question="制度", role=Role.HR, answerable=True,
        gold_chunk_ids=frozenset({"policy:v1:chunk-01"}),
        gold_versions=frozenset({("policy", 1)}), exact_terms=(),
        acl_filtered=False, forbidden_document_ids=frozenset(),
    )
    result = evaluate_ranked_results(
        (case,),
        {"json-array": RetrievalResult(
            candidates=[_candidate("policy:v1:chunk-01", 0.9)], suppressed_count=0,
        )},
        candidate_k=1, context_k=1, score_threshold=None,
    )
    payload = serialize_evaluation(result)["cases"][0]
    assert isinstance(payload["returned_chunk_ids"], list)
    assert isinstance(payload["returned_scores"], list)
    assert isinstance(payload["acl_leak_document_ids"], list)


def test_evaluation_does_not_hide_explicit_zero_k_or_non_finite_threshold() -> None:
    import pytest

    case = AcceptanceCase(
        case_id="invalid-config",
        split="calibration",
        category="semantic",
        question="配置边界",
        role=Role.HR,
        answerable=True,
        gold_chunk_ids=frozenset({"policy:v1:chunk-01"}),
        gold_versions=frozenset({("policy", 1)}),
        exact_terms=(),
        acl_filtered=False,
        forbidden_document_ids=frozenset(),
    )
    results = {
        case.case_id: RetrievalResult([_candidate("policy:v1:chunk-01", 0.9)], 0)
    }

    with pytest.raises(ValueError, match="k_must_be_positive"):
        evaluate_ranked_results(
            [case], results, candidate_k=0, context_k=1, score_threshold=None
        )
    with pytest.raises(ValueError, match="threshold_must_be_finite"):
        evaluate_ranked_results(
            [case], results, candidate_k=1, context_k=1, score_threshold=float("nan")
        )
