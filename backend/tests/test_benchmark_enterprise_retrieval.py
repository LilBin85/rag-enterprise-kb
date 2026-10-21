from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_kb.application.ingestion_worker import WorkerRunResult
from rag_kb.models import Role
from rag_kb.retrieval_acceptance import AcceptanceCase, load_acceptance_dataset
from scripts.benchmark_enterprise_retrieval import (
    BACKEND_ROOT,
    REPO_ROOT,
    CaseObservation,
    _drain_ingestion_jobs,
    _evaluate_document_version_cases,
    _select_pipeline_configuration,
    main,
)


class _Worker:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def run_once(self):
        return next(self.outcomes)


class _Jobs:
    def __init__(self, states):
        self.states = {key: iter(value) for key, value in states.items()}
        self.current = {}

    def get_ingestion_job(self, job_id, _auth):
        try:
            self.current[job_id] = next(self.states[job_id])
        except StopIteration:
            pass
        status = self.current.get(job_id)
        return None if status is None else SimpleNamespace(status=status)


def test_ingestion_drain_tolerates_idle_until_all_target_jobs_succeed() -> None:
    clock = iter((0.0, 0.0, 0.1, 0.2, 0.3))
    outcomes, diagnostics = _drain_ingestion_jobs(
        _Worker(
            (
                WorkerRunResult("idle"),
                WorkerRunResult("retry_scheduled", "job-2", "safe_code"),
                WorkerRunResult("succeeded", "job-2"),
            )
        ),
        _Jobs(
            {
                "job-1": ("pending", "succeeded"),
                "job-2": ("pending", "pending", "pending", "succeeded"),
            }
        ),
        None,
        ("job-1", "job-2"),
        diagnostics=lambda _ids: {"unexpected": True},
        timeout_seconds=1,
        poll_seconds=0,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
    )

    assert [item["status"] for item in outcomes] == [
        "idle",
        "retry_scheduled",
        "succeeded",
    ]
    assert diagnostics == {
        "unexpected": True,
        "error_code": None,
        "unrelated_job_ids": [],
        "duplicate_job_ids": [],
    }


def test_ingestion_drain_reports_terminal_failure() -> None:
    expected_diagnostics = {"targets": [{"job_id": "job-1", "status": "failed"}]}
    outcomes, diagnostics = _drain_ingestion_jobs(
        _Worker((WorkerRunResult("failed", "job-1", "safe_code"),)),
        _Jobs({"job-1": ("pending",)}),
        None,
        ("job-1",),
        diagnostics=lambda _ids: expected_diagnostics,
        timeout_seconds=1,
        monotonic=lambda: 0.0,
    )

    assert outcomes == [{"status": "failed", "job_id": "job-1", "error_code": "safe_code"}]
    assert diagnostics == {
        **expected_diagnostics,
        "error_code": "ingestion_drain_worker_failure",
        "unrelated_job_ids": [],
        "duplicate_job_ids": [],
    }


def test_ingestion_drain_reports_timeout_after_repeated_idle() -> None:
    clock = iter((0.0, 0.0, 1.0))
    expected_diagnostics = {"missing_job_ids": [], "queue_snapshot": []}
    outcomes, diagnostics = _drain_ingestion_jobs(
        _Worker((WorkerRunResult("idle"),)),
        _Jobs({"job-1": ("pending",)}),
        None,
        ("job-1",),
        diagnostics=lambda _ids: expected_diagnostics,
        timeout_seconds=1,
        poll_seconds=0,
        monotonic=lambda: next(clock),
        sleep=lambda _seconds: None,
    )

    assert outcomes == [{"status": "idle", "job_id": None, "error_code": None}]
    assert diagnostics == {
        **expected_diagnostics,
        "error_code": "ingestion_drain_timeout",
        "unrelated_job_ids": [],
        "duplicate_job_ids": [],
    }


def test_ingestion_drain_records_unrelated_and_duplicate_worker_outcomes() -> None:
    outcomes, diagnostics = _drain_ingestion_jobs(
        _Worker(
            (
                WorkerRunResult("succeeded", "other-job"),
                WorkerRunResult("succeeded", "job-1"),
                WorkerRunResult("succeeded", "job-1"),
            )
        ),
        _Jobs({"job-1": ("pending", "pending", "pending", "succeeded")}),
        None,
        ("job-1",),
        diagnostics=lambda _ids: {"targets": []},
        timeout_seconds=1,
        poll_seconds=0,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert len(outcomes) == 3
    assert diagnostics["error_code"] is None
    assert diagnostics["unrelated_job_ids"] == ["other-job"]
    assert diagnostics["duplicate_job_ids"] == ["job-1"]


def _case(
    *,
    case_id: str,
    answerable: bool,
    forbidden=(),
    out_of_scope: bool = False,
) -> AcceptanceCase:
    return AcceptanceCase(
        case_id=case_id,
        split="calibration",
        category="acl" if forbidden else ("no_answer" if out_of_scope else "semantic"),
        question="冻结问题",
        role=Role.HR,
        answerable=answerable,
        gold_chunk_ids=frozenset(),
        gold_versions=frozenset({("policy", 2)}) if answerable else frozenset(),
        exact_terms=(),
        acl_filtered=bool(forbidden),
        forbidden_document_ids=frozenset(forbidden),
        no_answer_kind=(
            "acl_denied"
            if forbidden
            else ("out_of_scope" if not answerable else None)
        ),
    )


def test_enterprise_metrics_use_document_version_not_chunk_ids() -> None:
    positive = _case(case_id="positive", answerable=True)
    denied = _case(case_id="denied", answerable=False, forbidden=("restricted",))
    evaluation = _evaluate_document_version_cases(
        (positive, denied),
        {
            "positive": CaseObservation("positive", True, (("policy", 2, 0.9),), 1.5),
            "denied": CaseObservation("denied", False, (("restricted", 1, 0.8),), 2.0),
        },
    )

    assert evaluation["metrics"]["metric_contract"] == "document_version_level_not_legacy_chunk_level"
    assert evaluation["metrics"]["recall_at_5"] == 1.0
    assert evaluation["metrics"]["mrr"] == 1.0
    # acl_denied 样本不计入拒答准确率（由 acl_leak_rate 考核）；无 out_of_scope 样本时为 None
    assert evaluation["metrics"]["no_answer_accuracy"] is None
    assert evaluation["metrics"]["no_answer_case_count"] == 0
    assert evaluation["metrics"]["acl_leak_rate"] == 1.0
    assert evaluation["cases"][0]["returned_document_versions"] == [
        {"document_id": "policy", "version": 2, "dense_score": 0.9}
    ]
    denied_row = next(row for row in evaluation["cases"] if row["case_id"] == "denied")
    assert denied_row["no_answer_correct"] is None


def test_out_of_scope_cases_carry_no_answer_accuracy() -> None:
    """拒答准确率只由 out_of_scope 负样本承载；命中确定性拦截的应计为正确。"""

    positive = _case(case_id="positive", answerable=True)
    blocked = _case(case_id="blocked", answerable=False, out_of_scope=True)
    leaked = _case(case_id="leaked", answerable=False, out_of_scope=True)
    denied = _case(case_id="denied", answerable=False, forbidden=("restricted",))
    evaluation = _evaluate_document_version_cases(
        (positive, blocked, leaked, denied),
        {
            "positive": CaseObservation("positive", True, (("policy", 2, 0.9),), 1.0),
            "blocked": CaseObservation("blocked", False, (), 1.0),
            "leaked": CaseObservation("leaked", False, (("policy", 2, 0.5),), 1.0),
            "denied": CaseObservation("denied", False, (), 1.0),
        },
    )

    assert evaluation["metrics"]["no_answer_case_count"] == 2
    assert evaluation["metrics"]["no_answer_accuracy"] == 0.5
    # acl_denied 样本检索层无泄漏，leak 门仍为 0
    assert evaluation["metrics"]["acl_leak_rate"] == 0.0
    rows = {row["case_id"]: row for row in evaluation["cases"]}
    assert rows["blocked"]["no_answer_correct"] is True
    assert rows["leaked"]["no_answer_correct"] is False
    assert rows["denied"]["no_answer_correct"] is None


def test_pipeline_selection_prioritizes_acl_then_dense_only_tie_break() -> None:
    positive = _case(case_id="positive", answerable=True)
    denied = _case(case_id="denied", answerable=False, forbidden=("restricted",))
    clean = _evaluate_document_version_cases(
        (positive, denied),
        {
            "positive": CaseObservation("positive", True, (("policy", 2, 0.9),), 1.0),
            "denied": CaseObservation("denied", False, (), 1.0),
        },
    )
    leaky = _evaluate_document_version_cases(
        (positive, denied),
        {
            "positive": CaseObservation("positive", True, (("policy", 2, 0.9),), 1.0),
            "denied": CaseObservation("denied", False, (("restricted", 1, 0.8),), 1.0),
        },
    )
    selected = _select_pipeline_configuration(
        (
            {
                "pipeline": "hybrid",
                "hybrid_enabled": True,
                "dense_score_threshold": 0.2,
                "evaluation": clean,
            },
            {
                "pipeline": "dense_only",
                "hybrid_enabled": False,
                "dense_score_threshold": None,
                "evaluation": clean,
            },
            {
                "pipeline": "dense_only",
                "hybrid_enabled": False,
                "dense_score_threshold": 0.8,
                "evaluation": leaky,
            },
        )
    )

    assert selected["pipeline"] == "dense_only"
    assert selected["hybrid_enabled"] is False
    assert selected["dense_score_threshold"] is None


def test_pipeline_selector_does_not_trade_recall_for_retrieval_refusal() -> None:
    """拒答正确率不参与管线选择：检索层只优化无泄漏与召回/MRR，
    out_of_scope 的最终拒答由 verdict 层保证。"""

    positive = _case(case_id="positive", answerable=True)
    blocked = _case(case_id="blocked", answerable=False, out_of_scope=True)
    high_threshold = _evaluate_document_version_cases(
        (positive, blocked),
        {
            "positive": CaseObservation("positive", True, (), 1.0),  # 正样本被阈值误杀
            "blocked": CaseObservation("blocked", False, (), 1.0),
        },
    )
    low_threshold = _evaluate_document_version_cases(
        (positive, blocked),
        {
            "positive": CaseObservation("positive", True, (("policy", 2, 0.9),), 1.0),
            "blocked": CaseObservation("blocked", False, (("policy", 2, 0.5),), 1.0),
        },
    )
    assert high_threshold["metrics"]["no_answer_accuracy"] == 1.0
    assert low_threshold["metrics"]["no_answer_accuracy"] == 0.0

    selected = _select_pipeline_configuration(
        (
            {
                "pipeline": "dense_only_high",
                "hybrid_enabled": False,
                "dense_score_threshold": 0.6,
                "evaluation": high_threshold,
            },
            {
                "pipeline": "dense_only_low",
                "hybrid_enabled": False,
                "dense_score_threshold": None,
                "evaluation": low_threshold,
            },
        )
    )
    assert selected["pipeline"] == "dense_only_low"


def test_document_version_evaluator_accepts_an_explicit_evaluated_subset() -> None:
    calibration = _case(case_id="calibration", answerable=True)
    development = _case(case_id="development", answerable=True)
    evaluation = _evaluate_document_version_cases(
        (calibration,),
        {
            "calibration": CaseObservation(
                "calibration", True, (("policy", 2, 0.9),), 1.0
            )
        },
    )

    assert evaluation["metrics"]["case_count"] == 1
    assert development.case_id not in {item["case_id"] for item in evaluation["cases"]}


def test_cli_defaults_to_v2_dataset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return {"status": "completed"}, 0

    monkeypatch.setenv("RAG_KB_DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(
        "scripts.benchmark_enterprise_retrieval.run_benchmark", fake_run_benchmark
    )

    assert main([]) == 0
    assert captured["dataset_path"] == REPO_ROOT / "data" / "benchmarks" / "retrieval_acceptance_v2.yaml"
    assert captured["schema_path"] == BACKEND_ROOT / "schemas" / "retrieval-acceptance-v2.schema.json"


def test_v2_acceptance_dataset_loads() -> None:
    dataset = load_acceptance_dataset(
        REPO_ROOT / "data" / "benchmarks" / "retrieval_acceptance_v2.yaml",
        BACKEND_ROOT / "schemas" / "retrieval-acceptance-v2.schema.json",
    )

    assert dataset.version == 2
    assert {"development", "calibration", "test"} <= set(dataset.split_counts)
