import json
from pathlib import Path

from rag_kb.evaluation import load_task_evaluation_dataset, run_dataset
from scripts.build_project_acceptance_report import build_report


def test_frozen_task_dataset_has_repeatable_offline_contract_report() -> None:
    dataset = load_task_evaluation_dataset(
        Path("data/benchmarks/enterprise_task_evaluation.yaml"),
        Path("schemas/enterprise-task-evaluation.schema.json"),
    )
    first = run_dataset(dataset)
    second = run_dataset(dataset)

    assert first == second
    assert first["status"] == "passed"
    assert first["live"] is False
    assert first["dataset"]["case_count"] == 20
    cases = {case["case_id"]: case for case in first["cases"]}
    assert cases["prompt-injection-document"]["security_gates"]["injection_ignored"]["status"] == "passed"
    assert cases["readonly-cannot-write"]["actions"][1]["outcome"] == "mode_denied"
    assert cases["idempotent-artifact"]["observations"]["artifact_attempts"] == 2
    assert cases["idempotent-artifact"]["observations"]["artifact_count"] == 1


def test_contract_report_is_consumed_without_claiming_live_model_metrics(tmp_path: Path) -> None:
    dataset = load_task_evaluation_dataset(
        Path("data/benchmarks/enterprise_task_evaluation.yaml"),
        Path("schemas/enterprise-task-evaluation.schema.json"),
    )
    task_report = tmp_path / "task.json"
    task_report.write_text(json.dumps(run_dataset(dataset), ensure_ascii=False), encoding="utf-8")

    report = build_report(
        pytest_xml=None,
        retrieval_report=None,
        task_report=task_report,
        dataset=Path("data/benchmarks/enterprise_task_evaluation.yaml"),
        config=None,
    )
    assert report["tool"]["status"] == "passed"
    assert report["model"]["status"] == "unverified"
    assert next(gate for gate in report["gates"] if gate["name"] == "task_contract")["status"] == "passed"
