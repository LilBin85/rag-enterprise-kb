import json
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.build_project_acceptance_report import SCHEMA, build_report, validate_report


def test_report_hash_and_schema_with_observed_inputs(tmp_path: Path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text('<testsuite tests="3" failures="1" errors="0" skipped="1" time="1.5"/>', encoding="utf-8")
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps({"status": "completed", "test": {"metrics": {"recall_at_5": 1.0, "mrr": 1.0, "no_answer_accuracy": 1.0, "acl_leak_rate": 0.0}}}), encoding="utf-8")
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "status": "passed",
                "execution_mode": "deterministic_contract_simulation",
                "summary": {"passed": 2, "failed": 0},
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("version: test\n", encoding="utf-8")
    report = build_report(pytest_xml=junit, retrieval_report=retrieval, task_report=task, dataset=dataset, config=None)
    validate_report(report)
    assert report["report_hash"] != "0" * 64
    assert report["outcomes"]["values"]["pytest"]["passed"] == 1
    assert report["config"]["status"] == "unverified"
    assert report["quality"]["status"] == "passed"
    assert report["security"]["status"] == "passed"
    assert next(item for item in report["gates"] if item["name"] == "task_contract")["status"] == "passed"


def test_missing_inputs_are_unverified_and_never_fabricated(tmp_path: Path) -> None:
    report = build_report(pytest_xml=None, retrieval_report=None, task_report=None, dataset=None, config=None)
    validate_report(report)
    assert report["verdict"]["status"] == "unverified"
    assert report["model"]["status"] == "unverified"
    assert report["quality"]["values"] == {}
    assert all(item["status"] == "missing" for item in report["evidence_artifacts"])


def test_existing_config_uses_schema_status_vocabulary(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    report = build_report(
        pytest_xml=None,
        retrieval_report=None,
        task_report=None,
        dataset=None,
        config=config,
    )

    validate_report(report)
    assert report["config"]["status"] == "passed"


def test_quality_gate_failure_controls_final_verdict(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(
        json.dumps(
            {
                "status": "completed",
                "test": {
                    "metrics": {
                        "recall_at_5": 0.9,
                        "mrr": 0.9,
                        "no_answer_accuracy": 0.75,
                        "acl_leak_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = build_report(
        pytest_xml=None,
        retrieval_report=retrieval,
        task_report=None,
        dataset=None,
        config=None,
    )

    assert report["quality"]["status"] == "failed"
    assert report["verdict"]["status"] == "failed"
