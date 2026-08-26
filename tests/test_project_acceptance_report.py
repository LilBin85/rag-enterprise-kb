import json
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts import build_project_acceptance_report as report_builder
from scripts.build_project_acceptance_report import SCHEMA, build_report, pytest_values, validate_report


def test_report_hash_and_schema_with_observed_inputs(tmp_path: Path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        """<testsuite tests="3" failures="1" errors="0" skipped="1" time="1.5">
<testcase classname="tests.test_unit" name="passed"/>
<testcase classname="tests.test_unit" name="failed"><failure/></testcase>
<testcase classname="tests.test_integration" name="needs_qdrant"><skipped message="qdrant_unavailable"/></testcase>
</testsuite>""",
        encoding="utf-8",
    )
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
    assert report["outcomes"]["values"]["pytest"]["skipped_details"] == [
        {"test": "tests.test_integration::needs_qdrant", "reason": "Qdrant 不可达"}
    ]
    assert len(report["code"]["values"]["worktree_sha256"]) == 64
    assert report["config"]["status"] == "unverified"
    assert report["quality"]["status"] == "passed"
    assert report["security"]["status"] == "passed"
    assert report["tool"]["status"] == "unverified"
    assert next(item for item in report["gates"] if item["name"] == "legacy_task_contract")["status"] == "passed"


def test_skip_details_do_not_publish_unapproved_message(tmp_path: Path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuite tests="1" skipped="1"><testcase name="skip"><skipped message="postgres://user:secret@host"/></testcase></testsuite>',
        encoding="utf-8",
    )

    _status, values = pytest_values(junit)

    assert values["skipped_details"] == [
        {"test": "skip", "reason": "跳过详情未写入报告"}
    ]


def test_worktree_fingerprint_includes_untracked_file_content(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("first", encoding="utf-8")

    def check_output(command, **_kwargs):
        return b"tracked" if command[1] == "diff" else b"marker.txt\0"

    monkeypatch.setattr(report_builder, "ROOT", tmp_path)
    monkeypatch.setattr(report_builder.subprocess, "check_output", check_output)
    first = report_builder.git_worktree_fingerprint()
    marker.write_text("second", encoding="utf-8")
    second = report_builder.git_worktree_fingerprint()

    assert first is not None and second is not None and first != second


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


def test_m1_e2e_requires_the_exact_live_research_trace(tmp_path: Path) -> None:
    m1_e2e = tmp_path / "m1-e2e.json"
    m1_e2e.write_text(
        json.dumps(
            {
                "status": "passed",
                "execution_mode": "live_postgres_msgpack",
                "document_version_endpoint": "/v1/documents/{document_id}/versions",
                "document_versions": 2,
                "tool_trace": [
                    "list_document_versions",
                    "compare_document_versions",
                    "search_knowledge",
                    "fetch_evidence",
                ],
                "citation_count": 1,
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        pytest_xml=None,
        retrieval_report=None,
        task_report=None,
        dataset=None,
        config=None,
        m1_e2e_report=m1_e2e,
    )

    assert report["model"]["status"] == "passed"
    assert report["tool"]["values"]["tool_trace"] == [
        "list_document_versions",
        "compare_document_versions",
        "search_knowledge",
        "fetch_evidence",
    ]
    assert next(item for item in report["gates"] if item["name"] == "m1_research_e2e")["status"] == "passed"
