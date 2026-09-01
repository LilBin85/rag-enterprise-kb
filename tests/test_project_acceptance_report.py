import hashlib
import json
from pathlib import Path

from scripts import build_project_acceptance_report as report_builder
from scripts.build_project_acceptance_report import build_report, pytest_values, validate_report


def _source_binding(monkeypatch) -> dict[str, str]:
    binding = {"git_revision": "revision", "worktree_sha256": "a" * 64}
    monkeypatch.setattr(report_builder, "git_revision", lambda: binding["git_revision"])
    monkeypatch.setattr(report_builder, "git_worktree_fingerprint", lambda: binding["worktree_sha256"])
    return binding


def _strict_msgpack() -> dict[str, object]:
    return {
        "runtime_flag": True,
        "serializer": "JsonPlusSerializer",
        "pickle_fallback": False,
        "allowed_msgpack_modules": sorted(
            report_builder._STRICT_MSGPACK_ALLOWLIST
        ),
    }


def _m1_e2e(binding: dict[str, str]) -> dict[str, object]:
    return {
        "status": "passed",
        "source_binding": binding,
        "execution_mode": "live_postgres_msgpack",
        "strict_msgpack": _strict_msgpack(),
        "checkpoint_encodings": ["msgpack"],
        "document_version_endpoint": "/v1/documents/{document_id}/versions",
        "document_versions": 2,
        "tool_trace": [
            "list_document_versions",
            "compare_document_versions",
            "search_knowledge",
            "fetch_evidence",
        ],
        "tool_calls": 4,
        "citation_count": 1,
        "result_readable": True,
    }


def _m2_e2e(binding: dict[str, str]) -> dict[str, object]:
    ledger = [
        {
            "task_id": f"task-{index}",
            "status": "succeeded",
            "stop_reason": "completed",
            "error_code": None,
            "correlation_id": f"correlation-{index}",
            "model_calls": 2,
            "tool_calls": 1,
        }
        for index in range(1, 31)
    ]
    return {
        "status": "passed",
        "source_binding": binding,
        "execution_mode": "live_postgres_msgpack",
        "capability": "task-agent",
        "protocol": "m2-fenced-v1",
        "strict_msgpack": _strict_msgpack(),
        "checkpoint_encodings": ["msgpack", "null"],
        "tool_exposure": {
            "read": [
                "list_document_versions",
                "compare_document_versions",
                "search_knowledge",
                "fetch_evidence",
            ],
            "write": ["create_report_artifact", "create_work_item_draft"],
        },
        "tool_trace": ["search_knowledge", "fetch_evidence", "create_report_artifact"],
        "binding": {
            "task_id": "task-1",
            "thread_id": "task-1",
            "checkpoint_id": "checkpoint-1",
            "interrupt_id": "interrupt-1",
            "approval_id": "approval-1",
            "tool_call_id": "tool-call-1",
            "artifact_id": "artifact-1",
            "artifact_hash": "sha256:" + "a" * 64,
            "result_artifact_id": "artifact-1",
        },
        "artifact": {
            "artifact_id": "artifact-1",
            "content_hash": "sha256:" + "a" * 64,
            "status": "ready",
            "content_readable": True,
            "hash_verified": True,
        },
        "terminal_binding": {
            "approval": {
                "approval_id": "approval-1",
                "checkpoint_action_id": "call-1",
                "status": "executed",
            },
            "tool_call": {
                "approval_id": "approval-1",
                "status": "succeeded",
                "tool_call_id": "tool-call-1",
            },
            "artifact": {
                "artifact_id": "artifact-1",
                "content_hash": "sha256:" + "a" * 64,
                "status": "ready",
                "tool_call_id": "tool-call-1",
            },
            "task_result": {
                "artifact_id": "artifact-1",
                "result_kind": "controlled_action",
                "status": "succeeded",
                "stop_reason": "completed",
                "task_id": "task-1",
            },
        },
        "hitl": {"adapter_calls_before_approval": 0, "adapter_calls_after_approval": 1},
        "side_effects": {"unapproved": 0, "duplicate": 0, "unknown_replay": 0},
        "access_checks": {"cross_tenant_status": 404, "nonowner_status": 404},
        "privacy_scan": {
            "storage_key": 0,
            "prompt": 0,
            "content": 0,
            "credentials": 0,
            "resume_payload": 0,
        },
        "cleanup": {"completed": True},
        "coverage": {
            "task_count": 30,
            "scenario_count": 20,
            "crash_recovery": {
                "before_execution": True,
                "after_execution_before_checkpoint": True,
                "after_checkpoint_before_sync": True,
            },
        },
        "task_ledger": ledger,
    }


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
    assert "config" not in report
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

    status, values = pytest_values(junit)

    assert status == "failed"
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


def test_m1_e2e_requires_the_exact_live_research_trace(tmp_path: Path, monkeypatch) -> None:
    binding = _source_binding(monkeypatch)
    m1_e2e = tmp_path / "m1-e2e.json"
    m1_e2e.write_text(
        json.dumps(_m1_e2e(binding)),
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
    assert report["model"]["values"]["strict_msgpack"] == {
        "runtime_flag": True,
        "serializer": "JsonPlusSerializer",
        "pickle_fallback": False,
        "allowed_msgpack_modules": sorted(
            report_builder._STRICT_MSGPACK_ALLOWLIST
        ),
    }
    assert report["tool"]["values"]["tool_trace"] == [
        "list_document_versions",
        "compare_document_versions",
        "search_knowledge",
        "fetch_evidence",
    ]
    assert report["tool"]["values"]["tool_calls"] == 4
    assert next(item for item in report["gates"] if item["name"] == "m1_research_e2e")["status"] == "passed"


def test_m1_e2e_hardcoded_legacy_report_fails_new_gates(tmp_path: Path) -> None:
    """缺少 strict 证据、tool_calls 或可读结果时，旧式硬编码报告必须判 failed。"""
    from scripts.build_project_acceptance_report import m1_e2e_status

    binding = {"git_revision": "revision", "worktree_sha256": "a" * 64}
    legacy = {
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
        "citation_count": 5,
    }

    assert m1_e2e_status("passed", legacy, "revision", "a" * 64) == "failed"

    for mutation in (
        {**legacy, "strict_msgpack": _strict_msgpack()},
        {
            **legacy,
            "strict_msgpack": _strict_msgpack(),
            "tool_calls": 3,
        },
        {
            **legacy,
            "strict_msgpack": {**_strict_msgpack(), "pickle_fallback": True},
            "tool_calls": 4,
        },
        {
            **legacy,
            "strict_msgpack": {**_strict_msgpack(), "serializer": "pickle"},
            "tool_calls": 4,
        },
        {
            **legacy,
            "strict_msgpack": {**_strict_msgpack(), "allowed_msgpack_modules": ["anything"]},
            "tool_calls": 4,
            "checkpoint_encodings": ["msgpack", "pickle"],
        },
        {
            **legacy,
            "strict_msgpack": _strict_msgpack(),
            "tool_calls": 4,
            "result_readable": False,
        },
    ):
        assert m1_e2e_status("passed", mutation, "revision", "a" * 64) == "failed"

    complete = {
        **mutation_all(legacy),
        "checkpoint_encodings": ["msgpack"],
        "tool_calls": 4,
        "result_readable": True,
        "strict_msgpack": _strict_msgpack(),
        "source_binding": binding,
    }
    assert m1_e2e_status("passed", complete, "revision", "a" * 64) == "passed"
    assert m1_e2e_status(
        "passed",
        {key: value for key, value in complete.items() if key != "source_binding"},
        "revision",
        "a" * 64,
    ) == "failed"


def mutation_all(legacy: dict) -> dict:
    return {**legacy}


def test_final_pass_requires_fresh_m1_m2_and_zero_skips(tmp_path: Path, monkeypatch) -> None:
    binding = _source_binding(monkeypatch)
    junit = tmp_path / "pytest.xml"
    junit.write_text('<testsuite tests="3" failures="0" errors="0" skipped="0" time="1"/>', encoding="utf-8")
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(
        json.dumps(
            {
                "status": "completed",
                "test": {
                    "metrics": {
                        "recall_at_5": 1.0,
                        "mrr": 1.0,
                        "no_answer_accuracy": 1.0,
                        "acl_leak_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("version: test\n", encoding="utf-8")
    m1_e2e = tmp_path / "m1-e2e.json"
    m1_e2e.write_text(json.dumps(_m1_e2e(binding)), encoding="utf-8")
    m2_e2e = tmp_path / "m2-e2e.json"
    m2_e2e.write_text(json.dumps(_m2_e2e(binding)), encoding="utf-8")

    kwargs = {
        "pytest_xml": junit,
        "retrieval_report": retrieval,
        "task_report": None,
        "dataset": dataset,
        "config": None,
        "m1_e2e_report": m1_e2e,
    }
    report = build_report(**kwargs, m2_e2e_report=m2_e2e)

    validate_report(report)
    assert report["verdict"]["status"] == "passed"
    assert report["m2"]["status"] == "passed"
    assert "cost" not in report
    assert all(gate["status"] == "passed" for gate in report["gates"])
    tampered = json.loads(json.dumps(report))
    tampered["m2"]["values"]["coverage"]["task_count"] = 31
    canonical = json.dumps(
        {key: value for key, value in tampered.items() if key != "report_hash"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() != report["report_hash"]

    no_m2 = build_report(**kwargs)
    assert no_m2["verdict"]["status"] == "unverified"


def test_m2_e2e_rejects_side_effects_and_stale_source_binding(monkeypatch) -> None:
    binding = _source_binding(monkeypatch)
    report = _m2_e2e(binding)

    report["side_effects"] = {"unapproved": 0, "duplicate": 1, "unknown_replay": 0}
    assert report_builder.m2_e2e_status("passed", report, "revision", "a" * 64) == "failed"

    report = _m2_e2e({"git_revision": "old", "worktree_sha256": "a" * 64})
    assert report_builder.m2_e2e_status("passed", report, "revision", "a" * 64) == "failed"


def test_m2_e2e_requires_terminal_audit_and_execution_binding(monkeypatch) -> None:
    binding = _source_binding(monkeypatch)

    missing_ledger = _m2_e2e(binding)
    del missing_ledger["task_ledger"]
    assert report_builder.m2_e2e_status("passed", missing_ledger, "revision", "a" * 64) == "failed"

    negative_model_calls = _m2_e2e(binding)
    negative_model_calls["task_ledger"][0]["model_calls"] = -1
    assert report_builder.m2_e2e_status("passed", negative_model_calls, "revision", "a" * 64) == "failed"

    pending_approval = _m2_e2e(binding)
    pending_approval["terminal_binding"]["approval"]["status"] = "pending"
    assert report_builder.m2_e2e_status("passed", pending_approval, "revision", "a" * 64) == "failed"

    mismatched_artifact = _m2_e2e(binding)
    mismatched_artifact["terminal_binding"]["task_result"]["artifact_id"] = "artifact-other"
    assert report_builder.m2_e2e_status("passed", mismatched_artifact, "revision", "a" * 64) == "failed"
