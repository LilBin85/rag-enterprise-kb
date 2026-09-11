"""验证验收脚本会拒绝假通过，不连接模型或任何外部服务。"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from scripts.isolated_e2e_scenarios import (
    _pending_ingestion,
    _points,
    _verify_projection,
    _wait_task,
    metric_value,
)
from scripts.run_isolated_e2e import (
    OWNER_LABEL,
    RUN_LABEL,
    REPO,
    Harness,
    Redactor,
    evidence_directory,
    isolated_compose,
    verify_isolated_compose,
)
from rag_kb.domain.indexing import compute_chunk_manifest_digest


def check(condition, code):
    if not condition:
        raise RuntimeError(code)


def test_metric_requires_exact_name_and_outcome():
    text = ('rag_kb_task_runs_total{mode="research",outcome="succeeded"} 2\n'
            'rag_kb_task_runs_total{mode="controlled_action",outcome="failure"} 7\n'
            'rag_kb_checkpoint_writes_total{outcome="succeeded"} 9\n')
    assert metric_value(text, "rag_kb_task_runs_total", outcome="succeeded") == 2
    assert metric_value(text, "rag_kb_task_runs_total", outcome="missing") == 0


@pytest.mark.parametrize("status", ["failed", "refused", "cancelled", "unknown", "waiting_for_user"])
def test_terminal_or_unexpected_clarification_fails_immediately(status):
    harness = SimpleNamespace(
        check=check, request=lambda *args, **kwargs: SimpleNamespace(json=lambda: {"status": status}),
        poll=lambda label, fn, **kwargs: fn(),
    )
    with pytest.raises(RuntimeError, match="task_unexpected_terminal_or_clarification"):
        _wait_task(harness, "task-1")


@pytest.mark.parametrize("status,attempts", [("running", 1), ("succeeded", 1), ("pending", 1)])
def test_backlog_rejects_consumption_while_worker_stopped(status, attempts):
    harness = SimpleNamespace(check=check, request=lambda *a, **kw: SimpleNamespace(
        json=lambda: {"status": status, "attempts": attempts}))
    with pytest.raises(RuntimeError, match="ingestion_consumed_without_worker"):
        _pending_ingestion(harness, {"ingestion_job_id": "job-1"})


def test_scroll_keeps_binding_and_follows_all_pages():
    calls = []

    def qdrant(method, path, json):
        calls.append(json)
        return {"result": {"points": [{"id": len(calls)}],
                           "next_page_offset": "page-two" if len(calls) == 1 else None}}

    harness = SimpleNamespace(collection="isolated", qdrant=qdrant)
    binding = {"tenant_id": "demo", "document_id": "test", "applied_projection_version_id": "v1",
               "applied_projection_generation": 3}
    assert [p["id"] for p in _points(harness, binding)] == [1, 2]
    assert calls[1]["offset"] == "page-two"
    filters = {item["key"]: item["match"]["value"] for item in calls[0]["filter"]["must"]}
    assert filters == {"tenant_id": "demo", "document_id": "test", "version_id": "v1",
                        "projection_generation": 3}


@pytest.mark.parametrize("run_id", ["short", "../escape", "run/escape", "Run-001"])
def test_evidence_directory_rejects_invalid_or_traversing_run_ids(run_id):
    with pytest.raises(ValueError, match="invalid_run_id"):
        evidence_directory(run_id)


def test_isolated_compose_rejects_shared_ports_and_unlabelled_build(tmp_path):
    source = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    config = isolated_compose(source, run_id="run-001", owner="owner-001", model_cache=tmp_path)
    verify_isolated_compose(config, run_id="run-001", owner="owner-001", model_cache=tmp_path)
    assert config["services"]["postgres"].get("ports") is None
    assert config["services"]["qdrant"].get("ports") is None
    assert config["services"]["api"]["build"]["labels"] == {
        OWNER_LABEL: "owner-001", RUN_LABEL: "run-001"
    }

    corrupted = deepcopy(config)
    corrupted["services"]["postgres"]["ports"] = ["127.0.0.1:5432:5432"]
    with pytest.raises(ValueError, match="isolated_shared_port_invalid"):
        verify_isolated_compose(corrupted, run_id="run-001", owner="owner-001", model_cache=tmp_path)

    corrupted = deepcopy(config)
    corrupted["services"]["api"]["build"]["labels"] = {}
    with pytest.raises(ValueError, match="isolated_build_labels_invalid"):
        verify_isolated_compose(corrupted, run_id="run-001", owner="owner-001", model_cache=tmp_path)


def test_sql_accepts_only_registered_read_queries(tmp_path):
    harness = Harness("run-001", tmp_path)
    captured = {}
    harness.exec_python = lambda code, payload: captured.update(payload) or "[]"
    assert harness.sql("document_projection", {"id": "document-1"}) == []
    assert captured["params"] == {"id": "document-1"}
    assert "FROM documents" in captured["query"]
    with pytest.raises(RuntimeError, match="sql_query_not_allowlisted"):
        harness.sql("SELECT * FROM documents")
    with pytest.raises(RuntimeError, match="sql_parameters_invalid"):
        harness.sql("document_projection")


@pytest.mark.parametrize("field, value", [("current", False), ("lifecycle_status", "indexing"), ("document_id", "other-document")])
def test_qdrant_fault_requires_the_exact_active_point_binding(tmp_path, field, value):
    harness = Harness("run-001", tmp_path)
    payload = {
        "tenant_id": "local-demo", "document_id": "e2e-run-001-projection", "version_id": "version-1",
        "projection_generation": 2, "projection_key": "sha256:projection", "current": True,
        "lifecycle_status": "active",
    }
    expected = {"point_id": "point-1", **payload}
    calls = []
    harness.container_identity = lambda service: {"id": service}
    harness.record = lambda *args: None

    def internal_http(method, url, body):
        calls.append((method, url, body))
        if url.endswith("/points"):
            return json.dumps({"result": [{"id": "point-1", "payload": payload}]})
        return json.dumps({"result": {"status": "completed"}})

    harness.internal_http = internal_http
    payload[field] = value
    with pytest.raises(RuntimeError, match="fault_point_binding_invalid"):
        harness.qdrant("POST", f"/collections/{harness.collection}/points/delete?wait=true",
                       {"points": ["point-1"]}, expected_binding=expected)
    assert not any(url.endswith("/points/delete?wait=true") for _, url, _ in calls)


def test_browser_report_rejects_missing_or_skipped_live_test(tmp_path):
    harness = Harness("run-001", tmp_path)
    browser = tmp_path / "browser"
    browser.mkdir()
    (browser / "browser-report.json").write_text(json.dumps({
        "stats": {"expected": 0, "skipped": 1, "unexpected": 0, "flaky": 0},
        "errors": [], "suites": [],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="browser_report_not_clean"):
        harness.verify_browser_report()


def _owned_container(harness, *, created="created"):
    return {
        "id": "container-1", "name": "/container-1", "created": created, "state": "created",
        "started": "", "health": None, "exit_code": 0, "image": "sha256:image",
        "labels": {"com.docker.compose.project": harness.project, "com.docker.compose.service": "api",
                   OWNER_LABEL: harness.owner, RUN_LABEL: harness.run_id},
    }


def test_cleanup_discovers_and_removes_resources_after_partial_startup(tmp_path):
    harness = Harness("run-001", tmp_path, docker="docker")
    harness.manifest.update(daemon_id="daemon", existing_container_ids=[])
    resources = {"container-1": _owned_container(harness)}
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        if args[1:3] == ["ps", "-a"]:
            return SimpleNamespace(stdout="container-1\n" if resources else "")
        if args[1:3] == ["rm", "-f"]:
            resources.pop(args[-1])
        return SimpleNamespace(stdout="")

    harness._run = run
    harness.docker_json = lambda *args: "daemon"
    harness.inspect_resource = lambda kind, identifier: resources[identifier]
    harness.discover()
    assert harness.cleanup()
    assert not resources
    assert any(command[1:3] == ["rm", "-f"] for command in commands)


def test_cleanup_identity_mismatch_never_deletes_resource(tmp_path):
    harness = Harness("run-001", tmp_path, docker="docker")
    expected = _owned_container(harness)
    harness.manifest.update(daemon_id="daemon", existing_container_ids=[], containers={expected["id"]: expected})
    commands = []
    harness.docker_json = lambda *args: "daemon"
    harness.discover = lambda: None
    harness.inspect_resource = lambda kind, identifier: _owned_container(harness, created="different")
    harness._run = lambda args, **kwargs: commands.append(args) or SimpleNamespace(stdout="")
    assert not harness.cleanup()
    assert not any(command[1:3] == ["rm", "-f"] for command in commands)


def test_cleanup_is_reentrant_and_redacts_credentials(tmp_path):
    harness = Harness("run-001", tmp_path)
    assert harness.cleanup()
    assert harness.cleanup()
    redactor = Redactor(["model-secret"])
    assert "model-secret" not in redactor.text("Bearer model-secret postgres://user:model-secret@host/db")


def test_container_inspection_allows_services_without_a_healthcheck(tmp_path):
    harness = Harness("run-001", tmp_path, docker="docker")
    harness.docker_json = lambda *args: {
        "id": "container-1", "name": "/container-1", "labels": {}, "created": "created",
        "state_detail": {"Status": "created", "StartedAt": "", "ExitCode": 0}, "image": "sha256:image",
    }
    assert harness.inspect_resource("containers", "container-1") == {
        "id": "container-1", "name": "/container-1", "labels": {}, "created": "created",
        "state": "created", "started": "", "health": None, "exit_code": 0, "image": "sha256:image",
    }


def test_wait_healthy_checks_the_container_before_returning(tmp_path):
    harness = Harness("run-001", tmp_path)
    healthy = {"state": "running", "health": "healthy"}
    harness.container_identity = lambda service: healthy
    harness.poll = lambda label, probe, **kwargs: probe()
    assert harness.wait_healthy("postgres") is healthy


@pytest.mark.parametrize("corruption", ["manifest", "old_current", "count", "text"])
def test_projection_verification_rejects_inconsistent_evidence(corruption):
    import hashlib

    chunk = {"chunk_id": "chunk-1", "content_hash": "sha256:" + hashlib.sha256(b"hello").hexdigest(),
             "source_locator": "line-1"}
    digest = compute_chunk_manifest_digest([SimpleNamespace(**chunk)])
    binding = {"tenant_id": "demo", "document_id": "test", "applied_projection_version_id": "v1",
               "applied_projection_generation": 3}
    receipt = {"indexed_chunk_count": 1, "index_manifest_digest": digest, "index_completed_at": "now"}
    if corruption == "manifest":
        receipt["index_manifest_digest"] = "sha256:wrong"
    if corruption == "count":
        receipt["indexed_chunk_count"] = 2

    def sql(query, params):
        if query == "document_projection":
            return [binding]
        if query == "document_version_projection":
            return [{"lifecycle_status": "active", "applied_projection_generation": 3}]
        if query == "ingestion_receipt":
            return [receipt]
        return [chunk]

    def qdrant(method, path, json):
        payload = {**chunk, "text": "wrong" if corruption == "text" else "hello",
                   "current": True, "lifecycle_status": "active"}
        points = [{"id": "point-1", "payload": payload}]
        if corruption == "old_current" and any(
            f["key"] == "current" for f in json["filter"]["must"]
        ):
            points.append({"id": "old-point", "payload": deepcopy(payload)})
        return {"result": {"points": points, "next_page_offset": None}}

    harness = SimpleNamespace(collection="isolated", sql=sql, qdrant=qdrant, check=check)
    with pytest.raises(RuntimeError, match="projection|old_projection"):
        _verify_projection(harness, {"document_id": "test", "version_id": "v1", "ingestion_job_id": "j1"})
