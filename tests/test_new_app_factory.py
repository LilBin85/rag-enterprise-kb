import subprocess
import sys

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SpanExporter

from rag_kb import observability as observability_module
from rag_kb.config import Settings
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import ReadinessFailure, build_readiness_probe


class ReadyProbe:
    def check(self) -> dict[str, str]:
        return {"qdrant": "ok"}


class FailedProbe:
    def check(self) -> dict[str, str]:
        raise ReadinessFailure("postgresql")


def test_live_probe_has_no_external_dependency() -> None:
    client = TestClient(create_app())

    assert client.get("/health/live").json() == {"status": "ok"}


def test_api_startup_does_not_import_worker_or_agent_runtime() -> None:
    blocked = (
        "rag_kb.worker",
        "rag_kb.agent.factory",
        "rag_kb.agent.checkpointer",
        "rag_kb.agent.tools",
        "rag_kb.embedder",
        "rag_kb.answerer",
    )
    code = f"""
import builtins
real_import = builtins.__import__
blocked = {blocked!r}
def guarded_import(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from rag_kb.config import Settings
from rag_kb.web.app import create_app
create_app(Settings(_env_file=None))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ready_probe_reports_dependencies() -> None:
    app = create_app()
    app.dependency_overrides[build_readiness_probe] = lambda: ReadyProbe()

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "dependencies": {"qdrant": "ok"}}


def test_ready_probe_fails_closed() -> None:
    app = create_app()
    app.dependency_overrides[build_readiness_probe] = lambda: FailedProbe()

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "dependencies": {"postgresql": "unavailable"},
    }


def test_factory_only_exposes_versioned_contracts() -> None:
    paths = create_app().openapi()["paths"]

    assert "/v1/conversations" in paths
    assert "/v1/conversations/{conversation_id}/messages" in paths
    assert "/v1/knowledge/query" not in paths
    assert "/v1/tasks" not in paths
    assert "/questions" not in paths
    assert "/documents" not in paths


def test_http_observability_returns_request_ids_and_uses_route_templates_for_metrics() -> None:
    app = create_app(Settings(_env_file=None, metrics_enabled=True))
    with TestClient(app) as client:
        live = client.get("/health/live", headers={"X-Request-ID": "invalid request id"})
        dynamic = client.get("/v1/tasks/task-visible")
        metrics = client.get("/metrics")

    assert live.status_code == 200
    assert len(live.headers["X-Request-ID"]) == 32
    assert dynamic.status_code == 403
    assert dynamic.headers["X-Request-ID"]
    assert metrics.status_code == 200
    assert 'route="/v1/tasks/{task_id}"' in metrics.text
    assert "task-visible" not in metrics.text


def test_metrics_disabled_keeps_business_routes_available() -> None:
    app = create_app(
        Settings(
            _env_file=None,
            observability_enabled=True,
            metrics_enabled=False,
            otel_exporter_otlp_endpoint="",
        )
    )

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/metrics").status_code == 404


def test_unavailable_otlp_exporter_does_not_fail_business_request(monkeypatch) -> None:
    class FailingExporter(SpanExporter):
        def export(self, _spans):
            raise OSError("synthetic endpoint unavailable")

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(
        observability_module,
        "OTLPSpanExporter",
        lambda **_kwargs: FailingExporter(),
    )
    app = create_app(
        Settings(
            _env_file=None,
            observability_enabled=True,
            metrics_enabled=True,
            otel_exporter_otlp_endpoint="http://127.0.0.1:1",
        )
    )

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200


def test_multiple_dynamic_ids_do_not_create_high_cardinality_metric_labels() -> None:
    app = create_app(Settings(_env_file=None, metrics_enabled=True))
    dynamic_markers = ("task-alpha", "task-beta", "document-alpha", "document-beta")

    with TestClient(app) as client:
        for marker in dynamic_markers[:2]:
            assert client.get(f"/v1/tasks/{marker}").status_code == 403
        for marker in dynamic_markers[2:]:
            assert client.get(f"/v1/documents/{marker}").status_code == 403

    forbidden_labels = {
        "document_id",
        "task_id",
        "run_id",
        "user_id",
        "url",
        "question",
        "content",
        "hash",
    }
    samples = [
        sample
        for metric in app.state.observability.registry.collect()
        for sample in metric.samples
    ]

    assert samples
    assert all(forbidden_labels.isdisjoint(sample.labels) for sample in samples)
    assert all(
        marker not in str(sample.labels)
        for sample in samples
        for marker in dynamic_markers
    )
    routes = {
        sample.labels.get("route")
        for sample in samples
        if sample.name == "rag_kb_http_requests_total"
    }
    assert {"/v1/tasks/{task_id}", "/v1/documents/{document_id}"} <= routes
