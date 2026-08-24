from fastapi.testclient import TestClient

from rag_kb.web.app import create_app
from rag_kb.web.dependencies import ReadinessFailure, build_readiness_probe


class ReadyProbe:
    def check(self) -> dict[str, str]:
        return {"qdrant": "ok"}


class FailedProbe:
    def check(self) -> dict[str, str]:
        raise ReadinessFailure("postgresql")


def test_live_probe_has_no_external_dependency() -> None:
    client = TestClient(create_app(include_legacy_routes=False))

    assert client.get("/health/live").json() == {"status": "ok"}


def test_ready_probe_reports_dependencies() -> None:
    app = create_app(include_legacy_routes=False)
    app.dependency_overrides[build_readiness_probe] = lambda: ReadyProbe()

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "dependencies": {"qdrant": "ok"}}


def test_ready_probe_fails_closed() -> None:
    app = create_app(include_legacy_routes=False)
    app.dependency_overrides[build_readiness_probe] = lambda: FailedProbe()

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "dependencies": {"postgresql": "unavailable"},
    }


def test_explicit_compatibility_factory_exposes_new_and_legacy_contracts() -> None:
    app = create_app(include_legacy_routes=True)

    paths = app.openapi()["paths"]

    assert "/health/live" in paths
    assert "/health/ready" in paths
    assert "/v1/knowledge/query" in paths
    assert "/questions" in paths
    assert "/policy-comparisons" in paths
    assert "/policy-rollout-tasks" in paths


def test_default_factory_does_not_expose_unauthenticated_legacy_routes() -> None:
    paths = create_app().openapi()["paths"]

    assert "/v1/knowledge/query" in paths
    assert "/questions" not in paths
    assert "/documents" not in paths
