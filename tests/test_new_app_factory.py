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
    client = TestClient(create_app())

    assert client.get("/health/live").json() == {"status": "ok"}


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

    assert "/v1/knowledge/query" in paths
    assert "/questions" not in paths
    assert "/documents" not in paths
