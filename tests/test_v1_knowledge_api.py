from fastapi.testclient import TestClient

from rag_kb.application.query_service import KnowledgeQueryResult
from rag_kb.domain.identity import AuthContext, Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.models import Refusal
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import build_identity_provider, build_knowledge_query_service


class CapturingQueryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AuthContext]] = []

    def query(self, question: str, auth: AuthContext) -> KnowledgeQueryResult:
        self.calls.append((question, auth))
        return KnowledgeQueryResult(
            verdict="refusal",
            refusal=Refusal(kind="test", message="测试拒答"),
            citations=(),
        )


def test_v1_query_uses_server_owned_identity_without_body_role() -> None:
    app = create_app()
    service = CapturingQueryService()
    provider = DemoTokenIdentityProvider(
        {
            "server-token": Principal(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=frozenset({"engineer"}),
                groups=frozenset({"research"}),
            )
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_knowledge_query_service] = lambda: service

    response = TestClient(app).post(
        "/v1/knowledge/query",
        json={"question": "发布前需要哪些检查？"},
        headers={"Authorization": "Bearer server-token", "X-Request-ID": "request-1"},
    )

    assert response.status_code == 200
    assert service.calls[0][0] == "发布前需要哪些检查？"
    assert service.calls[0][1].tenant_id == "tenant-a"
    assert service.calls[0][1].user_id == "user-a"
    assert service.calls[0][1].request_id == "request-1"
    assert response.headers["X-Request-ID"] == "request-1"


def test_v1_query_rejects_body_role_and_unknown_token() -> None:
    app = create_app()
    provider = DemoTokenIdentityProvider(
        {
            "valid": Principal(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=frozenset({"admin"}),
            )
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_knowledge_query_service] = lambda: CapturingQueryService()
    client = TestClient(app)

    body_role = client.post(
        "/v1/knowledge/query",
        json={"question": "问题", "role": "admin"},
        headers={"Authorization": "Bearer valid"},
    )
    no_identity = client.post(
        "/v1/knowledge/query",
        json={"question": "问题"},
        headers={"Authorization": "Bearer unknown"},
    )

    assert body_role.status_code == 422
    assert no_identity.status_code == 403
    assert body_role.headers["X-Request-ID"]
    assert no_identity.headers["X-Request-ID"]


def test_v1_query_normalizes_invalid_request_id_and_preserves_64_char_boundary() -> None:
    app = create_app()
    service = CapturingQueryService()
    app.dependency_overrides[build_identity_provider] = lambda: DemoTokenIdentityProvider(
        {
            "valid": Principal(
                tenant_id="tenant-a",
                user_id="user-a",
                roles=frozenset({"admin"}),
            )
        }
    )
    app.dependency_overrides[build_knowledge_query_service] = lambda: service

    client = TestClient(app)
    valid_boundary = client.post(
        "/v1/knowledge/query",
        json={"question": "问题"},
        headers={"Authorization": "Bearer valid", "X-Request-ID": "x" * 64},
    )
    invalid = client.post(
        "/v1/knowledge/query",
        json={"question": "问题"},
        headers={"Authorization": "Bearer valid", "X-Request-ID": "x" * 65},
    )

    assert valid_boundary.status_code == 200
    assert valid_boundary.headers["X-Request-ID"] == "x" * 64
    assert invalid.status_code == 200
    assert invalid.headers["X-Request-ID"] != "x" * 65
    assert len(invalid.headers["X-Request-ID"]) == 32
    assert service.calls[0][1].request_id == "x" * 64
    assert service.calls[1][1].request_id == invalid.headers["X-Request-ID"]
