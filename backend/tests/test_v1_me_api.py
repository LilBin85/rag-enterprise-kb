from fastapi.testclient import TestClient

from rag_kb.domain.identity import Permission, Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import build_identity_provider


def test_me_returns_only_effective_identity_facts() -> None:
    app = create_app()
    app.dependency_overrides[build_identity_provider] = lambda: DemoTokenIdentityProvider(
        {
            "admin-token": Principal(
                tenant_id="tenant-a",
                user_id="admin-a",
                roles=frozenset({"admin", "operator"}),
                attributes={"permissions": ["unknown:permission"], "secret": "hidden"},
            ),
        }
    )
    with TestClient(app) as client:
        response = client.get("/v1/me", headers={"Authorization": "Bearer admin-token"})
        missing_identity = client.get("/v1/me")

    assert response.status_code == 200
    assert response.json() == {
        "tenant_id": "tenant-a",
        "user_id": "admin-a",
        "roles": ["admin", "operator"],
        "permissions": [permission.value for permission in Permission],
    }
    assert missing_identity.status_code == 403
