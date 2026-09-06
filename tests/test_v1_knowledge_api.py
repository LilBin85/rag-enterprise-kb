"""旧快速问答写入口已收敛到统一会话。"""

from fastapi.testclient import TestClient

from rag_kb.web.app import create_app


def test_legacy_knowledge_post_is_removed_from_router_and_openapi() -> None:
    app = create_app()
    response = TestClient(app).post(
        "/v1/knowledge/query", json={"question": "问题", "role": "admin"}
    )

    assert response.status_code == 404
    assert "/v1/knowledge/query" not in app.openapi()["paths"]
