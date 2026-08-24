"""旧 API 测试迁移入口；业务断言位于版本化/兼容公开路由测试。"""

from fastapi.testclient import TestClient

from rag_kb.web.app import create_app


def test_legacy_routes_are_opt_in() -> None:
    client = TestClient(create_app())
    assert client.post("/questions", json={"question": "问题", "role": "engineer"}).status_code == 404


def test_explicit_legacy_app_exposes_thin_compatibility_routes() -> None:
    client = TestClient(create_app(include_legacy_routes=True))
    # 未提供身份时路由应进入认证，而不是 404 或旧实现。
    assert client.post("/questions", json={"question": "问题", "role": "engineer"}).status_code == 403
