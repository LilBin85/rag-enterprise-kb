"""FastAPI 接口集成测试(蓝图 8 节接口契约 + 10.2 上传校验用例)。

通过 dependency_overrides 注入测试 lifecycle(确定性假 embedder + 独立 collection),
验证端点行为与状态语义,不触碰真实模型与真实 collection。
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_kb.api import app, build_lifecycle
from rag_kb.embedder import DeterministicEmbedder
from rag_kb.lifecycle import DocumentLifecycle
from rag_kb.store import ChunkStore

METADATA = {
    "document_id": "api-leave-policy",
    "title": "API 测试制度",
    "department": "hr",
    "allowed_roles": ["hr", "admin"],
}

DOC_V1 = "# API 测试制度\n## 年假\n员工每年可享受带薪年假 10 天。\n"
DOC_V2 = "# API 测试制度\n## 年假\n员工每年可享受带薪年假 15 天。\n"


@pytest.fixture()
def client(store: ChunkStore, embedder: DeterministicEmbedder, tmp_path: Path) -> TestClient:
    test_lifecycle = DocumentLifecycle(
        store=store, embedder=embedder, upload_dir=tmp_path / "uploads", max_chars=800
    )
    app.dependency_overrides[build_lifecycle] = lambda: test_lifecycle
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _post(client: TestClient, content: str, name: str = "制度-v1.md", metadata: dict | None = None):
    return client.post(
        "/documents",
        files={"file": (name, content.encode("utf-8"), "text/markdown")},
        data={"metadata": json.dumps(metadata or METADATA, ensure_ascii=False)},
    )


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["qdrant"] == "ok"


def test_post_document_success(client: TestClient, store: ChunkStore) -> None:
    resp = _post(client, DOC_V1)
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "api-leave-policy"
    assert body["version"] == 1
    assert body["chunk_count"] == 1

    chunks = store.scroll_document("api-leave-policy")
    assert len(chunks) == 1
    assert chunks[0]["status"] == "active"
    assert chunks[0]["allowed_roles"] == ["hr", "admin"]


def test_post_rejects_unsupported_extension(client: TestClient, store: ChunkStore) -> None:
    resp = _post(client, "MZ 伪造内容", name="evil.exe")
    assert resp.status_code == 422
    assert store.scroll_document("api-leave-policy") == []  # 拒绝时不写入任何块


def test_put_creates_new_version_and_supersedes(client: TestClient) -> None:
    assert _post(client, DOC_V1).status_code == 200
    resp = client.put(
        "/documents/api-leave-policy",
        files={"file": ("制度-v2.md", DOC_V2.encode("utf-8"), "text/markdown")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 2
    assert body["previous_version"] == 1

    versions = client.get("/documents/api-leave-policy/versions").json()["versions"]
    assert [(v["version"], v["status"]) for v in versions] == [(1, "superseded"), (2, "active")]


def test_delete_document(client: TestClient) -> None:
    assert _post(client, DOC_V1).status_code == 200
    resp = client.delete("/documents/api-leave-policy", params={"mode": "delete"})
    assert resp.status_code == 200
    versions = client.get("/documents/api-leave-policy/versions").json()["versions"]
    assert versions and all(v["status"] == "deleted" for v in versions)


def test_withdraw_document(client: TestClient) -> None:
    assert _post(client, DOC_V1).status_code == 200
    resp = client.delete("/documents/api-leave-policy", params={"mode": "withdraw"})
    assert resp.status_code == 200
    versions = client.get("/documents/api-leave-policy/versions").json()["versions"]
    assert versions and all(v["status"] == "withdrawn" for v in versions)


def test_delete_rejects_bad_mode(client: TestClient) -> None:
    assert _post(client, DOC_V1).status_code == 200
    resp = client.delete("/documents/api-leave-policy", params={"mode": "bogus"})
    assert resp.status_code == 422
