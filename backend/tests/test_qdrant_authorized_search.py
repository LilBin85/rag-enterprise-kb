from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import (
    AuthorizedSearchError,
    QdrantAuthorizedSearch,
)
from rag_kb.infrastructure.vectorstore.qdrant_document_index import projection_key


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class SpyClient:
    def __init__(self, points=()):
        self.points = tuple(points)
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)


def point(**overrides):
    text = "授权正文"
    payload = {
        "tenant_id": "tenant-a",
        "document_id": "doc-a",
        "version_id": "version-a",
        "projection_generation": 1,
        "projection_key": projection_key("tenant-a", "version-a", 1),
        "document_version": 1,
        "chunk_id": "chunk-a",
        "content_hash": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_locator": "Section A",
        "source_file_name": "policy.md",
        "title": "Travel Policy",
        "text": text,
    }
    payload.update(overrides)
    return SimpleNamespace(payload=payload, score=0.9)


def condition_by_key(query_filter: models.Filter) -> dict[str, models.FieldCondition]:
    return {condition.key: condition for condition in query_filter.must}


def test_dense_and_sparse_use_same_server_authorization_filter() -> None:
    client = SpyClient([point()])
    adapter = QdrantAuthorizedSearch(client, FakeEmbedder(), collection="test-v2")
    arguments = {
        "tenant_id": "tenant-a",
        "version_ids": frozenset({"version-b", "version-a"}),
        "projection_generations": frozenset({("version-a", 1), ("version-b", 1)}),
        "allowed_principals": frozenset({"group:finance", "user:alice"}),
        "limit": 7,
    }

    dense = adapter.search_dense("差旅", **arguments)
    sparse = adapter.search_sparse("差旅", **arguments)

    assert dense[0].dense_score == 0.9 and dense[0].sparse_score is None
    assert dense[0].document_title == "Travel Policy"
    assert sparse[0].sparse_score == 0.9 and sparse[0].dense_score is None
    assert [call["using"] for call in client.calls] == ["dense", "keyword"]
    first, second = (condition_by_key(call["query_filter"]) for call in client.calls)
    assert first.keys() == second.keys() == {
        "tenant_id", "version_id", "projection_key", "lifecycle_status", "current", "allowed_principals"
    }
    assert first["tenant_id"].match.value == "tenant-a"
    assert first["version_id"].match.any == ["version-a", "version-b"]
    assert first["projection_key"].match.any == [
        projection_key("tenant-a", "version-a", 1),
        projection_key("tenant-a", "version-b", 1),
    ]
    assert first["lifecycle_status"].match.value == "active"
    assert first["current"].match.value is True
    assert first["allowed_principals"].match.any == ["group:finance", "user:alice"]
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["with_vectors"] is False


def test_admin_bypass_only_omits_principal_filter() -> None:
    client = SpyClient()
    adapter = QdrantAuthorizedSearch(client, FakeEmbedder())
    adapter.search_dense(
        "query",
        tenant_id="tenant-a",
        version_ids=frozenset({"version-a"}),
        projection_generations=frozenset({("version-a", 1)}),
        allowed_principals=frozenset({"role:admin"}),
        limit=5,
        principal_filter_required=False,
    )

    conditions = condition_by_key(client.calls[0]["query_filter"])
    assert "allowed_principals" not in conditions
    assert set(conditions) == {"tenant_id", "version_id", "projection_key", "lifecycle_status", "current"}


@pytest.mark.parametrize(
    "payload_patch",
    [
        {"text": ""},
        {"version_id": None},
        {"document_version": True},
    ],
)
def test_payload_missing_security_binding_fails_closed(payload_patch) -> None:
    adapter = QdrantAuthorizedSearch(SpyClient([point(**payload_patch)]), FakeEmbedder())
    with pytest.raises(AuthorizedSearchError, match="payload_invalid"):
        adapter.search_dense(
            "query",
            tenant_id="tenant-a",
            version_ids=frozenset({"version-a"}),
            projection_generations=frozenset({("version-a", 1)}),
            allowed_principals=frozenset({"user:alice"}),
            limit=5,
        )


def test_invalid_or_empty_scope_fails_before_qdrant() -> None:
    client = SpyClient()
    adapter = QdrantAuthorizedSearch(client, FakeEmbedder())
    with pytest.raises(AuthorizedSearchError, match="scope_empty"):
        adapter.search_dense(
            "query",
            tenant_id="tenant-a",
            version_ids=frozenset(),
            allowed_principals=frozenset({"user:alice"}),
            limit=5,
        )
    with pytest.raises(AuthorizedSearchError, match="principals_invalid"):
        adapter.search_dense(
            "query",
            tenant_id="tenant-a",
            version_ids=frozenset({"version-a"}),
            projection_generations=frozenset({("version-a", 1)}),
            allowed_principals=frozenset({"alice"}),
            limit=5,
        )
    assert client.calls == []


def test_tampered_text_with_stale_hash_fails_closed() -> None:
    adapter = QdrantAuthorizedSearch(
        SpyClient([point(text="被篡改的正文")]),
        FakeEmbedder(),
    )
    with pytest.raises(AuthorizedSearchError, match="content_hash_mismatch"):
        adapter.search_dense(
            "query",
            tenant_id="tenant-a",
            version_ids=frozenset({"version-a"}),
            projection_generations=frozenset({("version-a", 1)}),
            allowed_principals=frozenset({"user:alice"}),
            limit=5,
        )
