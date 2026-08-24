from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from rag_kb.infrastructure.vectorstore.qdrant_collection import (
    DocumentIndexError,
    ensure_document_collection,
    validate_document_collection,
)


class FakeQdrantClient:
    def __init__(self, *, exists: bool = False, dimension: int = 2) -> None:
        self.exists = exists
        self.created: list[dict] = []
        self.payload_indexes: list[dict] = []
        self.dimension = dimension
        self.distance = models.Distance.COSINE
        self.modifier = models.Modifier.IDF
        self.payload_schema: dict[str, object] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs) -> None:
        self.created.append(kwargs)
        self.exists = True

    def get_collection(self, collection_name: str):
        return SimpleNamespace(
            payload_schema=self.payload_schema,
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "dense": models.VectorParams(
                            size=self.dimension, distance=self.distance
                        )
                    },
                    sparse_vectors={
                        "keyword": models.SparseVectorParams(modifier=self.modifier)
                    },
                )
            ),
        )

    def create_payload_index(self, **kwargs) -> None:
        self.payload_indexes.append(kwargs)
        self.payload_schema[kwargs["field_name"]] = SimpleNamespace(
            data_type=kwargs["field_schema"]
        )


def test_creates_complete_named_collection_and_is_idempotent() -> None:
    client = FakeQdrantClient()

    ensure_document_collection(client, collection="kb_chunks_v2", dimension=2)
    ensure_document_collection(client, collection="kb_chunks_v2", dimension=2)

    assert len(client.created) == 1
    assert client.created[0]["vectors_config"]["dense"].distance == models.Distance.COSINE
    assert client.created[0]["sparse_vectors_config"]["keyword"].modifier == models.Modifier.IDF
    assert len(client.payload_indexes) == 9
    assert all(call["wait"] is True for call in client.payload_indexes)


@pytest.mark.parametrize("schema_fault", ["dense", "sparse", "payload"])
def test_incompatible_existing_collection_fails_closed_without_recreating(schema_fault: str) -> None:
    client = FakeQdrantClient(exists=True)
    if schema_fault == "dense":
        client.dimension = 3
    elif schema_fault == "sparse":
        client.modifier = models.Modifier.NONE
    else:
        client.payload_schema["tenant_id"] = SimpleNamespace(
            data_type=models.PayloadSchemaType.INTEGER
        )

    with pytest.raises(DocumentIndexError, match="document_index_collection_schema_invalid"):
        ensure_document_collection(client, collection="kb_chunks_v2", dimension=2)

    assert client.created == []


def test_create_race_accepts_only_a_compatible_collection() -> None:
    class RaceClient(FakeQdrantClient):
        def create_collection(self, **kwargs) -> None:
            self.exists = True
            raise RuntimeError("already exists")

    client = RaceClient()
    ensure_document_collection(client, collection="kb_chunks_v2", dimension=2)
    assert len(client.payload_indexes) == 9


def test_readonly_validation_rejects_missing_payload_index() -> None:
    client = FakeQdrantClient(exists=True)

    with pytest.raises(DocumentIndexError, match="document_index_collection_schema_invalid"):
        validate_document_collection(client.get_collection("kb_chunks_v2"), dimension=2)
