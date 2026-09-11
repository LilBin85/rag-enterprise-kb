from types import SimpleNamespace

from rag_kb import qdrant_init
from rag_kb.application.ports.indexing import DocumentIndexError


def test_qdrant_init_uses_runtime_collection_contract(monkeypatch) -> None:
    settings = SimpleNamespace(
        qdrant_url="http://qdrant:6333",
        qdrant_v2_collection="kb_chunks_v2",
        embedding_dim=1024,
    )
    client = object()
    captured = {}
    monkeypatch.setattr(qdrant_init, "get_settings", lambda: settings)
    monkeypatch.setattr(qdrant_init, "QdrantClient", lambda *, url: client)

    def ensure(actual_client, *, collection, dimension):
        captured.update(
            client=actual_client,
            collection=collection,
            dimension=dimension,
        )

    monkeypatch.setattr(qdrant_init, "ensure_document_collection", ensure)

    assert qdrant_init.run() == 0
    assert captured == {
        "client": client,
        "collection": "kb_chunks_v2",
        "dimension": 1024,
    }


def test_qdrant_init_fails_closed_with_safe_error(monkeypatch) -> None:
    settings = SimpleNamespace(
        qdrant_url="http://qdrant:6333",
        qdrant_v2_collection="kb_chunks_v2",
        embedding_dim=1024,
    )
    monkeypatch.setattr(qdrant_init, "get_settings", lambda: settings)
    monkeypatch.setattr(qdrant_init, "QdrantClient", lambda *, url: object())

    def fail(*args, **kwargs):
        raise DocumentIndexError("document_index_collection_schema_invalid")

    monkeypatch.setattr(qdrant_init, "ensure_document_collection", fail)

    assert qdrant_init.run() == 1
