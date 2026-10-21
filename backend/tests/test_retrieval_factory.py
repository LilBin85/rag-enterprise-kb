from types import SimpleNamespace

import pytest

from rag_kb.infrastructure.retrieval import factory


def _settings(**overrides):
    values = {
        "qdrant_url": "http://qdrant.test:6333",
        "qdrant_v2_collection": "documents",
        "retrieval_top_k": 5,
        "retrieval_candidate_k": 9,
        "retrieval_context_k": 4,
        "retrieval_context_token_budget": 1234,
        "retrieval_rrf_k": 77,
        "enable_hybrid_retrieval": True,
        "enable_reranker": True,
        "reranker_model": "BAAI/test-reranker",
        "retrieval_score_threshold": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factory_preserves_all_retrieval_settings(monkeypatch) -> None:
    database = object()
    embedder = object()
    client = object()
    reranker = object()

    class Authorization:
        def __init__(self, received_database) -> None:
            self.database = received_database

    class Vectors:
        def __init__(self, received_client, received_embedder, *, collection) -> None:
            self.client = received_client
            self.embedder = received_embedder
            self.collection = collection

    class Retriever:
        def __init__(self, authorization, vectors, **kwargs) -> None:
            self.authorization = authorization
            self.vectors = vectors
            self.kwargs = kwargs

    monkeypatch.setattr(factory, "BgeM3Embedder", lambda: embedder)
    monkeypatch.setattr(factory, "QdrantClient", lambda *, url: client)
    monkeypatch.setattr(factory, "DatabaseAuthorizationPort", Authorization)
    monkeypatch.setattr(factory, "QdrantAuthorizedSearch", Vectors)
    monkeypatch.setattr(factory, "BgeReranker", lambda model_name: reranker)
    monkeypatch.setattr(factory, "EnterpriseRetriever", Retriever)

    retriever = factory.build_enterprise_retriever(_settings(), database, consumer="durable")

    assert retriever.authorization.database is database
    assert retriever.vectors.client is client
    assert retriever.vectors.embedder is embedder
    assert retriever.vectors.collection == "documents"
    assert retriever.kwargs == {
        "candidate_k": 9,
        "context_k": 4,
        "context_token_budget": 1234,
        "rrf_k": 77,
        "hybrid_enabled": True,
        "dense_score_threshold": 0.25,
        "consumer": "durable",
        "reranker": reranker,
    }


def test_factory_omits_reranker_when_disabled(monkeypatch) -> None:
    class Retriever:
        def __init__(self, _authorization, _vectors, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(factory, "BgeM3Embedder", lambda: object())
    monkeypatch.setattr(factory, "QdrantClient", lambda *, url: object())
    monkeypatch.setattr(factory, "DatabaseAuthorizationPort", lambda _database: object())
    monkeypatch.setattr(factory, "QdrantAuthorizedSearch", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(factory, "EnterpriseRetriever", Retriever)

    retriever = factory.build_enterprise_retriever(
        _settings(enable_reranker=False), object(), consumer="fast"
    )

    assert retriever.kwargs["reranker"] is None
    assert retriever.kwargs["hybrid_enabled"] is True


def test_factory_rejects_invalid_limits_before_initializing_adapters(monkeypatch) -> None:
    initialized: list[str] = []

    def initialized_adapter(*_args, **_kwargs):
        initialized.append("adapter")
        raise AssertionError("invalid limits must fail first")

    monkeypatch.setattr(factory, "BgeM3Embedder", initialized_adapter)
    monkeypatch.setattr(factory, "QdrantClient", initialized_adapter)
    monkeypatch.setattr(factory, "DatabaseAuthorizationPort", initialized_adapter)
    monkeypatch.setattr(factory, "QdrantAuthorizedSearch", initialized_adapter)

    with pytest.raises(ValueError, match="retrieval_context_k_cannot_exceed_candidate_k"):
        factory.build_enterprise_retriever(
            _settings(retrieval_candidate_k=3, retrieval_context_k=4),
            object(),
            consumer="fast",
        )

    assert initialized == []
