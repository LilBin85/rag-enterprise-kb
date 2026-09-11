from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_kb.application.enterprise_retriever import EnterpriseRetriever, RetrievalSecurityError
from rag_kb.application.ports.retrieval import AuthorizedVersionSet, RankedChunk
from rag_kb.domain.identity import AuthContext
from rag_kb.observability import Observability


def auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="alice",
        groups=frozenset({"finance"}),
        attributes={"permissions": ["knowledge:read"]},
        request_id="retrieval-security",
    )


def chunk() -> RankedChunk:
    text = "授权差旅正文"
    return RankedChunk(
        tenant_id="tenant-a",
        document_id="policy",
        version_id="version-1",
        document_version=1,
        chunk_id="chunk-1",
        content_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_locator="section/1",
        source_file_name="policy.md",
        text=text,
        dense_score=0.9,
        sparse_score=0.8,
    )


class Authorization:
    def __init__(self, *, exact=True):
        self.exact = exact
        self.exact_calls = []

    def authorized_active_versions(self, identity):
        return AuthorizedVersionSet(
            identity.tenant_id,
            frozenset({"version-1"}),
            frozenset({("version-1", 1)}),
            frozenset({("policy", "version-1", 1)}),
        )

    def revalidate_versions(self, identity, version_ids):
        return version_ids & frozenset({"version-1"})

    def revalidate_candidates(self, identity, candidates):
        self.exact_calls.append(tuple(candidates))
        return (
            frozenset({("version-1", "chunk-1")})
            if self.exact
            else frozenset()
        )


class Vectors:
    def __init__(self, item=None):
        self.item = item or chunk()
        self.calls = []

    def search_dense(self, query, **kwargs):
        self.calls.append(("dense", kwargs))
        return (self.item,)

    def search_sparse(self, query, **kwargs):
        self.calls.append(("sparse", kwargs))
        return (self.item,)


class Reranker:
    def __init__(self, transform):
        self.transform = transform

    def rerank(self, query, candidates):
        return tuple(self.transform(item) for item in candidates)


def _telemetry() -> tuple[Observability, InMemorySpanExporter]:
    observability = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    observability.start()
    exporter = InMemorySpanExporter()
    assert observability.provider is not None
    observability.provider.add_span_processor(SimpleSpanProcessor(exporter))
    return observability, exporter


def test_hybrid_routes_share_scope_and_exact_sql_revalidation() -> None:
    authorization = Authorization()
    vectors = Vectors()
    result = EnterpriseRetriever(
        authorization,
        vectors,
        hybrid_enabled=True,
        candidate_k=5,
        context_k=2,
        context_token_budget=100,
    ).search("差旅", auth())

    assert [call[0] for call in vectors.calls] == ["dense", "sparse"]
    assert all(call[1]["tenant_id"] == "tenant-a" for call in vectors.calls)
    assert all(call[1]["version_ids"] == frozenset({"version-1"}) for call in vectors.calls)
    assert all(call[1]["projection_generations"] == frozenset({("version-1", 1)}) for call in vectors.calls)
    assert all(call[1]["allowed_principals"] == auth().acl_subjects for call in vectors.calls)
    assert authorization.exact_calls
    assert [item.chunk_id for item in result.candidates] == ["chunk-1"]


def test_explicit_document_scope_intersects_authorized_versions() -> None:
    vectors = Vectors()
    retriever = EnterpriseRetriever(Authorization(), vectors)

    result = retriever.search("差旅", auth(), scope_document_id="policy")
    unavailable = retriever.search("差旅", auth(), scope_document_id="other")

    assert [item.document_id for item in result.candidates] == ["policy"]
    assert vectors.calls[0][1]["version_ids"] == frozenset({"version-1"})
    assert unavailable.candidates == ()
    assert unavailable.insufficiency_code == "document_scope_unavailable"


def test_explicit_document_scope_rejects_mismatched_vector_binding() -> None:
    with pytest.raises(RetrievalSecurityError, match="acl_filter_contract"):
        EnterpriseRetriever(
            Authorization(),
            Vectors(replace(chunk(), document_id="other")),
        ).search("差旅", auth(), scope_document_id="policy")


def test_retrieval_telemetry_records_all_stages_without_dynamic_input_labels() -> None:
    observability, exporter = _telemetry()
    secret = "query private-token document/body C:/private/object"
    try:
        with observability.activate():
            result = EnterpriseRetriever(
                Authorization(),
                Vectors(),
                hybrid_enabled=True,
            reranker=Reranker(lambda item: item),
            candidate_k=5,
            context_k=2,
            context_token_budget=100,
            consumer="durable",
            ).search(secret, auth())
        payload = observability.metrics_payload().decode()
        spans = exporter.get_finished_spans()
        rendered_spans = repr(spans)
    finally:
        observability.shutdown()

    names = {span.name for span in spans}
    assert {
        "retrieval.search",
        "retrieval.authorization",
        "retrieval.dense",
        "retrieval.sparse",
        "retrieval.fuse",
        "retrieval.rerank",
        "retrieval.revalidate",
        "retrieval.pack",
    } <= names
    assert 'stage="dense"' in payload
    assert 'stage="fused"' in payload
    assert 'stage="revalidated"' in payload
    assert 'stage="packed"' in payload
    assert 'consumer="durable"' in payload
    assert all(
        span.attributes["retrieval_consumer"] == "durable"
        for span in spans
        if span.name.startswith("retrieval.")
    )
    assert secret not in payload
    assert secret not in rendered_spans


def test_vector_tenant_and_text_hash_contracts_fail_closed() -> None:
    with pytest.raises(RetrievalSecurityError, match="acl_filter_contract"):
        EnterpriseRetriever(Authorization(), Vectors(replace(chunk(), tenant_id="tenant-b"))).search(
            "差旅", auth()
        )
    with pytest.raises(RetrievalSecurityError, match="content_hash_contract"):
        EnterpriseRetriever(Authorization(), Vectors(replace(chunk(), text="被篡改"))).search(
            "差旅", auth()
        )


def test_security_boundary_metrics_use_only_fixed_kinds() -> None:
    observability, _exporter = _telemetry()
    try:
        with observability.activate():
            with pytest.raises(RetrievalSecurityError):
                EnterpriseRetriever(
                    Authorization(), Vectors(replace(chunk(), tenant_id="tenant-b"))
                ).search("差旅", auth())
            with pytest.raises(RetrievalSecurityError):
                EnterpriseRetriever(
                    Authorization(), Vectors(replace(chunk(), text="被篡改"))
                ).search("差旅", auth())
        payload = observability.metrics_payload().decode()
    finally:
        observability.shutdown()

    assert 'kind="acl"' in payload
    assert 'kind="content_hash"' in payload


def test_reranker_cannot_change_binding_set_or_emit_nan() -> None:
    with pytest.raises(RetrievalSecurityError, match="binding_changed"):
        EnterpriseRetriever(
            Authorization(),
            Vectors(),
            reranker=Reranker(lambda item: replace(item, source_locator="forged")),
        ).search("差旅", auth())
    with pytest.raises(RetrievalSecurityError, match="score_invalid"):
        EnterpriseRetriever(
            Authorization(),
            Vectors(),
            reranker=Reranker(lambda item: replace(item, rerank_score=float("nan"))),
        ).search("差旅", auth())


def test_sql_exact_binding_rejection_removes_candidate_before_context() -> None:
    result = EnterpriseRetriever(Authorization(exact=False), Vectors()).search("差旅", auth())
    assert result.candidates == ()
    assert result.sufficient is False
