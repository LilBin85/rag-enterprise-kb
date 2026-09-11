import hashlib
import json
from types import SimpleNamespace

from rag_kb.agent.tools import render_tool_message
from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.ports.retrieval import AuthorizedVersionSet, RankedChunk
from rag_kb.application.query_service import KnowledgeQueryService
from rag_kb.application.tool_execution_service import ToolExecutionResult
from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.infrastructure.tools.read_tools import DatabaseReadToolAdapter

from test_stage3_acl_evidence import auth, fixture_db


class _Authorization:
    def __init__(self, exact: frozenset[tuple[str, str]]) -> None:
        self.exact = exact
        self.exact_calls: list[tuple[RankedChunk, ...]] = []

    def authorized_active_versions(self, identity):
        return AuthorizedVersionSet(
            identity.tenant_id,
            frozenset({"version-a1"}),
            frozenset({("version-a1", 1)}),
        )

    def revalidate_versions(self, _identity, version_ids):
        return version_ids & frozenset({"version-a1"})

    def revalidate_candidates(self, _identity, candidates):
        self.exact_calls.append(tuple(candidates))
        return self.exact


class _Vectors:
    def __init__(self, chunks: tuple[RankedChunk, ...]) -> None:
        self.chunks = chunks
        self.dense_calls: list[dict[str, object]] = []

    def search_dense(self, _query, **kwargs):
        self.dense_calls.append(kwargs)
        return self.chunks

    def search_sparse(self, _query, **_kwargs):
        raise AssertionError("hybrid retrieval is disabled")


class _RecordingRetriever:
    def __init__(self, retriever: EnterpriseRetriever) -> None:
        self.retriever = retriever
        self.results = []

    def search(self, query, identity):
        result = self.retriever.search(query, identity)
        self.results.append(result)
        return result


class _FastLifecycle:
    def __init__(self) -> None:
        self.starts = []

    def start_fenced_query_with_evidence(self, **kwargs):
        self.starts.append(kwargs)
        return tuple(
            SimpleNamespace(evidence_id=f"evidence-{candidate.chunk_id}")
            for candidate in kwargs["candidates"]
        )

    def finish_fenced_query_with_citations(self, **_kwargs):
        return ()

    def close_fenced_query(self, **_kwargs):
        return True


class _Model:
    def __init__(self) -> None:
        self.calls = []

    def answer(self, _question, chunks):
        self.calls.append(chunks)
        return SimpleNamespace(
            verdict="answer",
            answer="受支持的回答",
            refusal_kind=None,
            used_chunk_ids=[chunks[0].chunk_id],
        )


class _Evidence:
    def __init__(self) -> None:
        self.calls = []

    def record_candidates(self, _session, **kwargs):
        self.calls.append(kwargs)
        return tuple(
            SimpleNamespace(evidence_id=f"evidence-{candidate.chunk_id}")
            for candidate in kwargs["candidates"]
        )


def _chunk(chunk_id: str) -> RankedChunk:
    text = f"{chunk_id} 的授权差旅政策"
    return RankedChunk(
        tenant_id="tenant-a",
        document_id="policy",
        version_id="version-a1",
        document_version=1,
        projection_generation=1,
        chunk_id=chunk_id,
        content_hash="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        source_locator=f"policy/{chunk_id}",
        source_file_name="policy.md",
        text=text,
        dense_score=1.0,
    )


def _retriever(authorization, vectors) -> _RecordingRetriever:
    return _RecordingRetriever(
        EnterpriseRetriever(
            authorization,
            vectors,
            candidate_k=3,
            context_k=3,
            context_token_budget=100,
            token_estimator=lambda _text: 1,
        )
    )


def _context():
    return ToolExecutionContext(
        auth=auth("user-a"),
        task_id="task-a",
        tool_call_id="durable-call",
        plan_revision=None,
        plan_digest=None,
        evidence_digest=None,
    )


def test_fast_and_durable_share_facts_and_durable_limit_is_a_prefix(fixture_db) -> None:
    authorization = _Authorization(
        frozenset({("version-a1", "chunk-3"), ("version-a1", "chunk-1")})
    )
    vectors = _Vectors((_chunk("chunk-3"), _chunk("chunk-1"), _chunk("chunk-2")))
    retriever = _retriever(authorization, vectors)
    lifecycle = _FastLifecycle()
    model = _Model()
    evidence = _Evidence()

    fast = KnowledgeQueryService(lifecycle, retriever, model).query("差旅政策", auth("user-a"))
    durable = DatabaseReadToolAdapter(
        fixture_db,
        operation="search_knowledge",
        retriever=retriever,
        evidence=evidence,
    ).execute({"query": "差旅政策", "limit": 1}, _context())

    fast_fact, durable_fact = retriever.results
    assert fast.verdict == "answer"
    assert fast_fact.query_fingerprint == durable_fact.query_fingerprint
    assert fast_fact.insufficiency_code == durable_fact.insufficiency_code is None
    assert [item.chunk_id for item in fast_fact.candidates] == ["chunk-3", "chunk-1"]
    assert [item.chunk_id for item in durable_fact.candidates] == ["chunk-3", "chunk-1"]
    assert [call["limit"] for call in vectors.dense_calls] == [3, 3]
    assert [item["chunk_id"] for item in durable["items"]] == ["chunk-3"]
    assert [item.chunk_id for item in evidence.calls[0]["candidates"]] == ["chunk-3"]
    assert [
        tuple(item.chunk_id for item in candidates)
        for candidates in authorization.exact_calls
    ] == [("chunk-3", "chunk-1", "chunk-2")] * 2


def test_fast_refusal_and_durable_tool_message_keep_the_same_empty_fact(fixture_db) -> None:
    authorization = _Authorization(frozenset())
    vectors = _Vectors((_chunk("chunk-1"),))
    retriever = _retriever(authorization, vectors)
    lifecycle = _FastLifecycle()
    model = _Model()
    evidence = _Evidence()

    fast = KnowledgeQueryService(lifecycle, retriever, model).query("差旅政策", auth("user-a"))
    durable = DatabaseReadToolAdapter(
        fixture_db,
        operation="search_knowledge",
        retriever=retriever,
        evidence=evidence,
    ).execute({"query": "差旅政策", "limit": 1}, _context())
    message = json.loads(
        render_tool_message(
            "search_knowledge",
            ToolExecutionResult(
                tool_call_id="durable-call",
                status="succeeded",
                result=durable,
            ),
        )
    )

    fast_fact, durable_fact = retriever.results
    assert fast.verdict == "refusal"
    assert fast.refusal is not None and fast.refusal.kind == "insufficient_evidence"
    assert model.calls == [] and lifecycle.starts == []
    assert fast_fact.query_fingerprint == durable_fact.query_fingerprint
    assert fast_fact.insufficiency_code == durable_fact.insufficiency_code == "no_candidates"
    assert fast_fact.candidates == durable_fact.candidates == ()
    assert evidence.calls[0]["candidates"] == ()
    assert message["status"] == "succeeded"
    assert message["error_code"] is None
    assert message["evidence_ids"] == []
    assert message["content"] == []
