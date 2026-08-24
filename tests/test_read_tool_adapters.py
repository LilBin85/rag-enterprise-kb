"""阶段5四个只读工具的 SQLite 安全契约测试。"""

import hashlib

import pytest
from sqlalchemy import select

from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.domain.retrieval import RetrievalResult
from rag_kb.infrastructure.database.models import EvidenceRecordRow
from rag_kb.infrastructure.tools.read_tools import DatabaseReadToolAdapter

from test_stage3_acl_evidence import auth, candidate, fixture_db


def _context(identity, task_id="task-a") -> ToolExecutionContext:
    return ToolExecutionContext(
        auth=identity,
        task_id=task_id,
        tool_call_id="tool-read-1",
        plan_revision=1,
        plan_digest="sha256:plan",
        evidence_digest="sha256:evidence",
    )


class _FakeRetriever:
    def __init__(self, item):
        self.item = item
        self.calls = []

    def search(self, query, identity):
        self.calls.append((query, identity))
        return RetrievalResult(
            query_fingerprint="query-fingerprint",
            candidates=(self.item,),
            context_token_estimate=1,
            sufficient=True,
        )


def test_search_records_ledger_without_persisting正文_and_verify_detects_tamper(fixture_db) -> None:
    identity = auth("user-a")
    item = candidate()
    retriever = _FakeRetriever(item)
    adapter = DatabaseReadToolAdapter(fixture_db, operation="search_knowledge", retriever=retriever)
    result = adapter.execute({"query": "制度", "limit": 5}, _context(identity))
    assert result["evidence_ids"] and result["items"][0]["text"] == item.text
    assert adapter.verify(result, _context(identity)) is True
    tampered = {**result, "items": [{**result["items"][0], "text": "伪造正文"}]}
    assert adapter.verify(tampered, _context(identity)) is False
    with fixture_db.sessions() as session:
        row = session.scalar(select(EvidenceRecordRow))
        assert row is not None and "text" not in EvidenceRecordRow.__table__.columns
        assert row.content_hash == "sha256:" + hashlib.sha256(item.text.encode()).hexdigest()
    assert len(retriever.calls) == 1


def test_fetch_evidence_rechecks_ledger_acl_and_reference(fixture_db) -> None:
    owner = auth("user-a")
    search = DatabaseReadToolAdapter(
        fixture_db, operation="search_knowledge", retriever=_FakeRetriever(candidate())
    )
    result = search.execute({"query": "制度"}, _context(owner))
    evidence_id = result["evidence_ids"][0]
    fetch = DatabaseReadToolAdapter(fixture_db, operation="fetch_evidence")
    fetched = fetch.execute({"evidence_ids": [evidence_id]}, _context(owner))
    assert fetched["evidence_ids"] == [evidence_id]
    assert fetch.verify(fetched, _context(owner)) is True
    with pytest.raises(Exception, match="evidence_task_binding_invalid|evidence_authorization"):
        fetch.execute({"evidence_ids": [evidence_id]}, _context(auth("reader-a")))


@pytest.mark.parametrize("operation", ["list_document_versions", "compare_document_versions"])
def test_version_tools_are_acl_scoped_and_tenant_scoped(fixture_db, operation: str) -> None:
    adapter = DatabaseReadToolAdapter(fixture_db, operation=operation)
    arguments = (
        {"document_id": "same-name"}
        if operation == "list_document_versions"
        else {"document_id": "same-name", "left_version": 1, "right_version": 2}
    )
    owner_result = adapter.execute(arguments, _context(auth("user-a")))
    assert owner_result["items"]
    assert adapter.verify(owner_result, _context(auth("user-a"))) is True
    with pytest.raises(PermissionError, match="document_read_denied"):
        adapter.execute(arguments, _context(auth("reader-a")))
    with pytest.raises(PermissionError, match="document_read_denied"):
        adapter.execute(arguments, _context(auth("user-b", tenant_id="tenant-a")))

