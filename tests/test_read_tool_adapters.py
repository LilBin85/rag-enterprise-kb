"""阶段5四个只读工具的 SQLite 安全契约测试。"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.domain.indexing import compute_chunk_manifest_digest
from rag_kb.domain.retrieval import RetrievalResult
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentTaskRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.tools.read_tools import DatabaseReadToolAdapter

from test_stage3_acl_evidence import auth, candidate, fixture_db


def _context(identity, task_id="task-a") -> ToolExecutionContext:
    return ToolExecutionContext(
        auth=identity,
        task_id=task_id,
        tool_call_id="tool-read-1",
        plan_revision=None,
        plan_digest=None,
        evidence_digest=None,
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


def test_list_document_versions_is_acl_scoped_and_tenant_scoped(fixture_db) -> None:
    adapter = DatabaseReadToolAdapter(fixture_db, operation="list_document_versions")
    arguments = {"document_id": "same-name"}
    owner_result = adapter.execute(arguments, _context(auth("user-a")))
    assert owner_result["items"]
    assert adapter.verify(owner_result, _context(auth("user-a"))) is True
    with pytest.raises(RuntimeError, match="document_read_denied"):
        adapter.execute(arguments, _context(auth("reader-a")))
    with pytest.raises(RuntimeError, match="document_read_denied"):
        adapter.execute(arguments, _context(auth("user-b", tenant_id="tenant-a")))


def _comparison_adapter(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'comparison.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    store = LocalObjectStore(tmp_path / "objects")
    parser = LocalDocumentParser(store, max_chars=400)
    sources = (
        ("version-1", 1, "versions/v1.md", "# 采购制度\n\n## 审批\n经理审批即可。\n".encode("utf-8")),
        (
            "version-2",
            2,
            "versions/v2.md",
            "# 采购制度\n\n## 审批\n经理和财务共同审批。\n\n## 阈值\n超过一千元需要财务复核。\n".encode("utf-8"),
        ),
    )
    parsed = []
    for version_id, version, key, raw in sources:
        store.put_exclusive(key, raw)
        parsed.append(
            parser.parse_version(
                tenant_id="tenant-a",
                document_id="procurement",
                version_id=version_id,
                document_version=version,
                source_object_key=key,
                source_file_name=f"v{version}.md",
                content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
            )
        )
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add_all(
            [
                PrincipalRow(tenant_id="tenant-a", user_id="user-a", roles=[], attributes={}),
                PrincipalRow(tenant_id="tenant-a", user_id="reader-a", roles=[], attributes={}),
            ]
        )
        session.flush()
        session.add(
            DocumentRow(
                tenant_id="tenant-a",
                document_id="procurement",
                title="采购制度",
                department="财务",
                status="active",
                created_by="user-a",
                projection_generation=2,
                applied_projection_generation=2,
                applied_projection_version_id="version-2",
            )
        )
        session.flush()
        version_rows = []
        for (version_id, version, key, raw), document in zip(sources, parsed, strict=True):
            generation = version
            version_rows.append(
                DocumentVersionRow(
                    version_id=version_id,
                    tenant_id="tenant-a",
                    document_id="procurement",
                    version=version,
                    lifecycle_status="superseded" if version == 1 else "active",
                    source_object_key=key,
                    source_file_name=f"v{version}.md",
                    content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
                    index_version="kb-v1",
                    desired_projection_generation=generation,
                    applied_projection_generation=generation,
                )
            )
        session.add_all(version_rows)
        session.flush()
        for (version_id, version, key, raw), document in zip(sources, parsed, strict=True):
            generation = version
            session.add(
                IngestionJobRow(
                    job_id=f"job-{version}",
                    tenant_id="tenant-a",
                    version_id=version_id,
                    status="succeeded",
                    projection_generation=generation,
                    projection_lease_generation=generation,
                    indexed_chunk_count=len(document.chunks),
                    index_manifest_digest=compute_chunk_manifest_digest(document.chunks),
                    index_completed_at=datetime.now(UTC),
                )
            )
            session.add_all(
                ChunkMetadataRow(
                    chunk_id=chunk.chunk_id,
                    version_id=version_id,
                    tenant_id="tenant-a",
                    document_id="procurement",
                    document_version=version,
                    content_hash=chunk.content_hash,
                    source_locator=chunk.source_locator,
                    source_file_name=f"v{version}.md",
                    index_version="kb-v1",
                )
                for chunk in document.chunks
            )
        session.add(
            ACLGrantRow(
                grant_id="grant-owner",
                tenant_id="tenant-a",
                resource_type="document",
                resource_id="procurement",
                subject_type="user",
                subject_id="user-a",
                permissions=["knowledge:read", "document:read"],
            )
        )
        session.add(
            AgentTaskRow(
                task_id="task-a",
                tenant_id="tenant-a",
                user_id="user-a",
                task_mode="research",
                status="running",
                goal="比较采购制度",
                auth_context_ref="auth-a",
            )
        )
    return database, DatabaseReadToolAdapter(
        database, operation="compare_document_versions", source_parser=parser
    )


def test_compare_versions_replays_authorized_sources_and_records_historical_evidence(tmp_path: Path) -> None:
    database, adapter = _comparison_adapter(tmp_path)
    context = _context(auth("user-a"))
    result = adapter.execute(
        {"document_id": "procurement", "left_version": 1, "right_version": 2}, context
    )

    assert result["evidence_ids"]
    assert adapter.verify(result, context) is True
    changes = result["items"][0]["changes"]
    rendered = repr(changes)
    assert "经理审批即可" in rendered
    assert "财务共同审批" in rendered
    assert "versions/v1.md" not in rendered
    with database.sessions() as session:
        scopes = set(session.scalars(select(EvidenceRecordRow.evidence_scope)))
        assert scopes == {"version_comparison"}

    with pytest.raises(RuntimeError, match="document_read_denied"):
        adapter.execute(
            {"document_id": "procurement", "left_version": 1, "right_version": 2},
            _context(auth("reader-a")),
        )
    with database.sessions.begin() as session:
        session.query(ACLGrantRow).filter_by(grant_id="grant-owner").delete()
    assert adapter.verify(result, context) is False
