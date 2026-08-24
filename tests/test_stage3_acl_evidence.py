"""Stage 3 SQLite ACL、文档资源权限与证据账本事务测试。"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from rag_kb.application.document_service import DocumentService
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentTaskRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    GroupMembershipRow,
    GroupRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.object_store import LocalObjectStore


def auth(
    user_id: str,
    *,
    tenant_id: str = "tenant-a",
    groups: tuple[str, ...] = (),
    permissions: tuple[str, ...] = ("knowledge:read", "document:read"),
) -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        groups=frozenset(groups),
        attributes={"permissions": list(permissions)},
        request_id=f"request-{user_id}",
    )


def candidate(
    *,
    tenant_id: str = "tenant-a",
    document_id: str = "same-name",
    version_id: str = "version-a1",
    version: int = 1,
    chunk_id: str = "chunk-a1",
    source_locator: str | None = None,
) -> RetrievalCandidate:
    text = "正文不得进入证据账本"
    return RetrievalCandidate(
        tenant_id=tenant_id,
        document_id=document_id,
        version_id=version_id,
        document_version=version,
        projection_generation=1,
        chunk_id=chunk_id,
        content_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_locator=source_locator or {
            "chunk-a1": "policy/a1",
            "chunk-old": "policy/old",
            "chunk-withdrawn": "policy/withdrawn",
            "chunk-b1": "policy/b1",
        }.get(chunk_id, f"policy/{chunk_id}"),
        source_file_name="b.md" if tenant_id == "tenant-b" else "a.md",
        dense_score=1.0,
        sparse_score=None,
        fused_score=1.0,
        text=text,
    )


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Database:
    chunk_hash = "sha256:" + hashlib.sha256(
        "正文不得进入证据账本".encode("utf-8")
    ).hexdigest()
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'stage3.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add_all([TenantRow(tenant_id="tenant-a", name="租户甲"), TenantRow(tenant_id="tenant-b", name="租户乙")])
        session.flush()
        session.add_all(
            [
                PrincipalRow(tenant_id="tenant-a", user_id="user-a", roles=[], attributes={}),
                PrincipalRow(tenant_id="tenant-a", user_id="reader-a", roles=[], attributes={}),
                PrincipalRow(tenant_id="tenant-a", user_id="group-user", roles=[], attributes={}),
                PrincipalRow(tenant_id="tenant-b", user_id="user-b", roles=[], attributes={}),
                GroupRow(tenant_id="tenant-a", group_id="finance", name="财务组"),
            ]
        )
        session.flush()
        session.add(GroupMembershipRow(tenant_id="tenant-a", group_id="finance", user_id="group-user"))
        session.add_all(
            [
                DocumentRow(
                    tenant_id="tenant-a", document_id="same-name", title="制度甲", department="财务",
                    status="active", created_by="user-a",
                    projection_generation=1, applied_projection_generation=1,
                    applied_projection_version_id="version-a1",
                ),
                DocumentRow(
                    tenant_id="tenant-b", document_id="same-name", title="制度乙", department="财务",
                    status="active", created_by="user-b",
                    projection_generation=1, applied_projection_generation=1,
                    applied_projection_version_id="version-b1",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                DocumentVersionRow(
                    version_id="version-a1", tenant_id="tenant-a", document_id="same-name", version=1,
                    lifecycle_status="active", source_object_key="a1", source_file_name="a.md",
                    content_hash="hash-chunk-a1", index_version="kb-v1",
                    desired_projection_generation=1, applied_projection_generation=1,
                ),
                DocumentVersionRow(
                    version_id="version-a-old", tenant_id="tenant-a", document_id="same-name", version=2,
                    lifecycle_status="superseded", source_object_key="a-old", source_file_name="a.md",
                    content_hash="hash-chunk-old", index_version="kb-v1",
                    desired_projection_generation=2, applied_projection_generation=2,
                ),
                DocumentVersionRow(
                    version_id="version-a-withdrawn", tenant_id="tenant-a", document_id="same-name", version=3,
                    lifecycle_status="withdrawn", source_object_key="a-withdrawn", source_file_name="a.md",
                    content_hash="hash-chunk-withdrawn", index_version="kb-v1",
                ),
                DocumentVersionRow(
                    version_id="version-b1", tenant_id="tenant-b", document_id="same-name", version=1,
                    lifecycle_status="active", source_object_key="b1", source_file_name="b.md",
                    content_hash="hash-chunk-b1", index_version="kb-v1",
                    desired_projection_generation=1, applied_projection_generation=1,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                IngestionJobRow(
                    job_id="job-a1", tenant_id="tenant-a", version_id="version-a1",
                    status="succeeded", indexed_chunk_count=1,
                    index_manifest_digest="sha256:manifest-a",
                    index_completed_at=datetime.now(UTC),
                    projection_generation=1, projection_lease_generation=1,
                ),
                IngestionJobRow(
                    job_id="job-b1", tenant_id="tenant-b", version_id="version-b1",
                    status="succeeded", indexed_chunk_count=1,
                    index_manifest_digest="sha256:manifest-b",
                    index_completed_at=datetime.now(UTC),
                    projection_generation=1, projection_lease_generation=1,
                ),
            ]
        )
        session.add_all(
            [
                ACLGrantRow(
                    grant_id="grant-a-owner", tenant_id="tenant-a", resource_type="document", resource_id="same-name",
                    subject_type="user", subject_id="user-a",
                    permissions=["knowledge:read", "document:read", "document:write"],
                ),
                ACLGrantRow(
                    grant_id="grant-a-group", tenant_id="tenant-a", resource_type="document", resource_id="same-name",
                    subject_type="group", subject_id="finance", permissions=["knowledge:read"],
                ),
                ACLGrantRow(
                    grant_id="grant-b-owner", tenant_id="tenant-b", resource_type="document", resource_id="same-name",
                    subject_type="user", subject_id="user-b", permissions=["knowledge:read", "document:read"],
                ),
            ]
        )
        session.add_all(
            [
                ChunkMetadataRow(
                    chunk_id="chunk-a1", version_id="version-a1", tenant_id="tenant-a", document_id="same-name",
                    document_version=1, content_hash=chunk_hash, source_locator="policy/a1",
                    source_file_name="a.md", index_version="kb-v1",
                ),
                ChunkMetadataRow(
                    chunk_id="chunk-old", version_id="version-a-old", tenant_id="tenant-a", document_id="same-name",
                    document_version=2, content_hash=chunk_hash, source_locator="policy/old",
                    source_file_name="a.md", index_version="kb-v1",
                ),
                ChunkMetadataRow(
                    chunk_id="chunk-withdrawn", version_id="version-a-withdrawn", tenant_id="tenant-a", document_id="same-name",
                    document_version=3, content_hash=chunk_hash, source_locator="policy/withdrawn",
                    source_file_name="a.md", index_version="kb-v1",
                ),
                ChunkMetadataRow(
                    chunk_id="chunk-b1", version_id="version-b1", tenant_id="tenant-b", document_id="same-name",
                    document_version=1, content_hash=chunk_hash, source_locator="policy/b1",
                    source_file_name="b.md", index_version="kb-v1",
                ),
            ]
        )
        session.add_all(
            [
                AgentTaskRow(
                    task_id="task-a", tenant_id="tenant-a", user_id="user-a", task_mode="knowledge_query",
                    status="running", goal="查询制度", auth_context_ref="auth-a",
                ),
                AgentTaskRow(
                    task_id="task-b", tenant_id="tenant-b", user_id="user-b", task_mode="knowledge_query",
                    status="running", goal="查询制度", auth_context_ref="auth-b",
                ),
            ]
        )
    return database


def test_creator_default_user_grant_and_resource_acl(fixture_db: Database, tmp_path: Path) -> None:
    service = DocumentService(fixture_db, LocalObjectStore(tmp_path / "objects"))
    owner = auth("user-a", permissions=("document:write", "document:read", "knowledge:read"))
    result = service.register_upload(
        document_id="created-by-user",
        title="新制度",
        department="财务",
        source_file_name="new.md",
        content=b"# new",
        idempotency_key="creator-grant",
        auth=owner,
    )
    with fixture_db.sessions() as session:
        grants = ACLRepository().allowed_principals_for_document(
            session, tenant_id="tenant-a", document_id="created-by-user", permission=Permission.DOCUMENT_READ
        )
    assert "user:user-a" in grants
    assert service.get_document("same-name", owner) is not None
    assert service.get_document("same-name", auth("reader-a")) is None
    assert service.get_ingestion_job("job-a1", auth("reader-a")) is None
    assert service.get_document("same-name", auth("user-a", tenant_id="tenant-b")) is None


def test_group_grant_cross_tenant_and_lifecycle_filter(fixture_db: Database) -> None:
    repository = ACLRepository()
    with fixture_db.sessions() as session:
        group_user = repository.authorized_active_version_ids(
            session, auth=auth("group-user", groups=("finance",))
        )
        other_tenant = repository.authorized_active_version_ids(
            session, auth=auth("user-b", tenant_id="tenant-b")
        )
        assert group_user.version_ids == frozenset({"version-a1"})
        assert other_tenant.version_ids == frozenset({"version-b1"})
        assert "version-a-old" not in group_user.version_ids
        assert "version-a-withdrawn" not in group_user.version_ids


def test_evidence_ledger_is_idempotent_and_never_stores正文(fixture_db: Database) -> None:
    repository = EvidenceLedgerRepository()
    item = candidate()
    with fixture_db.sessions.begin() as session:
        first = repository.record_candidates(
            session, task_id="task-a", query_fingerprint="query-a", auth=auth("user-a"), candidates=(item,)
        )
        second = repository.record_candidates(
            session, task_id="task-a", query_fingerprint="query-a", auth=auth("user-a"), candidates=(item,)
        )
        assert first[0].evidence_id == second[0].evidence_id
        assert first[0].retrieved_at == second[0].retrieved_at
        assert first[0].version_id == "version-a1"
        assert session.scalar(select(func.count()).select_from(EvidenceRecordRow)) == 1
        assert "text" not in EvidenceRecordRow.__table__.columns
        citations = repository.citations_from_ledger(
            session, task_id="task-a", evidence_ids=(first[0].evidence_id,), auth=auth("user-a")
        )
        assert citations[0].chunk_id == "chunk-a1"


@pytest.mark.parametrize(
    "bad_candidate",
    [
        candidate(tenant_id="tenant-b", version_id="version-b1", chunk_id="chunk-b1"),
        candidate(version_id="version-a-old", version=2, chunk_id="chunk-old"),
        candidate(version_id="version-a-withdrawn", version=3, chunk_id="chunk-withdrawn"),
    ],
)
def test_evidence_record_rejects_cross_tenant_or_stale_candidates(fixture_db: Database, bad_candidate: RetrievalCandidate) -> None:
    with fixture_db.sessions.begin() as session:
        with pytest.raises(EvidenceLedgerConflict):
            EvidenceLedgerRepository().record_candidates(
                session,
                task_id="task-a",
                query_fingerprint="query-bad",
                auth=auth("user-a"),
                candidates=(bad_candidate,),
            )


def test_evidence_citations_require_same_task_tenant_and_current_acl(fixture_db: Database) -> None:
    repository = EvidenceLedgerRepository()
    with fixture_db.sessions.begin() as session:
        records = repository.record_candidates(
            session, task_id="task-a", query_fingerprint="query-a", auth=auth("user-a"), candidates=(candidate(),)
        )
        evidence_id = records[0].evidence_id
        with pytest.raises(EvidenceLedgerConflict):
            repository.citations_from_ledger(
                session, task_id="task-b", evidence_ids=(evidence_id,), auth=auth("user-a")
            )
        with pytest.raises(EvidenceLedgerConflict, match="task_binding_invalid"):
            repository.citations_from_ledger(
                session,
                task_id="task-a",
                evidence_ids=(evidence_id,),
                auth=auth("group-user", groups=("finance",)),
            )
        session.query(ACLGrantRow).filter_by(grant_id="grant-a-owner").delete()
        with pytest.raises(EvidenceLedgerConflict, match="authorization_stale|binding_invalid"):
            repository.citations_from_ledger(
                session, task_id="task-a", evidence_ids=(evidence_id,), auth=auth("user-a")
            )


def test_evidence_id_conflict_never_rewrites_immutable_binding(fixture_db: Database) -> None:
    repository = EvidenceLedgerRepository()
    with fixture_db.sessions.begin() as session:
        records = repository.record_candidates(
            session,
            task_id="task-a",
            query_fingerprint="query-a",
            auth=auth("user-a"),
            candidates=(candidate(),),
        )
        row = session.get(EvidenceRecordRow, records[0].evidence_id)
        assert row is not None
        row.source_locator = "tampered/locator"
        session.flush()
        with pytest.raises(EvidenceLedgerConflict, match="immutable_binding_conflict"):
            repository.record_candidates(
                session,
                task_id="task-a",
                query_fingerprint="query-a",
                auth=auth("user-a"),
                candidates=(candidate(),),
            )
