"""真实 PostgreSQL Stage 3 ACL 与证据账本组件测试。"""

from __future__ import annotations

import json
import hashlib
import os
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import AgentTaskRow, EvidenceRecordRow


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")
CHUNK_TEXT = "正文仅用于测试输入，不写入账本"
CHUNK_HASH = "sha256:" + hashlib.sha256(CHUNK_TEXT.encode("utf-8")).hexdigest()


def _auth(tenant_id: str, user_id: str, *, groups: tuple[str, ...] = ()) -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        groups=frozenset(groups),
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id=f"request-{user_id}",
    )


def _candidate(tenant_id: str, document_id: str, version_id: str, chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        tenant_id=tenant_id,
        document_id=document_id,
        version_id=version_id,
        document_version=1,
        projection_generation=1,
        chunk_id=chunk_id,
        content_hash=CHUNK_HASH,
        source_locator="section/1",
        source_file_name="policy.md",
        fused_score=1.0,
        text=CHUNK_TEXT,
    )


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="未配置 RAG_KB_TEST_POSTGRES_URL")
def test_postgres_acl_active_versions_and_evidence_revalidation() -> None:
    assert POSTGRES_TEST_URL is not None
    database = build_database(POSTGRES_TEST_URL)
    assert database.engine.dialect.name == "postgresql"
    suffix = uuid4().hex
    tenant_a, tenant_b = f"stage3-a-{suffix}", f"stage3-b-{suffix}"
    user_a, group_user, user_b = f"user-a-{suffix}", f"group-user-{suffix}", f"user-b-{suffix}"
    group_id = f"group-{suffix}"
    document_id = f"same-document-{suffix}"
    version_a, version_old, version_b = f"version-a-{suffix}", f"version-old-{suffix}", f"version-b-{suffix}"
    chunk_a, chunk_old, chunk_b = f"chunk-a-{suffix}", f"chunk-old-{suffix}", f"chunk-b-{suffix}"
    task_id = f"task-{suffix}"
    try:
        with database.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (tenant_id, name, active, created_at) VALUES (:id, :name, TRUE, CURRENT_TIMESTAMP)"),
                [{"id": tenant_a, "name": tenant_a}, {"id": tenant_b, "name": tenant_b}],
            )
            connection.execute(
                text(
                    "INSERT INTO principals (tenant_id, user_id, roles, attributes, active, created_at) "
                    "VALUES (:tenant, :user, CAST('[]' AS jsonb), CAST('{}' AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                [
                    {"tenant": tenant_a, "user": user_a},
                    {"tenant": tenant_a, "user": group_user},
                    {"tenant": tenant_b, "user": user_b},
                ],
            )
            connection.execute(
                text("INSERT INTO groups (tenant_id, group_id, name) VALUES (:tenant, :group, :name)"),
                {"tenant": tenant_a, "group": group_id, "name": group_id},
            )
            connection.execute(
                text("INSERT INTO group_memberships (tenant_id, group_id, user_id) VALUES (:tenant, :group, :user)"),
                {"tenant": tenant_a, "group": group_id, "user": group_user},
            )
            connection.execute(
                text(
                    "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, "
                    "projection_generation, applied_projection_generation, applied_projection_version_id, "
                    "created_at, updated_at) "
                    "VALUES (:tenant, :doc, :title, 'finance', 'active', :user, 1, 1, :version, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                [
                    {"tenant": tenant_a, "doc": document_id, "title": "A", "user": user_a, "version": version_a},
                    {"tenant": tenant_b, "doc": document_id, "title": "B", "user": user_b, "version": version_b},
                ],
            )
            connection.execute(
                text(
                    "INSERT INTO document_versions "
                    "(version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, source_file_name, content_hash, index_version, desired_projection_generation, applied_projection_generation, created_at) "
                    "VALUES (:version, :tenant, :doc, :number, :status, :object, 'policy.md', :hash, 'index-v1', :number, :number, CURRENT_TIMESTAMP)"
                ),
                [
                    {"version": version_a, "tenant": tenant_a, "doc": document_id, "number": 1, "status": "active", "object": f"a/{suffix}", "hash": "sha256:a"},
                    {"version": version_old, "tenant": tenant_a, "doc": document_id, "number": 2, "status": "superseded", "object": f"old/{suffix}", "hash": "sha256:old"},
                    {"version": version_b, "tenant": tenant_b, "doc": document_id, "number": 1, "status": "active", "object": f"b/{suffix}", "hash": "sha256:b"},
                ],
            )
            connection.execute(
                text(
                    "INSERT INTO chunk_metadata "
                    "(chunk_id, version_id, tenant_id, document_id, document_version, content_hash, source_locator, source_file_name, index_version) "
                    "VALUES (:chunk, :version, :tenant, :doc, :number, :hash, 'section/1', 'policy.md', 'index-v1')"
                ),
                [
                    {"chunk": chunk_a, "version": version_a, "tenant": tenant_a, "doc": document_id, "number": 1, "hash": CHUNK_HASH},
                    {"chunk": chunk_old, "version": version_old, "tenant": tenant_a, "doc": document_id, "number": 2, "hash": CHUNK_HASH},
                    {"chunk": chunk_b, "version": version_b, "tenant": tenant_b, "doc": document_id, "number": 1, "hash": CHUNK_HASH},
                ],
            )
            connection.execute(
                text(
                    "INSERT INTO ingestion_jobs (job_id, tenant_id, version_id, status, attempts, available_at, "
                    "lease_generation, projection_generation, projection_lease_generation, created_at, updated_at, "
                    "indexed_chunk_count, index_manifest_digest, index_completed_at) "
                    "VALUES (:job, :tenant, :version, 'succeeded', 1, CURRENT_TIMESTAMP, 1, 1, 1, "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1, 'sha256:manifest', CURRENT_TIMESTAMP)"
                ),
                [
                    {"job": f"job-a-{suffix}", "tenant": tenant_a, "version": version_a},
                    {"job": f"job-b-{suffix}", "tenant": tenant_b, "version": version_b},
                ],
            )
            permissions = json.dumps(["knowledge:read", "document:read"])
            connection.execute(
                text(
                    "INSERT INTO acl_grants (grant_id, tenant_id, resource_type, resource_id, subject_type, subject_id, permissions, created_at) "
                    "VALUES (:grant, :tenant, 'document', :doc, :subject_type, :subject, CAST(:permissions AS jsonb), CURRENT_TIMESTAMP)"
                ),
                [
                    {"grant": f"grant-user-{suffix}", "tenant": tenant_a, "doc": document_id, "subject_type": "user", "subject": user_a, "permissions": permissions},
                    {"grant": f"grant-group-{suffix}", "tenant": tenant_a, "doc": document_id, "subject_type": "group", "subject": group_id, "permissions": permissions},
                    {"grant": f"grant-b-{suffix}", "tenant": tenant_b, "doc": document_id, "subject_type": "user", "subject": user_b, "permissions": permissions},
                ],
            )
        with database.sessions.begin() as session:
            session.add(
                AgentTaskRow(
                    task_id=task_id,
                    tenant_id=tenant_a,
                    user_id=user_a,
                    task_mode="knowledge_query",
                    status="running",
                    goal="component test",
                    auth_context_ref="sha256:test",
                )
            )

        acl = ACLRepository()
        with database.sessions() as session:
            user_versions = acl.authorized_active_version_ids(session, auth=_auth(tenant_a, user_a))
            group_versions = acl.authorized_active_version_ids(session, auth=_auth(tenant_a, group_user, groups=(group_id,)))
            other_versions = acl.authorized_active_version_ids(session, auth=_auth(tenant_b, user_b))
            assert user_versions.version_ids == frozenset({version_a})
            assert group_versions.version_ids == frozenset({version_a})
            assert other_versions.version_ids == frozenset({version_b})
            assert version_old not in user_versions.version_ids and version_b not in user_versions.version_ids

        repository = EvidenceLedgerRepository()
        item = _candidate(tenant_a, document_id, version_a, chunk_a)
        with database.sessions.begin() as session:
            records = repository.record_candidates(
                session, task_id=task_id, query_fingerprint=f"query-{suffix}", auth=_auth(tenant_a, user_a), candidates=(item,)
            )
            row = session.scalar(select(EvidenceRecordRow).where(EvidenceRecordRow.evidence_id == records[0].evidence_id))
            assert row is not None and row.version_id == version_a
            with pytest.raises(EvidenceLedgerConflict):
                repository.record_candidates(
                    session, task_id=task_id, query_fingerprint=f"query-old-{suffix}", auth=_auth(tenant_a, user_a),
                    candidates=(_candidate(tenant_a, document_id, version_old, chunk_old),),
                )
            evidence_id = records[0].evidence_id

        with database.engine.begin() as connection:
            connection.execute(text("DELETE FROM acl_grants WHERE tenant_id = :tenant AND resource_id = :doc"), {"tenant": tenant_a, "doc": document_id})
        with database.sessions() as session:
            with pytest.raises(EvidenceLedgerConflict):
                repository.citations_from_ledger(session, task_id=task_id, evidence_ids=(evidence_id,), auth=_auth(tenant_a, user_a))
    finally:
        with database.engine.begin() as connection:
            connection.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"), {"a": tenant_a, "b": tenant_b})
        database.engine.dispose()
