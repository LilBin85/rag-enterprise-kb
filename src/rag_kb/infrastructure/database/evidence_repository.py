"""证据账本：只持久化授权来源映射和哈希，不复制正文。"""

import hashlib
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from rag_kb.domain.evidence import EvidenceRecord
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
)
from rag_kb.models import Citation


class EvidenceLedgerConflict(RuntimeError):
    pass


class EvidenceLedgerRepository:
    def __init__(self, acl_repository: ACLRepository | None = None) -> None:
        self._acl = acl_repository or ACLRepository()

    def record_candidates(
        self,
        session: Session,
        *,
        task_id: str,
        query_fingerprint: str,
        auth: AuthContext,
        candidates: tuple[RetrievalCandidate, ...],
    ) -> tuple[EvidenceRecord, ...]:
        task = session.scalar(
            select(AgentTaskRow).where(
                AgentTaskRow.tenant_id == auth.tenant_id,
                AgentTaskRow.task_id == task_id,
                AgentTaskRow.user_id == auth.user_id,
            )
        )
        if task is None:
            raise EvidenceLedgerConflict("evidence_task_binding_invalid")
        now = datetime.now(UTC)
        records: list[EvidenceRecord] = []
        for candidate in candidates:
            if candidate.tenant_id != auth.tenant_id:
                raise EvidenceLedgerConflict("cross_tenant_evidence_denied")
            candidate_hash = "sha256:" + hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
            if candidate_hash != candidate.content_hash:
                raise EvidenceLedgerConflict("evidence_content_hash_mismatch")
            version = session.scalar(
                select(DocumentVersionRow)
                .join(
                    DocumentRow,
                    and_(
                        DocumentRow.tenant_id == DocumentVersionRow.tenant_id,
                        DocumentRow.document_id == DocumentVersionRow.document_id,
                    ),
                )
                .join(
                    IngestionJobRow,
                    and_(
                        IngestionJobRow.tenant_id == DocumentVersionRow.tenant_id,
                        IngestionJobRow.version_id == DocumentVersionRow.version_id,
                    ),
                )
                .where(
                    DocumentVersionRow.tenant_id == auth.tenant_id,
                    DocumentVersionRow.version_id == candidate.version_id,
                    DocumentVersionRow.document_id == candidate.document_id,
                    DocumentVersionRow.version == candidate.document_version,
                    DocumentVersionRow.lifecycle_status == "active",
                    DocumentVersionRow.applied_projection_generation
                    == candidate.projection_generation,
                    DocumentRow.applied_projection_version_id == candidate.version_id,
                    DocumentRow.applied_projection_generation
                    == candidate.projection_generation,
                    IngestionJobRow.status == "succeeded",
                    IngestionJobRow.projection_generation == candidate.projection_generation,
                )
            )
            metadata = session.scalar(
                select(ChunkMetadataRow).where(
                    ChunkMetadataRow.tenant_id == auth.tenant_id,
                    ChunkMetadataRow.version_id == candidate.version_id,
                    ChunkMetadataRow.document_id == candidate.document_id,
                    ChunkMetadataRow.document_version == candidate.document_version,
                    ChunkMetadataRow.chunk_id == candidate.chunk_id,
                    ChunkMetadataRow.content_hash == candidate.content_hash,
                    ChunkMetadataRow.source_locator == candidate.source_locator,
                    ChunkMetadataRow.source_file_name == candidate.source_file_name,
                )
            )
            if (
                version is None
                or metadata is None
                or not self._acl.document_is_allowed(
                    session,
                    auth=auth,
                    document_id=candidate.document_id,
                    permission=Permission.KNOWLEDGE_READ,
                )
            ):
                raise EvidenceLedgerConflict("evidence_authorization_stale")
            evidence_id = uuid5(
                NAMESPACE_URL,
                "|".join(
                    (
                        auth.tenant_id,
                        task_id,
                        query_fingerprint,
                        candidate.version_id,
                        candidate.chunk_id,
                        candidate.content_hash,
                    )
                ),
            ).hex
            values = {
                "evidence_id": evidence_id,
                "task_id": task_id,
                "query_fingerprint": query_fingerprint,
                "tenant_id": auth.tenant_id,
                "document_id": candidate.document_id,
                "version_id": candidate.version_id,
                "document_version": candidate.document_version,
                "chunk_id": candidate.chunk_id,
                "content_hash": candidate.content_hash,
                "source_locator": candidate.source_locator,
                "permission_granted": True,
                "supported_claim_ids": [],
                "retrieved_at": now,
            }
            dialect = session.get_bind().dialect.name
            if dialect == "postgresql":
                statement = postgresql_insert(EvidenceRecordRow).values(**values)
            elif dialect == "sqlite":
                statement = sqlite_insert(EvidenceRecordRow).values(**values)
            else:
                raise RuntimeError("unsupported_database_dialect")
            session.execute(
                statement.on_conflict_do_nothing(
                    index_elements=["evidence_id"],
                )
            )
            persisted = session.get(EvidenceRecordRow, evidence_id)
            if persisted is None:
                raise EvidenceLedgerConflict("evidence_insert_not_visible")
            immutable_binding = {
                "task_id": task_id,
                "query_fingerprint": query_fingerprint,
                "tenant_id": auth.tenant_id,
                "document_id": candidate.document_id,
                "version_id": candidate.version_id,
                "document_version": candidate.document_version,
                "chunk_id": candidate.chunk_id,
                "content_hash": candidate.content_hash,
                "source_locator": candidate.source_locator,
                "permission_granted": True,
                "supported_claim_ids": [],
            }
            if any(
                getattr(persisted, field) != expected
                for field, expected in immutable_binding.items()
            ):
                raise EvidenceLedgerConflict("evidence_immutable_binding_conflict")
            retrieved_at = persisted.retrieved_at
            if retrieved_at.tzinfo is None:
                retrieved_at = retrieved_at.replace(tzinfo=UTC)
            records.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    task_id=task_id,
                    query_fingerprint=query_fingerprint,
                    tenant_id=auth.tenant_id,
                    document_id=candidate.document_id,
                    version_id=candidate.version_id,
                    document_version=candidate.document_version,
                    chunk_id=candidate.chunk_id,
                    content_hash=candidate.content_hash,
                    source_locator=candidate.source_locator,
                    retrieved_at=retrieved_at,
                    permission_granted=True,
                )
            )
        return tuple(records)

    def citations_from_ledger(
        self,
        session: Session,
        *,
        task_id: str,
        evidence_ids: tuple[str, ...],
        auth: AuthContext,
    ) -> tuple[Citation, ...]:
        if len(evidence_ids) != len(set(evidence_ids)):
            raise EvidenceLedgerConflict("duplicate_evidence_id")
        if not evidence_ids:
            return ()
        task = session.scalar(
            select(AgentTaskRow).where(
                AgentTaskRow.tenant_id == auth.tenant_id,
                AgentTaskRow.task_id == task_id,
                AgentTaskRow.user_id == auth.user_id,
            )
        )
        if task is None:
            raise EvidenceLedgerConflict("evidence_task_binding_invalid")
        rows = session.execute(
            select(EvidenceRecordRow, ChunkMetadataRow)
            .join(
                ChunkMetadataRow,
                and_(
                    ChunkMetadataRow.tenant_id == EvidenceRecordRow.tenant_id,
                    ChunkMetadataRow.version_id == EvidenceRecordRow.version_id,
                    ChunkMetadataRow.document_id == EvidenceRecordRow.document_id,
                    ChunkMetadataRow.document_version == EvidenceRecordRow.document_version,
                    ChunkMetadataRow.chunk_id == EvidenceRecordRow.chunk_id,
                    ChunkMetadataRow.content_hash == EvidenceRecordRow.content_hash,
                    ChunkMetadataRow.source_locator == EvidenceRecordRow.source_locator,
                ),
            )
            .join(
                DocumentVersionRow,
                and_(
                    DocumentVersionRow.tenant_id == EvidenceRecordRow.tenant_id,
                    DocumentVersionRow.document_id == EvidenceRecordRow.document_id,
                    DocumentVersionRow.version == EvidenceRecordRow.document_version,
                    DocumentVersionRow.version_id == EvidenceRecordRow.version_id,
                    DocumentVersionRow.lifecycle_status == "active",
                ),
            )
            .join(
                DocumentRow,
                and_(
                    DocumentRow.tenant_id == DocumentVersionRow.tenant_id,
                    DocumentRow.document_id == DocumentVersionRow.document_id,
                    DocumentRow.applied_projection_version_id == DocumentVersionRow.version_id,
                    DocumentRow.applied_projection_generation
                    == DocumentVersionRow.applied_projection_generation,
                ),
            )
            .join(
                IngestionJobRow,
                and_(
                    IngestionJobRow.tenant_id == DocumentVersionRow.tenant_id,
                    IngestionJobRow.version_id == DocumentVersionRow.version_id,
                    IngestionJobRow.status == "succeeded",
                    IngestionJobRow.projection_generation
                    == DocumentVersionRow.applied_projection_generation,
                ),
            )
            .where(
                EvidenceRecordRow.tenant_id == auth.tenant_id,
                EvidenceRecordRow.task_id == task_id,
                EvidenceRecordRow.evidence_id.in_(evidence_ids),
                EvidenceRecordRow.permission_granted.is_(True),
            )
        ).all()
        by_id = {evidence.evidence_id: (evidence, metadata) for evidence, metadata in rows}
        if set(by_id) != set(evidence_ids):
            raise EvidenceLedgerConflict("evidence_ledger_binding_invalid")
        citations: list[Citation] = []
        for evidence_id in evidence_ids:
            evidence, metadata = by_id[evidence_id]
            if not self._acl.document_is_allowed(
                session,
                auth=auth,
                document_id=evidence.document_id,
                permission=Permission.KNOWLEDGE_READ,
            ):
                raise EvidenceLedgerConflict("evidence_authorization_stale")
            citations.append(
                Citation(
                    document_id=evidence.document_id,
                    document_version=evidence.document_version,
                    source_file_name=metadata.source_file_name,
                    source_locator=metadata.source_locator,
                    chunk_id=evidence.chunk_id,
                )
            )
        return tuple(citations)
