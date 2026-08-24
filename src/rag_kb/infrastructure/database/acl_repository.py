"""以 PostgreSQL 业务事实为准的文档 ACL 与当前版本查询。"""

from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


_SUBJECT_TYPES = frozenset({"user", "group", "role"})


def _subject_pairs(subjects: frozenset[str]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for subject in sorted(subjects):
        subject_type, separator, subject_id = subject.partition(":")
        if separator and subject_type in _SUBJECT_TYPES and subject_id:
            pairs.append((subject_type, subject_id))
    if not pairs:
        raise PermissionError("identity_has_no_acl_subjects")
    return tuple(pairs)


def _grants_permission(raw_permissions: object, permission: Permission) -> bool:
    if not isinstance(raw_permissions, (list, tuple, set, frozenset)):
        return False
    values = {str(value) for value in raw_permissions}
    return str(permission) in values or str(Permission.ADMIN) in values


@dataclass(frozen=True)
class AuthorizedVersionSet:
    tenant_id: str
    version_ids: frozenset[str]
    projection_generations: frozenset[tuple[str, int]] = frozenset()


class ACLRepository:
    def allowed_principals_for_document(
        self,
        session: Session,
        *,
        tenant_id: str,
        document_id: str,
        permission: Permission = Permission.KNOWLEDGE_READ,
    ) -> frozenset[str]:
        grants = session.scalars(
            select(ACLGrantRow).where(
                ACLGrantRow.tenant_id == tenant_id,
                ACLGrantRow.resource_type == "document",
                ACLGrantRow.resource_id == document_id,
            )
        )
        return frozenset(
            f"{grant.subject_type}:{grant.subject_id}"
            for grant in grants
            if grant.subject_type in _SUBJECT_TYPES
            and _grants_permission(grant.permissions, permission)
        )

    def authorized_active_version_ids(
        self,
        session: Session,
        *,
        auth: AuthContext,
        permission: Permission = Permission.KNOWLEDGE_READ,
        max_versions: int = 1000,
    ) -> AuthorizedVersionSet:
        if max_versions <= 0 or max_versions > 5000:
            raise ValueError("invalid_authorized_version_limit")
        if auth.has_permission(Permission.ADMIN):
            rows = tuple(
                session.execute(
                    select(
                        DocumentVersionRow.version_id,
                        DocumentVersionRow.applied_projection_generation,
                    )
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
                        DocumentVersionRow.lifecycle_status == "active",
                        DocumentVersionRow.applied_projection_generation.is_not(None),
                        DocumentRow.applied_projection_version_id == DocumentVersionRow.version_id,
                        DocumentRow.applied_projection_generation
                        == DocumentVersionRow.applied_projection_generation,
                        IngestionJobRow.status == "succeeded",
                        IngestionJobRow.projection_generation
                        == DocumentVersionRow.applied_projection_generation,
                        IngestionJobRow.indexed_chunk_count.is_not(None),
                        IngestionJobRow.index_manifest_digest.is_not(None),
                        IngestionJobRow.index_completed_at.is_not(None),
                    )
                    .limit(max_versions + 1)
                )
            )
            if len(rows) > max_versions:
                raise RuntimeError("authorized_version_limit_exceeded")
            projections = frozenset(
                (version_id, generation) for version_id, generation in rows
            )
            return AuthorizedVersionSet(
                auth.tenant_id,
                frozenset(version_id for version_id, _generation in projections),
                projections,
            )
        pairs = _subject_pairs(auth.acl_subjects)
        subject_filter = or_(
            *(
                and_(
                    ACLGrantRow.subject_type == subject_type,
                    ACLGrantRow.subject_id == subject_id,
                )
                for subject_type, subject_id in pairs
            )
        )
        rows = session.execute(
            select(
                DocumentVersionRow.version_id,
                DocumentVersionRow.applied_projection_generation,
                ACLGrantRow.permissions,
            )
            .join(
                DocumentRow,
                and_(
                    DocumentRow.tenant_id == DocumentVersionRow.tenant_id,
                    DocumentRow.document_id == DocumentVersionRow.document_id,
                ),
            )
            .join(
                ACLGrantRow,
                and_(
                    ACLGrantRow.tenant_id == DocumentVersionRow.tenant_id,
                    ACLGrantRow.resource_type == "document",
                    ACLGrantRow.resource_id == DocumentVersionRow.document_id,
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
                DocumentVersionRow.lifecycle_status == "active",
                DocumentVersionRow.applied_projection_generation.is_not(None),
                DocumentRow.applied_projection_version_id == DocumentVersionRow.version_id,
                DocumentRow.applied_projection_generation
                == DocumentVersionRow.applied_projection_generation,
                IngestionJobRow.status == "succeeded",
                IngestionJobRow.projection_generation
                == DocumentVersionRow.applied_projection_generation,
                IngestionJobRow.indexed_chunk_count.is_not(None),
                IngestionJobRow.index_manifest_digest.is_not(None),
                IngestionJobRow.index_completed_at.is_not(None),
                subject_filter,
            )
            .limit(max_versions + 1)
        ).all()
        projections = {
            (version_id, generation)
            for version_id, generation, permissions in rows
            if _grants_permission(permissions, permission)
        }
        if len(projections) > max_versions:
            raise RuntimeError("authorized_version_limit_exceeded")
        frozen_projections = frozenset(projections)
        return AuthorizedVersionSet(
            auth.tenant_id,
            frozenset(version_id for version_id, _generation in frozen_projections),
            frozen_projections,
        )

    def document_is_allowed(
        self,
        session: Session,
        *,
        auth: AuthContext,
        document_id: str,
        permission: Permission,
    ) -> bool:
        if auth.has_permission(Permission.ADMIN):
            return True
        allowed = self.allowed_principals_for_document(
            session,
            tenant_id=auth.tenant_id,
            document_id=document_id,
            permission=permission,
        )
        return bool(allowed & auth.acl_subjects)


class DatabaseAuthorizationPort:
    """为检索入口提供每次请求的新 SQL 快照。"""

    def __init__(self, database: Database, repository: ACLRepository | None = None) -> None:
        self._database = database
        self._repository = repository or ACLRepository()

    def authorized_active_versions(self, auth: AuthContext) -> AuthorizedVersionSet:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self._repository.authorized_active_version_ids(work.session, auth=auth)

    def revalidate_versions(
        self,
        auth: AuthContext,
        version_ids: frozenset[str],
    ) -> frozenset[str]:
        current = self.authorized_active_versions(auth).version_ids
        return current & version_ids

    def revalidate_candidates(
        self,
        auth: AuthContext,
        candidates,
    ) -> frozenset[tuple[str, str]]:
        """按 SQL 块元数据和当前 ACL 对 Qdrant 返回绑定做最终精确复核。"""
        if len(candidates) > 100:
            raise ValueError("candidate_revalidation_limit_exceeded")
        accepted: set[tuple[str, str]] = set()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            for candidate in candidates:
                if candidate.tenant_id != auth.tenant_id:
                    continue
                metadata = work.session.scalar(
                    select(ChunkMetadataRow)
                    .join(
                        DocumentVersionRow,
                        and_(
                            DocumentVersionRow.tenant_id == ChunkMetadataRow.tenant_id,
                            DocumentVersionRow.version_id == ChunkMetadataRow.version_id,
                        ),
                    )
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
                        ChunkMetadataRow.tenant_id == auth.tenant_id,
                        ChunkMetadataRow.version_id == candidate.version_id,
                        ChunkMetadataRow.document_id == candidate.document_id,
                        ChunkMetadataRow.document_version == candidate.document_version,
                        ChunkMetadataRow.chunk_id == candidate.chunk_id,
                        ChunkMetadataRow.content_hash == candidate.content_hash,
                        ChunkMetadataRow.source_locator == candidate.source_locator,
                        ChunkMetadataRow.source_file_name == candidate.source_file_name,
                        DocumentVersionRow.lifecycle_status == "active",
                        DocumentVersionRow.applied_projection_generation
                        == candidate.projection_generation,
                        DocumentRow.applied_projection_version_id == candidate.version_id,
                        DocumentRow.applied_projection_generation
                        == candidate.projection_generation,
                        IngestionJobRow.status == "succeeded",
                        IngestionJobRow.projection_generation
                        == candidate.projection_generation,
                        IngestionJobRow.indexed_chunk_count.is_not(None),
                        IngestionJobRow.index_manifest_digest.is_not(None),
                        IngestionJobRow.index_completed_at.is_not(None),
                    )
                )
                if metadata is not None and self._repository.document_is_allowed(
                    work.session,
                    auth=auth,
                    document_id=candidate.document_id,
                    permission=Permission.KNOWLEDGE_READ,
                ):
                    accepted.add((candidate.version_id, candidate.chunk_id))
        return frozenset(accepted)
