"""从 PostgreSQL 业务事实生成 Qdrant 文档投影元数据。"""

from sqlalchemy import and_, select

from rag_kb.application.ports.indexing import (
    ClaimedIngestionJob,
    DocumentIndexError,
    DocumentIndexMetadata,
    DocumentIndexMetadataPort,
)
from rag_kb.domain.identity import Permission
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.models import (
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class DatabaseDocumentIndexMetadataPort(DocumentIndexMetadataPort):
    """按任务、版本、文档和租户的完整绑定读取投影元数据。"""

    def __init__(self, database: Database, acl_repository: ACLRepository | None = None) -> None:
        self._database = database
        self._acl = acl_repository or ACLRepository()

    def get_metadata(self, job: ClaimedIngestionJob) -> DocumentIndexMetadata:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.execute(
                select(DocumentRow.title, DocumentRow.department)
                .join(
                    DocumentVersionRow,
                    and_(
                        DocumentVersionRow.tenant_id == DocumentRow.tenant_id,
                        DocumentVersionRow.document_id == DocumentRow.document_id,
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
                    DocumentRow.tenant_id == job.tenant_id,
                    DocumentRow.document_id == job.document_id,
                    DocumentVersionRow.version_id == job.version_id,
                    DocumentVersionRow.version == job.version,
                    DocumentVersionRow.source_object_key == job.source_object_key,
                    DocumentVersionRow.content_hash == job.content_hash,
                    DocumentVersionRow.index_version == job.index_version,
                    IngestionJobRow.job_id == job.job_id,
                )
            ).one_or_none()
            if row is None:
                raise DocumentIndexError("document_index_metadata_binding_invalid")
            principals = self._acl.allowed_principals_for_document(
                work.session,
                tenant_id=job.tenant_id,
                document_id=job.document_id,
                permission=Permission.KNOWLEDGE_READ,
            )
            if not principals:
                raise DocumentIndexError("document_index_acl_empty")
            return DocumentIndexMetadata(
                title=row.title,
                department=row.department,
                allowed_principals=tuple(sorted(principals)),
            )
