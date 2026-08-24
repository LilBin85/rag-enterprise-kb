"""文档注册表与摄取状态领域契约。"""

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class DocumentLifecycleStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    DELETED = "deleted"
    FAILED = "failed"


class IngestionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DomainRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(DomainRecord):
    tenant_id: str
    document_id: str
    title: str
    department: str
    created_by: str
    status: str = "active"


class DocumentVersion(DomainRecord):
    version_id: str
    tenant_id: str
    document_id: str
    version: int = Field(ge=1)
    lifecycle_status: DocumentLifecycleStatus
    source_object_key: str
    source_file_name: str
    content_hash: str
    index_version: str
    created_at: AwareDatetime


class ChunkMetadata(DomainRecord):
    chunk_id: str
    version_id: str
    tenant_id: str
    document_id: str
    document_version: int = Field(ge=1)
    content_hash: str
    source_locator: str
    source_file_name: str
    index_version: str


class IngestionJob(DomainRecord):
    job_id: str
    tenant_id: str
    version_id: str
    status: IngestionStatus
    attempts: int = Field(default=0, ge=0)
    available_at: AwareDatetime
    lease_owner: str | None = None
    lease_expires_at: AwareDatetime | None = None
    error_code: str | None = None
