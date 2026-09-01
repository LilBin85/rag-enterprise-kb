"""文档注册、读取、删除与上传恢复的稳定应用契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ContextManager, Protocol

from rag_kb.domain.identity import AuthContext


class DocumentAccessDenied(PermissionError):
    pass


class DocumentRegistryConflict(RuntimeError):
    """文档注册冲突，同时保留可安全返回给调用方的恢复引用。"""

    def __init__(self, message: str, *, recovery: RegisteredUpload | None = None) -> None:
        super().__init__(message)
        self.code = message
        self.recovery = recovery


class IdempotencyConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredUpload:
    tenant_id: str
    document_id: str
    version_id: str
    version: int
    job_id: str
    source_object_key: str
    lifecycle_status: str
    ingestion_status: str
    outcome: str = "created"


@dataclass(frozen=True)
class DocumentSnapshot:
    tenant_id: str
    document_id: str
    title: str
    department: str
    status: str
    versions: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class IngestionJobSnapshot:
    job_id: str
    version_id: str
    status: str
    attempts: int
    error_code: str | None


@dataclass(frozen=True)
class PendingObjectReservation:
    tenant_id: str
    user_id: str
    idempotency_key: str
    request_hash: str
    source_object_key: str
    content_hash: str


@dataclass(frozen=True)
class DocumentDeletionReservation:
    tenant_id: str
    document_id: str
    source_object_keys: tuple[str, ...]
    version_count: int
    job_count: int


class DocumentRegistryPort(Protocol):
    def document_projection_guard(
        self,
        tenant_id: str,
        document_id: str,
    ) -> ContextManager[None]: ...

    def reserve_pending_upload(
        self,
        *,
        auth: AuthContext,
        document_id: str,
        title: str,
        department: str,
        source_file_name: str,
        content_hash: str,
        idempotency_key: str,
        request_hash: str,
        expect_existing: bool,
    ) -> tuple[RegisteredUpload, bool]: ...

    def mark_object_ready(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> RegisteredUpload: ...

    def get_ingestion_job(
        self,
        *,
        auth: AuthContext,
        job_id: str,
    ) -> IngestionJobSnapshot | None: ...

    def get_document(
        self,
        *,
        auth: AuthContext,
        document_id: str,
    ) -> DocumentSnapshot | None: ...

    def list_ingestion_jobs(
        self,
        *,
        auth: AuthContext,
        document_id: str,
        limit: int,
    ) -> tuple[IngestionJobSnapshot, ...] | None: ...

    def prepare_document_deletion(
        self,
        *,
        auth: AuthContext,
        document_id: str,
    ) -> DocumentDeletionReservation | None: ...

    def finalize_document_deletion(self, *, tenant_id: str, document_id: str) -> bool: ...


class DocumentUploadRecoveryPort(Protocol):
    def list_pending_object_reservations(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> tuple[PendingObjectReservation, ...]: ...

    def mark_object_ready(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> RegisteredUpload: ...

    def defer_pending_object_reservation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> None: ...
