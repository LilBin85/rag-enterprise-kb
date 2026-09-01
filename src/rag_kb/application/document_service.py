"""文档注册与异步摄取用例。"""

import hashlib
import hmac
import json
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rag_kb.application.ports.documents import (
    DocumentAccessDenied,
    DocumentRegistryPort,
    RegisteredUpload,
)
from rag_kb.application.ports.indexing import DocumentVectorDeletionPort
from rag_kb.application.ports.object_store import ObjectConflictError, ObjectStorePort
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.ingestion import validate_upload


class DocumentDeletionUnavailable(RuntimeError):
    pass


class DocumentDeletionFailed(RuntimeError):
    pass


class DocumentUploadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    version_id: str
    version: int
    ingestion_job_id: str
    lifecycle_status: str
    ingestion_status: str
    outcome: str = "created"


class IngestionJobResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    version_id: str
    status: str
    attempts: int
    error_code: str | None = None


class DocumentDeletionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    deleted_versions: int
    deleted_ingestion_jobs: int


class DocumentVersionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str
    version: int
    lifecycle_status: str


class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    department: str
    status: str
    versions: tuple[DocumentVersionResult, ...]


class DocumentService:
    def __init__(
        self,
        registry: DocumentRegistryPort,
        object_store: ObjectStorePort,
        *,
        vector_deleter: DocumentVectorDeletionPort | None = None,
    ) -> None:
        self._registry = registry
        self._object_store = object_store
        self._vector_deleter = vector_deleter

    def register_upload(
        self,
        *,
        document_id: str,
        title: str,
        department: str,
        source_file_name: str,
        content: bytes,
        idempotency_key: str,
        auth: AuthContext,
        expect_existing: bool = False,
    ) -> DocumentUploadResult:
        if not auth.has_permission(Permission.DOCUMENT_WRITE):
            raise DocumentAccessDenied("document_write_denied")
        self._validate_identifier(document_id, "document_id")
        self._validate_identifier(auth.tenant_id, "tenant_id")
        self._validate_filename(source_file_name)
        self._validate_text(title, "title", max_length=500)
        self._validate_text(department, "department", max_length=200)
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("invalid_idempotency_key")
        validate_upload(source_file_name, len(content))
        content_hash = "sha256:" + hashlib.sha256(content).hexdigest()
        request_hash = self._request_hash(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            document_id=document_id,
            title=title,
            department=department,
            source_file_name=source_file_name,
            content_hash=content_hash,
            expect_existing=expect_existing,
        )
        # 上传和删除共享文档锁，避免删除提交后迟到的上传留下无数据库引用的原文对象。
        with self._registry.document_projection_guard(auth.tenant_id, document_id):
            # 跨数据库和对象存储不伪装成单事务：先提交可恢复预留，再写对象，最后
            # 将任务转为 pending。任一提交结果不确定时都保留对象/预留供同键重试收敛。
            reservation, needs_object_write = self._registry.reserve_pending_upload(
                auth=auth,
                document_id=document_id,
                title=title,
                department=department,
                source_file_name=source_file_name,
                content_hash=content_hash,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                expect_existing=expect_existing,
            )
            if needs_object_write:
                try:
                    self._object_store.put_exclusive(reservation.source_object_key, content)
                except ObjectConflictError:
                    existing_hash = "sha256:" + hashlib.sha256(
                        self._object_store.read(reservation.source_object_key)
                    ).hexdigest()
                    if not hmac.compare_digest(existing_hash, content_hash):
                        raise ObjectConflictError("object_content_mismatch")
                ready_reservation = self._registry.mark_object_ready(
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                )
                if reservation.outcome != "created":
                    ready_reservation = replace(ready_reservation, outcome=reservation.outcome)
                reservation = ready_reservation
        return self._result(reservation)

    def get_ingestion_job(self, job_id: str, auth: AuthContext) -> IngestionJobResult | None:
        if not auth.has_permission(Permission.DOCUMENT_READ):
            raise DocumentAccessDenied("document_read_denied")
        job = self._registry.get_ingestion_job(auth=auth, job_id=job_id)
        if job is None:
            return None
        return IngestionJobResult(
            job_id=job.job_id,
            version_id=job.version_id,
            status=job.status,
            attempts=job.attempts,
            error_code=job.error_code,
        )

    def get_document(self, document_id: str, auth: AuthContext) -> DocumentResult | None:
        if not auth.has_permission(Permission.DOCUMENT_READ):
            raise DocumentAccessDenied("document_read_denied")
        snapshot = self._registry.get_document(auth=auth, document_id=document_id)
        if snapshot is None:
            return None
        return DocumentResult(
            document_id=snapshot.document_id,
            title=snapshot.title,
            department=snapshot.department,
            status=snapshot.status,
            versions=tuple(
                DocumentVersionResult(
                    version_id=version_id,
                    version=version,
                    lifecycle_status=status,
                )
                for version_id, version, status in snapshot.versions
            ),
        )

    def list_ingestion_jobs(
        self,
        document_id: str,
        auth: AuthContext,
        *,
        limit: int = 100,
    ) -> tuple[IngestionJobResult, ...] | None:
        if not auth.has_permission(Permission.DOCUMENT_READ):
            raise DocumentAccessDenied("document_read_denied")
        jobs = self._registry.list_ingestion_jobs(
            auth=auth,
            document_id=document_id,
            limit=limit,
        )
        if jobs is None:
            return None
        return tuple(
            IngestionJobResult(
                job_id=job.job_id,
                version_id=job.version_id,
                status=job.status,
                attempts=job.attempts,
                error_code=job.error_code,
            )
            for job in jobs
        )

    def delete_document(
        self,
        document_id: str,
        auth: AuthContext,
    ) -> DocumentDeletionResult | None:
        if not auth.has_permission(Permission.DOCUMENT_WRITE):
            raise DocumentAccessDenied("document_write_denied")
        if self._vector_deleter is None:
            raise DocumentDeletionUnavailable("document_vector_deleter_unavailable")
        with self._registry.document_projection_guard(auth.tenant_id, document_id):
            reservation = self._registry.prepare_document_deletion(
                auth=auth,
                document_id=document_id,
            )
            if reservation is None:
                return None
            try:
                self._vector_deleter.delete_document(
                    tenant_id=reservation.tenant_id,
                    document_id=reservation.document_id,
                )
            except Exception as exc:
                raise DocumentDeletionFailed("document_vector_cleanup_failed") from exc
            try:
                for source_object_key in reservation.source_object_keys:
                    self._object_store.delete_exact(source_object_key)
            except Exception as exc:
                raise DocumentDeletionFailed("document_object_cleanup_failed") from exc
            finalized = self._registry.finalize_document_deletion(
                tenant_id=reservation.tenant_id,
                document_id=reservation.document_id,
            )
        if not finalized:
            raise DocumentDeletionFailed("document_deletion_finalize_missing")
        return DocumentDeletionResult(
            document_id=reservation.document_id,
            deleted_versions=reservation.version_count,
            deleted_ingestion_jobs=reservation.job_count,
        )

    @staticmethod
    def _result(reservation: RegisteredUpload) -> DocumentUploadResult:
        return DocumentUploadResult(
            document_id=reservation.document_id,
            version_id=reservation.version_id,
            version=reservation.version,
            ingestion_job_id=reservation.job_id,
            lifecycle_status=reservation.lifecycle_status,
            ingestion_status=reservation.ingestion_status,
            outcome=reservation.outcome,
        )

    @staticmethod
    def _validate_identifier(value: str, field: str) -> None:
        normalized = value.replace("\\", "/")
        if (
            not value
            or len(value) > 200
            or Path(normalized).name != normalized
            or value in {".", ".."}
            or value.rstrip(" .") != value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"invalid_{field}")

    @staticmethod
    def _validate_filename(value: str) -> None:
        normalized = value.replace("\\", "/")
        reserved = {"CON", "PRN", "AUX", "NUL"}
        reserved.update(f"COM{index}" for index in range(1, 10))
        reserved.update(f"LPT{index}" for index in range(1, 10))
        stem = value.split(".", 1)[0].upper()
        if (
            not value
            or len(value) > 255
            or Path(normalized).name != normalized
            or value.rstrip(" .") != value
            or any(character in '<>:"|?*' or ord(character) < 32 for character in value)
            or stem in reserved
        ):
            raise ValueError("invalid_source_file_name")

    @staticmethod
    def _validate_text(value: str, field: str, *, max_length: int) -> None:
        if not value.strip() or len(value) > max_length or any(ord(character) == 0 for character in value):
            raise ValueError(f"invalid_{field}")

    @staticmethod
    def _request_hash(**values: object) -> str:
        canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
