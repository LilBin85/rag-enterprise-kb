"""文档摄取、索引与投影收敛的稳定应用契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import ContextManager, Protocol

from rag_kb.domain.indexing import IndexReceipt, ParsedDocument


@dataclass(frozen=True)
class ClaimedIngestionJob:
    """传给解析与索引端口的最小任务元数据，不包含原文正文。"""

    tenant_id: str
    job_id: str
    version_id: str
    document_id: str
    version: int
    source_object_key: str
    source_file_name: str
    content_hash: str
    index_version: str
    attempts: int
    lease_owner: str
    lease_generation: int
    projection_generation: int | None = None
    previous_projection_generation: int | None = None
    previous_projection_version_id: str | None = None
    phase: str = "ingest"
    lifecycle_status: str = "indexing"
    index_receipt: IndexReceipt | None = None
    pending_traceparent: str | None = None
    pending_tracestate: str | None = None


@dataclass(frozen=True)
class IngestionQueueSnapshot:
    status: str
    depth: int
    oldest_age_seconds: float


class RetryableIngestionError(RuntimeError):
    """端口可显式标记为可重试的失败；其消息不会被持久化。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DocumentIndexError(RuntimeError):
    """仅传播固定安全错误码，绝不暴露底层索引异常。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DocumentIndexMetadata:
    title: str
    department: str
    allowed_principals: tuple[str, ...]


@dataclass(frozen=True)
class ClaimedProjectionRepair:
    job: ClaimedIngestionJob
    reconcile_lease_generation: int


class DocumentParserPort(Protocol):
    def parse(self, job: ClaimedIngestionJob) -> ParsedDocument: ...


class DocumentIndexPort(Protocol):
    def index(self, job: ClaimedIngestionJob, parsed_document: ParsedDocument) -> IndexReceipt: ...

    def publish(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        lifecycle_status: str,
    ) -> None: ...

    def retire(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> None: ...

    def discard(self, job: ClaimedIngestionJob) -> None: ...

    def discard_unbound(self, job: ClaimedIngestionJob) -> None: ...


class DocumentIndexMetadataPort(Protocol):
    def get_metadata(self, job: ClaimedIngestionJob) -> DocumentIndexMetadata: ...


class DocumentVectorDeletionPort(Protocol):
    def delete_document(self, *, tenant_id: str, document_id: str) -> None: ...


class IngestionJobPort(Protocol):
    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> ClaimedIngestionJob | None: ...

    def persist_manifest(self, job: ClaimedIngestionJob, parsed: ParsedDocument) -> None: ...

    def allocate_projection(self, job: ClaimedIngestionJob) -> ClaimedIngestionJob: ...

    def renew_lease(self, job: ClaimedIngestionJob, *, lease_seconds: int) -> None: ...

    def load_index_receipt(self, job: ClaimedIngestionJob) -> IndexReceipt: ...

    def activate(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str: ...

    def bind_projection(self, job: ClaimedIngestionJob, receipt: IndexReceipt) -> str: ...

    def current_lifecycle(self, job: ClaimedIngestionJob) -> str: ...

    def projection_guard(self, job: ClaimedIngestionJob) -> ContextManager[None]: ...

    def projection_is_bound(self, job: ClaimedIngestionJob) -> bool: ...

    def refresh_projection_binding(self, job: ClaimedIngestionJob) -> ClaimedIngestionJob: ...

    def complete_projection(
        self,
        job: ClaimedIngestionJob,
        receipt: IndexReceipt,
        *,
        expected_lifecycle_status: str | None = None,
    ) -> None: ...

    def record_failure(
        self,
        job: ClaimedIngestionJob,
        *,
        error_code: str,
        retry: bool,
        retry_after: timedelta,
    ) -> None: ...


class ProjectionReconcilePort(Protocol):
    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> ClaimedProjectionRepair | None: ...

    def projection_guard(self, claim: ClaimedProjectionRepair) -> ContextManager[None]: ...

    def projection_is_bound(self, claim: ClaimedProjectionRepair) -> bool: ...

    def renew(self, claim: ClaimedProjectionRepair, *, lease_seconds: int) -> None: ...

    def prepare_rebuild(self, claim: ClaimedProjectionRepair) -> ClaimedProjectionRepair: ...

    def complete(
        self,
        claim: ClaimedProjectionRepair,
        receipt: IndexReceipt,
        *,
        audit_after: timedelta,
    ) -> None: ...

    def fail(self, claim: ClaimedProjectionRepair, *, retry_after: timedelta) -> None: ...
