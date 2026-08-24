"""以 PostgreSQL 当前活动版本为真相的周期性 Qdrant 投影收敛器。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from rag_kb.application.ingestion_worker import (
    ClaimedIngestionJob,
    DocumentIndexPort,
    DocumentParserPort,
)
from rag_kb.domain.indexing import IndexReceipt


@dataclass(frozen=True)
class ClaimedProjectionRepair:
    job: ClaimedIngestionJob
    reconcile_lease_generation: int


@dataclass(frozen=True)
class ProjectionReconcileResult:
    status: str
    document_id: str | None = None
    error_code: str | None = None


class ProjectionReconcilePort(Protocol):
    def claim_next(
        self, *, worker_id: str, lease_seconds: int
    ) -> ClaimedProjectionRepair | None: ...

    def projection_guard(self, claim: ClaimedProjectionRepair): ...

    def complete(
        self,
        claim: ClaimedProjectionRepair,
        receipt: IndexReceipt,
        *,
        audit_after: timedelta,
    ) -> None: ...

    def fail(self, claim: ClaimedProjectionRepair, *, retry_after: timedelta) -> None: ...


class ProjectionReconciler:
    """重建当前活动版本并验证唯一 current 世代；失败时只记录安全错误码。"""

    def __init__(
        self,
        repairs: ProjectionReconcilePort,
        parser: DocumentParserPort,
        indexer: DocumentIndexPort,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        audit_interval: timedelta = timedelta(minutes=5),
        retry_delay: timedelta = timedelta(seconds=30),
    ) -> None:
        if (
            not worker_id
            or lease_seconds <= 0
            or audit_interval.total_seconds() <= 0
            or retry_delay.total_seconds() < 0
        ):
            raise ValueError("invalid_projection_reconciler_limits")
        self._repairs = repairs
        self._parser = parser
        self._indexer = indexer
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._audit_interval = audit_interval
        self._retry_delay = retry_delay

    def run_once(self) -> ProjectionReconcileResult:
        claim = self._repairs.claim_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if claim is None:
            return ProjectionReconcileResult(status="idle")
        job = claim.job
        try:
            guard_factory = getattr(self._repairs, "projection_guard", None)
            guard = guard_factory(claim) if guard_factory is not None else nullcontext()
            with guard:
                parsed = self._parser.parse(job)
                receipt = self._indexer.index(job, parsed)
                self._indexer.publish(job, receipt, lifecycle_status="active")
                self._repairs.complete(
                    claim,
                    receipt,
                    audit_after=self._audit_interval,
                )
        except Exception:
            try:
                self._repairs.fail(claim, retry_after=self._retry_delay)
            except Exception:
                return ProjectionReconcileResult(
                    status="failure_recording_failed",
                    document_id=job.document_id,
                    error_code="projection_reconcile_failure_recording_failed",
                )
            return ProjectionReconcileResult(
                status="retry_scheduled",
                document_id=job.document_id,
                error_code="projection_reconcile_failed",
            )
        return ProjectionReconcileResult(status="succeeded", document_id=job.document_id)
