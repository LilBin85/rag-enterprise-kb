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
    _LeaseHeartbeat,
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
            renew = getattr(self._repairs, "renew", None)
            heartbeat = _LeaseHeartbeat(
                (lambda: renew(claim, lease_seconds=self._lease_seconds))
                if renew is not None
                else None,
                self._lease_seconds,
            )
            # 等待文档 guard 也可能超过租约，心跳必须在阻塞前启动。
            with heartbeat:
                with guard:
                    heartbeat.renew_now()
                    receipt = job.index_receipt
                    verify = getattr(self._indexer, "verify", None)
                    if job.phase == "audit" and receipt is not None and verify is not None:
                        healthy = verify(job, receipt)
                        heartbeat.renew_now()
                        if healthy:
                            retire = getattr(self._indexer, "retire", None)
                            if retire is not None:
                                retire(job, receipt)
                            heartbeat.stop()
                            self._repairs.complete(
                                claim,
                                receipt,
                                audit_after=self._audit_interval,
                            )
                            return ProjectionReconcileResult(
                                status="succeeded",
                                document_id=job.document_id,
                            )
                        prepare_rebuild = getattr(self._repairs, "prepare_rebuild", None)
                        if prepare_rebuild is not None:
                            claim = prepare_rebuild(claim)
                            job = claim.job
                    if job.phase == "reconcile":
                        discard_unbound = getattr(self._indexer, "discard_unbound", None)
                        if discard_unbound is not None:
                            discard_unbound(job)
                    parsed = self._parser.parse(job)
                    heartbeat.renew_now()
                    receipt = self._indexer.index(job, parsed)
                    heartbeat.renew_now()
                    self._indexer.publish(job, receipt, lifecycle_status="active")
                    heartbeat.stop()
                    self._repairs.complete(
                        claim,
                        receipt,
                        audit_after=self._audit_interval,
                    )
                    retire = getattr(self._indexer, "retire", None)
                    if retire is not None:
                        try:
                            retire(job, receipt)
                        except Exception:
                            # SQL 已绑定精确世代；下一轮轻量审计会重试清理。
                            pass
        except Exception:
            if (
                job.phase == "reconcile"
                and job.projection_generation is not None
                and job.projection_generation != job.previous_projection_generation
            ):
                projection_is_bound = getattr(self._repairs, "projection_is_bound", None)
                try:
                    bound = (
                        projection_is_bound(claim)
                        if projection_is_bound is not None
                        else False
                    )
                except Exception:
                    bound = True
                if bound:
                    return ProjectionReconcileResult(
                        status="succeeded",
                        document_id=job.document_id,
                    )
                discard = getattr(self._indexer, "discard", None)
                if discard is not None:
                    try:
                        discard(job)
                    except Exception:
                        pass
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
