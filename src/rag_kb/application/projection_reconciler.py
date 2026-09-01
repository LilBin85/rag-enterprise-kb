"""以 PostgreSQL 当前活动版本为真相的周期性 Qdrant 投影收敛器。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter

from opentelemetry import context as otel_context

from rag_kb.application.lease_heartbeat import LeaseHeartbeat
from rag_kb.application.ports.indexing import (
    ClaimedIngestionJob,
    ClaimedProjectionRepair,
    DocumentIndexPort,
    DocumentParserPort,
    ProjectionReconcilePort,
)
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


@dataclass(frozen=True)
class ProjectionReconcileResult:
    status: str
    document_id: str | None = None
    error_code: str | None = None


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
        started_at = perf_counter()
        run_phase = job.phase
        telemetry = current_observability()
        with observe_span(
            "projection.reconcile",
            attributes={
                "component": "projection_reconciler",
                "operation": "projection.reconcile",
                "phase": run_phase,
            },
        ) as span:
            result = ProjectionReconcileResult(status="retry_scheduled")
            try:
                guard_factory = getattr(self._repairs, "projection_guard", None)
                guard = guard_factory(claim) if guard_factory is not None else nullcontext()
                renew = getattr(self._repairs, "renew", None)
                heartbeat_context = otel_context.get_current()

                def record_lease_failure(exc: Exception) -> None:
                    if telemetry is not None:
                        telemetry.record_lease_failure(
                            kind="projection",
                            operation="projection.lease.renew",
                            error_code=stable_error_code(exc),
                            parent_context=heartbeat_context,
                        )

                heartbeat = LeaseHeartbeat(
                    (lambda: renew(claim, lease_seconds=self._lease_seconds))
                    if renew is not None
                    else None,
                    self._lease_seconds,
                    on_failure=record_lease_failure,
                )
                # 等待文档 guard 也可能超过租约，心跳必须在阻塞前启动。
                with heartbeat:
                    with guard:
                        heartbeat.renew_now()
                        receipt = job.index_receipt
                        verify = getattr(self._indexer, "verify", None)
                        if job.phase == "audit" and receipt is not None and verify is not None:
                            verify_started_at = perf_counter()
                            verify_outcome = "succeeded"
                            verify_error_code: str | None = None
                            with observe_span(
                                "projection.verify",
                                attributes={
                                    "component": "projection_reconciler",
                                    "operation": "projection.verify",
                                    "phase": "audit",
                                },
                            ) as verify_span:
                                try:
                                    healthy = verify(job, receipt)
                                    if not healthy:
                                        verify_outcome = "mismatch"
                                except Exception as exc:
                                    verify_outcome = "failure"
                                    verify_error_code = stable_error_code(exc)
                                    raise
                                finally:
                                    finish_observed_span(
                                        verify_span,
                                        outcome=verify_outcome,
                                        error_category=error_category_for(verify_error_code)
                                        if verify_error_code is not None
                                        else "none",
                                        error_code=verify_error_code,
                                        attributes={
                                            "duration_ms": (perf_counter() - verify_started_at)
                                            * 1_000
                                        },
                                    )
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
                                result = ProjectionReconcileResult(
                                    status="succeeded",
                                    document_id=job.document_id,
                                )
                                return result
                            prepare_rebuild = getattr(self._repairs, "prepare_rebuild", None)
                            if prepare_rebuild is not None:
                                claim = prepare_rebuild(claim)
                                job = claim.job
                                run_phase = job.phase
                                span.set_attribute("phase", run_phase)
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
                result = ProjectionReconcileResult(
                    status="succeeded",
                    document_id=job.document_id,
                )
                return result
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
                        result = ProjectionReconcileResult(
                            status="succeeded",
                            document_id=job.document_id,
                        )
                        return result
                    discard = getattr(self._indexer, "discard", None)
                    if discard is not None:
                        try:
                            discard(job)
                        except Exception:
                            pass
                try:
                    self._repairs.fail(claim, retry_after=self._retry_delay)
                except Exception:
                    result = ProjectionReconcileResult(
                        status="failure_recording_failed",
                        document_id=job.document_id,
                        error_code="projection_reconcile_failure_recording_failed",
                    )
                    return result
                result = ProjectionReconcileResult(
                    status="retry_scheduled",
                    document_id=job.document_id,
                    error_code="projection_reconcile_failed",
                )
                return result
            finally:
                duration_seconds = perf_counter() - started_at
                category = (
                    error_category_for(result.error_code)
                    if result.error_code is not None
                    else "none"
                )
                finish_observed_span(
                    span,
                    outcome=result.status,
                    error_category=category if result.status != "succeeded" else "none",
                    error_code=result.error_code,
                    attributes={"duration_ms": duration_seconds * 1_000},
                )
                if telemetry is not None:
                    telemetry.record_projection_reconciliation(
                        phase=run_phase,
                        outcome=result.status,
                        duration_seconds=duration_seconds,
                    )
                log_event(
                    "projection.reconcile.completed",
                    level=40 if result.status != "succeeded" else 20,
                    component="projection_reconciler",
                    operation="projection.reconcile",
                    phase=run_phase,
                    outcome=result.status,
                    error_category=category if result.status != "succeeded" else "none",
                    error_code=result.error_code,
                    duration_ms=duration_seconds * 1_000,
                )
