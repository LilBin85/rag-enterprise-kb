"""可恢复的异步文档摄取工作器。

工作器只持久化安全错误码，绝不把正文、解析异常文本或对象内容写入任务错误。
索引端口必须按 ``(tenant_id, version_id)`` 实现幂等；进程在索引完成但激活前
终止时，租约到期后的再次执行因此仍是安全的。
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter

from opentelemetry import context as otel_context

from rag_kb.application.lease_heartbeat import LeaseHeartbeat
from rag_kb.application.ports.indexing import (
    DocumentIndexPort,
    DocumentParserPort,
    IngestionJobPort,
    RetryableIngestionError,
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
class WorkerRunResult:
    status: str
    job_id: str | None = None
    error_code: str | None = None


class ProjectionCleanupError(RetryableIngestionError):
    pass


class PermanentIngestionError(RuntimeError):
    """端口可显式标记为不可重试的失败；其消息不会被持久化。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# 兼容现有工作器级测试导入；实现位于独立共享模块，避免投影器反向依赖此模块。
_LeaseHeartbeat = LeaseHeartbeat


@contextmanager
def _ingestion_phase(phase: str, span_name: str):
    """统一结束摄取阶段 span 和低基数时长指标。"""
    started_at = perf_counter()
    outcome = "succeeded"
    error_code: str | None = None
    with observe_span(
        span_name,
        attributes={
            "component": "ingestion_worker",
            "operation": "ingestion.run",
            "phase": phase,
        },
    ) as span:
        try:
            yield
        except Exception as exc:
            outcome = "failure"
            error_code = stable_error_code(exc)
            raise
        finally:
            duration_seconds = perf_counter() - started_at
            finish_observed_span(
                span,
                outcome=outcome,
                error_category=error_category_for(error_code)
                if error_code is not None
                else "none",
                error_code=error_code,
                attributes={"duration_ms": duration_seconds * 1_000},
            )
            telemetry = current_observability()
            if telemetry is not None:
                telemetry.record_ingestion_phase(
                    phase=phase,
                    outcome=outcome,
                    duration_seconds=duration_seconds,
                )


class IngestionWorker:
    """一次领取一个任务的工作器；重复运行只会处理尚未成功的任务。"""

    def __init__(
        self,
        jobs: IngestionJobPort,
        parser: DocumentParserPort,
        indexer: DocumentIndexPort,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        max_attempts: int = 3,
        retry_delay: timedelta = timedelta(seconds=30),
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id_required")
        if lease_seconds <= 0 or max_attempts <= 0 or retry_delay.total_seconds() < 0:
            raise ValueError("invalid_ingestion_worker_limits")
        self._jobs = jobs
        self._parser = parser
        self._indexer = indexer
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def run_once(self) -> WorkerRunResult:
        job = self._jobs.claim_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
        )
        if job is None:
            return WorkerRunResult(status="idle")
        started_at = perf_counter()
        run_phase = job.phase
        telemetry = current_observability()
        attributes = {
            "component": "ingestion_worker",
            "operation": "ingestion.run",
            "phase": run_phase,
        }
        scope = (
            telemetry.ingestion_run_span(
                traceparent=job.pending_traceparent,
                tracestate=job.pending_tracestate,
                attributes=attributes,
            )
            if telemetry is not None
            else observe_span("ingestion.run", attributes=attributes)
        )
        with scope as span:
            result = WorkerRunResult(status="failed", job_id=job.job_id)
            try:
                renew = getattr(self._jobs, "renew_lease", None)
                heartbeat_context = otel_context.get_current()

                def record_lease_failure(exc: Exception) -> None:
                    if telemetry is not None:
                        telemetry.record_lease_failure(
                            kind="ingestion",
                            operation="ingestion.lease.renew",
                            error_code=stable_error_code(exc),
                            parent_context=heartbeat_context,
                        )

                heartbeat = _LeaseHeartbeat(
                    (lambda: renew(job, lease_seconds=self._lease_seconds))
                    if renew is not None
                    else None,
                    self._lease_seconds,
                    on_failure=record_lease_failure,
                )
                with heartbeat:
                    if job.phase == "cleanup":
                        guard_factory = getattr(self._jobs, "projection_guard", None)
                        guard = guard_factory(job) if guard_factory is not None else nullcontext()
                        with guard:
                            heartbeat.renew_now()
                            refresh = getattr(self._jobs, "refresh_projection_binding", None)
                            if refresh is not None:
                                job = refresh(job)
                            discard_unbound = getattr(self._indexer, "discard_unbound", None)
                            if discard_unbound is None:
                                raise ProjectionCleanupError("ingestion_projection_cleanup_failed")
                            try:
                                discard_unbound(job)
                            except Exception as exc:
                                raise ProjectionCleanupError(
                                    "ingestion_projection_cleanup_failed"
                                ) from exc
                        heartbeat.stop()
                        self._jobs.record_failure(
                            job,
                            error_code="ingestion_retry_exhausted",
                            retry=False,
                            retry_after=self._retry_delay,
                        )
                        result = WorkerRunResult(
                            status="failed",
                            job_id=job.job_id,
                            error_code="ingestion_retry_exhausted",
                        )
                        return result
                    if job.phase == "ingest" and job.projection_generation is not None:
                        guard_factory = getattr(self._jobs, "projection_guard", None)
                        guard = guard_factory(job) if guard_factory is not None else nullcontext()
                        with guard:
                            heartbeat.renew_now()
                            refresh = getattr(self._jobs, "refresh_projection_binding", None)
                            if refresh is not None:
                                job = refresh(job)
                            discard_unbound = getattr(self._indexer, "discard_unbound", None)
                            if discard_unbound is not None:
                                try:
                                    discard_unbound(job)
                                except Exception as exc:
                                    raise ProjectionCleanupError(
                                        "ingestion_projection_cleanup_failed"
                                    ) from exc
                    receipt = job.index_receipt
                    lifecycle_status = job.lifecycle_status
                    parsed_document = None
                    if job.phase == "ingest":
                        with _ingestion_phase("parse", "ingestion.parse"):
                            parsed_document = self._parser.parse(job)
                            heartbeat.check()
                            self._jobs.persist_manifest(job, parsed_document)
                            heartbeat.check()
                    guard_factory = getattr(self._jobs, "projection_guard", None)
                    guard = guard_factory(job) if guard_factory is not None else nullcontext()
                    with guard:
                        heartbeat.renew_now()
                        if job.phase == "ingest":
                            allocator = getattr(self._jobs, "allocate_projection", None)
                            if allocator is not None:
                                job = allocator(job)
                            discard_unbound = getattr(self._indexer, "discard_unbound", None)
                            if discard_unbound is not None:
                                try:
                                    discard_unbound(job)
                                except Exception as exc:
                                    raise ProjectionCleanupError(
                                        "ingestion_projection_cleanup_failed"
                                    ) from exc
                        if job.projection_generation is None:
                            raise PermanentIngestionError("projection_generation_missing")
                        if job.phase == "ingest":
                            if parsed_document is None:
                                raise PermanentIngestionError("parsed_document_missing")
                            with _ingestion_phase("index", "projection.index"):
                                receipt = self._indexer.index(job, parsed_document)
                            heartbeat.renew_now()
                            # 新世代先可见；SQL 精确世代仍指向旧投影，因此切换期间两者均可安全存在。
                            with _ingestion_phase("publish", "projection.publish"):
                                self._indexer.publish(job, receipt, lifecycle_status="active")
                            heartbeat.stop()
                            with _ingestion_phase("bind", "projection.bind"):
                                binder = getattr(self._jobs, "bind_projection", None)
                                if binder is not None:
                                    lifecycle_status = binder(job, receipt)
                                    if lifecycle_status != "active":
                                        try:
                                            with _ingestion_phase(
                                                "publish", "projection.publish"
                                            ):
                                                self._indexer.publish(
                                                    job,
                                                    receipt,
                                                    lifecycle_status=lifecycle_status,
                                                )
                                        except Exception:
                                            # 精确 SQL 绑定不会授权 superseded 世代；活动投影审计会清理它。
                                            pass
                                else:
                                    lifecycle_status = self._jobs.activate(job, receipt)
                                    if lifecycle_status != "active":
                                        with _ingestion_phase("publish", "projection.publish"):
                                            self._indexer.publish(
                                                job,
                                                receipt,
                                                lifecycle_status=lifecycle_status,
                                            )
                                    self._jobs.complete_projection(
                                        job,
                                        receipt,
                                        expected_lifecycle_status=lifecycle_status,
                                    )
                        else:
                            receipt_loader = getattr(self._jobs, "load_index_receipt", None)
                            if receipt_loader is not None:
                                receipt = receipt_loader(job)
                            current_lifecycle = getattr(self._jobs, "current_lifecycle", None)
                            if current_lifecycle is not None:
                                lifecycle_status = current_lifecycle(job)
                            if receipt is None:
                                raise PermanentIngestionError("index_receipt_missing")
                            with _ingestion_phase("publish", "projection.publish"):
                                self._indexer.publish(
                                    job,
                                    receipt,
                                    lifecycle_status=lifecycle_status,
                                )
                            heartbeat.stop()
                            with _ingestion_phase("bind", "projection.bind"):
                                self._jobs.complete_projection(
                                    job,
                                    receipt,
                                    expected_lifecycle_status=lifecycle_status,
                                )
                        if receipt is None:
                            raise PermanentIngestionError("index_receipt_missing")
                        if lifecycle_status == "active":
                            retire = getattr(self._indexer, "retire", None)
                            if retire is not None:
                                try:
                                    with _ingestion_phase("retire", "projection.retire"):
                                        retire(job, receipt)
                                except Exception:
                                    # SQL 已精确绑定新世代；周期审计会安全重试旧投影清理。
                                    pass
                result = WorkerRunResult(status="succeeded", job_id=job.job_id)
                return result
            except Exception as exc:
                error_code, retry = self._failure_disposition(exc, attempts=job.attempts)
                cleanup_failed = job.phase == "cleanup" or isinstance(exc, ProjectionCleanupError)
                if (
                    job.projection_generation is not None
                    and job.projection_generation != job.previous_projection_generation
                ):
                    projection_is_bound = getattr(self._jobs, "projection_is_bound", None)
                    try:
                        bound = projection_is_bound(job) if projection_is_bound is not None else False
                    except Exception:
                        bound = True
                    if bound and not cleanup_failed:
                        result = WorkerRunResult(status="succeeded", job_id=job.job_id)
                        return result
                    discard = getattr(self._indexer, "discard", None)
                    if discard is not None:
                        try:
                            discard(job)
                        except Exception:
                            # 数据保留清理优先于终态；清理恢复前持续保持可领取。
                            cleanup_failed = True
                if cleanup_failed:
                    error_code, retry = "ingestion_projection_cleanup_failed", True
                try:
                    self._jobs.record_failure(
                        job,
                        error_code=error_code,
                        retry=retry,
                        retry_after=self._retry_delay,
                    )
                except Exception:
                    # 错误持久化失败时不能暴露原始异常；租约到期后由适配层恢复。
                    result = WorkerRunResult(
                        status="failure_recording_failed",
                        job_id=job.job_id,
                        error_code="ingestion_failure_recording_failed",
                    )
                    return result
                result = WorkerRunResult(
                    status="retry_scheduled" if retry else "failed",
                    job_id=job.job_id,
                    error_code=error_code,
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
                    telemetry.record_ingestion_job(
                        phase=run_phase,
                        outcome=result.status,
                        error_category=category if result.status != "succeeded" else "none",
                    )
                log_event(
                    "ingestion.job.completed",
                    level=40 if result.status != "succeeded" else 20,
                    component="ingestion_worker",
                    operation="ingestion.run",
                    phase=run_phase,
                    outcome=result.status,
                    error_category=category if result.status != "succeeded" else "none",
                    error_code=result.error_code,
                    duration_ms=duration_seconds * 1_000,
                )

    def _failure_disposition(self, exc: Exception, *, attempts: int) -> tuple[str, bool]:
        if isinstance(exc, PermanentIngestionError):
            return "ingestion_permanent_failure", False
        if attempts >= self._max_attempts:
            return "ingestion_retry_exhausted", False
        if isinstance(exc, RetryableIngestionError):
            return "ingestion_retryable_failure", True
        return "ingestion_processing_failed", True
