"""受控工具执行的数据库原子生命周期。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from rag_kb.application.ports.approvals import principal_fingerprint
from rag_kb.application.ports.tool_execution import (
    ToolExecutionCompletion,
    ToolExecutionFacts,
    ToolExecutionLifecycleError,
    ToolExecutionLifecyclePort,
    ToolExecutionRequest,
    ToolExecutionReservation,
    ToolExecutionRunningCall,
)
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tools import ToolMode
from rag_kb.infrastructure.database.approval_lifecycle import (
    consume_approved_execution_in_session,
)
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    ArtifactRow,
    PlanRevisionRow,
    PrincipalRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskEvidencePort,
    TaskExecutionContextError,
    current_task_auth,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


logger = logging.getLogger(__name__)

_SYSTEM_REJECT_ACTOR = "system:approval-reaper"


def _aware_datetime(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


def _default_task_content_retention_days() -> int:
    from rag_kb.config import get_settings

    return get_settings().task_result_retention_days


class DatabaseToolExecutionLifecycle(ToolExecutionLifecyclePort):
    def __init__(
        self,
        database: Database,
        *,
        evidence_validator: DatabaseTaskEvidencePort | None = None,
        task_content_retention_days: int | None = None,
        now=None,
    ) -> None:
        self._database = database
        self._evidence_validator = evidence_validator or DatabaseTaskEvidencePort(database)
        self._task_content_retention_days = (
            task_content_retention_days
            if task_content_retention_days is not None
            else _default_task_content_retention_days()
        )
        self._now = now or (lambda: datetime.now(UTC))

    def current_auth(
        self,
        request: ToolExecutionRequest,
        *,
        principal_hash: str,
    ) -> AuthContext:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            _, current_auth = self._require_current_execution_context(
                work.session,
                request,
                principal_hash=principal_hash,
            )
            return current_auth

    def preflight_write(
        self,
        request: ToolExecutionRequest,
        *,
        principal_hash: str,
    ) -> None:
        """重放前也复核当前计划、证据和主体；原子门仍在 reservation 内重查。"""
        try:
            with self._database.sessions() as session:
                task = session.scalar(
                    select(AgentTaskRow).where(
                        AgentTaskRow.tenant_id == request.auth.tenant_id,
                        AgentTaskRow.task_id == request.task_id,
                        AgentTaskRow.user_id == request.auth.user_id,
                    )
                )
                plan = session.scalar(
                    select(PlanRevisionRow).where(
                        PlanRevisionRow.tenant_id == request.auth.tenant_id,
                        PlanRevisionRow.task_id == request.task_id,
                        PlanRevisionRow.revision == request.plan_revision,
                        PlanRevisionRow.plan_digest == request.plan_digest,
                        PlanRevisionRow.evidence_digest == request.evidence_digest,
                    )
                )
                if task is None or plan is None or task.task_mode != "controlled_action":
                    raise ToolExecutionLifecycleError("write_task_mode_denied")
                try:
                    current_auth, plan_evidence = (
                        self._evidence_validator.validate_plan_in_session(
                            session,
                            task=task,
                            plan=plan,
                            request_id=request.auth.request_id,
                            require_non_empty=True,
                        )
                    )
                    requested_evidence = canonical_evidence_snapshot(
                        request.evidence_ids,
                        require_non_empty=True,
                    )
                except (TaskExecutionContextError, ValueError) as exc:
                    raise ToolExecutionLifecycleError(
                        getattr(exc, "code", str(exc))
                    ) from exc
                if (
                    requested_evidence != plan_evidence
                    or principal_fingerprint(current_auth) != principal_hash
                ):
                    raise ToolExecutionLifecycleError("tool_current_binding_invalid")
        except ToolExecutionLifecycleError as exc:
            self._reject_stale_approval(
                tenant_id=request.auth.tenant_id,
                approval_id=request.approval_id,
                reason=exc.code,
            )
            raise

    def reserve_or_replay(
        self,
        request: ToolExecutionRequest,
        facts: ToolExecutionFacts,
        *,
        tool_call_id: str,
    ) -> ToolExecutionReservation:
        duplicate = self._existing_idempotent_result(request, facts)
        if duplicate is not None:
            return duplicate
        try:
            self._persist_running(request, facts, tool_call_id=tool_call_id)
        except IntegrityError:
            duplicate = self._existing_idempotent_result(request, facts)
            if duplicate is not None:
                return duplicate
            if request.approval_id is not None:
                with self._database.sessions() as session:
                    consumed = session.scalar(
                        select(ToolCallRow.tool_call_id).where(
                            ToolCallRow.tenant_id == request.auth.tenant_id,
                            ToolCallRow.approval_id == request.approval_id,
                        )
                    )
                if consumed is not None:
                    raise ToolExecutionLifecycleError("approval_already_consumed")
            raise
        return ToolExecutionReservation(
            tool_call_id=tool_call_id,
            mode=facts.mode,
            status="running",
        )

    def confirm_replay(
        self,
        request: ToolExecutionRequest,
        reservation: ToolExecutionReservation,
        *,
        principal_hash: str,
    ) -> bool:
        historical_principal_hash = reservation.historical_principal_hash
        if historical_principal_hash is None:
            raise ToolExecutionLifecycleError("tool_replay_binding_invalid")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._require_current_execution_context(
                work.session,
                request,
                principal_hash=principal_hash,
                historical_principal_hash=historical_principal_hash,
            )
            latest = work.session.scalar(
                select(ToolCallRow)
                .where(
                    ToolCallRow.tool_call_id == reservation.tool_call_id,
                    ToolCallRow.tenant_id == request.auth.tenant_id,
                    ToolCallRow.task_id == request.task_id,
                )
                .with_for_update()
            )
            now = self._execution_now(work.session)
            return not (
                latest is None
                or latest.status != "succeeded"
                or latest.result_ref != reservation.result_ref
                or latest.result_hash != reservation.result_hash
                or latest.result_purged_at is not None
                or (
                    _aware_datetime(latest.result_expires_at) is not None
                    and _aware_datetime(latest.result_expires_at) <= now
                )
            )

    def prepare_completion(
        self,
        request: ToolExecutionRequest,
        *,
        tool_call_id: str,
        principal_hash: str,
    ) -> ToolExecutionRunningCall:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._require_current_execution_context(
                work.session,
                request,
                principal_hash=principal_hash,
            )
            row = work.session.scalar(
                select(ToolCallRow)
                .where(
                    ToolCallRow.tool_call_id == tool_call_id,
                    ToolCallRow.tenant_id == request.auth.tenant_id,
                    ToolCallRow.task_id == request.task_id,
                )
                .with_for_update()
            )
            if row is None or row.status != "running":
                raise ToolExecutionLifecycleError("tool_call_state_conflict")
            if row.principal_hash != principal_hash:
                raise ToolExecutionLifecycleError("task_auth_context_changed")
            return ToolExecutionRunningCall(
                tool_call_id=row.tool_call_id,
                mode=ToolMode(row.mode),
            )

    def complete(
        self,
        request: ToolExecutionRequest,
        *,
        tool_call_id: str,
        principal_hash: str,
        status: str,
        error_code: str | None,
        result_hash: str | None,
        result_ref: str | None,
        result_size: int | None,
        artifact_id: str | None,
        tool_result_ttl_seconds: int,
    ) -> ToolExecutionCompletion:
        final_status = status
        final_error = error_code
        final_hash = result_hash
        final_ref = result_ref
        final_size = result_size
        result_expires_at = None
        result_to_discard: str | None = None
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            task, _ = self._require_current_execution_context(
                work.session,
                request,
                principal_hash=principal_hash,
            )
            row = work.session.scalar(
                select(ToolCallRow)
                .where(
                    ToolCallRow.tool_call_id == tool_call_id,
                    ToolCallRow.tenant_id == request.auth.tenant_id,
                    ToolCallRow.task_id == request.task_id,
                )
                .with_for_update()
            )
            if row is None or row.status != "running":
                raise ToolExecutionLifecycleError("tool_call_state_conflict")
            if row.principal_hash != principal_hash:
                raise ToolExecutionLifecycleError("task_auth_context_changed")
            now = self._execution_now(work.session)
            execution_deadline = _aware_datetime(row.execution_deadline)
            if (
                final_ref is not None
                and execution_deadline is not None
                and execution_deadline <= now
            ):
                result_to_discard = final_ref
                final_status = "failed" if row.mode == "read" else "unknown"
                final_error = "execution_deadline_expired"
                final_hash = None
                final_ref = None
                final_size = None
            if final_ref is not None:
                result_expires_at = now + timedelta(seconds=tool_result_ttl_seconds)
            row.status = final_status
            row.result_hash = final_hash
            row.error_code = final_error
            row.result_ref = final_ref
            row.result_size = final_size
            row.result_expires_at = result_expires_at
            row.completed_at = now
            row.verified_at = now if final_status == "succeeded" else None
            if row.mode == "write" and final_status == "unknown":
                expires_at = None
                if task.task_mode == "controlled_action":
                    expires_at = now + timedelta(days=self._task_content_retention_days)
                    task.content_expires_at = expires_at
                for artifact in work.session.scalars(
                    select(ArtifactRow).where(
                        ArtifactRow.tenant_id == row.tenant_id,
                        ArtifactRow.task_id == row.task_id,
                        ArtifactRow.content_purged_at.is_(None),
                    )
                ):
                    if artifact.status in ("creating", "ready"):
                        artifact.status = "unknown"
                    if expires_at is not None:
                        artifact.content_expires_at = expires_at
                task.status = "unknown"
                task.last_error_code = final_error
                task.lease_owner = None
                task.lease_expires_at = None
                task.state_version += 1
            elif (
                row.mode == "write"
                and final_status == "succeeded"
                and artifact_id is not None
                and artifact_id not in task.artifact_refs
            ):
                # 业务引用只绑定稳定 artifact ID，不写对象存储路径。
                task.artifact_refs = [*task.artifact_refs, artifact_id]
                task.state_version += 1
        return ToolExecutionCompletion(
            status=final_status,
            result_hash=final_hash,
            result_ref=final_ref,
            error_code=final_error,
            result_to_discard=result_to_discard,
        )

    def is_result_bound(self, *, tool_call_id: str, result_ref: str) -> bool:
        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                row = work.session.scalar(
                    select(ToolCallRow)
                    .where(ToolCallRow.tool_call_id == tool_call_id)
                    .with_for_update()
                )
                return row is not None and row.result_ref == result_ref
        except Exception:
            # 提交确认丢失时不能猜测未绑定；保留对象交给孤儿扫描收敛。
            logger.error("工具结果绑定状态确认失败 tool_call_id=%s", tool_call_id)
            return True

    def expire_stale_running(self, *, limit: int) -> int:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._execution_clock_expression(work.session)
            candidates = tuple(
                work.session.execute(
                    select(
                        ToolCallRow.tenant_id,
                        ToolCallRow.task_id,
                        ToolCallRow.tool_call_id,
                    )
                    .where(
                        ToolCallRow.status == "running",
                        ToolCallRow.execution_deadline <= clock,
                    )
                    .order_by(ToolCallRow.execution_deadline)
                    .limit(limit)
                )
            )
            now = self._execution_now(work.session)
            expired = 0
            for tenant_id, task_id, tool_call_id in candidates:
                # 与 complete() 一致：task 总在 tool call 之前锁定。
                task = work.session.scalar(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.tenant_id == tenant_id,
                        AgentTaskRow.task_id == task_id,
                    )
                    .with_for_update(skip_locked=True)
                )
                if task is None:
                    continue
                row = work.session.scalar(
                    select(ToolCallRow)
                    .where(
                        ToolCallRow.tenant_id == tenant_id,
                        ToolCallRow.task_id == task_id,
                        ToolCallRow.tool_call_id == tool_call_id,
                        ToolCallRow.status == "running",
                        ToolCallRow.execution_deadline <= clock,
                    )
                    .with_for_update(skip_locked=True)
                )
                if row is None:
                    continue
                # READ 崩溃可安全收敛为 failed；WRITE 副作用未知必须保持 unknown。
                row.status = "failed" if row.mode == "read" else "unknown"
                row.error_code = "execution_deadline_expired"
                row.completed_at = now
                if row.mode == "write":
                    self._settle_unknown_write(work.session, row, task=task, now=now)
                expired += 1
            return expired

    def _existing_idempotent_result(
        self,
        request: ToolExecutionRequest,
        facts: ToolExecutionFacts,
    ) -> ToolExecutionReservation | None:
        if request.idempotency_key is None:
            return None
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._require_current_execution_context(
                work.session,
                request,
                principal_hash=facts.principal_hash,
            )
            row = work.session.scalar(
                select(ToolCallRow)
                .where(
                    ToolCallRow.tenant_id == request.auth.tenant_id,
                    ToolCallRow.task_id == request.task_id,
                    ToolCallRow.tool_name == request.tool_name,
                    ToolCallRow.idempotency_key == request.idempotency_key,
                )
                .with_for_update()
            )
            if row is None:
                return None
            self._validate_idempotency_binding(row, request, facts)
            mode = ToolMode(row.mode)
            now = self._execution_now(work.session)
            expires_at = _aware_datetime(row.result_expires_at)
            if row.status == "succeeded" and (
                row.result_purged_at is not None
                or expires_at is not None and expires_at <= now
            ):
                return ToolExecutionReservation(
                    tool_call_id=row.tool_call_id,
                    mode=mode,
                    status="failed" if mode is ToolMode.READ else "unknown",
                    result_hash=row.result_hash,
                    error_code="tool_result_expired",
                    duplicate=True,
                )
            if row.status != "succeeded":
                return ToolExecutionReservation(
                    tool_call_id=row.tool_call_id,
                    mode=mode,
                    status=(
                        row.status
                        if row.status != "running"
                        else "failed" if mode is ToolMode.READ else "unknown"
                    ),
                    result_hash=row.result_hash,
                    result_ref=row.result_ref,
                    error_code=(
                        "previous_execution_in_progress"
                        if row.status == "running"
                        else row.error_code
                    ),
                    duplicate=True,
                )
            if not row.result_ref:
                return ToolExecutionReservation(
                    tool_call_id=row.tool_call_id,
                    mode=mode,
                    status="failed" if mode is ToolMode.READ else "unknown",
                    result_hash=row.result_hash,
                    error_code="tool_result_readback_failed",
                    duplicate=True,
                )
            return ToolExecutionReservation(
                tool_call_id=row.tool_call_id,
                mode=mode,
                status="succeeded",
                result_hash=row.result_hash,
                result_ref=row.result_ref,
                duplicate=True,
                requires_result_read=True,
                historical_principal_hash=row.principal_hash,
            )

    def _persist_running(
        self,
        request: ToolExecutionRequest,
        facts: ToolExecutionFacts,
        *,
        tool_call_id: str,
    ) -> None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            task, current_auth = self._require_current_execution_context(
                work.session,
                request,
                principal_hash=facts.principal_hash,
            )
            plan = None
            if facts.mode is ToolMode.WRITE:
                plan = work.session.scalar(
                    select(PlanRevisionRow)
                    .where(
                        PlanRevisionRow.tenant_id == request.auth.tenant_id,
                        PlanRevisionRow.task_id == request.task_id,
                        PlanRevisionRow.revision == request.plan_revision,
                        PlanRevisionRow.plan_digest == request.plan_digest,
                        PlanRevisionRow.evidence_digest == request.evidence_digest,
                    )
                    .with_for_update()
                )
            if facts.mode is ToolMode.WRITE and plan is None:
                raise ToolExecutionLifecycleError("tool_plan_binding_invalid")
            if facts.mode is ToolMode.WRITE:
                if task.task_mode != "controlled_action":
                    raise ToolExecutionLifecycleError("write_task_mode_denied")
                try:
                    current_auth, plan_evidence = (
                        self._evidence_validator.validate_plan_in_session(
                            work.session,
                            task=task,
                            plan=plan,
                            request_id=request.auth.request_id,
                            require_non_empty=True,
                        )
                    )
                    requested_evidence = canonical_evidence_snapshot(
                        request.evidence_ids,
                        require_non_empty=True,
                    )
                except (TaskExecutionContextError, ValueError) as exc:
                    raise ToolExecutionLifecycleError(
                        getattr(exc, "code", str(exc))
                    ) from exc
                if (
                    requested_evidence != plan_evidence
                    or principal_fingerprint(current_auth) != facts.principal_hash
                ):
                    raise ToolExecutionLifecycleError("tool_current_binding_invalid")
            approval_id = None
            now = self._execution_now(work.session)
            if facts.mode is ToolMode.WRITE:
                if request.approval_id is None:
                    raise ToolExecutionLifecycleError("approval_required")
                try:
                    approval_id = consume_approved_execution_in_session(
                        work.session,
                        auth=request.auth,
                        approval_id=request.approval_id,
                        task_id=request.task_id,
                        plan_revision=request.plan_revision,
                        plan_digest=request.plan_digest,
                        evidence_digest=request.evidence_digest,
                        evidence_ids=request.evidence_ids,
                        tool_name=request.tool_name,
                        tool_version=facts.tool_version,
                        arguments_hash=facts.arguments_hash,
                        now=now,
                    )
                except Exception as exc:
                    code = getattr(exc, "code", None)
                    if isinstance(code, str) and code:
                        raise ToolExecutionLifecycleError(code) from exc
                    raise
            work.session.add(
                ToolCallRow(
                    tool_call_id=tool_call_id,
                    task_id=request.task_id,
                    tenant_id=request.auth.tenant_id,
                    approval_id=approval_id,
                    plan_revision=request.plan_revision,
                    plan_digest=request.plan_digest,
                    evidence_digest=request.evidence_digest,
                    tool_name=request.tool_name,
                    tool_version=facts.tool_version,
                    arguments_hash=facts.arguments_hash,
                    principal_hash=facts.principal_hash,
                    mode=facts.mode.value,
                    idempotency_key=request.idempotency_key,
                    risk_level=facts.risk_level,
                    status="running",
                    created_at=now,
                    execution_deadline=now + timedelta(seconds=facts.timeout_seconds),
                )
            )

    def _require_active_lease(self, session, request: ToolExecutionRequest) -> AgentTaskRow:
        task = session.scalar(
            select(AgentTaskRow)
            .where(
                AgentTaskRow.tenant_id == request.auth.tenant_id,
                AgentTaskRow.task_id == request.task_id,
                AgentTaskRow.user_id == request.auth.user_id,
            )
            .with_for_update()
        )
        if task is None:
            raise ToolExecutionLifecycleError("tool_task_missing")
        now = self._execution_now(session)
        lease_expires_at = _aware_datetime(task.lease_expires_at)
        if (
            task.lease_owner != request.worker_id
            or task.lease_generation != request.lease_generation
            or lease_expires_at is None
            or lease_expires_at <= now
        ):
            raise ToolExecutionLifecycleError("tool_task_lease_stale")
        deadline = _aware_datetime(task.deadline)
        if (
            task.cancelled
            or task.status not in {"planning", "running"}
            or deadline is not None and deadline <= now
        ):
            raise ToolExecutionLifecycleError("tool_task_not_executable")
        return task

    def _require_current_execution_context(
        self,
        session,
        request: ToolExecutionRequest,
        *,
        principal_hash: str,
        historical_principal_hash: str | None = None,
    ) -> tuple[AgentTaskRow, AuthContext]:
        task = self._require_active_lease(session, request)
        try:
            current_auth = current_task_auth(
                session,
                task=task,
                auth_context_ref=task.auth_context_ref,
                request_id=request.auth.request_id,
            )
        except TaskExecutionContextError as exc:
            raise ToolExecutionLifecycleError(
                getattr(exc, "code", str(exc))
            ) from exc
        current_principal_hash = principal_fingerprint(current_auth)
        if (
            current_principal_hash != principal_hash
            or historical_principal_hash is not None
            and current_principal_hash != historical_principal_hash
        ):
            raise ToolExecutionLifecycleError("task_auth_context_changed")
        return task, current_auth

    @staticmethod
    def _validate_idempotency_binding(
        row: ToolCallRow,
        request: ToolExecutionRequest,
        facts: ToolExecutionFacts,
    ) -> None:
        if row.principal_hash != facts.principal_hash:
            raise ToolExecutionLifecycleError("task_auth_context_changed")
        binding = (
            row.plan_revision,
            row.plan_digest,
            row.evidence_digest,
            row.tool_version,
            row.arguments_hash,
        )
        expected = (
            request.plan_revision,
            request.plan_digest,
            request.evidence_digest,
            facts.tool_version,
            facts.arguments_hash,
        )
        if binding != expected:
            raise ToolExecutionLifecycleError("idempotency_key_conflict")

    def _reject_stale_approval(
        self,
        *,
        tenant_id: str,
        approval_id: str | None,
        reason: str,
    ) -> None:
        if approval_id is None:
            return
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            actor = work.session.scalar(
                select(PrincipalRow.user_id).where(
                    PrincipalRow.tenant_id == tenant_id,
                    PrincipalRow.user_id == _SYSTEM_REJECT_ACTOR,
                )
            )
            if actor is None:
                work.session.add(
                    PrincipalRow(
                        tenant_id=tenant_id,
                        user_id=_SYSTEM_REJECT_ACTOR,
                        roles=[],
                        attributes={"permissions": []},
                    )
                )
            row = work.session.scalar(
                select(ApprovalRow)
                .where(
                    ApprovalRow.tenant_id == tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
                .with_for_update()
            )
            if row is not None and row.status in {"pending", "approved"}:
                row.status = "rejected"
                row.rejected_by = _SYSTEM_REJECT_ACTOR
                row.rejected_at = self._execution_now(work.session)
                row.rejection_reason = f"auto:{reason}"[:200]

    def _settle_unknown_write(
        self,
        session,
        row: ToolCallRow,
        *,
        task: AgentTaskRow,
        now: datetime,
    ) -> None:
        expires_at = None
        if task.task_mode == "controlled_action":
            expires_at = now + timedelta(days=self._task_content_retention_days)
            task.content_expires_at = expires_at
        for artifact in session.scalars(
            select(ArtifactRow).where(
                ArtifactRow.tenant_id == row.tenant_id,
                ArtifactRow.task_id == row.task_id,
                ArtifactRow.content_purged_at.is_(None),
            )
        ):
            if artifact.status in ("creating", "ready"):
                artifact.status = "unknown"
            if expires_at is not None:
                artifact.content_expires_at = expires_at
        task.status = "unknown"
        task.last_error_code = "execution_deadline_expired"
        task.lease_owner = None
        task.lease_expires_at = None
        task.state_version += 1

    def _execution_now(self, session) -> datetime:
        if session.get_bind().dialect.name != "postgresql":
            return self._now()
        value = session.scalar(select(func.clock_timestamp()))
        if not isinstance(value, datetime):
            raise RuntimeError("database_clock_unavailable")
        return _aware_datetime(value) or value

    def _execution_clock_expression(self, session):
        if session.get_bind().dialect.name == "postgresql":
            return func.clock_timestamp()
        return self._now()
