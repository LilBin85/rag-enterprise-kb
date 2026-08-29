"""受控工具执行：先持久化意图，再执行，并对不确定副作用关闭重试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
from threading import Lock
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from rag_kb.application.approval_service import (
    ApprovalService,
    arguments_fingerprint,
    principal_fingerprint,
)
from rag_kb.application.tool_registry import (
    RegisteredTool,
    ToolExecutionContext,
    ToolRegistry,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tools import ToolMode
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
from rag_kb.infrastructure.object_store.local import LocalObjectStore, ObjectConflictError
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


logger = logging.getLogger(__name__)

# 失租与身份失效都必须中止图执行，避免继续写 ToolMessage/checkpoint；
# 但 TaskWorker 只可把前者视为 stale，后者仍应在持租约时失败关闭。
_STALE_TOOL_CODES = frozenset({"tool_task_lease_stale", "task_checkpoint_lease_stale"})
_AUTH_TOOL_CODES = frozenset(
    {
        "task_auth_context_changed",
        "task_auth_context_binding_mismatch",
        "task_principal_unavailable",
    }
)
_FATAL_TOOL_CODES = _STALE_TOOL_CODES | _AUTH_TOOL_CODES

# 自动失效的固定服务端 actor（M2_PLAN D3）：principals 行由拒绝事务内确保存在，
# 满足 fk_approvals_rejected_by_principal；reason 统一带 auto: 稳定前缀。
_SYSTEM_REJECT_ACTOR = "system:approval-reaper"


def _aware_datetime(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


def _default_tool_result_ttl_seconds() -> int:
    from rag_kb.config import get_settings

    return get_settings().tool_result_ttl_seconds


def _default_task_content_retention_days() -> int:
    from rag_kb.config import get_settings

    return get_settings().task_result_retention_days


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ToolExecutionRequest:
    auth: AuthContext
    task_id: str
    tool_name: str
    arguments: dict[str, object]
    worker_id: str
    lease_generation: int
    idempotency_key: str | None = None
    approval_id: str | None = None
    # M1 起 READ 工具不携带计划绑定；WRITE（M2）仍要求完整计划三元组。
    plan_revision: int | None = None
    plan_digest: str | None = None
    evidence_digest: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    status: str
    result_hash: str | None = None
    result_ref: str | None = None
    result: dict[str, object] | None = None
    error_code: str | None = None
    duplicate: bool = False


class ToolExecutionService:
    def __init__(
        self,
        database: Database,
        registry: ToolRegistry,
        approval_service: ApprovalService,
        *,
        evidence_validator: DatabaseTaskEvidencePort | None = None,
        tool_results: LocalObjectStore | None = None,
        tool_result_ttl_seconds: int | None = None,
        now=None,
    ) -> None:
        self._database = database
        self._registry = registry
        self._approvals = approval_service
        self._evidence_validator = evidence_validator or DatabaseTaskEvidencePort(database)
        self._tool_results = tool_results
        self._tool_result_ttl_seconds = (
            tool_result_ttl_seconds
            if tool_result_ttl_seconds is not None
            else _default_tool_result_ttl_seconds()
        )
        self._now = now or (lambda: datetime.now(UTC))
        # ponytail: 线程无法被强杀；超时后换池恢复服务，需硬终止时改用进程隔离。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-kb-tool")
        self._executor_lock = Lock()

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        registered = self._registry.get(request.tool_name)
        definition = registered.definition
        started_at = datetime.now(UTC)
        result: ToolExecutionResult | None = None
        error_code: str | None = None
        with observe_span(
            "tool.execute",
            attributes={
                "component": "tool_execution",
                "operation": "tool.execute",
                "tool_name": definition.name,
                "tool_mode": definition.mode.value,
                "task_id": request.task_id,
            },
        ) as span:
            try:
                result = self._execute(request, registered)
                error_code = result.error_code
                category = (
                    "none"
                    if result.status == "succeeded"
                    else "unknown"
                    if result.status == "unknown"
                    else error_category_for(error_code)
                )
                finish_observed_span(
                    span,
                    outcome=result.status,
                    error_category=category,
                    error_code=error_code,
                )
                return result
            except Exception as exc:
                error_code = stable_error_code(exc)
                finish_observed_span(
                    span,
                    outcome="failure",
                    error_category=error_category_for(error_code),
                    error_code=error_code,
                )
                raise
            finally:
                duration_seconds = (datetime.now(UTC) - started_at).total_seconds()
                outcome = result.status if result is not None else "failure"
                category = (
                    "none"
                    if outcome == "succeeded"
                    else "unknown"
                    if outcome == "unknown"
                    else error_category_for(error_code)
                )
                telemetry = current_observability()
                if telemetry is not None:
                    telemetry.record_tool(
                        tool=definition.name,
                        mode=definition.mode.value,
                        outcome=outcome,
                        error_category=category,
                        duration_seconds=duration_seconds,
                    )
                    if definition.mode is ToolMode.WRITE and outcome == "unknown":
                        telemetry.record_security_boundary_violation(kind="write_unknown")
                log_event(
                    "tool.execution.completed",
                    level=logging.ERROR if category != "none" else logging.INFO,
                    component="tool_execution",
                    operation="tool.execute",
                    outcome=outcome,
                    error_category=category,
                    error_code=error_code,
                    duration_ms=duration_seconds * 1_000,
                    task_id=request.task_id,
                    tool_name=definition.name,
                    tool_mode=definition.mode.value,
                )

    def _execute(
        self, request: ToolExecutionRequest, registered: RegisteredTool
    ) -> ToolExecutionResult:
        self._approvals.expire_due(tenant_id=request.auth.tenant_id)
        definition = registered.definition
        self._registry.validate_input(registered, request.arguments)
        arguments_hash = arguments_fingerprint(request.arguments)
        principal_hash = principal_fingerprint(request.auth)
        # 首次执行也从任务行锁内重建当前身份，不能只信任 invoke 开始时快照。
        current_auth = self._current_execution_auth(request, principal_hash=principal_hash)
        permissions = frozenset(
            permission for permission in Permission if current_auth.has_permission(permission)
        )
        self._registry.authorize(
            request.tool_name,
            permissions=permissions,
            approval=request.approval_id,
        )
        if definition.mode is ToolMode.WRITE and not request.idempotency_key:
            raise ToolExecutionError("idempotency_key_required")
        if definition.mode is ToolMode.WRITE and (
            request.plan_revision is None
            or request.plan_digest is None
            or request.evidence_digest is None
        ):
            raise ToolExecutionError("tool_plan_binding_required")

        if definition.mode is ToolMode.WRITE:
            try:
                self._preflight_write(request)
            except ToolExecutionError as exc:
                self._reject_stale_approval(
                    tenant_id=request.auth.tenant_id,
                    approval_id=request.approval_id,
                    reason=exc.code,
                )
                raise

        duplicate = self._existing_idempotent_result(
            request, registered, arguments_hash, principal_hash
        )
        if duplicate is not None:
            return duplicate

        tool_call_id = uuid4().hex
        raced = self._persist_running(
            request,
            registered,
            tool_call_id=tool_call_id,
            arguments_hash=arguments_hash,
            principal_hash=principal_hash,
        )
        if raced is not None:
            return raced
        context = ToolExecutionContext(
            auth=request.auth,
            task_id=request.task_id,
            tool_call_id=tool_call_id,
            plan_revision=request.plan_revision,
            plan_digest=request.plan_digest,
            evidence_digest=request.evidence_digest,
            idempotency_key=request.idempotency_key,
        )
        with self._executor_lock:
            executor = self._executor
            adapter_context = copy_context()
            future = executor.submit(
                adapter_context.run,
                self._execute_adapter,
                registered,
                request.arguments,
                context,
            )
        try:
            result = future.result(timeout=definition.timeout_seconds)
        except FutureTimeout:
            if not future.cancel():
                with self._executor_lock:
                    if self._executor is executor:
                        self._executor = ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="rag-kb-tool",
                        )
                executor.shutdown(wait=False, cancel_futures=True)
            return self._complete_observed(
                request,
                registered,
                tool_call_id,
                status="unknown" if definition.mode is ToolMode.WRITE else "failed",
                error_code="tool_timeout",
            )
        except Exception as exc:
            logger.error("受控工具执行失败 error_type=%s", type(exc).__name__)
            status = "unknown" if definition.mode is ToolMode.WRITE else "failed"
            code = getattr(exc, "code", None)
            if not isinstance(code, str) or not code:
                code = f"tool_execution_failed_{type(exc).__name__.lower()}"
            return self._complete_observed(
                request, registered, tool_call_id, status=status, error_code=code
            )
        verify_error: str | None = None
        with observe_span(
            "tool.verify",
            attributes={
                "component": "tool_execution",
                "operation": "tool.verify",
                "tool_name": definition.name,
                "tool_mode": definition.mode.value,
                "task_id": request.task_id,
            },
        ) as verify_span:
            try:
                self._registry.validate_output(registered, result)
                verified = registered.adapter.verify(result, context)
            except Exception as exc:
                verified = False
                verify_error = stable_error_code(exc)
            finish_observed_span(
                verify_span,
                outcome="succeeded" if verified else "failure",
                error_category="none" if verified else error_category_for(verify_error),
                error_code=verify_error,
            )
        if not verified:
            status = "unknown" if definition.mode is ToolMode.WRITE else "failed"
            return self._complete_observed(
                request,
                registered,
                tool_call_id,
                status=status,
                error_code=verify_error or "tool_result_unverified",
            )
        result_hash = canonical_fingerprint("tool_result", result)
        return self._complete_observed(
            request,
            registered,
            tool_call_id,
            status="succeeded",
            result_hash=result_hash,
            result=result,
        )

    @staticmethod
    def _execute_adapter(
        registered: RegisteredTool,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> dict[str, object]:
        definition = registered.definition
        with observe_span(
            "tool.adapter",
            attributes={
                "component": "tool_execution",
                "operation": "tool.adapter",
                "tool_name": definition.name,
                "tool_mode": definition.mode.value,
                "task_id": context.task_id,
            },
        ):
            return registered.adapter.execute(arguments, context)

    def close(self) -> None:
        with self._executor_lock:
            self._executor.shutdown(wait=False, cancel_futures=True)

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

    def _existing_idempotent_result(
        self,
        request: ToolExecutionRequest,
        registered: RegisteredTool,
        arguments_hash: str,
        principal_hash: str,
    ) -> ToolExecutionResult | None:
        if request.idempotency_key is None:
            return None
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._require_current_execution_context(
                work.session, request, principal_hash=principal_hash
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
            self._validate_idempotency_binding(
                row, request, registered, arguments_hash, principal_hash
            )
            now = self._execution_now(work.session)
            expires_at = _aware_datetime(row.result_expires_at)
            if row.status == "succeeded" and (
                row.result_purged_at is not None
                or expires_at is not None and expires_at <= now
            ):
                return ToolExecutionResult(
                    tool_call_id=row.tool_call_id,
                    status="failed",
                    result_hash=row.result_hash,
                    error_code="tool_result_expired",
                    duplicate=True,
                )
            if row.status != "succeeded":
                return ToolExecutionResult(
                    tool_call_id=row.tool_call_id,
                    status=row.status if row.status != "running" else (
                        "failed" if row.mode == "read" else "unknown"
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
            if not row.result_ref or self._tool_results is None:
                return ToolExecutionResult(
                    tool_call_id=row.tool_call_id,
                    status="failed" if row.mode == "read" else "unknown",
                    result_hash=row.result_hash,
                    error_code="tool_result_readback_failed",
                    duplicate=True,
                )
            tool_call_id = row.tool_call_id
            mode = row.mode
            result_ref = row.result_ref
            result_hash = row.result_hash
            historical_principal_hash = row.principal_hash

        try:
            parsed = json.loads(self._tool_results.read(result_ref))
            if (
                not isinstance(parsed, dict)
                or canonical_fingerprint("tool_result", parsed) != result_hash
            ):
                parsed = None
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None

        # 对象读取不受数据库事务保护；返回正文前必须再次重过租约和当前身份门。
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
                    ToolCallRow.tool_call_id == tool_call_id,
                    ToolCallRow.tenant_id == request.auth.tenant_id,
                    ToolCallRow.task_id == request.task_id,
                )
                .with_for_update()
            )
            now = self._execution_now(work.session)
            if (
                latest is None
                or latest.status != "succeeded"
                or latest.result_ref != result_ref
                or latest.result_hash != result_hash
                or latest.result_purged_at is not None
                or (
                    _aware_datetime(latest.result_expires_at) is not None
                    and _aware_datetime(latest.result_expires_at) <= now
                )
            ):
                return ToolExecutionResult(
                    tool_call_id=tool_call_id,
                    status="failed" if mode == "read" else "unknown",
                    result_hash=result_hash,
                    error_code="tool_result_expired",
                    duplicate=True,
                )
        if parsed is None:
            # 不可解引用或校验失败的成功结果不得凭 hash 伪装正文。
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                status="failed" if mode == "read" else "unknown",
                result_hash=result_hash,
                error_code="tool_result_readback_failed",
                duplicate=True,
            )
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status="succeeded",
            result_hash=result_hash,
            result_ref=result_ref,
            result=parsed,
            duplicate=True,
        )

    def _require_active_lease(
        self, session, request: ToolExecutionRequest
    ) -> AgentTaskRow:
        """租约 fencing 共享校验：owner/generation/expiry + cancelled + deadline。"""
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
            raise ToolExecutionError("tool_task_missing")
        now = self._execution_now(session)
        lease_expires_at = task.lease_expires_at
        if lease_expires_at is not None and lease_expires_at.tzinfo is None:
            lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
        if (
            task.lease_owner != request.worker_id
            or task.lease_generation != request.lease_generation
            or lease_expires_at is None
            or lease_expires_at <= now
        ):
            raise ToolExecutionError("tool_task_lease_stale")
        deadline = task.deadline
        if deadline is not None and deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        if (
            task.cancelled
            or task.status not in {"planning", "running"}
            or deadline is not None and deadline <= now
        ):
            raise ToolExecutionError("tool_task_not_executable")
        return task

    def _current_execution_auth(
        self, request: ToolExecutionRequest, *, principal_hash: str
    ) -> AuthContext:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            _, current_auth = self._require_current_execution_context(
                work.session, request, principal_hash=principal_hash
            )
            return current_auth

    def _require_current_execution_context(
        self,
        session,
        request: ToolExecutionRequest,
        *,
        principal_hash: str,
        historical_principal_hash: str | None = None,
    ) -> tuple[AgentTaskRow, AuthContext]:
        """共享任务行锁门：当前租约、当前身份和历史主体绑定必须同时成立。"""
        task = self._require_active_lease(session, request)
        try:
            current_auth = current_task_auth(
                session,
                task=task,
                auth_context_ref=task.auth_context_ref,
                request_id=request.auth.request_id,
            )
        except TaskExecutionContextError as exc:
            raise ToolExecutionError(getattr(exc, "code", str(exc))) from exc
        current_principal_hash = principal_fingerprint(current_auth)
        if (
            current_principal_hash != principal_hash
            or historical_principal_hash is not None
            and current_principal_hash != historical_principal_hash
        ):
            raise ToolExecutionError("task_auth_context_changed")
        return task, current_auth

    @staticmethod
    def _validate_idempotency_binding(
        row: ToolCallRow,
        request: ToolExecutionRequest,
        registered: RegisteredTool,
        arguments_hash: str,
        principal_hash: str,
    ) -> None:
        if row.principal_hash != principal_hash:
            raise ToolExecutionError("task_auth_context_changed")
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
            registered.definition.version,
            arguments_hash,
        )
        if binding != expected:
            raise ToolExecutionError("idempotency_key_conflict")

    def _persist_running(
        self,
        request: ToolExecutionRequest,
        registered: RegisteredTool,
        *,
        tool_call_id: str,
        arguments_hash: str,
        principal_hash: str,
    ) -> ToolExecutionResult | None:
        definition = registered.definition
        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                task, current_auth = self._require_current_execution_context(
                    work.session, request, principal_hash=principal_hash
                )
                plan = None
                if definition.mode is ToolMode.WRITE:
                    plan = work.session.scalar(
                        select(PlanRevisionRow).where(
                            PlanRevisionRow.tenant_id == request.auth.tenant_id,
                            PlanRevisionRow.task_id == request.task_id,
                            PlanRevisionRow.revision == request.plan_revision,
                            PlanRevisionRow.plan_digest == request.plan_digest,
                            PlanRevisionRow.evidence_digest == request.evidence_digest,
                        ).with_for_update()
                    )
                if definition.mode is ToolMode.WRITE and plan is None:
                    raise ToolExecutionError("tool_plan_binding_invalid")
                if definition.mode is ToolMode.WRITE:
                    if task.task_mode != "controlled_action":
                        raise ToolExecutionError("write_task_mode_denied")
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
                        raise ToolExecutionError(getattr(exc, "code", str(exc))) from exc
                    if (
                        requested_evidence != plan_evidence
                        or principal_fingerprint(current_auth) != principal_hash
                    ):
                        raise ToolExecutionError("tool_current_binding_invalid")
                approval: ApprovalRow | None = None
                if definition.mode is ToolMode.WRITE:
                    if request.approval_id is None:
                        raise ToolExecutionError("approval_required")
                    approval = self._approvals.validate_in_session(
                        work.session,
                        auth=request.auth,
                        approval_id=request.approval_id,
                        task_id=request.task_id,
                        plan_revision=request.plan_revision,
                        plan_digest=request.plan_digest,
                        evidence_digest=request.evidence_digest,
                        evidence_ids=request.evidence_ids,
                        tool_name=request.tool_name,
                        tool_version=definition.version,
                        arguments_hash=arguments_hash,
                    )
                now = self._execution_now(work.session)
                work.session.add(
                    ToolCallRow(
                        tool_call_id=tool_call_id,
                        task_id=request.task_id,
                        tenant_id=request.auth.tenant_id,
                        approval_id=approval.approval_id if approval else None,
                        plan_revision=request.plan_revision,
                        plan_digest=request.plan_digest,
                        evidence_digest=request.evidence_digest,
                        tool_name=request.tool_name,
                        tool_version=definition.version,
                        arguments_hash=arguments_hash,
                        principal_hash=principal_hash,
                        mode=definition.mode.value,
                        idempotency_key=request.idempotency_key,
                        risk_level=definition.risk_level.value,
                        status="running",
                        created_at=now,
                        execution_deadline=now + timedelta(seconds=definition.timeout_seconds),
                    )
                )
                if approval is not None:
                    # 批准授权的是一次写尝试，而不是一次“成功”。意图落库即消费，
                    # timeout/UNKNOWN 后也绝不重新开放同一批准。
                    approval.status = "executed"
            return None
        except IntegrityError:
            duplicate = self._existing_idempotent_result(
                request, registered, arguments_hash, principal_hash
            )
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
                    raise ToolExecutionError("approval_already_consumed")
            raise

    def _preflight_write(self, request: ToolExecutionRequest) -> None:
        """幂等结果回放也必须通过当前授权；原子门仍在 `_persist_running` 内重查。"""

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
                raise ToolExecutionError("write_task_mode_denied")
            try:
                current_auth, plan_evidence = self._evidence_validator.validate_plan_in_session(
                    session,
                    task=task,
                    plan=plan,
                    request_id=request.auth.request_id,
                    require_non_empty=True,
                )
                requested_evidence = canonical_evidence_snapshot(
                    request.evidence_ids,
                    require_non_empty=True,
                )
            except (TaskExecutionContextError, ValueError) as exc:
                raise ToolExecutionError(getattr(exc, "code", str(exc))) from exc
            if (
                requested_evidence != plan_evidence
                or principal_fingerprint(current_auth) != principal_fingerprint(request.auth)
            ):
                raise ToolExecutionError("tool_current_binding_invalid")

    def _reject_stale_approval(
        self,
        *,
        tenant_id: str,
        approval_id: str | None,
        reason: str,
    ) -> None:
        """审批失效的自动收敛：服务端固定 actor + 稳定 code 的完整审计。

        带 checkpoint 绑定的行受 ck_approvals_rejected_audit_complete 约束，
        rejected_by/rejected_at/rejection_reason 缺一不可（批次 2 修复 C-1）。
        """
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

    def _complete_observed(
        self,
        request: ToolExecutionRequest,
        registered: RegisteredTool,
        tool_call_id: str,
        **kwargs: object,
    ) -> ToolExecutionResult:
        definition = registered.definition
        with observe_span(
            "tool.complete",
            attributes={
                "component": "tool_execution",
                "operation": "tool.complete",
                "tool_name": definition.name,
                "tool_mode": definition.mode.value,
                "task_id": request.task_id,
            },
        ) as span:
            result = self._complete(request, tool_call_id, **kwargs)
            category = (
                "none"
                if result.status == "succeeded"
                else "unknown"
                if result.status == "unknown"
                else error_category_for(result.error_code)
            )
            finish_observed_span(
                span,
                outcome=result.status,
                error_category=category,
                error_code=result.error_code,
            )
            return result

    def _complete(
        self,
        request: ToolExecutionRequest,
        tool_call_id: str,
        *,
        status: str,
        error_code: str | None = None,
        result_hash: str | None = None,
        result: dict[str, object] | None = None,
    ) -> ToolExecutionResult:
        principal_hash = principal_fingerprint(request.auth)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._require_current_execution_context(
                work.session, request, principal_hash=principal_hash
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
                raise ToolExecutionError("tool_call_state_conflict")
            if row.principal_hash != principal_hash:
                raise ToolExecutionError("task_auth_context_changed")
            task_id = row.task_id
            mode = row.mode

        final_status = status
        final_error = error_code
        final_hash = result_hash
        result_ref: str | None = None
        result_size: int | None = None
        result_expires_at = None
        result_to_discard: str | None = None
        wrote_result = False
        if status == "succeeded" and (result is None or result_hash is None):
            final_status = "failed" if mode == "read" else "unknown"
            final_error = "tool_result_missing"
            final_hash = None
        elif status == "succeeded" and self._tool_results is None:
            final_status = "failed" if mode == "read" else "unknown"
            final_error = "tool_result_store_unavailable"
            final_hash = None
        elif status == "succeeded" and result is not None:
            assert self._tool_results is not None
            payload = json.dumps(
                result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            result_ref = f"tool-results/{task_id}/{tool_call_id}.json"
            try:
                self._tool_results.put_exclusive(result_ref, payload)
                wrote_result = True
            except ObjectConflictError:
                try:
                    existing = json.loads(self._tool_results.read(result_ref))
                except (OSError, ValueError, json.JSONDecodeError):
                    existing = None
                if (
                    not isinstance(existing, dict)
                    or canonical_fingerprint("tool_result", existing) != result_hash
                ):
                    final_status = "failed" if mode == "read" else "unknown"
                    final_error = "tool_result_persist_conflict"
                    final_hash = None
                    result_ref = None
            except (OSError, ValueError):
                # ponytail: READ 结果无法持久化时可安全失败；WRITE 副作用已发生，
                # 因而保持 unknown 交由人工收敛（计划 §6.11）。
                logger.error("工具结果对象持久化失败 tool_call_id=%s", tool_call_id)
                final_status = "failed" if mode == "read" else "unknown"
                final_error = "tool_result_persist_failed"
                final_hash = None
                result_ref = None
            else:
                result_size = len(payload)
            if result_ref is not None and result_size is None:
                result_size = len(payload)

        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                task, _ = self._require_current_execution_context(
                    work.session, request, principal_hash=principal_hash
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
                    raise ToolExecutionError("tool_call_state_conflict")
                if row.principal_hash != principal_hash:
                    raise ToolExecutionError("task_auth_context_changed")
                now = self._execution_now(work.session)
                execution_deadline = _aware_datetime(row.execution_deadline)
                if (
                    result_ref is not None
                    and execution_deadline is not None
                    and execution_deadline <= now
                ):
                    result_to_discard = result_ref
                    final_status = "failed" if row.mode == "read" else "unknown"
                    final_error = "execution_deadline_expired"
                    final_hash = None
                    result_ref = None
                    result_size = None
                if result_ref is not None:
                    result_expires_at = now + timedelta(
                        seconds=self._tool_result_ttl_seconds
                    )
                row.status = final_status
                row.result_hash = final_hash
                row.error_code = final_error
                row.result_ref = result_ref
                row.result_size = result_size
                row.result_expires_at = result_expires_at
                row.completed_at = now
                row.verified_at = now if final_status == "succeeded" else None
                if row.mode == "write" and final_status == "unknown":
                    expires_at = None
                    if task.task_mode == "controlled_action":
                        expires_at = now + timedelta(
                            days=_default_task_content_retention_days()
                        )
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
                elif row.mode == "write" and final_status == "succeeded" and result is not None:
                    # 业务引用只绑定稳定 artifact ID，不写对象存储路径。
                    artifact_id = result.get("artifact_id")
                    if isinstance(artifact_id, str) and artifact_id not in task.artifact_refs:
                        task.artifact_refs = [*task.artifact_refs, artifact_id]
                        task.state_version += 1
        except Exception:
            cleanup_ref = result_to_discard or result_ref
            if wrote_result and cleanup_ref is not None:
                self._delete_unbound_result(cleanup_ref, tool_call_id)
            raise
        if wrote_result and result_to_discard is not None:
            self._delete_unbound_result(result_to_discard, tool_call_id)
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status=final_status,
            result_hash=final_hash,
            result_ref=result_ref,
            result=result if final_status == "succeeded" else None,
            error_code=final_error,
        )

    def _delete_unbound_result(self, result_ref: str, tool_call_id: str) -> None:
        if self._result_is_bound(result_ref, tool_call_id):
            return
        assert self._tool_results is not None
        try:
            self._tool_results.delete_exact(result_ref)
        except OSError:
            logger.error("未绑定工具结果补偿删除失败 tool_call_id=%s", tool_call_id)

    def _result_is_bound(self, result_ref: str, tool_call_id: str) -> bool:
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

    def expire_stale_running(self, *, limit: int = 100) -> int:
        """崩溃恢复清理器只收敛状态，绝不重放工具。"""
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._execution_clock_expression(work.session)
            rows = tuple(
                work.session.scalars(
                    select(ToolCallRow)
                    .where(
                        ToolCallRow.status == "running",
                        ToolCallRow.execution_deadline <= clock,
                    )
                    .order_by(ToolCallRow.execution_deadline)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            now = self._execution_now(work.session)
            for row in rows:
                # READ 崩溃可安全收敛为 failed；WRITE 副作用未知必须保持 unknown。
                row.status = "failed" if row.mode == "read" else "unknown"
                row.error_code = "execution_deadline_expired"
                row.completed_at = now
                if row.mode == "write":
                    expires_at = None
                    task = work.session.scalar(
                        select(AgentTaskRow)
                        .where(
                            AgentTaskRow.tenant_id == row.tenant_id,
                            AgentTaskRow.task_id == row.task_id,
                        )
                        .with_for_update()
                    )
                    if task is not None and task.task_mode == "controlled_action":
                        expires_at = now + timedelta(
                            days=_default_task_content_retention_days()
                        )
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
                    if task is not None:
                        task.status = "unknown"
                        task.last_error_code = "execution_deadline_expired"
                        task.lease_owner = None
                        task.lease_expires_at = None
                        task.state_version += 1
            return len(rows)
