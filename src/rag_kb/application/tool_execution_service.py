"""受控工具执行：先持久化意图，再执行，并对不确定副作用关闭重试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from threading import Lock
from uuid import uuid4

from sqlalchemy import select
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
    ToolRegistryError,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.domain.tasks import TaskBudgets
from rag_kb.domain.tools import ToolMode
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    ArtifactRow,
    PlanRevisionRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskEvidencePort,
    TaskExecutionContextError,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ToolExecutionRequest:
    auth: AuthContext
    task_id: str
    plan_revision: int
    plan_digest: str
    evidence_digest: str
    evidence_ids: tuple[str, ...]
    tool_name: str
    arguments: dict[str, object]
    worker_id: str
    lease_generation: int
    idempotency_key: str | None = None
    approval_id: str | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    status: str
    result_hash: str | None = None
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
        now=None,
    ) -> None:
        self._database = database
        self._registry = registry
        self._approvals = approval_service
        self._evidence_validator = evidence_validator or DatabaseTaskEvidencePort(database)
        self._now = now or (lambda: datetime.now(UTC))
        # ponytail: 线程无法被强杀；超时后换池恢复服务，需硬终止时改用进程隔离。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-kb-tool")
        self._executor_lock = Lock()

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        self._approvals.expire_due(tenant_id=request.auth.tenant_id)
        registered = self._registry.get(request.tool_name)
        definition = registered.definition
        self._registry.validate_input(registered, request.arguments)
        permissions = frozenset(
            permission for permission in Permission if request.auth.has_permission(permission)
        )
        self._registry.authorize(
            request.tool_name,
            permissions=permissions,
            approval=request.approval_id,
        )
        if definition.mode is ToolMode.WRITE and not request.idempotency_key:
            raise ToolExecutionError("idempotency_key_required")

        if definition.mode is ToolMode.WRITE:
            try:
                self._preflight_write(request)
            except ToolExecutionError:
                self._reject_stale_approval(
                    tenant_id=request.auth.tenant_id,
                    approval_id=request.approval_id,
                )
                raise

        arguments_hash = arguments_fingerprint(request.arguments)
        principal_hash = principal_fingerprint(request.auth)
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
            future = executor.submit(registered.adapter.execute, request.arguments, context)
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
            return self._complete(tool_call_id, status="unknown", error_code="tool_timeout")
        except BaseException as exc:
            logger.exception("受控工具执行失败；调用状态将按副作用风险关闭")
            status = "unknown" if definition.mode is ToolMode.WRITE else "failed"
            code = (
                exc.code
                if isinstance(exc, ToolRegistryError)
                else f"tool_execution_failed_{type(exc).__name__.lower()}"
            )
            return self._complete(tool_call_id, status=status, error_code=code)
        try:
            self._registry.validate_output(registered, result)
            verified = registered.adapter.verify(result, context)
        except BaseException:
            verified = False
        if not verified:
            status = "unknown" if definition.mode is ToolMode.WRITE else "failed"
            return self._complete(tool_call_id, status=status, error_code="tool_result_unverified")
        result_hash = canonical_fingerprint("tool_result", result)
        return self._complete(
            tool_call_id,
            status="succeeded",
            result_hash=result_hash,
            result=result,
        )

    def _existing_idempotent_result(
        self,
        request: ToolExecutionRequest,
        registered: RegisteredTool,
        arguments_hash: str,
        principal_hash: str,
    ) -> ToolExecutionResult | None:
        if request.idempotency_key is None:
            return None
        with self._database.sessions() as session:
            row = session.scalar(
                select(ToolCallRow).where(
                    ToolCallRow.tenant_id == request.auth.tenant_id,
                    ToolCallRow.task_id == request.task_id,
                    ToolCallRow.tool_name == request.tool_name,
                    ToolCallRow.idempotency_key == request.idempotency_key,
                )
            )
            if row is None:
                return None
            binding = (
                row.plan_revision,
                row.plan_digest,
                row.evidence_digest,
                row.tool_version,
                row.arguments_hash,
                row.principal_hash,
            )
            expected = (
                request.plan_revision,
                request.plan_digest,
                request.evidence_digest,
                registered.definition.version,
                arguments_hash,
                principal_hash,
            )
            if binding != expected:
                raise ToolExecutionError("idempotency_key_conflict")
            return ToolExecutionResult(
                tool_call_id=row.tool_call_id,
                status="unknown" if row.status == "running" else row.status,
                result_hash=row.result_hash,
                error_code="previous_execution_in_progress" if row.status == "running" else row.error_code,
                duplicate=True,
            )

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
                task = work.session.scalar(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.tenant_id == request.auth.tenant_id,
                        AgentTaskRow.task_id == request.task_id,
                        AgentTaskRow.user_id == request.auth.user_id,
                    )
                    .with_for_update()
                )
                plan = work.session.scalar(
                    select(PlanRevisionRow).where(
                        PlanRevisionRow.tenant_id == request.auth.tenant_id,
                        PlanRevisionRow.task_id == request.task_id,
                        PlanRevisionRow.revision == request.plan_revision,
                        PlanRevisionRow.plan_digest == request.plan_digest,
                        PlanRevisionRow.evidence_digest == request.evidence_digest,
                    ).with_for_update()
                )
                if task is None or plan is None or task.current_plan_revision != request.plan_revision:
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
                lease_expires_at = task.lease_expires_at
                if lease_expires_at is not None and lease_expires_at.tzinfo is None:
                    lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
                if (
                    task.lease_owner != request.worker_id
                    or task.lease_generation != request.lease_generation
                    or lease_expires_at is None
                    or lease_expires_at <= self._now()
                ):
                    raise ToolExecutionError("tool_task_lease_stale")
                deadline = task.deadline
                if deadline is not None and deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=UTC)
                if task.cancelled or deadline is not None and deadline <= self._now():
                    raise ToolExecutionError("tool_task_not_executable")
                budgets = TaskBudgets.model_validate(task.budgets)
                if task.tool_calls >= budgets.max_tool_calls:
                    raise ToolExecutionError("tool_call_budget_exhausted")
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
                        created_at=self._now(),
                        execution_deadline=self._now()
                        + timedelta(seconds=definition.timeout_seconds),
                    )
                )
                if approval is not None:
                    # 批准授权的是一次写尝试，而不是一次“成功”。意图落库即消费，
                    # timeout/UNKNOWN 后也绝不重新开放同一批准。
                    approval.status = "executed"
                task.tool_calls += 1
                task.state_version += 1
                task.updated_at = self._now()
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
    ) -> None:
        if approval_id is None:
            return
        with UnitOfWork(self._database) as work:
            assert work.session is not None
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

    def _complete(
        self,
        tool_call_id: str,
        *,
        status: str,
        error_code: str | None = None,
        result_hash: str | None = None,
        result: dict[str, object] | None = None,
    ) -> ToolExecutionResult:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(ToolCallRow).where(ToolCallRow.tool_call_id == tool_call_id).with_for_update()
            )
            if row is None or row.status != "running":
                raise ToolExecutionError("tool_call_state_conflict")
            row.status = status
            row.result_hash = result_hash
            row.error_code = error_code
            row.completed_at = self._now()
            row.verified_at = self._now() if status == "succeeded" else None
            task = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == row.tenant_id,
                    AgentTaskRow.task_id == row.task_id,
                )
                .with_for_update()
            )
            if task is None:
                raise ToolExecutionError("tool_task_missing")
            if row.mode == "write" and status == "unknown":
                for artifact in work.session.scalars(
                    select(ArtifactRow).where(
                        ArtifactRow.tenant_id == row.tenant_id,
                        ArtifactRow.task_id == row.task_id,
                        ArtifactRow.tool_call_id == row.tool_call_id,
                        ArtifactRow.status.in_(("creating", "ready")),
                    )
                ):
                    artifact.status = "unknown"
                task.status = "unknown"
                task.current_node = "terminal"
                task.last_error_code = error_code
                task.lease_owner = None
                task.lease_expires_at = None
                task.state_version += 1
            elif row.mode == "write" and status == "succeeded" and result is not None:
                storage_key = result.get("storage_key")
                if isinstance(storage_key, str) and storage_key not in task.artifact_refs:
                    task.artifact_refs = [*task.artifact_refs, storage_key]
                    task.state_version += 1
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status=status,
            result_hash=result_hash,
            result=result,
            error_code=error_code,
        )

    def expire_stale_running(self, *, limit: int = 100) -> int:
        """崩溃恢复清理器只收敛状态，绝不重放工具。"""
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            rows = tuple(
                work.session.scalars(
                    select(ToolCallRow)
                    .where(
                        ToolCallRow.status == "running",
                        ToolCallRow.execution_deadline <= self._now(),
                    )
                    .order_by(ToolCallRow.execution_deadline)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for row in rows:
                row.status = "unknown"
                row.error_code = "execution_deadline_expired"
                row.completed_at = self._now()
                if row.mode == "write":
                    for artifact in work.session.scalars(
                        select(ArtifactRow).where(
                            ArtifactRow.tenant_id == row.tenant_id,
                            ArtifactRow.task_id == row.task_id,
                            ArtifactRow.tool_call_id == row.tool_call_id,
                            ArtifactRow.status.in_(("creating", "ready")),
                        )
                    ):
                        artifact.status = "unknown"
                    task = work.session.scalar(
                        select(AgentTaskRow)
                        .where(
                            AgentTaskRow.tenant_id == row.tenant_id,
                            AgentTaskRow.task_id == row.task_id,
                        )
                        .with_for_update()
                    )
                    if task is not None:
                        task.status = "unknown"
                        task.current_node = "terminal"
                        task.last_error_code = "execution_deadline_expired"
                        task.lease_owner = None
                        task.lease_expires_at = None
                        task.state_version += 1
            return len(rows)
