"""受控工具执行：先持久化意图，再执行，并对不确定副作用关闭重试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
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


logger = logging.getLogger(__name__)

# 工具执行中的致命错误码：失租/身份失效的旧 Worker 必须立即终止，
# 禁止渲染 ToolMessage 继续模型循环或回写 checkpoint。
_FATAL_TOOL_CODES = {"tool_task_lease_stale", "task_auth_context_changed"}


def _default_tool_result_ttl_seconds() -> int:
    from rag_kb.config import get_settings

    return get_settings().tool_result_ttl_seconds


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
        if definition.mode is ToolMode.WRITE and (
            request.plan_revision is None
            or request.plan_digest is None
            or request.evidence_digest is None
        ):
            raise ToolExecutionError("tool_plan_binding_required")

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
        # 幂等回读之前先过租约门：失租 Worker 连旧结果都不得回放。
        with self._database.sessions() as session:
            self._require_active_lease(session, request)
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
            return self._complete(
                tool_call_id,
                status="unknown" if definition.mode is ToolMode.WRITE else "failed",
                error_code="tool_timeout",
            )
        except BaseException as exc:
            logger.error("受控工具执行失败 error_type=%s", type(exc).__name__)
            status = "unknown" if definition.mode is ToolMode.WRITE else "failed"
            code = getattr(exc, "code", None)
            if not isinstance(code, str) or not code:
                code = f"tool_execution_failed_{type(exc).__name__.lower()}"
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

    def close(self) -> None:
        with self._executor_lock:
            self._executor.shutdown(wait=False, cancel_futures=True)

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
            expires_at = row.result_expires_at
            if expires_at is not None and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if row.status == "succeeded" and expires_at is not None and expires_at <= self._now():
                # TTL 已过：正文视为不存在，绝不凭 hash 回放（不可重试）。
                return ToolExecutionResult(
                    tool_call_id=row.tool_call_id,
                    status="failed",
                    result_hash=row.result_hash,
                    error_code="tool_result_expired",
                    duplicate=True,
                )
            readback: dict[str, object] | None = None
            if row.status == "succeeded":
                if not row.result_ref or self._tool_results is None:
                    return ToolExecutionResult(
                        tool_call_id=row.tool_call_id,
                        status="failed" if row.mode == "read" else "unknown",
                        result_hash=row.result_hash,
                        error_code="tool_result_readback_failed",
                        duplicate=True,
                    )
                try:
                    parsed = json.loads(self._tool_results.read(row.result_ref))
                    if (
                        not isinstance(parsed, dict)
                        or canonical_fingerprint("tool_result", parsed) != row.result_hash
                    ):
                        parsed = None
                except (OSError, ValueError, json.JSONDecodeError):
                    parsed = None
                if parsed is None:
                    # 不可解引用或校验失败的成功结果不得凭 hash 伪装正文。
                    return ToolExecutionResult(
                        tool_call_id=row.tool_call_id,
                        status="failed",
                        result_hash=row.result_hash,
                        error_code="tool_result_readback_failed",
                        duplicate=True,
                    )
                readback = parsed
            return ToolExecutionResult(
                tool_call_id=row.tool_call_id,
                status=row.status if row.status != "running" else (
                    "failed" if row.mode == "read" else "unknown"
                ),
                result_hash=row.result_hash,
                result_ref=row.result_ref,
                result=readback,
                error_code=(
                    "previous_execution_in_progress"
                    if row.status == "running"
                    else row.error_code
                ),
                duplicate=True,
            )

    def _require_active_lease(
        self, session, request: ToolExecutionRequest
    ) -> AgentTaskRow | None:
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
            return None
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
        return task

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
                task = self._require_active_lease(work.session, request)
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
                if task is None or (
                    definition.mode is ToolMode.WRITE and plan is None
                ):
                    raise ToolExecutionError("tool_plan_binding_invalid")
                # 逐工具身份重解析：行锁内重建当前身份，角色/组撤销立即生效。
                try:
                    current_auth = current_task_auth(
                        work.session,
                        task=task,
                        auth_context_ref=task.auth_context_ref,
                        request_id=request.auth.request_id,
                    )
                except TaskExecutionContextError as exc:
                    raise ToolExecutionError(getattr(exc, "code", str(exc))) from exc
                if principal_fingerprint(current_auth) != principal_hash:
                    raise ToolExecutionError("task_auth_context_changed")
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
            final_status = status
            final_error = error_code
            final_hash = result_hash
            result_ref: str | None = None
            result_size: int | None = None
            result_expires_at = None
            if status == "succeeded" and result is None:
                final_status = "failed" if row.mode == "read" else "unknown"
                final_error = "tool_result_missing"
                final_hash = None
            elif status == "succeeded" and self._tool_results is None:
                final_status = "failed" if row.mode == "read" else "unknown"
                final_error = "tool_result_store_unavailable"
                final_hash = None
            elif status == "succeeded" and result is not None:
                assert self._tool_results is not None
                payload = json.dumps(
                    result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                result_ref = f"tool-results/{row.task_id}/{tool_call_id}.json"
                try:
                    self._tool_results.put_exclusive(result_ref, payload)
                except ObjectConflictError:
                    try:
                        existing = json.loads(self._tool_results.read(result_ref))
                    except (OSError, ValueError, json.JSONDecodeError):
                        existing = None
                    if (
                        not isinstance(existing, dict)
                        or canonical_fingerprint("tool_result", existing) != result_hash
                    ):
                        final_status = "failed" if row.mode == "read" else "unknown"
                        final_error = "tool_result_persist_conflict"
                        final_hash = None
                        result_ref = None
                    else:
                        result_size = len(payload)
                        result_expires_at = self._now() + timedelta(
                            seconds=self._tool_result_ttl_seconds
                        )
                except (OSError, ValueError):
                    # ponytail: READ 结果无法持久化时可安全失败；WRITE 副作用已发生，
                    # 因而保持 unknown 交由人工收敛（计划 §6.11）。
                    logger.error("工具结果对象持久化失败 tool_call_id=%s", tool_call_id)
                    final_status = "failed" if row.mode == "read" else "unknown"
                    final_error = "tool_result_persist_failed"
                    final_hash = None
                    result_ref = None
                else:
                    result_size = len(payload)
                    result_expires_at = self._now() + timedelta(
                        seconds=self._tool_result_ttl_seconds
                    )
            now = self._now()
            row.status = final_status
            row.result_hash = final_hash
            row.error_code = final_error
            row.result_ref = result_ref
            row.result_size = result_size
            row.result_expires_at = result_expires_at
            row.completed_at = now
            row.verified_at = now if final_status == "succeeded" else None
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
            if row.mode == "write" and final_status == "unknown":
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
                task.last_error_code = final_error
                task.lease_owner = None
                task.lease_expires_at = None
                task.state_version += 1
            elif row.mode == "write" and final_status == "succeeded" and result is not None:
                storage_key = result.get("storage_key")
                if isinstance(storage_key, str) and storage_key not in task.artifact_refs:
                    task.artifact_refs = [*task.artifact_refs, storage_key]
                    task.state_version += 1
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status=final_status,
            result_hash=final_hash,
            result_ref=result_ref,
            result=result if final_status == "succeeded" else None,
            error_code=final_error,
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
                # READ 崩溃可安全收敛为 failed；WRITE 副作用未知必须保持 unknown。
                row.status = "failed" if row.mode == "read" else "unknown"
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
                        task.last_error_code = "execution_deadline_expired"
                        task.lease_owner = None
                        task.lease_expires_at = None
                        task.state_version += 1
            return len(rows)

    def purge_expired_tool_results(self, *, limit: int = 100) -> int:
        """删除过期工具结果对象并置 result_ref=NULL；hash 保留供审计。

        幂等可重扫：已清理行的 result_ref 为空，天然不再命中。
        """
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
        if self._tool_results is None:
            return 0
        purged = 0
        while purged < limit:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                rows = tuple(
                    work.session.scalars(
                        select(ToolCallRow)
                        .where(
                            ToolCallRow.result_ref.is_not(None),
                            ToolCallRow.result_expires_at <= self._now(),
                        )
                        .order_by(ToolCallRow.result_expires_at)
                        .limit(min(25, limit - purged))
                        .with_for_update(skip_locked=True)
                    )
                )
                if not rows:
                    break
                for row in rows:
                    try:
                        self._tool_results.delete_exact(row.result_ref)
                    except OSError:
                        # 对象缺失不阻塞收敛；引用仍会被置空，避免永久重扫。
                        logger.error(
                            "工具结果对象删除失败 tool_call_id=%s", row.tool_call_id
                        )
                    # succeeded 行必须携带完整结果引用（CHECK），TTL 清理后
                    # 按模式收敛终态：READ 失败、WRITE 保持 unknown。
                    row.status = "unknown" if row.mode == "write" else "failed"
                    row.error_code = "tool_result_expired"
                    row.result_ref = None
                    row.result_size = None
                    row.result_expires_at = None
                    purged += 1
        return purged
