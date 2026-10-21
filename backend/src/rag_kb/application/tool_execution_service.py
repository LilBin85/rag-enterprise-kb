"""受控工具执行：先持久化意图，再执行，并对不确定副作用关闭重试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from datetime import UTC, datetime
import json
import logging
from threading import Lock
from uuid import uuid4

from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.ports.approvals import (
    arguments_fingerprint,
    principal_fingerprint,
)
from rag_kb.application.ports.object_store import ObjectConflictError, ObjectStorePort
from rag_kb.application.ports.tool_execution import (
    ToolExecutionFacts,
    ToolExecutionLifecycleError,
    ToolExecutionLifecyclePort,
    ToolExecutionRequest,
    ToolExecutionReservation,
    ToolExecutionResult,
)
from rag_kb.application.tool_registry import (
    RegisteredTool,
    ToolExecutionContext,
    ToolRegistry,
)
from rag_kb.domain.identity import Permission
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tools import ToolMode
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


logger = logging.getLogger(__name__)

_WRITE_RENDER_FIELDS = ("artifact_id", "artifact_type", "content_hash")
_RETRYABLE_TOOL_CODES = frozenset(
    {
        "tool_timeout",
        "tool_execution_unavailable",
        "previous_execution_in_progress",
        "tool_result_readback_failed",
        "execution_deadline_expired",
    }
)


def render_write_tool_message(
    tool_name: str,
    result: ToolExecutionResult | None,
    *,
    error_code: str | None = None,
) -> str:
    """渲染 WRITE 工具的低敏结果契约，供 Agent 和恢复编排共同使用。"""
    status = result.status if result is not None else "failed"
    code = error_code if result is None else result.error_code
    payload: dict[str, object] = {
        "tool_call_id": result.tool_call_id if result is not None else None,
        "tool_name": tool_name,
        "status": status,
        "error_code": code,
        "retryable": status == "failed" and code in _RETRYABLE_TOOL_CODES,
    }
    if result is not None and result.result is not None:
        for field in _WRITE_RENDER_FIELDS:
            if field in result.result:
                value = result.result[field]
                payload[field] = value[:2_000] if isinstance(value, str) else value
    if result is not None:
        payload["result_ref"] = result.tool_call_id
        if result.result_hash:
            payload["result_hash"] = result.result_hash
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

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


def _default_tool_result_ttl_seconds() -> int:
    from rag_kb.config import get_settings

    return get_settings().tool_result_ttl_seconds


class ToolExecutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ToolExecutionService:
    def __init__(
        self,
        lifecycle: ToolExecutionLifecyclePort,
        registry: ToolRegistry,
        approval_service: ApprovalService,
        *,
        tool_results: ObjectStorePort | None = None,
        tool_result_ttl_seconds: int | None = None,
    ) -> None:
        self._lifecycle = lifecycle
        self._registry = registry
        self._approvals = approval_service
        self._tool_results = tool_results
        self._tool_result_ttl_seconds = (
            tool_result_ttl_seconds
            if tool_result_ttl_seconds is not None
            else _default_tool_result_ttl_seconds()
        )
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
        try:
            current_auth = self._lifecycle.current_auth(
                request,
                principal_hash=principal_hash,
            )
        except ToolExecutionLifecycleError as exc:
            raise ToolExecutionError(exc.code) from exc
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
                self._lifecycle.preflight_write(
                    request,
                    principal_hash=principal_hash,
                )
            except ToolExecutionLifecycleError as exc:
                raise ToolExecutionError(exc.code) from exc

        facts = ToolExecutionFacts(
            tool_version=definition.version,
            mode=definition.mode,
            risk_level=definition.risk_level.value,
            timeout_seconds=definition.timeout_seconds,
            arguments_hash=arguments_hash,
            principal_hash=principal_hash,
        )
        try:
            reservation = self._lifecycle.reserve_or_replay(
                request,
                facts,
                tool_call_id=uuid4().hex,
            )
        except ToolExecutionLifecycleError as exc:
            raise ToolExecutionError(exc.code) from exc
        if reservation.duplicate:
            return self._read_replay(request, reservation, principal_hash=principal_hash)

        context = ToolExecutionContext(
            auth=request.auth,
            task_id=request.task_id,
            tool_call_id=reservation.tool_call_id,
            plan_revision=request.plan_revision,
            plan_digest=request.plan_digest,
            evidence_digest=request.evidence_digest,
            idempotency_key=request.idempotency_key,
            scope_document_id=request.scope_document_id,
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
                reservation.tool_call_id,
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
                request,
                registered,
                reservation.tool_call_id,
                status=status,
                error_code=code,
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
                reservation.tool_call_id,
                status=status,
                error_code=verify_error or "tool_result_unverified",
            )
        result_hash = canonical_fingerprint("tool_result", result)
        return self._complete_observed(
            request,
            registered,
            reservation.tool_call_id,
            status="succeeded",
            result_hash=result_hash,
            result=result,
        )

    def _read_replay(
        self,
        request: ToolExecutionRequest,
        reservation: ToolExecutionReservation,
        *,
        principal_hash: str,
    ) -> ToolExecutionResult:
        if not reservation.requires_result_read:
            return ToolExecutionResult(
                tool_call_id=reservation.tool_call_id,
                status=reservation.status,
                result_hash=reservation.result_hash,
                result_ref=reservation.result_ref,
                error_code=reservation.error_code,
                duplicate=True,
            )
        if reservation.result_ref is None or self._tool_results is None:
            return ToolExecutionResult(
                tool_call_id=reservation.tool_call_id,
                status="failed" if reservation.mode is ToolMode.READ else "unknown",
                result_hash=reservation.result_hash,
                error_code="tool_result_readback_failed",
                duplicate=True,
            )
        try:
            parsed = json.loads(self._tool_results.read(reservation.result_ref))
            if (
                not isinstance(parsed, dict)
                or canonical_fingerprint("tool_result", parsed) != reservation.result_hash
            ):
                parsed = None
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = None
        try:
            still_current = self._lifecycle.confirm_replay(
                request,
                reservation,
                principal_hash=principal_hash,
            )
        except ToolExecutionLifecycleError as exc:
            raise ToolExecutionError(exc.code) from exc
        if not still_current:
            return ToolExecutionResult(
                tool_call_id=reservation.tool_call_id,
                status="failed" if reservation.mode is ToolMode.READ else "unknown",
                result_hash=reservation.result_hash,
                error_code="tool_result_expired",
                duplicate=True,
            )
        if parsed is None:
            # 不可解引用或校验失败的成功结果不得凭 hash 伪装正文。
            return ToolExecutionResult(
                tool_call_id=reservation.tool_call_id,
                status="failed" if reservation.mode is ToolMode.READ else "unknown",
                result_hash=reservation.result_hash,
                error_code="tool_result_readback_failed",
                duplicate=True,
            )
        return ToolExecutionResult(
            tool_call_id=reservation.tool_call_id,
            status="succeeded",
            result_hash=reservation.result_hash,
            result_ref=reservation.result_ref,
            result=parsed,
            duplicate=True,
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
        try:
            running = self._lifecycle.prepare_completion(
                request,
                tool_call_id=tool_call_id,
                principal_hash=principal_hash,
            )
        except ToolExecutionLifecycleError as exc:
            raise ToolExecutionError(exc.code) from exc

        final_status = status
        final_error = error_code
        final_hash = result_hash
        result_ref: str | None = None
        result_size: int | None = None
        wrote_result = False
        if status == "succeeded" and (result is None or result_hash is None):
            final_status = "failed" if running.mode is ToolMode.READ else "unknown"
            final_error = "tool_result_missing"
            final_hash = None
        elif status == "succeeded" and self._tool_results is None:
            final_status = "failed" if running.mode is ToolMode.READ else "unknown"
            final_error = "tool_result_store_unavailable"
            final_hash = None
        elif status == "succeeded" and result is not None:
            assert self._tool_results is not None
            payload = json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            result_ref = f"tool-results/{request.task_id}/{tool_call_id}.json"
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
                    final_status = "failed" if running.mode is ToolMode.READ else "unknown"
                    final_error = "tool_result_persist_conflict"
                    final_hash = None
                    result_ref = None
            except (OSError, ValueError):
                # ponytail: READ 结果无法持久化时可安全失败；WRITE 副作用已发生，
                # 因而保持 unknown 交由人工收敛（计划 §6.11）。
                logger.error("工具结果对象持久化失败 tool_call_id=%s", tool_call_id)
                final_status = "failed" if running.mode is ToolMode.READ else "unknown"
                final_error = "tool_result_persist_failed"
                final_hash = None
                result_ref = None
            else:
                result_size = len(payload)
            if result_ref is not None and result_size is None:
                result_size = len(payload)

        artifact_id = result.get("artifact_id") if result is not None else None
        if not isinstance(artifact_id, str):
            artifact_id = None
        try:
            completion = self._lifecycle.complete(
                request,
                tool_call_id=tool_call_id,
                principal_hash=principal_hash,
                status=final_status,
                error_code=final_error,
                result_hash=final_hash,
                result_ref=result_ref,
                result_size=result_size,
                artifact_id=artifact_id,
                tool_result_ttl_seconds=self._tool_result_ttl_seconds,
            )
        except Exception as exc:
            if wrote_result and result_ref is not None:
                self._delete_unbound_result(result_ref, tool_call_id)
            if isinstance(exc, ToolExecutionLifecycleError):
                raise ToolExecutionError(exc.code) from exc
            raise
        if wrote_result and completion.result_to_discard is not None:
            self._delete_unbound_result(completion.result_to_discard, tool_call_id)
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status=completion.status,
            result_hash=completion.result_hash,
            result_ref=completion.result_ref,
            result=result if completion.status == "succeeded" else None,
            error_code=completion.error_code,
        )

    def _delete_unbound_result(self, result_ref: str, tool_call_id: str) -> None:
        if self._lifecycle.is_result_bound(
            tool_call_id=tool_call_id,
            result_ref=result_ref,
        ):
            return
        assert self._tool_results is not None
        try:
            self._tool_results.delete_exact(result_ref)
        except OSError:
            logger.error("未绑定工具结果补偿删除失败 tool_call_id=%s", tool_call_id)

    def expire_stale_running(self, *, limit: int = 100) -> int:
        """崩溃恢复清理器只收敛状态，绝不重放工具。"""
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
        try:
            return self._lifecycle.expire_stale_running(limit=limit)
        except ToolExecutionLifecycleError as exc:
            raise ToolExecutionError(exc.code) from exc
