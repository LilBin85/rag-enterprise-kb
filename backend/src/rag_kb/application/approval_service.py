"""持久化批准协议的应用编排；数据库原子性由 lifecycle adapter 保证。"""

from __future__ import annotations

from datetime import UTC, datetime

from rag_kb.application.ports.approvals import (
    ActionBinding,
    ApprovalCheckpointBinding,
    ApprovalDetail,
    ApprovalLifecycleError,
    ApprovalLifecyclePort,
    arguments_fingerprint,
    checkpoint_binding_fingerprint,
    principal_fingerprint,
    stable_write_action_key,
)
from rag_kb.application.tool_registry import RegisteredTool, ToolRegistry, ToolRegistryError
from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.resume_commands import ApprovalResumeDecision, ResumeCommand
from rag_kb.domain.tools import ToolMode
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
)


class ApprovalServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ApprovalService:
    def __init__(
        self,
        lifecycle: ApprovalLifecyclePort,
        *,
        registry: ToolRegistry | None = None,
        now=None,
    ) -> None:
        self._lifecycle = lifecycle
        # 信任边界：tool/version 只从注册表白名单派生，拒绝任意工具或伪造版本。
        self._registry = registry or ToolRegistry()
        self._now = now or (lambda: datetime.now(UTC))

    def _record_transition(
        self,
        *,
        decision: str,
        outcome: str,
        task_id: str,
        created_at: datetime | None = None,
        error_code: str | None = None,
    ) -> None:
        wait_seconds = None
        if created_at is not None:
            created_at = (
                created_at.replace(tzinfo=UTC)
                if created_at.tzinfo is None
                else created_at
            )
            wait_seconds = (self._now() - created_at).total_seconds()
        category = error_category_for(error_code) if error_code is not None else "none"
        with observe_span(
            "approval.transition",
            attributes={
                "component": "approval",
                "operation": "approval.transition",
                "decision": decision,
                "outcome": outcome,
                "task_id": task_id,
            },
        ) as span:
            finish_observed_span(
                span,
                outcome=outcome,
                error_category=category,
                error_code=error_code,
            )
            telemetry = current_observability()
            if telemetry is not None:
                telemetry.record_approval(
                    decision=decision,
                    outcome=outcome,
                    wait_seconds=wait_seconds,
                )
            log_event(
                "approval.transitioned",
                level=40 if category != "none" else 20,
                component="approval",
                operation="approval.transition",
                decision=decision,
                outcome=outcome,
                error_category=category,
                error_code=error_code,
                task_id=task_id,
                duration_ms=wait_seconds * 1_000 if wait_seconds is not None else None,
            )

    def _require_writable_tool(self, tool_name: str) -> RegisteredTool:
        try:
            registered = self._registry.get(tool_name)
        except ToolRegistryError as exc:
            raise ApprovalServiceError("tool_not_registered") from exc
        if registered.definition.mode is not ToolMode.WRITE:
            raise ApprovalServiceError("approval_tool_mode_invalid")
        return registered

    @staticmethod
    def _require_valid_checkpoint_binding(
        checkpoint_binding: ApprovalCheckpointBinding,
    ) -> None:
        if (
            not isinstance(checkpoint_binding.thread_id, str)
            or not checkpoint_binding.thread_id.strip()
            or not isinstance(checkpoint_binding.checkpoint_ns, str)
            or not isinstance(checkpoint_binding.checkpoint_id, str)
            or not checkpoint_binding.checkpoint_id.strip()
            or not isinstance(checkpoint_binding.interrupt_id, str)
            or not checkpoint_binding.interrupt_id.strip()
            or not isinstance(checkpoint_binding.tool_call_id, str)
            or not checkpoint_binding.tool_call_id.strip()
        ):
            raise ApprovalServiceError("approval_checkpoint_binding_invalid")

    @staticmethod
    def _raise_lifecycle(exc: ApprovalLifecycleError) -> None:
        raise ApprovalServiceError(exc.code) from exc

    def ensure_action_binding(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        tool_name: str,
        arguments: dict[str, object],
        checkpoint_binding: ApprovalCheckpointBinding,
    ) -> ActionBinding:
        registered = self._require_writable_tool(tool_name)
        self._require_valid_checkpoint_binding(checkpoint_binding)
        try:
            return self._lifecycle.ensure_action_binding(
                auth=auth,
                task_id=task_id,
                tool_name=tool_name,
                tool_version=registered.definition.version,
                arguments_hash=arguments_fingerprint(arguments),
                checkpoint_binding=checkpoint_binding,
            )
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)

    def request(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        plan_revision: int,
        plan_digest: str,
        evidence_digest: str,
        evidence_ids: tuple[str, ...],
        tool_name: str,
        tool_version: str | None = None,
        arguments: dict[str, object],
        expires_at: datetime,
        checkpoint_binding: ApprovalCheckpointBinding | None = None,
    ) -> ApprovalBinding:
        registered = self._require_writable_tool(tool_name)
        if tool_version is not None and tool_version != registered.definition.version:
            raise ApprovalServiceError("approval_tool_version_invalid")
        if checkpoint_binding is not None:
            self._require_valid_checkpoint_binding(checkpoint_binding)
        self.expire_due()
        try:
            result = self._lifecycle.request(
                auth=auth,
                task_id=task_id,
                plan_revision=plan_revision,
                plan_digest=plan_digest,
                evidence_digest=evidence_digest,
                evidence_ids=evidence_ids,
                tool_name=tool_name,
                tool_version=registered.definition.version,
                arguments_hash=arguments_fingerprint(arguments),
                expires_at=expires_at,
                checkpoint_binding=checkpoint_binding,
            )
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)
        if result.created:
            self._record_transition(
                decision="requested",
                outcome="succeeded",
                task_id=result.binding.task_id,
            )
        return result.binding

    def approve(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        expected_task_id: str | None = None,
        expected_plan_revision: int | None = None,
        expected_task_state_version: int | None = None,
    ) -> ApprovalBinding:
        if not auth.has_permission(Permission.TASK_APPROVE):
            raise ApprovalServiceError("approver_permission_denied")
        self.expire_due(tenant_id=auth.tenant_id)
        try:
            result = self._lifecycle.approve(
                auth=auth,
                approval_id=approval_id,
                expected_task_id=expected_task_id,
                expected_plan_revision=expected_plan_revision,
                expected_task_state_version=expected_task_state_version,
            )
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)
        if result.transition is not None:
            if result.invalid_code is not None:
                self._record_transition(
                    decision="rejected",
                    outcome="failed",
                    task_id=result.transition.task_id,
                    created_at=result.transition.created_at,
                    error_code=result.invalid_code,
                )
                raise ApprovalServiceError(result.invalid_code)
            self._record_transition(
                decision="approved",
                outcome="succeeded",
                task_id=result.transition.task_id,
                created_at=result.transition.created_at,
            )
        return result.binding

    def decide_and_enqueue(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        expected_task_state_version: int,
        decision: str,
        reason: str | None,
    ) -> ResumeCommand:
        if not auth.has_permission(Permission.TASK_APPROVE):
            raise ApprovalServiceError("approver_permission_denied")
        try:
            resume_decision = {
                "approve": ApprovalResumeDecision.APPROVED,
                "reject": ApprovalResumeDecision.REJECTED,
            }[decision]
        except KeyError as exc:
            raise ValueError("approval_decision_invalid") from exc
        try:
            result = self._lifecycle.decide_and_enqueue(
                auth=auth,
                approval_id=approval_id,
                expected_task_state_version=expected_task_state_version,
                decision=resume_decision,
                reason=reason,
            )
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)
        if result.transition is not None:
            self._record_transition(
                decision=resume_decision.value,
                outcome="succeeded",
                task_id=result.transition.task_id,
                created_at=result.transition.created_at,
            )
        return result.command

    def reject(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        reason: str | None = None,
        expected_task_id: str | None = None,
        expected_task_state_version: int | None = None,
    ) -> ApprovalBinding:
        if not auth.has_permission(Permission.TASK_APPROVE):
            raise ApprovalServiceError("approver_permission_denied")
        self.expire_due(tenant_id=auth.tenant_id)
        try:
            result = self._lifecycle.reject(
                auth=auth,
                approval_id=approval_id,
                reason=reason,
                expected_task_id=expected_task_id,
                expected_task_state_version=expected_task_state_version,
            )
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)
        if result.transition is not None:
            self._record_transition(
                decision="rejected",
                outcome="succeeded",
                task_id=result.transition.task_id,
                created_at=result.transition.created_at,
            )
        return result.binding

    def get(self, *, tenant_id: str, approval_id: str) -> ApprovalBinding:
        try:
            return self._lifecycle.get(tenant_id=tenant_id, approval_id=approval_id)
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)

    def get_detail(self, *, tenant_id: str, approval_id: str) -> ApprovalDetail:
        try:
            return self._lifecycle.get_detail(tenant_id=tenant_id, approval_id=approval_id)
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)

    def list(
        self,
        *,
        tenant_id: str,
        status: str | None = "pending",
        task_id: str | None = None,
    ) -> tuple[ApprovalBinding, ...]:
        return self._lifecycle.list(
            tenant_id=tenant_id,
            status=status,
            task_id=task_id,
        )

    def list_details(
        self,
        *,
        tenant_id: str,
        status: str | None = "pending",
        task_id: str | None = None,
    ) -> tuple[ApprovalDetail, ...]:
        return self._lifecycle.list_details(
            tenant_id=tenant_id,
            status=status,
            task_id=task_id,
        )

    def load_action_facts(
        self, *, auth: AuthContext, approval_id: str
    ) -> tuple[ApprovalBinding, tuple[str, ...]]:
        try:
            return self._lifecycle.load_action_facts(
                auth=auth,
                approval_id=approval_id,
            )
        except ApprovalLifecycleError as exc:
            self._raise_lifecycle(exc)

    def load_approved_for_replay(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        tool_call_id: str,
    ) -> tuple[ApprovalBinding, tuple[str, ...]] | None:
        return self._lifecycle.load_approved_for_replay(
            auth=auth,
            task_id=task_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            tool_call_id=tool_call_id,
        )

    def expire_due(self, *, tenant_id: str | None = None) -> int:
        transitions = self._lifecycle.expire_due(tenant_id=tenant_id)
        for transition in transitions:
            self._record_transition(
                decision="expired",
                outcome="succeeded",
                task_id=transition.task_id,
                created_at=transition.created_at,
            )
        return len(transitions)
