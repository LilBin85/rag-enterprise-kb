"""单智能体显式状态图；每个公开步骤都通过仓储 CAS 持久化。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.task_runtime import (
    BudgetOperation,
    TaskRuntimeError,
    canonical_fingerprint,
    check_budget,
    ensure_unique_fingerprint,
    query_fingerprint,
    validate_transition,
)
from rag_kb.domain.tasks import (
    AskUserAction,
    CallToolAction,
    FinishAction,
    NextAction,
    PersistableAction,
    ProposeAction,
    RefuseAction,
    SearchAction,
    TaskBudgets,
    TaskNode,
    TaskMode,
    TaskState,
    TaskStatus,
    UpdatePlanAction,
)


_NEXT_ACTION_ADAPTER = TypeAdapter(NextAction)
_TERMINAL = {
    TaskStatus.SUCCEEDED,
    TaskStatus.REFUSED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.UNKNOWN,
}


class TaskRuntimeRepositoryPort(Protocol):
    def load(self, task_id: str) -> TaskState: ...

    def save(
        self,
        state: TaskState,
        *,
        expected_state_version: int,
        event: dict[str, object] | None = None,
    ) -> TaskState: ...

    def save_pending_action(
        self,
        state: TaskState,
        *,
        expected_state_version: int,
        action: PersistableAction,
        expires_at: datetime,
        event: dict[str, object] | None = None,
    ) -> TaskState: ...

    def load_pending_action(self) -> PersistableAction: ...

    def save_plan_revision(self, state: TaskState, *, steps: tuple[str, ...]) -> TaskState: ...

    def release_lease(self) -> None: ...


class NextActionPlannerPort(Protocol):
    def next_action(self, state: TaskState, observations=()) -> object: ...


class TaskEvidenceValidationPort(Protocol):
    def validate_task(
        self,
        *,
        task_id: str,
        auth_context_ref: str,
        evidence_ids: tuple[str, ...],
        expected_digest: str | None = None,
        request_id: str = "task-evidence-revalidation",
        require_non_empty: bool = True,
    ) -> object: ...


class TaskRuntime:
    """不执行任意工具；这里只验证动作、预算并推进可恢复图状态。"""

    def __init__(
        self,
        repository: TaskRuntimeRepositoryPort,
        planner: NextActionPlannerPort,
        *,
        budgets: TaskBudgets | None = None,
        evidence_validator: TaskEvidenceValidationPort | None = None,
        now=None,
    ) -> None:
        self._repository = repository
        self._planner = planner
        self._budgets = budgets or TaskBudgets()
        self._evidence_validator = evidence_validator
        self._now = now or (lambda: datetime.now(UTC))

    def step(self, task_id: str) -> TaskState:
        state = self._load_active(task_id)
        # 已发布的动作是崩溃恢复边界；绝不能为了恢复再调用 planner。
        if state.pending_action is not None:
            return state
        try:
            check_budget(
                state,
                self._budgets,
                operation=BudgetOperation.MODEL_CALL,
                now=self._now(),
            )
        except TaskRuntimeError as exc:
            try:
                self._fail_closed(state, exc.code)
            except Exception:
                pass
            raise
        counted = self._save(
            state.model_copy(update={"model_calls": state.model_calls + 1}),
            event_type="model_call_started",
        )
        try:
            return self._step_active(counted)
        except ValidationError as exc:
            try:
                self._fail_closed(counted, "next_action_invalid")
            except Exception:
                pass
            raise TaskRuntimeError("next_action_invalid") from exc
        except TaskRuntimeError as exc:
            try:
                self._fail_closed(counted, exc.code)
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                self._fail_closed(counted, "next_action_failed")
            except Exception:
                pass
            raise TaskRuntimeError("next_action_failed") from exc

    def _step_active(self, state: TaskState) -> TaskState:
        action = _NEXT_ACTION_ADAPTER.validate_python(
            self._planner.next_action(state, observations=state.observations)
        )
        action_fp = canonical_fingerprint("action", action.model_dump(mode="json"))
        ensure_unique_fingerprint(
            action_fp,
            state.repeated_action_fingerprints,
            duplicate_code="duplicate_action",
        )
        state = state.model_copy(
            update={
                "repeated_action_fingerprints": (
                    *state.repeated_action_fingerprints,
                    action_fp,
                )
            }
        )
        if isinstance(action, SearchAction):
            query_fp = query_fingerprint(action.query)
            ensure_unique_fingerprint(
                query_fp,
                state.query_fingerprints,
                duplicate_code="duplicate_query",
            )
            return self._stage_action(
                state.model_copy(
                    update={"query_fingerprints": (*state.query_fingerprints, query_fp)}
                ),
                action,
                target=TaskNode.EXECUTE_READ_TOOL,
                event_type="next_action_selected",
            )
        if isinstance(action, CallToolAction):
            return self._stage_action(
                state,
                action,
                target=TaskNode.EXECUTE_READ_TOOL,
                event_type="next_action_selected",
            )
        if isinstance(action, AskUserAction):
            return self._pause_for_user(state, action.question)
        if isinstance(action, UpdatePlanAction):
            check_budget(
                state,
                self._budgets,
                operation=BudgetOperation.REPLAN,
                now=self._now(),
            )
            try:
                return self._repository.save_plan_revision(state, steps=action.steps)
            except AttributeError as exc:
                raise TaskRuntimeError("task_plan_store_unavailable") from exc
        if isinstance(action, ProposeAction):
            if state.task_mode is not TaskMode.CONTROLLED_ACTION:
                raise TaskRuntimeError("write_task_mode_denied")
            try:
                snapshot = canonical_evidence_snapshot(
                    action.evidence_ids,
                    require_non_empty=True,
                )
            except ValueError as exc:
                raise TaskRuntimeError(str(exc)) from exc
            if any(item not in state.evidence_ids for item in snapshot.evidence_ids):
                raise TaskRuntimeError("write_evidence_invalid")
            # 先保存原始动作；随后 worker 创建真实 ApprovalRow 并调用
            # ``await_approval``。两阶段间崩溃时也会复用同一个动作。
            return self._stage_action(
                state,
                action,
                target=None,
                event_type="write_action_staged",
            )
        if isinstance(action, FinishAction):
            return self.finish(state.task_id, action, state=state)
        if isinstance(action, RefuseAction):
            return self.refuse(state.task_id, action, state=state)
        raise TaskRuntimeError("next_action_invalid")

    def transition(self, task_id: str, target: TaskNode) -> TaskState:
        state = self._load_active(task_id)
        return self._advance(state, target, event_type="task_transitioned")

    def record_tool_result(
        self,
        task_id: str,
        *,
        result_reference: str,
        evidence_ids: tuple[str, ...] = (),
        token_usage: int = 0,
        execution_already_counted: bool = False,
    ) -> TaskState:
        if not result_reference or token_usage < 0:
            raise ValueError("invalid_tool_result")
        state = self._load_active(task_id)
        try:
            return self._record_tool_result_active(
                state,
                result_reference=result_reference,
                evidence_ids=evidence_ids,
                token_usage=token_usage,
                execution_already_counted=execution_already_counted,
            )
        except TaskRuntimeError as exc:
            try:
                self._fail_closed(state, exc.code)
            except Exception:
                pass
            raise

    def _record_tool_result_active(
        self,
        state: TaskState,
        *,
        result_reference: str,
        evidence_ids: tuple[str, ...],
        token_usage: int,
        execution_already_counted: bool,
    ) -> TaskState:
        if state.current_node is not TaskNode.EXECUTE_READ_TOOL:
            raise TaskRuntimeError("task_transition_invalid")
        if not execution_already_counted:
            check_budget(
                state,
                self._budgets,
                operation=BudgetOperation.TOOL_CALL,
                now=self._now(),
            )
        result_fp = canonical_fingerprint("result", result_reference)
        ensure_unique_fingerprint(
            result_fp,
            state.result_fingerprints,
            duplicate_code="duplicate_result",
        )
        previous_evidence = frozenset(state.evidence_ids)
        merged_evidence = tuple(dict.fromkeys((*state.evidence_ids, *evidence_ids)))
        no_new = state.no_new_evidence_count + 1 if frozenset(merged_evidence) == previous_evidence else 0
        updated = state.model_copy(
            update={
                "tool_calls": state.tool_calls + (0 if execution_already_counted else 1),
                "token_usage": state.token_usage + token_usage,
                "observations": (*state.observations, {"ref": result_reference}),
                "evidence_ids": merged_evidence,
                "result_fingerprints": (*state.result_fingerprints, result_fp),
                "no_new_evidence_count": no_new,
                "pending_action": None,
            }
        )
        if no_new >= self._budgets.max_no_new_evidence:
            raise TaskRuntimeError("duplicate_no_new_evidence")
        return self._advance(updated, TaskNode.RECORD_EVIDENCE, event_type="tool_result_recorded")

    def load_pending_action(self, task_id: str) -> PersistableAction:
        """回读已绑定动作；载荷校验失败由 store 以安全码关闭执行。"""
        state = self._load_active(task_id)
        if state.pending_action is None:
            raise TaskRuntimeError("pending_action_missing")
        try:
            return self._repository.load_pending_action()
        except TaskRuntimeError:
            raise
        except Exception as exc:
            raise TaskRuntimeError("pending_action_unavailable") from exc

    def current_state(self, task_id: str) -> TaskState:
        """仅供 worker 在已领取 lease 内读取最新 checkpoint。"""
        return self._load_active(task_id)

    def refresh_lease(self, task_id: str) -> TaskState:
        """工具执行已更新计数时，回读同 generation 的最新状态再 checkpoint。"""
        self._load_active(task_id)
        try:
            return self._repository.refresh_lease()
        except AttributeError as exc:
            raise TaskRuntimeError("task_lease_refresh_unavailable") from exc

    def release_lease(self, task_id: str) -> None:
        """让尚未终结的 checkpoint 重新可领取，避免单次调度被租约卡住。"""
        state = self._load_active(task_id)
        if state.status not in {TaskStatus.PLANNING, TaskStatus.RUNNING}:
            return
        try:
            self._repository.release_lease()
        except AttributeError as exc:
            raise TaskRuntimeError("task_lease_release_unavailable") from exc

    def fail_closed(self, task_id: str, error_code: str) -> None:
        """worker 的不可恢复安全错误统一收敛为 terminal failed。"""
        self._fail_closed(self._load_active(task_id), error_code)

    def await_approval(self, task_id: str, *, approval_id: str) -> TaskState:
        """把已持久化的 write 动作绑定到真实审批记录并释放 lease。"""
        state = self._load_active(task_id)
        if (
            state.pending_action is None
            or state.pending_action.kind != "propose_action"
            or state.current_node is not TaskNode.SELECT_ACTION
            or not approval_id
        ):
            raise TaskRuntimeError("approval_action_binding_invalid")
        validate_transition(state.current_node, TaskNode.AWAIT_APPROVAL)
        return self._save(
            state.model_copy(
                update={
                    "status": TaskStatus.WAITING_FOR_APPROVAL,
                    "current_node": TaskNode.AWAIT_APPROVAL,
                    "pending_approval": approval_id,
                }
            ),
            event_type="approval_requested",
        )

    def complete_approved_action(
        self,
        task_id: str,
        *,
        result_reference: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> TaskState:
        """写工具已由执行层计数；这里只记录可验证结果并清除 pending binding。"""
        state = self._load_active(task_id)
        if state.current_node is not TaskNode.EXECUTE_APPROVED_ACTION:
            raise TaskRuntimeError("task_transition_invalid")
        if state.pending_action is None or state.pending_action.kind != "propose_action":
            raise TaskRuntimeError("approval_action_binding_invalid")
        result_fp = canonical_fingerprint("result", result_reference)
        ensure_unique_fingerprint(
            result_fp,
            state.result_fingerprints,
            duplicate_code="duplicate_result",
        )
        merged_evidence = tuple(dict.fromkeys((*state.evidence_ids, *evidence_ids)))
        completed = state.model_copy(
            update={
                "observations": (*state.observations, {"ref": result_reference}),
                "evidence_ids": merged_evidence,
                "result_fingerprints": (*state.result_fingerprints, result_fp),
                "pending_action": None,
                "pending_approval": None,
            }
        )
        return self._advance(completed, TaskNode.VERIFY_ACTION, event_type="write_action_verified")

    def ask_user(self, task_id: str, *, question: str) -> TaskState:
        state = self._load_active(task_id)
        return self._pause_for_user(state, question)

    def resume(self, task_id: str, answer: object) -> TaskState:
        state = self._repository.load(task_id)
        if state.status is not TaskStatus.WAITING_FOR_USER:
            raise TaskRuntimeError("task_not_waiting_for_user")
        answer_ref = canonical_fingerprint("user_answer", answer)
        resumed = state.model_copy(
            update={
                "status": TaskStatus.RUNNING,
                "current_node": TaskNode.UPDATE_PLAN,
                "pending_question": None,
                "observations": (*state.observations, {"ref": answer_ref}),
            }
        )
        return self._save(resumed, event_type="task_resumed")

    def cancel(self, task_id: str) -> TaskState:
        state = self._load_active(task_id)
        cancelled = state.model_copy(
            update={
                "status": TaskStatus.CANCELLED,
                "current_node": TaskNode.TERMINAL,
                "cancelled": True,
            }
        )
        return self._save(cancelled, event_type="task_cancelled")

    def finish(
        self,
        task_id: str,
        action: FinishAction,
        *,
        state: TaskState | None = None,
    ) -> TaskState:
        current = state or self._load_active(task_id)
        if current.cancelled:
            raise TaskRuntimeError("task_cancelled")
        validate_transition(current.current_node, TaskNode.TERMINAL)
        if any(claim not in current.evidence_ids for claim in action.supported_claim_ids):
            raise TaskRuntimeError("finish_evidence_invalid")
        try:
            snapshot = canonical_evidence_snapshot(
                action.supported_claim_ids,
                require_non_empty=True,
            )
        except ValueError as exc:
            raise TaskRuntimeError(str(exc)) from exc
        if self._evidence_validator is None:
            raise TaskRuntimeError("finish_evidence_validator_unavailable")
        try:
            self._evidence_validator.validate_task(
                task_id=current.task_id,
                auth_context_ref=current.auth_context_ref,
                evidence_ids=snapshot.evidence_ids,
                require_non_empty=True,
            )
        except Exception as exc:
            code = getattr(exc, "code", "finish_evidence_revalidation_failed")
            raise TaskRuntimeError(str(code)) from exc
        finished = current.model_copy(
            update={
                "status": TaskStatus.SUCCEEDED,
                "current_node": TaskNode.TERMINAL,
                "last_error": None,
            }
        )
        return self._save(finished, event_type="task_succeeded")

    def refuse(
        self,
        task_id: str,
        action: RefuseAction,
        *,
        state: TaskState | None = None,
    ) -> TaskState:
        current = state or self._load_active(task_id)
        validate_transition(current.current_node, TaskNode.TERMINAL)
        refused = current.model_copy(
            update={
                "status": TaskStatus.REFUSED,
                "current_node": TaskNode.TERMINAL,
                "last_error": action.refusal_code,
            }
        )
        return self._save(refused, event_type="task_refused")

    def _pause_for_user(self, state: TaskState, question: str) -> TaskState:
        if not question.strip():
            raise ValueError("task_question_required")
        validate_transition(state.current_node, TaskNode.CHECK_CLARIFICATION)
        paused = state.model_copy(
            update={
                "status": TaskStatus.WAITING_FOR_USER,
                "current_node": TaskNode.CHECK_CLARIFICATION,
                "pending_question": question.strip(),
            }
        )
        return self._save(paused, event_type="clarification_requested")

    def _advance(
        self,
        state: TaskState,
        target: TaskNode,
        *,
        event_type: str,
    ) -> TaskState:
        validate_transition(state.current_node, target)
        advanced = state.model_copy(update={"current_node": target})
        return self._save(advanced, event_type=event_type)

    def _stage_action(
        self,
        state: TaskState,
        action: PersistableAction,
        *,
        target: TaskNode | None,
        event_type: str,
    ) -> TaskState:
        if target is not None:
            validate_transition(state.current_node, target)
            state = state.model_copy(update={"current_node": target})
        expires_at = self._now() + timedelta(seconds=self._budgets.max_runtime_seconds)
        if state.deadline is not None and expires_at > state.deadline:
            expires_at = state.deadline
        expected = state.state_version
        versioned = state.model_copy(update={"state_version": expected + 1})
        try:
            return self._repository.save_pending_action(
                versioned,
                expected_state_version=expected,
                action=action,
                expires_at=expires_at,
                event={"event_type": event_type, "status": versioned.status.value},
            )
        except AttributeError as exc:
            raise TaskRuntimeError("pending_action_store_unavailable") from exc

    def _save(self, state: TaskState, *, event_type: str) -> TaskState:
        expected = state.state_version
        versioned = state.model_copy(update={"state_version": expected + 1})
        return self._repository.save(
            versioned,
            expected_state_version=expected,
            event={"event_type": event_type, "status": versioned.status.value},
        )

    def _load_active(self, task_id: str) -> TaskState:
        state = self._repository.load(task_id)
        if state.cancelled:
            raise TaskRuntimeError("task_cancelled")
        if state.status in _TERMINAL:
            raise TaskRuntimeError("task_terminal_state")
        return state

    def _fail_closed(self, state: TaskState, error_code: str) -> None:
        failed = state.model_copy(
            update={
                "status": TaskStatus.FAILED,
                "current_node": TaskNode.TERMINAL,
                "last_error": error_code,
                "pending_action": None,
                "pending_approval": None,
            }
        )
        self._save(failed, event_type="task_failed")
