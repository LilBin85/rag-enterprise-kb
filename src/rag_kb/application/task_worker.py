"""可恢复的通用任务 worker：动作对象是恢复边界，不重调 planner。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_runtime import TaskEvidenceValidationPort, TaskRuntime
from rag_kb.application.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExecutionService,
)
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.task_runtime import TaskRuntimeError, canonical_fingerprint
from rag_kb.domain.tasks import CallToolAction, ProposeAction, SearchAction, TaskNode
from rag_kb.domain.tools import ToolMode
from rag_kb.infrastructure.database.task_execution import TaskPlanBinding
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError
from rag_kb.infrastructure.database.task_runtime_adapter import LeasedTaskRuntimeRepository


class TaskWorkerError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class TaskAuthContextPort(Protocol):
    def resolve(self, *, task_id: str, auth_context_ref: str, request_id: str) -> AuthContext: ...


class TaskPlanBindingPort(Protocol):
    def load(self, *, auth: AuthContext, task_id: str, revision: int) -> TaskPlanBinding: ...


@dataclass(frozen=True)
class TaskWorkerResult:
    status: str
    task_id: str | None = None
    error_code: str | None = None


class TaskWorker:
    def __init__(
        self,
        repository: DatabaseTaskRepository,
        planner,
        tools: ToolExecutionService,
        registry: ToolRegistry,
        approvals: ApprovalService,
        auth_contexts: TaskAuthContextPort,
        plans: TaskPlanBindingPort,
        evidence_validator: TaskEvidenceValidationPort,
        pending_actions,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now=None,
    ) -> None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_worker")
        self._repository = repository
        self._planner = planner
        self._tools = tools
        self._registry = registry
        self._approvals = approvals
        self._auth_contexts = auth_contexts
        self._plans = plans
        self._evidence_validator = evidence_validator
        self._pending_actions = pending_actions
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._now = now or (lambda: datetime.now(UTC))

    def run_once(self) -> TaskWorkerResult:
        lease = self._repository.claim_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if lease is None:
            return TaskWorkerResult(status="idle")
        run_id = uuid4().hex
        runtime_repo = LeasedTaskRuntimeRepository(
            self._repository,
            lease,
            trace_id=run_id,
            run_id=run_id,
            pending_actions=self._pending_actions,
        )
        runtime = TaskRuntime(
            runtime_repo,
            self._planner,
            budgets=runtime_repo.budgets,
            evidence_validator=self._evidence_validator,
            now=self._now,
        )
        try:
            if lease.state.plan_revision == 0:
                lease = self._repository.bootstrap_initial_plan(
                    runtime_repo.lease,
                    trace_id=run_id,
                    run_id=run_id,
                )
                runtime_repo = LeasedTaskRuntimeRepository(
                    self._repository,
                    lease,
                    trace_id=run_id,
                    run_id=run_id,
                    pending_actions=self._pending_actions,
                )
                runtime = TaskRuntime(
                    runtime_repo,
                    self._planner,
                    budgets=runtime_repo.budgets,
                    evidence_validator=self._evidence_validator,
                    now=self._now,
                )
            self._advance_deterministic_nodes(runtime, lease.state.task_id)
            state = runtime.step(lease.state.task_id)
            if state.pending_action is None:
                self._release_runnable_lease(runtime, state.task_id)
                return TaskWorkerResult(status="progressed", task_id=state.task_id)
            auth = self._auth_contexts.resolve(
                task_id=state.task_id,
                auth_context_ref=state.auth_context_ref,
                request_id=run_id,
            )
            action = runtime.load_pending_action(state.task_id)
            if isinstance(action, ProposeAction) and state.current_node is TaskNode.SELECT_ACTION:
                result = self._await_approval(runtime, auth, state.task_id, action)
            else:
                result = self._execute_pending(runtime, auth, state.task_id, action)
            if result.status != "unknown":
                self._release_runnable_lease(runtime, state.task_id)
            return result
        except TaskRepositoryError as exc:
            return TaskWorkerResult(status="stale", task_id=lease.state.task_id, error_code=exc.code)
        except ToolExecutionError as exc:
            stale = exc.code == "tool_task_lease_stale"
            if not stale:
                try:
                    runtime.fail_closed(lease.state.task_id, exc.code)
                except Exception:
                    pass
            status = "stale" if stale else "failed"
            return TaskWorkerResult(status=status, task_id=lease.state.task_id, error_code=exc.code)
        except Exception as exc:
            # 不把模型、对象或工具正文写入日志/状态；可恢复失败保持安全码。
            code = getattr(exc, "code", "task_worker_failed")
            try:
                runtime.fail_closed(lease.state.task_id, str(code))
            except Exception:
                # stale lease 时不能覆盖其他 worker 的 checkpoint。
                pass
            return TaskWorkerResult(
                status="failed",
                task_id=lease.state.task_id,
                error_code=code,
            )

    def expire_stale_tool_calls(self, *, limit: int = 100) -> int:
        """由生产 worker 周期调用；工具恢复仍由执行服务统一实现。"""
        return self._tools.expire_stale_running(limit=limit)

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int:
        return self._repository.expire_stale_knowledge_queries(limit=limit)

    @staticmethod
    def _release_runnable_lease(runtime: TaskRuntime, task_id: str) -> None:
        """等待/终态由 checkpoint 自行释放；其余节点交回队列以继续推进。"""
        try:
            state = runtime.current_state(task_id)
        except TaskRuntimeError:
            return
        if state.status.value in {"planning", "running"}:
            runtime.release_lease(task_id)

    @staticmethod
    def _advance_deterministic_nodes(runtime: TaskRuntime, task_id: str) -> None:
        """只推进没有模型决策空间的状态图边，确保下一次模型调用处于合法节点。"""

        while True:
            state = runtime.current_state(task_id)
            if state.current_node is TaskNode.RECORD_EVIDENCE:
                runtime.transition(task_id, TaskNode.EVALUATE_EVIDENCE)
                continue
            if state.current_node in {TaskNode.EVALUATE_EVIDENCE, TaskNode.UPDATE_PLAN}:
                runtime.transition(task_id, TaskNode.SELECT_ACTION)
                continue
            if state.current_node is TaskNode.VERIFY_ACTION:
                # 写结果已经过验证；生成结果节点只允许 finish/refuse 等终结动作。
                runtime.transition(task_id, TaskNode.GENERATE_RESULT)
                continue
            return

    def _await_approval(
        self,
        runtime: TaskRuntime,
        auth: AuthContext,
        task_id: str,
        action: ProposeAction,
    ) -> TaskWorkerResult:
        state = runtime.current_state(task_id)
        if state.pending_action is None:
            raise TaskWorkerError("pending_action_missing")
        definition = self._registry.get(action.tool_name).definition
        if definition.mode is not ToolMode.WRITE:
            raise TaskWorkerError("proposed_tool_not_write")
        plan = self._plans.load(
            auth=auth,
            task_id=state.task_id,
            revision=state.plan_revision,
        )
        try:
            proposed_evidence = canonical_evidence_snapshot(
                action.evidence_ids,
                require_non_empty=True,
            )
        except ValueError as exc:
            raise TaskWorkerError(str(exc)) from exc
        if proposed_evidence.evidence_ids != plan.evidence_ids:
            raise TaskWorkerError("approval_evidence_exact_set_invalid")
        approval = self._approvals.request(
            auth=auth,
            task_id=state.task_id,
            plan_revision=plan.plan_revision,
            plan_digest=plan.plan_digest,
            evidence_digest=plan.evidence_digest,
            evidence_ids=plan.evidence_ids,
            tool_name=action.tool_name,
            tool_version=definition.version,
            arguments=action.arguments,
            expires_at=state.pending_action.expires_at,
        )
        paused = runtime.await_approval(state.task_id, approval_id=approval.approval_id)
        return TaskWorkerResult(status="waiting_for_approval", task_id=paused.task_id)

    def _execute_pending(
        self, runtime: TaskRuntime, auth: AuthContext, task_id: str, action
    ) -> TaskWorkerResult:
        state = runtime.current_state(task_id)
        if state.pending_action is None:
            raise TaskWorkerError("pending_action_missing")
        if isinstance(action, SearchAction):
            tool_name, arguments, expected_mode = "search_knowledge", {"query": action.query}, ToolMode.READ
            evidence_ids: tuple[str, ...] = ()
        elif isinstance(action, CallToolAction):
            tool_name, arguments, expected_mode = action.tool_name, action.arguments, ToolMode.READ
            evidence_ids = ()
        elif isinstance(action, ProposeAction):
            tool_name, arguments, expected_mode = action.tool_name, action.arguments, ToolMode.WRITE
            evidence_ids = action.evidence_ids
        else:
            raise TaskWorkerError("pending_action_kind_invalid")
        definition = self._registry.get(tool_name).definition
        if definition.mode is not expected_mode:
            raise TaskWorkerError("pending_action_tool_mode_invalid")
        if expected_mode is ToolMode.READ and state.current_node is not TaskNode.EXECUTE_READ_TOOL:
            raise TaskWorkerError("pending_action_node_invalid")
        if expected_mode is ToolMode.WRITE and (
            state.current_node is not TaskNode.EXECUTE_APPROVED_ACTION
            or not state.pending_approval
        ):
            raise TaskWorkerError("pending_approval_binding_invalid")
        plan = self._plans.load(auth=auth, task_id=state.task_id, revision=state.plan_revision)
        if expected_mode is ToolMode.WRITE:
            try:
                proposed_evidence = canonical_evidence_snapshot(
                    evidence_ids,
                    require_non_empty=True,
                )
            except ValueError as exc:
                raise TaskWorkerError(str(exc)) from exc
            if proposed_evidence.evidence_ids != plan.evidence_ids:
                raise TaskWorkerError("tool_evidence_exact_set_invalid")
        key = canonical_fingerprint(
            "task_tool_execution",
            {
                "task_id": state.task_id,
                "plan_revision": plan.plan_revision,
                "pending_action_hash": state.pending_action.content_hash,
                "tool_name": tool_name,
                "tool_version": definition.version,
            },
        )
        result: ToolExecutionResult = self._tools.execute(
            ToolExecutionRequest(
                auth=auth,
                task_id=state.task_id,
                plan_revision=plan.plan_revision,
                plan_digest=plan.plan_digest,
                evidence_digest=plan.evidence_digest,
                evidence_ids=plan.evidence_ids,
                tool_name=tool_name,
                arguments=arguments,
                worker_id=self._worker_id,
                lease_generation=state.lease_generation,
                idempotency_key=key,
                approval_id=state.pending_approval if expected_mode is ToolMode.WRITE else None,
            )
        )
        if expected_mode is ToolMode.WRITE and result.status == "unknown":
            # 写副作用不确定时 ToolExecutionService 已关闭任务并释放 lease，绝不重放。
            return TaskWorkerResult(status="unknown", task_id=state.task_id, error_code=result.error_code)
        runtime.refresh_lease(state.task_id)
        result_reference = result.result_hash or canonical_fingerprint(
            "tool_result_terminal",
            {"tool_call_id": result.tool_call_id, "status": result.status, "error_code": result.error_code},
        )
        if expected_mode is ToolMode.READ:
            result_evidence = ()
            if result.result is not None:
                raw = result.result.get("evidence_ids")
                if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                    result_evidence = tuple(raw)
            completed = runtime.record_tool_result(
                state.task_id,
                result_reference=result_reference,
                evidence_ids=result_evidence,
                execution_already_counted=True,
            )
        else:
            completed = runtime.complete_approved_action(
                state.task_id,
                result_reference=result_reference,
                evidence_ids=evidence_ids,
            )
        return TaskWorkerResult(status="progressed", task_id=completed.task_id)
