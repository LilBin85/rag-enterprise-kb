"""checkpointer 驱动的 Research Agent worker。"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Protocol
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from sqlalchemy import func, select

from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.result import (
    AgentResultStore,
    ArtifactSummary,
    TaskResultCandidate,
    build_controlled_task_result,
    build_task_result,
)
from rag_kb.agent.tools import CONTROLLED_WRITE_TOOL_NAMES
from rag_kb.application.approval_service import (
    ApprovalCheckpointBinding,
    ApprovalService,
    stable_write_action_key,
)
from rag_kb.application.tool_execution_service import (
    _STALE_TOOL_CODES,
    ToolExecutionError,
    ToolExecutionService,
)
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskStatus
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import AgentTaskRow, ArtifactRow, ToolCallRow
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
    TaskRepositoryError,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore

logger = logging.getLogger(__name__)

_STALE_REPOSITORY_CODES = frozenset(
    {"task_lease_generation_stale", "task_lease_expired"}
)
_STALE_CHECKPOINT_CODE = "task_checkpoint_lease_stale"
_HITL_ALLOWED_DECISIONS = frozenset({"approve", "reject"})
_APPROVAL_RESUME_DECISIONS = frozenset({"approve", "reject"})
# approval 有效期沿用现状：默认 15 分钟、应用进程时间源（开放问题 5 未拍板）。
_APPROVAL_EXPIRY_SECONDS = 15 * 60
# reject decision 的固定模型反馈文案；审计细节（rejection_reason）不进模型上下文。
_REJECT_DECISION_MESSAGE = "操作未获业务审批，已由服务端拒绝。"


def _approval_resume_payload(decision: str) -> dict[str, object]:
    """服务端固定 resume payload；客户端与模型不能提交任何 resume 数据。"""
    if decision == "approve":
        return {"decisions": [{"type": "approve"}]}
    return {"decisions": [{"type": "reject", "message": _REJECT_DECISION_MESSAGE}]}


class TaskWorkerError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _TaskLeaseHeartbeat:
    """仅在同步 Agent 调用期间保持同一代租约有效。"""

    def __init__(self, renew: Callable[[], TaskLease], lease_seconds: int) -> None:
        self._renew = renew
        self._interval = max(0.1, lease_seconds / 3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def __enter__(self) -> "_TaskLeaseHeartbeat":
        self.renew_now()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._renew()
            except Exception as exc:
                self._error = exc
                self._stop.set()

    def check(self) -> None:
        if self._error is not None:
            raise self._error

    def renew_now(self) -> None:
        self.check()
        self._renew()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
            self._thread = None
        self.check()


class TaskAuthContextPort(Protocol):
    def resolve(self, *, task_id: str, auth_context_ref: str, request_id: str) -> AuthContext: ...


@dataclass(frozen=True)
class TaskWorkerResult:
    status: str
    task_id: str | None = None
    error_code: str | None = None


class TaskWorker:
    """领取业务任务，推进或恢复唯一 `thread_id = task_id` 的 Agent。"""

    def __init__(
        self,
        *,
        repository: DatabaseTaskRepository,
        auth_contexts: TaskAuthContextPort,
        agent_for: Callable[[TaskBudgets], Any],
        results: AgentResultStore,
        evidence: EvidenceLedgerRepository,
        tools: ToolExecutionService,
        database,
        worker_id: str,
        lease_seconds: int = 300,
        capability_name: str | None = None,
        protocol_version: str | None = None,
        controlled_agent_for: Callable[[TaskBudgets], Any] | None = None,
        approvals: ApprovalService | None = None,
        objects: LocalObjectStore | None = None,
        close=None,
    ) -> None:
        if (
            not worker_id
            or lease_seconds <= 0
            or (capability_name is None) != (protocol_version is None)
            or capability_name is not None
            and (not capability_name or not protocol_version)
        ):
            raise ValueError("invalid_task_worker")
        self._repository = repository
        self._auth_contexts = auth_contexts
        self._agent_for = agent_for
        self._results = results
        self._evidence = evidence
        self._tools = tools
        self._database = database
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._capability_name = capability_name
        self._protocol_version = protocol_version
        # 受控动作（controlled_action）需要独立编译的 HITL Agent 与 ApprovalService；
        # 未配置时受控任务保持 fail-closed，M1 Research 轨迹不变。
        self._controlled_agent_for = controlled_agent_for
        self._approvals = approvals
        # 受控任务 settle 前 artifact 正文 hash 校验所需的对象存储。
        self._objects = objects
        self._close = close
        self._accepting_tasks = True

    def close(self) -> None:
        if not self._accepting_tasks:
            return
        self._accepting_tasks = False
        close = self._close
        self._close = None
        try:
            close_tools = getattr(self._tools, "close", None)
            if close_tools is not None:
                close_tools()
        finally:
            if close is not None:
                close()

    def run_once(self) -> TaskWorkerResult:
        if not self._accepting_tasks:
            return TaskWorkerResult(status="idle")
        claim: dict[str, object] = {
            "worker_id": self._worker_id,
            "lease_seconds": self._lease_seconds,
        }
        if self._capability_name is not None and self._protocol_version is not None:
            claim.update(
                capability_name=self._capability_name,
                protocol_version=self._protocol_version,
            )
        lease = self._repository.claim_next(**claim)
        if lease is None:
            return TaskWorkerResult(status="idle")
        return self._run_claimed(lease)

    def resume_for_user(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        answer: str,
    ) -> TaskWorkerResult:
        """仅在保存的 user clarification interrupt 上恢复同一线程。"""
        if not answer.strip() or len(answer) > 2_000:
            raise ValueError("task_resume_answer_invalid")
        if not self._accepting_tasks:
            raise TaskWorkerError("task_worker_closed")
        claim: dict[str, object] = {
            "auth": auth,
            "task_id": task_id,
            "expected_state_version": expected_state_version,
            "worker_id": self._worker_id,
            "lease_seconds": self._lease_seconds,
        }
        if self._capability_name is not None and self._protocol_version is not None:
            claim.update(
                capability_name=self._capability_name,
                protocol_version=self._protocol_version,
            )
        lease = self._repository.claim_user_resume(**claim)
        return self._run_claimed(lease, resume_answer=answer.strip())

    def resume_for_approval(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        approval_id: str,
        decision: str,
        reason: str | None = None,
    ) -> TaskWorkerResult:
        """审批 decision → 受控 resume：decision 先落库，服务端驱动同 thread 恢复。

        客户端只提交业务 decision；Command(resume=...) 由服务端从保存的
        checkpoint interrupt 构造，模型/客户端零 payload（M2_PLAN 不变量）。
        """
        if decision not in _APPROVAL_RESUME_DECISIONS:
            raise ValueError("task_approval_decision_invalid")
        if not self._accepting_tasks:
            raise TaskWorkerError("task_worker_closed")
        if self._approvals is None:
            raise TaskWorkerError("task_mode_not_supported")
        if decision == "approve":
            self._approvals.approve(
                auth=auth,
                approval_id=approval_id,
                expected_task_id=task_id,
                expected_task_state_version=expected_state_version,
            )
        else:
            self._approvals.reject(
                auth=auth,
                approval_id=approval_id,
                reason=reason,
                expected_task_id=task_id,
                expected_task_state_version=expected_state_version,
            )
        claim: dict[str, object] = {
            "auth": auth,
            "task_id": task_id,
            "expected_state_version": expected_state_version,
            "approval_id": approval_id,
            "worker_id": self._worker_id,
            "lease_seconds": self._lease_seconds,
        }
        if self._capability_name is not None and self._protocol_version is not None:
            claim.update(
                capability_name=self._capability_name,
                protocol_version=self._protocol_version,
            )
        lease = self._repository.claim_approval_resume(**claim)
        return self._run_claimed(
            lease,
            resume_decision=decision,
            resume_approval_id=approval_id,
        )

    def expire_stale_tool_calls(self, *, limit: int = 100) -> int:
        return self._tools.expire_stale_running(limit=limit)

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int:
        return self._repository.expire_stale_knowledge_queries(limit=limit)

    def _run_claimed(
        self,
        lease: TaskLease,
        *,
        resume_answer: str | None = None,
        resume_decision: str | None = None,
        resume_approval_id: str | None = None,
    ) -> TaskWorkerResult:
        run_id = uuid4().hex
        try:
            return self._advance(
                lease,
                run_id=run_id,
                resume_answer=resume_answer,
                resume_decision=resume_decision,
                resume_approval_id=resume_approval_id,
            )
        except TaskRepositoryError as exc:
            if exc.code not in _STALE_REPOSITORY_CODES:
                logger.error(
                    "Research Agent 仓储执行失败 task_id=%s error_code=%s",
                    lease.state.task_id,
                    exc.code,
                    exc_info=True,
                )
                stale_code = self._fail_safely(lease, exc.code, run_id)
                if stale_code is not None:
                    return TaskWorkerResult(
                        status="stale",
                        task_id=lease.state.task_id,
                        error_code=stale_code,
                    )
                return TaskWorkerResult(
                    status="failed", task_id=lease.state.task_id, error_code=exc.code
                )
            return TaskWorkerResult(
                status="stale", task_id=lease.state.task_id, error_code=exc.code
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or "task_agent_failed"
            if (
                code == _STALE_CHECKPOINT_CODE
                or isinstance(exc, ToolExecutionError) and exc.code in _STALE_TOOL_CODES
            ):
                # 失租 Worker 禁止更新业务状态：未完成调用由新 Worker 恢复重执行。
                logger.error(
                    "Research Agent 遭遇失租错误 task_id=%s error_code=%s",
                    lease.state.task_id,
                    code,
                )
                return TaskWorkerResult(
                    status="stale", task_id=lease.state.task_id, error_code=code
                )
            logger.error(
                "Research Agent 执行失败 task_id=%s error_code=%s",
                lease.state.task_id,
                code,
                exc_info=True,
            )
            stale_code = self._fail_safely(lease, code, run_id)
            if stale_code is not None:
                return TaskWorkerResult(
                    status="stale",
                    task_id=lease.state.task_id,
                    error_code=stale_code,
                )
            return TaskWorkerResult(
                status="failed", task_id=lease.state.task_id, error_code=code
            )

    def _advance(
        self,
        lease: TaskLease,
        *,
        run_id: str | None = None,
        resume_answer: str | None = None,
        resume_decision: str | None = None,
        resume_approval_id: str | None = None,
    ) -> TaskWorkerResult:
        lease = self._repository.refresh_lease(
            lease, lease_seconds=self._lease_seconds
        )
        state = lease.state
        if state.task_mode is TaskMode.RESEARCH:
            agent = self._agent_for(lease.budgets)
        elif state.task_mode is TaskMode.CONTROLLED_ACTION:
            # 未装配受控 Agent（M1 Worker）时受控任务 fail-closed，不静默降级。
            if self._controlled_agent_for is None or self._approvals is None:
                raise TaskWorkerError("task_mode_not_supported")
            agent = self._controlled_agent_for(lease.budgets)
        else:
            raise TaskWorkerError("task_mode_not_supported")
        if self._capability_name is not None and (
            state.required_capability_name != self._capability_name
            or state.required_protocol_version != self._protocol_version
        ):
            raise TaskWorkerError("task_protocol_mismatch")
        if state.deadline is not None and self._repository.current_time() >= state.deadline:
            raise TaskWorkerError("task_deadline_expired")

        run_id = run_id or uuid4().hex
        auth = self._auth_contexts.resolve(
            task_id=state.task_id,
            auth_context_ref=state.auth_context_ref,
            request_id=run_id,
        )
        context = AgentRunContext(
            task_id=state.task_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            task_mode=state.task_mode,
            worker_id=self._worker_id,
            lease_generation=lease.lease_generation,
            run_id=run_id,
            auth=auth,
        )
        configurable = {
            "thread_id": state.task_id,
            "checkpoint_tenant_id": auth.tenant_id,
            "checkpoint_worker_id": self._worker_id,
            "checkpoint_lease_generation": lease.lease_generation,
        }
        if self._capability_name is not None:
            configurable.update(
                checkpoint_required_capability_name=state.required_capability_name,
                checkpoint_required_protocol_version=state.required_protocol_version,
            )
        config = {
            "configurable": configurable,
            # 每个 middleware 节点各消耗一个 superstep，预算之外必须留出
            # 图遍历中转余量；真正的停止边界由 ModelCallLimit / ToolCallLimit /
            # ExecutionGuard 中间件按线程强制（M1_FIX_PLAN 修复 4）。
            "recursion_limit": max(
                50,
                (lease.budgets.max_model_calls + lease.budgets.max_tool_calls) * 3 + 20,
            ),
        }
        snapshot = agent.get_state(config)
        invoke_agent = True
        approval_facts: tuple[object, tuple[str, ...]] | None = None
        if resume_answer is not None:
            self._require_user_interrupt(snapshot)
            invoke_input: object = Command(resume={"answer": resume_answer})
        elif resume_decision is not None:
            # 服务端复核保存的 interrupt 与审批 checkpoint 绑定逐项一致；
            # 任一不符 fail-closed，不进入 resume（恢复侧形状纪律同批次 1）。
            approval_facts = self._require_approval_interrupt(
                lease, snapshot, auth=auth, approval_id=resume_approval_id
            )
            context = self._approval_run_context(
                lease, auth, run_id, approval_facts
            )
            invoke_input = Command(resume=_approval_resume_payload(resume_decision))
        elif snapshot.interrupts:
            return self._route_interrupt(lease, snapshot, run_id)
        elif not snapshot.values:
            invoke_input = {"messages": [HumanMessage(content=state.goal)]}
        elif snapshot.next:
            # 崩溃恢复只从同一个 checkpoint 继续，不重新投递 goal 或工具调用。
            invoke_input = None
        else:
            result = snapshot.values
            invoke_input = None
            invoke_agent = False

        if invoke_agent:
            if (
                resume_decision is None
                and resume_answer is None
                and lease.state.task_mode is TaskMode.CONTROLLED_ACTION
                and self._approvals is not None
                and snapshot.next
            ):
                # 崩溃重放（resume 已消费、interrupt_ref 已清）：从 checkpoint
                # 待执行 tool call 反查 approved 审批，重建服务端 context 与
                # 同一 D7 key；查不到则保持基础 context（最终门 fail-closed）。
                replay_facts = self._replay_approval_facts(lease, auth, snapshot)
                if replay_facts is not None:
                    context = self._approval_run_context(
                        lease, auth, run_id, replay_facts
                    )
            with _TaskLeaseHeartbeat(
                lambda: self._repository.refresh_lease(
                    lease, lease_seconds=self._lease_seconds
                ),
                self._lease_seconds,
            ):
                result = agent.invoke(invoke_input, config, context=context)
            lease = self._repository.refresh_lease(
                lease, lease_seconds=self._lease_seconds
            )
            # deadline 终检：invoke 期间过期的任务不得再进入 settle/终态提交。
            fresh_deadline = lease.state.deadline
            if (
                fresh_deadline is not None
                and self._repository.current_time() >= fresh_deadline
            ):
                raise TaskWorkerError("task_deadline_expired")

        latest = agent.get_state(config)
        if latest.interrupts:
            return self._route_interrupt(lease, latest, run_id)
        candidate = self._extract_candidate(result) or self._extract_candidate(latest.values)
        if candidate is None:
            raise TaskWorkerError("task_agent_no_result")
        candidate.require_valid()
        if candidate.status == "needs_clarification":
            # middleware 必须把它转为 interrupt；否则拒绝伪造业务暂停。
            raise TaskWorkerError("task_clarification_interrupt_missing")
        return self._settle(lease, candidate, auth, run_id, latest)

    @staticmethod
    def _require_user_interrupt(snapshot) -> None:
        interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
        if len(interrupts) != 1:
            raise TaskWorkerError("task_resume_interrupt_missing")
        payload = getattr(interrupts[0], "value", None)
        if not isinstance(payload, dict) or payload.get("kind") != "user_clarification":
            raise TaskWorkerError("task_resume_kind_invalid")

    def _require_approval_interrupt(
        self,
        lease: TaskLease,
        snapshot,
        *,
        auth: AuthContext,
        approval_id: str | None,
    ) -> tuple[object, tuple[str, ...]]:
        """恢复前复核：保存的 interrupt 形状与审批 checkpoint 绑定必须逐项一致。"""
        if self._approvals is None or not approval_id:
            raise TaskWorkerError("task_mode_not_supported")
        facts, evidence_ids = self._approvals.load_action_facts(
            auth=auth, approval_id=approval_id
        )
        interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
        if len(interrupts) != 1:
            raise TaskWorkerError("task_interrupt_shape_invalid")
        payload = getattr(interrupts[0], "value", None)
        if not (
            isinstance(payload, dict)
            and set(payload) == {"action_requests", "review_configs"}
        ):
            raise TaskWorkerError("task_interrupt_kind_invalid")
        action_requests = payload.get("action_requests")
        if (
            not isinstance(action_requests, list)
            or len(action_requests) != 1
            or not isinstance(action_requests[0], dict)
        ):
            raise TaskWorkerError("task_interrupt_shape_invalid")
        action = action_requests[0]
        tool_name = action.get("name")
        arguments = action.get("args")
        if tool_name != facts.tool_name or not isinstance(arguments, dict):
            raise TaskWorkerError("task_approval_action_invalid")
        configurable = dict(
            (getattr(snapshot, "config", None) or {}).get("configurable") or {}
        )
        if (
            configurable.get("thread_id") != lease.state.task_id
            or configurable.get("thread_id") != facts.thread_id
            or configurable.get("checkpoint_ns") != facts.checkpoint_ns
            or configurable.get("checkpoint_id") != facts.checkpoint_id
            or getattr(interrupts[0], "id", None) != facts.interrupt_id
        ):
            raise TaskWorkerError("task_checkpoint_binding_invalid")
        tool_call_id = self._resolve_action_tool_call_id(
            snapshot, tool_name=facts.tool_name, arguments=arguments
        )
        if tool_call_id != facts.tool_call_id:
            raise TaskWorkerError("task_checkpoint_binding_invalid")
        return (facts, evidence_ids)

    def _replay_approval_facts(
        self,
        lease: TaskLease,
        auth: AuthContext,
        snapshot,
    ) -> tuple[object, tuple[str, ...]] | None:
        """从 checkpoint 待执行 WRITE tool call 反查 approved 审批事实。

        与 _pause_for_approval 同一形状纪律：恰好一个 WRITE tool call、
        checkpoint 坐标完整；任一不符返回 None（不猜测）。
        """
        values = getattr(snapshot, "values", None)
        messages = values.get("messages") if isinstance(values, dict) else None
        if not isinstance(messages, (list, tuple)):
            return None
        pending: dict[str, object] | None = None
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            calls = [
                call
                for call in getattr(message, "tool_calls", ()) or ()
                if isinstance(call, dict)
                and call.get("name") in CONTROLLED_WRITE_TOOL_NAMES
            ]
            if len(calls) > 1:
                return None
            if len(calls) == 1:
                pending = calls[0]
                break
        if pending is None:
            return None
        configurable = dict(
            (getattr(snapshot, "config", None) or {}).get("configurable") or {}
        )
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        checkpoint_id = configurable.get("checkpoint_id")
        tool_call_id = pending.get("id")
        if (
            thread_id != lease.state.task_id
            or not isinstance(checkpoint_ns, str)
            or not isinstance(checkpoint_id, str)
            or not checkpoint_id.strip()
            or not isinstance(tool_call_id, str)
            or not tool_call_id.strip()
        ):
            return None
        assert self._approvals is not None
        return self._approvals.load_approved_for_replay(
            auth=auth,
            task_id=lease.state.task_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            tool_call_id=tool_call_id,
        )

    def _approval_run_context(
        self,
        lease: TaskLease,
        auth: AuthContext,
        run_id: str,
        approval_facts: tuple[object, tuple[str, ...]],
    ) -> AgentRunContext:
        """受控 resume 的服务端 context：审批事实 + D7 稳定 action key。"""
        facts, evidence_ids = approval_facts
        idempotency_key = stable_write_action_key(
            task_id=lease.state.task_id,
            thread_id=facts.thread_id,
            checkpoint_ns=facts.checkpoint_ns,
            checkpoint_id=facts.checkpoint_id,
            interrupt_id=facts.interrupt_id,
            tool_call_id=facts.tool_call_id,
            tool_name=facts.tool_name,
            tool_version=facts.tool_version,
            arguments_hash=facts.canonical_arguments_hash,
            evidence_digest=facts.evidence_hash,
        )
        return AgentRunContext(
            task_id=lease.state.task_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            task_mode=lease.state.task_mode,
            worker_id=self._worker_id,
            lease_generation=lease.lease_generation,
            run_id=run_id,
            auth=auth,
            approval_id=facts.approval_id,
            plan_revision=facts.plan_revision,
            plan_digest=facts.plan_digest,
            evidence_digest=facts.evidence_hash,
            evidence_ids=evidence_ids,
            idempotency_key=idempotency_key,
        )

    def _pause_for_interrupt(self, lease: TaskLease, snapshot, run_id: str) -> TaskWorkerResult:
        self._require_user_interrupt(snapshot)
        payload = snapshot.interrupts[0].value
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise TaskWorkerError("task_interrupt_payload_invalid")
        paused = self._repository.pause_for_user(
            lease, question=question, trace_id=run_id
        )
        return TaskWorkerResult(status="waiting_for_user", task_id=paused.task_id)

    def _route_interrupt(
        self, lease: TaskLease, snapshot, run_id: str
    ) -> TaskWorkerResult:
        """按保存的 interrupt 形状严格分流：clarification 与 approval 互斥。

        未知形状、多 interrupt、多 action、未知工具一律 fail-closed，
        不进入任何恢复入口。
        """
        interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
        if len(interrupts) != 1:
            raise TaskWorkerError("task_interrupt_shape_invalid")
        payload = getattr(interrupts[0], "value", None)
        if isinstance(payload, dict) and payload.get("kind") == "user_clarification":
            return self._pause_for_interrupt(lease, snapshot, run_id)
        if (
            isinstance(payload, dict)
            and set(payload) == {"action_requests", "review_configs"}
        ):
            return self._pause_for_approval(lease, snapshot, run_id)
        raise TaskWorkerError("task_interrupt_kind_invalid")

    def _pause_for_approval(
        self, lease: TaskLease, snapshot, run_id: str
    ) -> TaskWorkerResult:
        if lease.state.task_mode is not TaskMode.CONTROLLED_ACTION:
            # HITL 审批形状只允许出现在受控任务上；Research 线程出现即 fail-closed。
            raise TaskWorkerError("task_approval_mode_invalid")
        assert self._approvals is not None
        interrupt = snapshot.interrupts[0]
        payload = interrupt.value
        action_requests = payload.get("action_requests")
        review_configs = payload.get("review_configs")
        if (
            not isinstance(action_requests, list)
            or len(action_requests) != 1
            or not isinstance(review_configs, list)
            or len(review_configs) != 1
        ):
            raise TaskWorkerError("task_interrupt_shape_invalid")
        action = action_requests[0]
        review = review_configs[0]
        if not isinstance(action, dict) or not isinstance(review, dict):
            raise TaskWorkerError("task_interrupt_shape_invalid")
        tool_name = action.get("name")
        arguments = action.get("args")
        if (
            not isinstance(tool_name, str)
            or tool_name not in CONTROLLED_WRITE_TOOL_NAMES
            or not isinstance(arguments, dict)
        ):
            raise TaskWorkerError("task_approval_action_invalid")
        if (
            review.get("action_name") != tool_name
            or set(review.get("allowed_decisions") or ()) != _HITL_ALLOWED_DECISIONS
        ):
            raise TaskWorkerError("task_approval_review_config_invalid")
        tool_call_id = self._resolve_action_tool_call_id(
            snapshot, tool_name=tool_name, arguments=arguments
        )
        interrupt_id = getattr(interrupt, "id", None)
        if not isinstance(interrupt_id, str) or not interrupt_id.strip():
            raise TaskWorkerError("task_interrupt_id_invalid")
        configurable = dict(
            (getattr(snapshot, "config", None) or {}).get("configurable") or {}
        )
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        checkpoint_id = configurable.get("checkpoint_id")
        if (
            thread_id != lease.state.task_id
            or not isinstance(checkpoint_ns, str)
            or not isinstance(checkpoint_id, str)
            or not checkpoint_id.strip()
        ):
            raise TaskWorkerError("task_checkpoint_binding_invalid")
        binding = ApprovalCheckpointBinding(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            interrupt_id=interrupt_id,
            tool_call_id=tool_call_id,
        )
        auth = self._auth_contexts.resolve(
            task_id=lease.state.task_id,
            auth_context_ref=lease.state.auth_context_ref,
            request_id=run_id,
        )
        action_binding = self._approvals.ensure_action_binding(
            auth=auth,
            task_id=lease.state.task_id,
            tool_name=tool_name,
            arguments=arguments,
            checkpoint_binding=binding,
        )
        expires_at = datetime.now(UTC) + timedelta(seconds=_APPROVAL_EXPIRY_SECONDS)
        deadline = lease.state.deadline
        if deadline is not None:
            # 审批有效期不得越过任务 deadline（request 内会再次强制）。
            expires_at = min(expires_at, deadline)
        approval = self._approvals.request(
            auth=auth,
            task_id=lease.state.task_id,
            plan_revision=action_binding.plan_revision,
            plan_digest=action_binding.plan_digest,
            evidence_digest=action_binding.evidence_digest,
            evidence_ids=action_binding.evidence_ids,
            tool_name=tool_name,
            arguments=arguments,
            expires_at=expires_at,
            checkpoint_binding=binding,
        )
        paused = self._repository.pause_for_approval(
            lease,
            interrupt_ref=approval.approval_id,
            summary=self._render_approval_summary(tool_name, arguments),
            trace_id=run_id,
        )
        return TaskWorkerResult(status="waiting_for_approval", task_id=paused.task_id)

    @staticmethod
    def _resolve_action_tool_call_id(
        snapshot, *, tool_name: str, arguments: dict[str, object]
    ) -> str:
        """HITL payload 不携带 tool_call_id；从保存的最后一条 AI tool call 派生。

        按 (name, args) 精确匹配；无匹配即 fail-closed（payload 与 checkpoint
        事实不一致，可能是篡改或版本漂移）。
        """
        values = getattr(snapshot, "values", None)
        messages = values.get("messages") if isinstance(values, dict) else None
        if not isinstance(messages, (list, tuple)):
            raise TaskWorkerError("task_approval_tool_call_missing")
        from langchain_core.messages import AIMessage

        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            for call in getattr(message, "tool_calls", ()) or ():
                if (
                    isinstance(call, dict)
                    and call.get("name") == tool_name
                    and call.get("args") == arguments
                    and isinstance(call.get("id"), str)
                    and call["id"].strip()
                ):
                    return call["id"]
        raise TaskWorkerError("task_approval_tool_call_missing")

    @staticmethod
    def _render_approval_summary(
        tool_name: str, arguments: dict[str, object]
    ) -> str:
        """服务端裁剪摘要：只含工具名与标题，不复制正文或完整参数。"""
        title = arguments.get("title")
        safe_title = title.strip()[:80] if isinstance(title, str) else ""
        label = {
            "create_report_artifact": "报告产物",
            "create_work_item_draft": "工作项草稿",
        }.get(tool_name, "受控写操作")
        return f"等待人工审批：{label}「{safe_title}」" if safe_title else f"等待人工审批：{label}"

    @staticmethod
    def _extract_candidate(result: object) -> TaskResultCandidate | None:
        if not isinstance(result, dict):
            return None
        raw = result.get("structured_response")
        if isinstance(raw, TaskResultCandidate):
            return raw
        if isinstance(raw, dict):
            try:
                return TaskResultCandidate.model_validate(raw)
            except ValueError:
                return None
        return None

    def _settle(
        self,
        lease: TaskLease,
        candidate: TaskResultCandidate,
        auth: AuthContext,
        run_id: str,
        agent_snapshot: object,
    ) -> TaskWorkerResult:
        # citation 复核必须使用执行结束后的最新服务端身份，而非 invoke 前快照。
        auth = self._auth_contexts.resolve(
            task_id=lease.state.task_id,
            auth_context_ref=lease.state.auth_context_ref,
            request_id=run_id,
        )
        citations = ()
        if candidate.status == "succeeded":
            try:
                snapshot = canonical_evidence_snapshot(
                    candidate.evidence_ids, require_non_empty=True
                )
            except ValueError as exc:
                raise TaskWorkerError("task_evidence_candidate_invalid") from exc
            with self._database.sessions() as session:
                try:
                    citations = self._evidence.citations_from_ledger(
                        session,
                        task_id=lease.state.task_id,
                        evidence_ids=snapshot.evidence_ids,
                        auth=auth,
                    )
                except EvidenceLedgerConflict as exc:
                    raise TaskWorkerError("task_evidence_revalidation_failed") from exc

        if lease.state.task_mode is TaskMode.CONTROLLED_ACTION:
            artifacts = ()
            if candidate.artifact_ids:
                artifacts = self._verified_artifacts(
                    tenant_id=auth.tenant_id,
                    task_id=lease.state.task_id,
                    artifact_ids=candidate.artifact_ids,
                )
            result = build_controlled_task_result(
                candidate,
                task_id=lease.state.task_id,
                citations=citations,
                artifacts=artifacts,
            )
        else:
            result = build_task_result(
                candidate, task_id=lease.state.task_id, citations=citations
            )
        result_ref, result_hash = self._results.save_task_result(lease.state.task_id, result)
        status = (
            TaskStatus.SUCCEEDED
            if candidate.status == "succeeded"
            else TaskStatus.REFUSED
        )
        counts = self._terminal_counts(
            tenant_id=auth.tenant_id,
            task_id=lease.state.task_id,
            agent_snapshot=agent_snapshot,
        )
        try:
            finished = self._repository.finish_task(
                lease,
                status=status,
                result_kind=result.result_kind,
                result_ref=result_ref,
                result_hash=result_hash,
                trace_id=run_id,
                counts=counts,
            )
        except Exception:
            self._delete_unbound_task_result(result_ref, lease.state.task_id)
            raise
        if finished.status is TaskStatus.FAILED:
            self._delete_unbound_task_result(result_ref, finished.task_id)
            return TaskWorkerResult(
                status="failed",
                task_id=finished.task_id,
                error_code="task_deadline_expired",
            )
        return TaskWorkerResult(status="finished", task_id=finished.task_id)

    def _terminal_counts(
        self,
        *,
        tenant_id: str,
        task_id: str,
        agent_snapshot: object,
    ) -> dict[str, int]:
        values = getattr(agent_snapshot, "values", None)
        messages = values.get("messages", ()) if isinstance(values, dict) else ()
        model_calls = (
            sum(isinstance(message, AIMessage) for message in messages)
            if isinstance(messages, (list, tuple))
            else 0
        )
        with self._database.sessions() as session:
            tool_calls = session.scalar(
                select(func.count())
                .select_from(ToolCallRow)
                .where(
                    ToolCallRow.tenant_id == tenant_id,
                    ToolCallRow.task_id == task_id,
                )
            )
        return {"model_calls": model_calls, "tool_calls": int(tool_calls or 0)}

    def _verified_artifacts(
        self,
        *,
        tenant_id: str,
        task_id: str,
        artifact_ids: tuple[str, ...],
    ) -> tuple[ArtifactSummary, ...]:
        """settle 前逐个校验产物归属与一致性；任一失败 fail-closed。

        校验项：tenant/task 归属、status=ready、对象正文 sha256 与
        content_hash 一致。候选声明之外的产物不会被注入最终结果。
        """
        if self._objects is None:
            raise TaskWorkerError("task_artifact_binding_invalid")
        with self._database.sessions() as session:
            rows = {
                row.artifact_id: row
                for row in session.scalars(
                    select(ArtifactRow).where(
                        ArtifactRow.tenant_id == tenant_id,
                        ArtifactRow.task_id == task_id,
                        ArtifactRow.artifact_id.in_(artifact_ids),
                    )
                )
            }
        summaries: list[ArtifactSummary] = []
        for artifact_id in artifact_ids:
            row = rows.get(artifact_id)
            if row is None or row.status != "ready":
                raise TaskWorkerError("task_artifact_binding_invalid")
            try:
                content = self._objects.read(row.storage_key)
            except (OSError, ValueError):
                raise TaskWorkerError("task_artifact_binding_invalid") from None
            digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
            if digest != row.content_hash:
                raise TaskWorkerError("task_artifact_binding_invalid")
            summaries.append(
                ArtifactSummary(
                    artifact_id=row.artifact_id,
                    artifact_type=row.artifact_type,
                    content_hash=row.content_hash,
                    status="ready",
                )
            )
        return tuple(summaries)

    def _delete_unbound_task_result(self, result_ref: str, task_id: str) -> None:
        if self._task_result_is_bound(result_ref, task_id):
            return
        try:
            self._results.delete_task_result(result_ref)
        except OSError:
            logger.error("未绑定任务结果补偿删除失败 task_id=%s", task_id)

    def _task_result_is_bound(self, result_ref: str, task_id: str) -> bool:
        try:
            with self._database.sessions.begin() as session:
                row = session.scalar(
                    select(AgentTaskRow)
                    .where(AgentTaskRow.task_id == task_id)
                    .with_for_update()
                )
                return row is not None and row.result_ref == result_ref
        except Exception:
            # 提交确认丢失时不能猜测未绑定；保留对象交给孤儿扫描收敛。
            logger.error("任务结果绑定状态确认失败 task_id=%s", task_id)
            return True

    def _fail_safely(self, lease: TaskLease, code: str, run_id: str) -> str | None:
        try:
            self._repository.finish_task(
                lease,
                status=TaskStatus.FAILED,
                error_code=code[:100],
                trace_id=run_id,
            )
        except TaskRepositoryError as exc:
            if exc.code in _STALE_REPOSITORY_CODES:
                return exc.code
            logger.error("Research Agent 终态同步失败 task_id=%s", lease.state.task_id)
        except Exception:
            logger.error("Research Agent 终态同步失败 task_id=%s", lease.state.task_id)
        return None
