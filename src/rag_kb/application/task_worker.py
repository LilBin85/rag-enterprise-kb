"""checkpointer 驱动的 Research Agent worker。"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Protocol
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy import or_, select

from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.result import (
    AgentResultStore,
    TaskResultCandidate,
    build_task_result,
)
from rag_kb.application.tool_execution_service import (
    _FATAL_TOOL_CODES,
    ToolExecutionError,
    ToolExecutionService,
)
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import AgentTaskRow
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
    TaskRepositoryError,
)

logger = logging.getLogger(__name__)


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
        agent: Any,
        results: AgentResultStore,
        evidence: EvidenceLedgerRepository,
        tools: ToolExecutionService,
        database,
        worker_id: str,
        lease_seconds: int = 300,
        close=None,
        now=None,
        checkpointer=None,
    ) -> None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_worker")
        self._repository = repository
        self._auth_contexts = auth_contexts
        self._agent = agent
        self._results = results
        self._evidence = evidence
        self._tools = tools
        self._database = database
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._close = close
        self._now = now or (lambda: datetime.now(UTC))
        self._checkpointer = checkpointer

    def close(self) -> None:
        close_tools = getattr(self._tools, "close", None)
        if close_tools is not None:
            close_tools()
        if self._close is not None:
            self._close()
            self._close = None

    def run_once(self) -> TaskWorkerResult:
        lease = self._repository.claim_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
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
        lease = self._repository.claim_user_resume(
            auth=auth,
            task_id=task_id,
            expected_state_version=expected_state_version,
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        return self._run_claimed(lease, resume_answer=answer.strip())

    def expire_stale_tool_calls(self, *, limit: int = 100) -> int:
        return self._tools.expire_stale_running(limit=limit)

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int:
        return self._repository.expire_stale_knowledge_queries(limit=limit)

    def purge_expired_tool_results(self, *, limit: int = 100) -> int:
        purge = getattr(self._tools, "purge_expired_tool_results", None)
        if purge is None:
            return 0
        return purge(limit=limit)

    def purge_finished_task_threads(self, *, limit: int = 100) -> int:
        """终态任务超过保留期后删除 checkpoint 线程与结果正文（幂等可重扫）。"""
        from rag_kb.config import get_settings
        from rag_kb.domain.tasks import TaskStatus
        from rag_kb.infrastructure.database.unit_of_work import UnitOfWork

        retention = timedelta(days=get_settings().task_result_retention_days)
        cutoff = self._now() - retention
        terminal = tuple(
            item.value
            for item in TaskStatus
            if item
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.REFUSED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.UNKNOWN,
            }
        )
        purged = 0
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            rows = tuple(
                work.session.scalars(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.status.in_(terminal),
                        AgentTaskRow.updated_at <= cutoff,
                        or_(
                            AgentTaskRow.checkpoint_purged_at.is_(None),
                            AgentTaskRow.result_purged_at.is_(None),
                        ),
                    )
                    .order_by(AgentTaskRow.updated_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for row in rows:
                now = self._now()
                if row.checkpoint_purged_at is None:
                    if self._checkpointer is None:
                        raise ValueError("task_thread_purger_required")
                    self._checkpointer.delete_thread(row.task_id)
                    row.checkpoint_purged_at = now
                if row.result_purged_at is None:
                    if row.result_ref:
                        self._results.delete_task_result(row.result_ref)
                        row.result_ref = None
                    row.result_purged_at = now
                purged += 1
        return purged

    def _run_claimed(
        self,
        lease: TaskLease,
        *,
        resume_answer: str | None = None,
    ) -> TaskWorkerResult:
        try:
            return self._advance(lease, resume_answer=resume_answer)
        except TaskRepositoryError as exc:
            return TaskWorkerResult(
                status="stale", task_id=lease.state.task_id, error_code=exc.code
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or "task_agent_failed"
            if isinstance(exc, ToolExecutionError) and exc.code in _FATAL_TOOL_CODES:
                # 失租 Worker 禁止更新业务状态：致命码只上报，不改写终态；
                # 未完成调用由新 Worker 从 checkpoint 恢复重执行。
                logger.error(
                    "Research Agent 遭遇致命工具错误 task_id=%s error_code=%s",
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
            self._fail_safely(lease, code)
            return TaskWorkerResult(
                status="failed", task_id=lease.state.task_id, error_code=code
            )

    def _advance(
        self,
        lease: TaskLease,
        *,
        resume_answer: str | None,
    ) -> TaskWorkerResult:
        lease = self._repository.refresh_lease(
            lease, lease_seconds=self._lease_seconds
        )
        state = lease.state
        if state.task_mode is not TaskMode.RESEARCH:
            raise TaskWorkerError("task_mode_not_supported")
        if state.deadline is not None and self._now() >= state.deadline:
            raise TaskWorkerError("task_deadline_expired")

        run_id = uuid4().hex
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
        config = {
            "configurable": {"thread_id": state.task_id},
            # 每个 middleware 节点各消耗一个 superstep，预算之外必须留出
            # 图遍历中转余量；真正的停止边界由 ModelCallLimit / ToolCallLimit /
            # ExecutionGuard 中间件按线程强制（M1_FIX_PLAN 修复 4）。
            "recursion_limit": max(
                50,
                (lease.budgets.max_model_calls + lease.budgets.max_tool_calls) * 3 + 20,
            ),
        }
        snapshot = self._agent.get_state(config)
        invoke_agent = True
        if resume_answer is not None:
            self._require_user_interrupt(snapshot)
            invoke_input: object = Command(resume={"answer": resume_answer})
        elif snapshot.interrupts:
            return self._pause_for_interrupt(lease, snapshot, run_id)
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
            with _TaskLeaseHeartbeat(
                lambda: self._repository.refresh_lease(
                    lease, lease_seconds=self._lease_seconds
                ),
                self._lease_seconds,
            ):
                result = self._agent.invoke(invoke_input, config, context=context)
            lease = self._repository.refresh_lease(
                lease, lease_seconds=self._lease_seconds
            )
            # deadline 终检：invoke 期间过期的任务不得再进入 settle/终态提交。
            fresh_deadline = lease.state.deadline
            if fresh_deadline is not None and self._now() >= fresh_deadline:
                raise TaskWorkerError("task_deadline_expired")

        latest = self._agent.get_state(config)
        if latest.interrupts:
            return self._pause_for_interrupt(lease, latest, run_id)
        candidate = self._extract_candidate(result) or self._extract_candidate(latest.values)
        if candidate is None:
            raise TaskWorkerError("task_agent_no_result")
        candidate.require_valid()
        if candidate.status == "needs_clarification":
            # middleware 必须把它转为 interrupt；否则拒绝伪造业务暂停。
            raise TaskWorkerError("task_clarification_interrupt_missing")
        return self._settle(lease, candidate, auth, run_id)

    @staticmethod
    def _require_user_interrupt(snapshot) -> None:
        interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
        if len(interrupts) != 1:
            raise TaskWorkerError("task_resume_interrupt_missing")
        payload = getattr(interrupts[0], "value", None)
        if not isinstance(payload, dict) or payload.get("kind") != "user_clarification":
            raise TaskWorkerError("task_resume_kind_invalid")

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

        result = build_task_result(
            candidate, task_id=lease.state.task_id, citations=citations
        )
        result_ref, result_hash = self._results.save_task_result(lease.state.task_id, result)
        status = (
            TaskStatus.SUCCEEDED
            if candidate.status == "succeeded"
            else TaskStatus.REFUSED
        )
        finished = self._repository.finish_task(
            lease,
            status=status,
            result_kind=result.result_kind,
            result_ref=result_ref,
            result_hash=result_hash,
            trace_id=run_id,
        )
        return TaskWorkerResult(status="finished", task_id=finished.task_id)

    def _fail_safely(self, lease: TaskLease, code: str) -> None:
        try:
            self._repository.finish_task(
                lease,
                status=TaskStatus.FAILED,
                error_code=code[:100],
                trace_id=code[:64],
            )
        except Exception:
            logger.error("Research Agent 终态同步失败 task_id=%s", lease.state.task_id)
