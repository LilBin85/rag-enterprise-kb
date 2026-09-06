"""任务 worker 持久化生命周期的稳定应用契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from rag_kb.application.ports.task_read import ArtifactSummary
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.resume_commands import ResumeCommand, ResumeCommandCandidate
from rag_kb.domain.tasks import TaskBudgets, TaskState, TaskStatus
from rag_kb.models import Citation


class TaskWorkerPersistenceError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TaskLease:
    state: TaskState
    worker_id: str
    lease_generation: int
    budgets: TaskBudgets
    pending_traceparent: str | None = None
    pending_tracestate: str | None = None


@dataclass(frozen=True)
class ResumeCommandLease:
    """同一事务领取的 task/command 双 lease；正文不参与 repr。"""

    command: ResumeCommand = field(repr=False)
    task_lease: TaskLease = field(repr=False)
    worker_id: str
    command_lease_generation: int


class TaskWorkerLifecyclePort(Protocol):
    def resume_command_candidates(
        self, *, limit: int
    ) -> tuple[ResumeCommandCandidate, ...]: ...

    def next_claimable_task_eligible_at(
        self,
        *,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> datetime | None: ...

    def claim_resume_command(
        self,
        *,
        command_id: str,
        worker_id: str,
        lease_seconds: int,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> ResumeCommandLease | None: ...

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> TaskLease | None: ...

    def enqueue_user_resume_command(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        answer: str,
    ) -> ResumeCommand: ...

    def validate_resume_command_execution(
        self,
        lease: ResumeCommandLease,
        *,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> ResumeCommand: ...

    def refresh_resume_command_lease(
        self, lease: ResumeCommandLease, *, lease_seconds: int
    ) -> ResumeCommandLease: ...

    def refresh_lease(
        self, lease: TaskLease, *, lease_seconds: int | None = None
    ) -> TaskLease: ...

    def current_time(self) -> datetime: ...

    def consume_resume_command(self, lease: ResumeCommandLease) -> ResumeCommand: ...

    def pause_for_user(
        self,
        lease: TaskLease,
        *,
        question: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        interrupt_id: str,
        run_id: str,
    ) -> TaskState: ...

    def pause_for_approval(
        self,
        lease: TaskLease,
        *,
        interrupt_ref: str,
        summary: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        interrupt_id: str,
        run_id: str,
    ) -> TaskState: ...

    def finish_task(
        self,
        lease: TaskLease,
        *,
        status: TaskStatus,
        result_kind: str | None = None,
        result_ref: str | None = None,
        result_hash: str | None = None,
        error_code: str | None = None,
        run_id: str,
        auth_request_id: str | None = None,
        counts: dict[str, int] | None = None,
        conversation_message: str | None = None,
        conversation_evidence_ids: tuple[str, ...] = (),
        conversation_artifacts: tuple[dict[str, object], ...] = (),
    ) -> TaskState: ...

    def settle_applied_resume_command(self, *, command_id: str) -> ResumeCommand | None: ...

    def fail_resume_command(
        self, lease: ResumeCommandLease, *, failure_code: str
    ) -> ResumeCommand: ...

    def expire_claimed_resume_command(
        self, lease: ResumeCommandLease, *, failure_code: str = "task_resume_expired"
    ) -> ResumeCommand: ...

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int: ...


class TaskWorkerSettlementPort(Protocol):
    def revalidate_citations(
        self,
        *,
        task_id: str,
        evidence_ids: tuple[str, ...],
        auth: AuthContext,
    ) -> tuple[Citation, ...]: ...

    def tool_call_count(self, *, tenant_id: str, task_id: str) -> int: ...

    def verify_artifacts(
        self,
        *,
        tenant_id: str,
        task_id: str,
        artifact_ids: tuple[str, ...],
    ) -> tuple[ArtifactSummary, ...]: ...

    def is_task_result_bound(self, *, task_id: str, result_ref: str) -> bool: ...
