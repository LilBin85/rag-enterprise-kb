"""可恢复单智能体任务的 PostgreSQL/SQLite 持久仓储与租约栅栏。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, or_, select

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.domain.tasks import (
    PendingActionBinding,
    TaskBudgets,
    TaskMode,
    TaskNode,
    TaskState,
    TaskStatus,
)
from rag_kb.infrastructure.database.models import (
    ApprovalRow,
    AgentTaskRow,
    PlanRevisionRow,
    TaskEventRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


_TERMINAL_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.REFUSED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.UNKNOWN,
}


class TaskRepositoryError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TaskLease:
    state: TaskState
    worker_id: str
    lease_generation: int
    budgets: TaskBudgets


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _pending_action_from_row(row: AgentTaskRow) -> PendingActionBinding | None:
    values = (
        row.pending_action_ref,
        row.pending_action_hash,
        row.pending_action_kind,
        row.pending_action_expires_at,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise TaskRepositoryError("task_pending_action_binding_corrupt")
    return PendingActionBinding(
        storage_ref=row.pending_action_ref,
        content_hash=row.pending_action_hash,
        kind=row.pending_action_kind,
        expires_at=_aware(row.pending_action_expires_at),
    )


def _state_from_row(row: AgentTaskRow) -> TaskState:
    budgets = TaskBudgets.model_validate(row.budgets)
    return TaskState(
        task_id=row.task_id,
        auth_context_ref=row.auth_context_ref,
        goal=row.goal,
        task_mode=TaskMode(row.task_mode),
        status=TaskStatus(row.status),
        current_node=TaskNode(row.current_node),
        state_version=row.state_version,
        lease_generation=row.lease_generation,
        plan_revision=row.current_plan_revision,
        current_step=row.current_step,
        observations=tuple({"ref": item} for item in row.observation_refs),
        evidence_ids=tuple(row.evidence_refs),
        pending_question=row.pending_question,
        pending_approval=row.pending_approval_id,
        pending_action=_pending_action_from_row(row),
        artifacts=tuple(row.artifact_refs),
        model_calls=row.model_calls,
        tool_calls=row.tool_calls,
        replans=row.replans,
        consecutive_failures=row.consecutive_failures,
        deadline=_aware(row.deadline),
        token_budget=budgets.max_context_tokens,
        token_usage=row.token_usage,
        last_error=row.last_error_code,
        repeated_action_fingerprints=tuple(row.repeated_action_fingerprints),
        query_fingerprints=tuple(row.query_fingerprints),
        result_fingerprints=tuple(row.result_fingerprints),
        no_new_evidence_count=row.no_new_evidence_count,
        cancelled=row.cancelled,
    )


class DatabaseTaskRepository:
    def __init__(self, database: Database, *, now=None) -> None:
        self._database = database
        self._now = now or (lambda: datetime.now(UTC))

    def create_task(
        self,
        *,
        auth: AuthContext,
        goal: str,
        task_mode: TaskMode,
        budgets: TaskBudgets | None = None,
        deadline: datetime | None = None,
        task_id: str | None = None,
    ) -> TaskState:
        if not goal.strip():
            raise ValueError("task_goal_required")
        configured = budgets or TaskBudgets()
        deadline = _aware(deadline)
        if deadline is None:
            deadline = self._now() + timedelta(seconds=configured.max_runtime_seconds)
        if deadline is not None and deadline <= self._now():
            raise TaskRepositoryError("task_deadline_expired")
        auth_ref = canonical_fingerprint(
            "auth_context",
            {
                "tenant_id": auth.tenant_id,
                "user_id": auth.user_id,
                "roles": sorted(auth.roles),
                "groups": sorted(auth.groups),
                "attributes": auth.attributes,
            },
        )
        row = AgentTaskRow(
            task_id=task_id or uuid4().hex,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            task_mode=task_mode.value,
            status=TaskStatus.PLANNING.value,
            current_node=TaskNode.LOAD_TASK.value,
            goal=goal.strip(),
            auth_context_ref=auth_ref,
            budgets=configured.model_dump(mode="json"),
            deadline=deadline,
            available_at=self._now(),
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            work.session.add(row)
            work.session.flush()
            row.checkpoint_digest = self._checkpoint_digest(row)
            self._append_event(
                work.session,
                row,
                event_type="task_created",
                trace_id=auth.request_id,
                run_id=auth.request_id,
            )
            return _state_from_row(row)

    def get_task(self, *, auth: AuthContext, task_id: str) -> TaskState:
        with self._database.sessions() as session:
            row = session.scalar(
                select(AgentTaskRow).where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.user_id == auth.user_id,
                )
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            return _state_from_row(row)

    def claim_next(self, *, worker_id: str, lease_seconds: int) -> TaskLease | None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_claim")
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.status.in_(
                        (TaskStatus.PLANNING.value, TaskStatus.RUNNING.value)
                    ),
                    AgentTaskRow.task_mode != TaskMode.KNOWLEDGE_QUERY.value,
                    AgentTaskRow.cancelled.is_(False),
                    AgentTaskRow.available_at <= now,
                    or_(
                        AgentTaskRow.lease_owner.is_(None),
                        AgentTaskRow.lease_expires_at <= now,
                    ),
                )
                .order_by(AgentTaskRow.available_at, AgentTaskRow.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            row.lease_owner = worker_id
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.lease_generation += 1
            row.status = TaskStatus.RUNNING.value
            row.updated_at = now
            work.session.flush()
            return TaskLease(
                state=_state_from_row(row),
                worker_id=worker_id,
                lease_generation=row.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
            )

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int:
        """同步 API 崩溃后只做终态收敛，不交给通用 planner 重放。"""
        if limit <= 0:
            raise ValueError("invalid_sync_query_reaper_limit")
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            rows = tuple(
                work.session.scalars(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.task_mode == "knowledge_query",
                        AgentTaskRow.status == TaskStatus.RUNNING.value,
                        AgentTaskRow.lease_expires_at.is_not(None),
                        AgentTaskRow.lease_expires_at <= now,
                    )
                    .order_by(AgentTaskRow.lease_expires_at, AgentTaskRow.task_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for row in rows:
                row.status = TaskStatus.FAILED.value
                row.current_node = TaskNode.TERMINAL.value
                row.last_error_code = "knowledge_query_lease_expired"
                row.lease_owner = None
                row.lease_expires_at = None
                row.state_version += 1
                row.updated_at = now
            return len(rows)

    def claim_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> TaskLease | None:
        """领取明确任务；并发 continue/worker 不会误领其他租户的工作。"""
        if not task_id or not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_claim")
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.status.in_(
                        (TaskStatus.PLANNING.value, TaskStatus.RUNNING.value)
                    ),
                    AgentTaskRow.cancelled.is_(False),
                    AgentTaskRow.available_at <= now,
                    or_(
                        AgentTaskRow.lease_owner.is_(None),
                        AgentTaskRow.lease_expires_at <= now,
                    ),
                )
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            row.lease_owner = worker_id
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.lease_generation += 1
            row.status = TaskStatus.RUNNING.value
            row.updated_at = now
            work.session.flush()
            return TaskLease(
                state=_state_from_row(row),
                worker_id=worker_id,
                lease_generation=row.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
            )

    def bootstrap_initial_plan(
        self,
        lease: TaskLease,
        *,
        trace_id: str,
        run_id: str,
    ) -> TaskLease:
        """以固定的 LOAD→CLASSIFY→UPDATE→SELECT 序列创建首个可执行计划。

        这一步不调用模型，因此新任务不会在尚无计划绑定时尝试执行模型动作。
        """

        if lease.state.plan_revision > 0:
            return lease
        if lease.state.current_node is not TaskNode.LOAD_TASK:
            raise TaskRepositoryError("task_initial_plan_state_invalid")
        initial_plan = {
            "schema_version": 1,
            "steps": ["classify_goal", "select_action"],
            "evidence_ids": [],
        }
        return self._persist_plan_revision(
            lease,
            plan=initial_plan,
            revision=1,
            replans=lease.state.replans,
            current_node=TaskNode.SELECT_ACTION,
            event_type="plan_created",
            trace_id=trace_id,
            run_id=run_id,
        )

    def create_plan_revision(
        self,
        lease: TaskLease,
        state: TaskState,
        *,
        steps: tuple[str, ...],
        trace_id: str,
        run_id: str,
    ) -> TaskLease:
        """把模型给出的重规划落为不可变 revision，并使其成为当前唯一 active 计划。"""

        if state.task_id != lease.state.task_id or state.state_version != lease.state.state_version:
            raise TaskRepositoryError("task_state_conflict")
        normalized_steps = tuple(step.strip() for step in steps)
        if not normalized_steps or any(not step for step in normalized_steps):
            raise TaskRepositoryError("task_plan_steps_invalid")
        if state.plan_revision <= 0:
            raise TaskRepositoryError("task_plan_binding_missing")
        plan = {
            "schema_version": 1,
            "steps": list(normalized_steps),
            "evidence_ids": list(state.evidence_ids),
        }
        return self._persist_plan_revision(
            lease,
            plan=plan,
            revision=state.plan_revision + 1,
            replans=state.replans + 1,
            current_node=TaskNode.SELECT_ACTION,
            event_type="plan_revised",
            trace_id=trace_id,
            run_id=run_id,
        )

    def _persist_plan_revision(
        self,
        lease: TaskLease,
        *,
        plan: dict[str, object],
        revision: int,
        replans: int,
        current_node: TaskNode,
        event_type: str,
        trace_id: str,
        run_id: str,
    ) -> TaskLease:
        """在租约 CAS 内写 revision，避免 current revision 指向不存在的计划。"""

        evidence_ids = plan.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
            raise TaskRepositoryError("task_plan_evidence_invalid")
        snapshot = canonical_evidence_snapshot(evidence_ids)
        plan = {**plan, "evidence_ids": list(snapshot.evidence_ids)}
        plan_digest = canonical_fingerprint("task_plan", plan)
        evidence_digest = snapshot.digest
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(AgentTaskRow.task_id == lease.state.task_id)
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            self._verify_lease(row, lease, now=self._now())
            if row.state_version != lease.state.state_version:
                raise TaskRepositoryError("task_state_conflict")
            if row.current_plan_revision != lease.state.plan_revision:
                raise TaskRepositoryError("task_plan_binding_missing")
            if revision != row.current_plan_revision + 1:
                raise TaskRepositoryError("task_plan_revision_invalid")
            if revision > 1:
                previous = work.session.scalar(
                    select(PlanRevisionRow).where(
                        PlanRevisionRow.tenant_id == row.tenant_id,
                        PlanRevisionRow.task_id == row.task_id,
                        PlanRevisionRow.revision == row.current_plan_revision,
                    )
                )
                if previous is None:
                    raise TaskRepositoryError("task_plan_binding_missing")
                previous.status = "superseded"
            work.session.add(
                PlanRevisionRow(
                    plan_revision_id=uuid4().hex,
                    tenant_id=row.tenant_id,
                    task_id=row.task_id,
                    revision=revision,
                    schema_version=1,
                    plan=plan,
                    plan_digest=plan_digest,
                    evidence_digest=evidence_digest,
                    status="active",
                )
            )
            work.session.flush()
            next_state = _state_from_row(row).model_copy(
                update={
                    "plan_revision": revision,
                    "replans": replans,
                    "current_node": current_node,
                }
            )
            persisted = self.checkpoint_in_session(
                work.session,
                lease,
                next_state,
                event_type=event_type,
                trace_id=trace_id,
                run_id=run_id,
            )
            return TaskLease(
                state=persisted,
                worker_id=lease.worker_id,
                lease_generation=lease.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
            )

    def release_lease(self, lease: TaskLease) -> None:
        """一次 worker 调度结束后释放仍可运行的任务，供下一次领取继续状态机。"""

        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(AgentTaskRow.task_id == lease.state.task_id)
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            self._verify_lease(row, lease, now=self._now())
            row.lease_owner = None
            row.lease_expires_at = None
            row.updated_at = self._now()
            work.session.flush()

    def refresh_lease(
        self,
        lease: TaskLease,
        *,
        lease_seconds: int | None = None,
    ) -> TaskLease:
        """在不改变 generation 的前提下回读外部事务更新后的最新 checkpoint。"""

        if lease_seconds is not None and lease_seconds <= 0:
            raise ValueError("invalid_task_claim")
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(AgentTaskRow.task_id == lease.state.task_id)
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            self._verify_lease(row, lease, now=now)
            if lease_seconds is not None:
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.updated_at = now
                work.session.flush()
            return TaskLease(
                state=_state_from_row(row),
                worker_id=lease.worker_id,
                lease_generation=lease.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
            )

    def checkpoint(
        self,
        lease: TaskLease,
        state: TaskState,
        *,
        event_type: str,
        trace_id: str,
        run_id: str,
        release_lease: bool = False,
    ) -> TaskState:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self.checkpoint_in_session(
                work.session,
                lease,
                state,
                event_type=event_type,
                trace_id=trace_id,
                run_id=run_id,
                release_lease=release_lease,
            )

    def checkpoint_in_session(
        self,
        session,
        lease: TaskLease,
        state: TaskState,
        *,
        event_type: str,
        trace_id: str,
        run_id: str,
        release_lease: bool = False,
    ) -> TaskState:
        """复用调用方事务的 CAS checkpoint，供批准创建与任务暂停原子组合。"""

        if state.task_id != lease.state.task_id:
            raise TaskRepositoryError("task_checkpoint_binding_invalid")
        now = self._now()
        row = session.scalar(
            select(AgentTaskRow)
            .where(AgentTaskRow.task_id == state.task_id)
            .with_for_update()
        )
        if row is None:
            raise TaskRepositoryError("task_not_found")
        self._verify_lease(row, lease, now=now)
        if row.state_version != lease.state.state_version:
            raise TaskRepositoryError("task_state_conflict")
        self._validate_pending_state(session, row, state, now=now)
        self._apply_state(row, state)
        row.state_version += 1
        row.updated_at = now
        should_release = (
            release_lease
            or TaskStatus(row.status) in _TERMINAL_STATUSES
            or row.status
            in {
                TaskStatus.WAITING_FOR_USER.value,
                TaskStatus.WAITING_FOR_APPROVAL.value,
            }
        )
        if should_release:
            row.lease_owner = None
            row.lease_expires_at = None
        row.checkpoint_digest = self._checkpoint_digest(row)
        session.flush()
        self._append_event(
            session,
            row,
            event_type=event_type,
            trace_id=trace_id,
            run_id=run_id,
        )
        return _state_from_row(row)

    def resume_user(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        answer_ref: str,
    ) -> TaskState:
        if not answer_ref:
            raise ValueError("answer_ref_required")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.user_id == auth.user_id,
                )
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            if row.state_version != expected_state_version:
                raise TaskRepositoryError("task_state_conflict")
            if row.status != TaskStatus.WAITING_FOR_USER.value:
                raise TaskRepositoryError("task_not_waiting_for_user")
            row.observation_refs = [*row.observation_refs, answer_ref]
            row.pending_question = None
            row.status = TaskStatus.PLANNING.value
            row.current_node = TaskNode.UPDATE_PLAN.value
            row.available_at = self._now()
            row.state_version += 1
            row.updated_at = self._now()
            row.checkpoint_digest = self._checkpoint_digest(row)
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_resumed",
                trace_id=auth.request_id,
                run_id=auth.request_id,
            )
            return _state_from_row(row)

    def cancel(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
    ) -> TaskState:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.user_id == auth.user_id,
                )
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            if row.state_version != expected_state_version:
                raise TaskRepositoryError("task_state_conflict")
            if TaskStatus(row.status) in _TERMINAL_STATUSES:
                raise TaskRepositoryError("task_terminal_state")
            row.cancelled = True
            row.status = TaskStatus.CANCELLED.value
            row.current_node = TaskNode.TERMINAL.value
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = self._now()
            row.checkpoint_digest = self._checkpoint_digest(row)
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_cancelled",
                trace_id=auth.request_id,
                run_id=auth.request_id,
            )
            return _state_from_row(row)

    @staticmethod
    def _verify_lease(row: AgentTaskRow, lease: TaskLease, *, now: datetime) -> None:
        expires_at = _aware(row.lease_expires_at)
        if row.lease_owner != lease.worker_id or row.lease_generation != lease.lease_generation:
            raise TaskRepositoryError("task_lease_generation_stale")
        if expires_at is None or expires_at <= now:
            raise TaskRepositoryError("task_lease_expired")

    @staticmethod
    def _validate_pending_state(
        session,
        row: AgentTaskRow,
        state: TaskState,
        *,
        now: datetime,
    ) -> None:
        required_nodes = {
            TaskNode.EXECUTE_READ_TOOL,
            TaskNode.AWAIT_APPROVAL,
            TaskNode.EXECUTE_APPROVED_ACTION,
        }
        pending = state.pending_action
        if state.current_node in required_nodes and pending is None:
            raise TaskRepositoryError("task_pending_action_binding_missing")
        if pending is not None and pending.expires_at <= now:
            raise TaskRepositoryError("task_pending_action_expired")
        if state.status is TaskStatus.WAITING_FOR_APPROVAL and (
            pending is None or pending.kind != "propose_action"
        ):
            raise TaskRepositoryError("task_pending_action_binding_invalid")
        if state.pending_approval is None:
            return
        if pending is None or pending.kind != "propose_action":
            raise TaskRepositoryError("task_approval_action_binding_invalid")
        approval_id = session.scalar(
            select(ApprovalRow.approval_id).where(
                ApprovalRow.tenant_id == row.tenant_id,
                ApprovalRow.task_id == row.task_id,
                ApprovalRow.plan_revision == state.plan_revision,
                ApprovalRow.approval_id == state.pending_approval,
            )
        )
        if approval_id is None:
            raise TaskRepositoryError("task_pending_approval_not_found")

    @staticmethod
    def _apply_state(row: AgentTaskRow, state: TaskState) -> None:
        row.status = state.status.value
        row.current_node = state.current_node.value
        row.current_plan_revision = state.plan_revision
        row.current_step = state.current_step
        row.observation_refs = [str(item["ref"]) for item in state.observations]
        row.evidence_refs = list(state.evidence_ids)
        row.pending_question = state.pending_question
        row.pending_approval_id = state.pending_approval
        if state.pending_action is None:
            row.pending_action_ref = None
            row.pending_action_hash = None
            row.pending_action_kind = None
            row.pending_action_expires_at = None
        else:
            row.pending_action_ref = state.pending_action.storage_ref
            row.pending_action_hash = state.pending_action.content_hash
            row.pending_action_kind = state.pending_action.kind
            row.pending_action_expires_at = state.pending_action.expires_at
        row.artifact_refs = list(state.artifacts)
        row.model_calls = state.model_calls
        row.tool_calls = state.tool_calls
        row.replans = state.replans
        row.consecutive_failures = state.consecutive_failures
        row.token_usage = state.token_usage
        row.last_error_code = state.last_error
        row.repeated_action_fingerprints = list(state.repeated_action_fingerprints)
        row.query_fingerprints = list(state.query_fingerprints)
        row.result_fingerprints = list(state.result_fingerprints)
        row.no_new_evidence_count = state.no_new_evidence_count
        row.cancelled = state.cancelled

    @staticmethod
    def _append_event(
        session,
        row: AgentTaskRow,
        *,
        event_type: str,
        trace_id: str,
        run_id: str,
    ) -> None:
        sequence = session.scalar(
            select(func.coalesce(func.max(TaskEventRow.sequence), -1) + 1).where(
                TaskEventRow.tenant_id == row.tenant_id,
                TaskEventRow.task_id == row.task_id,
            )
        )
        session.add(
            TaskEventRow(
                event_id=uuid4().hex,
                task_id=row.task_id,
                tenant_id=row.tenant_id,
                sequence=int(sequence or 0),
                event_type=event_type,
                trace_id=trace_id,
                run_id=run_id,
                stage=row.current_node,
                status=row.status,
                counts={
                    "model_calls": row.model_calls,
                    "tool_calls": row.tool_calls,
                    "replans": row.replans,
                },
                token_usage=row.token_usage,
                error_code=row.last_error_code,
            )
        )

    @staticmethod
    def _checkpoint_digest(row: AgentTaskRow) -> str:
        return canonical_fingerprint(
            "task_checkpoint",
            {
                "task_id": row.task_id,
                "status": row.status,
                "current_node": row.current_node,
                "state_version": row.state_version,
                "plan_revision": row.current_plan_revision,
                "current_step": row.current_step,
                "observation_refs": row.observation_refs,
                "evidence_refs": row.evidence_refs,
                "artifact_refs": row.artifact_refs,
                "pending_action_ref": row.pending_action_ref,
                "pending_action_hash": row.pending_action_hash,
                "pending_action_kind": row.pending_action_kind,
                "pending_action_expires_at": (
                    _aware(row.pending_action_expires_at).isoformat()
                    if row.pending_action_expires_at is not None
                    else None
                ),
                "pending_approval_id": row.pending_approval_id,
                "model_calls": row.model_calls,
                "tool_calls": row.tool_calls,
                "replans": row.replans,
                "token_usage": row.token_usage,
                "cancelled": row.cancelled,
            },
        )
