"""agent_tasks 仓储：只保存业务生命周期、租约、中断摘要与最终结果引用。

Agent 执行状态（消息、Todo、工具循环、interrupt 载荷）唯一保存在
LangGraph PostgreSQL checkpointer（thread_id = task_id），本仓储不复制。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskState, TaskStatus
from rag_kb.infrastructure.database.models import (
    AgentCapabilityLeaseRow,
    AgentTaskRow,
    TaskEventRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class TaskRepositoryError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.REFUSED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.UNKNOWN,
    }
)


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


@dataclass(frozen=True)
class TaskLease:
    state: TaskState
    worker_id: str
    lease_generation: int
    budgets: TaskBudgets


def _state_from_row(row: AgentTaskRow) -> TaskState:
    return TaskState(
        task_id=row.task_id,
        auth_context_ref=row.auth_context_ref,
        goal=row.goal,
        task_mode=TaskMode(row.task_mode),
        status=TaskStatus(row.status),
        state_version=row.state_version,
        lease_generation=row.lease_generation,
        interrupt_kind=row.interrupt_kind,
        interrupt_summary=row.interrupt_summary,
        result_kind=row.result_kind,
        result_ref=row.result_ref,
        result_hash=row.result_hash,
        content_expires_at=_aware(row.content_expires_at),
        result_purged_at=_aware(row.result_purged_at),
        deadline=_aware(row.deadline),
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

    def register_capability(
        self,
        *,
        capability_name: str,
        protocol_version: str,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        """注册或续租一个能力实例；同一三元主键只保留一行。"""
        self._validate_capability_args(
            capability_name, protocol_version, worker_id, lease_seconds
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            now = self._database_now(work.session)
            lease_expires_at = now + timedelta(seconds=lease_seconds)
            values = {
                "capability_name": capability_name,
                "protocol_version": protocol_version,
                "worker_id": worker_id,
                "lease_expires_at": lease_expires_at,
                "updated_at": now,
            }
            dialect = work.session.get_bind().dialect.name
            if dialect == "postgresql":
                statement = postgresql_insert(AgentCapabilityLeaseRow).values(**values)
            elif dialect == "sqlite":
                statement = sqlite_insert(AgentCapabilityLeaseRow).values(**values)
            else:
                raise RuntimeError("unsupported_database_dialect")
            work.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        AgentCapabilityLeaseRow.capability_name,
                        AgentCapabilityLeaseRow.protocol_version,
                        AgentCapabilityLeaseRow.worker_id,
                    ],
                    set_={
                        "lease_expires_at": case(
                            (
                                AgentCapabilityLeaseRow.lease_expires_at
                                > lease_expires_at,
                                AgentCapabilityLeaseRow.lease_expires_at,
                            ),
                            else_=lease_expires_at,
                        ),
                        "updated_at": case(
                            (
                                AgentCapabilityLeaseRow.updated_at > now,
                                AgentCapabilityLeaseRow.updated_at,
                            ),
                            else_=now,
                        ),
                    },
                )
            )

    def renew_capability(
        self,
        *,
        capability_name: str,
        protocol_version: str,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        """续租必须命中已注册实例，避免拼写错误悄悄创建能力。"""
        self._validate_capability_args(
            capability_name, protocol_version, worker_id, lease_seconds
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            now = self._database_now(work.session)
            lease_expires_at = now + timedelta(seconds=lease_seconds)
            updated = work.session.execute(
                update(AgentCapabilityLeaseRow)
                .where(
                    AgentCapabilityLeaseRow.capability_name == capability_name,
                    AgentCapabilityLeaseRow.protocol_version == protocol_version,
                    AgentCapabilityLeaseRow.worker_id == worker_id,
                )
                .values(
                    lease_expires_at=case(
                        (
                            AgentCapabilityLeaseRow.lease_expires_at > lease_expires_at,
                            AgentCapabilityLeaseRow.lease_expires_at,
                        ),
                        else_=lease_expires_at,
                    ),
                    updated_at=case(
                        (
                            AgentCapabilityLeaseRow.updated_at > now,
                            AgentCapabilityLeaseRow.updated_at,
                        ),
                        else_=now,
                    ),
                )
            )
            if updated.rowcount != 1:
                raise TaskRepositoryError("agent_capability_not_found")

    def unregister_capability(
        self,
        *,
        capability_name: str,
        protocol_version: str,
        worker_id: str,
    ) -> None:
        if not capability_name or not protocol_version or not worker_id:
            raise ValueError("invalid_agent_capability")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            work.session.execute(
                delete(AgentCapabilityLeaseRow).where(
                    AgentCapabilityLeaseRow.capability_name == capability_name,
                    AgentCapabilityLeaseRow.protocol_version == protocol_version,
                    AgentCapabilityLeaseRow.worker_id == worker_id,
                )
            )

    def has_active_capability(
        self, *, capability_name: str, protocol_version: str
    ) -> bool:
        if not capability_name or not protocol_version:
            raise ValueError("invalid_agent_capability")
        with self._database.sessions() as session:
            return self._has_active_capability_in_session(
                session,
                capability_name=capability_name,
                protocol_version=protocol_version,
            )

    def _has_active_capability_in_session(
        self,
        session,
        *,
        capability_name: str,
        protocol_version: str,
    ) -> bool:
        """供后续任务创建事务复用，避免 capability 检查与插入任务间的 TOCTOU。"""
        return (
            session.scalar(
                select(AgentCapabilityLeaseRow.worker_id)
                .where(
                    AgentCapabilityLeaseRow.capability_name == capability_name,
                    AgentCapabilityLeaseRow.protocol_version == protocol_version,
                    AgentCapabilityLeaseRow.lease_expires_at
                    > self._database_clock_expression(session),
                )
                .limit(1)
            )
            is not None
        )

    @staticmethod
    def _database_clock_expression(session):
        if session.get_bind().dialect.name == "postgresql":
            return func.clock_timestamp()
        return func.current_timestamp()

    @classmethod
    def _database_now(cls, session) -> datetime:
        value = session.scalar(select(cls._database_clock_expression(session)))
        if not isinstance(value, datetime):
            raise RuntimeError("database_clock_unavailable")
        return _aware(value) or value

    @staticmethod
    def _validate_capability_args(
        capability_name: str,
        protocol_version: str,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        if (
            not capability_name
            or len(capability_name) > 64
            or not protocol_version
            or len(protocol_version) > 64
            or not worker_id
            or len(worker_id) > 200
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
        ):
            raise ValueError("invalid_agent_capability")

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
                    AgentTaskRow.task_mode == TaskMode.RESEARCH.value,
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

    def claim_user_resume(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        worker_id: str,
        lease_seconds: int,
    ) -> TaskLease:
        """按服务端确认的澄清 interrupt 领取同一任务；回答不落 agent_tasks。"""
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_claim")
        now = self._now()
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
            if row.task_mode != TaskMode.RESEARCH.value:
                raise TaskRepositoryError("task_mode_not_supported")
            if row.status != TaskStatus.WAITING_FOR_USER.value:
                raise TaskRepositoryError("task_not_waiting_for_user")
            if row.interrupt_kind != "user_clarification":
                raise TaskRepositoryError("task_resume_kind_invalid")
            row.status = TaskStatus.RUNNING.value
            row.interrupt_kind = None
            row.interrupt_summary = None
            row.lease_owner = worker_id
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.lease_generation += 1
            row.state_version += 1
            row.updated_at = now
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_resume_started",
                trace_id=auth.request_id,
                run_id=auth.request_id,
            )
            return TaskLease(
                state=_state_from_row(row),
                worker_id=worker_id,
                lease_generation=row.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
            )

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int:
        """同步 API 崩溃后只做终态收敛，不交给通用 worker 重放。"""
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
                row.last_error_code = "knowledge_query_lease_expired"
                row.lease_owner = None
                row.lease_expires_at = None
                row.state_version += 1
                row.updated_at = now
            return len(rows)

    def release_lease(self, lease: TaskLease) -> None:
        """一次 worker 调度结束后释放仍可运行的任务，供下一次领取继续执行。"""
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
        """在不改变 generation 的前提下延长租约并回读最新业务状态。"""
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

    def pause_for_user(
        self,
        lease: TaskLease,
        *,
        question: str,
        trace_id: str,
    ) -> TaskState:
        """模型请求澄清：只保存可展示摘要，正文留在 checkpointer 线程内。"""
        if not question.strip():
            raise ValueError("task_question_required")
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
            row.status = TaskStatus.WAITING_FOR_USER.value
            row.interrupt_kind = "user_clarification"
            row.interrupt_summary = question.strip()[:2_000]
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = self._now()
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_waiting_for_user",
                trace_id=trace_id,
                run_id=trace_id,
            )
            return _state_from_row(row)

    def finish_task(
        self,
        lease: TaskLease,
        *,
        status: TaskStatus,
        result_kind: str | None = None,
        result_ref: str | None = None,
        result_hash: str | None = None,
        error_code: str | None = None,
        trace_id: str,
    ) -> TaskState:
        """终态写入：succeeded 必须原子绑定有效结果（计划 §11.4 门禁）。"""
        if status not in _TERMINAL_STATUSES:
            raise ValueError("task_status_not_terminal")
        if status in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED} and (
            result_kind != "research" or result_ref is None or result_hash is None
        ):
            raise ValueError("task_result_binding_required")
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
            row.status = status.value
            row.cancelled = status is TaskStatus.CANCELLED
            row.result_kind = result_kind
            row.result_ref = result_ref
            row.result_hash = result_hash
            row.last_error_code = error_code
            if row.task_mode == TaskMode.RESEARCH.value:
                row.content_expires_at = self._database_now(work.session) + timedelta(
                    days=self._task_result_retention_days()
                )
            row.interrupt_kind = None
            row.interrupt_summary = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = self._now()
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type=f"task_{status.value}",
                trace_id=trace_id,
                run_id=trace_id,
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
            if row.task_mode == TaskMode.RESEARCH.value:
                row.content_expires_at = self._database_now(work.session) + timedelta(
                    days=self._task_result_retention_days()
                )
            row.interrupt_kind = None
            row.interrupt_summary = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = self._now()
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
    def _task_result_retention_days() -> int:
        from rag_kb.config import get_settings

        return get_settings().task_result_retention_days

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
                stage=row.status,
                status=row.status,
                counts={},
                token_usage=None,
                error_code=row.last_error_code,
            )
        )
