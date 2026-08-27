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
from rag_kb.domain.tasks import (
    M1_RESEARCH_AGENT_CAPABILITY_NAME,
    M1_RESEARCH_AGENT_PROTOCOL_VERSION,
    TASK_AGENT_CAPABILITY_NAME as _TASK_AGENT_CAPABILITY_NAME,
    TASK_AGENT_PROTOCOL_VERSION as _TASK_AGENT_PROTOCOL_VERSION,
    TaskBudgets,
    TaskMode,
    TaskState,
    TaskStatus,
)
from rag_kb.infrastructure.database.models import (
    AgentCapabilityLeaseRow,
    AgentTaskRow,
    TaskEventRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_execution import (
    TaskExecutionContextError,
    current_task_auth,
)
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
RESEARCH_AGENT_CAPABILITY_NAME = M1_RESEARCH_AGENT_CAPABILITY_NAME
RESEARCH_AGENT_PROTOCOL_VERSION = M1_RESEARCH_AGENT_PROTOCOL_VERSION
TASK_AGENT_CAPABILITY_NAME = _TASK_AGENT_CAPABILITY_NAME
TASK_AGENT_PROTOCOL_VERSION = _TASK_AGENT_PROTOCOL_VERSION


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
        required_capability_name=row.required_capability_name,
        required_protocol_version=row.required_protocol_version,
        state_version=row.state_version,
        lease_generation=row.lease_generation,
        interrupt_kind=row.interrupt_kind,
        interrupt_summary=row.interrupt_summary,
        interrupt_ref=row.interrupt_ref,
        result_kind=row.result_kind,
        result_ref=row.result_ref,
        result_hash=row.result_hash,
        content_expires_at=_aware(row.content_expires_at),
        result_purged_at=_aware(row.result_purged_at),
        deadline=_aware(row.deadline),
        cancelled=row.cancelled,
    )


class DatabaseTaskRepository:
    def __init__(
        self,
        database: Database,
        *,
        default_budgets: TaskBudgets | None = None,
        now=None,
        research_capability_name: str | None = None,
        research_capability_protocol_version: str | None = None,
        task_capability_name: str | None = None,
        task_capability_protocol_version: str | None = None,
    ) -> None:
        if (research_capability_name is None) != (research_capability_protocol_version is None):
            raise ValueError("invalid_agent_capability")
        if (task_capability_name is None) != (task_capability_protocol_version is None):
            raise ValueError("invalid_task_capability")
        if (
            task_capability_name is not None
            and research_capability_name is not None
            and (
                task_capability_name != research_capability_name
                or task_capability_protocol_version != research_capability_protocol_version
            )
        ):
            raise ValueError("invalid_task_capability")
        self._database = database
        self._default_budgets = (
            default_budgets if default_budgets is not None else TaskBudgets()
        )
        self._now = now or (lambda: datetime.now(UTC))
        self._research_capability_name = research_capability_name
        self._research_capability_protocol_version = research_capability_protocol_version
        self._task_capability_name = (
            task_capability_name
            or research_capability_name
            or M1_RESEARCH_AGENT_CAPABILITY_NAME
        )
        self._task_capability_protocol_version = (
            task_capability_protocol_version
            or research_capability_protocol_version
            or M1_RESEARCH_AGENT_PROTOCOL_VERSION
        )
        self._creation_capability_required = (
            task_capability_name is not None or research_capability_name is not None
        )

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
        configured = budgets if budgets is not None else self._default_budgets
        deadline = _aware(deadline)
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
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            now = self._lease_now(work.session)
            if deadline is None:
                deadline = now + timedelta(seconds=configured.max_runtime_seconds)
            if deadline <= now:
                raise TaskRepositoryError("task_deadline_expired")
            if (
                task_mode is TaskMode.RESEARCH
                and self._creation_capability_required
                and not self._has_active_capability_in_session(
                    work.session,
                    capability_name=self._task_capability_name,
                    protocol_version=self._task_capability_protocol_version,
                    for_update=True,
                )
            ):
                raise TaskRepositoryError("research_agent_unavailable")
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
                available_at=now,
                required_capability_name=self._task_capability_name,
                required_protocol_version=self._task_capability_protocol_version,
            )
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
        for_update: bool = False,
    ) -> bool:
        """供后续任务创建事务复用，避免 capability 检查与插入任务间的 TOCTOU。"""
        statement = (
            select(AgentCapabilityLeaseRow.worker_id)
            .where(
                AgentCapabilityLeaseRow.capability_name == capability_name,
                AgentCapabilityLeaseRow.protocol_version == protocol_version,
                AgentCapabilityLeaseRow.lease_expires_at
                > self._database_clock_expression(session),
            )
            .limit(1)
        )
        if for_update:
            statement = statement.with_for_update()
        return (
            session.scalar(statement)
            is not None
        )

    def _has_active_worker_capability_in_session(
        self,
        session,
        *,
        capability_name: str,
        protocol_version: str,
        worker_id: str,
        for_update: bool = False,
    ) -> bool:
        statement = (
            select(AgentCapabilityLeaseRow.worker_id)
            .where(
                AgentCapabilityLeaseRow.capability_name == capability_name,
                AgentCapabilityLeaseRow.protocol_version == protocol_version,
                AgentCapabilityLeaseRow.worker_id == worker_id,
                AgentCapabilityLeaseRow.lease_expires_at
                > self._database_clock_expression(session),
            )
            .limit(1)
        )
        if for_update:
            statement = statement.with_for_update()
        return session.scalar(statement) is not None

    def _database_clock_expression(self, session):
        if session.get_bind().dialect.name == "postgresql":
            return func.clock_timestamp()
        return self._now()

    def _database_now(self, session) -> datetime:
        if session.get_bind().dialect.name != "postgresql":
            return self._now()
        value = session.scalar(select(self._database_clock_expression(session)))
        if not isinstance(value, datetime):
            raise RuntimeError("database_clock_unavailable")
        return _aware(value) or value

    def _lease_now(self, session) -> datetime:
        return self._database_now(session)

    def _lease_clock_expression(self, session):
        return self._database_clock_expression(session)

    def current_time(self) -> datetime:
        """返回与任务生命周期判定一致的时间源。"""
        with self._database.sessions() as session:
            return self._lease_now(session)

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

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> TaskLease | None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_claim")
        capability_name, protocol_version, require_capability = self._claim_protocol(
            capability_name, protocol_version
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            if require_capability and not self._has_active_worker_capability_in_session(
                work.session,
                capability_name=capability_name,
                protocol_version=protocol_version,
                worker_id=worker_id,
                for_update=True,
            ):
                return None
            clock = self._lease_clock_expression(work.session)
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.status.in_(
                        (TaskStatus.PLANNING.value, TaskStatus.RUNNING.value)
                    ),
                    AgentTaskRow.task_mode == TaskMode.RESEARCH.value,
                    AgentTaskRow.required_capability_name == capability_name,
                    AgentTaskRow.required_protocol_version == protocol_version,
                    AgentTaskRow.cancelled.is_(False),
                    AgentTaskRow.available_at <= clock,
                    or_(
                        AgentTaskRow.lease_owner.is_(None),
                        AgentTaskRow.lease_expires_at <= clock,
                    ),
                )
                .order_by(AgentTaskRow.available_at, AgentTaskRow.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            now = self._lease_now(work.session)
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
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> TaskLease:
        """按服务端确认的澄清 interrupt 领取同一任务；回答不落 agent_tasks。"""
        if not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_task_claim")
        capability_name, protocol_version, require_capability = self._claim_protocol(
            capability_name, protocol_version
        )
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
            if (
                row.required_capability_name != capability_name
                or row.required_protocol_version != protocol_version
            ):
                raise TaskRepositoryError("task_protocol_mismatch")
            if row.status != TaskStatus.WAITING_FOR_USER.value:
                raise TaskRepositoryError("task_not_waiting_for_user")
            if row.interrupt_kind != "user_clarification" or row.interrupt_ref is not None:
                raise TaskRepositoryError("task_resume_kind_invalid")
            if require_capability and not self._has_active_worker_capability_in_session(
                work.session,
                capability_name=capability_name,
                protocol_version=protocol_version,
                worker_id=worker_id,
                for_update=True,
            ):
                raise TaskRepositoryError("task_agent_unavailable")
            now = self._lease_now(work.session)
            deadline = _aware(row.deadline)
            if deadline is not None and deadline <= now:
                raise TaskRepositoryError("task_deadline_expired")
            resumed = work.session.execute(
                update(AgentTaskRow)
                .where(
                    AgentTaskRow.task_id == row.task_id,
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.user_id == auth.user_id,
                    AgentTaskRow.state_version == expected_state_version,
                    AgentTaskRow.task_mode == TaskMode.RESEARCH.value,
                    AgentTaskRow.required_capability_name == capability_name,
                    AgentTaskRow.required_protocol_version == protocol_version,
                    AgentTaskRow.status == TaskStatus.WAITING_FOR_USER.value,
                    AgentTaskRow.interrupt_kind == "user_clarification",
                    AgentTaskRow.interrupt_ref.is_(None),
                    or_(
                        AgentTaskRow.deadline.is_(None),
                        AgentTaskRow.deadline
                        > self._lease_clock_expression(work.session),
                    ),
                )
                .values(
                    status=TaskStatus.RUNNING.value,
                    interrupt_kind=None,
                    interrupt_summary=None,
                    interrupt_ref=None,
                    lease_owner=worker_id,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    lease_generation=row.lease_generation + 1,
                    state_version=row.state_version + 1,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if resumed.rowcount != 1:
                if deadline is not None and deadline <= self._lease_now(work.session):
                    raise TaskRepositoryError("task_deadline_expired")
                raise TaskRepositoryError("task_state_conflict")
            work.session.refresh(row)
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

    def expire_waiting_tasks(self, *, limit: int = 100) -> int:
        """把超时的澄清等待原子收敛为失败，后续交给正文保留策略。"""
        if limit <= 0:
            raise ValueError("invalid_waiting_reaper_limit")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._lease_clock_expression(work.session)
            rows = tuple(
                work.session.scalars(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.task_mode == TaskMode.RESEARCH.value,
                        AgentTaskRow.status == TaskStatus.WAITING_FOR_USER.value,
                        AgentTaskRow.deadline.is_not(None),
                        AgentTaskRow.deadline <= clock,
                    )
                    .order_by(AgentTaskRow.deadline, AgentTaskRow.task_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            now = self._lease_now(work.session)
            for row in rows:
                row.status = TaskStatus.FAILED.value
                row.last_error_code = "task_deadline_expired"
                row.content_expires_at = now + timedelta(
                    days=self._task_result_retention_days()
                )
                row.interrupt_kind = None
                row.interrupt_summary = None
                row.interrupt_ref = None
                row.lease_owner = None
                row.lease_expires_at = None
                row.state_version += 1
                row.updated_at = now
                work.session.flush()
                self._append_event(
                    work.session,
                    row,
                    event_type="task_failed",
                    trace_id="task_deadline_expired",
                    run_id="task_deadline_expired",
                )
            return len(rows)

    def expire_stale_knowledge_queries(self, *, limit: int = 100) -> int:
        """同步 API 崩溃后只做终态收敛，不交给通用 worker 重放。"""
        if limit <= 0:
            raise ValueError("invalid_sync_query_reaper_limit")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._lease_clock_expression(work.session)
            rows = tuple(
                work.session.scalars(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.task_mode == "knowledge_query",
                        AgentTaskRow.status == TaskStatus.RUNNING.value,
                        AgentTaskRow.lease_expires_at.is_not(None),
                        AgentTaskRow.lease_expires_at <= clock,
                    )
                    .order_by(AgentTaskRow.lease_expires_at, AgentTaskRow.task_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            now = self._lease_now(work.session)
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
            now = self._lease_now(work.session)
            self._verify_lease(row, lease, now=now)
            row.lease_owner = None
            row.lease_expires_at = None
            row.updated_at = now
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
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(AgentTaskRow.task_id == lease.state.task_id)
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            now = self._lease_now(work.session)
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
            now = self._lease_now(work.session)
            self._verify_lease(row, lease, now=now)
            if row.state_version != lease.state.state_version:
                raise TaskRepositoryError("task_state_conflict")
            row.status = TaskStatus.WAITING_FOR_USER.value
            row.interrupt_kind = "user_clarification"
            row.interrupt_summary = question.strip()[:2_000]
            row.interrupt_ref = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_waiting_for_user",
                trace_id=trace_id,
                run_id=trace_id,
            )
            return _state_from_row(row)

    def pause_for_approval(
        self,
        lease: TaskLease,
        *,
        interrupt_ref: str,
        summary: str,
        trace_id: str,
    ) -> TaskState:
        """工具审批暂停：业务表只保存裁剪摘要与不透明审批引用。

        原始工具参数、HITL payload 与 resume 数据一律不进入 agent_tasks；
        正文事实留在 checkpointer 线程内。
        """
        if not interrupt_ref.strip() or not summary.strip():
            raise ValueError("task_interrupt_fields_required")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow).where(AgentTaskRow.task_id == lease.state.task_id)
                .with_for_update()
            )
            if row is None:
                raise TaskRepositoryError("task_not_found")
            now = self._lease_now(work.session)
            self._verify_lease(row, lease, now=now)
            if row.state_version != lease.state.state_version:
                raise TaskRepositoryError("task_state_conflict")
            row.status = TaskStatus.WAITING_FOR_APPROVAL.value
            row.interrupt_kind = "tool_approval"
            row.interrupt_summary = summary.strip()[:2_000]
            row.interrupt_ref = interrupt_ref.strip()[:200]
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_waiting_for_approval",
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
        is_result_submission = status in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED}
        if is_result_submission and (
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
            now = self._lease_now(work.session)
            self._verify_lease(row, lease, now=now)
            if is_result_submission and (
                row.cancelled or row.status not in {"planning", "running"}
            ):
                raise TaskRepositoryError("task_lease_generation_stale")
            deadline = _aware(row.deadline)
            if (
                is_result_submission
                and deadline is not None
                and deadline <= now
            ):
                # 持有任务行锁后才判定 deadline，避免对象写后提交 succeeded/refused。
                status = TaskStatus.FAILED
                error_code = "task_deadline_expired"
                result_kind = None
                result_ref = None
                result_hash = None
            is_result_submission = status in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED}
            if is_result_submission:
                try:
                    current_task_auth(
                        work.session,
                        task=row,
                        auth_context_ref=lease.state.auth_context_ref,
                        request_id=trace_id,
                    )
                except TaskExecutionContextError as exc:
                    raise TaskRepositoryError("task_auth_context_changed") from exc
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
            row.interrupt_ref = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
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
            now = self._lease_now(work.session)
            row.cancelled = True
            row.status = TaskStatus.CANCELLED.value
            if row.task_mode == TaskMode.RESEARCH.value:
                row.content_expires_at = self._database_now(work.session) + timedelta(
                    days=self._task_result_retention_days()
                )
            row.interrupt_kind = None
            row.interrupt_summary = None
            row.interrupt_ref = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
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
    def _claim_protocol(
        capability_name: str | None,
        protocol_version: str | None,
    ) -> tuple[str, str, bool]:
        if (capability_name is None) != (protocol_version is None):
            raise ValueError("invalid_task_claim")
        if capability_name is None:
            return (
                M1_RESEARCH_AGENT_CAPABILITY_NAME,
                M1_RESEARCH_AGENT_PROTOCOL_VERSION,
                False,
            )
        if (
            not capability_name
            or len(capability_name) > 64
            or not protocol_version
            or len(protocol_version) > 64
        ):
            raise ValueError("invalid_task_claim")
        return capability_name, protocol_version, True

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
