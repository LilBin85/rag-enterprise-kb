"""agent_tasks 仓储：只保存业务生命周期、租约、中断摘要与最终结果引用。

Agent 执行状态（消息、Todo、工具循环、interrupt 载荷）唯一保存在
LangGraph PostgreSQL checkpointer（thread_id = task_id），本仓储不复制。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import case, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from rag_kb.application.ports.task_worker import (
    ResumeCommandLease,
    TaskLease,
    TaskWorkerPersistenceError,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.resume_commands import (
    ApprovalResumeDecision,
    ResumeCommand,
    ResumeCommandCandidate,
    ResumeCommandKind,
    ResumeCommandStatus,
)
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
    ApprovalRow,
    ArtifactRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TaskEventRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_execution import (
    TaskExecutionContextError,
    current_task_auth,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.observability import (
    current_request_id,
    current_span_ids,
    current_trace_headers,
    normalize_request_id,
)


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
# 保留既有导入路径，实际错误契约归 application 所有。
TaskRepositoryError = TaskWorkerPersistenceError


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


@dataclass(frozen=True)
class TaskQueueSnapshot:
    task_mode: str
    status: str
    depth: int
    oldest_age_seconds: float


def _command_from_row(row: TaskResumeCommandRow) -> ResumeCommand:
    return ResumeCommand(
        command_id=row.command_id,
        tenant_id=row.tenant_id,
        task_id=row.task_id,
        kind=ResumeCommandKind(row.kind),
        actor_principal_id=row.actor_principal_id,
        actor_kind=row.actor_kind,
        expected_task_state_version=row.expected_task_state_version,
        expected_interrupt_checkpoint_ns=row.expected_interrupt_checkpoint_ns,
        expected_interrupt_checkpoint_id=row.expected_interrupt_checkpoint_id,
        expected_interrupt_id=row.expected_interrupt_id,
        expected_protocol_version=row.expected_protocol_version,
        approval_id=row.approval_id,
        decision=(ApprovalResumeDecision(row.decision) if row.decision is not None else None),
        payload_body=row.payload_body,
        payload_sha256=row.payload_sha256,
        resume_target_key=row.resume_target_key,
        idempotency_key=row.idempotency_key,
        status=ResumeCommandStatus(row.status),
        attempt_count=row.attempt_count,
        next_attempt_at=_aware(row.next_attempt_at),
        expires_at=_aware(row.expires_at),
        lease_owner=row.lease_owner,
        lease_generation=row.lease_generation,
        lease_expires_at=_aware(row.lease_expires_at),
        claimed_at=_aware(row.claimed_at),
        claimed_task_state_version=row.claimed_task_state_version,
        claimed_task_lease_generation=row.claimed_task_lease_generation,
        applied_checkpoint_id=row.applied_checkpoint_id,
        applied_at=_aware(row.applied_at),
        consumed_at=_aware(row.consumed_at),
        failed_at=_aware(row.failed_at),
        expired_at=_aware(row.expired_at),
        failure_code=row.failure_code,
        body_purged_at=_aware(row.body_purged_at),
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
    )


def _state_from_row(row: AgentTaskRow) -> TaskState:
    return TaskState(
        task_id=row.task_id,
        conversation_id=row.conversation_id,
        turn_id=row.conversation_turn_id,
        auth_context_ref=row.auth_context_ref,
        goal=row.goal,
        scope_document_id=row.scope_document_id,
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
        conversation_id: str | None = None,
        turn_id: str | None = None,
        scope_document_id: str | None = None,
    ) -> TaskState:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self.create_task_in_session(
                work.session,
                auth=auth,
                goal=goal,
                task_mode=task_mode,
                budgets=budgets,
                deadline=deadline,
                task_id=task_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                scope_document_id=scope_document_id,
            )

    def create_task_in_session(
        self,
        session,
        *,
        auth: AuthContext,
        goal: str,
        task_mode: TaskMode,
        budgets: TaskBudgets | None = None,
        deadline: datetime | None = None,
        task_id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        scope_document_id: str | None = None,
    ) -> TaskState:
        """供会话编排复用的同一事务建任务原语。

        调用方必须已经锁住会话/轮次；这里仍在插入前以 capability 行锁完成
        可用性检查，避免把会话绑定到永远不可领取的耐久任务。
        """
        if not goal.strip():
            raise ValueError("task_goal_required")
        if (conversation_id is None) != (turn_id is None):
            raise ValueError("task_conversation_binding_invalid")
        if scope_document_id is not None and not scope_document_id.strip():
            raise ValueError("document_scope_invalid")
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
        now = self._lease_now(session)
        if deadline is None:
            deadline = now + timedelta(seconds=configured.max_runtime_seconds)
        if deadline <= now:
            raise TaskRepositoryError("task_deadline_expired")
        if (
            task_mode in {TaskMode.RESEARCH, TaskMode.CONTROLLED_ACTION}
            and self._creation_capability_required
            and not self._has_active_capability_in_session(
                session,
                capability_name=self._task_capability_name,
                protocol_version=self._task_capability_protocol_version,
                for_update=True,
            )
        ):
            raise TaskRepositoryError(
                "research_agent_unavailable"
                if task_mode is TaskMode.RESEARCH
                else "task_agent_unavailable"
            )
        pending_traceparent, pending_tracestate = current_trace_headers()
        row = AgentTaskRow(
            task_id=task_id or uuid4().hex,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            conversation_id=conversation_id,
            conversation_turn_id=turn_id,
            task_mode=task_mode.value,
            status=TaskStatus.PLANNING.value,
            goal=goal.strip(),
            auth_context_ref=auth_ref,
            scope_document_id=(
                scope_document_id.strip() if scope_document_id is not None else None
            ),
            budgets=configured.model_dump(mode="json"),
            deadline=deadline,
            available_at=now,
            required_capability_name=self._task_capability_name,
            required_protocol_version=self._task_capability_protocol_version,
            pending_traceparent=pending_traceparent,
            pending_tracestate=pending_tracestate,
        )
        session.add(row)
        session.flush()
        self._append_event(
            session,
            row,
            event_type="task_created",
            correlation_id=auth.request_id,
            run_id=None,
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

    def conversation_context_for_task(
        self, *, tenant_id: str, task_id: str
    ) -> tuple[tuple[dict[str, str], ...], str | None]:
        """读取任务创建时冻结的会话快照；恢复时只用于重建运行 context。"""
        from rag_kb.infrastructure.database.conversation_repository import context_hash
        from rag_kb.infrastructure.database.models import ConversationTurnRow

        with self._database.sessions() as session:
            row = session.scalar(
                select(ConversationTurnRow).where(
                    ConversationTurnRow.tenant_id == tenant_id,
                    ConversationTurnRow.task_id == task_id,
                )
            )
            if row is None:
                return (), None
            snapshot = tuple(
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                for item in (row.context_snapshot or [])
                if isinstance(item, dict) and str(item.get("content", "")).strip()
            )
            if row.context_hash != context_hash(snapshot):
                return (), None
            return snapshot, row.context_hash

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

    @staticmethod
    def _resume_lock(statement, session, *, skip_locked: bool = False):
        """SQLite 不伪造 SKIP LOCKED；写路径由 BEGIN IMMEDIATE 串行化。"""
        if session.get_bind().dialect.name == "postgresql":
            return statement.with_for_update(skip_locked=skip_locked)
        return statement

    @staticmethod
    def _begin_resume_write(session) -> None:
        if session.get_bind().dialect.name == "sqlite":
            # 这必须是该 UnitOfWork 的第一条数据库语句；SQLite 测试适配器
            # 以数据库写锁模拟 PostgreSQL 的行级 compare-and-set 边界。
            session.execute(text("BEGIN IMMEDIATE"))

    def _lock_resume_task(
        self,
        session,
        *,
        tenant_id: str,
        task_id: str,
        skip_locked: bool,
    ) -> AgentTaskRow | None:
        return session.scalar(
            self._resume_lock(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == tenant_id,
                    AgentTaskRow.task_id == task_id,
                ),
                session,
                skip_locked=skip_locked,
            )
        )

    def _lock_resume_approval(
        self,
        session,
        *,
        tenant_id: str,
        approval_id: str,
        skip_locked: bool,
    ) -> ApprovalRow | None:
        return session.scalar(
            self._resume_lock(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == tenant_id,
                    ApprovalRow.approval_id == approval_id,
                ),
                session,
                skip_locked=skip_locked,
            )
        )

    def _lock_resume_command(
        self,
        session,
        *,
        tenant_id: str,
        command_id: str,
        skip_locked: bool,
    ) -> TaskResumeCommandRow | None:
        return session.scalar(
            self._resume_lock(
                select(TaskResumeCommandRow).where(
                    TaskResumeCommandRow.tenant_id == tenant_id,
                    TaskResumeCommandRow.command_id == command_id,
                ),
                session,
                skip_locked=skip_locked,
            )
        )

    def lock_approval_resume_command_in_session(
        self,
        session,
        *,
        tenant_id: str,
        task_id: str,
        approval_id: str,
    ) -> ResumeCommand | None:
        """在 task、approval 已锁定后锁定该审批的既有命令。"""
        row = session.scalar(
            self._resume_lock(
                select(TaskResumeCommandRow).where(
                    TaskResumeCommandRow.tenant_id == tenant_id,
                    TaskResumeCommandRow.task_id == task_id,
                    TaskResumeCommandRow.kind
                    == ResumeCommandKind.TOOL_APPROVAL.value,
                    TaskResumeCommandRow.approval_id == approval_id,
                ),
                session,
                skip_locked=False,
            )
        )
        return _command_from_row(row) if row is not None else None

    @staticmethod
    def _complete_resume_binding(
        task: AgentTaskRow,
        *,
        kind: ResumeCommandKind,
        approval_id: str | None = None,
    ) -> bool:
        namespace = task.interrupt_checkpoint_ns
        required = (task.interrupt_checkpoint_id, task.interrupt_id)
        # ``""`` 是 LangGraph 根 checkpoint 的合法 namespace；只有 NULL
        # 才表示该低敏 binding 缺失。
        if not isinstance(namespace, str) or not all(
            isinstance(value, str) and value.strip() for value in required
        ):
            return False
        if kind is ResumeCommandKind.USER_CLARIFICATION:
            return (
                task.status == TaskStatus.WAITING_FOR_USER.value
                and task.interrupt_kind == ResumeCommandKind.USER_CLARIFICATION.value
                and task.interrupt_ref is None
                and task.interrupt_approval_id is None
                and approval_id is None
            )
        return (
            task.status == TaskStatus.WAITING_FOR_APPROVAL.value
            and task.interrupt_kind == ResumeCommandKind.TOOL_APPROVAL.value
            and task.interrupt_ref == approval_id
            and task.interrupt_approval_id == approval_id
            and isinstance(task.interrupt_approval_id, str)
            and bool(task.interrupt_approval_id.strip())
        )

    @staticmethod
    def _approval_matches_resume(
        task: AgentTaskRow,
        approval: ApprovalRow | None,
        *,
        approval_id: str | None,
        decision: ApprovalResumeDecision | None,
    ) -> bool:
        if (
            approval is None
            or approval_id is None
            or decision is None
            or approval.approval_id != approval_id
            or approval.tenant_id != task.tenant_id
            or approval.task_id != task.task_id
            or task.interrupt_ref != approval_id
            or task.interrupt_approval_id != approval_id
            or approval.thread_id != task.task_id
            or approval.checkpoint_ns != task.interrupt_checkpoint_ns
            or approval.checkpoint_id != task.interrupt_checkpoint_id
            or approval.interrupt_id != task.interrupt_id
            or not approval.tool_call_id
            or not approval.checkpoint_binding_hash
        ):
            return False
        expected_status = (
            "approved"
            if decision is ApprovalResumeDecision.APPROVED
            else "rejected"
        )
        return approval.status == expected_status

    @staticmethod
    def _approval_matches_claimed_resume(
        task: AgentTaskRow,
        approval: ApprovalRow | None,
        command: TaskResumeCommandRow,
    ) -> bool:
        """claimed 后 task 已清 interrupt，重领只能以 command+approval 交叉复核。"""
        if (
            approval is None
            or command.approval_id is None
            or approval.approval_id != command.approval_id
            or approval.tenant_id != task.tenant_id
            or approval.task_id != task.task_id
            or task.task_mode != TaskMode.CONTROLLED_ACTION.value
            or approval.thread_id != task.task_id
            or approval.checkpoint_ns != command.expected_interrupt_checkpoint_ns
            or approval.checkpoint_id != command.expected_interrupt_checkpoint_id
            or approval.interrupt_id != command.expected_interrupt_id
            or not isinstance(approval.tool_call_id, str)
            or not approval.tool_call_id.strip()
            or not isinstance(approval.checkpoint_binding_hash, str)
            or not approval.checkpoint_binding_hash.strip()
        ):
            return False
        try:
            decision = ApprovalResumeDecision(command.decision or "")
        except ValueError:
            return False
        return approval.status == (
            "approved"
            if decision is ApprovalResumeDecision.APPROVED
            else "rejected"
        )

    @staticmethod
    def _normalize_resume_body(
        body: str | None,
        *,
        required: bool,
        maximum_length: int,
    ) -> str | None:
        if body is None:
            if required:
                raise ValueError("resume_command_payload_required")
            return None
        if not isinstance(body, str):
            raise ValueError("resume_command_payload_invalid")
        normalized = body.strip()
        if not normalized:
            if required:
                raise ValueError("resume_command_payload_required")
            return None
        if len(normalized) > maximum_length:
            raise ValueError("resume_command_payload_too_long")
        return normalized

    @staticmethod
    def _resume_failure_code(value: str) -> str:
        if (
            not value
            or len(value) > 100
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
        ):
            raise ValueError("invalid_resume_failure_code")
        return value

    @staticmethod
    def _resume_target_key(
        task: AgentTaskRow,
        *,
        kind: ResumeCommandKind,
        approval_id: str | None,
    ) -> str:
        return canonical_fingerprint(
            "task_resume_target",
            {
                "tenant_id": task.tenant_id,
                "task_id": task.task_id,
                "kind": kind.value,
                "expected_task_state_version": task.state_version,
                "checkpoint_ns": task.interrupt_checkpoint_ns,
                "checkpoint_id": task.interrupt_checkpoint_id,
                "interrupt_id": task.interrupt_id,
                "protocol_version": task.required_protocol_version,
                "approval_id": approval_id,
            },
        )

    @staticmethod
    def _resume_payload_sha256(
        *,
        kind: ResumeCommandKind,
        decision: ApprovalResumeDecision | None,
        payload_body: str | None,
    ) -> str:
        return canonical_fingerprint(
            "task_resume_payload",
            {
                "kind": kind.value,
                "decision": decision.value if decision is not None else None,
                "payload_body": payload_body,
            },
        )

    @staticmethod
    def _resume_idempotency_key(*, target_key: str, payload_sha256: str) -> str:
        return canonical_fingerprint(
            "task_resume_idempotency",
            {"resume_target_key": target_key, "payload_sha256": payload_sha256},
        )

    def _resume_expiry(
        self,
        task: AgentTaskRow,
        *,
        approval: ApprovalRow | None,
        now: datetime,
    ) -> datetime:
        candidates = [now + timedelta(hours=24)]
        deadline = _aware(task.deadline)
        if deadline is not None:
            candidates.append(deadline)
        if approval is not None:
            candidates.append(_aware(approval.expires_at) or approval.expires_at)
        expires_at = min(candidates)
        if expires_at <= now:
            raise TaskRepositoryError(
                "approval_expired" if approval is not None else "task_deadline_expired"
            )
        return expires_at

    def _resume_task_lease(
        self,
        task: AgentTaskRow,
        *,
        worker_id: str,
    ) -> TaskLease:
        return TaskLease(
            state=_state_from_row(task),
            worker_id=worker_id,
            lease_generation=task.lease_generation,
            budgets=TaskBudgets.model_validate(task.budgets),
            pending_traceparent=task.pending_traceparent,
            pending_tracestate=task.pending_tracestate,
        )

    def _claimable_task_conditions(
        self,
        session,
        *,
        capability_name: str,
        protocol_version: str,
        statuses: tuple[str, ...] = (
            TaskStatus.PLANNING.value,
            TaskStatus.RUNNING.value,
        ),
    ):
        clock = self._lease_clock_expression(session)
        return (
            AgentTaskRow.status.in_(statuses),
            # 协议门（required_capability/protocol 精确匹配）已隔离 M1/M2
            # Worker；M2 Worker 因此同时领取 Research 与受控任务，
            # 后者是 D6 崩溃窗口 3/5 的唯一重领取入口。
            AgentTaskRow.task_mode.in_(
                (TaskMode.RESEARCH.value, TaskMode.CONTROLLED_ACTION.value)
            ),
            AgentTaskRow.required_capability_name == capability_name,
            AgentTaskRow.required_protocol_version == protocol_version,
            AgentTaskRow.cancelled.is_(False),
            # 未应用恢复命令的正文仍受 task 侧门闩保护；通用 task claim
            # 绝不能绕过该门闩把同一 checkpoint 当作普通崩溃恢复。
            AgentTaskRow.active_resume_command_id.is_(None),
            AgentTaskRow.available_at <= clock,
            or_(
                AgentTaskRow.lease_owner.is_(None),
                AgentTaskRow.lease_expires_at <= clock,
            ),
        )

    def next_claimable_task_eligible_at(
        self,
        *,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> datetime | None:
        """返回普通 task 的最早候选时间；不领取、不改变 lease。"""
        capability_name, protocol_version, _ = self._claim_protocol(
            capability_name,
            protocol_version,
        )
        with self._database.sessions() as session:
            available_at = session.scalar(
                select(AgentTaskRow.available_at)
                .where(
                    *self._claimable_task_conditions(
                        session,
                        capability_name=capability_name,
                        protocol_version=protocol_version,
                    )
                )
                .order_by(AgentTaskRow.available_at, AgentTaskRow.created_at)
                .limit(1)
            )
        return _aware(available_at)

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
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    *self._claimable_task_conditions(
                        work.session,
                        capability_name=capability_name,
                        protocol_version=protocol_version,
                    )
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
            self._sync_conversation_turn(work.session, row, now=now)
            work.session.flush()
            return TaskLease(
                state=_state_from_row(row),
                worker_id=worker_id,
                lease_generation=row.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
                pending_traceparent=row.pending_traceparent,
                pending_tracestate=row.pending_tracestate,
            )

    def queue_snapshot(
        self,
        *,
        capability_name: str | None = None,
        protocol_version: str | None = None,
        limit: int,
    ) -> tuple[TaskQueueSnapshot, ...]:
        """固定状态/模式的有界快照，复用普通领取的行条件。

        capability lease 另以 gauge 上报；缺少 lease 时不能把已排队任务从积压视图中隐藏。
        """
        if limit <= 0:
            raise ValueError("invalid_task_queue_snapshot_limit")
        capability_name, protocol_version, _ = self._claim_protocol(
            capability_name, protocol_version
        )
        snapshots: list[TaskQueueSnapshot] = []
        with self._database.sessions() as session:
            now = self._lease_now(session)
            for task_mode in (TaskMode.RESEARCH.value, TaskMode.CONTROLLED_ACTION.value):
                for status in (
                    TaskStatus.PLANNING.value,
                    TaskStatus.RUNNING.value,
                    TaskStatus.WAITING_FOR_USER.value,
                    TaskStatus.WAITING_FOR_APPROVAL.value,
                ):
                    if status in {TaskStatus.PLANNING.value, TaskStatus.RUNNING.value}:
                        conditions = (
                            AgentTaskRow.task_mode == task_mode,
                            *self._claimable_task_conditions(
                                session,
                                capability_name=capability_name,
                                protocol_version=protocol_version,
                                statuses=(status,),
                            ),
                        )
                    else:
                        conditions = (
                            AgentTaskRow.task_mode == task_mode,
                            AgentTaskRow.status == status,
                            AgentTaskRow.required_capability_name == capability_name,
                            AgentTaskRow.required_protocol_version == protocol_version,
                            AgentTaskRow.cancelled.is_(False),
                        )
                    # ponytail: cap the periodic read to avoid a full queue scan;
                    # add a saturation metric if exact values above this limit are needed.
                    available = tuple(
                        session.scalars(
                            select(AgentTaskRow.available_at)
                            .where(*conditions)
                            .order_by(AgentTaskRow.available_at, AgentTaskRow.created_at)
                            .limit(limit)
                        )
                    )
                    oldest = _aware(available[0]) if available else None
                    age_seconds = (
                        max(0.0, (now - oldest).total_seconds())
                        if oldest is not None
                        else 0.0
                    )
                    snapshots.append(
                        TaskQueueSnapshot(
                            task_mode=task_mode,
                            status=status,
                            depth=len(available),
                            oldest_age_seconds=age_seconds,
                        )
                    )
        return tuple(snapshots)

    def enqueue_user_resume_command(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        answer: str,
    ) -> ResumeCommand:
        """锁定 task 后只写耐久命令；不构造 Agent 或 checkpoint。"""
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self.enqueue_user_resume_command_in_session(
                work.session,
                auth=auth,
                task_id=task_id,
                expected_state_version=expected_state_version,
                answer=answer,
            )

    def enqueue_user_resume_command_in_session(
        self,
        session,
        *,
        auth: AuthContext,
        task_id: str,
        expected_state_version: int,
        answer: str,
    ) -> ResumeCommand:
        """在调用方事务中锁定任务并写入澄清恢复命令。"""
        payload_body = self._normalize_resume_body(
            answer, required=True, maximum_length=2_000
        )
        self._begin_resume_write(session)
        task = self._lock_resume_task(
            session,
            tenant_id=auth.tenant_id,
            task_id=task_id,
            skip_locked=False,
        )
        if task is None or task.user_id != auth.user_id:
            # 与既有 task 读取一致，不以跨租户/非所有者差异泄漏存在性。
            raise TaskRepositoryError("task_not_found")
        now = self._lease_now(session)
        return self.enqueue_resume_command_in_session(
            session,
            task=task,
            actor_principal_id=auth.user_id,
            actor_kind="user",
            kind=ResumeCommandKind.USER_CLARIFICATION,
            expected_task_state_version=expected_state_version,
            payload_body=payload_body,
            approval=None,
            decision=None,
            now=now,
        )

    def enqueue_resume_command_in_session(
        self,
        session,
        *,
        task: AgentTaskRow,
        actor_principal_id: str,
        actor_kind: str,
        kind: ResumeCommandKind,
        expected_task_state_version: int,
        payload_body: str | None,
        approval: ApprovalRow | None,
        decision: ApprovalResumeDecision | None,
        now: datetime | None = None,
    ) -> ResumeCommand:
        """A4 可复用的同事务插入原语；调用方必须已按 task→approval 锁定。"""
        if task.state_version != expected_task_state_version:
            raise TaskRepositoryError("task_state_conflict")
        if task.active_resume_command_id is not None:
            raise TaskRepositoryError("task_resume_conflict")
        if (
            not task.required_protocol_version
            or not task.required_protocol_version.strip()
        ):
            raise TaskRepositoryError("task_protocol_mismatch")
        approval_id = approval.approval_id if approval is not None else None
        if not self._complete_resume_binding(
            task,
            kind=kind,
            approval_id=approval_id,
        ):
            raise TaskRepositoryError("task_resume_binding_missing")
        if kind is ResumeCommandKind.TOOL_APPROVAL and not self._approval_matches_resume(
            task,
            approval,
            approval_id=approval_id,
            decision=decision,
        ):
            raise TaskRepositoryError("task_resume_kind_invalid")
        if kind is ResumeCommandKind.USER_CLARIFICATION:
            payload_body = self._normalize_resume_body(
                payload_body, required=True, maximum_length=2_000
            )
        else:
            payload_body = self._normalize_resume_body(
                payload_body, required=False, maximum_length=200
            )
        if actor_kind not in {"user", "service"} or not actor_principal_id.strip():
            raise ValueError("resume_command_actor_invalid")
        now = now or self._lease_now(session)
        expires_at = self._resume_expiry(task, approval=approval, now=now)
        target_key = self._resume_target_key(
            task,
            kind=kind,
            approval_id=approval_id,
        )
        payload_sha256 = self._resume_payload_sha256(
            kind=kind,
            decision=decision,
            payload_body=payload_body,
        )
        idempotency_key = self._resume_idempotency_key(
            target_key=target_key,
            payload_sha256=payload_sha256,
        )
        existing_target = session.scalar(
            select(TaskResumeCommandRow).where(
                TaskResumeCommandRow.tenant_id == task.tenant_id,
                TaskResumeCommandRow.resume_target_key == target_key,
            )
        )
        if existing_target is not None:
            if existing_target.idempotency_key == idempotency_key:
                return _command_from_row(existing_target)
            raise TaskRepositoryError("task_resume_conflict")
        existing_idempotency = session.scalar(
            select(TaskResumeCommandRow).where(
                TaskResumeCommandRow.tenant_id == task.tenant_id,
                TaskResumeCommandRow.idempotency_key == idempotency_key,
            )
        )
        if existing_idempotency is not None:
            return _command_from_row(existing_idempotency)
        row = TaskResumeCommandRow(
            command_id=uuid4().hex,
            tenant_id=task.tenant_id,
            task_id=task.task_id,
            kind=kind.value,
            actor_principal_id=actor_principal_id,
            actor_kind=actor_kind,
            expected_task_state_version=task.state_version,
            expected_interrupt_checkpoint_ns=task.interrupt_checkpoint_ns,
            expected_interrupt_checkpoint_id=task.interrupt_checkpoint_id,
            expected_interrupt_id=task.interrupt_id,
            expected_protocol_version=task.required_protocol_version,
            approval_id=approval_id,
            decision=decision.value if decision is not None else None,
            payload_body=payload_body,
            payload_sha256=payload_sha256,
            resume_target_key=target_key,
            idempotency_key=idempotency_key,
            status=ResumeCommandStatus.PENDING.value,
            attempt_count=0,
            next_attempt_at=now,
            expires_at=expires_at,
            lease_generation=0,
            created_at=now,
            updated_at=now,
        )
        try:
            with session.begin_nested():
                session.add(row)
                session.flush()
        except IntegrityError:
            # 全局唯一约束是最终并发防线；错误只映射为稳定冲突，不能把 SQL
            # 参数（其中包含正文）经异常链泄漏给调用方。
            winner = session.scalar(
                select(TaskResumeCommandRow).where(
                    TaskResumeCommandRow.tenant_id == task.tenant_id,
                    TaskResumeCommandRow.resume_target_key == target_key,
                )
            )
            if winner is not None and winner.idempotency_key == idempotency_key:
                return _command_from_row(winner)
            raise TaskRepositoryError("task_resume_conflict") from None
        return _command_from_row(row)

    def resume_command_candidates(
        self,
        *,
        limit: int,
    ) -> tuple[ResumeCommandCandidate, ...]:
        """无锁候选发现；真正领取始终重新按 task→approval→command 锁定。"""
        if limit <= 0:
            raise ValueError("invalid_resume_command_candidate_limit")
        with self._database.sessions() as session:
            now = self._lease_now(session)
            rows = tuple(
                session.execute(
                    select(
                        TaskResumeCommandRow.command_id,
                        TaskResumeCommandRow.tenant_id,
                        TaskResumeCommandRow.task_id,
                        TaskResumeCommandRow.kind,
                        TaskResumeCommandRow.approval_id,
                        TaskResumeCommandRow.status,
                        TaskResumeCommandRow.next_attempt_at,
                        TaskResumeCommandRow.lease_expires_at,
                        TaskResumeCommandRow.applied_checkpoint_id,
                    )
                    .where(
                        or_(
                            # checkpoint 后、consume 前崩溃：marker 已是唯一
                            # 重放分界，立即进入只结算路径，绝不等待 lease 到期
                            # 或再次构造 Command(resume=...)。
                            (
                                TaskResumeCommandRow.status
                                == ResumeCommandStatus.CLAIMED.value
                            )
                            & TaskResumeCommandRow.applied_checkpoint_id.is_not(None),
                            (
                                TaskResumeCommandRow.status
                                == ResumeCommandStatus.PENDING.value
                            )
                            & (TaskResumeCommandRow.next_attempt_at <= now)
                            & (TaskResumeCommandRow.expires_at > now),
                            (
                                TaskResumeCommandRow.status
                                == ResumeCommandStatus.CLAIMED.value
                            )
                            & (TaskResumeCommandRow.lease_expires_at <= now),
                        )
                    )
                    .order_by(
                        case(
                            (
                                TaskResumeCommandRow.applied_checkpoint_id.is_not(None),
                                TaskResumeCommandRow.next_attempt_at,
                            ),
                            (
                                TaskResumeCommandRow.status
                                == ResumeCommandStatus.PENDING.value,
                                TaskResumeCommandRow.next_attempt_at,
                            ),
                            else_=TaskResumeCommandRow.lease_expires_at,
                        ),
                        TaskResumeCommandRow.created_at,
                    )
                    .limit(limit)
                ).all()
            )
        return tuple(
            ResumeCommandCandidate(
                command_id=row.command_id,
                tenant_id=row.tenant_id,
                task_id=row.task_id,
                kind=ResumeCommandKind(row.kind),
                approval_id=row.approval_id,
                status=ResumeCommandStatus(row.status),
                eligible_at=_aware(
                    row.next_attempt_at
                    if row.status == ResumeCommandStatus.PENDING.value
                    or row.applied_checkpoint_id is not None
                    else row.lease_expires_at
                ),
            )
            for row in rows
        )

    def claim_resume_command(
        self,
        *,
        command_id: str,
        worker_id: str,
        lease_seconds: int,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> ResumeCommandLease | None:
        if not command_id or not worker_id or lease_seconds <= 0:
            raise ValueError("invalid_resume_command_claim")
        capability_name, protocol_version, require_capability = self._claim_protocol(
            capability_name,
            protocol_version,
        )
        with self._database.sessions() as session:
            row = session.execute(
                select(
                    TaskResumeCommandRow.command_id,
                    TaskResumeCommandRow.tenant_id,
                    TaskResumeCommandRow.task_id,
                    TaskResumeCommandRow.kind,
                    TaskResumeCommandRow.approval_id,
                    TaskResumeCommandRow.status,
                    TaskResumeCommandRow.next_attempt_at,
                        TaskResumeCommandRow.lease_expires_at,
                        TaskResumeCommandRow.applied_checkpoint_id,
                ).where(TaskResumeCommandRow.command_id == command_id)
            ).one_or_none()
        if row is None:
            return None
        candidate = ResumeCommandCandidate(
            command_id=row.command_id,
            tenant_id=row.tenant_id,
            task_id=row.task_id,
            kind=ResumeCommandKind(row.kind),
            approval_id=row.approval_id,
            status=ResumeCommandStatus(row.status),
            eligible_at=_aware(
                    row.next_attempt_at
                    if row.status == ResumeCommandStatus.PENDING.value
                    or row.applied_checkpoint_id is not None
                    else row.lease_expires_at
            )
            or datetime.now(UTC),
        )
        return self._claim_resume_candidate(
            candidate,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            capability_name=capability_name,
            protocol_version=protocol_version,
            require_capability=require_capability,
        )

    def _claim_resume_candidate(
        self,
        candidate: ResumeCommandCandidate,
        *,
        worker_id: str,
        lease_seconds: int,
        capability_name: str,
        protocol_version: str,
        require_capability: bool,
    ) -> ResumeCommandLease | None:
        """候选先无锁读取，再依固定锁序重读并原子领取。"""
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._begin_resume_write(work.session)
            task = self._lock_resume_task(
                work.session,
                tenant_id=candidate.tenant_id,
                task_id=candidate.task_id,
                skip_locked=True,
            )
            if task is None:
                return None
            if require_capability and (
                task.required_capability_name != capability_name
                or task.required_protocol_version != protocol_version
                or not self._has_active_worker_capability_in_session(
                    work.session,
                    capability_name=capability_name,
                    protocol_version=protocol_version,
                    worker_id=worker_id,
                    for_update=False,
                )
            ):
                return None
            approval: ApprovalRow | None = None
            if candidate.kind is ResumeCommandKind.TOOL_APPROVAL:
                if candidate.approval_id is None:
                    return None
                approval = self._lock_resume_approval(
                    work.session,
                    tenant_id=candidate.tenant_id,
                    approval_id=candidate.approval_id,
                    skip_locked=True,
                )
                if approval is None:
                    return None
            command = self._lock_resume_command(
                work.session,
                tenant_id=candidate.tenant_id,
                command_id=candidate.command_id,
                skip_locked=True,
            )
            if (
                command is None
                or command.task_id != task.task_id
                or command.kind != candidate.kind.value
                or command.approval_id != candidate.approval_id
            ):
                return None
            now = self._lease_now(work.session)
            if command.applied_checkpoint_id is not None:
                self._settle_applied_locked(task, command, now=now)
                return None
            if command.status == ResumeCommandStatus.PENDING.value:
                if not self._pending_resume_claimable(
                    task,
                    approval,
                    command,
                    now=now,
                ):
                    return None
                shared_expires_at = now + timedelta(seconds=lease_seconds)
                task.status = TaskStatus.RUNNING.value
                task.lease_owner = worker_id
                task.lease_expires_at = shared_expires_at
                task.lease_generation += 1
                task.state_version += 1
                task.active_resume_command_id = command.command_id
                task.interrupt_kind = None
                task.interrupt_summary = None
                task.interrupt_ref = None
                task.interrupt_checkpoint_ns = None
                task.interrupt_checkpoint_id = None
                task.interrupt_id = None
                task.interrupt_approval_id = None
                task.updated_at = now
                command.status = ResumeCommandStatus.CLAIMED.value
                command.lease_owner = worker_id
                command.lease_expires_at = shared_expires_at
                command.lease_generation += 1
                command.claimed_at = now
                command.claimed_task_state_version = task.state_version
                command.claimed_task_lease_generation = task.lease_generation
                command.attempt_count += 1
                command.updated_at = now
            elif command.status == ResumeCommandStatus.CLAIMED.value:
                if not self._claimed_resume_reclaimable(task, approval, command, now=now):
                    return None
                shared_expires_at = now + timedelta(seconds=lease_seconds)
                task.lease_owner = worker_id
                task.lease_expires_at = shared_expires_at
                task.lease_generation += 1
                task.updated_at = now
                command.lease_owner = worker_id
                command.lease_expires_at = shared_expires_at
                command.lease_generation += 1
                command.claimed_at = now
                command.claimed_task_lease_generation = task.lease_generation
                command.attempt_count += 1
                command.updated_at = now
            else:
                return None
            self._sync_conversation_turn(work.session, task, now=now)
            work.session.flush()
            return ResumeCommandLease(
                command=_command_from_row(command),
                task_lease=self._resume_task_lease(task, worker_id=worker_id),
                worker_id=worker_id,
                command_lease_generation=command.lease_generation,
            )

    def _pending_resume_claimable(
        self,
        task: AgentTaskRow,
        approval: ApprovalRow | None,
        command: TaskResumeCommandRow,
        *,
        now: datetime,
    ) -> bool:
        if (
            command.status != ResumeCommandStatus.PENDING.value
            or _aware(command.next_attempt_at) > now
            or _aware(command.expires_at) <= now
            or task.active_resume_command_id is not None
            or task.state_version != command.expected_task_state_version
            or task.required_protocol_version != command.expected_protocol_version
            or task.cancelled
            or (_aware(task.deadline) is not None and _aware(task.deadline) <= now)
        ):
            return False
        kind = ResumeCommandKind(command.kind)
        approval_id = command.approval_id
        if not self._complete_resume_binding(task, kind=kind, approval_id=approval_id):
            return False
        if (
            task.interrupt_checkpoint_ns != command.expected_interrupt_checkpoint_ns
            or task.interrupt_checkpoint_id != command.expected_interrupt_checkpoint_id
            or task.interrupt_id != command.expected_interrupt_id
        ):
            return False
        if kind is ResumeCommandKind.USER_CLARIFICATION:
            return command.approval_id is None and command.decision is None
        try:
            decision = ApprovalResumeDecision(command.decision or "")
        except ValueError:
            return False
        return self._approval_matches_resume(
            task,
            approval,
            approval_id=approval_id,
            decision=decision,
        )

    def _claimed_resume_reclaimable(
        self,
        task: AgentTaskRow,
        approval: ApprovalRow | None,
        command: TaskResumeCommandRow,
        *,
        now: datetime,
    ) -> bool:
        command_expires_at = _aware(command.lease_expires_at)
        task_expires_at = _aware(task.lease_expires_at)
        if (
            command.status != ResumeCommandStatus.CLAIMED.value
            or command.applied_checkpoint_id is not None
            or command.lease_owner is None
            or command_expires_at is None
            or task.status != TaskStatus.RUNNING.value
            or task.active_resume_command_id != command.command_id
            or task.lease_owner != command.lease_owner
            or task_expires_at is None
            or task_expires_at != command_expires_at
            or task_expires_at > now
            or command_expires_at > now
            or task.state_version != command.expected_task_state_version + 1
            or task.state_version != command.claimed_task_state_version
            or task.lease_generation != command.claimed_task_lease_generation
            or task.required_protocol_version != command.expected_protocol_version
            or _aware(command.expires_at) <= now
            or task.cancelled
            or (_aware(task.deadline) is not None and _aware(task.deadline) <= now)
        ):
            return False
        kind = ResumeCommandKind(command.kind)
        if kind is ResumeCommandKind.USER_CLARIFICATION:
            return command.approval_id is None and command.decision is None
        return self._approval_matches_claimed_resume(task, approval, command)

    def _lock_resume_lease_rows(
        self,
        session,
        lease: ResumeCommandLease,
        *,
        skip_locked: bool,
    ) -> tuple[AgentTaskRow, ApprovalRow | None, TaskResumeCommandRow] | None:
        task = self._lock_resume_task(
            session,
            tenant_id=lease.command.tenant_id,
            task_id=lease.command.task_id,
            skip_locked=skip_locked,
        )
        if task is None:
            return None
        approval: ApprovalRow | None = None
        if lease.command.kind is ResumeCommandKind.TOOL_APPROVAL:
            if lease.command.approval_id is None:
                return None
            approval = self._lock_resume_approval(
                session,
                tenant_id=lease.command.tenant_id,
                approval_id=lease.command.approval_id,
                skip_locked=skip_locked,
            )
            if approval is None:
                return None
        command = self._lock_resume_command(
            session,
            tenant_id=lease.command.tenant_id,
            command_id=lease.command.command_id,
            skip_locked=skip_locked,
        )
        if (
            command is None
            or command.task_id != task.task_id
            or command.kind != lease.command.kind.value
            or command.approval_id != lease.command.approval_id
        ):
            return None
        return task, approval, command

    def _lock_nonterminal_resume_commands_for_task(
        self,
        session,
        *,
        task: AgentTaskRow,
    ) -> tuple[TaskResumeCommandRow, ...] | None:
        """task 已锁定后，按 approval→command 顺序锁定其未结算命令。"""
        locators = tuple(
            session.execute(
                select(
                    TaskResumeCommandRow.command_id,
                    TaskResumeCommandRow.kind,
                    TaskResumeCommandRow.approval_id,
                )
                .where(
                    TaskResumeCommandRow.tenant_id == task.tenant_id,
                    TaskResumeCommandRow.task_id == task.task_id,
                    TaskResumeCommandRow.status.in_(
                        (
                            ResumeCommandStatus.PENDING.value,
                            ResumeCommandStatus.CLAIMED.value,
                        )
                    ),
                )
                .order_by(
                    TaskResumeCommandRow.approval_id,
                    TaskResumeCommandRow.command_id,
                )
            ).all()
        )
        approval_ids = sorted(
            {
                locator.approval_id
                for locator in locators
                if locator.kind == ResumeCommandKind.TOOL_APPROVAL.value
                and locator.approval_id is not None
            }
        )
        if any(
            locator.kind == ResumeCommandKind.TOOL_APPROVAL.value
            and locator.approval_id is None
            for locator in locators
        ):
            return None
        for approval_id in approval_ids:
            if self._lock_resume_approval(
                session,
                tenant_id=task.tenant_id,
                approval_id=approval_id,
                skip_locked=False,
            ) is None:
                return None
        commands: list[TaskResumeCommandRow] = []
        for locator in sorted(locators, key=lambda item: item.command_id):
            command = self._lock_resume_command(
                session,
                tenant_id=task.tenant_id,
                command_id=locator.command_id,
                skip_locked=False,
            )
            if (
                command is None
                or command.task_id != task.task_id
                or command.kind != locator.kind
                or command.approval_id != locator.approval_id
                or command.status
                not in {
                    ResumeCommandStatus.PENDING.value,
                    ResumeCommandStatus.CLAIMED.value,
                }
            ):
                return None
            commands.append(command)
        return tuple(commands)

    def _verify_resume_lease_fence(
        self,
        task: AgentTaskRow,
        command: TaskResumeCommandRow,
        lease: ResumeCommandLease,
        *,
        now: datetime,
        expect_active_command: bool,
    ) -> None:
        task_expires_at = _aware(task.lease_expires_at)
        command_expires_at = _aware(command.lease_expires_at)
        if (
            command.status != ResumeCommandStatus.CLAIMED.value
            or command.command_id != lease.command.command_id
            or command.lease_owner != lease.worker_id
            or command.lease_generation != lease.command_lease_generation
            or task.status != TaskStatus.RUNNING.value
            or task.lease_owner != lease.worker_id
            or task.lease_generation != lease.task_lease.lease_generation
            or command.claimed_task_state_version != command.expected_task_state_version + 1
            or command.claimed_task_lease_generation != lease.task_lease.lease_generation
            or task_expires_at is None
            or command_expires_at is None
            or task_expires_at != command_expires_at
        ):
            raise TaskRepositoryError("task_resume_lease_stale")
        if task_expires_at <= now or command_expires_at <= now:
            raise TaskRepositoryError("task_resume_lease_expired")
        if expect_active_command:
            if (
                task.active_resume_command_id != command.command_id
                or command.claimed_task_state_version != task.state_version
                or command.claimed_task_lease_generation != task.lease_generation
            ):
                raise TaskRepositoryError("task_resume_lease_stale")
        elif task.active_resume_command_id is not None:
            raise TaskRepositoryError("task_resume_lease_stale")

    def validate_resume_command_execution(
        self,
        lease: ResumeCommandLease,
        *,
        capability_name: str | None = None,
        protocol_version: str | None = None,
    ) -> ResumeCommand:
        """在注入 resume 前按双 lease 重读当前 actor 与完整业务 binding。

        该检查刻意不信任 enqueue 时的权限快照：命令只保存 actor 标识；
        每次真正恢复都从当前 principals/approval/task 事实重新判定。命令
        正文不参与任何异常或事件。
        """
        capability_name, protocol_version, require_capability = self._claim_protocol(
            capability_name,
            protocol_version,
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._begin_resume_write(work.session)
            rows = self._lock_resume_lease_rows(
                work.session,
                lease,
                skip_locked=False,
            )
            if rows is None:
                raise TaskRepositoryError("task_resume_lease_stale")
            task, approval, command = rows
            now = self._lease_now(work.session)
            self._verify_resume_lease_fence(
                task,
                command,
                lease,
                now=now,
                expect_active_command=True,
            )
            if (
                task.required_capability_name != capability_name
                or task.required_protocol_version != protocol_version
                or command.expected_protocol_version != protocol_version
            ):
                raise TaskRepositoryError("task_protocol_mismatch")
            if require_capability and not self._has_active_worker_capability_in_session(
                work.session,
                capability_name=capability_name,
                protocol_version=protocol_version,
                worker_id=lease.worker_id,
                for_update=True,
            ):
                raise TaskRepositoryError("task_agent_unavailable")
            deadline = _aware(task.deadline)
            if task.cancelled or deadline is not None and deadline <= now:
                raise TaskRepositoryError("task_resume_expired")
            if command.actor_kind != "user":
                raise TaskRepositoryError("task_resume_actor_unauthorized")
            actor = work.session.scalar(
                select(PrincipalRow)
                .where(
                    PrincipalRow.tenant_id == task.tenant_id,
                    PrincipalRow.user_id == command.actor_principal_id,
                )
                .with_for_update()
            )
            if actor is None or not actor.active:
                raise TaskRepositoryError("task_resume_actor_unavailable")
            actor_auth = AuthContext(
                tenant_id=actor.tenant_id,
                user_id=actor.user_id,
                roles=frozenset(str(role) for role in actor.roles),
                attributes=actor.attributes,
                request_id=current_request_id() or "task-resume-worker",
            )
            kind = ResumeCommandKind(command.kind)
            if kind is ResumeCommandKind.USER_CLARIFICATION:
                if (
                    command.approval_id is not None
                    or command.decision is not None
                    or command.actor_principal_id != task.user_id
                    or not actor_auth.has_permission(Permission.TASK_READ)
                ):
                    raise TaskRepositoryError("task_resume_actor_unauthorized")
                return _command_from_row(command)

            try:
                decision = ApprovalResumeDecision(command.decision or "")
            except ValueError as exc:
                raise TaskRepositoryError("task_resume_binding_invalid") from exc
            if (
                approval is None
                or command.approval_id is None
                or approval.approval_id != command.approval_id
                or approval.tenant_id != task.tenant_id
                or approval.task_id != task.task_id
                or task.task_mode != TaskMode.CONTROLLED_ACTION.value
                or approval.thread_id != task.task_id
                or approval.checkpoint_ns != command.expected_interrupt_checkpoint_ns
                or approval.checkpoint_id != command.expected_interrupt_checkpoint_id
                or approval.interrupt_id != command.expected_interrupt_id
                or not isinstance(approval.tool_call_id, str)
                or not approval.tool_call_id.strip()
                or not isinstance(approval.checkpoint_binding_hash, str)
                or not approval.checkpoint_binding_hash.strip()
                or command.actor_principal_id == task.user_id
                or not actor_auth.has_permission(Permission.TASK_APPROVE)
            ):
                raise TaskRepositoryError("task_resume_binding_invalid")
            if decision is ApprovalResumeDecision.APPROVED:
                if approval.status != "approved" or approval.approved_by != actor.user_id:
                    raise TaskRepositoryError("task_resume_binding_invalid")
            elif approval.status != "rejected" or approval.rejected_by != actor.user_id:
                raise TaskRepositoryError("task_resume_binding_invalid")
            return _command_from_row(command)

    @staticmethod
    def _terminalize_resume_command(
        command: TaskResumeCommandRow,
        *,
        status: ResumeCommandStatus,
        now: datetime,
        failure_code: str | None = None,
    ) -> None:
        command.status = status.value
        command.payload_body = None
        command.body_purged_at = now
        command.lease_owner = None
        command.lease_expires_at = None
        command.consumed_at = now if status is ResumeCommandStatus.CONSUMED else None
        command.failed_at = now if status is ResumeCommandStatus.FAILED else None
        command.expired_at = now if status is ResumeCommandStatus.EXPIRED else None
        command.failure_code = failure_code
        command.updated_at = now

    def refresh_resume_command_lease(
        self,
        lease: ResumeCommandLease,
        *,
        lease_seconds: int,
    ) -> ResumeCommandLease:
        if lease_seconds <= 0:
            raise ValueError("invalid_resume_command_claim")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._begin_resume_write(work.session)
            rows = self._lock_resume_lease_rows(
                work.session,
                lease,
                skip_locked=False,
            )
            if rows is None:
                raise TaskRepositoryError("task_resume_lease_stale")
            task, _, command = rows
            now = self._lease_now(work.session)
            self._verify_resume_lease_fence(
                task,
                command,
                lease,
                now=now,
                # 首次 checkpoint 已原子写 marker 后，task 门闩已经解除，
                # 但当前 Worker 仍须保持 command/task 同源 expiry，直至
                # consume。此时只允许同一 owner/generation 的后 marker 续约。
                expect_active_command=command.applied_checkpoint_id is None,
            )
            shared_expires_at = now + timedelta(seconds=lease_seconds)
            task.lease_expires_at = shared_expires_at
            task.updated_at = now
            command.lease_expires_at = shared_expires_at
            command.updated_at = now
            work.session.flush()
            return ResumeCommandLease(
                command=_command_from_row(command),
                task_lease=self._resume_task_lease(task, worker_id=lease.worker_id),
                worker_id=lease.worker_id,
                command_lease_generation=command.lease_generation,
            )

    def consume_resume_command(self, lease: ResumeCommandLease) -> ResumeCommand:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._begin_resume_write(work.session)
            rows = self._lock_resume_lease_rows(
                work.session,
                lease,
                skip_locked=False,
            )
            if rows is None:
                raise TaskRepositoryError("task_resume_lease_stale")
            task, _, command = rows
            now = self._lease_now(work.session)
            self._verify_resume_lease_fence(
                task,
                command,
                lease,
                now=now,
                expect_active_command=False,
            )
            if command.applied_checkpoint_id is None:
                raise TaskRepositoryError("task_resume_apply_marker_missing")
            self._terminalize_resume_command(
                command,
                status=ResumeCommandStatus.CONSUMED,
                now=now,
            )
            work.session.flush()
            return _command_from_row(command)

    def _settle_applied_locked(
        self,
        task: AgentTaskRow,
        command: TaskResumeCommandRow,
        *,
        now: datetime,
    ) -> bool:
        if (
            command.status != ResumeCommandStatus.CLAIMED.value
            or command.applied_checkpoint_id is None
            or task.active_resume_command_id is not None
        ):
            return False
        self._terminalize_resume_command(
            command,
            status=ResumeCommandStatus.CONSUMED,
            now=now,
        )
        return True

    def settle_applied_resume_command(self, *, command_id: str) -> ResumeCommand | None:
        """checkpoint 后、consume 前崩溃时只结算，不再次注入 resume。"""
        with self._database.sessions() as session:
            locator = session.execute(
                select(
                    TaskResumeCommandRow.tenant_id,
                    TaskResumeCommandRow.task_id,
                    TaskResumeCommandRow.kind,
                    TaskResumeCommandRow.approval_id,
                ).where(TaskResumeCommandRow.command_id == command_id)
            ).one_or_none()
        if locator is None:
            return None
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._begin_resume_write(work.session)
            task = self._lock_resume_task(
                work.session,
                tenant_id=locator.tenant_id,
                task_id=locator.task_id,
                skip_locked=False,
            )
            if task is None:
                return None
            if locator.kind == ResumeCommandKind.TOOL_APPROVAL.value:
                if locator.approval_id is None or self._lock_resume_approval(
                    work.session,
                    tenant_id=locator.tenant_id,
                    approval_id=locator.approval_id,
                    skip_locked=False,
                ) is None:
                    return None
            command = self._lock_resume_command(
                work.session,
                tenant_id=locator.tenant_id,
                command_id=command_id,
                skip_locked=False,
            )
            if command is None or command.task_id != task.task_id:
                return None
            if command.status == ResumeCommandStatus.CONSUMED.value:
                return _command_from_row(command)
            now = self._lease_now(work.session)
            if not self._settle_applied_locked(task, command, now=now):
                return None
            work.session.flush()
            return _command_from_row(command)

    def _terminalize_claimed_resume(
        self,
        lease: ResumeCommandLease,
        *,
        command_status: ResumeCommandStatus,
        failure_code: str,
    ) -> ResumeCommand:
        failure_code = self._resume_failure_code(failure_code)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._begin_resume_write(work.session)
            rows = self._lock_resume_lease_rows(
                work.session,
                lease,
                skip_locked=False,
            )
            if rows is None:
                raise TaskRepositoryError("task_resume_lease_stale")
            task, _, command = rows
            now = self._lease_now(work.session)
            self._verify_resume_lease_fence(
                task,
                command,
                lease,
                now=now,
                expect_active_command=True,
            )
            if command.applied_checkpoint_id is not None:
                raise TaskRepositoryError("task_resume_already_applied")
            task.status = TaskStatus.FAILED.value
            task.cancelled = False
            task.last_error_code = failure_code
            task.active_resume_command_id = None
            task.lease_owner = None
            task.lease_expires_at = None
            task.state_version += 1
            task.updated_at = now
            self._set_terminal_content_expiry(work.session, task, now=now)
            self._terminalize_resume_command(
                command,
                status=command_status,
                now=now,
                failure_code=failure_code,
            )
            self._sync_conversation_turn(work.session, task, now=now)
            work.session.flush()
            return _command_from_row(command)

    def fail_resume_command(
        self,
        lease: ResumeCommandLease,
        *,
        failure_code: str,
    ) -> ResumeCommand:
        return self._terminalize_claimed_resume(
            lease,
            command_status=ResumeCommandStatus.FAILED,
            failure_code=failure_code,
        )

    def expire_claimed_resume_command(
        self,
        lease: ResumeCommandLease,
        *,
        failure_code: str = "task_resume_expired",
    ) -> ResumeCommand:
        return self._terminalize_claimed_resume(
            lease,
            command_status=ResumeCommandStatus.EXPIRED,
            failure_code=failure_code,
        )

    def expire_waiting_tasks(self, *, limit: int = 100) -> int:
        """把超时的澄清/审批等待原子收敛为失败，后续交给正文保留策略。"""
        if limit <= 0:
            raise ValueError("invalid_waiting_reaper_limit")
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._lease_clock_expression(work.session)
            rows: list[AgentTaskRow] = []
            for task_mode, status in (
                (TaskMode.RESEARCH.value, TaskStatus.WAITING_FOR_USER.value),
                (TaskMode.CONTROLLED_ACTION.value, TaskStatus.WAITING_FOR_APPROVAL.value),
                (TaskMode.CONTROLLED_ACTION.value, TaskStatus.WAITING_FOR_USER.value),
            ):
                remaining = limit - len(rows)
                if remaining == 0:
                    break
                rows.extend(
                    work.session.scalars(
                        select(AgentTaskRow)
                        .where(
                            AgentTaskRow.task_mode == task_mode,
                            AgentTaskRow.status == status,
                            AgentTaskRow.deadline.is_not(None),
                            AgentTaskRow.deadline <= clock,
                        )
                        .order_by(AgentTaskRow.deadline, AgentTaskRow.task_id)
                        .limit(remaining)
                        .with_for_update(skip_locked=True)
                    )
                )
            now = self._lease_now(work.session)
            expired = 0
            for row in rows:
                commands = self._lock_nonterminal_resume_commands_for_task(
                    work.session,
                    task=row,
                )
                if commands is None:
                    continue
                for command in commands:
                    self._terminalize_resume_command(
                        command,
                        status=(
                            ResumeCommandStatus.CONSUMED
                            if command.applied_checkpoint_id is not None
                            else ResumeCommandStatus.EXPIRED
                        ),
                        now=now,
                        failure_code=(
                            None
                            if command.applied_checkpoint_id is not None
                            else "task_deadline_expired"
                        ),
                    )
                row.status = TaskStatus.FAILED.value
                row.last_error_code = "task_deadline_expired"
                self._set_terminal_content_expiry(work.session, row, now=now)
                row.interrupt_kind = None
                row.interrupt_summary = None
                row.interrupt_ref = None
                row.interrupt_checkpoint_ns = None
                row.interrupt_checkpoint_id = None
                row.interrupt_id = None
                row.interrupt_approval_id = None
                row.lease_owner = None
                row.lease_expires_at = None
                row.state_version += 1
                row.updated_at = now
                self._sync_conversation_turn(work.session, row, now=now)
                work.session.flush()
                self._append_event(
                    work.session,
                    row,
                    event_type="task_failed",
                    correlation_id="task_deadline_expired",
                    run_id=None,
                )
                expired += 1
            return expired

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
                self._sync_conversation_turn(work.session, row, now=now)
            return len(rows)

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
            if row.active_resume_command_id is not None:
                raise TaskRepositoryError("task_resume_lease_stale")
            if lease_seconds is not None:
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.updated_at = now
                work.session.flush()
            return TaskLease(
                state=_state_from_row(row),
                worker_id=lease.worker_id,
                lease_generation=lease.lease_generation,
                budgets=TaskBudgets.model_validate(row.budgets),
                pending_traceparent=row.pending_traceparent,
                pending_tracestate=row.pending_tracestate,
            )

    def pause_for_user(
        self,
        lease: TaskLease,
        *,
        question: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        interrupt_id: str,
        run_id: str,
    ) -> TaskState:
        """模型请求澄清：保存摘要及服务端验证过的低敏 resume binding。"""
        if (
            not question.strip()
            or not isinstance(checkpoint_ns, str)
            or not isinstance(checkpoint_id, str)
            or not checkpoint_id.strip()
            or not isinstance(interrupt_id, str)
            or not interrupt_id.strip()
        ):
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
            # 根 namespace 合法地为 ""；NULL 才表示 binding 不完整。
            row.interrupt_checkpoint_ns = checkpoint_ns
            row.interrupt_checkpoint_id = checkpoint_id.strip()
            row.interrupt_id = interrupt_id.strip()
            row.interrupt_approval_id = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
            self._sync_conversation_turn(
                work.session, row, now=now, clarification_question=row.interrupt_summary
            )
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_waiting_for_user",
                correlation_id=run_id,
                run_id=run_id,
            )
            return _state_from_row(row)

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
    ) -> TaskState:
        """工具审批暂停：业务表只保存裁剪摘要与不透明审批引用。

        原始工具参数、HITL payload 与 resume 数据一律不进入 agent_tasks；
        正文事实留在 checkpointer 线程内。
        """
        if (
            not interrupt_ref.strip()
            or not summary.strip()
            or not isinstance(checkpoint_ns, str)
            or not isinstance(checkpoint_id, str)
            or not checkpoint_id.strip()
            or not isinstance(interrupt_id, str)
            or not interrupt_id.strip()
        ):
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
            # approval binding 与 snapshot 的根 namespace 必须原样一致。
            row.interrupt_checkpoint_ns = checkpoint_ns
            row.interrupt_checkpoint_id = checkpoint_id.strip()
            row.interrupt_id = interrupt_id.strip()
            row.interrupt_approval_id = interrupt_ref.strip()[:64]
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
            self._sync_conversation_turn(work.session, row, now=now)
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_waiting_for_approval",
                correlation_id=run_id,
                run_id=run_id,
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
        run_id: str,
        auth_request_id: str | None = None,
        counts: dict[str, int] | None = None,
        conversation_message: str | None = None,
        conversation_evidence_ids: tuple[str, ...] = (),
        conversation_artifacts: tuple[dict[str, object], ...] = (),
    ) -> TaskState:
        """终态写入：succeeded 必须原子绑定有效结果（计划 §11.4 门禁）。"""
        if status not in _TERMINAL_STATUSES:
            raise ValueError("task_status_not_terminal")
        if counts is None:
            event_counts: dict[str, int] = {}
        elif not isinstance(counts, dict) or any(
            not isinstance(name, str) or type(value) is not int or value < 0
            for name, value in counts.items()
        ):
            raise ValueError("task_event_counts_invalid")
        else:
            event_counts = dict(counts)
        is_result_submission = status in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED}
        # result_kind 的精确值按 task mode 在行锁内校验（受控任务为
        # controlled_action）；此处只做通用非空检查。
        if is_result_submission and (result_ref is None or result_hash is None):
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
            if row.active_resume_command_id is not None:
                raise TaskRepositoryError("task_resume_lease_stale")
            if is_result_submission:
                expected_kind = (
                    "controlled_action"
                    if row.task_mode == TaskMode.CONTROLLED_ACTION.value
                    else "research"
                )
                if result_kind != expected_kind:
                    raise ValueError("task_result_binding_required")
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
                        request_id=auth_request_id or current_request_id() or "worker-auth-revalidation",
                    )
                except TaskExecutionContextError as exc:
                    raise TaskRepositoryError("task_auth_context_changed") from exc
            row.status = status.value
            row.cancelled = status is TaskStatus.CANCELLED
            row.result_kind = result_kind
            row.result_ref = result_ref
            row.result_hash = result_hash
            row.last_error_code = error_code
            self._set_terminal_content_expiry(work.session, row, now=now)
            row.interrupt_kind = None
            row.interrupt_summary = None
            row.interrupt_ref = None
            row.interrupt_checkpoint_ns = None
            row.interrupt_checkpoint_id = None
            row.interrupt_id = None
            row.interrupt_approval_id = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
            self._sync_conversation_turn(
                work.session,
                row,
                now=now,
                terminal_message=conversation_message,
                terminal_evidence_ids=conversation_evidence_ids,
                terminal_artifacts=conversation_artifacts,
            )
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type=f"task_{status.value}",
                correlation_id=run_id,
                run_id=run_id,
                counts=event_counts,
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
            self._begin_resume_write(work.session)
            row = self._lock_resume_task(
                work.session,
                tenant_id=auth.tenant_id,
                task_id=task_id,
                skip_locked=False,
            )
            if row is None or row.user_id != auth.user_id:
                raise TaskRepositoryError("task_not_found")
            if row.state_version != expected_state_version:
                raise TaskRepositoryError("task_state_conflict")
            if TaskStatus(row.status) in _TERMINAL_STATUSES:
                raise TaskRepositoryError("task_terminal_state")
            now = self._lease_now(work.session)
            commands = self._lock_nonterminal_resume_commands_for_task(
                work.session,
                task=row,
            )
            if commands is None:
                raise TaskRepositoryError("task_resume_binding_invalid")
            for command in commands:
                self._terminalize_resume_command(
                    command,
                    status=(
                        ResumeCommandStatus.CONSUMED
                        if command.applied_checkpoint_id is not None
                        else ResumeCommandStatus.FAILED
                    ),
                    now=now,
                    failure_code=(
                        None if command.applied_checkpoint_id is not None else "task_cancelled"
                    ),
                )
            row.cancelled = True
            row.status = TaskStatus.CANCELLED.value
            self._set_terminal_content_expiry(work.session, row, now=now)
            row.interrupt_kind = None
            row.interrupt_summary = None
            row.interrupt_ref = None
            row.interrupt_checkpoint_ns = None
            row.interrupt_checkpoint_id = None
            row.interrupt_id = None
            row.interrupt_approval_id = None
            row.active_resume_command_id = None
            row.lease_owner = None
            row.lease_expires_at = None
            row.state_version += 1
            row.updated_at = now
            self._sync_conversation_turn(work.session, row, now=now)
            work.session.flush()
            self._append_event(
                work.session,
                row,
                event_type="task_cancelled",
                correlation_id=auth.request_id,
                run_id=None,
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
    def _sync_conversation_turn(
        session,
        row: AgentTaskRow,
        *,
        now: datetime,
        clarification_question: str | None = None,
        terminal_message: str | None = None,
        terminal_evidence_ids: tuple[str, ...] = (),
        terminal_artifacts: tuple[dict[str, object], ...] = (),
    ) -> None:
        """在任务事务内同步绑定轮次；旧任务没有绑定时保持原行为。"""
        if row.conversation_id is None or row.conversation_turn_id is None:
            return
        from uuid import uuid4

        from rag_kb.infrastructure.database.conversation_repository import request_hash
        from rag_kb.infrastructure.database.models import (
            ConversationMessageRow,
            ConversationRow,
            ConversationTurnRow,
        )

        turn = session.scalar(
            select(ConversationTurnRow)
            .where(
                ConversationTurnRow.tenant_id == row.tenant_id,
                ConversationTurnRow.conversation_id == row.conversation_id,
                ConversationTurnRow.turn_id == row.conversation_turn_id,
                ConversationTurnRow.task_id == row.task_id,
            )
            .with_for_update()
        )
        if turn is None:
            raise TaskRepositoryError("task_conversation_binding_invalid")
        from rag_kb.domain.conversations import turn_status_for_task_status

        turn.status = turn_status_for_task_status(row.status).value
        turn.task_state_version = row.state_version
        turn.updated_at = now
        conversation = session.scalar(
            select(ConversationRow)
            .where(
                ConversationRow.tenant_id == row.tenant_id,
                ConversationRow.conversation_id == row.conversation_id,
            )
            .with_for_update()
        )
        if conversation is None:
            raise TaskRepositoryError("task_conversation_binding_invalid")
        conversation.state_version += 1
        conversation.updated_at = now
        if clarification_question:
            existing = session.scalar(
                select(ConversationMessageRow.message_id).where(
                    ConversationMessageRow.tenant_id == row.tenant_id,
                    ConversationMessageRow.conversation_id == row.conversation_id,
                    ConversationMessageRow.turn_id == row.conversation_turn_id,
                    ConversationMessageRow.kind == "clarification_request",
                    ConversationMessageRow.content_hash
                    == request_hash({"content": clarification_question}),
                )
            )
            if existing is None:
                sequence = int(
                    session.scalar(
                        select(func.coalesce(func.max(ConversationMessageRow.sequence), 0)).where(
                            ConversationMessageRow.tenant_id == row.tenant_id,
                            ConversationMessageRow.conversation_id == row.conversation_id,
                        )
                    )
                    or 0
                ) + 1
                session.add(
                    ConversationMessageRow(
                        message_id=uuid4().hex,
                        tenant_id=row.tenant_id,
                        conversation_id=row.conversation_id,
                        turn_id=row.conversation_turn_id,
                        sequence=sequence,
                        role="assistant",
                        kind="clarification_request",
                        content=clarification_question,
                        content_hash=request_hash({"content": clarification_question}),
                        task_id=row.task_id,
                        evidence_ids=[],
                        artifacts=[],
                        created_at=now,
                    )
                )
        if terminal_message and row.status in {
            TaskStatus.SUCCEEDED.value,
            TaskStatus.REFUSED.value,
        }:
            existing = session.scalar(
                select(ConversationMessageRow.message_id).where(
                    ConversationMessageRow.tenant_id == row.tenant_id,
                    ConversationMessageRow.conversation_id == row.conversation_id,
                    ConversationMessageRow.turn_id == row.conversation_turn_id,
                    ConversationMessageRow.role == "assistant",
                    ConversationMessageRow.kind.in_(("answer", "refusal")),
                )
            )
            if existing is None:
                sequence = int(
                    session.scalar(
                        select(func.coalesce(func.max(ConversationMessageRow.sequence), 0)).where(
                            ConversationMessageRow.tenant_id == row.tenant_id,
                            ConversationMessageRow.conversation_id == row.conversation_id,
                        )
                    )
                    or 0
                ) + 1
                session.add(
                    ConversationMessageRow(
                        message_id=uuid4().hex,
                        tenant_id=row.tenant_id,
                        conversation_id=row.conversation_id,
                        turn_id=row.conversation_turn_id,
                        sequence=sequence,
                        role="assistant",
                        kind="answer" if row.status == TaskStatus.SUCCEEDED.value else "refusal",
                        content=terminal_message[:20_000],
                        content_hash=request_hash({"content": terminal_message[:20_000]}),
                        task_id=row.task_id,
                        evidence_ids=list(terminal_evidence_ids),
                        artifacts=list(terminal_artifacts),
                        created_at=now,
                    )
                )

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

    def _set_terminal_content_expiry(self, session, row: AgentTaskRow, *, now: datetime) -> None:
        if row.task_mode not in {
            TaskMode.RESEARCH.value,
            TaskMode.CONTROLLED_ACTION.value,
        }:
            return
        expires_at = now + timedelta(days=self._task_result_retention_days())
        row.content_expires_at = expires_at
        if row.task_mode != TaskMode.CONTROLLED_ACTION.value:
            return
        # 终态字段和租约会在调用方同一次 flush 提交，不能让此查询提前刷出。
        with session.no_autoflush:
            artifacts = tuple(
                session.scalars(
                    select(ArtifactRow)
                    .where(
                        ArtifactRow.tenant_id == row.tenant_id,
                        ArtifactRow.task_id == row.task_id,
                        ArtifactRow.content_purged_at.is_(None),
                    )
                    .with_for_update()
                )
            )
        for artifact in artifacts:
            artifact.content_expires_at = expires_at

    @staticmethod
    def _append_event(
        session,
        row: AgentTaskRow,
        *,
        event_type: str,
        correlation_id: str,
        run_id: str | None,
        counts: dict[str, int] | None = None,
    ) -> None:
        sequence = session.scalar(
            select(func.coalesce(func.max(TaskEventRow.sequence), -1) + 1).where(
                TaskEventRow.tenant_id == row.tenant_id,
                TaskEventRow.task_id == row.task_id,
            )
        )
        otel_trace_id, otel_span_id = current_span_ids()
        session.add(
            TaskEventRow(
                event_id=uuid4().hex,
                task_id=row.task_id,
                tenant_id=row.tenant_id,
                sequence=int(sequence or 0),
                event_type=event_type,
                correlation_id=normalize_request_id(correlation_id),
                otel_trace_id=otel_trace_id,
                otel_span_id=otel_span_id,
                run_id=run_id,
                stage=row.status,
                status=row.status,
                counts=dict(counts or {}),
                token_usage=None,
                error_code=row.last_error_code,
            )
        )
