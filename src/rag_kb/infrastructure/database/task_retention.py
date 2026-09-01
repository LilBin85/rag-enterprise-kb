"""任务保留策略的数据库生命周期适配器。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Callable

from sqlalchemy import func, or_, select

from rag_kb.application.ports.task_retention import (
    ArtifactContentCandidate,
    ExpiredToolResult,
    OrphanReferenceSnapshot,
    TaskContentCandidate,
    TaskRetentionLifecyclePort,
)
from rag_kb.domain.tasks import TaskStatus
from rag_kb.infrastructure.database.models import AgentTaskRow, ArtifactRow, ToolCallRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


_TERMINAL_STATUS_VALUES = tuple(
    status.value
    for status in (
        TaskStatus.SUCCEEDED,
        TaskStatus.REFUSED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.UNKNOWN,
    )
)


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


class DatabaseTaskRetentionLifecycle(TaskRetentionLifecyclePort):
    def __init__(
        self,
        database: Database,
        repository: DatabaseTaskRepository,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._repository = repository
        self._now = now or (lambda: datetime.now(UTC))

    def expire_waiting_tasks(self, *, limit: int) -> int:
        return self._repository.expire_waiting_tasks(limit=limit)

    def expired_tool_results(self, *, limit: int) -> tuple[ExpiredToolResult, ...]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._clock_expression(work.session)
            rows = tuple(
                work.session.scalars(
                    select(ToolCallRow)
                    .where(
                        ToolCallRow.status == "succeeded",
                        ToolCallRow.result_ref.is_not(None),
                        ToolCallRow.result_expires_at <= clock,
                        ToolCallRow.result_purged_at.is_(None),
                    )
                    .order_by(ToolCallRow.result_expires_at, ToolCallRow.tool_call_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            return tuple(
                ExpiredToolResult(tool_call_id=row.tool_call_id, result_ref=row.result_ref)
                for row in rows
                if row.result_ref is not None
            )

    def mark_tool_result_purged(self, candidate: ExpiredToolResult) -> bool:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(ToolCallRow)
                .where(ToolCallRow.tool_call_id == candidate.tool_call_id)
                .with_for_update()
            )
            if (
                row is None
                or row.status != "succeeded"
                or row.result_ref != candidate.result_ref
                or row.result_purged_at is not None
            ):
                return False
            row.result_ref = None
            row.result_size = None
            row.result_purged_at = self._database_now(work.session)
            return True

    def expired_task_content(self, *, limit: int) -> tuple[TaskContentCandidate, ...]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._clock_expression(work.session)
            rows = tuple(
                work.session.scalars(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.status.in_(_TERMINAL_STATUS_VALUES),
                        AgentTaskRow.content_expires_at.is_not(None),
                        AgentTaskRow.content_expires_at <= clock,
                        or_(
                            AgentTaskRow.checkpoint_purged_at.is_(None),
                            AgentTaskRow.result_purged_at.is_(None),
                        ),
                    )
                    .order_by(AgentTaskRow.content_expires_at, AgentTaskRow.task_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            return tuple(
                TaskContentCandidate(
                    task_id=row.task_id,
                    checkpoint_pending=row.checkpoint_purged_at is None,
                    result_pending=row.result_purged_at is None,
                    result_ref=row.result_ref,
                )
                for row in rows
            )

    def mark_task_content_purged(
        self,
        candidate: TaskContentCandidate,
        *,
        checkpoint_deleted: bool,
        result_ready_to_mark: bool,
    ) -> bool:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(AgentTaskRow)
                .where(AgentTaskRow.task_id == candidate.task_id)
                .with_for_update()
            )
            if row is None:
                return False
            now = self._database_now(work.session)
            if not self._is_expired_terminal_task(row, now):
                return False

            changed = False
            if checkpoint_deleted and row.checkpoint_purged_at is None:
                row.checkpoint_purged_at = now
                changed = True
            if result_ready_to_mark and row.result_purged_at is None:
                if candidate.result_ref is None:
                    if row.result_ref is None:
                        row.result_purged_at = now
                        changed = True
                elif row.result_ref == candidate.result_ref:
                    row.result_ref = None
                    row.result_purged_at = now
                    changed = True
            return changed

    def expired_artifact_content(
        self,
        *,
        limit: int,
    ) -> tuple[ArtifactContentCandidate, ...]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            clock = self._clock_expression(work.session)
            rows = tuple(
                work.session.scalars(
                    select(ArtifactRow)
                    .where(
                        ArtifactRow.storage_key.is_not(None),
                        ArtifactRow.content_expires_at.is_not(None),
                        ArtifactRow.content_expires_at <= clock,
                        ArtifactRow.content_purged_at.is_(None),
                    )
                    .order_by(ArtifactRow.content_expires_at, ArtifactRow.artifact_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            return tuple(
                ArtifactContentCandidate(
                    artifact_id=row.artifact_id,
                    storage_key=row.storage_key,
                )
                for row in rows
                if isinstance(row.storage_key, str)
            )

    def mark_artifact_content_purged(self, candidate: ArtifactContentCandidate) -> bool:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(ArtifactRow)
                .where(ArtifactRow.artifact_id == candidate.artifact_id)
                .with_for_update()
            )
            if row is None or row.content_purged_at is not None:
                return False
            now = self._database_now(work.session)
            expires_at = _aware(row.content_expires_at)
            if (
                row.storage_key != candidate.storage_key
                or expires_at is None
                or expires_at > now
            ):
                return False
            row.storage_key = None
            row.content_purged_at = now
            return True

    def orphan_scan_exclusions(self) -> frozenset[str]:
        with self._database.sessions() as session:
            return frozenset(self._referenced_refs(session) | self._running_tool_refs(session))

    def orphan_reference_snapshot(
        self,
        refs: tuple[str, ...],
    ) -> OrphanReferenceSnapshot:
        with self._database.sessions() as session:
            protected = self._running_tool_result_refs(
                session,
                refs,
            ) | self._running_task_result_refs(session, refs)
            return OrphanReferenceSnapshot(
                referenced_refs=frozenset(self._referenced_refs(session)),
                protected_refs=frozenset(protected),
            )

    def _referenced_refs(self, session) -> set[str]:
        return {
            *session.scalars(
                select(ToolCallRow.result_ref).where(ToolCallRow.result_ref.is_not(None))
            ),
            *session.scalars(
                select(AgentTaskRow.result_ref).where(AgentTaskRow.result_ref.is_not(None))
            ),
            *session.scalars(
                select(ArtifactRow.storage_key).where(ArtifactRow.storage_key.is_not(None))
            ),
        }

    def _running_tool_refs(self, session) -> set[str]:
        clock = self._clock_expression(session)
        return {
            PurePosixPath("tool-results", task_id, f"{tool_call_id}.json").as_posix()
            for task_id, tool_call_id in session.execute(
                select(ToolCallRow.task_id, ToolCallRow.tool_call_id).where(
                    ToolCallRow.status == "running",
                    ToolCallRow.execution_deadline > clock,
                )
            )
        }

    def _running_tool_result_refs(self, session, refs: tuple[str, ...]) -> set[str]:
        return set(refs) & self._running_tool_refs(session)

    def _running_task_result_refs(self, session, refs: tuple[str, ...]) -> set[str]:
        task_refs = {
            ref: path.parts[1]
            for ref in refs
            if (path := PurePosixPath(ref)).parts[:1] == ("task-results",)
            and len(path.parts) == 3
            and path.name.endswith(".json")
        }
        if not task_refs:
            return set()
        task_ids = set(task_refs.values())
        running_task_ids = set(
            session.scalars(
                select(AgentTaskRow.task_id).where(
                    AgentTaskRow.task_id.in_(task_ids),
                    AgentTaskRow.status.in_(
                        (TaskStatus.PLANNING.value, TaskStatus.RUNNING.value)
                    ),
                    or_(
                        AgentTaskRow.deadline.is_(None),
                        AgentTaskRow.deadline > self._clock_expression(session),
                    ),
                )
            )
        )
        return {
            ref for ref, task_id in task_refs.items() if task_id in running_task_ids
        }

    def _database_now(self, session) -> datetime:
        if session.get_bind().dialect.name != "postgresql":
            return self._now()
        value = session.scalar(select(func.clock_timestamp()))
        if not isinstance(value, datetime):
            raise RuntimeError("database_clock_unavailable")
        return _aware(value) or value

    def _clock_expression(self, session):
        if session.get_bind().dialect.name == "postgresql":
            return func.clock_timestamp()
        return self._now()

    @staticmethod
    def _is_expired_terminal_task(row: AgentTaskRow, now: datetime) -> bool:
        expires_at = _aware(row.content_expires_at)
        return (
            row.status in _TERMINAL_STATUS_VALUES
            and expires_at is not None
            and expires_at <= now
        )
