"""任务事件与产物的租户/所有者隔离只读视图。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.infrastructure.database.models import AgentTaskRow, ArtifactRow, TaskEventRow
from rag_kb.infrastructure.database.session import Database


class TaskQueryError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TaskEventResult:
    sequence: int
    event_type: str
    stage: str
    status: str
    counts: dict[str, object]
    token_usage: int
    error_code: str | None
    created_at: datetime


@dataclass(frozen=True)
class TaskArtifactResult:
    artifact_id: str
    artifact_type: str
    content_hash: str
    status: str
    created_at: datetime


class DatabaseTaskQueryService:
    def __init__(self, database: Database) -> None:
        self._database = database

    def events(self, *, auth: AuthContext, task_id: str) -> tuple[TaskEventResult, ...]:
        self._require_read(auth)
        with self._database.sessions() as session:
            self._require_owner(session, auth, task_id)
            rows = tuple(
                session.scalars(
                    select(TaskEventRow)
                    .where(
                        TaskEventRow.tenant_id == auth.tenant_id,
                        TaskEventRow.task_id == task_id,
                    )
                    .order_by(TaskEventRow.sequence)
                )
            )
        return tuple(
            TaskEventResult(
                sequence=row.sequence,
                event_type=row.event_type,
                stage=row.stage,
                status=row.status,
                counts=row.counts,
                token_usage=row.token_usage,
                error_code=row.error_code,
                created_at=row.created_at,
            )
            for row in rows
        )

    def artifacts(self, *, auth: AuthContext, task_id: str) -> tuple[TaskArtifactResult, ...]:
        self._require_read(auth)
        with self._database.sessions() as session:
            self._require_owner(session, auth, task_id)
            rows = tuple(
                session.scalars(
                    select(ArtifactRow)
                    .where(
                        ArtifactRow.tenant_id == auth.tenant_id,
                        ArtifactRow.task_id == task_id,
                    )
                    .order_by(ArtifactRow.created_at, ArtifactRow.artifact_id)
                )
            )
        return tuple(
            TaskArtifactResult(
                artifact_id=row.artifact_id,
                artifact_type=row.artifact_type,
                content_hash=row.content_hash,
                status=row.status,
                created_at=row.created_at,
            )
            for row in rows
        )

    @staticmethod
    def _require_read(auth: AuthContext) -> None:
        if not auth.has_permission(Permission.TASK_READ):
            raise TaskQueryError("task_read_denied")

    @staticmethod
    def _require_owner(session, auth: AuthContext, task_id: str) -> AgentTaskRow:
        task = session.scalar(
            select(AgentTaskRow).where(
                AgentTaskRow.tenant_id == auth.tenant_id,
                AgentTaskRow.task_id == task_id,
                AgentTaskRow.user_id == auth.user_id,
            )
        )
        if task is None:
            raise TaskQueryError("task_not_found")
        return task
