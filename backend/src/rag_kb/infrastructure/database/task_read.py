"""任务事件、结果与产物正文读取的数据库适配器。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import func, select

from rag_kb.application.ports.object_store import ObjectStorePort
from rag_kb.application.ports.task_read import (
    ArtifactContentError,
    ArtifactContentPort,
    TaskArtifactResult,
    TaskEventResult,
    TaskQueryPort,
    TaskResult,
    TaskResultError,
    TaskResultObjectStorePort,
    TaskResultReaderPort,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskState, TaskStatus
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import AgentTaskRow, ArtifactRow, TaskEventRow
from rag_kb.infrastructure.database.session import Database


_EVIDENCE_AUTH_CODES = frozenset({"evidence_authorization_stale"})


def _aware(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


class DatabaseTaskQueryReader(TaskQueryPort):
    def __init__(self, database: Database) -> None:
        self._database = database

    def events(
        self,
        *,
        auth: AuthContext,
        task_id: str,
    ) -> tuple[TaskEventResult, ...] | None:
        with self._database.sessions() as session:
            if not self._owns_task(session, auth=auth, task_id=task_id):
                return None
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

    def artifacts(
        self,
        *,
        auth: AuthContext,
        task_id: str,
    ) -> tuple[TaskArtifactResult, ...] | None:
        with self._database.sessions() as session:
            if not self._owns_task(session, auth=auth, task_id=task_id):
                return None
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
    def _owns_task(session, *, auth: AuthContext, task_id: str) -> bool:
        return (
            session.scalar(
                select(AgentTaskRow.task_id).where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.user_id == auth.user_id,
                )
            )
            is not None
        )


class DatabaseArtifactContentReader(ArtifactContentPort):
    def __init__(
        self,
        database: Database,
        objects: ObjectStorePort,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._objects = objects
        self._now = now or (lambda: datetime.now(UTC))

    def read(self, *, auth: AuthContext, task_id: str, artifact_id: str) -> bytes:
        storage_key, content_hash = self._load_readable_binding(
            auth=auth,
            task_id=task_id,
            artifact_id=artifact_id,
        )
        try:
            content = self._objects.read(storage_key)
        except (OSError, ValueError):
            raise ArtifactContentError("artifact_content_integrity_failed") from None
        if f"sha256:{hashlib.sha256(content).hexdigest()}" != content_hash:
            raise ArtifactContentError("artifact_content_integrity_failed")
        self._verify_unchanged_binding(
            auth=auth,
            task_id=task_id,
            artifact_id=artifact_id,
            storage_key=storage_key,
            content_hash=content_hash,
        )
        return content

    def _load_readable_binding(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        artifact_id: str,
    ) -> tuple[str, str]:
        with self._database.sessions() as session:
            self._require_owner(session, auth=auth, task_id=task_id)
            row = session.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.tenant_id == auth.tenant_id,
                    ArtifactRow.task_id == task_id,
                    ArtifactRow.artifact_id == artifact_id,
                )
            )
            if row is None:
                raise ArtifactContentError("artifact_not_found")
            return self._readable_values(row, now=self._database_now(session))

    def _verify_unchanged_binding(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        artifact_id: str,
        storage_key: str,
        content_hash: str,
    ) -> None:
        with self._database.sessions() as session:
            self._require_owner(session, auth=auth, task_id=task_id)
            row = session.scalar(
                select(ArtifactRow)
                .where(
                    ArtifactRow.tenant_id == auth.tenant_id,
                    ArtifactRow.task_id == task_id,
                    ArtifactRow.artifact_id == artifact_id,
                )
                .with_for_update()
            )
            if row is None:
                raise ArtifactContentError("artifact_not_found")
            current_key, current_hash = self._readable_values(
                row,
                now=self._database_now(session),
            )
            if current_key != storage_key or current_hash != content_hash:
                raise ArtifactContentError("artifact_content_integrity_failed")

    @staticmethod
    def _require_owner(session, *, auth: AuthContext, task_id: str) -> None:
        task = session.scalar(
            select(AgentTaskRow).where(
                AgentTaskRow.tenant_id == auth.tenant_id,
                AgentTaskRow.task_id == task_id,
                AgentTaskRow.user_id == auth.user_id,
            )
        )
        if task is None:
            raise ArtifactContentError("artifact_not_found")

    @staticmethod
    def _readable_values(row: ArtifactRow, *, now: datetime) -> tuple[str, str]:
        if row.status != "ready":
            raise ArtifactContentError("artifact_content_unavailable")
        expires_at = _aware(row.content_expires_at)
        if row.content_purged_at is not None or (
            expires_at is not None and expires_at <= now
        ):
            raise ArtifactContentError("artifact_content_expired")
        if (
            not isinstance(row.storage_key, str)
            or not row.storage_key
            or not isinstance(row.content_hash, str)
            or not row.content_hash
        ):
            raise ArtifactContentError("artifact_content_integrity_failed")
        return row.storage_key, row.content_hash

    def _database_now(self, session) -> datetime:
        if session.get_bind().dialect.name != "postgresql":
            return self._now()
        value = session.scalar(select(func.clock_timestamp()))
        if not isinstance(value, datetime):
            raise ArtifactContentError("artifact_content_integrity_failed")
        return _aware(value) or value


class DatabaseTaskResultReader(TaskResultReaderPort):
    def __init__(
        self,
        database: Database,
        results: TaskResultObjectStorePort,
        evidence: EvidenceLedgerRepository | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._results = results
        self._evidence = evidence or EvidenceLedgerRepository()
        self._now = now or (lambda: datetime.now(UTC))

    def read(self, *, auth: AuthContext, state: TaskState) -> TaskResult:
        expected_kind = (
            "controlled_action"
            if state.task_mode.value == "controlled_action"
            else "research"
        )
        if (
            state.result_kind != expected_kind
            or state.result_hash is None
            or state.status not in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED}
        ):
            raise TaskResultError("task_result_unavailable")
        now = self._result_now()
        if state.result_purged_at is not None or (
            state.content_expires_at is not None and state.content_expires_at <= now
        ):
            raise TaskResultError("task_result_expired")
        if state.result_ref is None:
            raise TaskResultError("task_result_unavailable")
        try:
            result = self._results.read_task_result(state.result_ref, state.result_hash)
        except (OSError, ValueError) as exc:
            raise TaskResultError("task_result_integrity_failed") from exc
        expected_status = "succeeded" if state.status is TaskStatus.SUCCEEDED else "refused"
        if (
            result.task_id != state.task_id
            or result.status != expected_status
            or result.result_kind != expected_kind
        ):
            raise TaskResultError("task_result_binding_invalid")
        if result.status == "refused":
            if result.evidence_ids or result.citations:
                raise TaskResultError("task_result_binding_invalid")
        else:
            if not result.evidence_ids:
                raise TaskResultError("task_result_binding_invalid")
            try:
                with self._database.sessions() as session:
                    citations = self._evidence.citations_from_ledger(
                        session,
                        task_id=state.task_id,
                        evidence_ids=result.evidence_ids,
                        auth=auth,
                    )
            except EvidenceLedgerConflict as exc:
                code = exc.args[0] if exc.args else ""
                if code in _EVIDENCE_AUTH_CODES:
                    raise TaskResultError("task_result_evidence_unavailable") from exc
                raise TaskResultError("task_result_binding_invalid") from exc
            if citations != result.citations:
                raise TaskResultError("task_result_binding_invalid")
        with self._database.sessions() as session:
            row = session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == state.task_id,
                    AgentTaskRow.user_id == auth.user_id,
                )
                .with_for_update()
            )
            now = self._result_now()
            if row is None or TaskStatus(row.status) not in {
                TaskStatus.SUCCEEDED,
                TaskStatus.REFUSED,
            }:
                raise TaskResultError("task_result_unavailable")
            expires = _aware(row.content_expires_at)
            if row.result_purged_at is not None or (
                expires is not None and expires <= now
            ):
                raise TaskResultError("task_result_expired")
            if row.result_ref != state.result_ref or row.result_hash != state.result_hash:
                raise TaskResultError("task_result_binding_invalid")
        return result

    def _result_now(self) -> datetime:
        if self._database.engine.dialect.name != "postgresql":
            return self._now()
        with self._database.sessions() as session:
            value = session.scalar(select(func.clock_timestamp()))
        if not isinstance(value, datetime):
            raise RuntimeError("database_clock_unavailable")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
