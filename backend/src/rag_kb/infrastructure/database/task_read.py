"""任务事件、结果与产物正文读取的数据库适配器。"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import func, select, update

from rag_kb.application.ports.object_store import ObjectStorePort
from rag_kb.application.ports.task_read import (
    ArtifactContentError,
    ArtifactContentPort,
    ArtifactSummary,
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
from rag_kb.observability import current_observability


_EVIDENCE_AUTH_CODES = frozenset({"evidence_authorization_stale"})
logger = logging.getLogger(__name__)


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
        artifacts: ObjectStorePort | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._database = database
        self._results = results
        self._evidence = evidence or EvidenceLedgerRepository()
        self._artifacts = artifacts
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
        return result.model_copy(update={"artifacts": self._hydrate_artifacts(auth, state, result)})

    def _hydrate_artifacts(
        self,
        auth: AuthContext,
        state: TaskState,
        result: TaskResult,
    ) -> tuple[ArtifactSummary, ...]:
        if not result.artifacts:
            return ()
        artifact_ids = tuple(item.artifact_id for item in result.artifacts)
        with self._database.sessions() as session:
            rows = {
                row.artifact_id: row
                for row in session.scalars(
                    select(ArtifactRow).where(
                        ArtifactRow.tenant_id == auth.tenant_id,
                        ArtifactRow.task_id == state.task_id,
                        ArtifactRow.artifact_id.in_(artifact_ids),
                    )
                )
            }

        hydrated: list[ArtifactSummary] = []
        for summary in result.artifacts:
            row = rows.get(summary.artifact_id)
            artifact_type = row.artifact_type if row is not None else summary.artifact_type
            extension = self._artifact_extension(row.storage_key if row is not None else None, artifact_type)
            fallback = self._fallback_artifact_name(artifact_type, summary.artifact_id)
            display_name = (row.display_name or "").strip() if row is not None else ""
            if not display_name and row is not None:
                display_name = self._recover_display_name(row, fallback)
            hydrated.append(
                summary.model_copy(
                    update={
                        "artifact_type": artifact_type,
                        "display_name": display_name or fallback,
                        "file_extension": extension,
                    }
                )
            )
        return tuple(hydrated)

    def _recover_display_name(self, row: ArtifactRow, fallback: str) -> str:
        expires_at = _aware(row.content_expires_at)
        if (
            self._artifacts is None
            or not row.storage_key
            or row.status != "ready"
            or row.content_purged_at is not None
            or (expires_at is not None and expires_at <= self._result_now())
        ):
            return fallback
        try:
            content = self._artifacts.read(row.storage_key)
            if f"sha256:{hashlib.sha256(content).hexdigest()}" != row.content_hash:
                raise ValueError("artifact_content_hash_mismatch")
            display_name = self._parse_display_name(content, row.artifact_type) or fallback
            display_name = display_name.strip()[:300] or fallback
            with self._database.sessions.begin() as session:
                changed = session.execute(
                    update(ArtifactRow)
                    .where(
                        ArtifactRow.tenant_id == row.tenant_id,
                        ArtifactRow.task_id == row.task_id,
                        ArtifactRow.artifact_id == row.artifact_id,
                        ArtifactRow.display_name.is_(None),
                    )
                    .values(display_name=display_name)
                )
                if changed.rowcount == 0:
                    winner = session.scalar(
                        select(ArtifactRow.display_name).where(
                            ArtifactRow.tenant_id == row.tenant_id,
                            ArtifactRow.task_id == row.task_id,
                            ArtifactRow.artifact_id == row.artifact_id,
                        )
                    )
                    if isinstance(winner, str) and winner.strip():
                        display_name = winner.strip()
            return display_name
        except Exception:
            observability = current_observability()
            if observability is not None:
                observability.record_artifact_name_backfill_failure()
            logger.warning(
                "旧产物名称恢复失败 task_id=%s artifact_id=%s",
                row.task_id,
                row.artifact_id,
            )
            return fallback

    @staticmethod
    def _parse_display_name(content: bytes, artifact_type: str) -> str | None:
        text = content.decode("utf-8")
        if artifact_type == "report":
            for line in text.splitlines():
                if line.startswith("# "):
                    return line[2:].strip() or None
            return None
        document = json.loads(text)
        title = document.get("title") if isinstance(document, dict) else None
        return title.strip() if isinstance(title, str) and title.strip() else None

    @staticmethod
    def _artifact_extension(storage_key: str | None, artifact_type: str) -> str:
        if storage_key and storage_key.lower().endswith(".md"):
            return "md"
        return "json"

    @staticmethod
    def _fallback_artifact_name(artifact_type: str, artifact_id: str) -> str:
        label = "历史工作项草稿" if artifact_type == "work_item_draft" else "历史报告"
        return f"{label}-{artifact_id[:10]}"

    def _result_now(self) -> datetime:
        if self._database.engine.dialect.name != "postgresql":
            return self._now()
        with self._database.sessions() as session:
            value = session.scalar(select(func.clock_timestamp()))
        if not isinstance(value, datetime):
            raise RuntimeError("database_clock_unavailable")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
