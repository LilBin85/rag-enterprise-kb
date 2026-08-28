"""受控产物正文读取：授权、生命周期和 hash 在服务端逐项复核。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import func, select

from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.infrastructure.database.models import AgentTaskRow, ArtifactRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.object_store.local import LocalObjectStore


class ArtifactContentError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _aware(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


class ArtifactContentService:
    """只返回已授权且未过期的正文 bytes，不向上层暴露对象引用。"""

    def __init__(self, database: Database, objects: LocalObjectStore, *, now=None) -> None:
        self._database = database
        self._objects = objects
        self._now = now or (lambda: datetime.now(UTC))

    def read(self, *, auth: AuthContext, task_id: str, artifact_id: str) -> bytes:
        if not auth.has_permission(Permission.TASK_READ):
            raise ArtifactContentError("task_read_denied")
        storage_key, content_hash = self._load_readable_binding(
            auth=auth, task_id=task_id, artifact_id=artifact_id
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
        self, *, auth: AuthContext, task_id: str, artifact_id: str
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
                row, now=self._database_now(session)
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
