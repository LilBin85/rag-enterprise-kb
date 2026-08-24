"""只在服务端对象根目录创建不可覆盖的本地产物。"""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from sqlalchemy import select, update

from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.infrastructure.database.models import ArtifactRow, ToolCallRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore, ObjectConflictError


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class LocalArtifactToolAdapter:
    def __init__(self, database: Database, store: LocalObjectStore, *, artifact_type: str) -> None:
        if artifact_type not in {"report", "work_item_draft"}:
            raise ValueError("unsupported_artifact_type")
        self._database = database
        self._store = store
        self._artifact_type = artifact_type

    def execute(
        self, arguments: dict[str, object], context: ToolExecutionContext
    ) -> dict[str, object]:
        if any(name in arguments for name in ("path", "storage_key", "url")):
            raise ValueError("artifact_path_forbidden")
        content = self._serialize(arguments)
        content_hash = _sha256(content)
        tenant_part = hashlib.sha256(context.auth.tenant_id.encode()).hexdigest()[:24]
        task_part = hashlib.sha256(context.task_id.encode()).hexdigest()[:24]
        key_part = hashlib.sha256((context.idempotency_key or context.tool_call_id).encode()).hexdigest()
        extension = "md" if arguments.get("format") == "markdown" else "json"
        storage_key = f"artifacts/{tenant_part}/{task_part}/{self._artifact_type}/{key_part}.{extension}"
        artifact_id = uuid4().hex
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            work.session.add(
                ArtifactRow(
                    artifact_id=artifact_id,
                    task_id=context.task_id,
                    tenant_id=context.auth.tenant_id,
                    tool_call_id=context.tool_call_id,
                    artifact_type=self._artifact_type,
                    storage_key=storage_key,
                    content_hash=content_hash,
                    status="creating",
                )
            )
        try:
            self._store.put_exclusive(storage_key, content)
        except ObjectConflictError:
            # 仅接受同一内容的恢复，不允许覆盖或用新内容冒充成功。
            if _sha256(self._store.read(storage_key)) != content_hash:
                self._mark(artifact_id, "unknown")
                raise RuntimeError("artifact_exists")
        if _sha256(self._store.read(storage_key)) != content_hash:
            self._mark(artifact_id, "unknown")
            raise RuntimeError("artifact_readback_mismatch")
        self._mark(artifact_id, "ready")
        return {
            "artifact_type": self._artifact_type,
            "storage_key": storage_key,
            "content_hash": content_hash,
            "status": "ready",
        }

    def verify(self, result: dict[str, object], context: ToolExecutionContext) -> bool:
        storage_key = result.get("storage_key")
        content_hash = result.get("content_hash")
        if not isinstance(storage_key, str) or not isinstance(content_hash, str):
            return False
        try:
            if _sha256(self._store.read(storage_key)) != content_hash:
                return False
        except (OSError, ValueError):
            return False
        with self._database.sessions() as session:
            row = session.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.tenant_id == context.auth.tenant_id,
                    ArtifactRow.task_id == context.task_id,
                    ArtifactRow.tool_call_id == context.tool_call_id,
                    ArtifactRow.storage_key == storage_key,
                    ArtifactRow.content_hash == content_hash,
                    ArtifactRow.status == "ready",
                )
            )
            return row is not None

    def _mark(self, artifact_id: str, status: str) -> None:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            artifact = work.session.scalar(
                select(ArtifactRow).where(ArtifactRow.artifact_id == artifact_id)
            )
            if artifact is None:
                raise RuntimeError("artifact_state_missing")
            call_status = work.session.scalar(
                select(ToolCallRow.status)
                .where(ToolCallRow.tool_call_id == artifact.tool_call_id)
                .with_for_update()
            )
            if status == "ready" and call_status != "running":
                work.session.execute(
                    update(ArtifactRow)
                    .where(
                        ArtifactRow.artifact_id == artifact_id,
                        ArtifactRow.status.in_(("creating", "ready")),
                    )
                    .values(status="unknown")
                )
                return
            work.session.execute(
                update(ArtifactRow)
                .where(
                    ArtifactRow.artifact_id == artifact_id,
                    ArtifactRow.status == "creating",
                )
                .values(status=status)
            )

    def _serialize(self, arguments: dict[str, object]) -> bytes:
        if self._artifact_type == "report" and arguments.get("format") == "markdown":
            return f"# {arguments['title']}\n\n{arguments['content']}\n".encode("utf-8")
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
