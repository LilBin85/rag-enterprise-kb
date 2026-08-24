"""可恢复任务动作的不可变对象载荷。

动作参数可能含业务正文，因此关系数据库只保存对象引用、摘要、类型和过期时间。
对象必须先完整发布，任务 checkpoint 才能引用它。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.domain.tasks import (
    PendingActionBinding,
    PersistableAction,
)
from rag_kb.infrastructure.object_store.local import (
    LocalObjectStore,
    ObjectConflictError,
)


class PendingActionStoreError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class PendingActionEnvelope(BaseModel):
    """对象中的版本化规范载荷；不直接暴露为 HTTP 契约。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    task_id: str = Field(min_length=1, max_length=64)
    auth_context_ref: str = Field(min_length=1, max_length=500)
    plan_revision: int = Field(ge=0)
    source_state_version: int = Field(ge=1)
    action: PersistableAction
    expires_at: AwareDatetime


class PendingActionStore:
    def __init__(self, objects: LocalObjectStore, *, now=None) -> None:
        self._objects = objects
        self._now = now or (lambda: datetime.now(UTC))

    def stage(
        self,
        *,
        task_id: str,
        auth_context_ref: str,
        plan_revision: int,
        source_state_version: int,
        action: PersistableAction,
        expires_at: datetime,
    ) -> PendingActionBinding:
        """独占发布动作对象；同一规范载荷的重复发布可安全复用。"""

        envelope = PendingActionEnvelope(
            task_id=task_id,
            auth_context_ref=auth_context_ref,
            plan_revision=plan_revision,
            source_state_version=source_state_version,
            action=action,
            expires_at=expires_at,
        )
        if envelope.expires_at <= self._now():
            raise PendingActionStoreError("pending_action_expiry_invalid")
        payload = envelope.model_dump(mode="json")
        content = self._canonical_bytes(payload)
        content_hash = canonical_fingerprint("pending_action_v1", payload)
        storage_ref = self._storage_ref(
            task_id=envelope.task_id,
            source_state_version=envelope.source_state_version,
            content_hash=content_hash,
        )
        try:
            self._objects.put_exclusive(storage_ref, content)
        except ObjectConflictError:
            try:
                existing = self._objects.read(storage_ref)
            except OSError as exc:
                raise PendingActionStoreError("pending_action_object_unavailable") from exc
            if existing != content:
                raise PendingActionStoreError("pending_action_object_conflict")
        return PendingActionBinding(
            storage_ref=storage_ref,
            content_hash=content_hash,
            kind=envelope.action.kind,
            expires_at=envelope.expires_at,
        )

    def load(
        self,
        binding: PendingActionBinding,
        *,
        task_id: str,
        auth_context_ref: str,
        plan_revision: int,
    ) -> PersistableAction:
        """回读并逐项验证引用、规范编码、摘要与任务绑定。"""

        if binding.expires_at <= self._now():
            raise PendingActionStoreError("pending_action_expired")
        try:
            content = self._objects.read(binding.storage_ref)
        except OSError as exc:
            raise PendingActionStoreError("pending_action_object_unavailable") from exc
        try:
            raw = json.loads(content)
            envelope = PendingActionEnvelope.model_validate(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, TypeError) as exc:
            raise PendingActionStoreError("pending_action_payload_invalid") from exc
        payload = envelope.model_dump(mode="json")
        if content != self._canonical_bytes(payload):
            raise PendingActionStoreError("pending_action_payload_noncanonical")
        actual_hash = canonical_fingerprint("pending_action_v1", payload)
        if actual_hash != binding.content_hash:
            raise PendingActionStoreError("pending_action_hash_mismatch")
        expected_ref = self._storage_ref(
            task_id=envelope.task_id,
            source_state_version=envelope.source_state_version,
            content_hash=actual_hash,
        )
        if binding.storage_ref != expected_ref:
            raise PendingActionStoreError("pending_action_reference_invalid")
        if (
            envelope.task_id != task_id
            or envelope.auth_context_ref != auth_context_ref
            or envelope.plan_revision != plan_revision
            or envelope.action.kind != binding.kind
            or envelope.expires_at != binding.expires_at
        ):
            raise PendingActionStoreError("pending_action_binding_invalid")
        return envelope.action

    @staticmethod
    def _canonical_bytes(value: dict[str, object]) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _storage_ref(
        *, task_id: str, source_state_version: int, content_hash: str
    ) -> str:
        task_shard = canonical_fingerprint("pending_action_task", task_id)[7:31]
        digest = content_hash.removeprefix("sha256:")
        return f"task-actions/{task_shard}/{source_state_version}-{digest}.json"

