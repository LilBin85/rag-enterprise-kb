"""任务正文、产物与孤儿对象保留策略的稳定契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class ExpiredToolResult:
    tool_call_id: str
    result_ref: str


@dataclass(frozen=True)
class TaskContentCandidate:
    task_id: str
    checkpoint_pending: bool
    result_pending: bool
    result_ref: str | None


@dataclass(frozen=True)
class ArtifactContentCandidate:
    artifact_id: str
    storage_key: str


@dataclass(frozen=True)
class OrphanReferenceSnapshot:
    referenced_refs: frozenset[str]
    protected_refs: frozenset[str]


class RetentionObjectStorePort(Protocol):
    def delete_exact(self, storage_key: str) -> None: ...

    def iter_ref_entries(
        self,
        prefix: str,
        *,
        older_than: datetime,
        limit: int,
        after: str | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> tuple[tuple[tuple[float, str], ...], bool, str | None]: ...


class TaskResultDeletionPort(Protocol):
    def delete_task_result(self, ref: str) -> None: ...


class TaskThreadPurgerPort(Protocol):
    def delete_thread(self, task_id: str) -> None: ...


class TaskRetentionLifecyclePort(Protocol):
    def expire_waiting_tasks(self, *, limit: int) -> int: ...

    def expired_tool_results(self, *, limit: int) -> tuple[ExpiredToolResult, ...]: ...

    def mark_tool_result_purged(self, candidate: ExpiredToolResult) -> bool: ...

    def expired_task_content(self, *, limit: int) -> tuple[TaskContentCandidate, ...]: ...

    def mark_task_content_purged(
        self,
        candidate: TaskContentCandidate,
        *,
        checkpoint_deleted: bool,
        result_ready_to_mark: bool,
    ) -> bool: ...

    def expired_artifact_content(
        self,
        *,
        limit: int,
    ) -> tuple[ArtifactContentCandidate, ...]: ...

    def mark_artifact_content_purged(self, candidate: ArtifactContentCandidate) -> bool: ...

    def orphan_scan_exclusions(self) -> frozenset[str]: ...

    def orphan_reference_snapshot(
        self,
        refs: tuple[str, ...],
    ) -> OrphanReferenceSnapshot: ...
