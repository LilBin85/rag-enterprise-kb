"""任务正文、checkpoint、产物与孤儿对象的周期收敛。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import logging
from pathlib import PurePosixPath

from rag_kb.application.ports.task_retention import (
    RetentionObjectStorePort,
    TaskResultDeletionPort,
    TaskRetentionLifecyclePort,
    TaskThreadPurgerPort,
)


logger = logging.getLogger(__name__)

_RESULT_PREFIXES = ("tool-results", "task-results", "artifacts")
_ORPHAN_RETRY_CAPACITY = 100


class TaskRetentionService:
    """保留策略只处理正文与 checkpoint，不改写执行结论。"""

    def __init__(
        self,
        *,
        lifecycle: TaskRetentionLifecyclePort,
        objects: RetentionObjectStorePort,
        results: TaskResultDeletionPort,
        checkpointer: TaskThreadPurgerPort | None = None,
        orphan_grace_seconds: int,
        now: Callable[[], datetime] | None = None,
        close: Callable[[], None] | None = None,
        conversation_purger: Callable[..., int] | None = None,
        conversation_expirer: Callable[..., int] | None = None,
    ) -> None:
        if orphan_grace_seconds <= 0:
            raise ValueError("invalid_result_object_orphan_grace")
        self._lifecycle = lifecycle
        self._objects = objects
        self._results = results
        self._checkpointer = checkpointer
        self._orphan_grace = timedelta(seconds=orphan_grace_seconds)
        self._now = now or (lambda: datetime.now(UTC))
        self._close = close
        self._conversation_purger = conversation_purger
        self._conversation_expirer = conversation_expirer
        self._orphan_prefix_index = 0
        self._orphan_cursors: dict[str, str | None] = {
            prefix: None for prefix in _RESULT_PREFIXES
        }
        self._orphan_retries: dict[str, list[str]] = {
            prefix: [] for prefix in _RESULT_PREFIXES
        }
        self._orphan_retry_turn: dict[str, bool] = {
            prefix: True for prefix in _RESULT_PREFIXES
        }
        self._orphan_retry_overflowed: dict[str, bool] = {
            prefix: False for prefix in _RESULT_PREFIXES
        }

    def close(self) -> None:
        close = self._close
        self._close = None
        if close is not None:
            close()

    def expire_waiting_tasks(self, *, limit: int = 100) -> int:
        return self._lifecycle.expire_waiting_tasks(limit=limit)

    def purge_conversation_content(self, *, limit: int = 100) -> int:
        self._validate_limit(limit)
        if self._conversation_purger is None:
            return 0
        return self._conversation_purger(limit=limit)

    def expire_conversations(self, *, limit: int = 100) -> int:
        self._validate_limit(limit)
        if self._conversation_expirer is None:
            return 0
        return self._conversation_expirer(limit=limit)

    def purge_expired_tool_results(self, *, limit: int = 100) -> int:
        """正文到期后只清引用；工具的 succeeded 审计结论保持不变。"""
        self._validate_limit(limit)
        purged = 0
        for candidate in self._lifecycle.expired_tool_results(limit=limit):
            try:
                self._objects.delete_exact(candidate.result_ref)
            except (OSError, ValueError):
                logger.error("工具结果对象删除失败 tool_call_id=%s", candidate.tool_call_id)
                continue
            if self._lifecycle.mark_tool_result_purged(candidate):
                purged += 1
        return purged

    def purge_finished_task_content(self, *, limit: int = 100) -> int:
        """删除过期终态 checkpoint、结果与产物正文，分段提交 marker 以支持重扫。"""
        self._validate_limit(limit)
        purged = 0
        for candidate in self._lifecycle.expired_task_content(limit=limit):
            checkpoint_deleted = False
            if candidate.checkpoint_pending:
                if self._checkpointer is None:
                    raise ValueError("task_thread_purger_required")
                try:
                    self._checkpointer.delete_thread(candidate.task_id)
                except Exception:
                    logger.error("任务 checkpoint 删除失败 task_id=%s", candidate.task_id)
                else:
                    checkpoint_deleted = True

            result_ready_to_mark = False
            if candidate.result_pending:
                if candidate.result_ref is None:
                    result_ready_to_mark = True
                else:
                    try:
                        self._results.delete_task_result(candidate.result_ref)
                    except (OSError, ValueError):
                        logger.error("任务结果对象删除失败 task_id=%s", candidate.task_id)
                    else:
                        result_ready_to_mark = True

            if checkpoint_deleted or result_ready_to_mark:
                if self._lifecycle.mark_task_content_purged(
                    candidate,
                    checkpoint_deleted=checkpoint_deleted,
                    result_ready_to_mark=result_ready_to_mark,
                ):
                    purged += 1
        return purged + self.purge_expired_artifact_content(limit=limit)

    def purge_expired_artifact_content(self, *, limit: int = 100) -> int:
        """产物正文删除成功后才清内部引用并写 purge marker。"""
        self._validate_limit(limit)
        purged = 0
        for candidate in self._lifecycle.expired_artifact_content(limit=limit):
            try:
                self._objects.delete_exact(candidate.storage_key)
            except (OSError, ValueError):
                logger.error("产物正文删除失败 artifact_id=%s", candidate.artifact_id)
                continue
            if self._lifecycle.mark_artifact_content_purged(candidate):
                purged += 1
        return purged

    def purge_orphaned_result_objects(self, *, limit: int = 100) -> int:
        """清理未绑定、已过 grace 的服务端结果对象。"""
        self._validate_limit(limit)
        cutoff = self._now() - self._orphan_grace
        exclude = self._lifecycle.orphan_scan_exclusions()
        refs: list[str] = []
        prefixes = _RESULT_PREFIXES[self._orphan_prefix_index :] + _RESULT_PREFIXES[
            : self._orphan_prefix_index
        ]
        self._orphan_prefix_index = (self._orphan_prefix_index + 1) % len(
            _RESULT_PREFIXES
        )
        base_quota, remainder = divmod(limit, len(prefixes))
        for index, prefix in enumerate(prefixes):
            quota = base_quota + (1 if index < remainder else 0)
            if quota == 0:
                continue
            retry_limit = quota - 1 if quota > 1 else 0
            if quota == 1 and self._orphan_retries[prefix]:
                retry_limit = 1 if self._orphan_retry_turn[prefix] else 0
                self._orphan_retry_turn[prefix] = not self._orphan_retry_turn[prefix]
            retries = tuple(self._orphan_retries[prefix][:retry_limit])
            refs.extend(retries)
            remaining = quota - len(retries)
            if remaining:
                # 已入队失败项由重试通道单独跟踪；继续占枚举名额会让
                # limit 很小时的同前缀后续孤儿永久饿死。
                scanned, exhausted, resume = self._objects.iter_ref_entries(
                    prefix,
                    older_than=cutoff,
                    limit=remaining,
                    after=self._orphan_cursors[prefix],
                    exclude=exclude | frozenset(self._orphan_retries[prefix]),
                )
                # 位置游标：截断即前进（无论本轮是否命中），扫尽一律重置开新 epoch。
                if resume is not None:
                    self._orphan_cursors[prefix] = resume
                elif exhausted:
                    self._orphan_cursors[prefix] = None
                queued = set(self._orphan_retries[prefix])
                refs.extend(
                    ref for _mtime, ref in scanned if ref not in queued
                )
        if not refs:
            return 0

        # 先确认在途执行：若它随后合法绑定正文，仍由保护集合阻止删除。
        snapshot = self._lifecycle.orphan_reference_snapshot(tuple(refs))

        purged = 0
        for ref in refs:
            if ref in snapshot.referenced_refs or ref in snapshot.protected_refs:
                self._discard_orphan_retry(ref)
                continue
            try:
                self._objects.delete_exact(ref)
            except (OSError, ValueError):
                logger.error(
                    "孤儿对象删除失败 prefix=%s", PurePosixPath(ref).parts[0]
                )
                self._queue_orphan_retry(ref)
                continue
            self._discard_orphan_retry(ref)
            purged += 1
        return purged

    def _queue_orphan_retry(self, ref: str) -> None:
        prefix = PurePosixPath(ref).parts[0]
        retries = self._orphan_retries[prefix]
        if ref in retries:
            retries.remove(ref)
            retries.append(ref)
            return
        if len(retries) >= _ORPHAN_RETRY_CAPACITY:
            # 队列耗尽后再回卷补扫，不能让持续失败项挡住前向扫描。
            self._orphan_retry_overflowed[prefix] = True
            logger.error("孤儿结果对象重试队列已满 prefix=%s", prefix)
            return
        retries.append(ref)

    def _discard_orphan_retry(self, ref: str) -> None:
        prefix = PurePosixPath(ref).parts[0]
        retries = self._orphan_retries[prefix]
        if ref in retries:
            retries.remove(ref)
            if not retries and self._orphan_retry_overflowed[prefix]:
                self._orphan_cursors[prefix] = None
                self._orphan_retry_overflowed[prefix] = False

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
