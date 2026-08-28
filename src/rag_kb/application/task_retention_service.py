"""任务正文、checkpoint、产物与孤儿对象的周期收敛。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from pathlib import PurePosixPath

from sqlalchemy import func, or_, select

from rag_kb.agent.result import AgentResultStore
from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.infrastructure.database.models import AgentTaskRow, ArtifactRow, ToolCallRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore


logger = logging.getLogger(__name__)

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
_RESULT_PREFIXES = ("tool-results", "task-results", "artifacts")
_ORPHAN_RETRY_CAPACITY = 100


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


@dataclass(frozen=True)
class _TaskContentCandidate:
    task_id: str
    checkpoint_pending: bool
    result_pending: bool
    result_ref: str | None


@dataclass(frozen=True)
class _ArtifactContentCandidate:
    artifact_id: str
    storage_key: str


class TaskRetentionService:
    """保留策略只处理正文与 checkpoint，不改写执行结论。"""

    def __init__(
        self,
        *,
        database: Database,
        repository: DatabaseTaskRepository,
        objects: LocalObjectStore,
        results: AgentResultStore,
        checkpointer=None,
        orphan_grace_seconds: int,
        now=None,
        close=None,
    ) -> None:
        if orphan_grace_seconds <= 0:
            raise ValueError("invalid_result_object_orphan_grace")
        self._database = database
        self._repository = repository
        self._objects = objects
        self._results = results
        self._checkpointer = checkpointer
        self._orphan_grace = timedelta(seconds=orphan_grace_seconds)
        self._now = now or (lambda: datetime.now(UTC))
        self._close = close
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
        return self._repository.expire_waiting_tasks(limit=limit)

    def purge_expired_tool_results(self, *, limit: int = 100) -> int:
        """正文到期后只清引用；工具的 succeeded 审计结论保持不变。"""
        self._validate_limit(limit)
        purged = 0
        for tool_call_id, result_ref in self._expired_tool_result_refs(limit):
            try:
                self._objects.delete_exact(result_ref)
            except (OSError, ValueError):
                logger.error("工具结果对象删除失败 tool_call_id=%s", tool_call_id)
                continue
            if self._mark_tool_result_purged(tool_call_id, result_ref):
                purged += 1
        return purged

    def purge_finished_task_content(self, *, limit: int = 100) -> int:
        """删除过期终态 checkpoint、结果与产物正文，分段提交 marker 以支持重扫。"""
        self._validate_limit(limit)
        purged = 0
        for candidate in self._expired_task_content(limit):
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
                if self._mark_task_content_purged(
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
        for candidate in self._expired_artifact_content(limit):
            try:
                self._objects.delete_exact(candidate.storage_key)
            except (OSError, ValueError):
                logger.error("产物正文删除失败 artifact_id=%s", candidate.artifact_id)
                continue
            if self._mark_artifact_content_purged(candidate):
                purged += 1
        return purged

    def purge_orphaned_result_objects(self, *, limit: int = 100) -> int:
        """清理未绑定、已过 grace 的服务端结果对象。"""
        self._validate_limit(limit)
        cutoff = self._now() - self._orphan_grace
        with self._database.sessions() as session:
            exclude = frozenset(
                self._referenced_refs(session) | self._running_tool_refs(session)
            )
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

        with self._database.sessions() as session:
            # 先确认在途执行：若它随后合法绑定正文，仍由保护集合阻止删除。
            protected = self._running_tool_result_refs(
                session, refs
            ) | self._running_task_result_refs(session, refs)
            referenced = self._referenced_refs(session)

        purged = 0
        for ref in refs:
            if ref in referenced or ref in protected:
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

    def _expired_tool_result_refs(self, limit: int) -> tuple[tuple[str, str], ...]:
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
                (row.tool_call_id, row.result_ref)
                for row in rows
                if row.result_ref is not None
            )

    def _mark_tool_result_purged(self, tool_call_id: str, result_ref: str) -> bool:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            row = work.session.scalar(
                select(ToolCallRow)
                .where(ToolCallRow.tool_call_id == tool_call_id)
                .with_for_update()
            )
            if (
                row is None
                or row.status != "succeeded"
                or row.result_ref != result_ref
                or row.result_purged_at is not None
            ):
                return False
            row.result_ref = None
            row.result_size = None
            row.result_purged_at = self._database_now(work.session)
            return True

    def _expired_task_content(self, limit: int) -> tuple[_TaskContentCandidate, ...]:
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
                _TaskContentCandidate(
                    task_id=row.task_id,
                    checkpoint_pending=row.checkpoint_purged_at is None,
                    result_pending=row.result_purged_at is None,
                    result_ref=row.result_ref,
                )
                for row in rows
            )

    def _expired_artifact_content(self, limit: int) -> tuple[_ArtifactContentCandidate, ...]:
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
                _ArtifactContentCandidate(
                    artifact_id=row.artifact_id,
                    storage_key=row.storage_key,
                )
                for row in rows
                if isinstance(row.storage_key, str)
            )

    def _mark_artifact_content_purged(self, candidate: _ArtifactContentCandidate) -> bool:
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

    def _mark_task_content_purged(
        self,
        candidate: _TaskContentCandidate,
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

    def _running_tool_result_refs(self, session, refs: list[str]) -> set[str]:
        return set(refs) & self._running_tool_refs(session)

    def _running_task_result_refs(self, session, refs: list[str]) -> set[str]:
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

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
