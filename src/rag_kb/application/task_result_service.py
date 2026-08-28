"""任务结果读取时的完整性、当前授权与 citation 复核。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select

from rag_kb.agent.result import AgentResultStore, TaskResult
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskState, TaskStatus
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import AgentTaskRow
from rag_kb.infrastructure.database.session import Database


class TaskResultError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_EVIDENCE_AUTH_CODES = frozenset({"evidence_authorization_stale"})


def _aware(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


class TaskResultService:
    def __init__(
        self,
        database: Database,
        results: AgentResultStore,
        evidence: EvidenceLedgerRepository | None = None,
        now=None,
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
