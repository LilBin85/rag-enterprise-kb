"""任务结果读取时的完整性、当前授权与 citation 复核。"""

from __future__ import annotations

from rag_kb.agent.result import AgentResultStore, TaskResult
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskState, TaskStatus
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.session import Database


class TaskResultError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class TaskResultService:
    def __init__(
        self,
        database: Database,
        results: AgentResultStore,
        evidence: EvidenceLedgerRepository | None = None,
    ) -> None:
        self._database = database
        self._results = results
        self._evidence = evidence or EvidenceLedgerRepository()

    def read(self, *, auth: AuthContext, state: TaskState) -> TaskResult:
        if (
            state.result_kind != "research"
            or state.result_ref is None
            or state.result_hash is None
            or state.status not in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED}
        ):
            raise TaskResultError("task_result_unavailable")
        try:
            result = self._results.read_task_result(state.result_ref, state.result_hash)
        except (OSError, ValueError) as exc:
            raise TaskResultError("task_result_integrity_failed") from exc
        expected_status = "succeeded" if state.status is TaskStatus.SUCCEEDED else "refused"
        if result.task_id != state.task_id or result.status != expected_status:
            raise TaskResultError("task_result_binding_invalid")
        if result.status == "refused":
            if result.evidence_ids or result.citations:
                raise TaskResultError("task_result_binding_invalid")
            return result
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
            raise TaskResultError("task_result_evidence_unavailable") from exc
        if citations != result.citations:
            raise TaskResultError("task_result_binding_invalid")
        return result
