"""Fast Query 的任务、证据和 citation 原子生命周期适配器。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import update

from rag_kb.application.ports.knowledge_query import KnowledgeQueryLifecyclePort
from rag_kb.domain.evidence import EvidenceRecord
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import AgentTaskRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.models import Citation


class DatabaseKnowledgeQueryLifecycle(KnowledgeQueryLifecyclePort):
    def __init__(
        self,
        database: Database,
        evidence: EvidenceLedgerRepository | None = None,
    ) -> None:
        self._database = database
        self._evidence = evidence or EvidenceLedgerRepository()

    def start_fenced_query_with_evidence(
        self,
        *,
        task_id: str,
        query_fingerprint: str,
        auth: AuthContext,
        candidates: tuple[RetrievalCandidate, ...],
        lease_owner: str,
        lease_seconds: int,
        lease_generation: int,
    ) -> tuple[EvidenceRecord, ...]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            work.session.add(
                AgentTaskRow(
                    task_id=task_id,
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    task_mode="knowledge_query",
                    status="running",
                    goal=f"knowledge_query:{query_fingerprint}",
                    auth_context_ref=self._auth_context_ref(auth),
                    lease_owner=lease_owner,
                    lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_seconds),
                    lease_generation=lease_generation,
                )
            )
            work.session.flush()
            return self._evidence.record_candidates(
                work.session,
                task_id=task_id,
                query_fingerprint=query_fingerprint,
                auth=auth,
                candidates=candidates,
            )

    def finish_fenced_query_with_citations(
        self,
        *,
        task_id: str,
        auth: AuthContext,
        evidence_ids: tuple[str, ...],
        lease_owner: str,
        lease_generation: int,
    ) -> tuple[Citation, ...]:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            citations = self._evidence.citations_from_ledger(
                work.session,
                task_id=task_id,
                evidence_ids=evidence_ids,
                auth=auth,
            )
            if not self._close_owned_query(
                work.session,
                task_id=task_id,
                auth=auth,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                terminal_status="succeeded",
            ):
                raise RuntimeError("query_task_binding_lost")
            return citations

    def close_fenced_query(
        self,
        *,
        task_id: str,
        auth: AuthContext,
        lease_owner: str,
        lease_generation: int,
        terminal_status: Literal["succeeded", "failed"],
        error_code: str | None = None,
    ) -> bool:
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            return self._close_owned_query(
                work.session,
                task_id=task_id,
                auth=auth,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                terminal_status=terminal_status,
                error_code=error_code,
            )

    @staticmethod
    def _auth_context_ref(auth: AuthContext) -> str:
        return "sha256:" + hashlib.sha256(
            f"{auth.tenant_id}|{auth.user_id}|{auth.request_id}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _close_owned_query(
        session,
        *,
        task_id: str,
        auth: AuthContext,
        lease_owner: str,
        lease_generation: int,
        terminal_status: Literal["succeeded", "failed"],
        error_code: str | None = None,
    ) -> bool:
        now = datetime.now(UTC)
        result = session.execute(
            update(AgentTaskRow)
            .where(
                AgentTaskRow.task_id == task_id,
                AgentTaskRow.tenant_id == auth.tenant_id,
                AgentTaskRow.user_id == auth.user_id,
                AgentTaskRow.task_mode == "knowledge_query",
                AgentTaskRow.status == "running",
                AgentTaskRow.lease_owner == lease_owner,
                AgentTaskRow.lease_generation == lease_generation,
                AgentTaskRow.lease_expires_at.is_not(None),
                AgentTaskRow.lease_expires_at > now,
            )
            .values(
                status=terminal_status,
                last_error_code=error_code,
                lease_owner=None,
                lease_expires_at=None,
                state_version=AgentTaskRow.state_version + 1,
                updated_at=now,
            )
        )
        return result.rowcount == 1
