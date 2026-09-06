"""同步知识问答的持久化生命周期契约。"""

from __future__ import annotations

from typing import Literal, Protocol

from rag_kb.domain.evidence import EvidenceRecord
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.models import Citation


class KnowledgeQueryLifecyclePort(Protocol):
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
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]: ...

    def finish_fenced_query_with_citations(
        self,
        *,
        task_id: str,
        auth: AuthContext,
        evidence_ids: tuple[str, ...],
        lease_owner: str,
        lease_generation: int,
        defer_terminal: bool = False,
    ) -> tuple[Citation, ...]: ...

    def close_fenced_query(
        self,
        *,
        task_id: str,
        auth: AuthContext,
        lease_owner: str,
        lease_generation: int,
        terminal_status: Literal["succeeded", "failed"],
        error_code: str | None = None,
    ) -> bool: ...
