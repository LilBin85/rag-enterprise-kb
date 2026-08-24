"""单次企业知识问答：授权检索、证据入账、受约束生成、账本引用。"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate, RetrievalResult
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import AgentTaskRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.models import Citation, Refusal


logger = logging.getLogger(__name__)

_REFUSALS = {
    "no_authorized_evidence": "当前知识库中没有可用于回答的已授权资料",
    "insufficient_evidence": "现有已授权资料不足以可靠回答该问题",
    "conflicting_evidence": "检索到的资料不足以形成可靠结论",
    "retrieval_failure": "检索或生成服务暂时不可用，请稍后重试",
}

# 同步问答任务的自租约：请求期间持有租约，异步任务 worker 的领取查询会因此
# 跳过本任务（否则会误领为 agent 任务并以 lease 约束冲突破坏同步终态写入，
# 把可回答的查询误报为 conflicting_evidence）。终态写入时必须释放租约；
# 若 API 进程中途崩溃，租约过期后 worker 可将孤儿任务收尾为 failed。
_SYNC_LEASE_OWNER = "api-sync-knowledge-query"
_SYNC_LEASE_SECONDS = 120


class RetrieverPort(Protocol):
    def search(self, query: str, auth: AuthContext) -> RetrievalResult: ...


class ModelVerdictPort(Protocol):
    verdict: str
    answer: str
    refusal_kind: str | None
    used_chunk_ids: list[str]


class AnswerModelPort(Protocol):
    def answer(
        self,
        question: str,
        chunks: list[RetrievalCandidate],
    ) -> ModelVerdictPort: ...


class KnowledgeQueryResult(BaseModel):
    """响应不回显身份、查询正文或内部 evidence/task 标识。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["answer", "refusal"]
    answer: str | None = None
    refusal: Refusal | None = None
    citations: tuple[Citation, ...] = ()


class KnowledgeQueryService:
    """普通问答只运行一次检索管线，最终引用只能由证据账本构造。"""

    def __init__(
        self,
        database: Database,
        retriever: RetrieverPort,
        model: AnswerModelPort,
        evidence: EvidenceLedgerRepository | None = None,
    ) -> None:
        self._database = database
        self._retriever = retriever
        self._model = model
        self._evidence = evidence or EvidenceLedgerRepository()

    def query(self, question: str, auth: AuthContext) -> KnowledgeQueryResult:
        try:
            result = self._retriever.search(question, auth)
        except Exception:
            logger.exception(
                "企业知识检索失败 tenant=%s request_id=%s",
                auth.tenant_id,
                auth.request_id,
            )
            return self._refusal("retrieval_failure")
        if not result.candidates:
            kind = (
                "no_authorized_evidence"
                if result.insufficiency_code in {"no_authorized_evidence", "knowledge_read_denied"}
                else "insufficient_evidence"
            )
            return self._refusal(kind)

        task_id = uuid4().hex
        try:
            records = self._record_evidence(
                task_id=task_id,
                query_fingerprint=result.query_fingerprint,
                auth=auth,
                candidates=result.candidates,
            )
        except Exception:
            logger.exception(
                "证据入账失败 tenant=%s request_id=%s",
                auth.tenant_id,
                auth.request_id,
            )
            return self._refusal("retrieval_failure")

        try:
            verdict = self._model.answer(question, list(result.candidates))
        except Exception:
            logger.exception(
                "知识回答模型失败 tenant=%s task_id=%s",
                auth.tenant_id,
                task_id,
            )
            self._finish_task(task_id, auth, status="failed", error_code="answer_model_failed")
            return self._refusal("retrieval_failure")
        if verdict.verdict != "answer":
            kind = (
                "conflicting_evidence"
                if verdict.refusal_kind == "conflict"
                else "insufficient_evidence"
            )
            self._finish_task(task_id, auth, status="succeeded")
            return self._refusal(kind)

        used_ids = verdict.used_chunk_ids
        if len(used_ids) != len(set(used_ids)):
            self._finish_task(task_id, auth, status="failed", error_code="duplicate_model_citation")
            return self._refusal("conflicting_evidence")
        evidence_by_chunk = {
            candidate.chunk_id: record.evidence_id
            for candidate, record in zip(result.candidates, records, strict=True)
        }
        try:
            evidence_ids = tuple(evidence_by_chunk[chunk_id] for chunk_id in used_ids)
        except KeyError:
            self._finish_task(task_id, auth, status="failed", error_code="unknown_model_citation")
            return self._refusal("conflicting_evidence")
        if not evidence_ids or not verdict.answer.strip():
            self._finish_task(task_id, auth, status="failed", error_code="empty_supported_answer")
            return self._refusal("conflicting_evidence")

        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                citations = self._evidence.citations_from_ledger(
                    work.session,
                    task_id=task_id,
                    evidence_ids=evidence_ids,
                    auth=auth,
                )
                task = work.session.get(AgentTaskRow, task_id)
                if task is None or task.tenant_id != auth.tenant_id or task.user_id != auth.user_id:
                    raise RuntimeError("query_task_binding_lost")
                task.status = "succeeded"
                task.last_error_code = None
                task.lease_owner = None
                task.lease_expires_at = None
        except Exception:
            logger.exception(
                "账本引用复核失败 tenant=%s task_id=%s",
                auth.tenant_id,
                task_id,
            )
            self._finish_task(task_id, auth, status="failed", error_code="citation_revalidation_failed")
            return self._refusal("conflicting_evidence")
        return KnowledgeQueryResult(
            verdict="answer",
            answer=verdict.answer.strip(),
            citations=citations,
        )

    def _record_evidence(
        self,
        *,
        task_id: str,
        query_fingerprint: str,
        auth: AuthContext,
        candidates: tuple[RetrievalCandidate, ...],
    ):
        auth_ref = "sha256:" + hashlib.sha256(
            f"{auth.tenant_id}|{auth.user_id}|{auth.request_id}".encode("utf-8")
        ).hexdigest()
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
                    auth_context_ref=auth_ref,
                    lease_owner=_SYNC_LEASE_OWNER,
                    lease_expires_at=datetime.now(UTC)
                    + timedelta(seconds=_SYNC_LEASE_SECONDS),
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

    def _finish_task(
        self,
        task_id: str,
        auth: AuthContext,
        *,
        status: str,
        error_code: str | None = None,
    ) -> None:
        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                task = work.session.get(AgentTaskRow, task_id)
                if task is None or task.tenant_id != auth.tenant_id or task.user_id != auth.user_id:
                    return
                task.status = status
                task.last_error_code = error_code
                # 终态任务必须无租约（lease 约束），同步自租约在此释放。
                task.lease_owner = None
                task.lease_expires_at = None
        except Exception:
            logger.exception(
                "知识问答任务终态写入失败 tenant=%s task_id=%s",
                auth.tenant_id,
                task_id,
            )

    @staticmethod
    def _refusal(kind: str) -> KnowledgeQueryResult:
        return KnowledgeQueryResult(
            verdict="refusal",
            refusal=Refusal(kind=kind, message=_REFUSALS[kind]),
            citations=(),
        )
