"""单次企业知识问答：授权检索、证据入账、受约束生成、账本引用。"""

from __future__ import annotations

from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.application.ports.knowledge_query import KnowledgeQueryLifecyclePort
from rag_kb.application.ports.retrieval import RetrieverPort
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.models import Citation, Refusal
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


_REFUSALS = {
    "no_authorized_evidence": "当前知识库中没有可用于回答的已授权资料",
    "insufficient_evidence": "现有已授权资料不足以可靠回答该问题",
    "conflicting_evidence": "检索到的资料不足以形成可靠结论",
    "retrieval_failure": "检索或生成服务暂时不可用，请稍后重试",
}

# 同步问答任务的自租约用于栅栏终态写入；通用 worker 不领取此模式，显式恢复者
# 仍可在租约过期后递增 generation 接管，迟到的 API 请求不得覆盖恢复结果。
_SYNC_LEASE_OWNER = "api-sync-knowledge-query"
_SYNC_LEASE_SECONDS = 120
_SYNC_LEASE_GENERATION = 1


class _QueryFailure(RuntimeError):
    def __init__(self, code: str, refusal_kind: str = "retrieval_failure") -> None:
        super().__init__(code)
        self.code = code
        self.refusal_kind = refusal_kind


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
    # 只供统一会话持久化绑定；任何 HTTP 响应都不能暴露内部 evidence id。
    evidence_ids: tuple[str, ...] = Field(default=(), exclude=True, repr=False)


class KnowledgeQueryService:
    """普通问答只运行一次检索管线，最终引用只能由证据账本构造。"""

    def __init__(
        self,
        lifecycle: KnowledgeQueryLifecyclePort,
        retriever: RetrieverPort,
        model: AnswerModelPort,
    ) -> None:
        self._lifecycle = lifecycle
        self._retriever = retriever
        self._model = model

    def query(
        self,
        question: str,
        auth: AuthContext,
        *,
        task_id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> KnowledgeQueryResult:
        started_at = perf_counter()
        outcome = "failure"
        reason = "retrieval_failure"
        error_code: str | None = None
        with observe_span(
            "knowledge.query",
            attributes={"component": "knowledge", "operation": "knowledge.query"},
        ) as span:
            try:
                response = self._query(
                    question,
                    auth,
                    task_id=task_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
                if response.verdict == "answer":
                    outcome, reason = "answer", "none"
                elif response.refusal is not None and response.refusal.kind == "retrieval_failure":
                    outcome, reason = "failure", "retrieval_failure"
                    error_code = "retrieval_failure"
                else:
                    outcome = "refusal"
                    reason = response.refusal.kind if response.refusal is not None else "conflicting_evidence"
                return response
            except _QueryFailure as exc:
                error_code = exc.code
                return self._refusal(exc.refusal_kind)
            except Exception as exc:
                error_code = stable_error_code(exc)
                return self._refusal("retrieval_failure")
            finally:
                duration_seconds = perf_counter() - started_at
                category = error_category_for(error_code) if error_code is not None else "none"
                finish_observed_span(
                    span,
                    outcome=outcome,
                    error_category=category,
                    error_code=error_code,
                    attributes={"duration_ms": duration_seconds * 1_000},
                )
                telemetry = current_observability()
                if telemetry is not None:
                    telemetry.record_knowledge_query(
                        outcome=outcome,
                        reason=reason,
                        duration_seconds=duration_seconds,
                    )
                log_event(
                    "knowledge.query.completed",
                    level=40 if outcome == "failure" else 20,
                    component="knowledge",
                    operation="knowledge.query",
                    outcome=outcome,
                    error_category=category,
                    error_code=error_code,
                    duration_ms=duration_seconds * 1_000,
                )

    def _query(
        self,
        question: str,
        auth: AuthContext,
        *,
        task_id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> KnowledgeQueryResult:
        try:
            result = self._retriever.search(question, auth)
        except Exception:
            raise _QueryFailure("retrieval_failure") from None
        if not result.candidates:
            kind = (
                "no_authorized_evidence"
                if result.insufficiency_code in {"no_authorized_evidence", "knowledge_read_denied"}
                else "insufficient_evidence"
            )
            return self._refusal(kind)

        task_id = task_id or uuid4().hex
        conversation_bound = conversation_id is not None and turn_id is not None
        try:
            binding: dict[str, str | None] = {}
            if conversation_id is not None or turn_id is not None:
                binding = {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                }
            records = self._lifecycle.start_fenced_query_with_evidence(
                task_id=task_id,
                query_fingerprint=result.query_fingerprint,
                auth=auth,
                candidates=result.candidates,
                lease_owner=_SYNC_LEASE_OWNER,
                lease_seconds=_SYNC_LEASE_SECONDS,
                lease_generation=_SYNC_LEASE_GENERATION,
                **binding,
            )
        except Exception:
            raise _QueryFailure("evidence_record_failed") from None

        try:
            verdict = self._model.answer(question, list(result.candidates))
        except Exception:
            if conversation_bound:
                # 会话路径由同一事务写任务、轮次和助手拒答，不提前分裂结算。
                return self._refusal("retrieval_failure")
            self._finish_task(task_id, auth, status="failed", error_code="answer_model_failed")
            raise _QueryFailure("answer_model_failed") from None
        if verdict.verdict != "answer":
            kind = (
                "conflicting_evidence"
                if verdict.refusal_kind == "conflict"
                else "insufficient_evidence"
            )
            if not conversation_bound:
                self._finish_task(task_id, auth, status="succeeded")
            return self._refusal(kind)

        used_ids = verdict.used_chunk_ids
        if len(used_ids) != len(set(used_ids)):
            if not conversation_bound:
                self._finish_task(task_id, auth, status="failed", error_code="duplicate_model_citation")
            return self._refusal("conflicting_evidence")
        evidence_by_chunk = {
            candidate.chunk_id: record.evidence_id
            for candidate, record in zip(result.candidates, records, strict=True)
        }
        try:
            evidence_ids = tuple(evidence_by_chunk[chunk_id] for chunk_id in used_ids)
        except KeyError:
            if not conversation_bound:
                self._finish_task(task_id, auth, status="failed", error_code="unknown_model_citation")
            return self._refusal("conflicting_evidence")
        if not evidence_ids or not verdict.answer.strip():
            if not conversation_bound:
                self._finish_task(task_id, auth, status="failed", error_code="empty_supported_answer")
            return self._refusal("conflicting_evidence")

        try:
            citations = self._lifecycle.finish_fenced_query_with_citations(
                task_id=task_id,
                auth=auth,
                evidence_ids=evidence_ids,
                lease_owner=_SYNC_LEASE_OWNER,
                lease_generation=_SYNC_LEASE_GENERATION,
                defer_terminal=conversation_bound,
            )
        except Exception:
            if conversation_bound:
                return self._refusal("retrieval_failure")
            self._finish_task(task_id, auth, status="failed", error_code="citation_revalidation_failed")
            raise _QueryFailure(
                "citation_revalidation_failed", refusal_kind="conflicting_evidence"
            ) from None
        return KnowledgeQueryResult(
            verdict="answer",
            answer=verdict.answer.strip(),
            citations=citations,
            evidence_ids=evidence_ids,
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
            self._lifecycle.close_fenced_query(
                task_id=task_id,
                auth=auth,
                lease_owner=_SYNC_LEASE_OWNER,
                lease_generation=_SYNC_LEASE_GENERATION,
                terminal_status=status,
                error_code=error_code,
            )
        except Exception:
            log_event(
                "knowledge.query.task_finish_failed",
                level=40,
                component="knowledge",
                operation="knowledge.query",
                outcome="failure",
                error_category="storage",
                error_code="query_task_finish_failed",
                task_id=task_id,
            )

    @staticmethod
    def _refusal(kind: str) -> KnowledgeQueryResult:
        return KnowledgeQueryResult(
            verdict="refusal",
            refusal=Refusal(kind=kind, message=_REFUSALS[kind]),
            citations=(),
        )
