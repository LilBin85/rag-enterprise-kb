"""结构化任务结果：模型候选不可信，最终结果只由服务端复核构造。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.models import Citation


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskResultCandidate(_StrictModel):
    """create_agent response_format 产出的不可信候选；服务端复核后才可交付。"""

    status: Literal["succeeded", "needs_clarification", "refused"]
    answer: str = Field(default="", max_length=20_000)
    question: str = Field(default="", max_length=2_000)
    evidence_ids: tuple[str, ...] = ()
    refusal_code: str | None = Field(default=None, max_length=100)
    stop_reason: str = Field(default="", max_length=200)

    def require_valid(self) -> "TaskResultCandidate":
        if self.status == "succeeded" and (
            not self.answer.strip() or not self.evidence_ids
        ):
            raise ValueError("task_result_candidate_invalid")
        if self.status == "needs_clarification" and not self.question.strip():
            raise ValueError("task_result_candidate_invalid")
        # refused 不允许携带证据：否则 result_available=true 但读取必然冲突。
        if self.status == "refused" and (
            not (self.refusal_code or "").strip() or self.evidence_ids
        ):
            raise ValueError("task_result_candidate_invalid")
        return self


class TaskResult(_StrictModel):
    """最终可交付结果；引用只能由 Evidence Ledger 复核构造。"""

    task_id: str
    result_kind: Literal["research"] = "research"
    status: Literal["succeeded", "refused"]
    answer: str = Field(default="", max_length=20_000)
    refusal_code: str | None = Field(default=None, max_length=100)
    evidence_ids: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    stop_reason: str = Field(default="", max_length=200)
    created_at: datetime


class AgentResultStore:
    """任务结果与澄清回答的服务端短期对象；键不可由模型或客户端指定。"""

    def __init__(self, objects: LocalObjectStore) -> None:
        self._objects = objects

    def save_task_result(self, task_id: str, result: TaskResult) -> tuple[str, str]:
        document = result.model_dump(mode="json")
        payload = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        ref = f"task-results/{task_id}/{uuid4().hex}.json"
        self._objects.put_exclusive(ref, payload)
        return ref, canonical_fingerprint("task_result_object", document)

    def read_task_result(self, ref: str, expected_hash: str) -> TaskResult:
        payload = self._objects.read(ref)
        try:
            document = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("task_result_integrity_failed") from exc
        if canonical_fingerprint("task_result_object", document) != expected_hash:
            raise ValueError("task_result_integrity_failed")
        return TaskResult.model_validate(document)

    def delete_task_result(self, ref: str) -> None:
        """保留期届满后删除结果正文；hash 仍保留在 agent_tasks 供审计。"""
        self._objects.delete_exact(ref)


def build_task_result(
    candidate: TaskResultCandidate,
    *,
    task_id: str,
    citations: tuple[Citation, ...],
) -> TaskResult:
    """把通过证据复核的候选装配为最终结果；created_at 由服务端赋值。"""
    status = "succeeded" if candidate.status == "succeeded" else "refused"
    evidence_ids = canonical_evidence_snapshot(
        candidate.evidence_ids, require_non_empty=status == "succeeded"
    ).evidence_ids
    if status == "refused":
        # 防御性清空：refused 结果与证据/引用互斥，与读取侧契约保持一致。
        evidence_ids = ()
        citations = ()
    if status == "succeeded" and len(citations) != len(evidence_ids):
        raise ValueError("task_result_citation_mismatch")
    return TaskResult(
        task_id=task_id,
        status=status,
        answer=candidate.answer.strip(),
        refusal_code=candidate.refusal_code if status == "refused" else None,
        evidence_ids=evidence_ids,
        citations=citations,
        stop_reason=candidate.stop_reason or "completed",
        created_at=datetime.now(UTC),
    )
