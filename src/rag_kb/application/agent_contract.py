"""Agent 运行契约：运行上下文、工具集合、结果候选与服务端结果存储。

application 是这些契约的唯一定义点；agent 层（LangGraph/模型适配）只消费
它们。``rag_kb.agent.context``/``rag_kb.agent.result`` 保留 re-export，兼容
已持久化 MsgPack checkpoint 的类型引用与既有导入路径。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.application.ports.object_store import ObjectStorePort
from rag_kb.application.ports.task_read import ArtifactSummary, TaskResult
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode
from rag_kb.models import Citation


class AgentRunContext(BaseModel):
    """模型与工具均不可通过请求或工具参数覆盖任何控制字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    task_mode: TaskMode
    worker_id: str = Field(min_length=1, max_length=200)
    lease_generation: int = Field(ge=0)
    run_id: str = Field(min_length=1, max_length=64)
    auth: AuthContext
    # M2 受控写工具的服务端授权事实；批次 1 恒为默认 None（无审批即无执行），
    # 批次 2 的受控 resume 才由 Worker 填充。
    approval_id: str | None = Field(default=None, max_length=64)
    plan_revision: int | None = Field(default=None, ge=1)
    plan_digest: str | None = Field(default=None, max_length=80)
    evidence_digest: str | None = Field(default=None, max_length=80)
    idempotency_key: str | None = Field(default=None, max_length=200)
    # 服务端 plan 派生的精确证据集合；模型不可通过参数注入。
    evidence_ids: tuple[str, ...] = ()


# M2 批次 1 精确开放的两个 WRITE 工具；HITL interrupt_on 必须与之逐字一致。
CONTROLLED_WRITE_TOOL_NAMES = (
    "create_report_artifact",
    "create_work_item_draft",
)


class TaskResultCandidate(BaseModel):
    """create_agent response_format 产出的不可信候选；服务端复核后才可交付。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["succeeded", "needs_clarification", "refused"]
    answer: str = Field(default="", max_length=20_000)
    question: str = Field(default="", max_length=2_000)
    evidence_ids: tuple[str, ...] = ()
    # 模型声明的产物引用；服务端 settle 前逐个校验归属与一致性。
    artifact_ids: tuple[str, ...] = ()
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
        if len(self.artifact_ids) > 20 or any(
            not artifact_id.strip() or len(artifact_id) > 64
            for artifact_id in self.artifact_ids
        ):
            raise ValueError("task_result_candidate_invalid")
        return self


class AgentResultStore:
    """任务结果与澄清回答的服务端短期对象；键不可由模型或客户端指定。"""

    def __init__(self, objects: ObjectStorePort) -> None:
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


def build_controlled_task_result(
    candidate: TaskResultCandidate,
    *,
    task_id: str,
    citations: tuple[Citation, ...],
    artifacts: tuple[ArtifactSummary, ...] = (),
) -> TaskResult:
    """受控任务最终结果：result_kind 与 artifacts 摘要只由服务端构造。"""
    result = build_task_result(candidate, task_id=task_id, citations=citations)
    return result.model_copy(
        update={
            "result_kind": "controlled_action",
            "artifacts": () if result.status == "refused" else artifacts,
        }
    )
