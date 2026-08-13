"""已批准策略计划的确定性本地草稿执行；不含模型、工具或外部副作用。"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rag_kb.models import (
    ExecutionStatus,
    PolicyRolloutApprovalResponse,
    PolicyRolloutClarification,
    PolicyRolloutExecutionResponse,
    PolicyRolloutPlanVersion,
    Role,
    TodoCitation,
)
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore

_WORK_ITEM_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-enterprise-kb/policy-rollout-draft")
ACTION = "create_local_draft_work_item"


class ApprovalConflict(RuntimeError):
    pass


class ExecutionConflict(RuntimeError):
    pass


def canonical_plan_digest(goal: str, clarification: PolicyRolloutClarification | None, plan: PolicyRolloutPlanVersion, citation_hashes: list[tuple[str, str]]) -> str:
    """摘要只绑定服务端持久化计划及当前证据哈希，绝不接受客户端摘要。"""
    payload = {
        "citation_content_hashes": [{"chunk_id": chunk_id, "content_hash": content_hash} for chunk_id, content_hash in sorted(citation_hashes)],
        "clarification": clarification.model_dump(mode="json") if clarification else None,
        "goal": goal,
        "plan": plan.model_dump(mode="json"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class PolicyRolloutExecutionService:
    # ponytail: v2 keeps execution synchronous and local-only until an external side effect is justified.
    def __init__(self, store: PolicyRolloutTaskStore, draft_dir: Path, citation_hashes: Callable[[list[TodoCitation], Role], list[tuple[str, str]] | None]) -> None:
        self.store, self.draft_dir, self.citation_hashes = store, draft_dir, citation_hashes

    def approve(self, task_id: str, version: int, role: Role) -> PolicyRolloutApprovalResponse | None:
        resolved = self._resolved(task_id, version, role)
        if resolved is None:
            return None
        goal, clarification, plan, hashes = resolved
        approved = self.store.approve(task_id, version, role, canonical_plan_digest(goal, clarification, plan, hashes))
        if approved is None:
            raise ApprovalConflict("approval_conflict")
        return approved

    def execute(self, task_id: str, version: int, role: Role, idempotency_key: str, action: str) -> PolicyRolloutExecutionResponse | None:
        if action != ACTION:
            raise ExecutionConflict("invalid_action")
        resolved = self._resolved(task_id, version, role)
        if resolved is None:
            return None
        goal, clarification, plan, hashes = resolved
        digest = canonical_plan_digest(goal, clarification, plan, hashes)
        approval = self.store.approval(task_id, version, role)
        if approval is None or approval.digest != digest:
            raise ExecutionConflict("approval_required_or_stale")
        work_item_id = str(uuid.uuid5(_WORK_ITEM_NAMESPACE, f"{task_id}:{version}:{idempotency_key}"))
        existing = self.store.get_execution(task_id, role, idempotency_key=idempotency_key)
        if existing is not None:
            self._validate_execution_match(existing, version, role, action, digest, work_item_id)
            return self._reconcile(existing, goal=goal, clarification=clarification, plan=plan)
        response = PolicyRolloutExecutionResponse(
            execution_id=str(uuid.uuid4()), task_id=task_id, version=version, role=role,
            idempotency_key=idempotency_key, action=ACTION, status=ExecutionStatus.RUNNING,
            digest=digest, work_item_id=work_item_id, draft_path=self._draft_name(work_item_id), created_at=datetime.now(UTC),
        )
        claimed = self.store.claim_execution(response)
        self._validate_execution_match(claimed, version, role, action, digest, work_item_id)
        return self._reconcile(claimed, goal=goal, clarification=clarification, plan=plan)

    def get_execution(self, task_id: str, role: Role, *, idempotency_key: str | None, execution_id: str | None) -> PolicyRolloutExecutionResponse | None:
        response = self.store.get_execution(task_id, role, idempotency_key=idempotency_key, execution_id=execution_id)
        return self._reconcile(response) if response else None

    def _resolved(self, task_id: str, version: int, role: Role) -> tuple[str, PolicyRolloutClarification | None, PolicyRolloutPlanVersion, list[tuple[str, str]]] | None:
        found = self.store.get_plan(task_id, version, role)
        if found is None:
            return None
        goal, clarification, plan = found
        try:
            hashes = self.citation_hashes([citation for todo in plan.todos for citation in todo.citations], role)
        except Exception:
            return None
        return (goal, clarification, plan, hashes) if hashes is not None else None

    def _reconcile(self, response: PolicyRolloutExecutionResponse, *, goal: str | None = None, clarification: PolicyRolloutClarification | None = None, plan: PolicyRolloutPlanVersion | None = None) -> PolicyRolloutExecutionResponse:
        if response.status == ExecutionStatus.UNKNOWN:
            raise ExecutionConflict("unknown_execution")
        path = self._draft_path(response)
        expected = self._draft_json(response, goal, clarification, plan) if plan else None
        if path.exists():
            try:
                actual = path.read_bytes()
            except OSError as exc:
                self.store.update_execution_status(response.execution_id, ExecutionStatus.UNKNOWN)
                raise ExecutionConflict("draft_state_unknown") from exc
            if expected is not None and actual != expected:
                self.store.update_execution_status(response.execution_id, ExecutionStatus.UNKNOWN)
                raise ExecutionConflict("draft_conflict")
            if expected is None:
                try:
                    parsed = json.loads(actual)
                except (OSError, json.JSONDecodeError) as exc:
                    self.store.update_execution_status(response.execution_id, ExecutionStatus.UNKNOWN)
                    raise ExecutionConflict("draft_state_unknown") from exc
                if parsed.get("approval_digest") != response.digest or parsed.get("work_item_id") != response.work_item_id:
                    self.store.update_execution_status(response.execution_id, ExecutionStatus.UNKNOWN)
                    raise ExecutionConflict("draft_conflict")
            return self.store.update_execution_status(response.execution_id, ExecutionStatus.SUCCEEDED) or response
        if response.status == ExecutionStatus.SUCCEEDED:
            self.store.update_execution_status(response.execution_id, ExecutionStatus.UNKNOWN)
            raise ExecutionConflict("draft_missing")
        if expected is None:
            raise ExecutionConflict("execution_recovery_requires_request")
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            return self._reconcile(response, goal=goal, clarification=clarification, plan=plan)
        except OSError as exc:
            self.store.update_execution_status(response.execution_id, ExecutionStatus.UNKNOWN)
            raise ExecutionConflict("draft_write_ambiguous") from exc
        return self.store.update_execution_status(response.execution_id, ExecutionStatus.SUCCEEDED) or response

    def _draft_name(self, work_item_id: str) -> str:
        return f"{work_item_id}.json"

    def _validate_execution_match(
        self, execution: PolicyRolloutExecutionResponse, version: int, role: Role,
        action: str, digest: str, work_item_id: str,
    ) -> None:
        if (
            execution.version != version
            or execution.role != role
            or execution.action != action
            or execution.digest != digest
            or execution.work_item_id != work_item_id
            or execution.draft_path != self._draft_name(work_item_id)
        ):
            raise ExecutionConflict("idempotency_conflict")

    def _draft_path(self, response: PolicyRolloutExecutionResponse) -> Path:
        expected = self._draft_name(response.work_item_id)
        candidate = Path(response.draft_path)
        if candidate.name != expected or candidate != Path(expected):
            raise ExecutionConflict("draft_conflict")
        root = self.draft_dir.resolve()
        path = (root / candidate).resolve()
        if path.parent != root:
            raise ExecutionConflict("draft_conflict")
        return path

    @staticmethod
    def _draft_json(response: PolicyRolloutExecutionResponse, goal: str | None, clarification: PolicyRolloutClarification | None, plan: PolicyRolloutPlanVersion | None) -> bytes:
        if goal is None or plan is None:
            raise ExecutionConflict("missing_draft_payload")
        payload = {
            "approval_digest": response.digest,
            "citations": [citation.model_dump(mode="json") for todo in plan.todos for citation in todo.citations],
            "created_at": response.created_at.isoformat(),
            "task_id": response.task_id,
            "todos": [todo.model_dump(mode="json") for todo in plan.todos],
            "version": response.version,
            "work_item_id": response.work_item_id,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
