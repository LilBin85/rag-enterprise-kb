"""M1 任务业务状态与既有安全契约。"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.evidence import EvidenceRecord
from rag_kb.domain.identity import Permission
from rag_kb.domain.tasks import TaskBudgets, TaskMode, TaskState, TaskStatus
from rag_kb.domain.tools import ApprovalPolicy, RiskLevel, ToolDefinition, ToolMode


UTC_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def _task_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "task_id": "task-1",
        "auth_context_ref": "auth-1",
        "goal": "查询制度",
        "task_mode": TaskMode.RESEARCH,
        "status": TaskStatus.PLANNING,
        "deadline": UTC_NOW,
    }
    fields.update(overrides)
    return fields


def test_task_state_only_accepts_business_lifecycle_fields() -> None:
    assert TaskState(**_task_fields()).deadline == UTC_NOW

    with pytest.raises(ValidationError):
        TaskState(**_task_fields(deadline=datetime(2026, 8, 25, 12, 0)))
    with pytest.raises(ValidationError):
        TaskState(**_task_fields(current_node="select_action"))
    with pytest.raises(ValidationError):
        TaskState(**_task_fields(pending_question="不得复制 checkpoint 正文"))


def test_task_state_binds_interrupt_and_research_result_to_status() -> None:
    waiting = TaskState(
        **_task_fields(
            status=TaskStatus.WAITING_FOR_USER,
            interrupt_kind="user_clarification",
            interrupt_summary="请确认部门",
        )
    )
    assert waiting.interrupt_kind == "user_clarification"

    with pytest.raises(ValidationError, match="task_interrupt_binding_missing"):
        TaskState(**_task_fields(status=TaskStatus.WAITING_FOR_USER))
    with pytest.raises(ValidationError, match="task_interrupt_only_when_waiting"):
        TaskState(**_task_fields(interrupt_kind="user_clarification", interrupt_summary="问题"))
    with pytest.raises(ValidationError, match="task_result_binding_missing"):
        TaskState(
            **_task_fields(
                status=TaskStatus.SUCCEEDED,
                content_expires_at=UTC_NOW + timedelta(days=7),
            )
        )
    with pytest.raises(ValidationError, match="task_content_expiry_missing"):
        TaskState(**_task_fields(status=TaskStatus.FAILED))

    completed = TaskState(
        **_task_fields(
            status=TaskStatus.SUCCEEDED,
            result_kind="research",
            result_ref="task-results/task-1/result.json",
            result_hash="sha256:" + "a" * 64,
            content_expires_at=UTC_NOW + timedelta(days=7),
        )
    )
    assert completed.result_ref is not None
    assert completed.result_available(now=UTC_NOW)
    assert not completed.result_available(now=UTC_NOW + timedelta(days=8))

    purged = TaskState(
        **_task_fields(
            status=TaskStatus.REFUSED,
            result_kind="research",
            result_hash="sha256:" + "a" * 64,
            result_purged_at=UTC_NOW,
            content_expires_at=UTC_NOW + timedelta(days=7),
        )
    )
    assert not purged.result_available(now=UTC_NOW)

    with pytest.raises(ValidationError, match="task_purged_result_ref_present"):
        TaskState(
            **_task_fields(
                status=TaskStatus.REFUSED,
                result_kind="research",
                result_ref="task-results/task-1/result.json",
                result_hash="sha256:" + "a" * 64,
                result_purged_at=UTC_NOW,
                content_expires_at=UTC_NOW + timedelta(days=7),
            )
        )


def test_task_budgets_enforce_nonnegative_and_positive_runtime_bounds() -> None:
    assert TaskBudgets(max_model_calls=0, max_tool_calls=0).max_model_calls == 0
    with pytest.raises(ValidationError):
        TaskBudgets(max_model_calls=-1)
    with pytest.raises(ValidationError):
        TaskBudgets(max_runtime_seconds=0)


def test_tool_definition_requires_security_declaration_fields() -> None:
    required = {
        "name",
        "version",
        "description",
        "input_schema",
        "output_schema",
        "required_permissions",
        "mode",
        "risk_level",
        "timeout_seconds",
        "idempotent",
        "approval_policy",
        "result_verifier",
        "redaction_policy",
    }
    assert required <= set(ToolDefinition.model_fields)

    tool = ToolDefinition(
        name="read_policy",
        version="1",
        description="读取授权制度",
        input_schema={},
        output_schema={},
        required_permissions=frozenset({Permission.KNOWLEDGE_READ}),
        mode=ToolMode.READ,
        risk_level=RiskLevel.LOW,
        timeout_seconds=30,
        idempotent=True,
        approval_policy=ApprovalPolicy.NEVER,
        result_verifier="verify_evidence",
        redaction_policy="no_body_text",
    )
    assert tool.approval_policy is ApprovalPolicy.NEVER
    with pytest.raises(ValidationError):
        ToolDefinition(**tool.model_dump(), undeclared=True)


def test_approval_binding_requires_aware_expiry_and_positive_revision() -> None:
    values = {
        "approval_id": "approval-1",
        "task_id": "task-1",
        "tenant_id": "tenant-1",
        "plan_revision": 1,
        "plan_digest": "d" * 64,
        "tool_name": "create_draft",
        "tool_version": "1",
        "canonical_arguments_hash": "a" * 64,
        "principal_hash": "b" * 64,
        "evidence_hash": "c" * 64,
        "expires_at": UTC_NOW,
    }
    assert ApprovalBinding(**values).expires_at == UTC_NOW
    with pytest.raises(ValidationError):
        ApprovalBinding(**{**values, "expires_at": datetime(2026, 8, 25, 12, 0)})
    with pytest.raises(ValidationError):
        ApprovalBinding(**{**values, "plan_revision": 0})


def test_evidence_record_has_only_mapping_fields_and_no_body() -> None:
    assert "text" not in EvidenceRecord.model_fields
    assert "content" not in EvidenceRecord.model_fields
    values = {
        "evidence_id": "ev-1",
        "task_id": "task-1",
        "query_fingerprint": "q" * 64,
        "tenant_id": "tenant-1",
        "document_id": "doc-1",
        "version_id": "version-1",
        "document_version": 1,
        "chunk_id": "chunk-1",
        "content_hash": "c" * 64,
        "source_locator": "制度/第1条",
        "retrieved_at": UTC_NOW,
        "permission_granted": True,
    }
    assert EvidenceRecord(**values).retrieved_at == UTC_NOW
    with pytest.raises(ValidationError):
        EvidenceRecord(**{**values, "content": "正文不得入账"})
