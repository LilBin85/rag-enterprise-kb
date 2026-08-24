"""阶段1领域契约的快速、无外部组件单元测试。"""

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.evidence import EvidenceRecord
from rag_kb.domain.identity import Permission
from rag_kb.domain.tasks import (
    AskUserAction,
    NextAction,
    ProposeAction,
    SearchAction,
    TaskBudgets,
    TaskMode,
    TaskState,
    TaskStatus,
)
from rag_kb.domain.tools import (
    ApprovalPolicy,
    RiskLevel,
    ToolDefinition,
    ToolMode,
)


UTC_NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def test_next_action_discriminator_accepts_declared_actions() -> None:
    adapter = TypeAdapter(NextAction)

    assert isinstance(
        adapter.validate_python(
            {"kind": "search", "query": "制度", "reason_code": "need_evidence"}
        ),
        SearchAction,
    )
    assert isinstance(
        adapter.validate_python(
            {
                "kind": "ask_user",
                "question": "请确认范围",
                "reason_code": "missing_scope",
            }
        ),
        AskUserAction,
    )
    assert isinstance(
        adapter.validate_python(
            {
                "kind": "propose_action",
                "tool_name": "create_draft",
                "arguments": {},
                "evidence_ids": ("ev-1",),
                "reason_code": "grounded_proposal",
            }
        ),
        ProposeAction,
    )


def test_next_action_rejects_unknown_kind_and_extra_fields() -> None:
    adapter = TypeAdapter(NextAction)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "invented", "reason_code": "bad"})
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "search",
                "query": "制度",
                "reason_code": "need_evidence",
                "free_text": "不得进入动作契约",
            }
        )


def test_task_state_rejects_extra_fields_and_requires_aware_deadline() -> None:
    fields = {
        "task_id": "task-1",
        "auth_context_ref": "auth-1",
        "goal": "查询制度",
        "task_mode": TaskMode.KNOWLEDGE_QUERY,
        "status": TaskStatus.PLANNING,
    }
    assert TaskState(**fields, deadline=UTC_NOW).deadline == UTC_NOW

    with pytest.raises(ValidationError):
        TaskState(**fields, deadline=datetime(2026, 8, 22, 12, 0))
    with pytest.raises(ValidationError):
        TaskState(**fields, unsupported_state_field=True)


def test_task_budgets_enforce_nonnegative_and_positive_runtime_bounds() -> None:
    assert TaskBudgets(max_model_calls=0, max_tool_calls=0, max_replans=0).max_model_calls == 0
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
        ApprovalBinding(**{**values, "expires_at": datetime(2026, 8, 22, 12, 0)})
    with pytest.raises(ValidationError):
        ApprovalBinding(**{**values, "plan_revision": 0})


def test_evidence_record_has_only_mapping_fields_and_no正文入口() -> None:
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
        EvidenceRecord(**{**values, "retrieved_at": datetime(2026, 8, 22, 12, 0)})
    with pytest.raises(ValidationError):
        EvidenceRecord(**{**values, "content": "正文不得入账"})
