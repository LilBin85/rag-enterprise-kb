"""Stage 5 统一工具注册表先行契约测试。"""

import pytest
from pydantic import ValidationError

from rag_kb.domain.identity import Permission
from rag_kb.domain.tools import ApprovalPolicy, RiskLevel, ToolDefinition, ToolMode


EXPECTED_TOOLS = frozenset(
    {
        "search_knowledge",
        "fetch_evidence",
        "list_document_versions",
        "compare_document_versions",
        "create_report_artifact",
        "create_work_item_draft",
    }
)


def _definition(name: str = "search", *, mode=ToolMode.READ, approval=ApprovalPolicy.NEVER) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        version="1.0",
        description="structured test tool",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
        required_permissions=frozenset({Permission.KNOWLEDGE_READ}),
        mode=mode,
        risk_level=RiskLevel.LOW,
        timeout_seconds=10,
        idempotent=True,
        approval_policy=approval,
        result_verifier="hash",
        redaction_policy="metadata_only",
    )


def test_tool_definition_is_closed_and_contains_security_contract() -> None:
    definition = _definition()
    assert definition.required_permissions == frozenset({Permission.KNOWLEDGE_READ})
    assert definition.mode is ToolMode.READ
    assert definition.approval_policy is ApprovalPolicy.NEVER
    assert definition.result_verifier and definition.redaction_policy
    with pytest.raises(ValidationError):
        ToolDefinition(**{**definition.model_dump(), "unexpected": "field"})


def _registry_or_skip():
    module = pytest.importorskip("rag_kb.application.tool_registry")
    return module


def test_registry_has_exact_fixed_tool_set_and_rejects_unknown() -> None:
    module = _registry_or_skip()
    definitions = module.builtin_tool_definitions()
    assert {item.name for item in definitions} == EXPECTED_TOOLS
    for definition in definitions:
        assert definition.input_schema.get("additionalProperties") is False
        assert definition.result_verifier and definition.redaction_policy
        assert definition.required_permissions
        assert definition.idempotent
    with pytest.raises(Exception, match="tool_not_registered"):
        module.ToolRegistry().get("arbitrary_shell")


def test_registry_rejects_extra_arguments_and_missing_permission() -> None:
    module = _registry_or_skip()
    registry = module.ToolRegistry()
    with pytest.raises(Exception, match="tool_arguments_invalid"):
        registry.validate_input(
            registry.get("search_knowledge"), {"query": "q", "extra": "no"}
        )
    assert registry.get("search_knowledge").definition.required_permissions == frozenset({Permission.KNOWLEDGE_READ})
    with pytest.raises(Exception, match="tool_permission_denied"):
        registry.authorize("search_knowledge", permissions=frozenset())


def test_read_never_requires_approval_write_always_requires_it() -> None:
    module = _registry_or_skip()
    definitions = {d.name: d for d in module.builtin_tool_definitions()}
    assert all(definitions[name].approval_policy is ApprovalPolicy.NEVER for name in EXPECTED_TOOLS - {"create_report_artifact", "create_work_item_draft"})
    assert all(definitions[name].approval_policy is ApprovalPolicy.ALWAYS for name in {"create_report_artifact", "create_work_item_draft"})
    assert all(definitions[name].mode is ToolMode.READ for name in EXPECTED_TOOLS - {"create_report_artifact", "create_work_item_draft"})
    assert all(definitions[name].mode is ToolMode.WRITE for name in {"create_report_artifact", "create_work_item_draft"})
    registry = module.ToolRegistry()
    assert registry.requires_approval("search_knowledge") is False
    assert registry.requires_approval("create_report_artifact") is True
    with pytest.raises(Exception, match="approval_required"):
        registry.authorize("create_report_artifact", permissions=frozenset({Permission.ARTIFACT_WRITE}))


def test_work_item_draft_rejects_invalid_due_date_at_tool_boundary() -> None:
    module = _registry_or_skip()
    registry = module.ToolRegistry()
    valid = {"title": "草稿", "description": "描述", "due_date": "2026-08-27"}
    registry.validate_input(registry.get("create_work_item_draft"), valid)
    for invalid in ("not-a-date", "2026-13-45", "2026-8-27", "2026/08/27", "20260827"):
        with pytest.raises(Exception, match="tool_arguments_invalid"):
            registry.validate_input(
                registry.get("create_work_item_draft"),
                {"title": "草稿", "description": "描述", "due_date": invalid},
            )


def test_registry_rejects_model_injected_control_fields() -> None:
    module = _registry_or_skip()
    registry = module.ToolRegistry()
    injected = {
        "title": "报告",
        "content": "正文",
        "format": "markdown",
        "tenant_id": "tenant-evil",
        "roles": ["admin"],
        "approval_id": "forged",
        "idempotency_key": "forged",
        "thread_id": "evil-thread",
        "checkpoint_id": "evil-checkpoint",
        "storage_key": "secret/path",
    }
    with pytest.raises(Exception, match="tool_arguments_invalid"):
        registry.validate_input(registry.get("create_report_artifact"), injected)
