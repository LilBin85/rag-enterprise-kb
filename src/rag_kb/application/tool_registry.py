"""固定白名单工具注册表；模型无法访问注册表之外的能力。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.tools import ApprovalPolicy, RiskLevel, ToolDefinition, ToolMode


class ToolRegistryError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ToolExecutionContext:
    auth: AuthContext
    task_id: str
    tool_call_id: str
    plan_revision: int
    plan_digest: str
    evidence_digest: str
    idempotency_key: str | None = None


class ToolAdapter(Protocol):
    def execute(
        self, arguments: dict[str, object], context: ToolExecutionContext
    ) -> dict[str, object]: ...

    def verify(self, result: dict[str, object], context: ToolExecutionContext) -> bool: ...


class _UnavailableAdapter:
    """声明可见但尚未接线的工具必须关闭执行，而不是悄悄降级。"""

    def execute(
        self, arguments: dict[str, object], context: ToolExecutionContext
    ) -> dict[str, object]:
        raise ToolRegistryError("tool_adapter_unavailable")

    def verify(self, result: dict[str, object], context: ToolExecutionContext) -> bool:
        return False


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    adapter: ToolAdapter


class ToolRegistry:
    def __init__(
        self,
        definitions: tuple[ToolDefinition, ...] | list[ToolDefinition] | None = None,
        *,
        adapters: dict[str, ToolAdapter] | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        configured_adapters = adapters or {}
        for definition in definitions or builtin_tool_definitions():
            self.register(
                definition,
                configured_adapters.get(definition.name, _UnavailableAdapter()),
            )

    def register(self, definition: ToolDefinition, adapter: ToolAdapter) -> None:
        if definition.name in self._tools:
            raise ToolRegistryError("tool_already_registered")
        if definition.mode is ToolMode.READ and definition.approval_policy is not ApprovalPolicy.NEVER:
            raise ToolRegistryError("read_tool_approval_policy_invalid")
        if definition.mode is ToolMode.WRITE and (
            definition.approval_policy is ApprovalPolicy.NEVER or not definition.idempotent
        ):
            raise ToolRegistryError("write_tool_security_declaration_invalid")
        try:
            Draft202012Validator.check_schema(definition.input_schema)
            Draft202012Validator.check_schema(definition.output_schema)
        except SchemaError as exc:
            raise ToolRegistryError("tool_schema_invalid") from exc
        self._tools[definition.name] = RegisteredTool(definition, adapter)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError("tool_not_registered") from exc

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name].definition for name in sorted(self._tools))

    def requires_approval(self, name: str) -> bool:
        return self.get(name).definition.approval_policy is not ApprovalPolicy.NEVER

    def authorize(
        self,
        name: str,
        *,
        permissions: frozenset[Permission],
        approval: object | None = None,
    ) -> RegisteredTool:
        tool = self.get(name)
        if not tool.definition.required_permissions.issubset(permissions):
            raise ToolRegistryError("tool_permission_denied")
        if self.requires_approval(name) and approval is None:
            raise ToolRegistryError("approval_required")
        return tool

    def validate_arguments(self, name: str, arguments: dict[str, object]) -> None:
        self.validate_input(self.get(name), arguments)

    @staticmethod
    def validate_input(tool: RegisteredTool, arguments: dict[str, object]) -> None:
        try:
            Draft202012Validator(tool.definition.input_schema).validate(arguments)
        except ValidationError as exc:
            raise ToolRegistryError("tool_arguments_invalid") from exc

    @staticmethod
    def validate_output(tool: RegisteredTool, result: dict[str, object]) -> None:
        try:
            Draft202012Validator(tool.definition.output_schema).validate(result)
        except ValidationError as exc:
            raise ToolRegistryError("tool_result_invalid") from exc


def _object_schema(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def builtin_tool_definitions() -> tuple[ToolDefinition, ...]:
    read_result = _object_schema(
        {
            "result_reference": {"type": "string", "minLength": 1},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "items": {"type": "array", "items": {"type": "object"}},
        },
        ["result_reference", "evidence_ids", "items"],
    )
    artifact_result = _object_schema(
        {
            "artifact_type": {"enum": ["report", "work_item_draft"]},
            "storage_key": {"type": "string", "minLength": 1},
            "content_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "status": {"const": "ready"},
        },
        ["artifact_type", "storage_key", "content_hash", "status"],
    )
    common = {
        "version": "1.0.0",
        "timeout_seconds": 30,
        "result_verifier": "adapter_readback_v1",
        "redaction_policy": "persist_hashes_and_references_only_v1",
    }
    read_common = {
        "mode": ToolMode.READ,
        "risk_level": RiskLevel.LOW,
        "idempotent": True,
        "approval_policy": ApprovalPolicy.NEVER,
    }
    write_common = {
        "mode": ToolMode.WRITE,
        "risk_level": RiskLevel.LOW,
        "idempotent": True,
        "approval_policy": ApprovalPolicy.ALWAYS,
    }
    return (
        ToolDefinition(
            name="search_knowledge",
            description="在当前租户和用户 ACL 内检索知识证据。",
            input_schema=_object_schema(
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                ["query"],
            ),
            output_schema=read_result,
            required_permissions=frozenset({Permission.KNOWLEDGE_READ}),
            **read_common,
            **common,
        ),
        ToolDefinition(
            name="fetch_evidence",
            description="按证据账本标识复核当前仍有效的来源映射。",
            input_schema=_object_schema(
                {
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    }
                },
                ["evidence_ids"],
            ),
            output_schema=read_result,
            required_permissions=frozenset({Permission.KNOWLEDGE_READ}),
            **read_common,
            **common,
        ),
        ToolDefinition(
            name="list_document_versions",
            description="列出用户当前有权查看的文档版本元数据。",
            input_schema=_object_schema(
                {"document_id": {"type": "string", "minLength": 1, "maxLength": 200}},
                ["document_id"],
            ),
            output_schema=read_result,
            required_permissions=frozenset({Permission.DOCUMENT_READ}),
            **read_common,
            **common,
        ),
        ToolDefinition(
            name="compare_document_versions",
            description="比较两个授权文档版本的块清单与元数据。",
            input_schema=_object_schema(
                {
                    "document_id": {"type": "string", "minLength": 1, "maxLength": 200},
                    "left_version": {"type": "integer", "minimum": 1},
                    "right_version": {"type": "integer", "minimum": 1},
                },
                ["document_id", "left_version", "right_version"],
            ),
            output_schema=read_result,
            required_permissions=frozenset({Permission.DOCUMENT_READ}),
            **read_common,
            **common,
        ),
        ToolDefinition(
            name="create_report_artifact",
            description="在受控对象存储中创建不可覆盖的报告产物。",
            input_schema=_object_schema(
                {
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "content": {"type": "string", "minLength": 1, "maxLength": 200000},
                    "format": {"enum": ["markdown", "json"]},
                },
                ["title", "content", "format"],
            ),
            output_schema=artifact_result,
            required_permissions=frozenset({Permission.ARTIFACT_WRITE}),
            **write_common,
            **common,
        ),
        ToolDefinition(
            name="create_work_item_draft",
            description="仅创建本地工作项草稿，不发送到外部系统。",
            input_schema=_object_schema(
                {
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "description": {"type": "string", "minLength": 1, "maxLength": 200000},
                    "owner_group": {"type": "string", "minLength": 1, "maxLength": 200},
                    "due_date": {"type": "string", "format": "date"},
                },
                ["title", "description"],
            ),
            output_schema=artifact_result,
            required_permissions=frozenset({Permission.ARTIFACT_WRITE}),
            **write_common,
            **common,
        ),
    )
