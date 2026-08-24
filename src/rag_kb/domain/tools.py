"""统一工具注册表的静态声明契约。"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.identity import Permission


class ToolMode(StrEnum):
    READ = "read"
    WRITE = "write"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalPolicy(StrEnum):
    NEVER = "never"
    ALWAYS = "always"
    POLICY = "policy"


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    required_permissions: frozenset[Permission]
    mode: ToolMode
    risk_level: RiskLevel
    timeout_seconds: int = Field(ge=1, le=300)
    idempotent: bool
    approval_policy: ApprovalPolicy
    result_verifier: str
    redaction_policy: str
