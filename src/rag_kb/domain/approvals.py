"""批准绑定契约；任一绑定事实变化都必须使批准失效。"""

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ApprovalBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str
    task_id: str
    tenant_id: str
    plan_revision: int = Field(ge=1)
    plan_digest: str
    tool_name: str
    tool_version: str
    canonical_arguments_hash: str
    principal_hash: str
    evidence_hash: str
    expires_at: AwareDatetime
    approved_by: str | None = None
    approved_at: AwareDatetime | None = None
