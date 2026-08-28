"""M2 审批 HTTP 契约；不承载 checkpoint 或 resume 数据。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ApprovalStatus = Literal["pending", "approved", "rejected", "expired", "executed"]


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    reason: str | None = Field(default=None, max_length=200)
    expected_task_state_version: int = Field(ge=1)


class ApprovalResponse(BaseModel):
    """审批者所需的安全摘要；原始参数和 graph 绑定字段永不输出。"""

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    task_id: str
    summary: str | None = None
    plan_revision: int
    plan_digest: str
    evidence_hash: str
    canonical_arguments_hash: str
    checkpoint_binding_hash: str | None = None
    tool_name: str
    tool_version: str
    expires_at: datetime
    status: ApprovalStatus
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejected_by: str | None = None
    rejected_at: datetime | None = None
    rejection_reason: str | None = None
