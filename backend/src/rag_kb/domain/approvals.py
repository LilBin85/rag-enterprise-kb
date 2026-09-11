"""批准绑定契约；任一绑定事实变化都必须使批准失效。"""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


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
    status: str | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = None
    checkpoint_id: str | None = None
    interrupt_id: str | None = None
    tool_call_id: str | None = None
    checkpoint_binding_hash: str | None = None
    rejected_by: str | None = None
    rejected_at: AwareDatetime | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_checkpoint_and_rejection_bindings(self) -> "ApprovalBinding":
        checkpoint_fields = (
            self.thread_id,
            self.checkpoint_ns,
            self.checkpoint_id,
            self.interrupt_id,
            self.tool_call_id,
            self.checkpoint_binding_hash,
        )
        if any(value is not None for value in checkpoint_fields) and not (
            self.thread_id
            and self.checkpoint_ns is not None
            and self.checkpoint_id
            and self.interrupt_id
            and self.tool_call_id
            and self.checkpoint_binding_hash
        ):
            raise ValueError("approval_checkpoint_binding_missing")
        if self.status == "rejected" and self.checkpoint_binding_hash and not (
            self.rejected_by and self.rejected_at and self.rejection_reason
        ):
            raise ValueError("approval_rejection_audit_missing")
        return self
