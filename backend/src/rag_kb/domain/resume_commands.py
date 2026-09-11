"""耐久恢复命令的最小领域契约。

命令正文只在 Worker 消费前保存在 PostgreSQL；这些类型刻意把正文、摘要和
不透明标识从 ``repr`` 中隐藏，调用方不得把它们写入日志或异常消息。
"""

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from rag_kb.domain.tasks import StrictModel


class ResumeCommandKind(StrEnum):
    USER_CLARIFICATION = "user_clarification"
    TOOL_APPROVAL = "tool_approval"


class ResumeCommandStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    CONSUMED = "consumed"
    FAILED = "failed"
    EXPIRED = "expired"


class ApprovalResumeDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


_TERMINAL_STATUSES = frozenset(
    {
        ResumeCommandStatus.CONSUMED,
        ResumeCommandStatus.FAILED,
        ResumeCommandStatus.EXPIRED,
    }
)


class ResumeCommand(StrictModel):
    """Worker 可读取的已验证命令；敏感值均不参与 repr。"""

    command_id: str = Field(min_length=1, max_length=64, repr=False)
    tenant_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=64)
    kind: ResumeCommandKind
    actor_principal_id: str = Field(min_length=1, max_length=200)
    actor_kind: str = Field(min_length=1, max_length=32)
    expected_task_state_version: int = Field(ge=1)
    # LangGraph 根 checkpoint 的 namespace 合法地使用空串；缺失仅能是 NULL，
    # 而不是把根 namespace 改写为 sentinel。
    expected_interrupt_checkpoint_ns: str = Field(max_length=200, repr=False)
    expected_interrupt_checkpoint_id: str = Field(min_length=1, max_length=150, repr=False)
    expected_interrupt_id: str = Field(min_length=1, max_length=100, repr=False)
    expected_protocol_version: str = Field(min_length=1, max_length=64)
    approval_id: str | None = Field(default=None, max_length=64, repr=False)
    decision: ApprovalResumeDecision | None = None
    payload_body: str | None = Field(default=None, repr=False)
    payload_sha256: str = Field(min_length=1, max_length=80, repr=False)
    resume_target_key: str = Field(min_length=1, max_length=80, repr=False)
    idempotency_key: str = Field(min_length=1, max_length=80, repr=False)
    status: ResumeCommandStatus
    attempt_count: int = Field(ge=0)
    next_attempt_at: AwareDatetime
    expires_at: AwareDatetime
    lease_owner: str | None = Field(default=None, max_length=200, repr=False)
    lease_generation: int = Field(ge=0)
    lease_expires_at: AwareDatetime | None = None
    claimed_at: AwareDatetime | None = None
    claimed_task_state_version: int | None = Field(default=None, ge=1)
    claimed_task_lease_generation: int | None = Field(default=None, ge=1)
    applied_checkpoint_id: str | None = Field(default=None, max_length=150, repr=False)
    applied_at: AwareDatetime | None = None
    consumed_at: AwareDatetime | None = None
    failed_at: AwareDatetime | None = None
    expired_at: AwareDatetime | None = None
    failure_code: str | None = Field(default=None, max_length=100)
    body_purged_at: AwareDatetime | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_bindings(self) -> "ResumeCommand":
        terminal = self.status in _TERMINAL_STATUSES
        if self.kind is ResumeCommandKind.USER_CLARIFICATION:
            if self.approval_id is not None or self.decision is not None:
                raise ValueError("resume_command_kind_binding_invalid")
            if not terminal and not (self.payload_body and self.payload_body.strip()):
                raise ValueError("resume_command_payload_missing")
        elif self.approval_id is None or self.decision is None:
            raise ValueError("resume_command_approval_binding_missing")
        if self.status is ResumeCommandStatus.PENDING:
            if any(
                value is not None
                for value in (
                    self.lease_owner,
                    self.lease_expires_at,
                    self.claimed_at,
                    self.claimed_task_state_version,
                    self.claimed_task_lease_generation,
                )
            ):
                raise ValueError("resume_command_pending_lease_present")
        elif self.status is ResumeCommandStatus.CLAIMED and not (
            self.lease_owner
            and self.lease_expires_at
            and self.claimed_at
            and self.claimed_task_state_version is not None
            and self.claimed_task_lease_generation is not None
        ):
            raise ValueError("resume_command_claim_lease_missing")
        if terminal != (self.body_purged_at is not None):
            raise ValueError("resume_command_body_purge_binding_invalid")
        if terminal and self.payload_body is not None:
            raise ValueError("resume_command_terminal_body_present")
        if (self.applied_checkpoint_id is None) != (self.applied_at is None):
            raise ValueError("resume_command_apply_binding_invalid")
        if self.expires_at <= self.created_at:
            raise ValueError("resume_command_expiry_invalid")
        return self


class ResumeCommandCandidate(StrictModel):
    """候选只含低敏元数据，供 Worker 后续按固定锁序重读。"""

    command_id: str = Field(min_length=1, max_length=64, repr=False)
    tenant_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=64)
    kind: ResumeCommandKind
    approval_id: str | None = Field(default=None, max_length=64, repr=False)
    status: ResumeCommandStatus
    eligible_at: AwareDatetime
