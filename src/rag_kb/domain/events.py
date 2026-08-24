"""不包含正文、提示词或隐藏思维过程的统一任务事件契约。"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class TaskEventType(StrEnum):
    TASK_CREATED = "task_created"
    CLARIFICATION_REQUESTED = "clarification_requested"
    CLARIFICATION_RECEIVED = "clarification_received"
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    TOOL_SELECTED = "tool_selected"
    TOOL_STARTED = "tool_started"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    EVIDENCE_ADDED = "evidence_added"
    EVIDENCE_REJECTED = "evidence_rejected"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_INVALIDATED = "approval_invalidated"
    ACTION_EXECUTED = "action_executed"
    RESULT_VERIFIED = "result_verified"
    TASK_COMPLETED = "task_completed"
    TASK_REFUSED = "task_refused"
    TASK_FAILED = "task_failed"


class TaskEvent(BaseModel):
    """长期事件只允许结构化元数据和计数，不提供任意正文扩展字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    event_type: TaskEventType
    trace_id: str
    run_id: str
    task_id: str
    tenant_ref: str
    stage: str
    status: str
    tool_name: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    duration_ms: int | None = Field(default=None, ge=0)
    token_usage: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
