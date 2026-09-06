"""统一会话的最小领域契约。

会话只保存用户与助手之间的业务对话；任务图状态仍唯一属于 LangGraph
checkpoint，任务的租约、审批和证据边界不在这里复制。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    DELETING = "deleting"
    DELETED = "deleted"


class ConversationTurnStatus(StrEnum):
    ROUTING = "routing"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversationMode(StrEnum):
    AUTO = "auto"
    ANSWER = "answer"
    RESEARCH = "research"
    CONTROLLED_ACTION = "controlled_action"


class ConversationSubmissionKind(StrEnum):
    NEW = "new"
    CLARIFICATION = "clarification"


class ConversationMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


ACTIVE_TURN_STATUSES = frozenset(
    {
        ConversationTurnStatus.ROUTING,
        ConversationTurnStatus.PLANNING,
        ConversationTurnStatus.RUNNING,
        ConversationTurnStatus.WAITING_FOR_USER,
        ConversationTurnStatus.WAITING_FOR_APPROVAL,
    }
)

TERMINAL_TURN_STATUSES = frozenset(
    {
        ConversationTurnStatus.SUCCEEDED,
        ConversationTurnStatus.REFUSED,
        ConversationTurnStatus.FAILED,
        ConversationTurnStatus.CANCELLED,
    }
)


class ConversationDomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationState(ConversationDomainModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=200)
    status: ConversationStatus
    state_version: int = Field(ge=1)
    summary: str | None = Field(default=None, max_length=12_000, repr=False)
    summary_through_sequence: int = Field(default=0, ge=0)
    memory_version: int = Field(default=0, ge=0)
    memory_hash: str | None = Field(default=None, max_length=80)
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleting_at: AwareDatetime | None = None
    deleted_at: AwareDatetime | None = None


class ConversationTurnState(ConversationDomainModel):
    turn_id: str = Field(min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=64)
    task_id: str | None = Field(default=None, max_length=64)
    status: ConversationTurnStatus
    requested_mode: ConversationMode
    selected_mode: ConversationMode | None = None
    state_version: int = Field(ge=1)
    context_hash: str | None = Field(default=None, max_length=80)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_mode_binding(self) -> "ConversationTurnState":
        if self.status is ConversationTurnStatus.ROUTING:
            if self.selected_mode is not None or self.task_id is not None:
                raise ValueError("conversation_turn_routing_binding_invalid")
        elif self.selected_mode is None:
            raise ValueError("conversation_turn_selected_mode_missing")
        return self


class ConversationMessageState(ConversationDomainModel):
    message_id: str = Field(min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=64)
    turn_id: str = Field(min_length=1, max_length=64)
    role: ConversationMessageRole
    content: str = Field(max_length=20_000, repr=False)
    task_id: str | None = Field(default=None, max_length=64)
    evidence_ids: tuple[str, ...] = ()
    redacted: bool = False
    created_at: AwareDatetime


def turn_status_for_task_status(task_status: str) -> ConversationTurnStatus:
    """把已有任务生命周期映射到会话轮次，不扩展任务状态机。"""
    return {
        "planning": ConversationTurnStatus.PLANNING,
        "running": ConversationTurnStatus.RUNNING,
        "waiting_for_user": ConversationTurnStatus.WAITING_FOR_USER,
        "waiting_for_approval": ConversationTurnStatus.WAITING_FOR_APPROVAL,
        "succeeded": ConversationTurnStatus.SUCCEEDED,
        "refused": ConversationTurnStatus.REFUSED,
        "cancelled": ConversationTurnStatus.CANCELLED,
        "failed": ConversationTurnStatus.FAILED,
        "unknown": ConversationTurnStatus.FAILED,
    }.get(task_status, ConversationTurnStatus.FAILED)


def aware_datetime(value: datetime) -> datetime:
    """领域输入不得静默接受 naive 时间；该 helper 供仓储转换使用。"""
    if value.tzinfo is None:
        raise ValueError("conversation_time_must_be_aware")
    return value
