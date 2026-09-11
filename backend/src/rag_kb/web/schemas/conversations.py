"""统一会话 HTTP 契约；业务状态与持久化均由应用服务负责。"""

from datetime import datetime
from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationMessageNewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["new"]
    content: str = Field(min_length=1, max_length=8_000)
    mode: Literal["auto", "answer", "research", "controlled_action"] = "auto"
    scope_document_id: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationMessageClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["clarification"]
    content: str = Field(min_length=1, max_length=8_000)
    turn_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    expected_task_state_version: int = Field(ge=1)


ConversationMessageRequest = Annotated[
    ConversationMessageNewRequest | ConversationMessageClarificationRequest,
    Field(discriminator="kind"),
]


class ConversationSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    title: str | None
    state_version: int
    status: str
    created_at: datetime
    updated_at: datetime
    active_turn: dict[str, Any] | None = None


class ConversationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[ConversationSummaryResponse, ...]
    next_cursor: str | None = None


class UserMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    sequence: int


class ConversationTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str
    state_version: int
    result_available: bool


class ConversationTurnResponse(BaseModel):
    """提交消息后的稳定响应；不暴露内部执行载荷。"""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    conversation_state_version: int
    turn_id: str
    user_message: UserMessageResponse
    requested_mode: Literal["auto", "answer", "research", "controlled_action"]
    selected_mode: Literal["answer", "research", "controlled_action"]
    route_reason_code: str
    turn_status: str
    task: ConversationTaskResponse | None = None
    assistant_message: dict[str, Any] | None = None
    idempotent_replay: bool


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    turn_id: str | None = None
    task_id: str | None = None
    sequence: int
    role: str
    kind: str
    content: str | None = None
    visibility: str
    redaction_reason: str | None = None
    citations: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    created_at: datetime


class ConversationMessagesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[ConversationMessageResponse, ...]


class ConversationDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    status: Literal["deleting"] = "deleting"
