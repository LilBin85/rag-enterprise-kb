"""任务 HTTP 契约；不暴露图节点或 checkpoint 载荷。"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.tasks import TaskMode, TaskStatus


class TaskCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_state_version: int = Field(ge=1)


class TaskResponse(BaseModel):
    """只返回公开业务生命周期和可展示中断摘要。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    conversation_id: str | None = None
    turn_id: str | None = None
    goal: str
    task_mode: TaskMode
    status: TaskStatus
    state_version: int
    interrupt_kind: str | None = None
    interrupt_summary: str | None = None
    result_available: bool
    cancelled: bool


class ResumeCommandResponse(BaseModel):
    """已耐久接受的恢复命令；不回显正文或 checkpoint binding。"""

    model_config = ConfigDict(extra="forbid")

    command_id: str
    status: Literal["accepted"] = "accepted"


class TaskEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    event_type: str
    stage: str
    status: str
    counts: dict[str, Any]
    token_usage: int | None
    error_code: str | None
    created_at: datetime
