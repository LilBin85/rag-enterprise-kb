"""任务 HTTP 契约；不暴露图节点或 checkpoint 载荷。"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_kb.domain.tasks import TaskMode, TaskStatus


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=8_000)
    task_mode: TaskMode

    @field_validator("task_mode")
    @classmethod
    def require_public_mode(cls, value: TaskMode) -> TaskMode:
        if value not in {TaskMode.RESEARCH, TaskMode.CONTROLLED_ACTION}:
            raise ValueError("仅支持 research 或 controlled_action 任务")
        return value


class TaskContinueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=2_000)
    expected_state_version: int = Field(ge=1)


class TaskCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_state_version: int = Field(ge=1)


class TaskResponse(BaseModel):
    """只返回公开业务生命周期和可展示中断摘要。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
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


class TaskContinueResponse(TaskResponse):
    """continue 的兼容任务响应，额外表达命令已入队。"""

    resume_command: ResumeCommandResponse | None = None


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
