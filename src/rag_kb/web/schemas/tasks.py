"""通用任务 HTTP 契约；只暴露客户端完成协作所需的状态。"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.tasks import TaskMode, TaskNode, TaskStatus


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=8_000)
    task_mode: TaskMode


class TaskContinueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=2_000)
    expected_state_version: int = Field(ge=1)


class TaskCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_state_version: int = Field(ge=1)


class TaskApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1, max_length=64)


class TaskResponse(BaseModel):
    """刻意省略租约、认证快照和内部去重指纹。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    goal: str
    task_mode: TaskMode
    status: TaskStatus
    current_node: TaskNode
    state_version: int
    plan_revision: int
    current_step: int
    pending_question: str | None = None
    cancelled: bool


class TaskEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    event_type: str
    stage: str
    status: str
    counts: dict[str, Any]
    token_usage: int
    error_code: str | None
    created_at: datetime


class TaskArtifactResponse(BaseModel):
    """产物清单不泄露对象存储路径与内容摘要。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: str
    status: str
    created_at: datetime


class TaskApprovalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    task_id: str
    plan_revision: int
    status: str
    expires_at: datetime
    approved_at: datetime | None
