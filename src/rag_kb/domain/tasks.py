"""通用任务的业务状态模型；Agent 执行状态唯一保存在 LangGraph checkpointer。"""

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from rag_kb.config import AGENT_MAX_MODEL_CALLS, AGENT_MAX_TOOL_CALLS


class TaskMode(StrEnum):
    KNOWLEDGE_QUERY = "knowledge_query"
    RESEARCH = "research"
    CONTROLLED_ACTION = "controlled_action"


class TaskStatus(StrEnum):
    WAITING_FOR_USER = "waiting_for_user"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


_INTERRUPT_STATUSES = frozenset(
    {TaskStatus.WAITING_FOR_USER, TaskStatus.WAITING_FOR_APPROVAL}
)
_RESULT_STATUSES = frozenset({TaskStatus.SUCCEEDED, TaskStatus.REFUSED})
_RESEARCH_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.REFUSED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.UNKNOWN,
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskBudgets(StrictModel):
    """业务级执行预算；默认值与 Settings/build_research_agent 同源（config 常量）。"""

    max_model_calls: int = Field(default=AGENT_MAX_MODEL_CALLS, ge=0)
    max_tool_calls: int = Field(default=AGENT_MAX_TOOL_CALLS, ge=0)
    max_runtime_seconds: int = Field(default=600, ge=1)
    max_context_tokens: int = Field(default=32_000, ge=1)
    tool_timeout_seconds: int = Field(default=30, ge=1)


class TaskState(StrictModel):
    """只承载业务生命周期、中断摘要与最终结果引用；不保存消息、计划或观察。"""

    task_id: str
    auth_context_ref: str
    goal: str
    task_mode: TaskMode
    status: TaskStatus
    state_version: int = Field(default=1, ge=1)
    lease_generation: int = Field(default=0, ge=0)
    interrupt_kind: str | None = None
    interrupt_summary: str | None = None
    result_kind: str | None = None
    result_ref: str | None = None
    result_hash: str | None = None
    content_expires_at: AwareDatetime | None = None
    result_purged_at: AwareDatetime | None = None
    deadline: AwareDatetime | None = None
    cancelled: bool = False

    @model_validator(mode="after")
    def validate_pause_and_cancel_bindings(self) -> "TaskState":
        if (self.status is TaskStatus.CANCELLED) != self.cancelled:
            raise ValueError("task_cancel_binding_invalid")
        if self.status in _INTERRUPT_STATUSES and not (
            self.interrupt_kind and self.interrupt_summary
        ):
            raise ValueError("task_interrupt_binding_missing")
        if self.status not in _INTERRUPT_STATUSES and (
            self.interrupt_kind or self.interrupt_summary
        ):
            raise ValueError("task_interrupt_only_when_waiting")
        if self.result_ref is not None and not self.result_kind:
            raise ValueError("task_result_kind_missing")
        if (
            self.task_mode is TaskMode.RESEARCH
            and self.status in _RESEARCH_TERMINAL_STATUSES
        ):
            if self.content_expires_at is None:
                raise ValueError("task_content_expiry_missing")
        if self.task_mode is TaskMode.RESEARCH and self.status in _RESULT_STATUSES:
            if self.result_hash is None or self.result_kind != "research":
                raise ValueError("task_result_binding_missing")
            if self.result_purged_at is None and self.result_ref is None:
                raise ValueError("task_result_binding_missing")
            if self.result_purged_at is not None and self.result_ref is not None:
                raise ValueError("task_purged_result_ref_present")
        return self

    def result_available(self, *, now: datetime) -> bool:
        """正文可用性独立于执行终态；调用方必须传入可信时间源。"""
        if now.tzinfo is None:
            raise ValueError("task_result_time_must_be_aware")
        return (
            self.status in _RESULT_STATUSES
            and bool(self.result_ref and self.result_hash)
            and self.result_purged_at is None
            and self.content_expires_at is not None
            and self.content_expires_at > now
        )
