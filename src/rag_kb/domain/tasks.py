"""可暂停恢复的单智能体任务状态与结构化下一动作。"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


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


class TaskNode(StrEnum):
    LOAD_TASK = "load_task"
    CLASSIFY_GOAL = "classify_goal"
    CHECK_CLARIFICATION = "check_clarification"
    UPDATE_PLAN = "update_plan"
    SELECT_ACTION = "select_action"
    EXECUTE_READ_TOOL = "execute_read_tool"
    RECORD_EVIDENCE = "record_evidence"
    EVALUATE_EVIDENCE = "evaluate_evidence"
    AWAIT_APPROVAL = "await_approval"
    EXECUTE_APPROVED_ACTION = "execute_approved_action"
    VERIFY_ACTION = "verify_action"
    GENERATE_RESULT = "generate_result"
    TERMINAL = "terminal"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchAction(StrictModel):
    kind: Literal["search"] = "search"
    query: str = Field(min_length=1, max_length=4000)
    reason_code: str


class CallToolAction(StrictModel):
    kind: Literal["call_tool"] = "call_tool"
    tool_name: str
    arguments: dict[str, object]
    reason_code: str


class AskUserAction(StrictModel):
    kind: Literal["ask_user"] = "ask_user"
    question: str = Field(min_length=1, max_length=2000)
    required_fields: tuple[str, ...] = ()
    reason_code: str


class UpdatePlanAction(StrictModel):
    kind: Literal["update_plan"] = "update_plan"
    steps: tuple[str, ...]
    reason_code: str


class ProposeAction(StrictModel):
    kind: Literal["propose_action"] = "propose_action"
    tool_name: str
    arguments: dict[str, object]
    evidence_ids: tuple[str, ...]
    reason_code: str


class FinishAction(StrictModel):
    kind: Literal["finish"] = "finish"
    supported_claim_ids: tuple[str, ...]
    reason_code: str


class RefuseAction(StrictModel):
    kind: Literal["refuse"] = "refuse"
    refusal_code: str
    reason_code: str


NextAction = Annotated[
    SearchAction
    | CallToolAction
    | AskUserAction
    | UpdatePlanAction
    | ProposeAction
    | FinishAction
    | RefuseAction,
    Field(discriminator="kind"),
]


PersistableAction = Annotated[
    SearchAction | CallToolAction | ProposeAction,
    Field(discriminator="kind"),
]


class PendingActionBinding(StrictModel):
    """数据库内只保存的不可变动作对象绑定，不包含动作参数正文。"""

    storage_ref: str = Field(min_length=1, max_length=500)
    content_hash: str = Field(
        min_length=71,
        max_length=71,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    kind: Literal["search", "call_tool", "propose_action"]
    expires_at: AwareDatetime


class TaskBudgets(StrictModel):
    max_model_calls: int = Field(default=12, ge=0)
    max_tool_calls: int = Field(default=20, ge=0)
    max_replans: int = Field(default=4, ge=0)
    max_consecutive_failures: int = Field(default=3, ge=0)
    max_runtime_seconds: int = Field(default=600, ge=1)
    max_context_tokens: int = Field(default=32_000, ge=1)
    tool_timeout_seconds: int = Field(default=30, ge=1)
    max_no_new_evidence: int = Field(default=2, ge=1)


class TaskState(StrictModel):
    task_id: str
    auth_context_ref: str
    goal: str
    task_mode: TaskMode
    status: TaskStatus
    current_node: TaskNode = TaskNode.LOAD_TASK
    state_version: int = Field(default=1, ge=1)
    lease_generation: int = Field(default=0, ge=0)
    plan_revision: int = Field(default=0, ge=0)
    current_step: int = Field(default=0, ge=0)
    observations: tuple[dict[str, object], ...] = ()
    evidence_ids: tuple[str, ...] = ()
    pending_question: str | None = None
    pending_approval: str | None = None
    pending_action: PendingActionBinding | None = None
    artifacts: tuple[str, ...] = ()
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    replans: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    deadline: AwareDatetime | None = None
    token_budget: int = Field(default=32_000, ge=0)
    token_usage: int = Field(default=0, ge=0)
    last_error: str | None = None
    repeated_action_fingerprints: tuple[str, ...] = ()
    query_fingerprints: tuple[str, ...] = ()
    result_fingerprints: tuple[str, ...] = ()
    no_new_evidence_count: int = Field(default=0, ge=0)
    cancelled: bool = False

    @model_validator(mode="after")
    def validate_pause_and_cancel_bindings(self) -> "TaskState":
        if (self.status is TaskStatus.CANCELLED) != self.cancelled:
            raise ValueError("task_cancel_binding_invalid")
        if self.status is TaskStatus.WAITING_FOR_USER and not self.pending_question:
            raise ValueError("task_question_binding_missing")
        if self.status is TaskStatus.WAITING_FOR_APPROVAL and not self.pending_approval:
            raise ValueError("task_approval_binding_missing")
        if self.status is TaskStatus.WAITING_FOR_APPROVAL and (
            self.pending_action is None
            or self.pending_action.kind != "propose_action"
        ):
            raise ValueError("task_pending_action_binding_missing")
        if self.pending_approval is not None and (
            self.pending_action is None
            or self.pending_action.kind != "propose_action"
        ):
            raise ValueError("task_approval_action_binding_invalid")
        return self
