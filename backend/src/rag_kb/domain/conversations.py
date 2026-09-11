"""统一会话的最小领域契约。

会话只保存用户与助手之间的业务对话；任务图状态仍唯一属于 LangGraph
checkpoint，任务的租约、审批和证据边界不在这里复制。
"""

from enum import StrEnum


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
