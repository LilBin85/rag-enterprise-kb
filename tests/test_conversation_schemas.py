import pytest
from pydantic import ValidationError

from rag_kb.web.schemas.conversations import (
    ConversationMessageClarificationRequest,
    ConversationMessageNewRequest,
)


def test_new_message_contract_is_strict_and_bounded() -> None:
    assert ConversationMessageNewRequest(kind="new", content="问题").mode == "auto"
    with pytest.raises(ValidationError):
        ConversationMessageNewRequest(kind="new", content="问题", unexpected=True)
    with pytest.raises(ValidationError):
        ConversationMessageNewRequest(kind="new", content="", mode="answer")


def test_clarification_requires_binding_and_forbids_mode() -> None:
    assert ConversationMessageClarificationRequest(
        kind="clarification", content="答案", turn_id="turn-1", task_id="task-1", expected_task_state_version=1
    ).task_id == "task-1"
    with pytest.raises(ValidationError):
        ConversationMessageClarificationRequest(kind="clarification", content="答案", turn_id="turn-1", task_id="task-1")
    with pytest.raises(ValidationError):
        ConversationMessageClarificationRequest(kind="clarification", content="答案", turn_id="turn-1", task_id="task-1", expected_task_state_version=1, mode="research")
