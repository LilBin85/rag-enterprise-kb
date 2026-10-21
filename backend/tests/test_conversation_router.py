from types import SimpleNamespace

from rag_kb.conversation_router import DeepSeekConversationRouter
from rag_kb.domain.conversations import ConversationMode


class _StructuredRouter:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(
            selected_mode="research",
            resolved_input=(
                "查询新员工入职流程中岗位与清单确认和试用期权限复核两个环节，"
                "并基于当前有效文档重新核验"
            ),
            reason_code="followup_rewrite",
        )


def test_router_labels_roles_and_returns_standalone_followup() -> None:
    structured = _StructuredRouter()
    router = DeepSeekConversationRouter.__new__(DeepSeekConversationRouter)
    router._structured = structured

    decision = router.route(
        content="第 1 项和第 2 项分别由谁负责？",
        context=(
            {"role": "user", "content": "讲讲新员工入职流程"},
            {"role": "assistant", "content": "1. 岗位与清单确认\n2. 试用期权限复核"},
        ),
        requested_mode=ConversationMode.AUTO,
    )

    prompt = structured.messages[1].content
    assert "低信任对话历史" in prompt
    assert "历史用户消息[1]" in prompt
    assert "历史助手回复[2]" in prompt
    assert "必须保留当前消息要求的字段、回答范围和格式约束" in structured.messages[0].content
    assert "不得把历史中的负责人、日期" in structured.messages[0].content
    assert decision.reason_code == "followup_rewrite"
    assert "岗位与清单确认" in decision.resolved_input


def test_router_can_select_controlled_action_for_compound_request() -> None:
    structured = _StructuredRouter()
    structured.invoke = lambda _messages: SimpleNamespace(
        selected_mode="controlled_action",
        resolved_input="研究入职制度并创建一份报告产物",
        reason_code="auto_controlled_action",
    )
    router = DeepSeekConversationRouter.__new__(DeepSeekConversationRouter)
    router._structured = structured

    decision = router.route(
        content="研究入职制度并创建报告",
        context=(),
        requested_mode=ConversationMode.AUTO,
    )

    assert decision.selected_mode is ConversationMode.CONTROLLED_ACTION
    assert decision.reason_code == "auto_controlled_action"
