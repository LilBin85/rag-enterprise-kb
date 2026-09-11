"""唯一 create_agent 工厂；模型/工具预算由中间件按线程强制。"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.middleware import (
    ClarificationInterruptMiddleware,
    ExecutionGuardMiddleware,
    SingleBusinessToolCallMiddleware,
)
from rag_kb.agent.prompt import CONTROLLED_ACTION_AGENT_PROMPT, TASK_AGENT_PROMPT
from rag_kb.agent.result import TaskResultCandidate
from rag_kb.agent.tools import CONTROLLED_WRITE_TOOL_NAMES


def build_research_agent(
    model: Any,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    max_model_calls: int,
    max_tool_calls: int,
) -> Any:
    """编译研究 Agent：结构化结果 + 线程短期记忆 + 调用限制，全部 fail-closed。"""
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=TASK_AGENT_PROMPT,
        middleware=[
            TodoListMiddleware(),
            SingleBusinessToolCallMiddleware(),
            ClarificationInterruptMiddleware(),
            ExecutionGuardMiddleware(),
            ModelCallLimitMiddleware(
                thread_limit=max_model_calls, exit_behavior="error"
            ),
            ToolCallLimitMiddleware(
                thread_limit=max_tool_calls, exit_behavior="error"
            ),
        ],
        # DeepSeek 当前不提供本项目所需的 native response_format；结果仍由工具调用校验。
        response_format=ToolStrategy(TaskResultCandidate),
        context_schema=AgentRunContext,
        checkpointer=checkpointer,
    )


def build_controlled_action_agent(
    model: Any,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    max_model_calls: int,
    max_tool_calls: int,
) -> Any:
    """编译受控动作 Agent：两个 WRITE 精确经 HITL 暂停，decision 只允许 approve/reject。

    interrupt_on 与 CONTROLLED_WRITE_TOOL_NAMES 逐字一致：不配置第三个 WRITE，
    不开放 edit/respond；权限、审批绑定与副作用资格始终由服务端决定。
    """
    interrupt_on = {
        tool_name: {"allowed_decisions": ["approve", "reject"]}
        for tool_name in CONTROLLED_WRITE_TOOL_NAMES
    }
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=CONTROLLED_ACTION_AGENT_PROMPT,
        middleware=[
            TodoListMiddleware(),
            HumanInTheLoopMiddleware(
                interrupt_on=interrupt_on,
                description_prefix="受控写操作等待人工审批",
            ),
            # LangChain 的 after_model 按声明逆序执行，故此守卫须放在 HITL 后。
            SingleBusinessToolCallMiddleware(),
            ClarificationInterruptMiddleware(),
            ExecutionGuardMiddleware(),
            ModelCallLimitMiddleware(
                thread_limit=max_model_calls, exit_behavior="error"
            ),
            ToolCallLimitMiddleware(
                thread_limit=max_tool_calls, exit_behavior="error"
            ),
        ],
        response_format=ToolStrategy(TaskResultCandidate),
        context_schema=AgentRunContext,
        checkpointer=checkpointer,
    )
