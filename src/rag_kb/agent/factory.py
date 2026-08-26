"""唯一 create_agent 工厂；模型/工具预算由中间件按线程强制。"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from rag_kb.config import AGENT_MAX_MODEL_CALLS, AGENT_MAX_TOOL_CALLS
from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.middleware import (
    ClarificationInterruptMiddleware,
    ExecutionGuardMiddleware,
    SingleBusinessToolCallMiddleware,
)
from rag_kb.agent.prompt import TASK_AGENT_PROMPT
from rag_kb.agent.result import TaskResultCandidate


def build_research_agent(
    model: Any,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    max_model_calls: int = AGENT_MAX_MODEL_CALLS,
    max_tool_calls: int = AGENT_MAX_TOOL_CALLS,
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
