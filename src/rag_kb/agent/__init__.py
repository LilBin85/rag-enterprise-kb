"""LangChain create_agent 唯一任务运行时（M1：4 个只读工具）。"""

from rag_kb.agent.context import AgentRunContext
from rag_kb.agent.result import TaskResult, TaskResultCandidate

__all__ = [
    "AgentRunContext",
    "TaskResult",
    "TaskResultCandidate",
    "build_research_agent",
    "build_research_tools",
]


def __getattr__(name: str) -> object:
    if name == "build_research_agent":
        from rag_kb.agent.factory import build_research_agent

        return build_research_agent
    if name == "build_research_tools":
        from rag_kb.agent.tools import build_research_tools

        return build_research_tools
    raise AttributeError(name)
