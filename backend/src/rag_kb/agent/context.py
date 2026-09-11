"""服务端拥有的 Agent 运行上下文；契约已迁至 application，此处保留 re-export。"""

from rag_kb.application.agent_contract import AgentRunContext

__all__ = ["AgentRunContext"]
