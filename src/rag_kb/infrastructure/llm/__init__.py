"""大模型适配器；模型只产生结构化建议，不获得工具或权限。"""

from rag_kb.infrastructure.llm.task_planner import DeepSeekTaskPlanner

__all__ = ["DeepSeekTaskPlanner"]
