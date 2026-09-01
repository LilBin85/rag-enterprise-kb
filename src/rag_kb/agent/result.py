"""任务结果契约 shim；定义已迁至 application，保留 re-export。

历史 checkpoint（MsgPack serde）以 ``rag_kb.agent.result.TaskResultCandidate``
等路径引用类型，re-export 保证旧数据可继续反序列化。
"""

from rag_kb.application.agent_contract import (
    AgentResultStore,
    TaskResultCandidate,
    build_controlled_task_result,
    build_task_result,
)
from rag_kb.application.ports.task_read import ArtifactSummary, TaskResult

__all__ = [
    "AgentResultStore",
    "ArtifactSummary",
    "TaskResult",
    "TaskResultCandidate",
    "build_controlled_task_result",
    "build_task_result",
]
