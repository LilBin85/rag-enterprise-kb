"""策略比较技能的历史导入兼容层。

真实实现位于 :mod:`rag_kb.skills.policy_comparison`。保留此模块仅为兼容既有
调用方、依赖覆盖和测试；各符号均为同一对象，不在这里包装或复制行为。
"""

from rag_kb.skills.policy_comparison import (
    ALLOWED_TOOL_NAMES,
    REFUSAL_MESSAGE,
    BoundPolicyTools,
    BoundToolModel,
    DeepSeekPolicyAgentModel,
    Evidence,
    PolicyAgentVerdict,
    PolicyComparisonService,
    PolicyToolError,
    ToolBindable,
    _evidence,
    _section_map,
    _SingleToolCallBoundModel,
    deterministic_diff,
)

__all__ = [
    "ALLOWED_TOOL_NAMES",
    "REFUSAL_MESSAGE",
    "BoundPolicyTools",
    "BoundToolModel",
    "DeepSeekPolicyAgentModel",
    "Evidence",
    "PolicyAgentVerdict",
    "PolicyComparisonService",
    "PolicyToolError",
    "ToolBindable",
    "_SingleToolCallBoundModel",
    "_evidence",
    "_section_map",
    "deterministic_diff",
]
