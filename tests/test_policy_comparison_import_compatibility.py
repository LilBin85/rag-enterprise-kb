"""策略比较技能迁移后的导入兼容性测试。"""

import rag_kb.policy_compare as legacy
import rag_kb.skills.policy_comparison as skill


def test_legacy_policy_comparison_symbols_are_skill_objects() -> None:
    """旧路径必须重导出技能实现本身，不能引入包装层。"""
    names = (
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
    )

    for name in names:
        assert getattr(legacy, name) is getattr(skill, name)
