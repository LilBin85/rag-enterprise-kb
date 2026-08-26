"""M1 Research Agent 的最小执行边界。"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import interrupt

from rag_kb.agent.result import TaskResultCandidate
from rag_kb.domain.fingerprints import canonical_fingerprint


class AgentGuardError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SingleBusinessToolCallMiddleware(AgentMiddleware):
    """每轮至多放行一个业务工具；Todo 工具不计入业务调用。"""

    def after_model(self, state: dict[str, Any], runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or ()
        if not messages:
            return None
        calls = getattr(messages[-1], "tool_calls", ()) or ()
        business_calls = [call for call in calls if call.get("name") != "write_todos"]
        if len(business_calls) > 1:
            raise AgentGuardError("agent_parallel_tool_call_denied")
        return None


class ClarificationInterruptMiddleware(AgentMiddleware):
    """把模型的澄清候选转为同一 checkpoint 中可恢复的 LangGraph interrupt。"""

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime) -> dict[str, Any] | None:
        raw = state.get("structured_response")
        if isinstance(raw, TaskResultCandidate):
            candidate = raw
        elif isinstance(raw, dict):
            try:
                candidate = TaskResultCandidate.model_validate(raw)
            except ValueError:
                return None
        else:
            return None
        if candidate.status != "needs_clarification":
            return None

        resume = interrupt(
            {"kind": "user_clarification", "question": candidate.question.strip()}
        )
        answer = resume.get("answer") if isinstance(resume, dict) else None
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 2_000:
            raise AgentGuardError("task_resume_answer_invalid")
        return {
            "messages": [HumanMessage(content=answer.strip())],
            "jump_to": "model",
        }


_CONSECUTIVE_TOOL_FAILURES = 3
_REPEATED_TOOL_CALLS = 2


def _is_failed_tool_message(message: object) -> bool:
    if not isinstance(message, ToolMessage):
        return False
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "failed"


class ExecutionGuardMiddleware(AgentMiddleware):
    """纯函数扫描消息历史，强制停止边界；checkpoint 恢复后重算同样触发。

    - 连续 >= 3 条 failed ToolMessage：agent_consecutive_tool_failures
    - 同一 (tool_name, args) 指纹重复 >= 2 次：agent_repeated_tool_call
    """

    def before_model(self, state: dict[str, Any], runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or ()
        failures = 0
        for message in reversed(messages):
            if not _is_failed_tool_message(message):
                break
            failures += 1
        if failures >= _CONSECUTIVE_TOOL_FAILURES:
            raise AgentGuardError("agent_consecutive_tool_failures")
        fingerprints: dict[str, int] = {}
        for message in messages:
            for call in getattr(message, "tool_calls", ()) or ():
                fingerprint = canonical_fingerprint(
                    "agent_tool_call_repeat",
                    {"name": call.get("name"), "args": call.get("args")},
                )
                count = fingerprints.get(fingerprint, 0) + 1
                if count >= _REPEATED_TOOL_CALLS:
                    raise AgentGuardError("agent_repeated_tool_call")
                fingerprints[fingerprint] = count
        return None
