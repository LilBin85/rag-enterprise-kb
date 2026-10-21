"""M1 Research Agent 的最小执行边界。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

from rag_kb.agent.result import TaskResultCandidate
from rag_kb.application.agent_contract import CONTROLLED_WRITE_TOOL_NAMES
from rag_kb.agent.tools import RESEARCH_TOOL_NAMES
from rag_kb.domain.fingerprints import canonical_fingerprint


class AgentGuardError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_MAX_PARALLEL_READ_CALLS = 3
_READ_TOOL_NAMES = frozenset(RESEARCH_TOOL_NAMES)
_WRITE_TOOL_NAMES = frozenset(CONTROLLED_WRITE_TOOL_NAMES)


def _batch_error_message(call: dict[str, Any], code: str) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {
                "tool_name": call.get("name"),
                "status": "failed",
                "error_code": code,
                "retryable": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        tool_call_id=str(call.get("id") or "missing-tool-call-id"),
        name=call.get("name"),
        status="error",
    )


def _prior_batch_error_count(messages: tuple[object, ...] | list[object], code: str) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("error_code") == code:
            count += 1
    return count


@dataclass
class _BatchProgress:
    started_at: float
    total: int
    completed: int = 0
    succeeded: int = 0
    failed: int = 0


BatchObserver = Callable[[str, Any, dict[str, int]], None]


class BusinessToolBatchMiddleware(AgentMiddleware):
    """允许最多三个只读调用并行；所有写调用仍保持单调用审批路径。"""

    def __init__(self, observer: BatchObserver | None = None) -> None:
        self._observer = observer
        self._progress: dict[tuple[str, ...], _BatchProgress] = {}
        self._progress_lock = Lock()

    @staticmethod
    def _latest_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
        messages = state.get("messages") or ()
        latest = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        return list(latest.tool_calls or ()) if latest is not None else []

    @staticmethod
    def _batch_key(calls: list[dict[str, Any]], context: Any) -> tuple[str, ...]:
        return (str(getattr(context, "task_id", "")), *(str(call.get("id") or "") for call in calls))

    def _reject_batch(
        self,
        messages: tuple[object, ...] | list[object],
        calls: list[dict[str, Any]],
        code: str,
    ) -> dict[str, Any]:
        if _prior_batch_error_count(messages, code):
            raise AgentGuardError(code)
        return {"messages": [_batch_error_message(call, code) for call in calls]}

    def after_model(self, state: dict[str, Any], runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or ()
        calls = self._latest_calls(state)
        read_calls = [call for call in calls if call.get("name") in _READ_TOOL_NAMES]
        write_calls = [call for call in calls if call.get("name") in _WRITE_TOOL_NAMES]
        business_calls = [
            call for call in calls if call.get("name") in _READ_TOOL_NAMES | _WRITE_TOOL_NAMES
        ]
        if read_calls and write_calls:
            return self._reject_batch(messages, business_calls, "agent_read_write_batch_denied")
        if len(write_calls) > 1:
            return self._reject_batch(messages, write_calls, "agent_multiple_write_calls_denied")
        if len(read_calls) > _MAX_PARALLEL_READ_CALLS:
            return self._reject_batch(messages, read_calls, "agent_read_batch_limit_exceeded")
        fingerprints = {
            canonical_fingerprint(
                "agent_tool_call_batch",
                {"name": call.get("name"), "args": call.get("args")},
            )
            for call in read_calls
        }
        if len(fingerprints) != len(read_calls):
            return self._reject_batch(messages, read_calls, "agent_duplicate_batch_tool_call")
        completed_controlled_write = False
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name not in CONTROLLED_WRITE_TOOL_NAMES:
                continue
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                raise AgentGuardError("agent_controlled_write_result_invalid") from None
            if (
                not isinstance(payload, dict)
                or payload.get("tool_name") != message.name
                or payload.get("status") not in _CONTROLLED_TOOL_STATUSES
            ):
                raise AgentGuardError("agent_controlled_write_result_invalid")
            completed_controlled_write |= payload["status"] == "succeeded"
        if completed_controlled_write and any(
            call.get("name") in _WRITE_TOOL_NAMES for call in write_calls
        ):
            raise AgentGuardError("agent_second_controlled_write_denied")
        return None

    def wrap_tool_call(self, request, handler):
        calls = self._latest_calls(request.state)
        read_calls = [call for call in calls if call.get("name") in _READ_TOOL_NAMES]
        if len(read_calls) <= 1 or request.tool_call.get("name") not in _READ_TOOL_NAMES:
            return handler(request)
        key = self._batch_key(read_calls, request.runtime.context)
        with self._progress_lock:
            progress = self._progress.get(key)
            if progress is None:
                progress = _BatchProgress(started_at=monotonic(), total=len(read_calls))
                self._progress[key] = progress
                if self._observer is not None:
                    self._observer("tool_batch_started", request.runtime.context, {"total": progress.total})
        result = None
        failed = False
        try:
            result = handler(request)
            if isinstance(result, ToolMessage):
                try:
                    payload = json.loads(result.content)
                except (TypeError, ValueError):
                    payload = None
                failed = result.status == "error" or (
                    isinstance(payload, dict) and payload.get("status") != "succeeded"
                )
            return result
        except BaseException:
            failed = True
            raise
        finally:
            with self._progress_lock:
                progress.completed += 1
                progress.failed += int(failed)
                progress.succeeded += int(not failed)
                if progress.completed == progress.total:
                    self._progress.pop(key, None)
                    if self._observer is not None:
                        self._observer(
                            "tool_batch_completed",
                            request.runtime.context,
                            {
                                "total": progress.total,
                                "succeeded": progress.succeeded,
                                "failed": progress.failed,
                                "duration_ms": max(0, round((monotonic() - progress.started_at) * 1000)),
                            },
                        )


# 兼容旧导入名；行为已经升级为有界只读批次。
SingleBusinessToolCallMiddleware = BusinessToolBatchMiddleware


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
_BUSINESS_TOOL_NAMES = frozenset(RESEARCH_TOOL_NAMES)
_CONTROLLED_TOOL_STATUSES = frozenset({"succeeded", "failed", "unknown"})


def _business_tool_status(message: object) -> str | None:
    if not isinstance(message, ToolMessage):
        return None
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    tool_name = payload.get("tool_name")
    status = payload.get("status")
    if tool_name not in _BUSINESS_TOOL_NAMES or status not in _CONTROLLED_TOOL_STATUSES:
        return None
    return status


class ExecutionGuardMiddleware(AgentMiddleware):
    """纯函数扫描消息历史，强制停止边界；checkpoint 恢复后重算同样触发。

    - 连续 >= 3 条 failed ToolMessage：agent_consecutive_tool_failures
    - 同一 (tool_name, args) 指纹重复 >= 2 次：agent_repeated_tool_call
    """

    def before_model(self, state: dict[str, Any], runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or ()
        failures = 0
        for message in reversed(messages):
            status = _business_tool_status(message)
            if status is None:
                continue
            if status != "failed":
                break
            failures += 1
        if failures >= _CONSECUTIVE_TOOL_FAILURES:
            raise AgentGuardError("agent_consecutive_tool_failures")
        fingerprints: dict[str, int] = {}
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            message_fingerprints: set[str] = set()
            for call in getattr(message, "tool_calls", ()) or ():
                if not isinstance(call, dict) or call.get("name") not in _BUSINESS_TOOL_NAMES:
                    continue
                fingerprint = canonical_fingerprint(
                    "agent_tool_call_repeat",
                    {"name": call.get("name"), "args": call.get("args")},
                )
                if fingerprint in message_fingerprints:
                    continue
                message_fingerprints.add(fingerprint)
                count = fingerprints.get(fingerprint, 0) + 1
                if count >= _REPEATED_TOOL_CALLS:
                    raise AgentGuardError("agent_repeated_tool_call")
                fingerprints[fingerprint] = count
        return None
