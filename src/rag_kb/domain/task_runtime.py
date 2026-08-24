"""显式任务图的纯领域策略：转换、预算与循环指纹。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Any

from rag_kb.domain.tasks import TaskBudgets, TaskNode, TaskState


class TaskRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class BudgetOperation(StrEnum):
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    REPLAN = "replan"
    CHECKPOINT = "checkpoint"


_ALLOWED_TRANSITIONS: dict[TaskNode, frozenset[TaskNode]] = {
    TaskNode.LOAD_TASK: frozenset({TaskNode.CLASSIFY_GOAL}),
    TaskNode.CLASSIFY_GOAL: frozenset(
        {TaskNode.CHECK_CLARIFICATION, TaskNode.UPDATE_PLAN, TaskNode.TERMINAL}
    ),
    TaskNode.CHECK_CLARIFICATION: frozenset({TaskNode.UPDATE_PLAN, TaskNode.TERMINAL}),
    TaskNode.UPDATE_PLAN: frozenset({TaskNode.SELECT_ACTION, TaskNode.TERMINAL}),
    TaskNode.SELECT_ACTION: frozenset(
        {
            TaskNode.EXECUTE_READ_TOOL,
            TaskNode.CHECK_CLARIFICATION,
            TaskNode.UPDATE_PLAN,
            TaskNode.AWAIT_APPROVAL,
            TaskNode.GENERATE_RESULT,
            TaskNode.TERMINAL,
        }
    ),
    TaskNode.EXECUTE_READ_TOOL: frozenset({TaskNode.RECORD_EVIDENCE, TaskNode.TERMINAL}),
    TaskNode.RECORD_EVIDENCE: frozenset({TaskNode.EVALUATE_EVIDENCE, TaskNode.TERMINAL}),
    TaskNode.EVALUATE_EVIDENCE: frozenset(
        {TaskNode.SELECT_ACTION, TaskNode.UPDATE_PLAN, TaskNode.GENERATE_RESULT, TaskNode.TERMINAL}
    ),
    TaskNode.AWAIT_APPROVAL: frozenset({TaskNode.EXECUTE_APPROVED_ACTION, TaskNode.TERMINAL}),
    TaskNode.EXECUTE_APPROVED_ACTION: frozenset({TaskNode.VERIFY_ACTION, TaskNode.TERMINAL}),
    TaskNode.VERIFY_ACTION: frozenset(
        {TaskNode.RECORD_EVIDENCE, TaskNode.GENERATE_RESULT, TaskNode.TERMINAL}
    ),
    TaskNode.GENERATE_RESULT: frozenset({TaskNode.TERMINAL}),
    TaskNode.TERMINAL: frozenset(),
}


def validate_transition(current: TaskNode, target: TaskNode) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise TaskRuntimeError("task_transition_invalid")


def canonical_fingerprint(kind: str, value: Any) -> str:
    if not kind:
        raise ValueError("fingerprint_kind_required")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(kind.encode('utf-8') + b'\0' + encoded).hexdigest()}"


_QUERY_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def query_fingerprint(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query).casefold().strip()
    normalized = _QUERY_PUNCTUATION.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", normalized)
    if not normalized:
        raise TaskRuntimeError("query_empty")
    return canonical_fingerprint("query", normalized)


def ensure_unique_fingerprint(
    fingerprint: str,
    previous: tuple[str, ...],
    *,
    duplicate_code: str,
) -> None:
    if fingerprint in previous:
        raise TaskRuntimeError(duplicate_code)


def check_budget(
    state: TaskState,
    budgets: TaskBudgets,
    *,
    operation: BudgetOperation,
    now: datetime,
) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("budget_clock_must_be_timezone_aware")
    if state.cancelled:
        raise TaskRuntimeError("task_cancelled")
    if state.deadline is not None and now >= state.deadline:
        raise TaskRuntimeError("task_deadline_expired")
    if state.token_usage >= min(state.token_budget, budgets.max_context_tokens):
        raise TaskRuntimeError("budget_context_tokens_exceeded")
    if state.consecutive_failures >= budgets.max_consecutive_failures:
        raise TaskRuntimeError("budget_consecutive_failures_exceeded")
    if state.no_new_evidence_count >= budgets.max_no_new_evidence:
        raise TaskRuntimeError("duplicate_no_new_evidence")
    if operation is BudgetOperation.MODEL_CALL and state.model_calls >= budgets.max_model_calls:
        raise TaskRuntimeError("budget_model_calls_exceeded")
    if operation is BudgetOperation.TOOL_CALL and state.tool_calls >= budgets.max_tool_calls:
        raise TaskRuntimeError("budget_tool_calls_exceeded")
    if operation is BudgetOperation.REPLAN and state.replans >= budgets.max_replans:
        raise TaskRuntimeError("budget_replans_exceeded")
