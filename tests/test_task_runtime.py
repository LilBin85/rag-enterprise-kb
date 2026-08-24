"""Stage 4 任务运行时先行契约测试。

当前生产实现尚未提供 ``TaskRuntime`` 时，运行时用例会安全跳过；接口出现后
这些测试应直接成为阻断验收。测试不接真实模型，模型结果由 fake 提供。
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from rag_kb.domain.tasks import (
    AskUserAction,
    FinishAction,
    NextAction,
    RefuseAction,
    SearchAction,
    TaskBudgets,
    TaskMode,
    TaskNode,
    TaskState,
    TaskStatus,
)


def test_next_action_is_strict_discriminated_union() -> None:
    adapter = TypeAdapter(NextAction)
    assert isinstance(adapter.validate_python({"kind": "search", "query": "q", "reason_code": "r"}), SearchAction)
    assert isinstance(adapter.validate_python({"kind": "ask_user", "question": "q", "reason_code": "r"}), AskUserAction)
    assert isinstance(
        adapter.validate_python(
            {"kind": "finish", "supported_claim_ids": ("claim-1",), "reason_code": "done"}
        ),
        FinishAction,
    )
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "unknown", "reason_code": "bad"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "search", "query": "q", "reason_code": "r", "text": "正文"})


def test_domain_budget_and_aware_deadline_boundaries() -> None:
    with pytest.raises(ValidationError):
        TaskBudgets(max_runtime_seconds=0)
    with pytest.raises(ValidationError):
        TaskState(
            task_id="t",
            auth_context_ref="auth",
            goal="g",
            task_mode=TaskMode.RESEARCH,
            status=TaskStatus.RUNNING,
            deadline=datetime(2026, 1, 1),
        )
    state = TaskState(
        task_id="t",
        auth_context_ref="auth",
        goal="g",
        task_mode=TaskMode.RESEARCH,
        status=TaskStatus.RUNNING,
        current_node=TaskNode.SELECT_ACTION,
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    assert state.deadline is not None and state.deadline.tzinfo is not None


def _runtime_or_skip():
    module = pytest.importorskip("rag_kb.application.task_runtime")
    return module.TaskRuntime


class FakePlanner:
    def __init__(self, actions):
        self.actions = iter(actions)

    def next_action(self, state, observations=()):
        return next(self.actions)


class FakeRepository:
    """最小运行时端口：实现后应以真实 DatabaseTaskRepository 替换。"""

    def __init__(self, state):
        self.state = state
        self.events = []

    def load(self, task_id):
        return self.state

    def save(self, state, *, expected_state_version, event=None):
        assert state.state_version == expected_state_version + 1
        self.state = state
        if event is not None:
            self.events.append(event)
        return state


def _state(**changes):
    values = dict(
        task_id="task-1",
        auth_context_ref="auth-ref",
        goal="goal",
        task_mode=TaskMode.RESEARCH,
        status=TaskStatus.RUNNING,
        current_node=TaskNode.SELECT_ACTION,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    values.update(changes)
    return TaskState(**values)


def test_runtime_rejects_illegal_transition_and_terminal_actions() -> None:
    TaskRuntime = _runtime_or_skip()
    runtime = TaskRuntime(FakeRepository(_state()), FakePlanner([]))
    with pytest.raises(Exception, match="task_transition_invalid"):
        runtime.transition("task-1", TaskNode.RECORD_EVIDENCE)
    terminal = TaskRuntime(FakeRepository(_state(status=TaskStatus.SUCCEEDED)), FakePlanner([]))
    with pytest.raises(Exception, match="task_terminal_state"):
        terminal.step("task-1")


def test_runtime_persistent_budget_limits_fail_closed() -> None:
    TaskRuntime = _runtime_or_skip()
    with pytest.raises(Exception, match="budget_model_calls_exceeded"):
        TaskRuntime(
            FakeRepository(_state()), FakePlanner([]), budgets=TaskBudgets(max_model_calls=0)
        ).step("task-1")
    with pytest.raises(Exception, match="budget_tool_calls_exceeded"):
        TaskRuntime(
            FakeRepository(_state(current_node=TaskNode.EXECUTE_READ_TOOL)),
            FakePlanner([]),
            budgets=TaskBudgets(max_tool_calls=0),
        ).record_tool_result("task-1", result_reference="result-1")
    with pytest.raises(Exception, match="budget_replans_exceeded"):
        TaskRuntime(
            FakeRepository(_state()),
            FakePlanner([{"kind": "update_plan", "steps": ["one"], "reason_code": "r"}]),
            budgets=TaskBudgets(max_replans=0),
        ).step("task-1")
    with pytest.raises(Exception, match="budget_context_tokens_exceeded"):
        TaskRuntime(
            FakeRepository(_state(token_usage=1, token_budget=1)), FakePlanner([])
        ).step("task-1")


def test_runtime_deadline_duplicate_protection_and_no_new_evidence() -> None:
    TaskRuntime = _runtime_or_skip()
    expired = TaskRuntime(
        FakeRepository(_state(deadline=datetime.now(UTC) - timedelta(seconds=1))), FakePlanner([])
    )
    with pytest.raises(Exception, match="task_deadline_expired"):
        expired.step("task-1")
    from rag_kb.domain.task_runtime import canonical_fingerprint, query_fingerprint

    search = SearchAction(query="采购 审批", reason_code="need_evidence")
    action_fp = canonical_fingerprint("action", search.model_dump(mode="json"))
    with pytest.raises(Exception, match="duplicate_action"):
        TaskRuntime(
            FakeRepository(_state(repeated_action_fingerprints=(action_fp,))),
            FakePlanner([search]),
        ).step("task-1")
    with pytest.raises(Exception, match="duplicate_query"):
        TaskRuntime(
            FakeRepository(_state(query_fingerprints=(query_fingerprint("采购审批"),))),
            FakePlanner([search]),
        ).step("task-1")
    result_fp = canonical_fingerprint("result", "result-1")
    with pytest.raises(Exception, match="duplicate_result"):
        TaskRuntime(
            FakeRepository(
                _state(
                    current_node=TaskNode.EXECUTE_READ_TOOL,
                    result_fingerprints=(result_fp,),
                )
            ),
            FakePlanner([]),
        ).record_tool_result("task-1", result_reference="result-1")
    stagnant = TaskRuntime(
        FakeRepository(
            _state(current_node=TaskNode.EXECUTE_READ_TOOL, no_new_evidence_count=1)
        ),
        FakePlanner([]),
        budgets=TaskBudgets(max_no_new_evidence=2),
    )
    with pytest.raises(Exception, match="duplicate_no_new_evidence"):
        stagnant.record_tool_result("task-1", result_reference="new-result-no-evidence")


def test_runtime_pause_resume_cancel_finish_and_refuse() -> None:
    TaskRuntime = _runtime_or_skip()
    runtime = TaskRuntime(FakeRepository(_state()), FakePlanner([]))
    paused = runtime.ask_user("task-1", question="需要哪个部门？")
    assert paused.status is TaskStatus.WAITING_FOR_USER
    resumed = runtime.resume("task-1", {"department": "finance"})
    assert resumed.status is TaskStatus.RUNNING
    cancelled = runtime.cancel("task-1")
    assert cancelled.status is TaskStatus.CANCELLED
    with pytest.raises(Exception, match="task_cancelled"):
        runtime.finish("task-1", FinishAction(supported_claim_ids=(), reason_code="done"))
    refused = TaskRuntime(FakeRepository(_state()), FakePlanner([])).refuse(
        "task-1", RefuseAction(refusal_code="insufficient_evidence", reason_code="safe")
    )
    assert refused.status is TaskStatus.REFUSED


def test_model_failure_is_counted_before_call_and_persisted_terminal() -> None:
    TaskRuntime = _runtime_or_skip()
    repository = FakeRepository(_state())
    runtime = TaskRuntime(repository, FakePlanner([]))

    with pytest.raises(Exception, match="next_action_failed"):
        runtime.step("task-1")

    assert repository.state.status is TaskStatus.FAILED
    assert repository.state.current_node is TaskNode.TERMINAL
    assert repository.state.model_calls == 1
    assert repository.state.last_error == "next_action_failed"
