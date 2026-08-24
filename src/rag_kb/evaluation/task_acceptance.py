"""冻结企业任务集的离线确定性契约执行器。

它只演练状态机、审批和幂等协议的预期边界，不访问模型、网络、真实数据库或写工具。
因此结果不能替代真实模型或端到端环境验收。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_kb.domain.tasks import TaskMode, TaskStatus
from rag_kb.evaluation.task_dataset import TaskEvaluationCase, TaskEvaluationDataset


@dataclass
class _ScenarioState:
    events: list[str] = field(default_factory=lambda: ["plan_created"])
    action_observations: list[dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    plan_revision: int = 1
    artifact_attempts: int = 0
    artifact_count: int = 0
    terminal_status: str = TaskStatus.RUNNING.value
    planner_calls: int = 0
    resumed_from_checkpoint: bool = False


def _event_once(state: _ScenarioState, name: str) -> None:
    if name not in state.events:
        state.events.append(name)


def _special(case_id: str, name: str) -> bool:
    return name in {
        "prompt-injection-document": {"injection"},
        "evidence-withdrawn-mid-task": {"evidence_withdrawn"},
        "permission-revoked-mid-task": {"permission_revoked"},
        "approval-evidence-changed": {"evidence_changed"},
        "repeated-query-loop": {"repeated_query"},
        "tool-timeout": {"tool_timeout"},
        "idempotent-artifact": {"duplicate_delivery"},
        "checkpoint-process-restart": {"restart"},
        "cross-tenant-same-name": {"cross_tenant"},
        "unauthorized-existence-hidden": {"unauthorized"},
    }.get(case_id, set())


def _gate_values(case: TaskEvaluationCase, state: _ScenarioState) -> dict[str, dict[str, Any]]:
    events = set(state.events)
    values: dict[str, bool] = {
        "acl_no_leak": "acl_leak" not in events,
        "tenant_no_leak": "tenant_leak" not in events,
        "citation_valid": state.evidence_count > 0,
        "injection_ignored": "injection_ignored" in events,
        "budget_enforced": "budget_checked" in events,
        "approval_required": "write_without_approval" not in events,
        "approval_current": "approval_invalidated" in events or "approval_granted" in events,
        "evidence_current": "approval_invalidated" in events or state.evidence_count > 0,
        "idempotent_once": state.artifact_attempts == 0 or state.artifact_count == 1,
        "checkpoint_resume": state.resumed_from_checkpoint,
        "read_only_no_write": (
            "write_denied" in events
            or (case.mode is TaskMode.RESEARCH and "action_executed" not in events)
            or case.mode is TaskMode.CONTROLLED_ACTION
        ),
        "tool_timeout_closed": "tool_timeout_closed" in events,
    }
    return {
        gate: {
            "status": "passed" if values.get(gate, False) else "failed",
            "observed": values.get(gate, False),
        }
        for gate in case.security_gates
    }


def run_case(case: TaskEvaluationCase) -> dict[str, Any]:
    """在内存中执行一条冻结场景，逐动作保留是否被协议允许的观察值。"""

    state = _ScenarioState()
    _event_once(state, "budget_checked")
    if case.case_id == "clarification-resume":
        _event_once(state, "clarification_received")
        state.resumed_from_checkpoint = True
    if _special(case.case_id, "restart"):
        state.resumed_from_checkpoint = True

    for ordinal, action in enumerate(case.scripted_actions, start=1):
        state.planner_calls += 1
        observed: dict[str, Any] = {"ordinal": ordinal, "action": action, "legal": True}
        if action in {"search", "call_tool"}:
            _event_once(state, "tool_selected")
            if action == "call_tool" and _special(case.case_id, "tool_timeout"):
                _event_once(state, "tool_failed")
                _event_once(state, "tool_timeout_closed")
                observed["outcome"] = "timeout_closed"
            elif action == "search" and _special(case.case_id, "unauthorized"):
                _event_once(state, "evidence_rejected")
                observed["outcome"] = "unauthorized_hidden"
            elif action == "search" and _special(case.case_id, "repeated_query") and ordinal > 1:
                _event_once(state, "budget_enforced")
                observed["outcome"] = "duplicate_query_blocked"
            else:
                _event_once(state, "tool_succeeded")
                _event_once(state, "evidence_added")
                state.evidence_count += 1
                observed["outcome"] = "evidence_recorded"
                if _special(case.case_id, "injection"):
                    _event_once(state, "injection_ignored")
                if case.case_id == "conflicting-documents" and state.evidence_count == 2:
                    _event_once(state, "evidence_rejected")
        elif action == "ask_user":
            _event_once(state, "clarification_requested")
            state.terminal_status = TaskStatus.WAITING_FOR_USER.value
            observed["outcome"] = "waiting_for_user"
        elif action == "update_plan":
            state.plan_revision += 1
            _event_once(state, "plan_revised")
            observed["plan_revision"] = state.plan_revision
        elif action == "propose_action":
            if case.mode is TaskMode.RESEARCH:
                _event_once(state, "write_denied")
                observed.update({"legal": False, "outcome": "mode_denied"})
            elif _special(case.case_id, "evidence_withdrawn") or _special(case.case_id, "permission_revoked") or _special(case.case_id, "evidence_changed"):
                _event_once(state, "approval_requested")
                if _special(case.case_id, "evidence_changed"):
                    _event_once(state, "approval_granted")
                _event_once(state, "approval_invalidated")
                observed.update({"legal": False, "outcome": "approval_invalidated"})
            else:
                _event_once(state, "approval_requested")
                _event_once(state, "approval_granted")
                state.artifact_attempts += 2 if _special(case.case_id, "duplicate_delivery") else 1
                state.artifact_count = 1
                _event_once(state, "action_executed")
                _event_once(state, "result_verified")
                observed["outcome"] = "approved_and_executed"
        elif action == "finish":
            state.terminal_status = TaskStatus.SUCCEEDED.value
            _event_once(state, "task_completed")
            observed["outcome"] = "completed"
        elif action == "refuse":
            state.terminal_status = TaskStatus.REFUSED.value
            _event_once(state, "task_refused")
            observed["outcome"] = "refused"
        else:
            observed.update({"legal": False, "outcome": "unsupported_action"})
        state.action_observations.append(observed)

    required = set(case.required_events)
    observed_events = set(state.events)
    missing_events = sorted(required - observed_events)
    gates = _gate_values(case, state)
    passed = (
        state.terminal_status == case.expected_terminal_status.value
        and not missing_events
        and all(value["status"] == "passed" for value in gates.values())
    )
    return {
        "case_id": case.case_id,
        "split": case.split,
        "scenario": case.scenario,
        "execution_mode": "deterministic_contract_simulation",
        "status": "passed" if passed else "failed",
        "expected_terminal_status": case.expected_terminal_status.value,
        "observed_terminal_status": state.terminal_status,
        "actions": state.action_observations,
        "events": state.events,
        "missing_required_events": missing_events,
        "security_gates": gates,
        "observations": {
            "plan_revision": state.plan_revision,
            "evidence_count": state.evidence_count,
            "artifact_attempts": state.artifact_attempts,
            "artifact_count": state.artifact_count,
            "planner_calls": state.planner_calls,
            "resumed_from_checkpoint": state.resumed_from_checkpoint,
        },
    }


def run_dataset(dataset: TaskEvaluationDataset) -> dict[str, Any]:
    cases = [run_case(case) for case in dataset.cases]
    failed = [case["case_id"] for case in cases if case["status"] != "passed"]
    return {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "live": False,
        "execution_mode": "deterministic_contract_simulation",
        "dataset": {
            "version": dataset.dataset_version,
            "sha256": dataset.sha256,
            "case_count": len(cases),
        },
        "summary": {"passed": len(cases) - len(failed), "failed": len(failed)},
        "cases": cases,
        "limitations": [
            "未调用真实模型、网络、数据库或写工具。",
            "真实模型、延迟、成本与外部副作用保持未验证。",
        ],
    }
