"""冻结 Planner Workflow-vs-Agent 评测；live 模式只在显式启用时调用模型。"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml
from jsonschema import validate
from rag_kb.models import PlanStatus, Role
from rag_kb.policy_rollout_planning import BoundPlanningTools, DeepSeekPlannerDecisionModel, GroundedPlanComposer, PlannerDecision, PlanningDraft, PolicyRolloutPlanningService, _todo_instruction_text
from rag_kb.retrieval_policy import QueryPolicy
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore
from rag_kb.retriever import RetrievalCandidate, RetrievalResult


METRICS = (
    "decision_path_correctness", "critical_clarification_recall",
    "invalid_or_over_clarification_rate", "initial_constraint_pass",
    "final_constraint_pass", "autonomous_repair_success", "citation_legality",
    "necessary_evidence_recall", "todo_grounding", "average_model_decisions",
    "average_search_calls", "search_overhead", "acl_leakage",
    "unexpected_budget_exhaustion", "refusal_correctness",
    "clarification_answer_utilization", "terminal_state_valid",
    "plan_constraint_pass", "task_outcome_correct",
    "clarification_question_quality", "todo_action_coverage",
    "todo_dependency_quality", "citation_support_alignment",
    # `decision_path_correctness` 作为旧字段保留；下面的 exact 字段明确它只
    # 是控制流诊断，semantic 字段才表示业务路径是否正确。
    "exact_control_trace_match", "semantic_business_path_correctness",
    "direct_proposal_success_rate", "grounded_composer_rate",
    "first_pass_plan_success_rate", "repair_rate", "emergency_fallback_rate",
)
LEGACY_METRICS = METRICS[:16]
DIAGNOSTIC_METRICS = frozenset({
    "decision_path_correctness", "exact_control_trace_match",
    "initial_constraint_pass", "autonomous_repair_success",
    "direct_proposal_success_rate", "grounded_composer_rate",
})
QUALITY_METRICS = tuple(
    metric for metric in METRICS
    if metric not in {
        "average_model_decisions", "average_search_calls", "search_overhead",
        *DIAGNOSTIC_METRICS,
    }
)
SCORE_METRICS = tuple(
    metric for metric in METRICS
    if metric not in {"average_model_decisions", "average_search_calls", "search_overhead"}
)
SMOKE_CASES = (
    "one-search-propose", "no-evidence-refusal", "acl-adversarial-refusal",
    "clarification-answer-used",
)
ALLOWLISTED_ERROR_CODES = frozenset({
    "agent_execution_failed", "retrieval_failed", "planning_failed", "planning_claim_lost",
    "task_persistence_failed", "invalid_model_output", "invalid_citations",
    "duplicate_todo_id", "invalid_dependencies", "cyclic_dependencies", "invalid_tool_call",
    "max_model_decisions_exceeded", "model_call_budget_reached",
    "benchmark_runtime_error", "quality_gate_failed",
    "max_searches_exceeded", "search_after_clarification_answers",
    "clarification_after_answers", "authorized_evidence_empty",
    "explicit_missing_business_variable", "business_input_is_not_evidence_query",
    "unsupported_todo_content", "incomplete_evidence_actions",
    "no_new_evidence", "semantically_duplicate_query", "query_rewrite_limit_exceeded",
    "search_after_terminal_evidence_state", "clarification_not_justified", "query_outside_goal_scope",
    "refusal_reconsideration_required",
    "json_truncated", "schema_validation_failed", "provider_timeout",
    "abnormal_finish_reason", "provider_empty_response", "adapter_protocol_error",
    "model_output_parse_failed", "provider_request_failed",
})
RUNTIME_ERROR_CODES = frozenset({
    "agent_execution_failed", "retrieval_failed", "planning_failed", "planning_claim_lost",
    "task_persistence_failed", "benchmark_runtime_error",
    "json_truncated", "schema_validation_failed", "provider_timeout",
    "abnormal_finish_reason", "provider_empty_response", "adapter_protocol_error",
    "model_output_parse_failed", "provider_request_failed",
})

LIVE_MODEL_QUALITY_THRESHOLDS = {
    "semantic_business_path_min": 0.90,
    # direct 与受约束 composer 都是正常的混合架构输出；紧急 fallback 不计入首稿。
    "first_pass_plan_success_min": 0.80,
    "repair_rate_max": 0.20,
    "emergency_fallback_rate_max": 0.10,
}

PROPOSAL_SOURCE_DIRECT = "direct_model_proposal"
PROPOSAL_SOURCE_COMPOSER = "grounded_composed_proposal"
PROPOSAL_SOURCE_FALLBACK = "emergency_grounded_fallback"
PROPOSAL_SOURCE_UNKNOWN = "unknown"
PROPOSAL_SOURCE_NOT_APPLICABLE = "not_applicable"

_COMPOSER_OUTCOMES = frozenset({
    "constrained_composer", "grounded_composed_proposal", "grounded_composer",
})
_COMPOSER_CODES = frozenset({
    "grounded_composer_after_validation", "grounded_composed_proposal",
})
_FALLBACK_OUTCOMES = frozenset({"grounded_fallback", "emergency_grounded_fallback"})


def ratio(values: Sequence[bool | int | float], average: bool = False, *, status: str | None = None):
    return {
        "value": sum(values) / len(values) if values and average else sum(bool(x) for x in values) / len(values) if values else None,
        "defined_n": len(values),
        "numerator": sum(values) if average else sum(bool(x) for x in values),
        "status": status or ("observed" if values else "not_observed"),
    }


class CaseRetriever:
    def __init__(self, routes, role):
        self.routes, self.role, self.calls = routes, role, []
        self.returned_chunk_ids: set[str] = set()

    def search(self, query, role):
        self.calls.append(query)
        normalized = re.sub(r"\s+", " ", query.strip().casefold()) if isinstance(query, str) else ""
        items = [
            item
            for route in self.routes.values()
            if normalized and any(
                re.sub(r"\s+", " ", alias.strip().casefold()) in normalized
                for alias in route["aliases"]
            )
            for item in route["candidates"]
        ]
        candidates = [
            RetrievalCandidate(
                item["chunk_id"], item["chunk_id"].split(":")[0], 2, "fixture",
                "fixture.md", item["locator"], 1.0, item["text"],
            )
            for item in items if str(role) in item["allowed_roles"]
        ]
        self.returned_chunk_ids.update(candidate.chunk_id for candidate in candidates)
        return RetrievalResult(candidates, 0)


class ScriptedModel:
    def __init__(self, script):
        self.script, self.index = script, 0

    def invoke(self, messages):
        step = self.script[self.index]
        self.index += 1
        human = json.loads(messages[-1].content)
        if "require_repair" in step:
            assert human["repair"]["constraint_codes"] == [step["require_repair"]]
        if "tool" in step:
            return PlannerDecision(action="search", search_query=step["tool"])
        draft = PlanningDraft.model_validate(step["draft"])
        if draft.action == "propose":
            return PlannerDecision(action="propose", proposal=draft)
        if draft.action == "clarify":
            return PlannerDecision(action="clarify", clarification_questions=draft.clarification_questions)
        return PlannerDecision(action="refuse", refusal_reason="scripted refusal")


class _ObservedModel:
    """只记录计数和 usage，不保留请求、回复、提示词或异常详情。"""

    def __init__(self, model, observation, remaining_calls):
        self._model, self._observation, self._remaining_calls = model, observation, remaining_calls

    def invoke(self, messages, *args, **kwargs):
        if self._remaining_calls[0] <= 0:
            self._observation["budget_exhausted"] = True
            raise RuntimeError("model_call_budget_reached")
        self._remaining_calls[0] -= 1
        self._observation["observed_model_calls"] += 1
        try:
            result = self._model.invoke(messages, *args, **kwargs)
        except Exception as exc:
            # 解析/Schema 失败也可能由适配器在安全异常属性上携带 usage；先读取，
            # 只有确实不存在时才算调用级缺失。
            _record_call_usage(exc, self._observation)
            raise
        _record_call_usage(result, self._observation)
        return result


class ObservedModel:
    def __init__(self, model, max_calls):
        self._model = model
        self._remaining_calls = [max_calls]
        self.observation = {
            "observed_model_calls": 0,
            "usage_observed_calls": 0,
            "usage_missing_calls": 0,
            "usage_partial_calls": 0,
            "budget_exhausted": False,
            "tokens": {"input": None, "output": None, "total": None},
        }

    def invoke(self, messages, *args, **kwargs):
        return _ObservedModel(self._model, self.observation, self._remaining_calls).invoke(messages, *args, **kwargs)


def _record_usage(message, totals):
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(message, "response_metadata", None)
        usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return "metadata_missing"
    values = {
        "input": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output": usage.get("output_tokens", usage.get("completion_tokens")),
        "total": usage.get("total_tokens"),
    }
    if values["total"] is None and all(
        isinstance(values[name], int) and not isinstance(values[name], bool) and values[name] >= 0
        for name in ("input", "output")
    ):
        values["total"] = values["input"] + values["output"]
    valid_fields = 0
    for key, value in values.items():
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            totals[key] = (totals[key] or 0) + value
            valid_fields += 1
    if valid_fields == 0:
        return "metadata_missing"
    return "observed" if valid_fields == len(values) else "partial_metadata"


def _record_call_usage(message, observation) -> None:
    usage_status = _record_usage(message, observation["tokens"])
    if usage_status == "metadata_missing":
        observation["usage_missing_calls"] += 1
        return
    observation["usage_observed_calls"] += 1
    if usage_status == "partial_metadata":
        observation["usage_partial_calls"] += 1


def action(status):
    return {
        PlanStatus.PLANNED: "propose", PlanStatus.CLARIFICATION: "clarify",
        PlanStatus.REFUSED: "refuse", PlanStatus.FAILED: "failed",
    }[status]


def plan_ids(response):
    return {
        todo.todo_id: [citation.chunk_id for citation in todo.citations]
        for plan in response.plan_versions for todo in plan.todos
    }


def plan_details(response):
    return {
        todo.todo_id: {
            "title": todo.title,
            "detail": todo.detail,
            "depends_on": list(todo.depends_on),
            "chunk_ids": [citation.chunk_id for citation in todo.citations],
        }
        for plan in response.plan_versions for todo in plan.todos
    }


def run_agent(case, routes, budgets):
    stale = [True]
    with tempfile.TemporaryDirectory() as root:
        store = PolicyRolloutTaskStore(Path(root) / "tasks.sqlite3")
        retriever = CaseRetriever(routes, case["role"])
        service = PolicyRolloutPlanningService(
            store, retriever, ScriptedModel(case["script"]),
            citation_checker=lambda _citations, _role: stale[0],
            max_tool_calls=budgets["max_search_calls"],
            max_model_decisions=budgets["max_model_decisions"],
            max_repairs=budgets["max_repairs"],
        )
        response = service.create(case["goal"], Role(case["role"]))
        if case.get("continuation_answers") and response.status == PlanStatus.CLARIFICATION:
            response = service.continue_task(response.task_id, Role(case["role"]), case["continuation_answers"]) or response
        trace = store.decision_trace(response.task_id)
        stale[0] = False if case.get("stale_check") else stale[0]
        stale_accessible = service.get(response.task_id, Role(case["role"])) is not None
    return observed(response, trace, retriever.calls, stale_accessible, retriever.returned_chunk_ids)


def run_workflow(case, routes, budgets):
    retriever = CaseRetriever(routes, case["role"])
    tools = BoundPlanningTools(retriever, Role(case["role"]), budgets["max_search_calls"])
    tools.search_authorized_evidence(case["workflow_query"])
    if not tools.candidates:
        return {
            "action": "refuse", "events": ["search", "refuse"], "model_decisions": 0,
            "semantic_business_path": ["search", "refuse"],
            "proposal_source": PROPOSAL_SOURCE_NOT_APPLICABLE,
            "used_grounded_fallback": False, "used_grounded_composer": False,
            "search_calls": len(retriever.calls), "repairs": 0, "todos": {},
            "citations": [], "stale_accessible": False,
            "attempted_searches": 1, "executed_searches": len(retriever.calls),
            "deduplicated_searches": 0, "rejected_searches": 0,
            "_retrieved_chunks": sorted(retriever.returned_chunk_ids),
            "_todo_details": {}, "_clarification_questions": [], "_clarification_answers": [],
        }
    # Workflow 不依赖模型自由文本；与生产安全边界一致，只从授权证据生成固定
    # Todo，避免基准路径绕过 composer 后与 Agent 走两套安全语义。
    chunk = next(iter(tools.candidates))
    draft = GroundedPlanComposer(QueryPolicy())._build_draft(
        [chunk], tools, case["goal"], case.get("continuation_answers", []),
    )
    plan = PolicyRolloutPlanningService._validated_plan(
        draft, tools, goal=case["goal"], answers=case.get("continuation_answers", []),
    )
    return {
        "action": "propose", "events": ["search", "propose"], "model_decisions": 0,
        "semantic_business_path": ["search", "propose"],
        "proposal_source": PROPOSAL_SOURCE_COMPOSER,
        "used_grounded_fallback": False, "used_grounded_composer": True,
        "search_calls": len(retriever.calls), "repairs": 0,
        "todos": {todo.todo_id: [citation.chunk_id for citation in todo.citations] for todo in plan.todos},
        "citations": [chunk], "stale_accessible": False,
        "attempted_searches": 1, "executed_searches": len(retriever.calls),
        "deduplicated_searches": 0, "rejected_searches": 0,
        "_retrieved_chunks": sorted(retriever.returned_chunk_ids),
        "_todo_details": {
            todo.todo_id: {
                "title": todo.title, "detail": todo.detail,
                "depends_on": list(todo.depends_on),
                "chunk_ids": [citation.chunk_id for citation in todo.citations],
            }
            for todo in plan.todos
        },
        "_clarification_questions": [], "_clarification_answers": [],
    }


def _append_stage(stages: list[str], stage: str) -> None:
    if not stages or stages[-1] != stage:
        stages.append(stage)


def _semantic_business_path(trace, terminal_action: str) -> list[str]:
    """将内部控制事件投影为业务阶段，不消费 accepted_paths 标签。

    被拒绝的模型动作、repair、composer/fallback 机制事件不会变成用户可见阶段；
    真正执行过的连续检索折叠为一个 search 阶段。
    """
    stages: list[str] = []
    executed_search_outcomes = {"success", "empty", "rehydrated", "no_new_evidence"}
    for event in trace:
        action_name, outcome = event.get("action"), event.get("outcome")
        if action_name == "search" and outcome in executed_search_outcomes:
            _append_stage(stages, "search")
        elif action_name == "clarify" and outcome == "clarification":
            _append_stage(stages, "clarify")
    _append_stage(stages, terminal_action)
    return stages


def _proposal_source(trace, terminal_action: str) -> str:
    """优先消费稳定 outcome/code；兼容早期 composer/fallback 命名。"""
    if terminal_action != "propose":
        return PROPOSAL_SOURCE_NOT_APPLICABLE
    outcomes = {str(event.get("outcome") or "") for event in trace}
    codes = {str(event.get("constraint_code") or "") for event in trace}
    if outcomes & _FALLBACK_OUTCOMES or any(code.startswith("grounded_fallback_") for code in codes):
        return PROPOSAL_SOURCE_FALLBACK
    if outcomes & _COMPOSER_OUTCOMES or codes & _COMPOSER_CODES:
        return PROPOSAL_SOURCE_COMPOSER
    # 新契约中的 direct 没有额外机制事件；旧报告也按这一规则兼容。
    return PROPOSAL_SOURCE_DIRECT


def _expected_semantic_business_path(case) -> list[str]:
    expected_action = case.get("expected_action", "failed" if case.get("expected_exhaustion") else "propose")
    stages: list[str] = []
    if case.get("continuation_answers") or expected_action == "clarify":
        _append_stage(stages, "clarify")
    if expected_action != "clarify" and case.get("workflow_query"):
        _append_stage(stages, "search")
    _append_stage(stages, expected_action)
    return stages


def observed(response, trace, calls, stale_accessible, retrieved_chunk_ids=()):
    todos = plan_ids(response)
    citations = [chunk for chunks in todos.values() for chunk in chunks]
    terminal = trace[-1] if trace else {"model_decisions": 0, "search_calls": len(calls), "repairs": 0}
    deduplicated = sum(event.get("action") == "search" and event.get("outcome") == "deduplicated" for event in trace)
    attempted = int(terminal["attempted_searches"]) if "attempted_searches" in terminal else int(terminal.get("search_calls", len(calls))) + deduplicated
    executed = int(terminal.get("executed_searches", len(calls)))
    rejected = int(terminal.get("rejected_searches", max(0, attempted - executed - deduplicated)))
    clarification = response.clarification
    terminal_action = action(response.status)
    proposal_source = _proposal_source(trace, terminal_action)
    return {
        "action": terminal_action, "events": [event["action"] for event in trace],
        "semantic_business_path": _semantic_business_path(trace, terminal_action),
        "proposal_source": proposal_source,
        "used_grounded_fallback": proposal_source == PROPOSAL_SOURCE_FALLBACK,
        "used_grounded_composer": proposal_source == PROPOSAL_SOURCE_COMPOSER,
        "constraint_codes": sorted({
            str(event.get("constraint_code"))
            for event in trace
            if event.get("constraint_code")
        }),
        "model_decisions": terminal["model_decisions"], "search_calls": terminal["search_calls"],
        "repairs": terminal["repairs"], "todos": todos, "citations": citations,
        "stale_accessible": stale_accessible,
        "attempted_searches": attempted, "executed_searches": executed,
        "deduplicated_searches": deduplicated, "rejected_searches": rejected,
        "_retrieved_chunks": sorted(set(retrieved_chunk_ids)),
        "_todo_details": plan_details(response),
        "_clarification_questions": list(clarification.questions) if clarification else [],
        "_clarification_answers": list(clarification.answers) if clarification else [],
        "_had_rejected_decision": any(
            event.get("outcome") == "rejected"
            and event.get("action") in {"propose", "clarify", "refuse"}
            for event in trace
        ),
    }


def _contains_terms(value: str, terms: Sequence[str]) -> bool:
    normalized = re.sub(r"\s+", "", value).casefold()
    return all(re.sub(r"\s+", "", term).casefold() in normalized for term in terms)


def _todo_quality(case, run):
    details = run.get("_todo_details", {})
    required = case.get("required_todos", [])
    if run["action"] != "propose":
        return None, None, None
    if not required:
        return False, False, False
    matches: dict[str, set[str]] = {}
    for spec in required:
        supporting = set(spec.get("supporting_chunks", []))
        candidates = {
            todo_id: todo
            for todo_id, todo in details.items()
            if todo.get("chunk_ids")
            and set(todo["chunk_ids"]).issubset(supporting)
        }
        combined = " ".join(
            _todo_instruction_text(todo["title"], todo["detail"])
            for todo in candidates.values()
        )
        if candidates and _contains_terms(combined, spec.get("required_terms", [])):
            matches[spec["key"]] = set(candidates)
    coverage = len(matches) == len(required)
    dependency_quality = bool(coverage)
    if dependency_quality:
        for spec in required:
            current_ids = matches[spec["key"]]
            for dependency_key in spec.get("depends_on", []):
                dependency_ids = matches.get(dependency_key, set())
                if not any(
                    dependency_id in dependency_ids
                    for todo_id in current_ids
                    for dependency_id in details[todo_id].get("depends_on", [])
                ):
                    dependency_quality = False
                    break
            if not dependency_quality:
                break
    support_alignment = bool(coverage) and all(
        all(
            bool(details[todo_id].get("chunk_ids"))
            and set(details[todo_id]["chunk_ids"]).issubset(set(spec.get("supporting_chunks", [])))
            for todo_id in matches[spec["key"]]
        )
        for spec in required
    )
    return coverage, dependency_quality, support_alignment


def _clarification_quality(case, run):
    missing = case.get("missing_business_variables", [])
    if not missing:
        return None
    questions = run.get("_clarification_questions", [])
    if not questions or run["events"].count("clarify") != 1:
        return False
    return all(
        any(any(_contains_terms(question, [cue]) for cue in variable["question_cues"]) for question in questions)
        for variable in missing
    )


def _metric_statuses(run, metrics):
    statuses = {}
    for name, value in metrics.items():
        if value is not None:
            statuses[name] = "observed"
        elif run["action"] == "failed":
            statuses[name] = "run_failed"
        else:
            statuses[name] = "not_applicable"
    return statuses


def score(case, run):
    gold, todos, path = set(case["gold_chunks"]), run["todos"], run["events"]
    propose = run["action"] == "propose"
    retrieved = set(run.get("_retrieved_chunks", []))
    legal = bool(run["citations"]) and set(run["citations"]).issubset(retrieved) if propose else None
    coverage, dependency_quality, support_alignment = _todo_quality(case, run)
    grounded = bool(legal and coverage and dependency_quality and support_alignment) if propose else None
    details = run.get("_todo_details", {})
    safe_blob = json.dumps({
        "events": path,
        "citations": run["citations"],
        "retrieved_chunk_ids": sorted(retrieved),
        "todos": todos,
    }, ensure_ascii=False)
    semantic_blob = " ".join(
        f"{todo['title']} {todo['detail']}" for todo in details.values()
    )
    clarification_quality = _clarification_quality(case, run)
    expected_action = case.get("expected_action", "failed" if case.get("expected_exhaustion") else "propose")
    exact_trace_match = path in case["accepted_paths"]
    semantic_path = run.get("semantic_business_path")
    if semantic_path is None:
        semantic_path = []
        for stage in path:
            if stage in {"search", "clarify"}:
                _append_stage(semantic_path, stage)
        _append_stage(semantic_path, run["action"])
        run["semantic_business_path"] = semantic_path
    proposal_source = run.get("proposal_source")
    if proposal_source is None:
        if run["action"] != "propose":
            proposal_source = PROPOSAL_SOURCE_NOT_APPLICABLE
        elif run.get("used_grounded_fallback"):
            proposal_source = PROPOSAL_SOURCE_FALLBACK
        elif run.get("used_grounded_composer"):
            proposal_source = PROPOSAL_SOURCE_COMPOSER
        else:
            proposal_source = PROPOSAL_SOURCE_DIRECT
        run["proposal_source"] = proposal_source
    proposal_quality_eligible = expected_action == "propose"
    clean_first_pass = bool(
        propose
        and run.get("repairs", 0) == 0
        and not run.get("_had_rejected_decision", False)
        and proposal_source != PROPOSAL_SOURCE_FALLBACK
    )
    terminal_valid = run["action"] == expected_action
    plan_constraint = bool(legal and grounded) if propose else None
    answer_utilization = None
    if case.get("answer_constraints"):
        answer_utilization = bool(
            propose
            and path.count("clarify") == 1
            and _contains_terms(semantic_blob, case["answer_constraints"])
        )
    if expected_action == "propose":
        task_outcome = bool(terminal_valid and plan_constraint)
    elif expected_action == "refuse":
        task_outcome = bool(terminal_valid and not any(marker in safe_blob for marker in case.get("forbidden_markers", [])))
    elif expected_action == "clarify":
        task_outcome = bool(terminal_valid and clarification_quality)
    else:
        task_outcome = bool(terminal_valid and (case.get("expected_exhaustion") or case.get("expected_failure")))
    metrics = {
        # 完整轨迹匹配仅用于诊断，不参与 live 质量通过判定。
        "decision_path_correctness": exact_trace_match,
        "critical_clarification_recall": "clarify" in path if case.get("missing_business_variables") else None,
        "invalid_or_over_clarification_rate": "clarify" in path and not case.get("missing_business_variables"),
        "initial_constraint_pass": "repair" not in path,
        "final_constraint_pass": plan_constraint,
        "autonomous_repair_success": propose if "repair" in path else None,
        "citation_legality": legal,
        "necessary_evidence_recall": gold.issubset(set(run["citations"])) if propose else None,
        "todo_grounding": grounded,
        "acl_leakage": any(marker in safe_blob for marker in case.get("forbidden_markers", [])),
        "unexpected_budget_exhaustion": run["action"] == "failed" and expected_action != "failed",
        "refusal_correctness": (run["action"] == "refuse") == (expected_action == "refuse"),
        "clarification_answer_utilization": answer_utilization,
        "terminal_state_valid": terminal_valid,
        "plan_constraint_pass": plan_constraint,
        "task_outcome_correct": task_outcome,
        "clarification_question_quality": clarification_quality,
        "todo_action_coverage": coverage,
        "todo_dependency_quality": dependency_quality,
        "citation_support_alignment": support_alignment,
        "exact_control_trace_match": exact_trace_match,
        "semantic_business_path_correctness": semantic_path == _expected_semantic_business_path(case),
        "direct_proposal_success_rate": (
            clean_first_pass and proposal_source == PROPOSAL_SOURCE_DIRECT
            if proposal_quality_eligible else None
        ),
        "grounded_composer_rate": (
            proposal_source == PROPOSAL_SOURCE_COMPOSER
            if proposal_quality_eligible else None
        ),
        "first_pass_plan_success_rate": clean_first_pass if proposal_quality_eligible else None,
        "repair_rate": run.get("repairs", 0) > 0 if proposal_quality_eligible else None,
        "emergency_fallback_rate": (
            proposal_source == PROPOSAL_SOURCE_FALLBACK
            if proposal_quality_eligible else None
        ),
    }
    run["metric_statuses"] = _metric_statuses(run, metrics)
    for private_name in (
        "_todo_details", "_clarification_questions", "_clarification_answers",
        "_retrieved_chunks", "_had_rejected_decision",
    ):
        run.pop(private_name, None)
    return metrics


def aggregate(runs):
    output = {}
    for name in METRICS:
        if name == "average_model_decisions":
            values, average = [run["model_decisions"] for run in runs], True
        elif name == "average_search_calls":
            values, average = [run["search_calls"] for run in runs], True
        elif name == "search_overhead":
            values, average = [max(0, run["search_calls"] - 1) for run in runs], True
        else:
            values, average = [run["metrics"][name] for run in runs if run["metrics"][name] is not None], False
        statuses = [run.get("metric_statuses", {}).get(name, "not_observed") for run in runs]
        aggregate_status = "observed" if values else next((status for status in statuses if status != "not_applicable"), "not_applicable")
        output[name] = ratio(values, average, status=aggregate_status)
    return output


def run_benchmark(manifest, manifest_name="data/benchmarks/planner_workflow_agent.yaml"):
    cases, workflow, agent = [], [], []
    for case in manifest["cases"]:
        w, a = run_workflow(case, manifest["routes"], manifest["budgets"]), run_agent(case, manifest["routes"], manifest["budgets"])
        w["metrics"], a["metrics"] = score(case, w), score(case, a)
        workflow.append(w)
        agent.append(a)
        cases.append({
            "name": case["name"], "workflow": w, "agent": a,
            "paired": {"same_inputs": True, "decision_delta": a["model_decisions"] - w["model_decisions"], "search_delta": a["search_calls"] - w["search_calls"]},
        })
    stale_runs = [run for case, run in zip(manifest["cases"], agent, strict=True) if case.get("stale_check")]
    gate_groups = _deterministic_gates(agent, stale_runs, manifest["budgets"])
    failures = [f"{group}.{name}" for group, checks in gate_groups.items() for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1, "benchmark": "planner-workflow-vs-agent", "manifest": manifest_name,
        "sample_size": len(cases),
        "metrics": {"workflow": aggregate(workflow), "agent": aggregate(agent), "paired": {"decision_delta": ratio([case["paired"]["decision_delta"] for case in cases], True), "search_delta": ratio([case["paired"]["search_delta"] for case in cases], True)}},
        "safety_gates": _gate_report(gate_groups["safety_gates"]),
        "quality_gates": _gate_report(gate_groups["quality_gates"]),
        "cost_gates": _gate_report(gate_groups["cost_gates"]),
        "gates": {"passed": not failures, "failures": failures}, "cases": cases,
    }


def _gate_report(checks):
    failures = [name for name, passed in checks.items() if not passed]
    return {"passed": not failures, "failures": failures, "checks": checks}


def _deterministic_gates(agent, stale_runs, budgets):
    return {
        "safety_gates": {
            "acl_leak": all(not run["metrics"]["acl_leakage"] for run in agent),
            "illegal_citation": all(run["metrics"]["citation_legality"] is not False for run in agent),
            "stale_evidence_accessibility": bool(stale_runs) and all(not run["stale_accessible"] for run in stale_runs),
        },
        "quality_gates": {
            "terminal_state_valid": all(run["metrics"]["terminal_state_valid"] for run in agent),
            "task_outcome_correct": all(run["metrics"]["task_outcome_correct"] for run in agent),
            "plan_constraint_pass": all(run["metrics"]["plan_constraint_pass"] is not False for run in agent),
        },
        "cost_gates": {
            "search_budget": all(run["executed_searches"] <= budgets["max_search_calls"] for run in agent),
            "unexpected_exhaustion": all(not run["metrics"]["unexpected_budget_exhaustion"] for run in agent),
        },
    }


def _sanitize_error_code(value):
    return value if value in ALLOWLISTED_ERROR_CODES else "planning_failed"


def _empty_metrics(status="not_observed"):
    return ({name: None for name in SCORE_METRICS}, {name: status for name in SCORE_METRICS})


def _run_live_agent(
    case, routes, budgets, model_factory, repetition, remaining_model_calls,
    *, preflight_named_policy_search=False,
):
    """每次 repetition 都使用独立 SQLite、Retriever 和 Planner 服务。"""
    started = time.perf_counter()
    observation = {
        "observed_model_calls": 0,
        "usage_observed_calls": 0,
        "usage_missing_calls": 0,
        "usage_partial_calls": 0,
        "budget_exhausted": False,
        "tokens": {"input": None, "output": None, "total": None},
    }
    observed_model = None
    empty_metrics, empty_statuses = _empty_metrics("run_failed")
    base = {
        "case_name": case["name"], "repetition": repetition, "action": "failed", "events": [], "model_decisions": 0,
        "semantic_business_path": ["failed"],
        "proposal_source": PROPOSAL_SOURCE_NOT_APPLICABLE,
        "used_grounded_fallback": False, "used_grounded_composer": False,
        "observed_model_calls": 0, "search_calls": 0, "attempted_searches": 0,
        "executed_searches": 0, "deduplicated_searches": 0, "rejected_searches": 0,
        "repairs": 0, "todos": {},
        "citations": [], "stale_accessible": False, "metrics": empty_metrics,
        "metric_statuses": empty_statuses,
        "duration_ms": 0.0, "status": "error", "error_code": "benchmark_runtime_error",
        "tokens": {
            "input": None, "output": None, "total": None,
            "status": "not_applicable", "reason": "no_model_call",
            "usage_observed_calls": 0, "usage_missing_calls": 0, "usage_partial_calls": 0,
        },
    }
    try:
        model = model_factory(case, repetition)
        observed_model = ObservedModel(model, remaining_model_calls)
        stale = [True]
        with tempfile.TemporaryDirectory() as root:
            store = PolicyRolloutTaskStore(Path(root) / "tasks.sqlite3")
            retriever = CaseRetriever(routes, case["role"])
            service = PolicyRolloutPlanningService(
                store, retriever, observed_model,
                citation_checker=lambda _citations, _role: stale[0],
                max_tool_calls=budgets["max_search_calls"],
                max_model_decisions=budgets["max_model_decisions"],
                max_repairs=budgets["max_repairs"],
                preflight_named_policy_search=preflight_named_policy_search,
            )
            response = service.create(case["goal"], Role(case["role"]))
            if case.get("continuation_answers") and response.status == PlanStatus.CLARIFICATION:
                response = service.continue_task(response.task_id, Role(case["role"]), case["continuation_answers"]) or response
            trace = store.decision_trace(response.task_id)
            stale[0] = False if case.get("stale_check") else stale[0]
            result = observed(
                response, trace, retriever.calls,
                service.get(response.task_id, Role(case["role"])) is not None,
                retriever.returned_chunk_ids,
            )
        result["metrics"] = score(case, result)
        result.update({
            "case_name": case["name"],
            "repetition": repetition,
            "observed_model_calls": observed_model.observation["observed_model_calls"],
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "tokens": _token_record(observed_model.observation),
        })
        trace_code = trace[-1].get("constraint_code") if trace else None
        if observed_model.observation["budget_exhausted"]:
            result.update(status="fail", error_code="model_call_budget_reached")
        elif result["action"] == "failed" and trace_code in RUNTIME_ERROR_CODES:
            result.update(status="error", error_code=_sanitize_error_code(trace_code))
        elif _live_quality_pass(result):
            result.update(status="pass", error_code=None)
        else:
            result.update(status="fail", error_code=_sanitize_error_code(trace_code) if result["action"] == "failed" else "quality_gate_failed")
        return result
    except Exception:
        base.update({
            "observed_model_calls": observed_model.observation["observed_model_calls"] if observed_model else 0,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "tokens": _token_record(observed_model.observation if observed_model else observation),
        })
        return base


def _live_quality_pass(run):
    metrics = run["metrics"]
    return bool(
        metrics["task_outcome_correct"]
        and metrics["terminal_state_valid"]
        and metrics["semantic_business_path_correctness"]
        and not metrics["acl_leakage"]
        and not metrics["unexpected_budget_exhaustion"]
        and metrics["citation_legality"] is not False
    )


def _metric_rate(runs, name: str) -> float | None:
    values = [run["metrics"].get(name) for run in runs if run["metrics"].get(name) is not None]
    return sum(bool(value) for value in values) / len(values) if values else None


def _live_model_quality_gate_checks(business_runs) -> dict[str, bool]:
    """模型质量门只消费 live_eligible 业务样本，不纳入确定性故障注入。"""
    semantic_rate = _metric_rate(business_runs, "semantic_business_path_correctness")
    first_pass_rate = _metric_rate(business_runs, "first_pass_plan_success_rate")
    repair_rate = _metric_rate(business_runs, "repair_rate")
    fallback_rate = _metric_rate(business_runs, "emergency_fallback_rate")
    return {
        "semantic_business_path": (
            semantic_rate is None
            or semantic_rate >= LIVE_MODEL_QUALITY_THRESHOLDS["semantic_business_path_min"]
        ),
        # direct 与受约束 composer 都可通过首稿门；emergency fallback 在 score
        # 中被明确排除，并由独立上限约束。
        "first_pass_plan_success": (
            first_pass_rate is None
            or first_pass_rate >= LIVE_MODEL_QUALITY_THRESHOLDS["first_pass_plan_success_min"]
        ),
        "repair_rate": (
            repair_rate is None
            or repair_rate <= LIVE_MODEL_QUALITY_THRESHOLDS["repair_rate_max"]
        ),
        "emergency_fallback_rate": (
            fallback_rate is None
            or fallback_rate <= LIVE_MODEL_QUALITY_THRESHOLDS["emergency_fallback_rate_max"]
        ),
    }


def _skipped_live_run(case_name, repetition):
    metrics, statuses = _empty_metrics("skipped")
    return {
        "case_name": case_name, "repetition": repetition, "status": "skipped", "action": "failed", "events": [],
        "semantic_business_path": ["failed"],
        "proposal_source": PROPOSAL_SOURCE_NOT_APPLICABLE,
        "used_grounded_fallback": False, "used_grounded_composer": False,
        "model_decisions": 0, "observed_model_calls": 0, "search_calls": 0,
        "attempted_searches": 0, "executed_searches": 0,
        "deduplicated_searches": 0, "rejected_searches": 0, "repairs": 0,
        "todos": {}, "citations": [], "stale_accessible": False,
        "metrics": metrics, "metric_statuses": statuses,
        "duration_ms": 0.0, "error_code": "model_call_budget_reached",
        "tokens": {
            "input": None, "output": None, "total": None, "status": "skipped",
            "usage_observed_calls": 0, "usage_missing_calls": 0, "usage_partial_calls": 0,
        },
    }


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def _token_record(observation, status=None):
    tokens = observation.get("tokens", {})
    output = {name: tokens.get(name) for name in ("input", "output", "total")}
    model_calls = int(observation.get("observed_model_calls", 0))
    usage_observed = int(observation.get("usage_observed_calls", 0))
    usage_missing = int(observation.get("usage_missing_calls", max(0, model_calls - usage_observed)))
    usage_partial = int(observation.get("usage_partial_calls", 0))
    if status is not None:
        token_status = status
    elif model_calls == 0:
        token_status = "not_applicable"
        output["reason"] = "no_model_call"
    elif usage_observed == 0:
        token_status = "metadata_missing"
    elif usage_missing > 0 or usage_partial > 0:
        token_status = "partial_metadata"
    else:
        token_status = "observed"
    output.update({
        "status": token_status,
        "usage_observed_calls": usage_observed,
        "usage_missing_calls": usage_missing,
        "usage_partial_calls": usage_partial,
    })
    return output


def _token_summary(runs):
    total_known = [run for run in runs if run["tokens"]["total"] is not None]
    model_call_runs = [run for run in runs if run.get("observed_model_calls", 0) > 0]
    no_model_call_runs = [run for run in runs if run.get("observed_model_calls", 0) == 0]
    missing_runs = [run for run in model_call_runs if run["tokens"].get("status") == "metadata_missing"]
    partial_runs = [run for run in model_call_runs if run["tokens"].get("status") == "partial_metadata"]
    usage_observed_calls = sum(int(run["tokens"].get("usage_observed_calls", 0)) for run in runs)
    usage_missing_calls = sum(int(run["tokens"].get("usage_missing_calls", 0)) for run in runs)
    usage_partial_calls = sum(int(run["tokens"].get("usage_partial_calls", 0)) for run in runs)
    def total(name):
        values = [run["tokens"][name] for run in runs if run["tokens"][name] is not None]
        return sum(values) if values else None
    def average(name):
        values = [run["tokens"][name] for run in runs if run["tokens"][name] is not None]
        return round(sum(values) / len(values), 3) if values else None
    totals = {name: total(name) for name in ("input", "output", "total")}
    if not model_call_runs:
        summary_status = "not_applicable"
    elif missing_runs and not total_known:
        summary_status = "metadata_missing"
    elif missing_runs or partial_runs:
        summary_status = "partial_metadata"
    else:
        summary_status = "observed"
    return {
        **totals,
        "average": {"input": average("input"), "output": average("output"), "total": average("total")},
        # 兼容字段：observed_runs 仍表示至少有 total token 的运行；missing_runs
        # 仅表示真实发生模型调用但完全缺失 usage 的运行，不再包含零调用任务。
        "observed_runs": len(total_known), "missing_runs": len(missing_runs),
        "partial_runs": len(partial_runs), "model_call_runs": len(model_call_runs),
        "no_model_call_runs": len(no_model_call_runs),
        "usage_observed_calls": usage_observed_calls,
        "usage_missing_calls": usage_missing_calls,
        "usage_partial_calls": usage_partial_calls,
        "status": summary_status,
    }


def _token_distribution(runs, name):
    values = [run["tokens"][name] for run in runs if run["tokens"][name] is not None]
    model_calls = sum(run.get("observed_model_calls", 0) for run in runs)
    return {
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "status": "observed" if values else ("metadata_missing" if model_calls else "not_applicable"),
    }


def aggregate_live_performance(runs):
    actual = [run for run in runs if run["status"] != "skipped"]
    durations = [run["duration_ms"] for run in actual]
    successful = [run for run in actual if run["metrics"].get("task_outcome_correct") is True]
    failed = [run for run in actual if run not in successful]
    tokens = _token_summary(actual)
    observed_calls = sum(run["observed_model_calls"] for run in actual)
    total_tokens = tokens["total"]
    complete_usage_calls = tokens["usage_observed_calls"] - tokens["usage_partial_calls"]
    tokens.update({
        # 调用级缺失率避免把服务端零模型调用的正确拒绝误记为 Provider 元数据缺失。
        "missing_rate": (
            tokens["usage_missing_calls"] / observed_calls if observed_calls else None
        ),
        "call_metadata_complete_rate": (
            complete_usage_calls / observed_calls if observed_calls else None
        ),
        "call_metadata_partial_rate": (
            tokens["usage_partial_calls"] / observed_calls if observed_calls else None
        ),
        "call_metadata_missing_rate": (
            tokens["usage_missing_calls"] / observed_calls if observed_calls else None
        ),
        "run_missing_rate": (
            (tokens["missing_runs"] + tokens["partial_runs"]) / tokens["model_call_runs"]
            if tokens["model_call_runs"] else None
        ),
        "no_model_call_rate": tokens["no_model_call_runs"] / len(actual) if actual else None,
        "average_per_business_run": {
            "input": round(tokens["input"] / len(actual), 3) if tokens["input"] is not None and actual else None,
            "output": round(tokens["output"] / len(actual), 3) if tokens["output"] is not None and actual else None,
            "total": round(total_tokens / len(actual), 3) if total_tokens is not None and actual else None,
        },
        "average_per_model_run": tokens["average"],
        "successful": _token_summary(successful),
        "failed_or_error": _token_summary(failed),
        "per_correct_plan": _token_summary([run for run in successful if run["action"] == "propose"])["average"],
        "distributions": {
            "input": _token_distribution(actual, "input"),
            "output": _token_distribution(actual, "output"),
            "total": _token_distribution(actual, "total"),
        },
        "provider_retries_observed": False,
    })
    return {
        "latency_ms": {"p50": _percentile(durations, 0.5), "p95": _percentile(durations, 0.95), "max": max(durations) if durations else None},
        "latency_scope": "end_to_end_planner_request_not_pure_model_inference",
        "latency_sample_size": len(durations),
        "total_model_decisions": sum(run["observed_model_calls"] for run in actual),
        "total_search_calls": sum(run["executed_searches"] for run in actual),
        "searches": {
            "attempted": sum(run["attempted_searches"] for run in actual),
            "executed": sum(run["executed_searches"] for run in actual),
            "deduplicated": sum(run["deduplicated_searches"] for run in actual),
            "rejected": sum(run["rejected_searches"] for run in actual),
        },
        "repair_count": sum(run["repairs"] for run in actual),
        "runtime_errors": sum(run["status"] == "error" for run in actual),
        "tokens": tokens,
        "budget_scope": "observed_model_decisions_provider_retries_not_observed",
    }


def _metric_deltas(left, right, names):
    return {
        name: None if left.get(name, {}).get("value") is None or right.get(name, {}).get("value") is None else left[name]["value"] - right[name]["value"]
        for name in names
    }


def _metric_delta_statuses(left, right, names):
    return {
        name: "observed" if left.get(name, {}).get("value") is not None and right.get(name, {}).get("value") is not None else "not_observed"
        for name in names
    }


def _select_live_cases(manifest, case_names=None, smoke=False):
    selected_names = SMOKE_CASES if smoke else tuple(case_names or ())
    all_cases = {case["name"]: case for case in manifest["cases"]}
    if selected_names:
        missing = set(selected_names) - set(all_cases)
        if missing:
            raise ValueError("unknown_benchmark_case")
        return [all_cases[name] for name in selected_names]
    # 故障注入项仍由离线状态机覆盖，但不能要求真实 Provider 复现人为构造的
    # 非法输出轨迹；默认 live 套件只包含具有正常业务终态的场景。
    return [case for case in manifest["cases"] if case.get("live_eligible", True)]


def _model_snapshot(settings, budgets, max_model_calls, *, preflight_named_policy_search=False):
    return {
        "model": str(settings.deepseek_model), "temperature": 0, "thinking": "disabled",
        "timeout_seconds": int(settings.agent_timeout_seconds),
        "provider_retries": 1,
        "budgets": {
            "max_model_decisions": budgets["max_model_decisions"],
            "max_search_calls": budgets["max_search_calls"], "max_repairs": budgets["max_repairs"],
            "max_model_calls": max_model_calls,
            "preflight_named_policy_search": preflight_named_policy_search,
        },
    }


def _compatible_previous_report(previous, report):
    if not isinstance(previous, dict) or previous.get("schema_version") != 2:
        return False
    if previous.get("benchmark") != report["benchmark"]:
        return False
    if previous.get("manifest") != report["manifest"]:
        return False
    if previous.get("model_snapshot", {}).get("model") != report["model_snapshot"]["model"]:
        return False
    if previous.get("model_snapshot", {}).get("temperature") != 0 or previous.get("model_snapshot", {}).get("thinking") != "disabled":
        return False
    metrics = previous.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("live_agent"), dict):
        return False
    if set(LEGACY_METRICS) - set(metrics["live_agent"]):
        return False
    if previous.get("repetitions") != report["repetitions"]:
        return False
    if previous.get("model_snapshot", {}).get("budgets") != report["model_snapshot"]["budgets"]:
        return False
    return [case.get("name") for case in previous.get("cases", [])] == [case["name"] for case in report["cases"]]


def _valid_previous_report_shape(previous) -> bool:
    """接受新旧 v2 live 报告的稳定比较子集。

    `--compare` 的输入可能由旧版 schema 生成；若先用当前 schema 校验，会在
    真正的配置兼容判断前错误拒绝历史基线。这里只校验比较实际消费的字段，
    具体配置是否一致仍由 `_compatible_previous_report` 决定。
    """
    if not isinstance(previous, dict):
        return False
    if previous.get("schema_version") != 2 or previous.get("mode") != "live":
        return False
    if not isinstance(previous.get("benchmark"), str) or not isinstance(previous.get("manifest"), str):
        return False
    if not isinstance(previous.get("repetitions"), int) or previous["repetitions"] < 1:
        return False
    if not isinstance(previous.get("model_snapshot"), dict) or not isinstance(previous.get("cases"), list):
        return False
    live_metrics = previous.get("metrics", {}).get("live_agent")
    if not isinstance(live_metrics, dict) or set(LEGACY_METRICS) - set(live_metrics):
        return False
    return all(
        isinstance(live_metrics[name], dict)
        and {"value", "defined_n", "numerator"} <= set(live_metrics[name])
        for name in LEGACY_METRICS
    )


def run_live_benchmark(
    manifest,
    manifest_name="data/benchmarks/planner_workflow_agent.yaml",
    *,
    repetitions=3,
    max_model_calls=200,
    case_names=None,
    smoke=False,
    model_factory: Callable[[dict, int], object] | None = None,
    settings=None,
    previous_report=None,
    preflight_named_policy_search=False,
):
    """显式 live 评测入口；默认工厂仅在真正的 repetition 开始时构造适配器。"""
    if repetitions < 1 or max_model_calls < 1:
        raise ValueError("live_benchmark_limits_must_be_positive")
    if settings is None:
        from rag_kb.config import get_settings
        settings = get_settings()
    if model_factory is None:
        model_factory = lambda _case, _repetition: DeepSeekPlannerDecisionModel(settings, max_retries=1)
    selected = _select_live_cases(manifest, case_names, smoke)
    workflow_runs, scripted_runs, live_runs, cases = [], [], [], []
    observed_model_calls = 0
    for case in selected:
        workflow = run_workflow(case, manifest["routes"], manifest["budgets"])
        scripted = run_agent(case, manifest["routes"], manifest["budgets"])
        workflow["metrics"], scripted["metrics"] = score(case, workflow), score(case, scripted)
        workflow_runs.append(workflow)
        scripted_runs.append(scripted)
        case_live_runs = []
        for repetition in range(1, repetitions + 1):
            if observed_model_calls >= max_model_calls:
                run = _skipped_live_run(case["name"], repetition)
            else:
                run = _run_live_agent(
                    case, manifest["routes"], manifest["budgets"], model_factory, repetition,
                    max_model_calls - observed_model_calls,
                    preflight_named_policy_search=preflight_named_policy_search,
                )
                observed_model_calls += run["observed_model_calls"]
            case_live_runs.append(run)
            live_runs.append(run)
        completed = [run for run in case_live_runs if run["status"] != "skipped"]
        cases.append({
            "name": case["name"], "workflow": workflow, "scripted_agent": scripted,
            "live_agent": {
                "runs": case_live_runs,
                "pass_rate": sum(run["status"] == "pass" for run in completed) / len(completed) if completed else None,
                "flaky": len({run["status"] for run in completed}) > 1,
                "metrics": aggregate(completed),
            },
        })
    workflow_metrics, scripted_metrics = aggregate(workflow_runs), aggregate(scripted_runs)
    live_metrics = aggregate([run for run in live_runs if run["status"] != "skipped"])
    actual_live = [run for run in live_runs if run["status"] != "skipped"]
    live_business_names = {
        case["name"] for case in selected if case.get("live_eligible", True)
    }
    live_business_runs = [
        run for run in actual_live if run["case_name"] in live_business_names
    ]
    model_quality_checks = _live_model_quality_gate_checks(live_business_runs)
    stale_runs = [
        run for case, entry in zip(selected, cases, strict=True) if case.get("stale_check")
        for run in entry["live_agent"]["runs"] if run["status"] != "skipped"
    ]
    gate_groups = {
        "safety_gates": {
            "acl_leak": all(not run["metrics"]["acl_leakage"] for run in actual_live),
            "illegal_citation": all(run["metrics"]["citation_legality"] is not False for run in actual_live),
            "stale_evidence_accessibility": all(not run["stale_accessible"] for run in stale_runs) if stale_runs else not any(case.get("stale_check") for case in selected),
        },
        "quality_gates": {
            "terminal_state_valid": all(run["metrics"]["terminal_state_valid"] is True for run in actual_live),
            "task_outcome_correct": all(run["metrics"]["task_outcome_correct"] is True for run in actual_live),
            "plan_constraint_pass": all(run["metrics"]["plan_constraint_pass"] is not False for run in actual_live),
            "clarification_question_quality": all(run["metrics"]["clarification_question_quality"] is not False for run in actual_live),
            "clarification_answer_utilization": all(run["metrics"]["clarification_answer_utilization"] is not False for run in actual_live),
            **model_quality_checks,
        },
        "cost_gates": {
            "model_call_budget": not any(
                run["status"] == "skipped" or run["error_code"] == "model_call_budget_reached"
                for run in live_runs
            ),
            "search_budget": all(run["executed_searches"] <= manifest["budgets"]["max_search_calls"] for run in actual_live),
            "unexpected_exhaustion": all(not run["metrics"]["unexpected_budget_exhaustion"] for run in actual_live),
        },
    }
    failures = [f"{group}.{name}" for group, checks in gate_groups.items() for name, passed in checks.items() if not passed]
    report = {
        "schema_version": 2, "benchmark": "planner-workflow-vs-agent", "manifest": manifest_name,
        "mode": "live", "sample_size": len(cases), "repetitions": repetitions,
        "model_snapshot": _model_snapshot(
            settings, manifest["budgets"], max_model_calls,
            preflight_named_policy_search=preflight_named_policy_search,
        ),
        "metrics": {"workflow": workflow_metrics, "scripted_agent": scripted_metrics, "live_agent": live_metrics},
        "metric_deltas": {
            "live_agent_minus_workflow_quality": _metric_deltas(live_metrics, workflow_metrics, QUALITY_METRICS),
            "live_agent_minus_scripted_agent_quality": _metric_deltas(live_metrics, scripted_metrics, QUALITY_METRICS),
        },
        "metric_delta_statuses": {
            "live_agent_minus_workflow_quality": _metric_delta_statuses(live_metrics, workflow_metrics, QUALITY_METRICS),
            "live_agent_minus_scripted_agent_quality": _metric_delta_statuses(live_metrics, scripted_metrics, QUALITY_METRICS),
        },
        "performance": aggregate_live_performance(live_runs),
        "safety_gates": _gate_report(gate_groups["safety_gates"]),
        "quality_gates": _gate_report(gate_groups["quality_gates"]),
        "cost_gates": _gate_report(gate_groups["cost_gates"]),
        "gates": {"passed": not failures, "failures": failures}, "cases": cases,
        "comparison_context": {
            "workflow_scope": "fixed_human_query_single_search_without_model_decisions",
            "agent_scope": "model_selects_search_clarification_refusal_and_proposal",
            "quality_deltas_are_diagnostic": True,
            "direct_superiority_claim_supported": False,
        },
        "statistical_scope": {
            "case_count": len(cases), "completed_run_count": len(actual_live),
            "small_sample": len(actual_live) < 30,
            "latency_percentiles_stable": len(actual_live) >= 30,
        },
        "quality_gate_thresholds": dict(LIVE_MODEL_QUALITY_THRESHOLDS),
        "previous_run_change": None,
    }
    if _compatible_previous_report(previous_report, report):
        assert previous_report is not None
        report["previous_run_change"] = {
            "metric_deltas": _metric_deltas(live_metrics, previous_report["metrics"]["live_agent"], METRICS),
            "metric_statuses": _metric_delta_statuses(live_metrics, previous_report["metrics"]["live_agent"], METRICS),
            "configuration_matched": True,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmarks/planner_workflow_agent.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/planner-workflow-agent-report.schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/local/benchmarks/planner-workflow-agent-report.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", dest="case_names")
    selection.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-model-calls", type=int, default=200)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if (args.case_names or args.smoke or args.compare) and not args.live:
        parser.error("--case, --smoke, and --compare require --live")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    previous = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    if previous is not None and not _valid_previous_report_shape(previous):
        parser.error("--compare must be a valid live benchmark report")
    report = run_live_benchmark(
        manifest, str(args.manifest), repetitions=args.repetitions,
        max_model_calls=args.max_model_calls, case_names=args.case_names, smoke=args.smoke,
        previous_report=previous,
        preflight_named_policy_search=True,
    ) if args.live else run_benchmark(manifest, str(args.manifest))
    validate(report, schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    smoke_failed = args.smoke and any(
        run["status"] in {"fail", "error"}
        for case in report.get("cases", [])
        for run in case.get("live_agent", {}).get("runs", [])
    )
    return 0 if report["gates"]["passed"] and not smoke_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
