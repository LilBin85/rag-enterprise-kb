"""冻结和 fake-live 评测均须直接运行真实 Planner 服务。"""
from copy import deepcopy
from itertools import count
import json
from pathlib import Path

from jsonschema import ValidationError, validate
import pytest
import yaml

import scripts.benchmark_planner_workflow_agent as benchmark
from rag_kb.models import Role
from rag_kb.policy_rollout_planning import PlannerDecision
from scripts.benchmark_planner_workflow_agent import CaseRetriever, _select_live_cases, run_benchmark, run_live_benchmark

def manifest(): return yaml.safe_load(Path("data/benchmarks/planner_workflow_agent.yaml").read_text(encoding="utf-8"))

def test_real_agent_paths_repairs_continuation_and_stale_gate():
    report = run_benchmark(manifest())
    by_name = {case["name"]: case for case in report["cases"]}
    # 离线套件保留故障注入契约；live 只运行正常业务语义场景。
    assert report["sample_size"] == 12
    assert "repair-exhaustion" in by_name
    assert by_name["adaptive-two-search-propose"]["agent"]["semantic_business_path"] == ["search", "propose"]
    assert by_name["adaptive-two-search-propose"]["agent"]["events"] == ["search", "search", "propose", "propose"]
    citation_case = by_name["citation-repair-success"]["agent"]
    assert citation_case["semantic_business_path"] == ["search", "propose"]
    assert citation_case["events"] == ["search", "propose", "repair", "search", "propose", "propose"]
    # 澄清答案利用按语义约束判断，不能依赖模型或组合器的固定 Todo ID。
    clarification_case = by_name["clarification-answer-used"]["agent"]
    assert clarification_case["metrics"]["clarification_answer_utilization"]
    assert not by_name["false-refusal-guard"]["agent"]["stale_accessible"]
    assert report["metrics"]["agent"]["autonomous_repair_success"] == {"value": 2 / 3, "defined_n": 3, "numerator": 2, "status": "observed"}


def test_live_selection_excludes_only_explicit_fault_injection_case():
    selected = _select_live_cases(manifest())
    assert len(selected) == 11
    assert "repair-exhaustion" not in {case["name"] for case in selected}

def test_expected_label_cannot_define_observed_agent_action():
    changed = deepcopy(manifest()); changed["cases"][0]["accepted_paths"] = [["search", "refuse"]]
    report = run_benchmark(changed)
    assert not report["cases"][0]["agent"]["metrics"]["decision_path_correctness"]


def test_exact_path_is_diagnostic_and_terminal_outcome_drives_quality():
    changed = deepcopy(manifest())
    changed["cases"][0]["accepted_paths"] = [["search", "search", "propose"]]
    run = run_benchmark(changed)["cases"][0]["agent"]
    assert not run["metrics"]["decision_path_correctness"]
    assert not run["metrics"]["exact_control_trace_match"]
    assert run["metrics"]["semantic_business_path_correctness"]
    assert run["semantic_business_path"] == ["search", "propose"]
    assert run["metrics"]["terminal_state_valid"]
    assert run["metrics"]["task_outcome_correct"]


def test_non_plan_terminal_has_explicit_metric_status_and_split_gates():
    report = run_benchmark(manifest())
    refusal = {case["name"]: case for case in report["cases"]}["no-evidence-refusal"]["agent"]
    assert refusal["metrics"]["plan_constraint_pass"] is None
    assert refusal["metric_statuses"]["plan_constraint_pass"] == "not_applicable"
    assert report["safety_gates"]["passed"]
    assert report["quality_gates"]["passed"]
    assert report["cost_gates"]["passed"]


def test_clarification_and_todo_metrics_are_semantic_not_id_only():
    clarification = deepcopy(manifest()["cases"][2])
    assert not benchmark._clarification_quality(clarification, {
        "_clarification_questions": ["请补充更多信息"], "events": ["clarify"],
    })
    run = run_benchmark(manifest())["cases"][2]["agent"]
    assert run["metrics"]["clarification_question_quality"]
    assert run["metrics"]["clarification_answer_utilization"]

    case = deepcopy(manifest()["cases"][0])
    semantic_miss = {
        "action": "propose", "events": ["search", "propose"],
        "todos": {"arbitrary-id": ["leave:v2:chunk-01"]},
        "citations": ["leave:v2:chunk-01"], "stale_accessible": True,
        "model_decisions": 2, "search_calls": 1, "repairs": 0,
        "attempted_searches": 1, "executed_searches": 1,
        "deduplicated_searches": 0, "rejected_searches": 0,
        "_retrieved_chunks": ["leave:v2:chunk-01"],
        "_todo_details": {
            "arbitrary-id": {
                "title": "泛化任务", "detail": "处理事项", "depends_on": [],
                "chunk_ids": ["leave:v2:chunk-01"],
            },
        },
        "_clarification_questions": [], "_clarification_answers": [],
    }
    metrics = benchmark.score(case, semantic_miss)
    assert metrics["citation_legality"]
    assert not metrics["todo_action_coverage"]
    assert not metrics["todo_grounding"]


def test_todo_quality_allows_split_actions_and_arbitrary_dependency_ids() -> None:
    case = deepcopy(manifest()["cases"][1])
    run = {
        "action": "propose", "events": ["search", "search", "propose"],
        "todos": {
            "step-a": ["leave:v2:chunk-01"], "step-b": ["leave:v2:chunk-01"],
            "step-c": ["leave:v2:chunk-02"], "step-d": ["leave:v2:chunk-02"],
        },
        "citations": ["leave:v2:chunk-01", "leave:v2:chunk-02"],
        "stale_accessible": True, "model_decisions": 3, "search_calls": 2, "repairs": 0,
        "attempted_searches": 2, "executed_searches": 2,
        "deduplicated_searches": 0, "rejected_searches": 0,
        "_retrieved_chunks": ["leave:v2:chunk-01", "leave:v2:chunk-02"],
        "_todo_details": {
            "step-a": {"title": "核对适用资格", "detail": "核对适用资格。", "depends_on": [], "chunk_ids": ["leave:v2:chunk-01"]},
            "step-b": {"title": "核对年假余额", "detail": "核对年假余额。", "depends_on": [], "chunk_ids": ["leave:v2:chunk-01"]},
            "step-c": {"title": "配置审批流程", "detail": "配置审批流程。", "depends_on": ["step-a"], "chunk_ids": ["leave:v2:chunk-02"]},
            "step-d": {"title": "通知全体员工", "detail": "通知全体员工。", "depends_on": ["step-b"], "chunk_ids": ["leave:v2:chunk-02"]},
        },
        "_clarification_questions": [], "_clarification_answers": [],
    }
    metrics = benchmark.score(case, run)
    assert metrics["todo_action_coverage"]
    assert metrics["todo_dependency_quality"]
    assert metrics["citation_support_alignment"]
    assert metrics["todo_grounding"]


def test_todo_action_coverage_ignores_appended_evidence_excerpt() -> None:
    case = deepcopy(manifest()["cases"][0])
    run = {
        "action": "propose", "events": ["search", "propose"],
        "semantic_business_path": ["search", "propose"],
        "proposal_source": benchmark.PROPOSAL_SOURCE_COMPOSER,
        "used_grounded_fallback": False, "used_grounded_composer": True,
        "todos": {"generic": ["leave:v2:chunk-01"]},
        "citations": ["leave:v2:chunk-01"], "stale_accessible": True,
        "model_decisions": 1, "search_calls": 1, "repairs": 0,
        "attempted_searches": 1, "executed_searches": 1,
        "deduplicated_searches": 0, "rejected_searches": 0,
        "_retrieved_chunks": ["leave:v2:chunk-01"],
        "_todo_details": {
            "generic": {
                "title": "处理相关事项",
                "detail": "按流程办理。授权制度依据摘要：核对适用资格和年假余额。",
                "depends_on": [], "chunk_ids": ["leave:v2:chunk-01"],
            }
        },
        "_clarification_questions": [], "_clarification_answers": [],
    }
    metrics = benchmark.score(case, run)
    assert not metrics["todo_action_coverage"]
    assert not metrics["todo_grounding"]


def test_citation_legality_uses_retrieved_acl_evidence_not_gold_relevance() -> None:
    case = deepcopy(manifest()["cases"][0])
    run = {
        "action": "propose", "events": ["search", "propose"],
        "todos": {"extra": ["leave:v2:chunk-02"]},
        "citations": ["leave:v2:chunk-02"], "stale_accessible": True,
        "model_decisions": 2, "search_calls": 1, "repairs": 0,
        "attempted_searches": 1, "executed_searches": 1,
        "deduplicated_searches": 0, "rejected_searches": 0,
        "_retrieved_chunks": ["leave:v2:chunk-02"],
        "_todo_details": {
            "extra": {
                "title": "实施额外流程", "detail": "按实施附录执行。",
                "depends_on": [], "chunk_ids": ["leave:v2:chunk-02"],
            }
        },
        "_clarification_questions": [], "_clarification_answers": [],
    }
    metrics = benchmark.score(case, run)
    assert metrics["citation_legality"]
    assert not metrics["necessary_evidence_recall"]
    assert not metrics["todo_grounding"]


def test_acl_leakage_detects_forbidden_retrieved_chunk_even_when_not_cited() -> None:
    case = next(item for item in manifest()["cases"] if item["name"] == "acl-adversarial-refusal")
    run = {
        "action": "refuse", "events": ["search", "refuse"],
        "semantic_business_path": ["search", "refuse"],
        "proposal_source": benchmark.PROPOSAL_SOURCE_NOT_APPLICABLE,
        "model_decisions": 1, "search_calls": 1, "repairs": 0,
        "todos": {}, "citations": [], "stale_accessible": False,
        "attempted_searches": 1, "executed_searches": 1,
        "deduplicated_searches": 0, "rejected_searches": 0,
        "_retrieved_chunks": ["secret:v1:chunk-01"],
        "_todo_details": {}, "_clarification_questions": [],
        "_clarification_answers": [], "_had_rejected_decision": False,
    }
    assert benchmark.score(case, run)["acl_leakage"]


def test_fixture_evidence_is_complete_chinese_policy_text():
    routes = manifest()["routes"]
    assert "全体员工" in routes["leave"]["candidates"][0]["text"]
    assert "审批流程" in routes["implementation"]["candidates"][0]["text"]
    assert routes["leave"]["candidates"][0]["text"] != "annual"

def test_script_route_and_stale_mutations_change_observed_metrics():
    changed = deepcopy(manifest()); changed["cases"][6]["script"][-1]["draft"]["used_chunk_ids"] = ["fake"]
    report = run_benchmark(changed)
    assert report["cases"][6]["agent"]["action"] == "failed"
    changed = deepcopy(manifest()); changed["routes"]["implementation"]["candidates"] = []
    report = run_benchmark(changed)
    assert not report["cases"][1]["agent"]["metrics"]["decision_path_correctness"]

def test_missing_stale_check_fails_the_actual_stale_gate():
    changed = deepcopy(manifest())
    changed["cases"][5].pop("stale_check")
    report = run_benchmark(changed)
    assert "stale_evidence_accessibility" in report["safety_gates"]["failures"]


class SettingsStub:
    deepseek_model = "deepseek-test"
    deepseek_api_key = "secret-api-key-must-not-appear"
    deepseek_base_url = "https://secret.example.invalid"
    agent_timeout_seconds = 7


def scripted_live_factory(case, _repetition):
    return benchmark.ScriptedModel(deepcopy(case["script"]))


def validate_report(report):
    validate(report, json.loads(Path("schemas/planner-workflow-agent-report.schema.json").read_text(encoding="utf-8")))


def test_report_schema_error_codes_cover_runtime_allowlist():
    schema = json.loads(Path("schemas/planner-workflow-agent-report.schema.json").read_text(encoding="utf-8"))
    schema_codes = set(schema["$defs"]["live_run"]["properties"]["error_code"]["enum"])
    assert benchmark.ALLOWLISTED_ERROR_CODES <= schema_codes


def test_live_v2_schema_requires_new_metrics_usage_and_split_gates():
    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=scripted_live_factory, settings=SettingsStub(),
    )
    validate_report(report)

    removals = (
        ("quality_gate_thresholds",),
        ("safety_gates",),
        ("quality_gates",),
        ("cost_gates",),
        ("quality_gates", "checks", "first_pass_plan_success"),
        ("metrics", "live_agent", "first_pass_plan_success_rate"),
        ("cases", 0, "live_agent", "runs", 0, "semantic_business_path"),
        ("cases", 0, "live_agent", "runs", 0, "metrics", "semantic_business_path_correctness"),
        ("cases", 0, "live_agent", "runs", 0, "tokens", "usage_observed_calls"),
        ("performance", "tokens", "call_metadata_partial_rate"),
    )
    for path in removals:
        changed = deepcopy(report)
        target = changed
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
        with pytest.raises(ValidationError):
            validate_report(changed)


def test_deterministic_report_stays_v1_and_strictly_valid():
    report = run_benchmark(manifest())
    assert report["schema_version"] == 1
    validate_report(report)


def test_case_retriever_matches_only_declared_normalized_aliases_and_keeps_acl():
    routes = manifest()["routes"]
    retriever = CaseRetriever(routes, "hr")
    assert [item.chunk_id for item in retriever.search(" 请核对年假制度 ", Role.HR).candidates] == ["leave:v2:chunk-01"]
    assert [item.chunk_id for item in retriever.search("年假制度和实施安排", Role.HR).candidates] == ["leave:v2:chunk-01", "leave:v2:chunk-02"]
    assert retriever.search("休假", Role.HR).candidates == []
    assert CaseRetriever(routes, "employee").search("受限政策", Role.EMPLOYEE).candidates == []


def test_adaptive_fixture_requires_two_distinct_declared_topics():
    routes = manifest()["routes"]
    retriever = CaseRetriever(routes, "hr")
    assert [item.chunk_id for item in retriever.search("年假制度", Role.HR).candidates] == ["leave:v2:chunk-01"]
    assert [item.chunk_id for item in retriever.search("实施安排", Role.HR).candidates] == ["leave:v2:chunk-02"]
    assert [item.chunk_id for item in retriever.search("年假制度和实施安排", Role.HR).candidates] == ["leave:v2:chunk-01", "leave:v2:chunk-02"]


def test_fake_live_smoke_repeats_selected_cases_and_excludes_sensitive_text():
    report = run_live_benchmark(
        manifest(), smoke=True, repetitions=2, max_model_calls=100,
        model_factory=scripted_live_factory, settings=SettingsStub(),
    )
    assert [case["name"] for case in report["cases"]] == list(benchmark.SMOKE_CASES)
    assert all(len(case["live_agent"]["runs"]) == 2 for case in report["cases"])
    assert all(run["case_name"] == case["name"] for case in report["cases"] for run in case["live_agent"]["runs"])
    assert report["performance"]["total_model_decisions"] > 0
    serialized = json.dumps(report, ensure_ascii=False)
    for forbidden in (SettingsStub.deepseek_api_key, SettingsStub.deepseek_base_url, "新版年假制度", "年假制度", "实施安排", "secret\"", "annual"):
        assert forbidden not in serialized
    validate_report(report)


def test_live_model_receives_case_goal_but_report_omits_it():
    received = []

    class GoalRecordingModel(benchmark.ScriptedModel):
        def invoke(self, messages):
            received.append(json.loads(messages[-1].content)["goal"])
            return super().invoke(messages)

    case = manifest()["cases"][0]
    report = run_live_benchmark(
        manifest(), case_names=[case["name"]], repetitions=1, max_model_calls=10,
        model_factory=lambda selected, _repetition: GoalRecordingModel(deepcopy(selected["script"])), settings=SettingsStub(),
    )
    assert received[0] == case["goal"]
    assert case["goal"] not in json.dumps(report, ensure_ascii=False)


def test_live_error_and_decision_fail_are_distinct():
    class RaisingModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        def invoke(self, _messages):
            raise RuntimeError("do not leak this exception")

    error_report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=lambda _case, _repetition: RaisingModel(), settings=SettingsStub(),
    )
    error_run = error_report["cases"][0]["live_agent"]["runs"][0]
    assert error_run["status"] == "error"
    assert error_run["error_code"] == "provider_request_failed"

    class WrongRefusalModel:
        def invoke(self, _messages):
            return PlannerDecision(action="refuse", refusal_reason="wrong")

    fail_report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=lambda _case, _repetition: WrongRefusalModel(), settings=SettingsStub(),
    )
    fail_run = fail_report["cases"][0]["live_agent"]["runs"][0]
    assert fail_run["status"] == "fail"
    assert fail_run["error_code"] == "quality_gate_failed"


def test_live_semantic_stop_precedes_tool_call_limit():
    class RepeatingSearchModel:
        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            return PlannerDecision(action="search", search_query=f"年假制度 {self.turn}")

    changed = deepcopy(manifest())
    changed["budgets"]["max_search_calls"] = 1
    report = run_live_benchmark(
        changed, case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=lambda _case, _repetition: RepeatingSearchModel(), settings=SettingsStub(),
    )
    run = report["cases"][0]["live_agent"]["runs"][0]
    assert run["status"] == "fail"
    assert run["error_code"] == "quality_gate_failed"
    assert run["attempted_searches"] == 3
    assert run["executed_searches"] == 1
    assert run["rejected_searches"] == 1
    assert run["deduplicated_searches"] == 1
    validate_report(report)


def test_live_budget_stops_following_runs_as_skipped():
    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=3, max_model_calls=1,
        model_factory=scripted_live_factory, settings=SettingsStub(),
    )
    runs = report["cases"][0]["live_agent"]["runs"]
    assert runs[0]["status"] == "fail" and runs[0]["error_code"] == "model_call_budget_reached"
    assert [run["status"] for run in runs[1:]] == ["skipped", "skipped"]
    assert "model_call_budget" in report["cost_gates"]["failures"]
    validate_report(report)


def test_live_usage_latency_aggregation_and_previous_metric_deltas(monkeypatch):
    ticks = count(0.0, 0.010)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))

    class UsageScriptedModel(benchmark.ScriptedModel):
        def invoke(self, messages):
            response = super().invoke(messages)
            object.__setattr__(response, "usage_metadata", {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
            return response

    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=2, max_model_calls=10,
        model_factory=lambda case, _repetition: UsageScriptedModel(deepcopy(case["script"])), settings=SettingsStub(),
    )
    previous = deepcopy(report)
    previous["metrics"]["live_agent"]["decision_path_correctness"]["value"] = 0.5
    comparison = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=2, max_model_calls=10,
        model_factory=lambda case, _repetition: UsageScriptedModel(deepcopy(case["script"])), settings=SettingsStub(),
        previous_report=previous,
    )
    performance = report["performance"]
    assert {name: performance["tokens"][name] for name in ("input", "output", "total")} == {"input": 8, "output": 12, "total": 20}
    assert performance["tokens"]["missing_rate"] == 0
    assert performance["tokens"]["usage_observed_calls"] == 4
    assert performance["tokens"]["usage_missing_calls"] == 0
    assert performance["tokens"]["successful"]["total"] == 20
    assert performance["tokens"]["failed_or_error"]["total"] is None
    assert performance["tokens"]["distributions"]["input"]["p50"] == 4
    assert not performance["tokens"]["provider_retries_observed"]
    assert performance["latency_scope"] == "end_to_end_planner_request_not_pure_model_inference"
    assert performance["latency_ms"]["p50"] is not None
    assert performance["latency_ms"]["p95"] >= performance["latency_ms"]["p50"]
    assert performance["latency_ms"]["max"] >= performance["latency_ms"]["p95"]
    assert comparison["previous_run_change"] is not None
    # delta 定义为“当前值 - 上次值”；组合器内部事件保留在完整控制轨迹中，
    # 因此本例当前精确轨迹为 0、手工基线为 0.5。
    assert comparison["previous_run_change"]["metric_deltas"]["decision_path_correctness"] == -0.5


def test_zero_model_call_is_not_token_metadata_missing():
    class MustNotBeInvoked:
        def invoke(self, _messages):
            raise AssertionError("授权预检为空时不应调用模型")

    report = run_live_benchmark(
        manifest(), case_names=["no-evidence-refusal"], repetitions=1, max_model_calls=10,
        model_factory=lambda _case, _repetition: MustNotBeInvoked(), settings=SettingsStub(),
        preflight_named_policy_search=True,
    )
    run = report["cases"][0]["live_agent"]["runs"][0]
    assert run["observed_model_calls"] == 0
    assert run["tokens"]["status"] == "not_applicable"
    assert run["tokens"]["reason"] == "no_model_call"
    assert run["tokens"]["usage_missing_calls"] == 0
    tokens = report["performance"]["tokens"]
    assert tokens["no_model_call_runs"] == 1
    assert tokens["missing_runs"] == 0
    assert tokens["missing_rate"] is None
    validate_report(report)


def test_usage_coverage_is_counted_per_model_call():
    class PartiallyObservedUsageModel(benchmark.ScriptedModel):
        def __init__(self, script):
            super().__init__(script)
            self.calls = 0

        def invoke(self, messages):
            response = super().invoke(messages)
            self.calls += 1
            if self.calls == 1:
                object.__setattr__(response, "usage_metadata", {
                    "input_tokens": 2, "output_tokens": 3, "total_tokens": 5,
                })
            return response

    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=lambda case, _repetition: PartiallyObservedUsageModel(deepcopy(case["script"])),
        settings=SettingsStub(),
    )
    run = report["cases"][0]["live_agent"]["runs"][0]
    assert run["observed_model_calls"] == 2
    assert run["tokens"]["status"] == "partial_metadata"
    assert run["tokens"]["usage_observed_calls"] == 1
    assert run["tokens"]["usage_missing_calls"] == 1
    assert report["performance"]["tokens"]["missing_rate"] == 0.5
    assert report["performance"]["tokens"]["call_metadata_complete_rate"] == 0.5
    assert report["performance"]["tokens"]["call_metadata_partial_rate"] == 0
    assert report["performance"]["tokens"]["call_metadata_missing_rate"] == 0.5
    validate_report(report)


def test_partial_call_metadata_has_own_rate_even_when_missing_rate_is_zero():
    class PartialUsageModel(benchmark.ScriptedModel):
        def invoke(self, messages):
            response = super().invoke(messages)
            object.__setattr__(response, "usage_metadata", {"total_tokens": 5})
            return response

    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=lambda case, _repetition: PartialUsageModel(deepcopy(case["script"])),
        settings=SettingsStub(),
    )
    tokens = report["performance"]["tokens"]
    assert tokens["missing_rate"] == 0
    assert tokens["call_metadata_complete_rate"] == 0
    assert tokens["call_metadata_partial_rate"] == 1
    assert tokens["call_metadata_missing_rate"] == 0
    validate_report(report)


def test_partial_token_fields_are_not_discarded_when_total_is_missing():
    runs = [{
        "observed_model_calls": 1,
        "tokens": {
            "input": 7, "output": None, "total": None,
            "status": "partial_metadata",
            "usage_observed_calls": 1, "usage_missing_calls": 0,
            "usage_partial_calls": 1,
        },
    }]
    summary = benchmark._token_summary(runs)
    assert summary["input"] == 7
    assert summary["output"] is None
    assert summary["total"] is None
    assert summary["average"]["input"] == 7
    assert summary["status"] == "partial_metadata"


def test_exception_usage_is_observed_before_parse_failure_is_rethrown():
    class ParseFailureWithUsage:
        def invoke(self, _messages):
            exc = RuntimeError("raw response parsing failed")
            exc.usage_metadata = {
                "input_tokens": 7, "output_tokens": 2, "total_tokens": 9,
            }
            raise exc

    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=lambda _case, _repetition: ParseFailureWithUsage(),
        settings=SettingsStub(),
    )
    run = report["cases"][0]["live_agent"]["runs"][0]
    assert run["status"] == "error"
    assert run["tokens"]["status"] == "observed"
    assert run["tokens"]["total"] == 9
    assert run["tokens"]["usage_observed_calls"] == 1
    assert run["tokens"]["usage_missing_calls"] == 0
    assert report["performance"]["tokens"]["call_metadata_complete_rate"] == 1
    validate_report(report)


def test_proposal_source_uses_stable_trace_outcome_and_code():
    assert benchmark._proposal_source(
        [{"action": "propose", "outcome": "planned", "constraint_code": None}], "propose",
    ) == benchmark.PROPOSAL_SOURCE_DIRECT
    assert benchmark._proposal_source(
        [
            {"action": "propose", "outcome": "constrained_composer", "constraint_code": "grounded_composer_after_validation"},
            {"action": "propose", "outcome": "planned", "constraint_code": "grounded_composer_after_validation"},
        ],
        "propose",
    ) == benchmark.PROPOSAL_SOURCE_COMPOSER
    assert benchmark._proposal_source(
        [{"action": "propose", "outcome": "grounded_fallback", "constraint_code": "grounded_fallback_after_unsupported_todo"}],
        "propose",
    ) == benchmark.PROPOSAL_SOURCE_FALLBACK


def test_constrained_composer_passes_first_pass_gate_but_emergency_fallback_does_not():
    composer_run = {
        "metrics": {
            "semantic_business_path_correctness": True,
            "first_pass_plan_success_rate": True,
            "repair_rate": False,
            "emergency_fallback_rate": False,
        }
    }
    fallback_run = {
        "metrics": {
            "semantic_business_path_correctness": True,
            "first_pass_plan_success_rate": False,
            "repair_rate": True,
            "emergency_fallback_rate": True,
        }
    }
    assert all(benchmark._live_model_quality_gate_checks([composer_run]).values())
    checks = benchmark._live_model_quality_gate_checks([fallback_run])
    assert not checks["first_pass_plan_success"]
    assert not checks["repair_rate"]
    assert not checks["emergency_fallback_rate"]


def test_fault_injection_case_does_not_enter_live_model_quality_gates():
    report = run_live_benchmark(
        manifest(), case_names=["repair-exhaustion"], repetitions=1, max_model_calls=10,
        model_factory=scripted_live_factory, settings=SettingsStub(),
    )
    for name in (
        "semantic_business_path", "first_pass_plan_success", "repair_rate", "emergency_fallback_rate",
    ):
        assert report["quality_gates"]["checks"][name]
    validate_report(report)


def test_incompatible_previous_report_is_omitted():
    report = run_live_benchmark(
        manifest(), case_names=["one-search-propose"], repetitions=1, max_model_calls=10,
        model_factory=scripted_live_factory, settings=SettingsStub(),
        previous_report={"schema_version": 2, "benchmark": "wrong"},
    )
    assert report["previous_run_change"] is None


def test_legacy_v2_previous_report_shape_is_accepted_without_current_schema_fields():
    legacy = {
        "schema_version": 2,
        "benchmark": "planner-workflow-vs-agent",
        "manifest": "data/benchmarks/planner_workflow_agent.yaml",
        "mode": "live",
        "repetitions": 1,
        "model_snapshot": {"model": "test", "budgets": {}},
        "cases": [],
        "metrics": {
            "live_agent": {
                name: {"value": None, "defined_n": 0, "numerator": 0}
                for name in benchmark.LEGACY_METRICS
            }
        },
    }
    assert benchmark._valid_previous_report_shape(legacy)
    del legacy["metrics"]["live_agent"][benchmark.LEGACY_METRICS[0]]["numerator"]
    assert not benchmark._valid_previous_report_shape(legacy)


def test_live_cli_returns_nonzero_for_hard_budget_gate(tmp_path, monkeypatch):
    class FakeDeepSeekModel:
        def __init__(self, _settings, *, max_retries):
            assert max_retries == 1
            self.model = benchmark.ScriptedModel(deepcopy(manifest()["cases"][0]["script"]))

        def invoke(self, messages):
            return self.model.invoke(messages)

    monkeypatch.setattr(benchmark, "DeepSeekPlannerDecisionModel", FakeDeepSeekModel)
    monkeypatch.setattr("rag_kb.config.get_settings", lambda: SettingsStub())
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--live", "--case", "one-search-propose", "--repetitions", "3", "--max-model-calls", "1", "--output", str(tmp_path / "report.json")],
    )
    assert benchmark.main() == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "model_call_budget" in report["cost_gates"]["failures"]


def test_live_smoke_cli_returns_nonzero_for_quality_fail(tmp_path, monkeypatch):
    class FakeDeepSeekModel:
        def __init__(self, _settings, *, max_retries):
            self.model = benchmark.ScriptedModel([{"draft": {"action": "refuse"}}])

        def invoke(self, messages):
            return self.model.invoke(messages)

    monkeypatch.setattr(benchmark, "DeepSeekPlannerDecisionModel", FakeDeepSeekModel)
    monkeypatch.setattr("rag_kb.config.get_settings", lambda: SettingsStub())
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--live", "--smoke", "--repetitions", "1", "--max-model-calls", "100", "--output", str(tmp_path / "report.json")],
    )
    assert benchmark.main() == 1
