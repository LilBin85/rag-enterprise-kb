"""策略落地规划的纯单元与 API 契约测试；不连接 Qdrant 或真实模型。"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from rag_kb.models import (
    PlanStatus, PolicyRolloutClarification, PolicyRolloutPlanVersion, PolicyRolloutTaskContinueRequest, PolicyRolloutTodo,
    Role, TodoCitation,
)
from rag_kb.policy_rollout_planning import (
    BoundPlanningTools, DeepSeekPlannerDecisionModel, PlannerDecision, PlanningDraft,
    PlanningError, PolicyRolloutPlanningService, _high_level_action_is_grounded,
    _todo_is_supported,
)
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore
from rag_kb.retriever import RetrievalCandidate, RetrievalResult
from rag_kb.retrieval_policy import QueryPolicy, hash_identifier


class FakeRetriever:
    def __init__(self, allowed_role: Role = Role.HR) -> None:
        self.allowed_role = allowed_role
        self.calls: list[tuple[str, Role]] = []
        self.error: Exception | None = None

    def search(self, query: str, role: Role) -> RetrievalResult:
        self.calls.append((query, role))
        if self.error:
            raise self.error
        if role != self.allowed_role:
            return RetrievalResult([], 0)
        return RetrievalResult([RetrievalCandidate(
            chunk_id="leave:v2:chunk-01", document_id="leave", document_version=2,
            title="休假制度", source_file_name="leave-v2.md", source_locator="年假",
            score=0.9, text="人力资源部门应核对年假规则，年假为十五天。"
        )], 0)


class FakePlanningModel:
    def __init__(self, drafts: list[dict]) -> None:
        self.drafts = drafts
        self.turn = 0

    def invoke(self, _messages):
        self.turn += 1
        if self.turn % 2:
            return PlannerDecision(action="search", search_query="休假政策")
        draft = self.drafts.pop(0)
        if draft.get("status") == "clarification":
            return PlannerDecision(action="clarify", clarification_questions=draft["clarification_questions"])
        if draft.get("status") == "refuse":
            return PlannerDecision(action="refuse", refusal_reason="test refusal")
        return PlannerDecision(action="propose", proposal=draft)


class RecordingPlanningModel(FakePlanningModel):
    def __init__(self, drafts: list[dict]) -> None:
        super().__init__(drafts)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return super().invoke(messages)


def _planned(answers: list[str] | None = None) -> dict:
    answers = answers or []
    scope = f"适用范围：{'、'.join(answers)}。" if answers else "确认制度适用范围。"
    return {
        "status": "planned",
        "used_chunk_ids": ["leave:v2:chunk-01"],
        "todos": [{
            "todo_id": "review", "title": "核对年假规则", "detail": scope,
            "depends_on": [], "chunk_ids": ["leave:v2:chunk-01"],
        }],
    }


def test_provider_draft_status_is_a_proposal_compatibility_alias() -> None:
    draft = PlanningDraft.model_validate({
        "action": "propose", "status": "draft", "used_chunk_ids": ["chunk-1"],
        "todos": [{
            "todo_id": "review", "title": "核对制度", "detail": "按制度核对。",
            "chunk_ids": ["chunk-1"],
        }],
    })
    assert draft.action == "propose"


def test_structured_planner_adapter_keeps_only_safe_token_metadata(monkeypatch) -> None:
    captured = {}

    class RawResponse:
        content = "SECRET_RAW_CONTENT"
        usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
        response_metadata = {"token_usage": {"prompt_tokens": 2, "completion_tokens": 3}, "request_id": "SECRET_REQUEST_ID"}

    class StructuredModel:
        def invoke(self, _messages):
            return {"raw": RawResponse(), "parsed": {"action": "refuse", "refusal_reason": "offline"}, "parsing_error": None}

    class FakeChatDeepSeek:
        def __init__(self, **_kwargs) -> None:
            pass

        def with_structured_output(self, schema, **kwargs):
            captured.update(schema=schema, kwargs=kwargs)
            return StructuredModel()

    monkeypatch.setattr("rag_kb.policy_rollout_planning.importlib.import_module", lambda _name: type("Module", (), {"ChatDeepSeek": FakeChatDeepSeek})())
    settings = type("SettingsStub", (), {"deepseek_model": "test", "deepseek_api_key": "key", "deepseek_base_url": "url", "agent_timeout_seconds": 1})()
    decision = DeepSeekPlannerDecisionModel(settings).invoke([])
    assert captured == {"schema": PlannerDecision, "kwargs": {"include_raw": True}}
    assert decision.action == "refuse"
    assert decision.usage_metadata == RawResponse.usage_metadata
    assert decision.response_metadata == {"token_usage": {"prompt_tokens": 2, "completion_tokens": 3}}
    assert "SECRET" not in f"{decision!r}{decision.model_dump()}"

    class FailureRawResponse:
        content = "SECRET_FAILURE_RAW_CONTENT"
        usage_metadata = {
            "input_tokens": 7, "output_tokens": 11, "total_tokens": 18,
            "secret_tokens": 999, "cached_tokens": True, "reasoning_tokens": -1,
        }
        response_metadata = {
            "token_usage": {
                "prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18,
                "secret_tokens": 999, "cached_tokens": True, "reasoning_tokens": -1,
            },
            "request_id": "SECRET_FAILURE_REQUEST_ID",
        }

    def assert_safe_failure_usage(error: PlanningError) -> None:
        assert error.usage_metadata == {
            "input_tokens": 7, "output_tokens": 11, "total_tokens": 18,
        }
        assert error.response_metadata == {"token_usage": {
            "prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18,
        }}
        assert error.__cause__ is None and error.__context__ is None
        serialized = repr((error.args, error.__dict__))
        assert "SECRET" not in serialized and "bad parse" not in serialized
        assert all(
            type(count) is int and count >= 0
            for metadata in (
                error.usage_metadata,
                error.response_metadata["token_usage"],
            )
            for count in metadata.values()
        )

    class BrokenStructuredModel:
        def invoke(self, _messages):
            return {
                "raw": FailureRawResponse(), "parsed": None,
                "parsing_error": ValueError("SECRET bad parse"),
            }

    adapter = DeepSeekPlannerDecisionModel(settings)
    adapter._model = BrokenStructuredModel()
    with pytest.raises(PlanningError, match="model_output_parse_failed") as parsing_failure:
        adapter.invoke([])
    assert_safe_failure_usage(parsing_failure.value)

    class InvalidParsedModel:
        def invoke(self, _messages):
            return {
                "raw": FailureRawResponse(),
                "parsed": {
                    "action": "search", "search_query": "年假制度",
                    "refusal_reason": "SECRET_INVALID_PARSED_TEXT",
                },
                "parsing_error": None,
            }

    adapter._model = InvalidParsedModel()
    with pytest.raises(PlanningError, match="schema_validation_failed") as validation_failure:
        adapter.invoke([])
    assert_safe_failure_usage(validation_failure.value)


def test_structured_planner_adapter_classifies_provider_failures_without_leaking_text(monkeypatch) -> None:
    class TimedOutModel:
        def invoke(self, _messages):
            raise TimeoutError("SECRET_PROVIDER_DETAIL timed out")

    class FakeChatDeepSeek:
        def __init__(self, **_kwargs) -> None:
            pass

        def with_structured_output(self, _schema, **_kwargs):
            return TimedOutModel()

    monkeypatch.setattr("rag_kb.policy_rollout_planning.importlib.import_module", lambda _name: type("Module", (), {"ChatDeepSeek": FakeChatDeepSeek})())
    settings = type("SettingsStub", (), {"deepseek_model": "test", "deepseek_api_key": "key", "deepseek_base_url": "url", "agent_timeout_seconds": 1})()
    with pytest.raises(PlanningError) as captured:
        DeepSeekPlannerDecisionModel(settings).invoke([])
    assert str(captured.value) == "provider_timeout"
    assert "SECRET_PROVIDER_DETAIL" not in str(captured.value)


def _service(tmp_path, drafts: list[dict], role: Role = Role.HR, **kwargs):
    return PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), FakeRetriever(role), FakePlanningModel(drafts),
        citation_checker=lambda _citations, _role: True,
        **kwargs,
    )


def test_valid_plan_persists_and_recovers_from_sqlite(tmp_path) -> None:
    service = _service(tmp_path, [_planned()])
    created = service.create("落地新版休假制度", Role.HR)
    recovered = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3").get(created.task_id, Role.HR, "recovery")
    assert created.status == PlanStatus.PLANNED
    assert recovered is not None and recovered.status == PlanStatus.PLANNED
    assert recovered.plan_versions[0].todos[0].citations[0].chunk_id == "leave:v2:chunk-01"


def test_clarification_continue_appends_immutable_plan_version(tmp_path) -> None:
    service = _service(tmp_path, [_planned(["全员"])])
    task = service.create("落地休假制度；适用部门待确认。", Role.HR)
    first = service.continue_task(task.task_id, Role.HR, ["全员"])
    second = service.continue_task(task.task_id, Role.HR, ["全员"])
    assert task.status == PlanStatus.CLARIFICATION
    assert first is not None and [p.version for p in first.plan_versions] == [1]
    assert first.clarification is not None and first.clarification.answers == ["全员"]
    assert second is None
    reopened = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3").get(task.task_id, Role.HR, "reopen")
    assert reopened is not None and [p.version for p in reopened.plan_versions] == [1]
    assert reopened.plan_versions[0].model_dump() == first.plan_versions[0].model_dump()


def test_clarification_answers_are_bounded_and_accept_boundary() -> None:
    boundary = "x" * 2000
    assert PolicyRolloutTaskContinueRequest(clarification_answers=[boundary]).clarification_answers == [boundary]
    with pytest.raises(ValueError):
        PolicyRolloutTaskContinueRequest(clarification_answers=[boundary + "x"])
    with pytest.raises(ValueError):
        PolicyRolloutClarification(questions=["范围？"], answers=[boundary + "x"])


@pytest.mark.parametrize("mutate", [lambda draft: draft.update(used_chunk_ids=["fabricated"]), lambda draft: draft["todos"][0].update(chunk_ids=["fabricated"]), lambda draft: draft["todos"][0].update(depends_on=["review"])])
def test_fabricated_evidence_or_invalid_dependencies_fail_closed(tmp_path, mutate) -> None:
    draft = _planned()
    mutate(draft)
    response = _service(tmp_path, [draft]).create("落地休假制度", Role.HR)
    assert response.status == PlanStatus.FAILED and response.plan_versions == []


def test_dependency_cycle_and_model_or_tool_failures_fail_closed(tmp_path) -> None:
    cycle = _planned()
    cycle["todos"].append({"todo_id": "publish", "title": "发布", "detail": "发布变更。", "depends_on": ["review"], "chunk_ids": ["leave:v2:chunk-01"]})
    cycle["todos"][0]["depends_on"] = ["publish"]
    assert _service(tmp_path, [cycle]).create("目标", Role.HR).status == PlanStatus.FAILED
    service = _service(tmp_path, [_planned()])
    service.retriever.error = RuntimeError("unavailable")
    assert service.create("目标", Role.HR).status == PlanStatus.FAILED
    failed_model = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "model.sqlite3"), FakeRetriever(), None,
        citation_checker=lambda _citations, _role: True,
    )
    assert failed_model.create("目标", Role.HR).status == PlanStatus.FAILED


def test_explicit_refusal_appends_no_plan_and_records_text_free_trace(tmp_path) -> None:
    service = _service(tmp_path, [{"status": "refuse"}])
    response = service.create("私密目标 SECRET_GOAL", Role.HR)
    trace = service.store.decision_trace(response.task_id)
    assert response.status == PlanStatus.REFUSED and response.plan_versions == []
    assert trace and trace[-1]["action"] == "refuse"
    assert "SECRET_GOAL" not in json.dumps(trace, ensure_ascii=False)
    assert {"action", "outcome", "model_decisions", "search_calls", "repairs", "constraint_code", "duration_ms"} <= trace[-1].keys()


def test_invalid_proposal_is_repaired_but_infrastructure_failure_is_not(tmp_path) -> None:
    invalid = _planned()
    invalid["todos"] = []
    repaired = _service(tmp_path, [invalid, _planned()])
    response = repaired.create("目标", Role.HR)
    assert response.status == PlanStatus.PLANNED
    trace = repaired.store.decision_trace(response.task_id)
    assert trace[-1]["repairs"] == 1
    assert trace[-1]["constraint_code"] == "grounded_composer_after_validation"
    assert any(
        event["action"] == "repair" and event["constraint_code"] == "invalid_citations"
        for event in trace
    )

    broken = _service(tmp_path, [_planned()])
    broken.retriever.error = RuntimeError("unavailable")
    failed = broken.create("目标", Role.HR)
    assert failed.status == PlanStatus.FAILED
    assert broken.store.decision_trace(failed.task_id)[-1]["repairs"] == 0


def test_structural_repair_exhaustion_appends_no_plan(tmp_path) -> None:
    invalid = _planned()
    invalid["todos"] = []
    service = _service(tmp_path, [invalid, invalid, invalid], max_repairs=2)
    response = service.create("目标", Role.HR)
    assert response.status == PlanStatus.FAILED and response.plan_versions == []
    assert service.store.decision_trace(response.task_id)[-1]["repairs"] == 2


def test_semantically_duplicate_search_is_not_executed_before_proposal(tmp_path) -> None:
    class TwoSearchModel(FakePlanningModel):
        def invoke(self, _messages):
            self.turn += 1
            if self.turn <= 2:
                return PlannerDecision(action="search", search_query=f"休假政策 {self.turn}")
            return PlannerDecision(action="propose", proposal=_planned())

    service = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), FakeRetriever(), TwoSearchModel([]),
        citation_checker=lambda _citations, _role: True, max_tool_calls=3,
    )
    response = service.create("目标", Role.HR)
    assert response.status == PlanStatus.PLANNED
    trace = service.store.decision_trace(response.task_id)
    assert trace[-1]["search_calls"] == 1
    assert [event["outcome"] for event in trace] == [
        "success", "deduplicated", "constrained_composer", "planned",
    ]


def test_search_for_explicitly_excluded_facet_is_rejected_before_proposal(tmp_path) -> None:
    class OutOfScopeSearchModel:
        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            if self.turn == 1:
                return PlannerDecision(action="search", search_query="年假制度")
            if self.turn == 2:
                return PlannerDecision(action="search", search_query="系统实施安排")
            return PlannerDecision(action="propose", proposal=_planned())

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    retriever = FakeRetriever()
    response = PolicyRolloutPlanningService(
        store, retriever, OutOfScopeSearchModel(), citation_checker=lambda _citations, _role: True,
    ).create("请核对年假规则；不需要系统实施安排。", Role.HR)
    assert response.status == PlanStatus.PLANNED
    assert len(retriever.calls) == 1
    trace = store.decision_trace(response.task_id)
    assert [event["outcome"] for event in trace] == [
        "success", "rejected", "constrained_composer", "planned",
    ]
    assert trace[1]["constraint_code"] == "query_outside_goal_scope"


def test_terminal_model_schema_failures_fall_back_to_grounded_checklist(tmp_path) -> None:
    class TerminalSchemaFailureModel:
        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            if self.turn == 1:
                return PlannerDecision(action="search", search_query="休假政策")
            if self.turn == 2:
                return PlannerDecision(action="search", search_query="年假余额")
            raise PlanningError("schema_validation_failed")

    model = TerminalSchemaFailureModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FakeRetriever(), model, citation_checker=lambda _citations, _role: True,
    ).create("请根据休假制度形成核对清单。", Role.HR)
    assert response.status == PlanStatus.PLANNED
    assert model.turn == 5
    todo = response.plan_versions[0].todos[0]
    assert todo.todo_id == "grounded-review-1"
    assert todo.citations[0].chunk_id == "leave:v2:chunk-01"
    assert "任务范围（非制度证据）" in todo.detail
    trace = store.decision_trace(response.task_id)
    assert any(event["outcome"] == "grounded_fallback" for event in trace)


def test_grounded_fallback_preserves_clarified_scope_after_repairs(tmp_path) -> None:
    """复现真实 clarification smoke：二次搜索被拒后，坏提案耗尽修复预算。"""

    class FixtureRetriever(FakeRetriever):
        def search(self, query: str, role: Role) -> RetrievalResult:
            self.calls.append((query, role))
            if role != self.allowed_role:
                return RetrievalResult([], 0)
            return RetrievalResult([RetrievalCandidate(
                chunk_id="leave:v2:chunk-01", document_id="leave", document_version=2,
                title="新版年假制度", source_file_name="leave-v2.md",
                source_locator="员工手册/年假制度/第二版/第3-5条", score=0.9,
                text=(
                    "新版年假制度适用于与公司签订劳动合同的全体员工。"
                    "人力资源部门应先核对员工适用资格和截至切换日的年假余额。"
                ),
            )], 0)

    unsupported = PlanningDraft.model_validate({
        "action": "propose",
        "used_chunk_ids": ["leave:v2:chunk-01"],
        "todos": [{
            "todo_id": "unsupported", "title": "采购年假主题礼品",
            "detail": "为试点员工订购礼品。", "chunk_ids": ["leave:v2:chunk-01"],
        }],
    })

    class RepairExhaustingModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return PlannerDecision(action="search", search_query="年假制度")
            if self.calls == 2:
                return PlannerDecision(action="search", search_query="员工休假规则")
            return PlannerDecision(action="propose", proposal=unsupported)

    goal = (
        "请根据休假制度制定分批落地计划；制度适用于全体员工，"
        "但首批试点部门尚未确定，应先确认试点范围。"
    )
    retriever = FixtureRetriever()
    model = RepairExhaustingModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, retriever, model, citation_checker=lambda _citations, _role: True,
    )

    clarification = service.create(goal, Role.HR)
    assert clarification.status == PlanStatus.CLARIFICATION
    planned = service.continue_task(
        clarification.task_id, Role.HR, ["首批试点部门为研发部"],
    )

    assert planned is not None and planned.status == PlanStatus.PLANNED
    assert model.calls == 5 and len(retriever.calls) == 1
    todo = planned.plan_versions[0].todos[0]
    semantic_todo = f"{todo.title} {todo.detail}"
    assert all(
        term in semantic_todo
        for term in ("研发部", "试点范围", "核对", "适用资格", "年假余额")
    )
    assert "用户已确认的业务约束（非制度证据）" in todo.detail
    assert "任务范围（非制度证据）" in todo.detail
    assert "授权制度依据" in todo.detail
    assert [citation.chunk_id for citation in todo.citations] == ["leave:v2:chunk-01"]
    trace = store.decision_trace(clarification.task_id)
    assert trace[-1]["constraint_code"] == "grounded_fallback_after_unsupported_todo"
    assert trace[-1]["executed_searches"] == 1
    assert trace[-1]["rejected_searches"] == 1


def test_semantic_stop_precedes_tool_call_limit(tmp_path) -> None:
    class RepeatingSearchModel:
        def __init__(self) -> None:
            self.turn = 0

        def invoke(self, _messages):
            self.turn += 1
            return PlannerDecision(action="search", search_query=f"休假政策 {self.turn}")

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, FakeRetriever(), RepeatingSearchModel(),
        citation_checker=lambda _citations, _role: True, max_tool_calls=1,
    )
    response = service.create("目标", Role.HR)
    assert response.status == PlanStatus.REFUSED
    final = store.decision_trace(response.task_id)[-1]
    assert final["constraint_code"] == "search_after_terminal_evidence_state"
    assert final["attempted_searches"] == 3
    assert final["executed_searches"] == final["search_calls"] == 1
    assert final["rejected_searches"] == 1
    assert final["deduplicated_searches"] == 1


def test_empty_authorized_search_refuses_without_query_rewrites(tmp_path) -> None:
    class SynonymSearchModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            queries = ["年假制度", "员工休假规则", "公司年假政策"]
            query = queries[min(self.calls, len(queries) - 1)]
            self.calls += 1
            return PlannerDecision(action="search", search_query=query)

    model = SynonymSearchModel()
    retriever = FakeRetriever(Role.EMPLOYEE)
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, retriever, model, citation_checker=lambda _citations, _role: True,
    ).create("为 HR 落地年假制度", Role.HR)
    assert response.status == PlanStatus.REFUSED
    assert model.calls == 1 and retriever.calls == [("年假制度", Role.HR)]
    assert [event["outcome"] for event in store.decision_trace(response.task_id)] == ["empty", "refused"]
    final = store.decision_trace(response.task_id)[-1]
    assert final["constraint_code"] == "authorized_evidence_empty"
    assert final["attempted_searches"] == final["executed_searches"] == 1


def test_no_new_chunk_for_structured_rewrite_forces_terminal_decision(tmp_path) -> None:
    class SequenceModel:
        def __init__(self) -> None:
            self.decisions = iter([
                PlannerDecision(action="search", search_query="年假制度"),
                PlannerDecision(action="search", search_query="实施安排"),
                PlannerDecision(action="propose", proposal=_planned()),
            ])

        def invoke(self, _messages):
            return next(self.decisions)

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, FakeRetriever(), SequenceModel(),
        citation_checker=lambda _citations, _role: True,
    )
    response = service.create("请根据年假制度制定计划。", Role.HR)
    assert response.status == PlanStatus.PLANNED
    trace = store.decision_trace(response.task_id)
    assert [event["outcome"] for event in trace] == [
        "success", "no_new_evidence", "constrained_composer", "planned",
    ]
    assert trace[-1]["executed_searches"] == 2
    with sqlite3.connect(tmp_path / "tasks.sqlite3") as connection:
        persisted = connection.execute(
            "SELECT seen_chunk_hashes_json FROM policy_rollout_tasks WHERE task_id = ?",
            (response.task_id,),
        ).fetchone()[0]
    assert "leave:v2:chunk-01" not in persisted
    assert hash_identifier("leave:v2:chunk-01") in persisted


def test_only_one_structurally_distinct_query_rewrite_is_executed(tmp_path) -> None:
    class FacetedRetriever(FakeRetriever):
        def search(self, query: str, role: Role) -> RetrievalResult:
            self.calls.append((query, role))
            suffix = "implementation" if "实施" in query else "leave"
            return RetrievalResult([RetrievalCandidate(
                chunk_id=f"{suffix}:v2:chunk-01", document_id=suffix,
                document_version=2, title="休假制度", source_file_name="leave-v2.md",
                source_locator="制度", score=0.9,
                text="人力资源部门应核对年假规则并实施配置。",
            )], 0)

    class SequenceModel:
        def __init__(self) -> None:
            self.decisions = iter([
                PlannerDecision(action="search", search_query="年假制度"),
                PlannerDecision(action="search", search_query="实施安排"),
                PlannerDecision(action="search", search_query="审批流程"),
            ])

        def invoke(self, _messages):
            return next(self.decisions)

    retriever = FacetedRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, retriever, SequenceModel(), citation_checker=lambda _citations, _role: True,
    ).create("请根据年假制度制定计划。", Role.HR)
    assert response.status == PlanStatus.REFUSED
    assert len(retriever.calls) == 2
    trace = store.decision_trace(response.task_id)
    assert trace[-2]["constraint_code"] == "search_after_terminal_evidence_state"
    assert trace[-1]["executed_searches"] == 2
    assert trace[-1]["rejected_searches"] == 1


def test_unjustified_clarification_is_rejected_then_refused(tmp_path) -> None:
    class ClarifyingModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return PlannerDecision(action="clarify", clarification_questions=["请补充更多信息？"])

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FakeRetriever(), ClarifyingModel(),
        citation_checker=lambda _citations, _role: True,
    ).create("请根据年假制度制定计划。", Role.HR)
    assert response.status == PlanStatus.REFUSED and response.clarification is None
    trace = store.decision_trace(response.task_id)
    assert [event["outcome"] for event in trace] == ["rejected", "rejected", "refused"]
    assert trace[-1]["constraint_code"] == "clarification_not_justified"


def test_named_policy_preflight_refuses_acl_empty_without_model_clarification(tmp_path) -> None:
    class MustNotInvoke:
        def invoke(self, _messages):
            raise AssertionError("ACL 空结果不应先调用模型")

    retriever = FakeRetriever(Role.HR)
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, retriever, MustNotInvoke(),
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("请为受限政策制定计划。", Role.EMPLOYEE)

    assert response.status == PlanStatus.REFUSED
    assert retriever.calls == [("受限政策", Role.EMPLOYEE)]
    trace = store.decision_trace(response.task_id)
    assert [event["action"] for event in trace] == ["search", "refuse"]
    assert trace[-1]["constraint_code"] == "authorized_evidence_empty"
    assert trace[-1]["model_decisions"] == 0


def test_named_policy_preflight_keeps_explicit_missing_variable_before_search(tmp_path) -> None:
    model = FakePlanningModel([_planned(["研发部"])])
    retriever = FakeRetriever()
    response = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), retriever, model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("首批试点部门尚未确定；请根据年假制度制定计划。", Role.HR)

    assert response.status == PlanStatus.CLARIFICATION
    assert retriever.calls == []
    assert model.turn == 0


def test_named_policy_preflight_exposes_authorized_evidence_before_model_decision(tmp_path) -> None:
    class ProposingModel:
        def __init__(self) -> None:
            self.messages = []

        def invoke(self, messages):
            self.messages.append(messages)
            return PlannerDecision(action="propose", proposal=_planned())

    model = ProposingModel()
    retriever = FakeRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, retriever, model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("请根据年假制度制定计划。", Role.HR)

    assert response.status == PlanStatus.PLANNED
    assert retriever.calls == [("年假制度", Role.HR)]
    request = json.loads(model.messages[0][-1].content)
    assert request["search_feedback"] == "preflight_authorized_evidence_terminal_only"
    assert request["evidence"][0]["chunk_id"] == "leave:v2:chunk-01"
    assert [event["outcome"] for event in store.decision_trace(response.task_id)] == [
        "success", "constrained_composer", "planned",
    ]


def test_named_policy_preflight_fetches_explicit_implementation_facet(tmp_path) -> None:
    class FacetRetriever:
        def __init__(self) -> None:
            self.calls = []

        def search(self, query, role):
            self.calls.append((query, role))
            review = RetrievalCandidate("chunk-1", "leave", 2, "年假制度", "leave.md", "第3条", 0.9, "先核对员工适用资格和年假余额")
            implementation = RetrievalCandidate("chunk-2", "leave", 2, "年假制度", "leave.md", "实施附录", 0.9, "配置审批流程并通知员工")
            return RetrievalResult([review, implementation] if "实施安排" in query else [review], 0)

    class ProposingModel:
        def __init__(self) -> None:
            self.messages = []

        def invoke(self, messages):
            self.messages.append(messages)
            return PlannerDecision(action="propose", proposal=PlanningDraft(
                action="propose", used_chunk_ids=["chunk-1", "chunk-2"], todos=[
                    {"todo_id": "review", "title": "核对员工适用资格和年假余额", "detail": "核对员工适用资格和年假余额", "chunk_ids": ["chunk-1"]},
                    {"todo_id": "implement", "title": "配置审批流程并通知员工", "detail": "配置审批流程并通知员工", "depends_on": ["review"], "chunk_ids": ["chunk-2"]},
                ],
            ))

    retriever, model = FacetRetriever(), ProposingModel()
    response = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), retriever, model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("请协调新版休假政策的落地工作，并确认执行安排。", Role.HR)

    assert response.status == PlanStatus.PLANNED
    assert retriever.calls == [("休假政策", Role.HR), ("休假政策 实施安排", Role.HR)]
    request = json.loads(model.messages[0][-1].content)
    assert {item["chunk_id"] for item in request["evidence"]} == {"chunk-1", "chunk-2"}


def test_preflight_missing_intent_coverage_requests_repair_before_composition(tmp_path) -> None:
    class FacetRetriever:
        def search(self, query, _role):
            review = RetrievalCandidate("chunk-1", "leave", 2, "年假制度", "leave.md", "第3条", 0.9, "先核对员工适用资格和年假余额")
            implementation = RetrievalCandidate("chunk-2", "leave", 2, "年假制度", "leave.md", "实施附录", 0.9, "配置审批流程并通知员工")
            return RetrievalResult([review, implementation] if "实施安排" in query else [review], 0)

    incomplete = PlanningDraft(
        action="propose", used_chunk_ids=["chunk-1", "chunk-2"], todos=[{
            "todo_id": "review", "title": "核对员工适用资格和年假余额",
            "detail": "核对员工适用资格和年假余额",
            "chunk_ids": ["chunk-1"],
        }],
    )
    complete = PlanningDraft(
        action="propose", used_chunk_ids=["chunk-1", "chunk-2"],
        action_intents=[
            {"intent_id": "review", "description": "核对员工适用资格和年假余额", "chunk_ids": ["chunk-1"]},
            {"intent_id": "implement", "description": "配置审批流程并通知员工", "chunk_ids": ["chunk-2"]},
        ],
    )

    class IncompleteModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return PlannerDecision(
                action="propose", proposal=incomplete if self.calls == 1 else complete,
            )

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    model = IncompleteModel()
    response = PolicyRolloutPlanningService(
        store, FacetRetriever(), model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True, max_repairs=1,
    ).create("请协调新版休假政策的落地工作，并确认执行安排。", Role.HR)

    assert response.status == PlanStatus.PLANNED
    todos = response.plan_versions[0].todos
    assert {citation.chunk_id for todo in todos for citation in todo.citations} == {"chunk-1", "chunk-2"}
    assert any(todo.depends_on for todo in todos if todo.todo_id.startswith("grounded-implement-"))
    trace = store.decision_trace(response.task_id)
    assert [event["outcome"] for event in trace[-2:]] == ["constrained_composer", "planned"]
    assert trace[-1]["constraint_code"] == "grounded_composer_after_validation"
    assert trace[-1]["repairs"] == 1 and model.calls == 2
    assert any(
        event["outcome"] == "requested"
        and event["constraint_code"] == "incomplete_evidence_actions"
        for event in trace
    )


def test_action_intents_are_composed_into_grounded_todos_without_model_repair(tmp_path) -> None:
    class FacetRetriever:
        def search(self, query, _role):
            review = RetrievalCandidate("chunk-1", "leave", 2, "年假制度", "leave.md", "第3条", 0.9, "先核对员工适用资格和年假余额")
            implementation = RetrievalCandidate("chunk-2", "leave", 2, "年假制度", "leave.md", "实施附录", 0.9, "配置审批流程并通知员工")
            return RetrievalResult([review, implementation] if "实施安排" in query else [review], 0)

    proposal = PlanningDraft(
        action="propose",
        used_chunk_ids=["chunk-1", "chunk-2"],
        action_intents=[
            {"intent_id": "review", "description": "核对员工适用资格和年假余额", "chunk_ids": ["chunk-1"]},
            {"intent_id": "implement", "description": "配置审批流程并通知员工", "chunk_ids": ["chunk-2"]},
        ],
    )

    class IntentModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return PlannerDecision(action="propose", proposal=proposal)

    model = IntentModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FacetRetriever(), model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("请协调新版休假政策的落地工作，并确认执行安排。", Role.HR)

    assert response.status == PlanStatus.PLANNED and model.calls == 1
    assert {citation.chunk_id for todo in response.plan_versions[0].todos for citation in todo.citations} == {"chunk-1", "chunk-2"}
    assert any(todo.depends_on for todo in response.plan_versions[0].todos if todo.todo_id.startswith("grounded-implement-"))
    trace = store.decision_trace(response.task_id)
    assert [event["outcome"] for event in trace[-2:]] == ["constrained_composer", "planned"]
    assert trace[-1]["repairs"] == 0


def test_constrained_composer_uses_clarification_as_scope_not_policy_evidence(tmp_path) -> None:
    proposal = PlanningDraft(
        action="propose", used_chunk_ids=["leave:v2:chunk-01"],
        action_intents=[{
            "intent_id": "review", "description": "核对年假规则",
            "chunk_ids": ["leave:v2:chunk-01"],
        }],
    )

    class ContinuationModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return PlannerDecision(action="search", search_query="年假制度")
            return PlannerDecision(action="propose", proposal=proposal)

    model = ContinuationModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, FakeRetriever(), model,
        citation_checker=lambda _citations, _role: True,
    )
    clarification = service.create(
        "请制定年假制度落地计划；首批试点部门尚未确定。", Role.HR,
    )
    response = service.continue_task(
        clarification.task_id, Role.HR, ["首批试点部门为研发部"],
    )

    assert response is not None and response.status == PlanStatus.PLANNED
    todo = response.plan_versions[0].todos[0]
    assert "用户已确认的业务约束（非制度证据）：首批试点部门为研发部" in todo.detail
    assert "任务范围（非制度证据）" in todo.detail
    assert [citation.chunk_id for citation in todo.citations] == ["leave:v2:chunk-01"]
    trace = store.decision_trace(response.task_id)
    assert trace[-2]["outcome"] == "constrained_composer"
    assert trace[-1]["repairs"] == 0


def test_action_intent_labels_and_used_chunks_are_safely_rehydrated() -> None:
    draft = PlanningDraft.model_validate({
        "action": "propose",
        "action_intents": [{
            "description": "核对员工适用资格和年假余额",
            "chunk_ids": ["chunk-1"],
        }],
    })
    assert draft.used_chunk_ids == ["chunk-1"]
    assert draft.action_intents[0].intent_id is None


def test_empty_action_intent_noise_is_ignored_without_relaxing_negation(tmp_path) -> None:
    proposal = PlanningDraft(
        action="propose", used_chunk_ids=["leave:v2:chunk-01"],
        action_intents=[
            {"intent_id": "review", "description": "核对年假规则", "chunk_ids": ["leave:v2:chunk-01"]},
            {"intent_id": "noise", "description": "根据制度要求完成安排", "chunk_ids": ["leave:v2:chunk-01"]},
        ],
    )

    class IntentModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return PlannerDecision(action="propose", proposal=proposal)

    model = IntentModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FakeRetriever(), model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("请根据年假制度制定计划。", Role.HR)

    assert response.status == PlanStatus.PLANNED and model.calls == 1
    trace = store.decision_trace(response.task_id)
    assert trace[-1]["constraint_code"] == "grounded_composer_after_validation"


def test_selected_evidence_only_is_composed_deterministically_without_free_text(tmp_path) -> None:
    proposal = PlanningDraft(
        action="propose", used_chunk_ids=["leave:v2:chunk-01"],
        action_intents=[
            {"intent_id": "noise", "description": "按既定范围形成清单", "chunk_ids": ["leave:v2:chunk-01"]},
        ],
    )

    class IntentModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return PlannerDecision(action="propose", proposal=proposal)

    model = IntentModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FakeRetriever(), model,
        citation_checker=lambda _citations, _role: True,
        preflight_named_policy_search=True,
    ).create("请根据年假制度制定计划。", Role.HR)

    assert response.status == PlanStatus.PLANNED and model.calls == 1
    todo = response.plan_versions[0].todos[0]
    assert "按既定范围形成清单" not in todo.title
    assert "按既定范围形成清单" not in todo.detail


@pytest.mark.parametrize(
    ("proposal", "expected_code"),
    [
        (
            PlanningDraft(
                action="propose", used_chunk_ids=["leave:v2:chunk-01"],
                action_intents=[{
                    "intent_id": "purchase", "description": "采购年假主题礼品",
                    "chunk_ids": ["leave:v2:chunk-01"],
                }],
            ),
            "unsupported_todo_content",
        ),
        (
            PlanningDraft(
                action="propose", used_chunk_ids=["leave:v2:chunk-01"],
                action_intents=[{
                    "intent_id": "negated", "description": "不要核对年假规则",
                    "chunk_ids": ["leave:v2:chunk-01"],
                }],
            ),
            "unsupported_todo_content",
        ),
        (
            PlanningDraft(
                action="propose", used_chunk_ids=["fabricated"],
                action_intents=[{
                    "intent_id": "review", "description": "核对年假规则",
                    "chunk_ids": ["fabricated"],
                }],
            ),
            "invalid_citations",
        ),
    ],
)
def test_constrained_composer_rejects_unsupported_actions_and_fabricated_chunks(
    tmp_path, proposal, expected_code,
) -> None:
    class IntentModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return PlannerDecision(action="search", search_query="年假制度")
            return PlannerDecision(action="propose", proposal=proposal)

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FakeRetriever(), IntentModel(),
        citation_checker=lambda _citations, _role: True, max_repairs=0,
    ).create("请根据年假制度制定计划。", Role.HR)

    assert response.status == PlanStatus.FAILED and response.plan_versions == []
    trace = store.decision_trace(response.task_id)
    assert trace[-1]["constraint_code"] == expected_code
    assert all(event["outcome"] != "constrained_composer" for event in trace)


@pytest.mark.parametrize("evidence", [
    "制度规定：不得在未经法务与安全委员会书面批准的任何情况下向外部发送员工资料。",
    "未经法务书面批准，不得向外部发送员工资料。",
    "除非获得法务书面批准，否则不得向外部发送员工资料。",
    "仅当法务书面批准时，才可向外部发送员工资料。",
    "在取得法务书面批准的前提下，方可向外部发送员工资料。",
    "经法务书面批准，可以向外部发送员工资料。",
    "如果获得法务批准，则可以向外部发送员工资料。",
    "若主管同意，可向外部发送员工资料。",
    "获得书面授权后向外部发送员工资料。",
])
def test_conditional_or_negative_clauses_never_ground_positive_actions(evidence) -> None:
    assert not _high_level_action_is_grounded("向外部发送员工资料", [evidence])
    assert not _high_level_action_is_grounded("批准后向外部发送员工资料", [evidence])


@pytest.mark.parametrize(("intent", "evidence"), [
    ("访问薪酬系统", "员工无权访问薪酬系统。"),
    ("读取董事会文件", "员工没有权限读取董事会文件。"),
    ("查阅董事会文件", "员工未获权限查阅董事会文件。"),
    ("访问薪酬系统", "薪酬系统仅限管理员访问。"),
    ("读取董事会文件", "董事会文件访问权限待确认，暂不读取。"),
])
def test_denied_limited_or_unknown_acl_sentences_never_ground_access(intent, evidence) -> None:
    assert not _high_level_action_is_grounded(intent, [evidence])


def test_positive_leave_actions_remain_grounded_after_clause_level_filtering() -> None:
    assert _high_level_action_is_grounded(
        "核对员工适用资格和年假余额", ["先核对员工适用资格和年假余额。"],
    )
    assert _high_level_action_is_grounded(
        "配置审批流程并通知员工", ["配置审批流程并通知员工。"],
    )
    mixed = "人力资源部门应核对员工适用资格。未经批准不得向外部发送员工资料。"
    assert _high_level_action_is_grounded("核对员工适用资格", [mixed])
    assert not _high_level_action_is_grounded("向外部发送员工资料", [mixed])


def test_negative_evidence_cannot_be_overridden_by_positive_goal_in_todo_validation() -> None:
    assert not _todo_is_supported(
        "配置审批流程并通知员工", "向员工通知新的审批流程。",
        ["制度明确规定：不得配置审批流程或通知员工。"],
        ["请配置审批流程并通知员工。"],
    )


def test_pasted_evidence_excerpt_cannot_self_prove_todo_action_coverage() -> None:
    tools = BoundPlanningTools(FakeRetriever(), Role.HR, 1)
    tools.search_authorized_evidence("年假制度")
    draft = PlanningDraft(
        action="propose", used_chunk_ids=["leave:v2:chunk-01"], todos=[{
            "todo_id": "vague", "title": "制定年假落地计划",
            "detail": "任务范围：年假制度。授权制度依据：人力资源部门应核对年假规则，年假为十五天。",
            "chunk_ids": ["leave:v2:chunk-01"],
        }],
    )

    with pytest.raises(PlanningError, match="unsupported_todo_content"):
        PolicyRolloutPlanningService._validated_plan(
            draft, tools, goal="请制定年假落地计划。", answers=[],
        )


@pytest.mark.parametrize("dangerous_text", ["外传员工隐私", "泄露员工数据", "制造危险物"])
def test_legacy_model_todos_are_rebuilt_without_unknown_dangerous_text(
    tmp_path, dangerous_text,
) -> None:
    proposal = _planned()
    proposal["todos"][0].update(
        title=f"{dangerous_text}并核对年假资格",
        detail="核对年假规则。",
    )
    service = _service(tmp_path, [proposal], max_repairs=0)

    response = service.create("请根据年假制度制定核对计划。", Role.HR)

    assert response.status == PlanStatus.PLANNED
    persisted = json.dumps(
        [todo.model_dump(mode="json") for todo in response.plan_versions[0].todos],
        ensure_ascii=False,
    )
    assert dangerous_text not in persisted
    trace = service.store.decision_trace(response.task_id)
    assert [event["outcome"] for event in trace[-2:]] == ["constrained_composer", "planned"]
    assert trace[-1]["constraint_code"] == "grounded_composer_after_validation"


def test_composer_fails_closed_for_negative_authorized_evidence(tmp_path) -> None:
    class NegativeRetriever:
        def search(self, _query, _role):
            return RetrievalResult([RetrievalCandidate(
                "negative:chunk-1", "negative", 1, "员工资料制度", "negative.md", "第8条", 0.9,
                "制度规定：不得在未经法务与安全委员会书面批准的任何情况下向外部发送员工资料。",
            )], 0)

    proposal = PlanningDraft(
        action="propose", used_chunk_ids=["negative:chunk-1"],
        action_intents=[{
            "intent_id": "send", "description": "向外部发送员工资料",
            "chunk_ids": ["negative:chunk-1"],
        }],
    )

    class IntentModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return PlannerDecision(action="search", search_query="员工资料发送制度")
            return PlannerDecision(action="propose", proposal=proposal)

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, NegativeRetriever(), IntentModel(),
        citation_checker=lambda _citations, _role: True, max_repairs=0,
    ).create("请根据员工资料制度制定计划。", Role.HR)

    assert response.status == PlanStatus.FAILED and response.plan_versions == []
    trace = store.decision_trace(response.task_id)
    assert trace[-1]["constraint_code"] == "unsupported_todo_content"
    assert all(event["outcome"] != "constrained_composer" for event in trace)


def test_emergency_fallback_fails_closed_for_negated_implementation_actions(tmp_path) -> None:
    class NegativeRetriever:
        def search(self, _query, _role):
            return RetrievalResult([RetrievalCandidate(
                "negative:chunk-1", "negative", 1, "年假制度", "negative.md", "第8条", 0.9,
                "制度明确规定：不得配置审批流程或通知员工。",
            )], 0)

    class SchemaFailureModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return PlannerDecision(action="search", search_query="年假制度")
            raise PlanningError("schema_validation_failed")

    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, NegativeRetriever(), SchemaFailureModel(),
        citation_checker=lambda _citations, _role: True,
    ).create("请根据年假制度制定实施计划。", Role.HR)

    assert response.status == PlanStatus.FAILED and response.plan_versions == []
    trace = store.decision_trace(response.task_id)
    assert any(event["outcome"] == "grounded_fallback" for event in trace)
    assert trace[-1]["constraint_code"] == "unsupported_todo_content"


def test_continuation_can_rehydrate_same_query_once_then_propose(tmp_path) -> None:
    class SequenceModel:
        def __init__(self) -> None:
            self.decisions = iter([
                PlannerDecision(action="search", search_query="休假政策"),
                PlannerDecision(action="propose", proposal=_planned(["研发部"])),
            ])
            self.messages = []

        def invoke(self, messages):
            self.messages.append(messages)
            return next(self.decisions)

    model = SequenceModel()
    retriever = FakeRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    goal = "落地休假制度"
    policy = QueryPolicy()
    routed = policy.classify(
        "休假政策", goal, business_terms=set(), missing_variables=[], answers=[],
    )
    assert routed.query is not None and routed.structure_fingerprint is not None
    old_claim = "old-claim"
    store.create("legacy-task", Role.HR, goal, PlanStatus.PLANNING, None, claim_token=old_claim)
    assert store.save_planning_progress(
        "legacy-task", Role.HR, old_claim, model_decisions=2, repairs=0,
        attempted_searches=1, executed_searches=1, deduplicated_searches=0,
        rejected_searches=0, seen_query_hashes={hash_identifier(routed.query.casefold())},
        seen_query_structures={routed.structure_fingerprint},
        seen_chunk_hashes={hash_identifier("leave:v2:chunk-01")}, query_rewrites=0,
    )
    clarification_state = PolicyRolloutClarification(questions=["适用哪些部门？"], answers=[])
    assert store.update_state(
        "legacy-task", Role.HR, PlanStatus.CLARIFICATION, clarification_state,
        claim_token=old_claim,
    )
    service = PolicyRolloutPlanningService(
        store, retriever, model, citation_checker=lambda _citations, _role: True,
    )
    response = service.continue_task("legacy-task", Role.HR, ["研发部"])
    assert response is not None and response.status == PlanStatus.PLANNED
    assert retriever.calls == [(routed.query, Role.HR)]
    trace = store.decision_trace("legacy-task")
    assert [event["outcome"] for event in trace] == [
        "rehydrated", "constrained_composer", "planned",
    ]
    assert trace[-1]["model_decisions"] == 4
    assert trace[-1]["attempted_searches"] == trace[-1]["executed_searches"] == 2
    continuation_request = json.loads(model.messages[0][-1].content)
    assert continuation_request["clarification_context"] == [
        {"question": "适用哪些部门？", "answer": "研发部"}
    ]
    assert "关键业务变量尚未确定并要求先确认" in model.messages[0][0].content


def test_explicit_missing_variable_clarifies_before_model_or_search(tmp_path) -> None:
    class SequenceModel:
        def __init__(self) -> None:
            self.decisions = iter([
                PlannerDecision(action="search", search_query="年假制度"),
                PlannerDecision(action="propose", proposal=_planned(["研发部"])),
            ])
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return next(self.decisions)

    model = SequenceModel()
    retriever = FakeRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, retriever, model, citation_checker=lambda _citations, _role: True,
    )
    clarification = service.create(
        "请制定分批落地计划；首批试点部门尚未确定，应先确认试点范围。", Role.HR,
    )
    assert clarification.status == PlanStatus.CLARIFICATION
    assert clarification.clarification is not None
    assert clarification.clarification.questions == ["请确认首批试点部门。"]
    assert model.calls == 0 and retriever.calls == []
    assert store.decision_trace(clarification.task_id)[-1]["constraint_code"] == "explicit_missing_business_variable"

    planned = service.continue_task(clarification.task_id, Role.HR, ["研发部"])
    assert planned is not None and planned.status == PlanStatus.PLANNED
    assert model.calls == 2 and retriever.calls == [("年假制度", Role.HR)]


def test_clarification_business_value_is_rejected_as_evidence_query(tmp_path) -> None:
    class SequenceModel:
        def __init__(self) -> None:
            self.decisions = iter([
                PlannerDecision(action="search", search_query="研发部 首批试点部门"),
                PlannerDecision(action="search", search_query="年假制度"),
                PlannerDecision(action="propose", proposal=_planned(["研发部"])),
            ])
            self.messages = []

        def invoke(self, messages):
            self.messages.append(messages)
            return next(self.decisions)

    model = SequenceModel()
    retriever = FakeRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, retriever, model, citation_checker=lambda _citations, _role: True,
    )
    clarification = service.create("首批试点部门尚未确定。", Role.HR)
    planned = service.continue_task(clarification.task_id, Role.HR, ["首批试点部门为研发部"])
    assert planned is not None and planned.status == PlanStatus.PLANNED
    assert retriever.calls == [("年假制度", Role.HR)]
    trace = store.decision_trace(clarification.task_id)
    assert [event["outcome"] for event in trace] == [
        "clarification", "rejected", "success", "constrained_composer", "planned",
    ]
    assert trace[1]["constraint_code"] == "business_input_is_not_evidence_query"
    corrected_request = json.loads(model.messages[1][-1].content)
    assert corrected_request["search_feedback"] == "business_input_is_not_evidence_query"


def test_continuation_second_search_is_rejected_before_terminal_proposal(tmp_path) -> None:
    class SequenceModel:
        def __init__(self) -> None:
            self.decisions = iter([
                PlannerDecision(action="search", search_query="休假政策"),
                PlannerDecision(action="search", search_query="员工休假规则"),
                PlannerDecision(action="propose", proposal=_planned(["全员"])),
            ])

        def invoke(self, _messages):
            return next(self.decisions)

    retriever = FakeRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    store.create(
        "clarified-task", Role.HR, "落地休假制度", PlanStatus.CLARIFICATION,
        PolicyRolloutClarification(questions=["适用范围？"], answers=[]),
    )
    service = PolicyRolloutPlanningService(
        store, retriever, SequenceModel(), citation_checker=lambda _citations, _role: True,
    )
    response = service.continue_task("clarified-task", Role.HR, ["全员"])
    assert response is not None and response.status == PlanStatus.PLANNED
    assert len(retriever.calls) == 1
    trace = store.decision_trace("clarified-task")
    rejected = next(
        event for event in trace
        if event["constraint_code"] == "search_after_clarification_answers"
    )
    assert rejected["outcome"] == "rejected"
    assert trace[-1]["attempted_searches"] == 2
    assert trace[-1]["executed_searches"] == 1
    assert trace[-1]["rejected_searches"] == 1


def test_model_decision_budget_is_shared_across_continuation(tmp_path) -> None:
    class ClarifyThenRefuseModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return PlannerDecision(action="clarify", clarification_questions=["适用范围？"])

    model = ClarifyThenRefuseModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    service = PolicyRolloutPlanningService(
        store, FakeRetriever(), model, citation_checker=lambda _citations, _role: True,
        max_model_decisions=1,
    )
    old_claim = "old-budget"
    store.create("budget-task", Role.HR, "落地休假制度", PlanStatus.PLANNING, None, claim_token=old_claim)
    assert store.save_planning_progress(
        "budget-task", Role.HR, old_claim, model_decisions=1, repairs=0,
        attempted_searches=0, executed_searches=0, deduplicated_searches=0,
        rejected_searches=0, seen_query_hashes=set(),
    )
    clarification_state = PolicyRolloutClarification(questions=["适用范围？"], answers=[])
    assert store.update_state(
        "budget-task", Role.HR, PlanStatus.CLARIFICATION, clarification_state,
        claim_token=old_claim,
    )
    response = service.continue_task("budget-task", Role.HR, ["全员"])
    assert response is not None and response.status == PlanStatus.CLARIFICATION
    assert model.calls == 0
    assert store.planning_progress("budget-task", Role.HR, "missing") is None


def test_normalized_duplicate_search_does_not_call_retriever_twice(tmp_path) -> None:
    class DuplicateSearchModel:
        def __init__(self) -> None:
            self.turn = 0
            self.messages = []

        def invoke(self, messages):
            self.turn += 1
            self.messages.append(messages)
            if self.turn == 3:
                return PlannerDecision(action="refuse", refusal_reason="done")
            return PlannerDecision(
                action="search", search_query=" 年假\t政策 " if self.turn == 1 else "年假 政策"
            )

    retriever = FakeRetriever()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    model = DuplicateSearchModel()
    response = PolicyRolloutPlanningService(
        store, retriever, model, citation_checker=lambda _citations, _role: True,
    ).create("目标", Role.HR)
    assert response.status == PlanStatus.REFUSED
    assert retriever.calls == [("年假 政策", Role.HR)]
    assert [event["outcome"] for event in store.decision_trace(response.task_id)] == ["success", "deduplicated", "refused"]
    third_request = json.loads(model.messages[2][-1].content)
    assert third_request["prior_search_queries"] == ["年假 政策"]
    assert third_request["search_feedback"] == "duplicate_search_query_terminal_only"
    assert "年假 政策" not in json.dumps(store.decision_trace(response.task_id), ensure_ascii=False)


def test_decision_budget_counts_failed_attempt_and_stops_before_next_invoke(tmp_path) -> None:
    class InvalidThenSearchModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return {"action": "search", "search_query": "x", "refusal_reason": "bad"}

    model = InvalidThenSearchModel()
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    response = PolicyRolloutPlanningService(
        store, FakeRetriever(), model, citation_checker=lambda _citations, _role: True,
        max_model_decisions=1, max_repairs=1,
    ).create("目标", Role.HR)
    assert response.status == PlanStatus.FAILED and model.calls == 1
    assert store.decision_trace(response.task_id)[-1]["constraint_code"] == "max_model_decisions_exceeded"


def test_terminal_decision_actions_and_evidence_stay_out_of_trace(tmp_path) -> None:
    class SequenceModel:
        def __init__(self, decisions) -> None:
            self.decisions = iter(decisions)

        def invoke(self, _messages):
            return next(self.decisions)

    clarification = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "clarify.sqlite3"), FakeRetriever(),
        SequenceModel([PlannerDecision(action="clarify", clarification_questions=["范围？"])]),
        citation_checker=lambda _citations, _role: True,
    ).create("适用范围待确认。", Role.HR)
    refusal_store = PolicyRolloutTaskStore(tmp_path / "refuse.sqlite3")
    refusal = PolicyRolloutPlanningService(
        refusal_store, FakeRetriever(),
        SequenceModel([PlannerDecision(action="search", search_query="SECRET_QUERY"), PlannerDecision(action="refuse", refusal_reason="SECRET_REASON")]),
        citation_checker=lambda _citations, _role: True,
    ).create("SECRET_GOAL", Role.HR)
    assert clarification.status == PlanStatus.CLARIFICATION and refusal.status == PlanStatus.REFUSED
    assert "SECRET" not in json.dumps(refusal_store.decision_trace(refusal.task_id), ensure_ascii=False)


def test_trace_persists_search_repair_and_terminal_events_without_text(tmp_path) -> None:
    invalid = _planned()
    invalid["todos"] = []
    service = _service(tmp_path, [invalid, _planned()])
    response = service.create("SECRET_GOAL", Role.HR)
    trace = service.store.decision_trace(response.task_id)
    assert [event["action"] for event in trace] == [
        "search", "propose", "repair", "search", "propose", "propose",
    ]
    assert all("SECRET_GOAL" not in json.dumps(event, ensure_ascii=False) for event in trace)
    assert trace[1]["outcome"] == "invalid_citations"
    assert trace[2]["constraint_code"] == "invalid_citations"
    assert trace[-2]["outcome"] == "constrained_composer"
    assert trace[-1]["outcome"] == "planned"


def test_repair_receives_prior_draft_and_allowlisted_code_only(tmp_path) -> None:
    invalid = _planned()
    invalid["todos"] = []
    model = RecordingPlanningModel([invalid, _planned()])
    service = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), FakeRetriever(), model,
        citation_checker=lambda _citations, _role: True,
    )
    assert service.create("目标", Role.HR).status == PlanStatus.PLANNED
    repair_payload = next(
        json.loads(messages[-1].content)
        for messages in model.messages
        if messages and isinstance(messages[-1].content, str) and '"repair"' in messages[-1].content
    )
    assert repair_payload["repair"]["constraint_codes"] == ["invalid_citations"]
    assert repair_payload["repair"]["prior_draft"]["todos"] == []
    assert "evidence" not in json.dumps(repair_payload["repair"], ensure_ascii=False)


def test_todo_content_must_be_supported_by_its_own_citations(tmp_path) -> None:
    unsupported = _planned()
    unsupported["todos"][0].update(title="订购年假主题生日蛋糕", detail="为年假活动采购蛋糕。")
    service = _service(tmp_path, [unsupported], max_repairs=0)
    response = service.create("落地休假制度", Role.HR)
    assert response.status == PlanStatus.FAILED and response.plan_versions == []
    trace = service.store.decision_trace(response.task_id)
    assert trace[-1]["constraint_code"] == "unsupported_todo_content"

    wrong_number = _planned()
    wrong_number["todos"][0]["detail"] = "按二十天假期执行，并记录为 20 天。"
    number_store = PolicyRolloutTaskStore(tmp_path / "numbers.sqlite3")
    number_service = PolicyRolloutPlanningService(
        number_store, FakeRetriever(), FakePlanningModel([wrong_number]),
        citation_checker=lambda _citations, _role: True, max_repairs=0,
    )
    numbered = number_service.create("落地休假制度", Role.HR)
    assert numbered.status == PlanStatus.FAILED
    assert number_store.decision_trace(numbered.task_id)[-1]["constraint_code"] == "unsupported_todo_content"


def test_stale_claim_cannot_append_trace_event(tmp_path) -> None:
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    store.create("task", Role.HR, "目标", PlanStatus.PLANNING, None, claim_token="live")
    assert not store.append_trace_event("task", Role.HR, "old", action="search", outcome="success", model_decisions=1, search_calls=1, repairs=0, constraint_code=None, duration_ms=0)
    assert store.decision_trace("task") == []


def test_role_isolation_never_returns_task_or_evidence(tmp_path) -> None:
    service = _service(tmp_path, [_planned()])
    task = service.create("只给 HR 的休假计划", Role.HR)
    assert service.get(task.task_id, Role.EMPLOYEE) is None
    assert service.continue_task(task.task_id, Role.EMPLOYEE, ["伪造答案"]) is None


def test_authorized_retrieval_tool_binds_role_and_hides_other_role_evidence() -> None:
    retriever = FakeRetriever(Role.HR)
    tools = BoundPlanningTools(retriever, Role.EMPLOYEE, max_tool_calls=3)
    assert json.loads(tools.search_authorized_evidence("休假")) == {"evidence": []}
    assert retriever.calls == [("休假", Role.EMPLOYEE)]
    assert tools.candidates == {}


def test_stale_or_revoked_citation_blocks_get_and_continuation(tmp_path) -> None:
    checker_calls: list[tuple[int, Role]] = []
    valid = True
    def checker(citations, role):
        checker_calls.append((len(citations), role))
        return valid
    service = PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), FakeRetriever(), FakePlanningModel([_planned()]),
        citation_checker=checker,
    )
    task = service.create("落地休假制度", Role.HR)
    assert task.status == PlanStatus.PLANNED
    valid = False
    assert service.get(task.task_id, Role.HR) is None
    assert service.continue_task(task.task_id, Role.HR, ["全员"]) is None
    assert checker_calls and all(call == (1, Role.HR) for call in checker_calls)


def test_store_claim_allows_only_one_continuation_and_one_version(tmp_path) -> None:
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    store.create("task", Role.HR, "目标", PlanStatus.CLARIFICATION, None)
    clarification = PolicyRolloutClarification(questions=["范围？"], answers=["全员"])
    claim_token = "claim-one"
    assert store.claim_clarification("task", Role.HR, clarification, claim_token=claim_token)
    assert not store.claim_clarification("task", Role.HR, clarification, claim_token="claim-two")
    plan = PolicyRolloutPlanVersion(
        version=0,
        todos=[PolicyRolloutTodo(
            todo_id="review", title="核对", detail="核对规则。",
            citations=[TodoCitation(
                chunk_id="leave:v2:chunk-01", document_id="leave", document_version=2,
                source_file_name="leave-v2.md", source_locator="年假",
            )],
        )],
        created_at=task_time(),
    )
    completed = store.complete_claim_and_append_plan("task", Role.HR, plan, clarification, claim_token)
    assert completed is not None and completed.version == 1
    assert store.complete_claim_and_append_plan("task", Role.HR, plan, clarification, claim_token) is None


def test_stale_claim_token_cannot_complete_new_claim(tmp_path) -> None:
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    now = task_time()
    clarification = PolicyRolloutClarification(questions=["范围？"], answers=["全员"])
    old_token = "old-claim"
    store.create("task", Role.HR, "目标", PlanStatus.PLANNING, clarification, claim_token=old_token, now=now)
    assert store.reclaim_stale_planning("task", Role.HR, 1, now=now.replace(year=now.year + 1))
    new_token = "new-claim"
    assert store.claim_clarification("task", Role.HR, clarification, claim_token=new_token)
    plan = PolicyRolloutPlanVersion(version=0, todos=[PolicyRolloutTodo(todo_id="review", title="核对", detail="核对规则。", citations=[TodoCitation(chunk_id="leave:v2:chunk-01", document_id="leave", document_version=2, source_file_name="leave-v2.md", source_locator="年假")])], created_at=task_time())
    assert store.complete_claim_and_append_plan("task", Role.HR, plan, clarification, old_token) is None
    assert not store.release_claim("task", Role.HR, clarification, old_token)
    assert store.complete_claim_and_append_plan("task", Role.HR, plan, clarification, new_token) is not None
    reopened = store.get("task", Role.HR, "reopen")
    assert reopened is not None and reopened.status == PlanStatus.PLANNED
    assert [version.version for version in reopened.plan_versions] == [1]


def test_stale_claim_token_cannot_terminally_overwrite_new_claim(tmp_path) -> None:
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    now = task_time()
    clarification = PolicyRolloutClarification(questions=["范围？"], answers=["全员"])
    store.create("task", Role.HR, "目标", PlanStatus.PLANNING, clarification, claim_token="old", now=now)
    assert store.reclaim_stale_planning("task", Role.HR, 1, now=now.replace(year=now.year + 1))
    assert store.claim_clarification("task", Role.HR, clarification, claim_token="new")
    assert not store.update_state("task", Role.HR, PlanStatus.REFUSED, None, claim_token="old")
    assert store.get("task", Role.HR, "reopen").status == PlanStatus.PLANNING


def test_stale_planning_with_invalid_clarification_fails(tmp_path) -> None:
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    now = task_time()
    store.create("task", Role.HR, "目标", PlanStatus.PLANNING, None, claim_token="expired", now=now)
    with store._connection() as connection:
        connection.execute("UPDATE policy_rollout_tasks SET clarification_json = ? WHERE task_id = ?", ("{}", "task"))
    assert store.reclaim_stale_planning("task", Role.HR, 1, now=now.replace(year=now.year + 1))
    recovered = store.get("task", Role.HR, "reopen")
    assert recovered is not None and recovered.status == PlanStatus.FAILED


def task_time():
    from datetime import UTC, datetime
    return datetime.now(UTC)


def test_planning_read_store_never_ensures_collection(monkeypatch) -> None:
    import rag_kb.api as api
    created = []
    class FakeStore:
        def __init__(self, url, collection) -> None:
            created.append((url, collection))
        def ensure_collection(self, _dim) -> None:
            raise AssertionError("planning builder must not write Qdrant metadata")
    monkeypatch.setattr(api, "ChunkStore", FakeStore)
    monkeypatch.setattr(api, "get_settings", lambda: type("SettingsStub", (), {"qdrant_url": "http://qdrant", "qdrant_collection": "chunks"})())
    api._build_read_store()
    assert created == [("http://qdrant", "chunks")]


def test_planning_builder_uses_structured_model_not_policy_compare_agent(tmp_path, monkeypatch) -> None:
    import rag_kb.api as api
    built = []

    class PlannerModel:
        def __init__(self, settings) -> None:
            built.append(settings)

    monkeypatch.setattr(api, "DeepSeekPlannerDecisionModel", PlannerModel)
    monkeypatch.setattr(api, "_build_read_store", lambda: type("ReadStore", (), {"citations_currently_authorized": lambda *_args: True})())
    monkeypatch.setattr(api, "BgeM3Embedder", lambda: object())
    monkeypatch.setattr(api, "Retriever", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "PolicyRolloutTaskStore", lambda _path: object())
    settings = type("SettingsStub", (), {
        "retrieval_top_k": 1, "retrieval_score_threshold": None, "agent_max_steps": 1,
        "agent_max_tool_calls": 1, "planner_max_model_decisions": 1, "planner_max_repairs": 0,
        "policy_rollout_claim_ttl_seconds": 1, "policy_rollout_db_path": tmp_path / "tasks.sqlite3",
    })()
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    api.build_policy_rollout_planning_service.cache_clear()
    try:
        assert isinstance(api.build_policy_rollout_planning_service().model, PlannerModel)
        assert built == [settings]
    finally:
        api.build_policy_rollout_planning_service.cache_clear()


def test_task_endpoints_enforce_token_role_and_validation(tmp_path, monkeypatch) -> None:
    import rag_kb.api as api
    service = _service(tmp_path, [_planned()])
    monkeypatch.setattr(api, "get_settings", lambda: type("SettingsStub", (), {"demo_role_tokens": {"hr": "hr-token", "employee": "employee-token"}})())
    api.app.dependency_overrides[api.build_policy_rollout_planning_service] = lambda: service
    try:
        client = TestClient(api.app)
        body = {"goal": "落地休假制度", "role": "hr"}
        assert client.post("/policy-rollout-tasks", json=body).status_code == 403
        assert client.post("/policy-rollout-tasks", json={"goal": "", "role": "hr"}, headers={"X-Demo-Role-Token": "hr-token"}).status_code == 422
        created = client.post("/policy-rollout-tasks", json=body, headers={"X-Demo-Role-Token": "hr-token"})
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        assert client.get(f"/policy-rollout-tasks/{task_id}", headers={"X-Demo-Role-Token": "employee-token"}).status_code == 403
    finally:
        api.app.dependency_overrides.clear()
