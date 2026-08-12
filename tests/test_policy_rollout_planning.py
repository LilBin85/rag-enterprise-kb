"""策略落地规划的纯单元与 API 契约测试；不连接 Qdrant 或真实模型。"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from rag_kb.models import (
    PlanStatus, PolicyRolloutClarification, PolicyRolloutPlanVersion, PolicyRolloutTodo,
    Role, TodoCitation,
)
from rag_kb.policy_rollout_planning import BoundPlanningTools, PolicyRolloutPlanningService
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore
from rag_kb.retriever import RetrievalCandidate, RetrievalResult


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
            score=0.9, text="年假为十五天。"
        )], 0)


class FakePlanningModel:
    def __init__(self, drafts: list[dict]) -> None:
        self.drafts = drafts
        self.turn = 0

    def bind_tools(self, _tools: list, **_kwargs):
        return self

    def invoke(self, _messages):
        self.turn += 1
        if self.turn % 2:
            return AIMessage(content="", tool_calls=[{"name": "search_authorized_evidence", "args": {"query": "休假政策"}, "id": str(self.turn)}])
        return AIMessage(content=json.dumps(self.drafts.pop(0), ensure_ascii=False))


def _planned() -> dict:
    return {"status": "planned", "used_chunk_ids": ["leave:v2:chunk-01"], "todos": [{"todo_id": "review", "title": "核对年假规则", "detail": "确认制度适用范围。", "depends_on": [], "chunk_ids": ["leave:v2:chunk-01"]}]}


def _service(tmp_path, drafts: list[dict], role: Role = Role.HR):
    return PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3"), FakeRetriever(role), FakePlanningModel(drafts),
        citation_checker=lambda _citations, _role: True,
    )


def test_valid_plan_persists_and_recovers_from_sqlite(tmp_path) -> None:
    service = _service(tmp_path, [_planned()])
    created = service.create("落地新版休假制度", Role.HR)
    recovered = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3").get(created.task_id, Role.HR, "recovery")
    assert created.status == PlanStatus.PLANNED
    assert recovered is not None and recovered.status == PlanStatus.PLANNED
    assert recovered.plan_versions[0].todos[0].citations[0].chunk_id == "leave:v2:chunk-01"


def test_clarification_continue_appends_immutable_plan_version(tmp_path) -> None:
    service = _service(tmp_path, [{"status": "clarification", "clarification_questions": ["适用哪些部门？"]}, _planned()])
    task = service.create("落地休假制度", Role.HR)
    first = service.continue_task(task.task_id, Role.HR, ["全员"])
    second = service.continue_task(task.task_id, Role.HR, ["全员"])
    assert task.status == PlanStatus.CLARIFICATION
    assert first is not None and [p.version for p in first.plan_versions] == [1]
    assert first.clarification is not None and first.clarification.answers == ["全员"]
    assert second is None
    reopened = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3").get(task.task_id, Role.HR, "reopen")
    assert reopened is not None and [p.version for p in reopened.plan_versions] == [1]
    assert reopened.plan_versions[0].model_dump() == first.plan_versions[0].model_dump()


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
    assert store.claim_clarification("task", Role.HR, clarification)
    assert not store.claim_clarification("task", Role.HR, clarification)
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
    completed = store.complete_claim_and_append_plan("task", Role.HR, plan, clarification)
    assert completed is not None and completed.version == 1
    assert store.complete_claim_and_append_plan("task", Role.HR, plan, clarification) is None
    reopened = store.get("task", Role.HR, "reopen")
    assert reopened is not None and reopened.status == PlanStatus.PLANNED
    assert [version.version for version in reopened.plan_versions] == [1]


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
