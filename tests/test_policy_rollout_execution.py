"""批准后的本地草稿执行测试；完全使用 tmp_path 与假引用哈希。"""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from rag_kb.models import (
    ExecutionStatus, PlanStatus, PolicyRolloutClarification, PolicyRolloutPlanVersion,
    PolicyRolloutExecutionResponse, PolicyRolloutTodo, Role, TodoCitation,
)
from rag_kb.policy_rollout_execution import ExecutionConflict, PolicyRolloutExecutionService, canonical_plan_digest
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore


def _plan() -> PolicyRolloutPlanVersion:
    return PolicyRolloutPlanVersion(version=0, created_at=datetime.now(UTC), todos=[PolicyRolloutTodo(
        todo_id="review", title="核对", detail="核对制度。", citations=[TodoCitation(
            chunk_id="leave:v2:chunk-01", document_id="leave", document_version=2,
            source_file_name="leave-v2.md", source_locator="年假",
        )],
    )])


def _prepared(tmp_path, hashes=lambda citations, _role: [(citation.chunk_id, "sha256:current") for citation in citations]):
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    store.create("task", Role.HR, "落地休假制度", PlanStatus.CLARIFICATION, None)
    clarification = PolicyRolloutClarification(questions=["范围？"], answers=["全员"])
    claim_token = "prepared-claim"
    assert store.claim_clarification("task", Role.HR, clarification, claim_token=claim_token)
    plan = store.complete_claim_and_append_plan("task", Role.HR, _plan(), clarification, claim_token)
    assert plan is not None
    return store, PolicyRolloutExecutionService(store, tmp_path / "drafts", hashes)


def test_schema_migration_is_idempotent_and_adds_claim_fields(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE policy_rollout_tasks(task_id TEXT PRIMARY KEY, role TEXT NOT NULL, goal TEXT NOT NULL, status TEXT NOT NULL, clarification_json TEXT, created_at TEXT NOT NULL)")
    PolicyRolloutTaskStore(path)
    PolicyRolloutTaskStore(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(policy_rollout_tasks)")}
        assert {"claimed_at", "claim_token"} <= columns


def test_stale_planning_reclaims_but_active_claim_does_not(tmp_path) -> None:
    store = PolicyRolloutTaskStore(tmp_path / "tasks.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store.create("task", Role.HR, "目标", PlanStatus.PLANNING, None, claim_token="initial", now=now)
    assert not store.reclaim_stale_planning("task", Role.HR, 60, now=now + timedelta(seconds=59))
    assert store.reclaim_stale_planning("task", Role.HR, 60, now=now + timedelta(seconds=61))
    assert store.get("task", Role.HR, "test").status == PlanStatus.FAILED


def test_approval_requires_current_evidence_and_is_idempotent(tmp_path) -> None:
    hashes = [("leave:v2:chunk-01", "sha256:current")]
    _store, service = _prepared(tmp_path, lambda *_: hashes)
    first = service.approve("task", 1, Role.HR)
    assert first is not None and service.approve("task", 1, Role.HR).model_dump() == first.model_dump()
    hashes[:] = [("leave:v2:chunk-01", "sha256:changed")]
    with pytest.raises(Exception):
        service.approve("task", 1, Role.HR)


def test_no_approval_or_stale_digest_creates_no_file(tmp_path) -> None:
    _store, service = _prepared(tmp_path)
    with pytest.raises(ExecutionConflict):
        service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    assert not (tmp_path / "drafts").exists()


def test_same_key_creates_exactly_one_draft_and_reconciles_crash(tmp_path) -> None:
    store, service = _prepared(tmp_path)
    assert service.approve("task", 1, Role.HR)
    first = service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    second = service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    assert first.status == second.status == ExecutionStatus.SUCCEEDED
    assert len(list((tmp_path / "drafts").glob("*.json"))) == 1
    store.update_execution_status(first.execution_id, ExecutionStatus.RUNNING)
    recovered = service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    assert recovered is not None and recovered.status == ExecutionStatus.SUCCEEDED


def test_unknown_conflicting_draft_fails_closed_without_overwrite(tmp_path) -> None:
    _store, service = _prepared(tmp_path)
    assert service.approve("task", 1, Role.HR)
    result = service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    path = tmp_path / "drafts" / f"{result.work_item_id}.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ExecutionConflict):
        service.get_execution("task", Role.HR, idempotency_key="same", execution_id=None)
    assert path.read_text(encoding="utf-8") == "{}"


def test_execution_endpoints_enforce_auth_and_preserve_existing_contract(tmp_path, monkeypatch) -> None:
    import rag_kb.api as api
    _store, service = _prepared(tmp_path)
    monkeypatch.setattr(api, "get_settings", lambda: type("SettingsStub", (), {"demo_role_tokens": {"hr": "hr-token", "employee": "employee-token"}})())
    api.app.dependency_overrides[api.build_policy_rollout_execution_service] = lambda: service
    try:
        client = TestClient(api.app)
        approval = "/policy-rollout-tasks/task/versions/1/approval"
        assert client.post(approval, json={"version": 1}).status_code == 403
        assert client.post(approval, json={"version": 1}, headers={"X-Demo-Role-Token": "hr-token"}).status_code == 200
        execution = "/policy-rollout-tasks/task/versions/1/executions"
        body = {"version": 1, "idempotency_key": "once", "action": "create_local_draft_work_item"}
        assert client.post(execution, json=body, headers={"X-Demo-Role-Token": "employee-token"}).status_code == 403
        response = client.post(execution, json=body, headers={"X-Demo-Role-Token": "hr-token"})
        assert response.status_code == 200
        assert response.json()["draft_path"] == f"{response.json()['work_item_id']}.json"
        assert str(tmp_path) not in response.json()["draft_path"] and ".." not in response.json()["draft_path"]
    finally:
        api.app.dependency_overrides.clear()


def test_chunk_hash_association_changes_digest(tmp_path) -> None:
    plan = _plan()
    first = canonical_plan_digest("目标", None, plan, [("chunk-a", "sha256:one"), ("chunk-b", "sha256:two")])
    swapped = canonical_plan_digest("目标", None, plan, [("chunk-a", "sha256:two"), ("chunk-b", "sha256:one")])
    assert first != swapped


def test_duplicate_execution_claim_returns_winning_row(tmp_path) -> None:
    store, service = _prepared(tmp_path)
    assert service.approve("task", 1, Role.HR)
    first = service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    duplicate = store.claim_execution(first.model_copy(update={"execution_id": "different", "status": ExecutionStatus.RUNNING}))
    assert duplicate.execution_id == first.execution_id


def test_post_claim_winner_mismatch_does_not_reconcile(tmp_path, monkeypatch) -> None:
    store, service = _prepared(tmp_path)
    assert service.approve("task", 1, Role.HR)
    winner = PolicyRolloutExecutionResponse(
        execution_id="winner", task_id="task", version=1, role=Role.HR,
        idempotency_key="same", action="create_local_draft_work_item",
        status=ExecutionStatus.RUNNING, digest="winner-digest", work_item_id="winner-work-item",
        draft_path="winner-work-item.json", created_at=datetime.now(UTC),
    ).model_copy(update={"version": 2, "role": Role.EMPLOYEE, "action": "other_action"})
    monkeypatch.setattr(store, "claim_execution", lambda _response: winner)

    with pytest.raises(ExecutionConflict, match="idempotency_conflict"):
        service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")

    assert winner.status == ExecutionStatus.RUNNING
    assert not (tmp_path / "drafts").exists()


def test_relative_draft_identifier_rejects_traversal(tmp_path) -> None:
    _store, service = _prepared(tmp_path)
    assert service.approve("task", 1, Role.HR)
    result = service.execute("task", 1, Role.HR, "same", "create_local_draft_work_item")
    result.draft_path = "../outside.json"
    with pytest.raises(ExecutionConflict):
        service._reconcile(result)
