"""Stage 6 通用任务 HTTP 契约：所有依赖均由内存假件注入。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from rag_kb.application.approval_service import ApprovalServiceError
from rag_kb.application.task_query_service import (
    TaskArtifactResult,
    TaskEventResult,
    TaskQueryError,
)
from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.identity import Principal
from rag_kb.domain.tasks import TaskMode, TaskNode, TaskState, TaskStatus
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database.task_repository import TaskRepositoryError
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import (
    build_approval_service,
    build_identity_provider,
    build_task_query_service,
    build_task_repository,
)


def _state(
    *,
    status: TaskStatus = TaskStatus.PLANNING,
    state_version: int = 1,
    current_node: TaskNode = TaskNode.LOAD_TASK,
    pending_question: str | None = None,
    cancelled: bool = False,
) -> TaskState:
    return TaskState(
        task_id="task-1",
        auth_context_ref="sha256:internal-auth-context",
        goal="核对上线制度",
        task_mode=TaskMode.RESEARCH,
        status=status,
        current_node=current_node,
        state_version=state_version,
        plan_revision=2,
        pending_question=pending_question,
        repeated_action_fingerprints=("sha256:internal-action",),
        query_fingerprints=("sha256:internal-query",),
        result_fingerprints=("sha256:internal-result",),
        cancelled=cancelled,
    )


class FakeTaskRepository:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.resumed: list[dict[str, object]] = []
        self.cancelled: list[dict[str, object]] = []
        self.current = _state()

    def create_task(self, **kwargs: object) -> TaskState:
        self.created.append(kwargs)
        return self.current

    def get_task(self, *, auth, task_id: str) -> TaskState:
        if task_id != "task-1" or auth.tenant_id != "tenant-a" or auth.user_id != "owner-a":
            raise TaskRepositoryError("task_not_found")
        return self.current

    def resume_user(self, **kwargs: object) -> TaskState:
        if kwargs["task_id"] == "missing":
            raise TaskRepositoryError("task_not_found")
        if kwargs["expected_state_version"] != self.current.state_version:
            raise TaskRepositoryError("task_state_conflict")
        self.resumed.append(kwargs)
        self.current = _state(
            status=TaskStatus.PLANNING,
            state_version=self.current.state_version + 1,
            current_node=TaskNode.UPDATE_PLAN,
        )
        return self.current

    def cancel(self, **kwargs: object) -> TaskState:
        if kwargs["expected_state_version"] != self.current.state_version:
            raise TaskRepositoryError("task_state_conflict")
        self.cancelled.append(kwargs)
        self.current = _state(
            status=TaskStatus.CANCELLED,
            state_version=self.current.state_version + 1,
            current_node=TaskNode.TERMINAL,
            cancelled=True,
        )
        return self.current


class FakeTaskQueryService:
    def _check(self, auth, task_id: str) -> None:
        if not auth.has_permission("task:read"):
            raise TaskQueryError("task_read_denied")
        if task_id != "task-1" or auth.tenant_id != "tenant-a" or auth.user_id != "owner-a":
            raise TaskQueryError("task_not_found")

    def events(self, *, auth, task_id: str):
        self._check(auth, task_id)
        return (
            TaskEventResult(
                sequence=0,
                event_type="task_created",
                stage="load_task",
                status="planning",
                counts={"model_calls": 0},
                token_usage=0,
                error_code=None,
                created_at=datetime(2026, 8, 22, tzinfo=UTC),
            ),
        )

    def artifacts(self, *, auth, task_id: str):
        self._check(auth, task_id)
        return (
            TaskArtifactResult(
                artifact_id="artifact-1",
                artifact_type="report",
                storage_key="artifacts/internal/location.md",
                content_hash="sha256:internal-content",
                status="ready",
                created_at=datetime(2026, 8, 22, tzinfo=UTC),
            ),
        )


class FakeApprovalService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def approve(self, **kwargs: object) -> ApprovalBinding:
        if not kwargs["auth"].has_permission("task:approve"):
            raise ApprovalServiceError("approver_permission_denied")
        if kwargs["approval_id"] == "missing":
            raise ApprovalServiceError("approval_not_found")
        if kwargs["expected_task_id"] != "task-1" or kwargs["expected_plan_revision"] != 2:
            raise ApprovalServiceError("approval_plan_binding_invalid")
        self.calls.append(kwargs)
        now = datetime.now(UTC)
        return ApprovalBinding(
            approval_id="approval-1",
            task_id="task-1",
            tenant_id="tenant-a",
            plan_revision=2,
            plan_digest="sha256:internal-plan",
            tool_name="create_report_artifact",
            tool_version="1.0.0",
            canonical_arguments_hash="sha256:internal-arguments",
            principal_hash="sha256:internal-principal",
            evidence_hash="sha256:internal-evidence",
            expires_at=now + timedelta(minutes=5),
            approved_by="approver-a",
            approved_at=now,
        )


@pytest.fixture()
def setup_app() -> Iterator[tuple[TestClient, FakeTaskRepository, FakeTaskQueryService, FakeApprovalService]]:
    app = create_app()
    repository = FakeTaskRepository()
    query_service = FakeTaskQueryService()
    approval_service = FakeApprovalService()
    provider = DemoTokenIdentityProvider(
        {
            "owner-token": Principal(
                tenant_id="tenant-a",
                user_id="owner-a",
                attributes={"permissions": ["task:create", "task:read"]},
            ),
            "approver-token": Principal(
                tenant_id="tenant-a",
                user_id="approver-a",
                attributes={"permissions": ["task:approve"]},
            ),
            "outsider-token": Principal(
                tenant_id="tenant-b",
                user_id="owner-b",
                attributes={"permissions": ["task:read"]},
            ),
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_task_repository] = lambda: repository
    app.dependency_overrides[build_task_query_service] = lambda: query_service
    app.dependency_overrides[build_approval_service] = lambda: approval_service
    with TestClient(app) as client:
        yield client, repository, query_service, approval_service
    app.dependency_overrides.clear()


def _headers(token: str = "owner-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Request-ID": "task-request"}


def test_create_and_get_task_expose_only_collaboration_state(setup_app) -> None:
    client, repository, _, _ = setup_app

    created = client.post(
        "/v1/tasks",
        headers=_headers(),
        json={"goal": "核对上线制度", "task_mode": "research"},
    )
    fetched = client.get("/v1/tasks/task-1", headers=_headers())

    assert created.status_code == 202
    assert fetched.status_code == 200
    assert repository.created[0]["task_mode"] is TaskMode.RESEARCH
    assert "auth_context_ref" not in fetched.json()
    assert "lease_generation" not in fetched.json()
    assert "repeated_action_fingerprints" not in fetched.json()
    assert fetched.json()["pending_question"] is None


def test_continue_and_cancel_use_state_version_and_server_answer_reference(setup_app) -> None:
    client, repository, _, _ = setup_app
    repository.current = _state(
        status=TaskStatus.WAITING_FOR_USER,
        current_node=TaskNode.CHECK_CLARIFICATION,
        pending_question="需要哪个部门？",
    )

    continued = client.post(
        "/v1/tasks/task-1/continue",
        headers=_headers(),
        json={"answer": "人力资源部", "expected_state_version": 1},
    )
    stale = client.post(
        "/v1/tasks/task-1/cancel",
        headers=_headers(),
        json={"expected_state_version": 1},
    )
    cancelled = client.post(
        "/v1/tasks/task-1/cancel",
        headers=_headers(),
        json={"expected_state_version": 2},
    )

    assert continued.status_code == 200
    assert repository.resumed[0]["answer_ref"].startswith("sha256:")
    assert repository.resumed[0]["answer_ref"] != "人力资源部"
    assert stale.status_code == 409
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_events_artifacts_and_approval_are_bound_and_redacted(setup_app) -> None:
    client, _, _, approvals = setup_app

    events = client.get("/v1/tasks/task-1/events", headers=_headers())
    artifacts = client.get("/v1/tasks/task-1/artifacts", headers=_headers())
    approval = client.post(
        "/v1/tasks/task-1/plans/2/approval",
        headers=_headers("approver-token"),
        json={"approval_id": "approval-1"},
    )

    assert events.status_code == 200
    assert artifacts.status_code == 200
    assert artifacts.json()[0] == {
        "artifact_id": "artifact-1",
        "artifact_type": "report",
        "status": "ready",
        "created_at": "2026-08-22T00:00:00Z",
    }
    assert approval.status_code == 200
    assert approval.json()["status"] == "approved"
    assert "plan_digest" not in approval.json()
    assert approvals.calls[0]["expected_task_id"] == "task-1"
    assert approvals.calls[0]["expected_plan_revision"] == 2


def test_permissions_not_found_conflicts_and_extra_body_are_stable(setup_app) -> None:
    client, _, _, _ = setup_app

    no_create = client.post(
        "/v1/tasks",
        headers=_headers("outsider-token"),
        json={"goal": "任务", "task_mode": "research"},
    )
    cross_tenant = client.get("/v1/tasks/task-1", headers=_headers("outsider-token"))
    extra = client.post(
        "/v1/tasks",
        headers=_headers(),
        json={"goal": "任务", "task_mode": "research", "tenant_id": "tenant-b"},
    )
    binding_mismatch = client.post(
        "/v1/tasks/task-1/plans/3/approval",
        headers=_headers("approver-token"),
        json={"approval_id": "approval-1"},
    )

    assert no_create.status_code == 403
    assert cross_tenant.status_code == 404
    assert extra.status_code == 422
    assert binding_mismatch.status_code == 409
