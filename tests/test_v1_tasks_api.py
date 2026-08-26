"""M1 Research Agent HTTP 契约，全部依赖由内存假件注入。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_kb.agent.result import TaskResult
from rag_kb.application.task_query_service import TaskEventResult, TaskQueryError
from rag_kb.domain.identity import Principal
from rag_kb.domain.tasks import TaskMode, TaskState, TaskStatus
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database.task_repository import TaskRepositoryError
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import (
    build_identity_provider,
    build_task_query_service,
    build_task_repository,
    build_task_result_service,
    build_task_resume_worker,
)


def _state(
    *,
    status: TaskStatus = TaskStatus.PLANNING,
    state_version: int = 1,
    interrupt_kind: str | None = None,
    interrupt_summary: str | None = None,
    result_ref: str | None = None,
    result_hash: str | None = None,
    cancelled: bool = False,
) -> TaskState:
    return TaskState(
        task_id="task-1",
        auth_context_ref="sha256:internal-auth-context",
        goal="核对上线制度",
        task_mode=TaskMode.RESEARCH,
        status=status,
        state_version=state_version,
        interrupt_kind=interrupt_kind,
        interrupt_summary=interrupt_summary,
        result_kind="research" if result_ref else None,
        result_ref=result_ref,
        result_hash=result_hash,
        content_expires_at=(
            datetime(2026, 9, 1, tzinfo=UTC)
            if status
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.REFUSED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.UNKNOWN,
            }
            else None
        ),
        cancelled=cancelled,
    )


class FakeTaskRepository:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.cancelled: list[dict[str, object]] = []
        self.current = _state()

    def create_task(self, **kwargs: object) -> TaskState:
        self.created.append(kwargs)
        return self.current

    def get_task(self, *, auth, task_id: str) -> TaskState:
        if task_id != "task-1" or auth.tenant_id != "tenant-a" or auth.user_id != "owner-a":
            raise TaskRepositoryError("task_not_found")
        return self.current

    def cancel(self, **kwargs: object) -> TaskState:
        if kwargs["expected_state_version"] != self.current.state_version:
            raise TaskRepositoryError("task_state_conflict")
        self.cancelled.append(kwargs)
        self.current = _state(
            status=TaskStatus.CANCELLED,
            state_version=self.current.state_version + 1,
            cancelled=True,
        )
        return self.current


class FakeResumeWorker:
    def __init__(self, repository: FakeTaskRepository) -> None:
        self._repository = repository
        self.calls: list[dict[str, object]] = []

    def resume_for_user(self, **kwargs: object) -> object:
        if kwargs["expected_state_version"] != self._repository.current.state_version:
            raise TaskRepositoryError("task_state_conflict")
        self.calls.append(kwargs)
        self._repository.current = _state(
            status=TaskStatus.REFUSED,
            state_version=self._repository.current.state_version + 1,
            result_ref="task-results/task-1/result.json",
            result_hash="sha256:" + "a" * 64,
        )
        return SimpleNamespace(status="finished")


class FakeTaskQueryService:
    def events(self, *, auth, task_id: str):
        if not auth.has_permission("task:read"):
            raise TaskQueryError("task_read_denied")
        if task_id != "task-1" or auth.tenant_id != "tenant-a" or auth.user_id != "owner-a":
            raise TaskQueryError("task_not_found")
        return (
            TaskEventResult(
                sequence=0,
                event_type="task_created",
                stage="planning",
                status="planning",
                counts={},
                token_usage=None,
                error_code=None,
                created_at=datetime(2026, 8, 25, tzinfo=UTC),
            ),
        )


class FakeResultService:
    def __init__(self) -> None:
        self.result = TaskResult(
            task_id="task-1",
            status="refused",
            refusal_code="insufficient_evidence",
            stop_reason="no_evidence",
            created_at=datetime(2026, 8, 25, tzinfo=UTC),
        )

    def read(self, *, auth, state: TaskState) -> TaskResult:
        if auth.user_id != "owner-a" or state.task_id != "task-1":
            raise ValueError("task_result_integrity_failed")
        return self.result


@pytest.fixture()
def setup_app() -> Iterator[tuple[TestClient, FakeTaskRepository, FakeResumeWorker]]:
    app = create_app()
    repository = FakeTaskRepository()
    worker = FakeResumeWorker(repository)
    provider = DemoTokenIdentityProvider(
        {
            "owner-token": Principal(
                tenant_id="tenant-a",
                user_id="owner-a",
                attributes={"permissions": ["task:create", "task:read"]},
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
    app.dependency_overrides[build_task_query_service] = lambda: FakeTaskQueryService()
    app.dependency_overrides[build_task_resume_worker] = lambda: worker
    app.dependency_overrides[build_task_result_service] = lambda: FakeResultService()
    with TestClient(app) as client:
        yield client, repository, worker
    app.dependency_overrides.clear()


def _headers(token: str = "owner-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Request-ID": "task-request"}


def test_create_and_get_task_expose_only_business_lifecycle(setup_app) -> None:
    client, repository, _ = setup_app
    created = client.post(
        "/v1/tasks",
        headers=_headers(),
        json={"goal": "核对上线制度", "task_mode": "research"},
    )
    fetched = client.get("/v1/tasks/task-1", headers=_headers())

    assert created.status_code == 202
    assert fetched.status_code == 200
    assert repository.created[0]["task_mode"] is TaskMode.RESEARCH
    assert {"auth_context_ref", "lease_generation", "current_node", "pending_question"}.isdisjoint(
        fetched.json()
    )
    assert fetched.json()["result_available"] is False


def test_create_rejects_non_research_task_modes(setup_app) -> None:
    client, repository, _ = setup_app
    response = client.post(
        "/v1/tasks",
        headers=_headers(),
        json={"goal": "查询制度", "task_mode": "knowledge_query"},
    )
    assert response.status_code == 422
    assert repository.created == []


def test_result_is_read_from_server_side_reference(setup_app) -> None:
    client, repository, _ = setup_app
    repository.current = _state(
        status=TaskStatus.REFUSED,
        result_ref="task-results/task-1/result.json",
        result_hash="sha256:" + "a" * 64,
    )

    result = client.get("/v1/tasks/task-1/result", headers=_headers())
    unavailable = client.get("/v1/tasks/task-1/result", headers=_headers("outsider-token"))

    assert result.status_code == 200
    assert result.json()["refusal_code"] == "insufficient_evidence"
    assert unavailable.status_code == 404


def test_continue_passes_answer_only_to_controlled_worker_resume(setup_app) -> None:
    client, repository, worker = setup_app
    repository.current = _state(
        status=TaskStatus.WAITING_FOR_USER,
        interrupt_kind="user_clarification",
        interrupt_summary="需要哪个部门？",
    )

    response = client.post(
        "/v1/tasks/task-1/continue",
        headers=_headers(),
        json={"answer": "人力资源部", "expected_state_version": 1},
    )

    assert response.status_code == 200
    assert len(worker.calls) == 1
    assert worker.calls[0]["task_id"] == "task-1"
    assert worker.calls[0]["expected_state_version"] == 1
    assert worker.calls[0]["answer"] == "人力资源部"
    assert "answer" not in response.json()
    assert response.json()["result_available"] is True


def test_cancel_events_and_cross_tenant_errors_are_stable(setup_app) -> None:
    client, repository, _ = setup_app
    events = client.get("/v1/tasks/task-1/events", headers=_headers())
    cross_tenant = client.get("/v1/tasks/task-1", headers=_headers("outsider-token"))
    cancelled = client.post(
        "/v1/tasks/task-1/cancel",
        headers=_headers(),
        json={"expected_state_version": 1},
    )
    stale = client.post(
        "/v1/tasks/task-1/cancel",
        headers=_headers(),
        json={"expected_state_version": 1},
    )

    assert events.status_code == 200
    assert events.json()[0]["event_type"] == "task_created"
    assert cross_tenant.status_code == 404
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert stale.status_code == 409
    assert len(repository.cancelled) == 1
