"""M1 Research Agent HTTP 契约，全部依赖由内存假件注入。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_kb.agent.result import TaskResult
from rag_kb.application.task_query_service import TaskEventResult, TaskQueryError
from rag_kb.application.task_result_service import TaskResultError
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
)
from rag_kb.web.routers import tasks as tasks_router


def _state(
    *,
    task_mode: TaskMode = TaskMode.RESEARCH,
    status: TaskStatus = TaskStatus.PLANNING,
    state_version: int = 1,
    interrupt_kind: str | None = None,
    interrupt_summary: str | None = None,
    interrupt_ref: str | None = None,
    result_ref: str | None = None,
    result_hash: str | None = None,
    content_expires_at: datetime | None = None,
    result_purged_at: datetime | None = None,
    cancelled: bool = False,
) -> TaskState:
    return TaskState(
        task_id="task-1",
        auth_context_ref="sha256:internal-auth-context",
        goal="核对上线制度",
        task_mode=task_mode,
        status=status,
        state_version=state_version,
        interrupt_kind=interrupt_kind,
        interrupt_summary=interrupt_summary,
        interrupt_ref=interrupt_ref,
        result_kind=(
            "controlled_action"
            if result_hash and task_mode is TaskMode.CONTROLLED_ACTION
            else "research"
            if result_hash
            else None
        ),
        result_ref=result_ref,
        result_hash=result_hash,
        content_expires_at=(
            content_expires_at
            if content_expires_at is not None
            else datetime.now(UTC) + timedelta(days=1)
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
        result_purged_at=result_purged_at,
        cancelled=cancelled,
    )


class FakeTaskRepository:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.cancelled: list[dict[str, object]] = []
        self.enqueued: list[dict[str, object]] = []
        self._resume_commands: dict[
            tuple[str, int], tuple[str, SimpleNamespace]
        ] = {}
        self.current = _state()
        self.create_error: str | None = None

    def create_task(self, **kwargs: object) -> TaskState:
        if self.create_error is not None:
            raise TaskRepositoryError(self.create_error)
        self.created.append(kwargs)
        return self.current

    def get_task(self, *, auth, task_id: str) -> TaskState:
        if task_id != "task-1" or auth.tenant_id != "tenant-a" or auth.user_id != "owner-a":
            raise TaskRepositoryError("task_not_found")
        return self.current

    def current_time(self) -> datetime:
        return datetime.now(UTC)

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

    def enqueue_user_resume_command(self, **kwargs: object) -> SimpleNamespace:
        auth = kwargs["auth"]
        task_id = kwargs["task_id"]
        expected_state_version = kwargs["expected_state_version"]
        answer = kwargs["answer"]
        if (
            task_id != "task-1"
            or auth.tenant_id != "tenant-a"
            or auth.user_id != "owner-a"
        ):
            raise TaskRepositoryError("task_not_found")
        if self.current.interrupt_kind != "user_clarification":
            raise TaskRepositoryError("task_resume_kind_invalid")
        if expected_state_version != self.current.state_version:
            raise TaskRepositoryError("task_state_conflict")
        target = (task_id, expected_state_version)
        existing = self._resume_commands.get(target)
        if existing is not None:
            if existing[0] == answer:
                return existing[1]
            raise TaskRepositoryError("task_resume_conflict")
        command = SimpleNamespace(command_id=f"resume-command-{len(self.enqueued) + 1}")
        self.enqueued.append(kwargs)
        self._resume_commands[target] = (answer, command)
        return command


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
        if state.status not in {TaskStatus.SUCCEEDED, TaskStatus.REFUSED} or (
            state.result_kind != "research" or state.result_hash is None
        ):
            raise TaskResultError("task_result_unavailable")
        if state.result_purged_at is not None or (
            state.content_expires_at is not None
            and state.content_expires_at <= datetime.now(UTC)
        ):
            raise TaskResultError("task_result_expired")
        if state.result_ref is None:
            raise TaskResultError("task_result_unavailable")
        if state.result_hash == "duplicate":
            raise TaskResultError("task_result_binding_invalid")
        if state.result_hash == "authorization_stale":
            raise TaskResultError("task_result_evidence_unavailable")
        if state.result_hash == "integrity":
            raise TaskResultError("task_result_integrity_failed")
        return self.result


@pytest.fixture()
def setup_app() -> Iterator[tuple[TestClient, FakeTaskRepository]]:
    app = create_app()
    repository = FakeTaskRepository()
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
            "denied-token": Principal(
                tenant_id="tenant-a",
                user_id="denied-a",
                attributes={"permissions": []},
            ),
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_task_repository] = lambda: repository
    app.dependency_overrides[build_task_query_service] = lambda: FakeTaskQueryService()
    app.dependency_overrides[build_task_result_service] = lambda: FakeResultService()
    with TestClient(app) as client:
        yield client, repository
    app.dependency_overrides.clear()


def _headers(token: str = "owner-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Request-ID": "task-request"}







def test_result_is_read_from_server_side_reference(setup_app) -> None:
    client, repository = setup_app
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


def test_result_contract_distinguishes_expired_unavailable_and_integrity(setup_app) -> None:
    client, repository = setup_app
    repository.current = _state(
        status=TaskStatus.REFUSED,
        result_ref="task-results/task-1/result.json",
        result_hash="sha256:" + "a" * 64,
        content_expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    expired = client.get("/v1/tasks/task-1/result", headers=_headers())
    state = client.get("/v1/tasks/task-1", headers=_headers())
    assert expired.status_code == 410
    assert state.json()["result_available"] is False

    repository.current = _state()
    assert client.get("/v1/tasks/task-1/result", headers=_headers()).status_code == 409

    repository.current = _state(
        status=TaskStatus.REFUSED,
        result_hash="sha256:" + "a" * 64,
        result_purged_at=datetime.now(UTC),
    )
    assert client.get("/v1/tasks/task-1/result", headers=_headers()).status_code == 410

    repository.current = _state(
        status=TaskStatus.REFUSED,
        result_ref="task-results/task-1/result.json",
        result_hash="integrity",
    )
    assert client.get("/v1/tasks/task-1/result", headers=_headers()).status_code == 500

    repository.current = _state(
        status=TaskStatus.REFUSED,
        result_ref="task-results/task-1/result.json",
        result_hash="duplicate",
    )
    assert client.get("/v1/tasks/task-1/result", headers=_headers()).status_code == 500

    repository.current = _state(
        status=TaskStatus.REFUSED,
        result_ref="task-results/task-1/result.json",
        result_hash="authorization_stale",
    )
    assert client.get("/v1/tasks/task-1/result", headers=_headers()).status_code == 409






def test_cancel_events_and_cross_tenant_errors_are_stable(setup_app) -> None:
    client, repository = setup_app
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




def test_legacy_task_write_endpoints_are_removed(setup_app) -> None:
    client, repository = setup_app

    assert client.post(
        "/v1/tasks",
        headers=_headers(),
        json={"goal": "问题", "task_mode": "research"},
    ).status_code == 404
    assert client.post(
        "/v1/tasks/task-1/continue",
        headers=_headers(),
        json={"answer": "答案", "expected_state_version": 1},
    ).status_code == 404
    paths = client.app.openapi()["paths"]
    assert "/v1/tasks" not in paths
    assert "/v1/tasks/{task_id}/continue" not in paths
    assert repository.created == []
    assert repository.enqueued == []
