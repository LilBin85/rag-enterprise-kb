"""旧路由仅作为新版服务的薄兼容层；不回归旧 planner/store 实现。"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_kb.application.query_service import KnowledgeQueryResult
from rag_kb.domain.identity import AuthContext, Principal
from rag_kb.domain.tasks import TaskMode, TaskState, TaskStatus
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.models import PolicyComparisonResponse, Refusal, Role
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import (
    build_compatibility_policy_comparison_service,
    build_identity_provider,
    build_knowledge_query_service,
    build_task_repository,
)


class _QueryService:
    def __init__(self):
        self.calls = []

    def query(self, question, auth):
        self.calls.append((question, auth))
        return KnowledgeQueryResult(
            verdict="refusal", refusal=Refusal(kind="test", message="拒答"), citations=()
        )


class _ComparisonService:
    def __init__(self):
        self.calls = []

    def compare(self, **kwargs):
        self.calls.append(kwargs)
        return PolicyComparisonResponse(
            verdict="refusal", refusal=Refusal(kind="test", message="拒答"), role=kwargs["role"], run_id="run-1"
        )


class _TaskRepository:
    def __init__(self):
        self.calls = []

    def create_task(self, **kwargs):
        self.calls.append(kwargs)
        return TaskState(
            task_id="task-1",
            auth_context_ref="auth-ref",
            goal=kwargs["goal"],
            task_mode=TaskMode.CONTROLLED_ACTION,
            status=TaskStatus.PLANNING,
        )


def _client():
    app = create_app(include_legacy_routes=True)
    provider = DemoTokenIdentityProvider(
        {
            "engineer-token": Principal(
                tenant_id="tenant-a",
                user_id="alice",
                roles=frozenset({"engineer"}),
                attributes={"permissions": ["task:create"]},
            )
        }
    )
    query = _QueryService()
    comparison = _ComparisonService()
    repository = _TaskRepository()
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_knowledge_query_service] = lambda: query
    app.dependency_overrides[build_compatibility_policy_comparison_service] = lambda: comparison
    app.dependency_overrides[build_task_repository] = lambda: repository
    return TestClient(app), query, comparison, repository


def test_legacy_questions_delegates_identity_and_emits_deprecation_headers() -> None:
    client, query, _comparison, _repository = _client()
    response = client.post(
        "/questions",
        json={"question": "问题", "role": "engineer"},
        headers={"Authorization": "Bearer engineer-token"},
    )
    assert response.status_code == 200
    assert query.calls[0][0] == "问题"
    assert query.calls[0][1].tenant_id == "tenant-a"
    assert response.headers["Deprecation"] == "true"
    assert "Sunset" in response.headers
    assert response.headers["Link"] == "</v1/knowledge/query>; rel=\"successor-version\""

    mismatch = client.post(
        "/questions",
        json={"question": "问题", "role": "admin"},
        headers={"Authorization": "Bearer engineer-token"},
    )
    assert mismatch.status_code == 403
    assert len(query.calls) == 1


def test_legacy_policy_comparison_delegates_only_to_compatibility_service() -> None:
    client, _query, comparison, _repository = _client()
    response = client.post(
        "/policy-comparisons",
        json={"document_id": "doc-1", "from_version": 1, "to_version": 2, "role": "engineer"},
        headers={"Authorization": "Bearer engineer-token"},
    )
    assert response.status_code == 200
    assert comparison.calls == [
        {
            "auth": comparison.calls[0]["auth"],
            "document_id": "doc-1",
            "from_version": 1,
            "to_version": 2,
            "role": Role.ENGINEER,
        }
    ]
    assert response.headers["Link"] == "</v1/tasks>; rel=\"successor-version\""


def test_legacy_policy_comparison_rejects_boolean_versions() -> None:
    client, _query, comparison, _repository = _client()
    response = client.post(
        "/policy-comparisons",
        json={"document_id": "doc-1", "from_version": True, "to_version": 2, "role": "engineer"},
        headers={"Authorization": "Bearer engineer-token"},
    )
    assert response.status_code == 422
    assert comparison.calls == []


def test_legacy_rollout_creation_delegates_to_task_repository() -> None:
    client, _query, _comparison, repository = _client()
    response = client.post(
        "/policy-rollout-tasks",
        json={"goal": "整理制度", "role": "engineer"},
        headers={"Authorization": "Bearer engineer-token"},
    )
    assert response.status_code == 202
    assert repository.calls and repository.calls[0]["goal"] == "整理制度"
    assert response.headers["Link"] == "</v1/tasks>; rel=\"successor-version\""


def test_legacy_routes_are_disabled_by_default() -> None:
    client = TestClient(create_app())
    response = client.post("/questions", json={"question": "问题", "role": "engineer"})
    assert response.status_code == 404
