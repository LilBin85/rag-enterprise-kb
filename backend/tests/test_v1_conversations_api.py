from datetime import UTC, datetime

from fastapi.testclient import TestClient

from rag_kb.application.conversation_service import ConversationServiceError
from rag_kb.domain.identity import Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import build_conversation_service, build_identity_provider


class FakeConversations:
    def create(self, *, auth, title=None):
        return _model(conversation_id="c-1", title=title, state_version=1, status="active", created_at=_now(), updated_at=_now())

    def list(self, *, auth, limit, cursor):
        return _model(items=(), next_cursor=None)

    def read(self, *, auth, conversation_id):
        if conversation_id == "missing":
            raise ConversationServiceError("conversation_not_found")
        return self.create(auth=auth, title=None)

    def messages(self, **_kwargs):
        return _model(items=())

    def delete(self, *, auth, conversation_id):
        return _model(conversation_id=conversation_id, status="deleting")

    def submit(self, **kwargs):
        if kwargs["idempotency_key"] == "conflict":
            raise ConversationServiceError("idempotency_conflict")
        answer = kwargs.get("mode") == "answer"
        return _model(
            conversation_id="c-1", conversation_state_version=2, turn_id="t-1",
            user_message={"message_id": "m-1", "sequence": 1}, requested_mode=kwargs.get("mode", "auto"),
            selected_mode="answer" if answer else "research", route_reason_code="test",
            turn_status="refused" if answer else "planning",
            task={"task_id": "task-1", "status": "refused" if answer else "planning", "state_version": 1, "result_available": False},
            assistant_message={"message_id": "m-2"} if answer else None,
            idempotent_replay=False, status_code=200 if answer else 202,
        )


class _model:
    def __init__(self, **values): self.values = values
    def model_dump(self, **_kwargs): return {key: value for key, value in self.values.items() if key != "status_code"}
    @property
    def status_code(self): return self.values.get("status_code", 200)


def _now(): return datetime(2026, 9, 6, tzinfo=UTC)


def test_conversation_routes_validate_and_map_service_results() -> None:
    app = create_app()
    app.dependency_overrides[build_identity_provider] = lambda: DemoTokenIdentityProvider({"token": Principal(tenant_id="t", user_id="u")})
    app.dependency_overrides[build_conversation_service] = FakeConversations
    client = TestClient(app)
    headers = {"Authorization": "Bearer token", "Idempotency-Key": "key"}

    assert client.post("/v1/conversations", headers=headers, json={}).status_code == 201
    assert client.get("/v1/conversations", headers=headers).json()["next_cursor"] is None
    assert client.get("/v1/conversations/missing", headers=headers).status_code == 404
    assert client.post("/v1/conversations/c-1/messages", headers=headers, json={"kind": "new", "content": "问题"}).status_code == 202
    answer = client.post("/v1/conversations/c-1/messages", headers=headers, json={"kind": "new", "content": "问题", "mode": "answer"})
    assert answer.status_code == 200
    assert answer.json()["task"]["result_available"] is False
    assert client.post("/v1/conversations/c-1/messages", json={"kind": "new", "content": "问题"}, headers={"Authorization": "Bearer token"}).status_code == 422
    conflict = client.post("/v1/conversations/c-1/messages", headers={**headers, "Idempotency-Key": "conflict"}, json={"kind": "new", "content": "问题"})
    assert conflict.json()["detail"] == {"code": "idempotency_conflict"}
    assert client.delete("/v1/conversations/c-1", headers=headers).status_code == 202
    for path in ("/v1/knowledge/query", "/v1/tasks", "/v1/tasks/task-1/continue"):
        assert client.post(path, headers=headers, json={}).status_code == 404
        assert path not in app.openapi()["paths"]
