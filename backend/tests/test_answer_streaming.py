from __future__ import annotations

from threading import Event
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_kb.application.conversation_service import ConversationService, PreparedAnswer
from rag_kb.application.query_service import KnowledgeQueryExecution
from rag_kb.domain.identity import Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.web.answer_streaming import AnswerExecutionManager, AnswerStream
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import build_conversation_service, build_identity_provider


def _prepared() -> PreparedAnswer:
    return PreparedAnswer(
        turn=SimpleNamespace(
            conversation_id="conversation-1",
            turn_id="turn-1",
            task_id="task-1",
            resolved_input="问题",
        ),
        user_message=SimpleNamespace(message_id="message-1", sequence=1),
        route_reason_code="explicit_answer",
        original_input="问题",
    )


class _Result:
    def model_dump(self, **_kwargs) -> dict[str, object]:
        return {
            "conversation_id": "conversation-1",
            "turn_id": "turn-1",
            "turn_status": "succeeded",
            "assistant_message": {"content": "正式答案"},
        }


def test_answer_stream_coalesces_updates_and_keeps_only_terminal_event() -> None:
    stream = AnswerStream()
    stream.publish_status("retrieving")
    stream.publish_status("generating")
    stream.publish_preview("草稿")
    assert stream.finish_done({"turn_id": "turn-1"}) is True
    assert stream.finish_error("late_error") is False

    payload = b"".join(stream.iter_sse({"turn_id": "turn-1"})).decode("utf-8")

    assert "event: turn.accepted" in payload
    assert "event: status\ndata: {\"stage\":\"generating\"}" in payload
    assert "event: answer.preview\ndata: {\"answer\":\"草稿\"}" in payload
    assert "event: done\ndata: {\"turn_id\":\"turn-1\"}" in payload
    assert "late_error" not in payload


def test_execution_manager_rejects_over_capacity_without_queueing() -> None:
    started = Event()
    release = Event()

    class BlockingService:
        def execute_prepared_answer(self, **kwargs):
            kwargs["status"]("generating")
            started.set()
            assert release.wait(3)
            return _Result()

        def fail_prepared_answer(self, **_kwargs):
            return True

    manager = AnswerExecutionManager(max_executions=1)
    try:
        first = manager.submit(service=BlockingService(), auth=SimpleNamespace(), prepared=_prepared())
        assert first is not None and started.wait(3)
        assert manager.submit(service=BlockingService(), auth=SimpleNamespace(), prepared=_prepared()) is None
        release.set()
        payload = b"".join(first.iter_sse({"turn_id": "turn-1"})).decode("utf-8")
        assert "event: done" in payload
    finally:
        release.set()
        manager.close()


def test_timeout_fences_task_and_late_done_cannot_replace_error() -> None:
    failed = Event()

    class SlowService:
        def execute_prepared_answer(self, **kwargs):
            kwargs["on_generation_started"](
                KnowledgeQueryExecution(
                    task_id="task-1",
                    lease_owner="owner",
                    lease_generation=1,
                )
            )
            assert failed.wait(3)
            return _Result()

        def fail_prepared_answer(self, **_kwargs):
            failed.set()
            return True

    manager = AnswerExecutionManager(max_executions=1, generation_timeout_seconds=1)
    try:
        stream = manager.submit(service=SlowService(), auth=SimpleNamespace(), prepared=_prepared())
        assert stream is not None
        payload = b"".join(stream.iter_sse({"turn_id": "turn-1"})).decode("utf-8")
        assert "event: error" in payload
        assert "answer_generation_timeout" in payload
        assert "event: done" not in payload
    finally:
        manager.close()


def test_fail_prepared_answer_preserves_fenced_repository_result() -> None:
    captured: dict[str, object] = {}

    def fail_fast_answer(**kwargs) -> bool:
        captured.update(kwargs)
        return True

    service = ConversationService(SimpleNamespace(fail_fast_answer=fail_fast_answer), None)

    assert service.fail_prepared_answer(
        auth=SimpleNamespace(),
        prepared=_prepared(),
        error_code="answer_generation_timeout",
        execution=KnowledgeQueryExecution(
            task_id="task-1", lease_owner="owner", lease_generation=1
        ),
    ) is True
    assert captured["task_lease_owner"] == "owner"
    assert captured["task_lease_generation"] == 1


def test_conversation_answer_uses_sse_only_when_requested() -> None:
    class StreamingService:
        def prepare_submission(self, **_kwargs):
            return _prepared()

        def execute_prepared_answer(self, **kwargs):
            kwargs["status"]("retrieving")
            kwargs["preview"]("草稿")
            return _Result()

        def fail_prepared_answer(self, **_kwargs):
            return True

    app = create_app()
    app.dependency_overrides[build_identity_provider] = lambda: DemoTokenIdentityProvider(
        {"token": Principal(tenant_id="tenant-1", user_id="user-1")}
    )
    app.dependency_overrides[build_conversation_service] = StreamingService
    headers = {
        "Authorization": "Bearer token",
        "Idempotency-Key": "stream-key",
        "Accept": "text/event-stream",
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/conversations/conversation-1/messages",
            headers=headers,
            json={"kind": "new", "content": "问题", "mode": "answer"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: turn.accepted" in response.text
    assert "event: answer.preview" in response.text
    assert "event: done" in response.text
