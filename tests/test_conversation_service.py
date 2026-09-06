from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from rag_kb.application.conversation_service import (
    ConversationRouteDecision,
    ConversationService,
    ConversationServiceError,
    _estimate_tokens,
)
from rag_kb.domain.conversations import ConversationMode
from rag_kb.application.query_service import KnowledgeQueryResult
from rag_kb.domain.identity import AuthContext
from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.conversation_repository import (
    DatabaseConversationRepository,
)
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ConversationMessageRow,
    ConversationRow,
    ConversationTurnRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TenantRow,
)
from rag_kb.infrastructure.database.session import build_database
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.models import Refusal


@pytest.fixture()
def service(tmp_path):
    database = build_database(f"sqlite:///{tmp_path / 'conversation.db'}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="租户A", active=True))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="owner-a",
                roles=[],
                attributes={},
                active=True,
            )
        )
        session.add(
            PrincipalRow(
                tenant_id="tenant-a",
                user_id="other-a",
                roles=[],
                attributes={},
                active=True,
            )
        )
    tasks = DatabaseTaskRepository(database)
    return ConversationService(DatabaseConversationRepository(database), tasks), database


def _auth(user_id: str = "owner-a") -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id=user_id,
        roles=frozenset(),
        groups=frozenset(),
        attributes={},
        request_id="request-1",
    )


def test_research_submit_is_durable_idempotent_and_single_active(service) -> None:
    conversations, _database = service
    created = conversations.create(auth=_auth())

    first = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="key-1",
        kind="new",
        content="比较两个制度版本",
        mode="research",
    )
    replay = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="key-1",
        kind="new",
        content="比较两个制度版本",
        mode="research",
    )

    assert first.status_code == 202
    assert first.task is not None
    assert first.task.task_id == replay.task.task_id
    assert replay.idempotent_replay is True
    with pytest.raises(ConversationServiceError, match="conversation_has_active_turn"):
        conversations.delete(auth=_auth(), conversation_id=created.conversation_id)
    with pytest.raises(ConversationServiceError, match="conversation_has_active_turn"):
        conversations.submit(
            auth=_auth(),
            conversation_id=created.conversation_id,
            idempotency_key="key-2",
            kind="new",
            content="另一个问题",
            mode="research",
        )
    with pytest.raises(ConversationServiceError, match="conversation_not_found"):
        conversations.read(auth=_auth("other-a"), conversation_id=created.conversation_id)


class _RefusingKnowledge:
    def query(self, _question, _auth, **_binding):
        return KnowledgeQueryResult(
            verdict="refusal",
            refusal=Refusal(kind="insufficient_evidence", message="证据不足"),
        )


class _CountingRouter:
    def __init__(self) -> None:
        self.calls = 0

    def route(self, *, content, context, requested_mode):
        self.calls += 1
        return ConversationRouteDecision(
            selected_mode=ConversationMode.RESEARCH,
            resolved_input=content,
            reason_code="test_research",
        )


class _BrokenKnowledge:
    def query(self, *_args, **_kwargs):
        raise RuntimeError("retrieval_down")


class _ControlledRouter:
    def route(self, *, content, context, requested_mode):
        return ConversationRouteDecision(
            selected_mode=ConversationMode.CONTROLLED_ACTION,
            resolved_input=content,
            reason_code="write_intent_requires_explicit",
        )


def test_explicit_answer_settles_message_and_releases_conversation(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database),
        DatabaseTaskRepository(database),
        knowledge_queries=_RefusingKnowledge(),
    )
    created = conversations.create(auth=_auth(), title="制度问答")
    result = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="answer-1",
        kind="new",
        content="现行制度是什么？",
        mode="answer",
    )

    assert result.status_code == 200
    assert result.turn_status == "refused"
    assert result.task is not None and result.task.result_available is False
    assert result.assistant_message is not None
    assert result.assistant_message.content == "证据不足"
    replay = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="answer-1",
        kind="new",
        content="现行制度是什么？",
        mode="answer",
    )
    assert replay.idempotent_replay is True
    assert replay.task is not None and replay.task.result_available is False
    with database.sessions() as session:
        task = session.get(AgentTaskRow, result.task.task_id)
        turn = session.get(ConversationTurnRow, result.turn_id)
    assert task is not None and turn is not None
    assert task.status == turn.status == result.task.status == result.turn_status
    assert task.state_version == turn.task_state_version == result.task.state_version
    state = DatabaseTaskRepository(database).get_task(auth=_auth(), task_id=result.task.task_id)
    assert state.result_available(now=datetime.now(UTC)) is False
    history = conversations.messages(
        auth=_auth(), conversation_id=created.conversation_id
    )
    assert [item.kind for item in history.items] == ["input", "refusal"]
    assert conversations.delete(
        auth=_auth(), conversation_id=created.conversation_id
    ).status == "deleting"


def test_clarification_reply_and_resume_command_commit_together(service) -> None:
    conversations, database = service
    created = conversations.create(auth=_auth())
    submitted = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="research-clarify",
        kind="new",
        content="比较制度版本",
        mode="research",
    )
    assert submitted.task is not None
    tasks = DatabaseTaskRepository(database)
    lease = tasks.claim_next(worker_id="worker-a", lease_seconds=30)
    assert lease is not None
    paused = tasks.pause_for_user(
        lease,
        question="需要比较哪些版本？",
        checkpoint_ns="",
        checkpoint_id="checkpoint-a",
        interrupt_id="interrupt-a",
        run_id="run-a",
    )

    resumed = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="clarification-1",
        kind="clarification",
        content="比较 2025 年版和 2024 年版",
        turn_id=submitted.turn_id,
        task_id=submitted.task.task_id,
        expected_task_state_version=paused.state_version,
    )
    replay = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="clarification-1",
        kind="clarification",
        content="比较 2025 年版和 2024 年版",
        turn_id=submitted.turn_id,
        task_id=submitted.task.task_id,
        expected_task_state_version=paused.state_version,
    )

    assert resumed.status_code == 202
    assert resumed.turn_status == "running"
    assert replay.status_code == 202
    assert replay.idempotent_replay is True
    assert replay.user_message == resumed.user_message
    with database.sessions() as session:
        turn = session.get(ConversationTurnRow, submitted.turn_id)
        commands = tuple(session.scalars(select(TaskResumeCommandRow)))
        replies = tuple(
            session.scalars(
                select(ConversationMessageRow).where(
                    ConversationMessageRow.kind == "clarification_reply"
                )
            )
        )
    assert turn is not None and turn.status == "running"
    assert len(commands) == 1
    assert len(replies) == 1


def test_replay_precedes_route_and_conflicting_key_never_reroutes(service) -> None:
    _conversations, database = service
    router = _CountingRouter()
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database), router=router
    )
    created = conversations.create(auth=_auth())
    first = conversations.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="route-once",
        kind="new", content="请研究两版制度", mode="auto",
    )
    replay = conversations.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="route-once",
        kind="new", content="请研究两版制度", mode="auto",
    )

    assert first.task is not None and replay.idempotent_replay is True
    assert router.calls == 1
    with pytest.raises(ConversationServiceError, match="idempotency_conflict"):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id, idempotency_key="route-once",
            kind="new", content="不同请求", mode="auto",
        )
    assert router.calls == 1


def test_auto_never_creates_controlled_action_turn(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        router=_ControlledRouter(), controlled_action_enabled=True,
    )
    created = conversations.create(auth=_auth())
    with pytest.raises(ConversationServiceError, match="controlled_action_explicit_mode_required"):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id, idempotency_key="auto-write",
            kind="new", content="创建工作项", mode="auto",
        )
    with database.sessions() as session:
        assert session.scalar(select(ConversationTurnRow)) is None


def test_task_cancel_synchronizes_bound_turn(service) -> None:
    conversations, database = service
    created = conversations.create(auth=_auth())
    submitted = conversations.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="cancel-bound",
        kind="new", content="研究制度", mode="research",
    )
    assert submitted.task is not None
    state = DatabaseTaskRepository(database).cancel(
        auth=_auth(), task_id=submitted.task.task_id,
        expected_state_version=submitted.task.state_version,
    )
    assert state.status.value == "cancelled"
    assert conversations.read(auth=_auth(), conversation_id=created.conversation_id).active_turn is None
    with database.sessions() as session:
        turn = session.get(ConversationTurnRow, submitted.turn_id)
    assert turn is not None and turn.status == "cancelled" and turn.task_state_version == state.state_version


def test_fast_path_exception_fails_turn_without_assistant_message(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        knowledge_queries=_BrokenKnowledge(),
    )
    created = conversations.create(auth=_auth())

    with pytest.raises(ConversationServiceError, match="conversation_content_unavailable"):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id, idempotency_key="broken-answer",
            kind="new", content="制度是什么", mode="answer",
        )
    with database.sessions() as session:
        turn = session.scalar(select(ConversationTurnRow))
        messages = tuple(session.scalars(select(ConversationMessageRow)))
    assert turn is not None and turn.status == "failed"
    assert [message.role for message in messages] == ["user"]
    recovered = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        knowledge_queries=_RefusingKnowledge(),
    )
    assert recovered.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="next-answer",
        kind="new", content="另一个问题", mode="answer",
    )


def test_memory_is_bounded_structured_and_persists_across_service_rebuild(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        knowledge_queries=_RefusingKnowledge(),
    )
    created = conversations.create(auth=_auth())
    for index in range(14):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id, idempotency_key=f"memory-{index}",
            kind="new",
            content=(
                f"比较供应商甲与供应商乙的 202{index % 6} 年方案；"
                "请用中文表格输出。" + "细节" * 300
            ),
            mode="answer",
        )

    repository = DatabaseConversationRepository(database)
    row = repository.get_owned(auth=_auth(), conversation_id=created.conversation_id)
    context = repository.context_messages(auth=_auth(), conversation_id=created.conversation_id)
    rebuilt = ConversationService(repository, DatabaseTaskRepository(database))
    assert rebuilt.messages(auth=_auth(), conversation_id=created.conversation_id).items
    assert row.summary is not None
    assert "主题与目标" in row.summary and "偏好" in row.summary
    assert "2020" in row.summary  # 多次压缩后仍保留最早的比较范围。
    assert _estimate_tokens(row.summary) <= 1_000
    assert sum(_estimate_tokens(item["content"]) for item in context) <= 4_000
    assert len(context) >= 7  # 一段摘要和优先保留的最近六条用户消息。


def test_memory_compaction_failure_degrades_to_recent_history(service, monkeypatch) -> None:
    _conversations, database = service
    repository = DatabaseConversationRepository(database)
    conversations = ConversationService(
        repository, DatabaseTaskRepository(database), knowledge_queries=_RefusingKnowledge()
    )
    created = conversations.create(auth=_auth())
    monkeypatch.setattr(
        repository, "_compact_memory_in_session", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    result = conversations.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="summary-failure",
        kind="new", content="仍然可以回答", mode="answer",
    )
    assert result.status_code == 200 and result.turn_status == "refused"


def test_sensitive_user_text_never_enters_summary_and_uncited_answer_is_hidden(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        knowledge_queries=_RefusingKnowledge(),
    )
    created = conversations.create(auth=_auth())
    for index in range(9):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id, idempotency_key=f"secret-{index}",
            kind="new", content=("密码 token=never-store；" if index == 0 else "比较范围；") + "填充" * 900,
            mode="answer",
        )
    repository = DatabaseConversationRepository(database)
    row = repository.get_owned(auth=_auth(), conversation_id=created.conversation_id)
    assert row.summary is not None and "token" not in row.summary.lower() and "密码" not in row.summary

    with database.sessions.begin() as session:
        turn = session.scalar(select(ConversationTurnRow).where(ConversationTurnRow.conversation_id == created.conversation_id))
        assert turn is not None
        session.add(
            ConversationMessageRow(
                message_id="uncited-answer", tenant_id="tenant-a", conversation_id=created.conversation_id,
                turn_id=turn.turn_id, sequence=999, role="assistant", kind="answer", content="企业事实",
                content_hash="sha256:" + "a" * 64, task_id=turn.task_id, evidence_ids=[], artifacts=[],
                created_at=datetime.now(UTC),
            )
        )
    hidden = conversations.messages(auth=_auth(), conversation_id=created.conversation_id).items[-1]
    assert hidden.content is None and hidden.visibility == "redacted"
    assert all(item["content"] != "企业事实" for item in repository.context_messages(
        auth=_auth(), conversation_id=created.conversation_id
    ))


def test_deleting_fast_conversation_cleans_empty_task_markers(service) -> None:
    _conversations, database = service
    repository = DatabaseConversationRepository(database)
    conversations = ConversationService(
        repository, DatabaseTaskRepository(database), knowledge_queries=_RefusingKnowledge()
    )
    created = conversations.create(auth=_auth(), title="待删除")
    conversations.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="delete-answer",
        kind="new", content="制度是什么", mode="answer",
    )
    assert conversations.delete(auth=_auth(), conversation_id=created.conversation_id).status == "deleting"
    assert conversations.delete(auth=_auth(), conversation_id=created.conversation_id).status == "deleting"
    assert repository.purge_deleting(limit=10) == 1
    with database.sessions() as session:
        row = session.get(ConversationRow, created.conversation_id)
        task = session.scalar(select(ConversationTurnRow).where(ConversationTurnRow.conversation_id == created.conversation_id))
    assert row is not None and row.status == "deleted" and row.title is None and row.summary is None
    assert task is not None
