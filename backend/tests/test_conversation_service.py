from datetime import UTC, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import func, select

from rag_kb.application.conversation_service import (
    ConversationRouteDecision,
    ConversationService,
    ConversationServiceError,
    _estimate_tokens,
)
from rag_kb.application.ports.conversations import ConversationRepositoryError
from rag_kb.domain.conversations import ConversationMode, ConversationStatus, ConversationTurnStatus
from rag_kb.application.query_service import KnowledgeQueryExecution, KnowledgeQueryResult
from rag_kb.domain.identity import AuthContext
from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.conversation_repository import (
    DatabaseConversationRepository,
    context_hash,
)
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ArtifactRow,
    ConversationMessageRow,
    ConversationRow,
    ConversationTurnRow,
    PrincipalRow,
    TaskResumeCommandRow,
    TenantRow,
    ToolCallRow,
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
    history = conversations.messages(auth=_auth(), conversation_id=created.conversation_id)
    assert history.items[0].task_id == first.task.task_id
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


def test_list_only_paginates_active_conversations(service) -> None:
    conversations, _database = service
    created = [conversations.create(auth=_auth()) for _ in range(3)]

    assert conversations.delete(
        auth=_auth(), conversation_id=created[0].conversation_id
    ).status == "deleting"
    assert conversations.delete(
        auth=_auth(), conversation_id=created[1].conversation_id
    ).status == "deleting"

    page = conversations.list(auth=_auth(), limit=1)

    assert [item.conversation_id for item in page.items] == [created[2].conversation_id]
    assert page.next_cursor is None


def test_explicit_document_scope_is_durable_and_part_of_idempotency(service) -> None:
    conversations, database = service
    created = conversations.create(auth=_auth())

    submitted = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="scoped-research",
        kind="new",
        content="总结这份文档",
        mode="research",
        scope_document_id="document-a",
    )

    assert submitted.task is not None
    with database.sessions() as session:
        turn = session.get(ConversationTurnRow, submitted.turn_id)
        task = session.get(AgentTaskRow, submitted.task.task_id)
    assert turn is not None and turn.scope_document_id == "document-a"
    assert task is not None and task.scope_document_id == "document-a"
    with pytest.raises(ConversationServiceError, match="idempotency_conflict"):
        conversations.submit(
            auth=_auth(),
            conversation_id=created.conversation_id,
            idempotency_key="scoped-research",
            kind="new",
            content="总结这份文档",
            mode="research",
            scope_document_id="document-b",
        )


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


class _FencedKnowledge:
    def __init__(self, database, after_start) -> None:
        self.database = database
        self.after_start = after_start
        self.calls = 0

    def query(self, _question, auth, *, task_id, conversation_id, turn_id, **_options):
        self.calls += 1
        with self.database.sessions.begin() as session:
            session.add(
                AgentTaskRow(
                    task_id=task_id,
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    conversation_id=conversation_id,
                    conversation_turn_id=turn_id,
                    task_mode="knowledge_query",
                    status="running",
                    goal="knowledge_query:test",
                    auth_context_ref="sha256:test",
                    lease_owner="api-sync-knowledge-query",
                    lease_expires_at=datetime.now(UTC) + timedelta(seconds=30),
                    lease_generation=1,
                )
            )
        self.after_start(task_id, auth)
        return KnowledgeQueryResult(
            verdict="refusal",
            refusal=Refusal(kind="insufficient_evidence", message="证据不足"),
            execution=KnowledgeQueryExecution(
                task_id=task_id,
                lease_owner="api-sync-knowledge-query",
                lease_generation=1,
            ),
        )


class _ControlledRouter:
    def route(self, *, content, context, requested_mode):
        return ConversationRouteDecision(
            selected_mode=ConversationMode.CONTROLLED_ACTION,
            resolved_input=content,
            reason_code="auto_controlled_action",
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
    assert result.task is None
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
    assert replay.task is None
    with database.sessions() as session:
        turn = session.get(ConversationTurnRow, result.turn_id)
        tasks = tuple(session.scalars(select(AgentTaskRow)))
    assert turn is not None
    assert turn.status == result.turn_status == "refused"
    assert turn.task_id is None and turn.task_state_version is None
    assert tasks == ()
    history = conversations.messages(
        auth=_auth(), conversation_id=created.conversation_id
    )
    assert [item.kind for item in history.items] == ["input", "refusal"]
    assert all(item.task_id is None for item in history.items)
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


def test_auto_creates_controlled_action_turn_for_write_intent(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        router=_ControlledRouter(), controlled_action_enabled=True,
    )
    created = conversations.create(auth=_auth())
    result = conversations.submit(
        auth=_auth(), conversation_id=created.conversation_id, idempotency_key="auto-write",
        kind="new", content="研究制度并创建工作项", mode="auto",
    )

    assert result.status_code == 202
    assert result.requested_mode is ConversationMode.AUTO
    assert result.selected_mode is ConversationMode.CONTROLLED_ACTION
    assert result.route_reason_code == "auto_controlled_action"
    assert result.task is not None
    with database.sessions() as session:
        turn = session.get(ConversationTurnRow, result.turn_id)
        task = session.get(AgentTaskRow, result.task.task_id)
    assert turn is not None and turn.requested_mode == "auto"
    assert turn.selected_mode == "controlled_action"
    assert task is not None and task.task_mode == "controlled_action"


def test_auto_write_intent_fails_closed_when_controlled_actions_are_disabled(service) -> None:
    _conversations, database = service
    conversations = ConversationService(
        DatabaseConversationRepository(database), DatabaseTaskRepository(database),
        router=_ControlledRouter(), controlled_action_enabled=False,
    )
    created = conversations.create(auth=_auth())

    with pytest.raises(ConversationServiceError, match="controlled_action_unavailable"):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id, idempotency_key="auto-write-disabled",
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


def test_cancelled_answer_cannot_be_settled_or_retried(service) -> None:
    _conversations, database = service

    def cancel(task_id, auth) -> None:
        DatabaseTaskRepository(database).cancel(
            auth=auth,
            task_id=task_id,
            expected_state_version=1,
        )

    knowledge = _FencedKnowledge(database, cancel)
    conversations = ConversationService(
        DatabaseConversationRepository(database),
        DatabaseTaskRepository(database),
        knowledge_queries=knowledge,
    )
    created = conversations.create(auth=_auth())

    with pytest.raises(ConversationServiceError, match="conversation_answer_stale"):
        conversations.submit(
            auth=_auth(),
            conversation_id=created.conversation_id,
            idempotency_key="cancelled-answer",
            kind="new",
            content="制度是什么？",
            mode="answer",
        )

    replay = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="cancelled-answer",
        kind="new",
        content="制度是什么？",
        mode="answer",
    )
    with database.sessions() as session:
        task = session.scalar(select(AgentTaskRow))
        messages = tuple(session.scalars(select(ConversationMessageRow)))
    assert knowledge.calls == 1
    assert replay.idempotent_replay is True
    assert task is not None and task.status == "cancelled"
    assert [message.role for message in messages] == ["user"]


def test_concurrent_answer_settlement_writes_one_terminal_message(service) -> None:
    _conversations, database = service
    repository = DatabaseConversationRepository(database)
    task_repository = DatabaseTaskRepository(database)
    created = repository.create(auth=_auth())
    turn, _message, _task, _replay, _decision = repository.begin_new_turn(
        auth=_auth(),
        conversation_id=created.conversation_id,
        content="制度是什么？",
        requested_mode=ConversationMode.ANSWER,
        idempotency_key="concurrent-settlement",
        request_fingerprint="sha256:concurrent-settlement",
        task_repository=task_repository,
        route=lambda _context: ConversationRouteDecision(
            selected_mode=ConversationMode.ANSWER,
            resolved_input="制度是什么？",
            reason_code="explicit_answer",
        ),
    )
    with database.sessions.begin() as session:
        session.add(
            AgentTaskRow(
                task_id=turn.task_id,
                tenant_id="tenant-a",
                user_id="owner-a",
                conversation_id=created.conversation_id,
                conversation_turn_id=turn.turn_id,
                task_mode="knowledge_query",
                status="running",
                goal="knowledge_query:test",
                auth_context_ref="sha256:test",
                lease_owner="api-sync-knowledge-query",
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=30),
                lease_generation=1,
            )
        )
    barrier = Barrier(2)

    def settle() -> str:
        barrier.wait()
        try:
            repository.settle_answer(
                auth=_auth(),
                conversation_id=created.conversation_id,
                turn_id=turn.turn_id,
                task_id=turn.task_id,
                task_lease_owner="api-sync-knowledge-query",
                task_lease_generation=1,
                content="证据不足",
                status=ConversationTurnStatus.REFUSED,
            )
            return "settled"
        except ConversationRepositoryError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: settle(), range(2)))
    with database.sessions() as session:
        messages = tuple(session.scalars(select(ConversationMessageRow)))

    assert sorted(outcomes) == ["conversation_answer_stale", "settled"]
    assert [message.role for message in messages] == ["user", "assistant"]


@pytest.mark.parametrize("transition", ["expired", "transferred"])
def test_late_answer_cannot_overwrite_expired_or_transferred_execution(service, transition) -> None:
    _conversations, database = service

    def mutate(task_id, _auth_context) -> None:
        with database.sessions.begin() as session:
            task = session.get(AgentTaskRow, task_id)
            assert task is not None
            if transition == "expired":
                task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
                return
            task.lease_owner = "other-worker"
            task.lease_generation = 2
            task.state_version = 2
            turn = session.scalar(
                select(ConversationTurnRow).where(ConversationTurnRow.task_id == task_id)
            )
            assert turn is not None
            turn.task_state_version = 2

    conversations = ConversationService(
        DatabaseConversationRepository(database),
        DatabaseTaskRepository(database),
        knowledge_queries=_FencedKnowledge(database, mutate),
    )
    created = conversations.create(auth=_auth())

    with pytest.raises(ConversationServiceError, match="conversation_answer_stale"):
        conversations.submit(
            auth=_auth(),
            conversation_id=created.conversation_id,
            idempotency_key=f"late-{transition}",
            kind="new",
            content="制度是什么？",
            mode="answer",
        )

    with database.sessions() as session:
        task = session.scalar(select(AgentTaskRow))
        messages = tuple(session.scalars(select(ConversationMessageRow)))
    assert task is not None and task.status == "running"
    assert task.lease_owner == (
        "api-sync-knowledge-query" if transition == "expired" else "other-worker"
    )
    assert [message.role for message in messages] == ["user"]


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


class _ContextEvidence:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def citations_from_ledger(self, _session, *, evidence_ids, **_kwargs):
        if self.fail:
            raise RuntimeError("acl_revoked")
        return tuple(object() for _ in evidence_ids)


def test_context_keeps_role_order_and_fail_closed_assistant_visibility(service) -> None:
    _conversations, database = service
    repository = DatabaseConversationRepository(database, _ContextEvidence())
    conversations = ConversationService(
        repository, DatabaseTaskRepository(database), knowledge_queries=_RefusingKnowledge()
    )
    created = conversations.create(auth=_auth())
    for index in range(2):
        conversations.submit(
            auth=_auth(), conversation_id=created.conversation_id,
            idempotency_key=f"role-context-{index}", kind="new",
            content=f"用户问题 {index}", mode="answer",
        )
    with database.sessions.begin() as session:
        answers = tuple(
            session.scalars(
                select(ConversationMessageRow)
                .where(ConversationMessageRow.role == "assistant")
                .order_by(ConversationMessageRow.sequence)
            )
        )
        answers[0].kind = "answer"
        answers[0].content = "1. 岗位与清单确认\n2. 试用期权限复核"
        answers[0].task_id = "task-visible"
        answers[0].evidence_ids = ["evidence-visible"]

    context = repository.context_messages(
        auth=_auth(), conversation_id=created.conversation_id
    )
    assert [item["role"] for item in context] == [
        "user", "assistant", "user", "assistant"
    ]
    assert "岗位与清单确认" in context[1]["content"]
    assert context_hash(context) == context_hash(tuple(dict(item) for item in context))

    hidden_repository = DatabaseConversationRepository(database, _ContextEvidence(fail=True))
    hidden = hidden_repository.context_messages(
        auth=_auth(), conversation_id=created.conversation_id
    )
    assert [item["role"] for item in hidden] == ["user", "user", "assistant"]
    assert all("岗位与清单确认" not in item["content"] for item in hidden)


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
        turn = session.scalar(select(ConversationTurnRow).where(ConversationTurnRow.conversation_id == created.conversation_id))
        messages = session.scalar(
            select(func.count())
            .select_from(ConversationMessageRow)
            .where(ConversationMessageRow.conversation_id == created.conversation_id)
        )
    # 真删除：会话、回合与消息行物理移除（快速路径不产生任务行）。
    assert row is None and turn is None and messages == 0


def test_purge_deleting_removes_legacy_deleted_husk_rows(service) -> None:
    _conversations, database = service
    repository = DatabaseConversationRepository(database)
    conversations = ConversationService(
        repository, DatabaseTaskRepository(database), knowledge_queries=_RefusingKnowledge()
    )
    created = conversations.create(auth=_auth(), title="旧壳行")
    # 模拟旧行为遗留的 deleted 空壳行（正文已清、行保留）。
    with database.sessions.begin() as session:
        row = session.get(ConversationRow, created.conversation_id)
        now = datetime.now(UTC)
        row.status = ConversationStatus.DELETED.value
        row.deleting_at = now
        row.deleted_at = now

    assert repository.purge_deleting(limit=10) == 1

    with database.sessions() as session:
        assert session.get(ConversationRow, created.conversation_id) is None


def test_deleting_conversation_expires_linked_artifact_content(service) -> None:
    _conversations, database = service
    repository = DatabaseConversationRepository(database)
    conversations = ConversationService(
        repository,
        DatabaseTaskRepository(database),
        router=_ControlledRouter(),
        controlled_action_enabled=True,
    )
    created = conversations.create(auth=_auth())
    submitted = conversations.submit(
        auth=_auth(),
        conversation_id=created.conversation_id,
        idempotency_key="delete-artifact",
        kind="new",
        content="生成制度摘要",
        mode="controlled_action",
    )
    assert submitted.task is not None
    now = datetime.now(UTC)
    with database.sessions.begin() as session:
        task = session.get(AgentTaskRow, submitted.task.task_id)
        turn = session.get(ConversationTurnRow, submitted.turn_id)
        assert task is not None and turn is not None
        task.status = "failed"
        task.checkpoint_purged_at = now
        task.result_purged_at = now
        turn.status = ConversationTurnStatus.FAILED.value
        session.add(
            ToolCallRow(
                tool_call_id="artifact-call",
                task_id=task.task_id,
                tenant_id="tenant-a",
                approval_id=None,
                plan_revision=None,
                plan_digest=None,
                evidence_digest=None,
                tool_name="search_knowledge",
                tool_version="1.0.0",
                arguments_hash="sha256:" + "a" * 64,
                principal_hash="sha256:" + "b" * 64,
                mode="read",
                idempotency_key=None,
                risk_level="low",
                status="running",
                execution_deadline=now,
            )
        )
        session.flush()
        session.add(
            ArtifactRow(
                artifact_id="artifact-delete",
                task_id=task.task_id,
                tenant_id="tenant-a",
                tool_call_id="artifact-call",
                artifact_type="report",
                storage_key="artifacts/tenant-a/task/report.md",
                content_hash="sha256:" + "c" * 64,
                status="ready",
                content_expires_at=now + timedelta(days=1),
            )
        )

    assert conversations.delete(
        auth=_auth(), conversation_id=created.conversation_id
    ).status == ConversationStatus.DELETING.value
    assert repository.purge_deleting(limit=10) == 1
    with database.sessions() as session:
        artifact = session.get(ArtifactRow, "artifact-delete")
        task = session.get(AgentTaskRow, submitted.task.task_id)
        turn = session.get(ConversationTurnRow, submitted.turn_id)
    assert artifact is not None and artifact.content_expires_at is not None
    # 真删除：回合行物理移除；任务行保留为独立记录（goal 打码、解除会话绑定）。
    assert turn is None
    assert task is not None and task.goal == "[conversation content purged]"
    assert task.conversation_id is None and task.conversation_turn_id is None
    expires_at = artifact.content_expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    assert expires_at <= datetime.now(UTC)
