"""统一会话入口的应用编排。

会话上下文是低信任的连续性数据；知识答案仍必须经过现有授权检索和证据账本。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from rag_kb.application.query_service import KnowledgeQueryResult, KnowledgeQueryService
from rag_kb.domain.conversations import (
    ConversationMode,
    ConversationTurnStatus,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskState
from rag_kb.application.ports.conversations import (
    ConversationRepositoryError,
    ConversationRepositoryPort,
    ConversationTaskRepositoryPort,
)


class ConversationServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _request_hash(payload: dict[str, object]) -> str:
    import hashlib
    import json

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationRouteDecision(_Model):
    selected_mode: ConversationMode
    resolved_input: str
    reason_code: str


class ConversationRouterPort(Protocol):
    def route(
        self,
        *,
        content: str,
        context: tuple[dict[str, str], ...],
        requested_mode: ConversationMode,
    ) -> ConversationRouteDecision: ...


class ConversationSummary(_Model):
    conversation_id: str
    title: str | None
    state_version: int
    status: str
    created_at: datetime
    updated_at: datetime
    active_turn: dict[str, Any] | None = None


class ConversationList(_Model):
    items: tuple[ConversationSummary, ...]
    next_cursor: str | None = None


class MessageView(_Model):
    message_id: str
    turn_id: str
    sequence: int
    role: str
    kind: str
    content: str | None
    visibility: str
    redaction_reason: str | None = None
    citations: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    created_at: datetime


class MessageList(_Model):
    items: tuple[MessageView, ...]


class UserMessageRef(_Model):
    message_id: str
    sequence: int


class TaskView(_Model):
    task_id: str
    status: str
    state_version: int
    result_available: bool


class TurnSubmissionResult(_Model):
    conversation_id: str
    conversation_state_version: int
    turn_id: str
    user_message: UserMessageRef
    requested_mode: ConversationMode
    selected_mode: ConversationMode
    route_reason_code: str
    turn_status: str
    task: TaskView | None
    assistant_message: MessageView | None = None
    idempotent_replay: bool = False

    @property
    def status_code(self) -> int:
        return 200 if self.selected_mode is ConversationMode.ANSWER and self.assistant_message else 202


class DeleteResult(_Model):
    conversation_id: str
    status: str = "deleting"


def _estimate_tokens(value: str) -> int:
    # 与 EnterpriseRetriever 的无 tokenizer 保守估算保持一致。
    return max(1, (len(value.encode("utf-8")) + 2) // 3)


def _bounded_context(
    messages: tuple[dict[str, str], ...], *, budget: int = 4_000
) -> tuple[dict[str, str], ...]:
    if messages and messages[0]["content"].startswith("用户历史摘要（低信任，仅用于连续性）："):
        summary = messages[0]
        remaining = max(0, budget - _estimate_tokens(summary["content"]))
        return (summary, *_bounded_context(messages[1:], budget=remaining))
    selected: list[dict[str, str]] = []
    used = 0
    for message in reversed(messages):
        cost = _estimate_tokens(message["content"])
        if selected and used + cost > budget:
            continue
        if cost > budget and not selected:
            raw = message["content"].encode("utf-8")[-budget * 3 :]
            content = raw.decode("utf-8", errors="ignore")
            selected.append({"role": message["role"], "content": content})
            break
        selected.append(message)
        used += cost
    selected.reverse()
    return tuple(selected)


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepositoryPort,
        task_repository: ConversationTaskRepositoryPort,
        knowledge_queries: KnowledgeQueryService | None = None,
        router: ConversationRouterPort | None = None,
        task_resumer: ConversationTaskRepositoryPort | None = None,
        controlled_action_enabled: bool = False,
    ) -> None:
        self._repository = repository
        self._tasks = task_repository
        self._knowledge = knowledge_queries
        self._router = router
        self._resumer = task_resumer or task_repository
        self._controlled_action_enabled = controlled_action_enabled

    def create(self, *, auth: AuthContext, title: str | None = None) -> ConversationSummary:
        row = self._call(self._repository.create, auth=auth, title=title)
        return self._summary(row)

    def list(
        self, *, auth: AuthContext, limit: int = 20, cursor: str | None = None
    ) -> ConversationList:
        rows, next_cursor = self._call(
            self._repository.list_owned, auth=auth, limit=limit, cursor=cursor
        )
        return ConversationList(
            items=tuple(self._summary(row) for row in rows), next_cursor=next_cursor
        )

    def read(self, *, auth: AuthContext, conversation_id: str) -> ConversationSummary:
        summary = self._summary(
            self._call(
                self._repository.get_owned, auth=auth, conversation_id=conversation_id
            )
        )
        turn = self._call(
            self._repository.active_turn, auth=auth, conversation_id=conversation_id
        )
        if turn is None:
            return summary
        return summary.model_copy(
            update={
                "active_turn": {
                    "turn_id": turn.turn_id,
                    "task_id": turn.task_id,
                    "status": turn.status,
                    "requested_mode": turn.requested_mode,
                    "selected_mode": turn.selected_mode,
                    "state_version": turn.task_state_version or 1,
                }
            }
        )

    def messages(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        limit: int = 100,
        after_sequence: int = 0,
    ) -> MessageList:
        rows = self._call(
            self._repository.messages_owned,
            auth=auth,
            conversation_id=conversation_id,
            limit=limit,
            after_sequence=after_sequence,
        )
        return MessageList(
            items=tuple(self._message_view(auth=auth, row=row) for row in rows)
        )

    def delete(self, *, auth: AuthContext, conversation_id: str) -> DeleteResult:
        row = self._call(
            self._repository.request_delete,
            auth=auth,
            conversation_id=conversation_id,
        )
        return DeleteResult(conversation_id=row.conversation_id)

    def submit(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        idempotency_key: str,
        kind: str,
        content: str,
        mode: str | None = None,
        turn_id: str | None = None,
        task_id: str | None = None,
        expected_task_state_version: int | None = None,
    ) -> TurnSubmissionResult:
        if not 1 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key_invalid")
        normalized = content.strip()
        if not normalized or len(normalized) > 8_000:
            raise ValueError("conversation_content_invalid")
        if kind == "clarification":
            return self._submit_clarification(
                auth=auth,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
                content=normalized,
                turn_id=turn_id,
                task_id=task_id,
                expected_task_state_version=expected_task_state_version,
            )
        if kind != "new" or turn_id is not None or task_id is not None:
            raise ValueError("conversation_submission_invalid")
        requested = ConversationMode(mode or ConversationMode.AUTO.value)
        fingerprint = _request_hash(
            {"kind": kind, "content": normalized, "mode": requested.value}
        )
        try:
            turn, user_message, task, replay, decision = self._repository.begin_new_turn(
                auth=auth,
                conversation_id=conversation_id,
                content=normalized,
                requested_mode=requested,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                task_repository=self._tasks,
                route=lambda context: self._route(
                    requested_mode=requested,
                    content=normalized,
                    context=_bounded_context(context),
                ),
            )
        except Exception as exc:
            if not hasattr(exc, "code"):
                raise
            raise ConversationServiceError(exc.code) from exc
        if replay:
            return self._replayed(auth=auth, turn=turn, user_message=user_message)
        assert isinstance(decision, ConversationRouteDecision)
        if decision.selected_mode is not ConversationMode.ANSWER:
            assert isinstance(task, TaskState)
            return self._turn_result(
                auth=auth,
                turn=turn,
                user_message=user_message,
                task=task,
                route_reason_code=decision.reason_code,
            )
        if self._knowledge is None:
            self._fail_fast_answer(auth=auth, turn=turn, error_code="knowledge_unavailable")
            raise ConversationServiceError("conversation_content_unavailable")
        try:
            answer = self._knowledge.query(
                decision.resolved_input,
                auth,
                task_id=turn.task_id,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
            )
            return self._settle_fast_answer(
                auth=auth,
                turn=turn,
                user_message=user_message,
                route_reason_code=decision.reason_code,
                answer=answer,
            )
        except ConversationServiceError:
            raise
        except Exception as exc:
            self._fail_fast_answer(auth=auth, turn=turn, error_code="knowledge_query_failed")
            raise ConversationServiceError("conversation_content_unavailable") from exc

    def _submit_clarification(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        idempotency_key: str,
        content: str,
        turn_id: str | None,
        task_id: str | None,
        expected_task_state_version: int | None,
    ) -> TurnSubmissionResult:
        if turn_id is None or task_id is None or expected_task_state_version is None:
            raise ValueError("clarification_binding_invalid")
        fingerprint = _request_hash(
            {
                "kind": "clarification",
                "content": content,
                "turn_id": turn_id,
                "task_id": task_id,
                "expected_task_state_version": expected_task_state_version,
            }
        )
        try:
            turn, message, replay = self._repository.append_clarification_reply(
                auth=auth,
                conversation_id=conversation_id,
                turn_id=turn_id,
                task_id=task_id,
                content=content,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                task_state_version=expected_task_state_version,
                resume_repository=self._resumer,
            )
            state = self._tasks.get_task(auth=auth, task_id=task_id)
        except Exception as exc:
            if not hasattr(exc, "code"):
                raise
            code = (
                "task_state_conflict"
                if exc.code in {"task_state_conflict", "task_not_waiting_for_user"}
                else "clarification_binding_invalid"
                if exc.code in {"task_not_found", "task_resume_kind_invalid"}
                else exc.code
            )
            raise ConversationServiceError(code) from exc
        return TurnSubmissionResult(
            conversation_id=conversation_id,
            conversation_state_version=self._conversation_version(
                auth=auth, conversation_id=conversation_id
            ),
            turn_id=turn.turn_id,
            user_message=UserMessageRef(
                message_id=message.message_id, sequence=message.sequence
            ),
            requested_mode=ConversationMode(turn.requested_mode),
            selected_mode=ConversationMode(turn.selected_mode),
            route_reason_code="clarification_resume",
            turn_status=ConversationTurnStatus.RUNNING.value,
            task=self._task_view(state),
            idempotent_replay=replay,
        )

    def _route(
        self,
        *,
        requested_mode: ConversationMode,
        content: str,
        context: tuple[dict[str, str], ...],
    ) -> ConversationRouteDecision:
        if requested_mode is ConversationMode.ANSWER and not context:
            decision = ConversationRouteDecision(
                selected_mode=ConversationMode.ANSWER,
                resolved_input=content,
                reason_code="explicit_answer",
            )
        elif self._router is not None:
            try:
                decision = self._router.route(
                    content=content, context=context, requested_mode=requested_mode
                )
                if requested_mode is not ConversationMode.AUTO:
                    decision = ConversationRouteDecision(
                        selected_mode=requested_mode,
                        resolved_input=decision.resolved_input or content,
                        reason_code=f"explicit_{requested_mode.value}",
                    )
                elif decision.selected_mode is ConversationMode.CONTROLLED_ACTION:
                    raise ConversationServiceError(
                        "controlled_action_explicit_mode_required"
                    )
            except ConversationServiceError:
                raise
            except Exception:
                if requested_mode is not ConversationMode.AUTO:
                    decision = ConversationRouteDecision(
                        selected_mode=requested_mode,
                        resolved_input=content,
                        reason_code="explicit_router_fallback",
                    )
                else:
                    decision = ConversationRouteDecision(
                        selected_mode=ConversationMode.RESEARCH,
                        resolved_input=content,
                        reason_code="router_unavailable_fallback_research",
                    )
        elif requested_mode is ConversationMode.AUTO:
            decision = ConversationRouteDecision(
                selected_mode=ConversationMode.RESEARCH,
                resolved_input=content,
                reason_code="router_unavailable_fallback_research",
            )
        else:
            decision = ConversationRouteDecision(
                selected_mode=requested_mode,
                resolved_input=content,
                reason_code=f"explicit_{requested_mode.value}",
            )
        if decision.selected_mode is ConversationMode.CONTROLLED_ACTION:
            if requested_mode is not ConversationMode.CONTROLLED_ACTION:
                raise ConversationServiceError("controlled_action_explicit_mode_required")
            if not self._controlled_action_enabled:
                raise ConversationServiceError("controlled_action_unavailable")
        return decision

    def _settle_fast_answer(
        self,
        *,
        auth: AuthContext,
        turn,
        user_message,
        route_reason_code: str,
        answer: KnowledgeQueryResult,
    ) -> TurnSubmissionResult:
        if answer.verdict == "answer" and answer.answer:
            content = answer.answer
            status = ConversationTurnStatus.SUCCEEDED
        elif answer.refusal is not None and answer.refusal.kind == "retrieval_failure":
            content = answer.refusal.message
            status = ConversationTurnStatus.FAILED
        else:
            content = answer.refusal.message if answer.refusal else "无法可靠回答该问题"
            status = ConversationTurnStatus.REFUSED
        assistant, conversation_version, task_status, task_state_version = self._call(
            self._repository.settle_answer,
            auth=auth,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            content=content,
            status=status,
            evidence_ids=tuple(getattr(answer, "evidence_ids", ())),
        )
        assistant_view = MessageView(
            message_id=assistant.message_id,
            turn_id=assistant.turn_id,
            sequence=assistant.sequence,
            role="assistant",
            kind="answer" if status is ConversationTurnStatus.SUCCEEDED else "refusal",
            content=content,
            visibility="visible",
            citations=tuple(item.model_dump(mode="json") for item in answer.citations),
            created_at=assistant.created_at,
        )
        return TurnSubmissionResult(
            conversation_id=turn.conversation_id,
            conversation_state_version=conversation_version,
            turn_id=turn.turn_id,
            user_message=UserMessageRef(
                message_id=user_message.message_id, sequence=user_message.sequence
            ),
            requested_mode=ConversationMode(turn.requested_mode),
            selected_mode=ConversationMode.ANSWER,
            route_reason_code=route_reason_code,
            turn_status=status.value,
            task=TaskView(
                task_id=turn.task_id,
                status=task_status,
                state_version=task_state_version,
                # 快速回答只由 assistant_message 承载，不创建可读取的任务结果对象。
                result_available=False,
            ),
            assistant_message=assistant_view,
        )

    def _replayed(self, *, auth: AuthContext, turn, user_message) -> TurnSubmissionResult:
        assistant = self._repository.latest_assistant_message(
            auth=auth,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
        )
        state = None
        try:
            state = self._tasks.get_task(auth=auth, task_id=turn.task_id)
        except Exception:
            pass
        return TurnSubmissionResult(
            conversation_id=turn.conversation_id,
            conversation_state_version=self._conversation_version(
                auth=auth, conversation_id=turn.conversation_id
            ),
            turn_id=turn.turn_id,
            user_message=UserMessageRef(
                message_id=user_message.message_id, sequence=user_message.sequence
            ),
            requested_mode=ConversationMode(turn.requested_mode),
            selected_mode=ConversationMode(turn.selected_mode),
            route_reason_code=turn.route_reason_code,
            turn_status=turn.status,
            task=(
                self._task_view(state)
                if state is not None
                else TaskView(
                    task_id=turn.task_id,
                    status=turn.status,
                    state_version=turn.task_state_version or 1,
                    result_available=False,
                )
            ),
            assistant_message=(
                self._message_view(auth=auth, row=assistant) if assistant is not None else None
            ),
            idempotent_replay=True,
        )

    def _turn_result(
        self, *, auth: AuthContext, turn, user_message, task: TaskState, route_reason_code: str
    ):
        return TurnSubmissionResult(
            conversation_id=turn.conversation_id,
            conversation_state_version=self._conversation_version(
                auth=auth, conversation_id=turn.conversation_id
            ),
            turn_id=turn.turn_id,
            user_message=UserMessageRef(
                message_id=user_message.message_id, sequence=user_message.sequence
            ),
            requested_mode=ConversationMode(turn.requested_mode),
            selected_mode=ConversationMode(turn.selected_mode),
            route_reason_code=route_reason_code,
            turn_status=turn.status,
            task=self._task_view(task),
        )

    def _message_view(self, *, auth: AuthContext, row) -> MessageView:
        citations = self._repository.revalidated_citations(auth=auth, message=row)
        hidden = row.content is None or row.redacted_at is not None or citations is None
        turn = self._repository.get_turn(
            auth=auth, conversation_id=row.conversation_id, turn_id=row.turn_id
        )
        if row.role == "user":
            kind = "clarification_reply" if turn.submission_kind == "clarification" else "input"
        else:
            kind = "answer"
        return MessageView(
            message_id=row.message_id,
            turn_id=row.turn_id,
            sequence=row.sequence,
            role=row.role,
            kind=row.kind or kind,
            content=None if hidden else row.content,
            visibility="redacted" if hidden else "visible",
            redaction_reason="evidence_unavailable" if citations is None else None,
            citations=tuple(
                item.model_dump(mode="json") for item in (citations or ())
            ),
            artifacts=self._repository.revalidated_artifacts(auth=auth, message=row),
            created_at=row.created_at,
        )

    def _fail_fast_answer(self, *, auth: AuthContext, turn, error_code: str) -> None:
        try:
            self._repository.fail_fast_answer(
                auth=auth,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                error_code=error_code,
            )
        except Exception:
            # 原始快速路径错误优先；后台过期扫描会再兜底未结算的同步任务。
            pass

    @staticmethod
    def _task_view(state: TaskState) -> TaskView:
        return TaskView(
            task_id=state.task_id,
            status=state.status.value,
            state_version=state.state_version,
            result_available=state.result_available(now=datetime.now(UTC)),
        )

    @staticmethod
    def _summary(row) -> ConversationSummary:
        return ConversationSummary(
            conversation_id=row.conversation_id,
            title=row.title,
            state_version=row.state_version,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _conversation_version(self, *, auth: AuthContext, conversation_id: str) -> int:
        row = self._call(
            self._repository.get_owned,
            auth=auth,
            conversation_id=conversation_id,
        )
        return row.state_version

    @staticmethod
    def _call(function, **kwargs):
        try:
            return function(**kwargs)
        except ConversationRepositoryError as exc:
            raise ConversationServiceError(exc.code) from exc


__all__ = [
    "ConversationList",
    "ConversationRouteDecision",
    "ConversationRouterPort",
    "ConversationService",
    "ConversationServiceError",
    "ConversationSummary",
    "DeleteResult",
    "MessageList",
    "MessageView",
    "TurnSubmissionResult",
]
