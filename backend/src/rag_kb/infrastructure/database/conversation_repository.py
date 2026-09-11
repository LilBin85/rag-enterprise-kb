"""统一会话的 PostgreSQL 仓储。

会话只保存对话连续性；企业事实仍由任务证据账本判定。
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from rag_kb.domain.conversations import (
    ACTIVE_TURN_STATUSES,
    ConversationMessageRole,
    ConversationMode,
    ConversationStatus,
    ConversationSubmissionKind,
    ConversationTurnStatus,
)
from rag_kb.application.ports.conversations import (
    ConversationRepositoryError,
    ConversationTaskRepositoryPort,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.database.models import (
    ConversationMessageRow,
    ConversationRow,
    ConversationTurnRow,
    AgentTaskRow,
    ArtifactRow,
    ToolCallRow,
    TenantRow,
)
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


def request_hash(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def context_hash(messages: tuple[dict[str, str], ...]) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value.encode("utf-8")) + 2) // 3)


def _truncate_to_tokens(value: str, budget: int) -> str:
    if _estimate_tokens(value) <= budget:
        return value
    return value.encode("utf-8")[: budget * 3].decode("utf-8", errors="ignore").strip()


def _safe_memory_text(value: str) -> str:
    """只保留连续性线索，避免把凭证、权限或执行载荷写进摘要。"""
    sensitive = (
        "password", "passwd", "token", "secret", "credential", "api key", "apikey",
        "密码", "令牌", "密钥", "凭证", "权限", "审批", "批准", "拒绝", "参数",
    )
    lines = (
        " ".join(line.split())
        for line in value.splitlines()
        if line.strip()
    )
    return " ".join(
        line for line in lines if not any(word in line.lower() for word in sensitive)
    )


def _merge_memory_summary(existing: str | None, messages: tuple[str, ...]) -> str:
    """以固定栏目合并旧摘要与新用户消息，始终保留早期连续性线索。"""
    sections = {
        "主题与目标": [],
        "偏好": [],
        "实体与范围": [],
        "未决问题与指代": [],
    }
    if existing:
        for line in existing.splitlines():
            name, separator, value = line.partition("：")
            if separator and name in sections:
                safe = _safe_memory_text(value)
                if safe:
                    sections[name].append(safe)
    for source in messages:
        safe = _safe_memory_text(source)
        if safe:
            lower = safe.lower()
            sections["主题与目标"].append(safe)
            if any(word in lower for word in ("格式", "语言", "输出", "表格", "列表", "中文", "英文")):
                sections["偏好"].append(safe)
            if any(word in lower for word in ("比较", "对比", "版本", "时间", "年", "月", "日", "与", "和")):
                sections["实体与范围"].append(safe)
            if "?" in safe or "？" in safe or any(
                word in lower for word in ("上一", "上述", "前者", "后者", "这个", "那个", "它")
            ):
                sections["未决问题与指代"].append(safe)
    lines: list[str] = []
    for name, values in sections.items():
        if not values:
            continue
        # 已压缩的旧摘要在同一栏目保留在前，新信息随后补充。
        unique = tuple(dict.fromkeys(values))
        lines.append(f"{name}：" + _truncate_to_tokens("；".join(unique), 240))
    return _truncate_to_tokens("\n".join(lines), 1_000)


def _cursor(updated_at: datetime, conversation_id: str) -> str:
    value = json.dumps([updated_at.isoformat(), conversation_id], separators=(",", ":"))
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[datetime, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = json.loads(base64.urlsafe_b64decode(padded).decode())
        when = datetime.fromisoformat(raw[0])
        if when.tzinfo is None or not isinstance(raw[1], str):
            raise ValueError
        return when, raw[1]
    except Exception as exc:
        raise ConversationRepositoryError("conversation_cursor_invalid") from exc


class DatabaseConversationRepository:
    def __init__(
        self, database: Database, evidence: EvidenceLedgerRepository | None = None
    ) -> None:
        self._database = database
        self._evidence = evidence or EvidenceLedgerRepository()

    @property
    def database(self) -> Database:
        return self._database

    def create(self, *, auth: AuthContext, title: str | None = None) -> ConversationRow:
        normalized = title.strip() if title is not None else None
        if normalized is not None and not 1 <= len(normalized) <= 200:
            raise ValueError("conversation_title_invalid")
        now = datetime.now(UTC)
        row = ConversationRow(
            conversation_id=uuid4().hex,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            title=normalized,
            status=ConversationStatus.ACTIVE.value,
            state_version=1,
            summary_through_sequence=0,
            memory_version=0,
            created_at=now,
            updated_at=now,
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            tenant = work.session.get(TenantRow, auth.tenant_id)
            if tenant is None:
                raise ConversationRepositoryError("conversation_not_found")
            if tenant.conversation_retention_days is not None:
                row.content_expires_at = now + timedelta(
                    days=tenant.conversation_retention_days
                )
            work.session.add(row)
            work.session.flush()
        return row

    def list_owned(
        self, *, auth: AuthContext, limit: int = 20, cursor: str | None = None
    ) -> tuple[tuple[ConversationRow, ...], str | None]:
        if not 1 <= limit <= 100:
            raise ValueError("conversation_limit_invalid")
        with self._database.sessions() as session:
            statement = select(ConversationRow).where(
                ConversationRow.tenant_id == auth.tenant_id,
                ConversationRow.user_id == auth.user_id,
                ConversationRow.status != ConversationStatus.DELETED.value,
            )
            if cursor:
                when, conversation_id = _decode_cursor(cursor)
                statement = statement.where(
                    (ConversationRow.updated_at < when)
                    | (
                        (ConversationRow.updated_at == when)
                        & (ConversationRow.conversation_id < conversation_id)
                    )
                )
            rows = tuple(
                session.scalars(
                    statement.order_by(
                        ConversationRow.updated_at.desc(),
                        ConversationRow.conversation_id.desc(),
                    ).limit(limit + 1)
                )
            )
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = _cursor(rows[-1].updated_at, rows[-1].conversation_id)
        return rows, next_cursor

    def get_owned(self, *, auth: AuthContext, conversation_id: str) -> ConversationRow:
        with self._database.sessions() as session:
            row = session.scalar(
                select(ConversationRow).where(
                    ConversationRow.tenant_id == auth.tenant_id,
                    ConversationRow.user_id == auth.user_id,
                    ConversationRow.conversation_id == conversation_id,
                    ConversationRow.status != ConversationStatus.DELETED.value,
                )
            )
            if row is None:
                raise ConversationRepositoryError("conversation_not_found")
            session.expunge(row)
            return row

    def messages_owned(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        limit: int = 100,
        after_sequence: int = 0,
    ) -> tuple[ConversationMessageRow, ...]:
        if not 1 <= limit <= 100 or after_sequence < 0:
            raise ValueError("conversation_message_page_invalid")
        self.get_owned(auth=auth, conversation_id=conversation_id)
        with self._database.sessions() as session:
            return tuple(
                session.scalars(
                    select(ConversationMessageRow)
                    .where(
                        ConversationMessageRow.tenant_id == auth.tenant_id,
                        ConversationMessageRow.conversation_id == conversation_id,
                        ConversationMessageRow.sequence > after_sequence,
                    )
                    .order_by(ConversationMessageRow.sequence)
                    .limit(limit)
                )
            )

    def context_messages(
        self, *, auth: AuthContext, conversation_id: str
    ) -> tuple[dict[str, str], ...]:
        """返回当前用户仍可见的低信任对话；企业事实仍必须重新检索。"""
        conversation = self.get_owned(auth=auth, conversation_id=conversation_id)
        with self._database.sessions() as session:
            return self._context_messages_in_session(session, conversation, auth=auth)

    @staticmethod
    def _recent_context(
        rows: tuple[ConversationMessageRow, ...], *, budget: int = 3_000
    ) -> tuple[dict[str, str], ...]:
        turn_ids = tuple(dict.fromkeys(row.turn_id for row in reversed(rows)))[:6]
        recent_turn_ids = frozenset(turn_ids)
        candidates = [row for row in rows if row.turn_id in recent_turn_ids]
        selected: list[dict[str, str]] = []
        used = 0
        for index, row in enumerate(candidates):
            assert row.content is not None
            content = row.content
            cost = _estimate_tokens(content)
            # 最近六条优先完整保留；过长时平均压缩，确保每条仍参与连续性解析。
            allowance = max(1, (budget - used) // (len(candidates) - index))
            if cost > allowance:
                content = _truncate_to_tokens(content, allowance)
                cost = _estimate_tokens(content)
            selected.append({"role": row.role, "content": content})
            used += cost
        return tuple(selected)

    def _context_messages_in_session(
        self, session, conversation: ConversationRow, *, auth: AuthContext
    ) -> tuple[dict[str, str], ...]:
        rows = tuple(
            reversed(
                tuple(
                    session.scalars(
                        select(ConversationMessageRow)
                        .join(
                            ConversationTurnRow,
                            (
                                ConversationTurnRow.tenant_id
                                == ConversationMessageRow.tenant_id
                            )
                            & (
                                ConversationTurnRow.conversation_id
                                == ConversationMessageRow.conversation_id
                            )
                            & (
                                ConversationTurnRow.turn_id
                                == ConversationMessageRow.turn_id
                            ),
                        )
                        .where(
                            ConversationMessageRow.tenant_id == conversation.tenant_id,
                            ConversationMessageRow.conversation_id == conversation.conversation_id,
                            ConversationMessageRow.sequence
                            > conversation.summary_through_sequence,
                            ConversationTurnRow.status.in_(
                                (
                                    ConversationTurnStatus.SUCCEEDED.value,
                                    ConversationTurnStatus.REFUSED.value,
                                )
                            ),
                        )
                        .order_by(ConversationMessageRow.sequence.desc())
                        .limit(100)
                    )
                )
            )
        )
        visible: list[ConversationMessageRow] = []
        for row in rows:
            if row.content is None or row.redacted_at is not None or row.content_purged_at is not None:
                continue
            if row.role == ConversationMessageRole.USER.value:
                visible.append(row)
                continue
            if row.kind == "refusal" and not row.evidence_ids:
                visible.append(row)
                continue
            if row.kind != "answer":
                continue
            if self._revalidated_citations_in_session(session, auth=auth, message=row) is not None:
                visible.append(row)
        recent = self._recent_context(tuple(visible))
        if conversation.summary:
            return (
                {
                    "role": "user",
                    "content": "用户历史摘要（低信任，仅用于连续性）：\n"
                    + _truncate_to_tokens(conversation.summary, 950),
                },
                *recent,
            )
        return recent

    def _compact_memory_in_session(self, session, conversation: ConversationRow) -> bool:
        rows = tuple(
            session.scalars(
                select(ConversationMessageRow)
                .where(
                    ConversationMessageRow.tenant_id == conversation.tenant_id,
                    ConversationMessageRow.conversation_id == conversation.conversation_id,
                    ConversationMessageRow.role == ConversationMessageRole.USER.value,
                    ConversationMessageRow.content.is_not(None),
                    ConversationMessageRow.redacted_at.is_(None),
                    ConversationMessageRow.content_purged_at.is_(None),
                    ConversationMessageRow.sequence > conversation.summary_through_sequence,
                )
                .order_by(ConversationMessageRow.sequence)
            )
        )
        token_count = sum(_estimate_tokens(row.content) for row in rows if row.content)
        if token_count <= 3_000 or len(rows) <= 6:
            return False
        older = rows[:-6]
        summary = _merge_memory_summary(
            conversation.summary,
            tuple(row.content for row in older if row.content),
        )
        conversation.summary = summary or None
        conversation.summary_through_sequence = older[-1].sequence
        conversation.memory_version += 1
        conversation.memory_hash = request_hash({"summary": summary}) if summary else None
        conversation.summary_updated_at = datetime.now(UTC)
        conversation.state_version += 1
        return True

    def begin_new_turn(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        content: str,
        requested_mode: ConversationMode,
        scope_document_id: str | None = None,
        idempotency_key: str,
        request_fingerprint: str,
        task_repository: ConversationTaskRepositoryPort,
        route,
    ) -> tuple[ConversationTurnRow, ConversationMessageRow, object | None, bool, object | None]:
        now = datetime.now(UTC)
        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                session = work.session
                conversation = session.scalar(
                    select(ConversationRow)
                    .where(
                        ConversationRow.tenant_id == auth.tenant_id,
                        ConversationRow.user_id == auth.user_id,
                        ConversationRow.conversation_id == conversation_id,
                    )
                    .with_for_update()
                )
                if conversation is None or conversation.status == ConversationStatus.DELETED.value:
                    raise ConversationRepositoryError("conversation_not_found")
                if conversation.status == ConversationStatus.DELETING.value:
                    raise ConversationRepositoryError("conversation_deleting")
                existing = session.scalar(
                    select(ConversationTurnRow).where(
                        ConversationTurnRow.tenant_id == auth.tenant_id,
                        ConversationTurnRow.user_id == auth.user_id,
                        ConversationTurnRow.conversation_id == conversation_id,
                        ConversationTurnRow.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    if existing.request_hash != request_fingerprint:
                        raise ConversationRepositoryError("idempotency_conflict")
                    message = session.scalar(
                        select(ConversationMessageRow).where(
                            ConversationMessageRow.tenant_id == auth.tenant_id,
                            ConversationMessageRow.conversation_id == conversation_id,
                            ConversationMessageRow.turn_id == existing.turn_id,
                            ConversationMessageRow.role == ConversationMessageRole.USER.value,
                        )
                    )
                    assert message is not None
                    return existing, message, None, True, None
                active = session.scalar(
                    select(ConversationTurnRow.turn_id).where(
                        ConversationTurnRow.tenant_id == auth.tenant_id,
                        ConversationTurnRow.conversation_id == conversation_id,
                        ConversationTurnRow.status.in_(tuple(item.value for item in ACTIVE_TURN_STATUSES)),
                    )
                )
                if active is not None:
                    raise ConversationRepositoryError("conversation_has_active_turn")
                # 幂等键与活动轮次均已在会话锁内确认后才允许调用路由模型。
                try:
                    self._compact_memory_in_session(session, conversation)
                except Exception:
                    # 摘要是可丢失的派生记忆；失败时使用未压缩的近期原话继续。
                    pass
                context_snapshot = self._context_messages_in_session(
                    session, conversation, auth=auth
                )
                decision = route(context_snapshot)
                selected_mode = ConversationMode(decision.selected_mode)
                route_reason_code = str(decision.reason_code)
                resolved_input = str(decision.resolved_input).strip()
                if not resolved_input:
                    raise ConversationRepositoryError("conversation_router_unavailable")
                turn_sequence = int(
                    session.scalar(
                        select(func.coalesce(func.max(ConversationTurnRow.sequence), 0)).where(
                            ConversationTurnRow.tenant_id == auth.tenant_id,
                            ConversationTurnRow.conversation_id == conversation_id,
                        )
                    )
                    or 0
                ) + 1
                message_sequence = int(
                    session.scalar(
                        select(func.coalesce(func.max(ConversationMessageRow.sequence), 0)).where(
                            ConversationMessageRow.tenant_id == auth.tenant_id,
                            ConversationMessageRow.conversation_id == conversation_id,
                        )
                    )
                    or 0
                ) + 1
                turn_id = uuid4().hex
                task_id = uuid4().hex
                status = (
                    ConversationTurnStatus.RUNNING
                    if selected_mode is ConversationMode.ANSWER
                    else ConversationTurnStatus.PLANNING
                )
                turn = ConversationTurnRow(
                    turn_id=turn_id,
                    tenant_id=auth.tenant_id,
                    conversation_id=conversation_id,
                    user_id=auth.user_id,
                    sequence=turn_sequence,
                    status=status.value,
                    submission_kind=ConversationSubmissionKind.NEW.value,
                    requested_mode=requested_mode.value,
                    selected_mode=selected_mode.value,
                    route_reason_code=route_reason_code,
                    resolved_input=resolved_input,
                    resolved_input_hash=request_hash({"resolved_input": resolved_input}),
                    scope_document_id=scope_document_id,
                    task_id=task_id,
                    task_state_version=1,
                    context_hash=context_hash(context_snapshot),
                    context_snapshot=list(context_snapshot),
                    idempotency_key=idempotency_key,
                    request_hash=request_fingerprint,
                    created_at=now,
                    updated_at=now,
                )
                message = ConversationMessageRow(
                    message_id=uuid4().hex,
                    tenant_id=auth.tenant_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    sequence=message_sequence,
                    role=ConversationMessageRole.USER.value,
                    kind="input",
                    content=content,
                    content_hash=request_hash({"content": content}),
                    idempotency_key=idempotency_key,
                    request_hash=request_fingerprint,
                    task_id=task_id,
                    evidence_ids=[],
                    artifacts=[],
                    created_at=now,
                )
                # 没有 ORM relationship 时显式按 FK 顺序落盘，避免 SQLite/PostgreSQL
                # 在同一 flush 中先写 message。
                session.add(turn)
                session.flush()
                session.add(message)
                task = None
                if selected_mode is not ConversationMode.ANSWER:
                    task = task_repository.create_task_in_session(
                        session,
                        auth=auth,
                        goal=resolved_input,
                        task_mode=(
                            TaskMode.RESEARCH
                            if selected_mode is ConversationMode.RESEARCH
                            else TaskMode.CONTROLLED_ACTION
                        ),
                        task_id=task_id,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        scope_document_id=scope_document_id,
                    )
                    turn.task_state_version = task.state_version
                if conversation.title is None:
                    conversation.title = " ".join(content.split())[:80]
                conversation.state_version += 1
                conversation.updated_at = now
                session.flush()
                return turn, message, task, False, decision
        except IntegrityError as exc:
            raise ConversationRepositoryError("conversation_has_active_turn") from exc

    def settle_answer(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        turn_id: str,
        task_id: str | None,
        task_lease_owner: str | None,
        task_lease_generation: int | None,
        content: str,
        status: ConversationTurnStatus,
        evidence_ids: tuple[str, ...] = (),
    ) -> tuple[ConversationMessageRow, int, str | None, int | None, tuple[object, ...]]:
        if (task_lease_owner is None) != (task_lease_generation is None):
            raise ValueError("conversation_task_fence_invalid")
        now = datetime.now(UTC)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            session = work.session
            # 任务取消、租约回收和最终结算统一按 task -> turn -> conversation 加锁。
            task = None
            if task_id is not None:
                task = session.scalar(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.tenant_id == auth.tenant_id,
                        AgentTaskRow.task_id == task_id,
                        AgentTaskRow.user_id == auth.user_id,
                    )
                    .with_for_update()
                )
            turn = session.scalar(
                select(ConversationTurnRow)
                .where(
                    ConversationTurnRow.tenant_id == auth.tenant_id,
                    ConversationTurnRow.user_id == auth.user_id,
                    ConversationTurnRow.conversation_id == conversation_id,
                    ConversationTurnRow.turn_id == turn_id,
                )
                .with_for_update()
            )
            if turn is None:
                raise ConversationRepositoryError("conversation_not_found")
            if (
                turn.selected_mode != ConversationMode.ANSWER.value
                or turn.status != ConversationTurnStatus.RUNNING.value
                or turn.task_id != task_id
            ):
                raise ConversationRepositoryError("conversation_answer_stale")
            final_status = status
            final_content = content
            final_evidence_ids = evidence_ids
            citations: tuple[object, ...] = ()
            if task is None:
                # 没有授权候选时不建立 agent task，也不保留一个不可读取的 task_id。
                if task_lease_owner is not None or final_status is ConversationTurnStatus.SUCCEEDED:
                    raise ConversationRepositoryError("conversation_answer_stale")
                final_evidence_ids = ()
                turn.task_id = None
                turn.task_state_version = None
                session.execute(
                    update(ConversationMessageRow)
                    .where(
                        ConversationMessageRow.tenant_id == auth.tenant_id,
                        ConversationMessageRow.conversation_id == conversation_id,
                        ConversationMessageRow.turn_id == turn_id,
                    )
                    .values(task_id=None)
                )
            else:
                lease_expires_at = task.lease_expires_at
                if lease_expires_at is not None and lease_expires_at.tzinfo is None:
                    lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
                if (
                    task.task_mode != TaskMode.KNOWLEDGE_QUERY.value
                    or task.conversation_id != conversation_id
                    or task.conversation_turn_id != turn_id
                    or task.cancelled
                    or task.status != "running"
                    or task_lease_owner is None
                    or task.lease_owner != task_lease_owner
                    or task.lease_generation != task_lease_generation
                    or lease_expires_at is None
                    or lease_expires_at <= now
                    or turn.task_state_version != task.state_version
                ):
                    raise ConversationRepositoryError("conversation_answer_stale")
                if final_status is ConversationTurnStatus.SUCCEEDED:
                    try:
                        citations = self._evidence.citations_from_ledger(
                            session,
                            task_id=task.task_id,
                            evidence_ids=evidence_ids,
                            auth=auth,
                        )
                        if not citations:
                            raise ValueError("empty_citations")
                    except Exception:
                        # 最终事务中的 ACL/版本复核失败时绝不保存模型回答。
                        final_status = ConversationTurnStatus.REFUSED
                        final_content = "当前资料权限或版本已变化，无法可靠回答该问题"
                        final_evidence_ids = ()
                        citations = ()
                        task.last_error_code = "citation_revalidation_failed"
                task.status = final_status.value
                task.lease_owner = None
                task.lease_expires_at = None
                task.state_version += 1
                task.updated_at = now
            existing = session.scalar(
                select(ConversationMessageRow.message_id).where(
                    ConversationMessageRow.tenant_id == auth.tenant_id,
                    ConversationMessageRow.conversation_id == conversation_id,
                    ConversationMessageRow.turn_id == turn_id,
                    ConversationMessageRow.role == ConversationMessageRole.ASSISTANT.value,
                    ConversationMessageRow.kind.in_(("answer", "refusal")),
                )
            )
            if existing is not None:
                raise ConversationRepositoryError("conversation_answer_stale")
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(ConversationMessageRow.sequence), 0)).where(
                        ConversationMessageRow.tenant_id == auth.tenant_id,
                        ConversationMessageRow.conversation_id == conversation_id,
                    )
                )
                or 0
            ) + 1
            message = ConversationMessageRow(
                message_id=uuid4().hex,
                tenant_id=auth.tenant_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                sequence=sequence,
                role=ConversationMessageRole.ASSISTANT.value,
                kind=("answer" if final_status is ConversationTurnStatus.SUCCEEDED else "refusal"),
                content=final_content,
                content_hash=request_hash({"content": final_content}),
                task_id=task.task_id if task is not None else None,
                evidence_ids=list(final_evidence_ids),
                artifacts=[],
                created_at=now,
            )
            session.add(message)
            turn.status = final_status.value
            if task is not None:
                turn.task_state_version = task.state_version
            turn.updated_at = now
            conversation = session.scalar(
                select(ConversationRow).where(
                    ConversationRow.tenant_id == auth.tenant_id,
                    ConversationRow.conversation_id == conversation_id,
                )
            )
            assert conversation is not None
            conversation.updated_at = now
            conversation.state_version += 1
            session.flush()
            return (
                message,
                conversation.state_version,
                task.status if task is not None else None,
                task.state_version if task is not None else None,
                citations,
            )

    def fail_fast_answer(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        turn_id: str,
        task_id: str | None,
        task_lease_owner: str | None = None,
        task_lease_generation: int | None = None,
        error_code: str,
    ) -> bool:
        """快速路径异常只结算失败状态，绝不生成半成品助手消息。"""
        if (task_lease_owner is None) != (task_lease_generation is None):
            raise ValueError("conversation_task_fence_invalid")
        now = datetime.now(UTC)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            session = work.session
            task = None
            if task_id is not None:
                task = session.scalar(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.tenant_id == auth.tenant_id,
                        AgentTaskRow.task_id == task_id,
                        AgentTaskRow.user_id == auth.user_id,
                    )
                    .with_for_update()
                )
            turn = session.scalar(
                select(ConversationTurnRow)
                .where(
                    ConversationTurnRow.tenant_id == auth.tenant_id,
                    ConversationTurnRow.user_id == auth.user_id,
                    ConversationTurnRow.conversation_id == conversation_id,
                    ConversationTurnRow.turn_id == turn_id,
                )
                .with_for_update()
            )
            if (
                turn is None
                or turn.selected_mode != ConversationMode.ANSWER.value
                or turn.status != ConversationTurnStatus.RUNNING.value
                or turn.task_id != task_id
            ):
                return False
            if task is None:
                if task_lease_owner is not None:
                    return False
                turn.task_id = None
                turn.task_state_version = None
                session.execute(
                    update(ConversationMessageRow)
                    .where(
                        ConversationMessageRow.tenant_id == auth.tenant_id,
                        ConversationMessageRow.conversation_id == conversation_id,
                        ConversationMessageRow.turn_id == turn_id,
                    )
                    .values(task_id=None)
                )
            else:
                lease_expires_at = task.lease_expires_at
                if lease_expires_at is not None and lease_expires_at.tzinfo is None:
                    lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
                if (
                    task.task_mode != TaskMode.KNOWLEDGE_QUERY.value
                    or task.conversation_id != conversation_id
                    or task.conversation_turn_id != turn_id
                    or task.cancelled
                    or task.status != "running"
                    or task_lease_owner is None
                    or task.lease_owner != task_lease_owner
                    or task.lease_generation != task_lease_generation
                    or lease_expires_at is None
                    or lease_expires_at <= now
                    or turn.task_state_version != task.state_version
                ):
                    return False
                task.status = ConversationTurnStatus.FAILED.value
                task.last_error_code = error_code[:100]
                task.lease_owner = None
                task.lease_expires_at = None
                task.state_version += 1
                task.updated_at = now
            if turn.status == ConversationTurnStatus.RUNNING.value:
                turn.status = ConversationTurnStatus.FAILED.value
                if task is not None:
                    turn.task_state_version = task.state_version
                turn.updated_at = now
                conversation = session.scalar(
                    select(ConversationRow)
                    .where(
                        ConversationRow.tenant_id == auth.tenant_id,
                        ConversationRow.conversation_id == conversation_id,
                    )
                    .with_for_update()
                )
                if conversation is not None:
                    conversation.updated_at = now
                    conversation.state_version += 1
            return True

    def request_delete(self, *, auth: AuthContext, conversation_id: str) -> ConversationRow:
        now = datetime.now(UTC)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            session = work.session
            row = session.scalar(
                select(ConversationRow)
                .where(
                    ConversationRow.tenant_id == auth.tenant_id,
                    ConversationRow.user_id == auth.user_id,
                    ConversationRow.conversation_id == conversation_id,
                )
                .with_for_update()
            )
            if row is None or row.status == ConversationStatus.DELETED.value:
                raise ConversationRepositoryError("conversation_not_found")
            active = session.scalar(
                select(ConversationTurnRow.turn_id).where(
                    ConversationTurnRow.tenant_id == auth.tenant_id,
                    ConversationTurnRow.conversation_id == conversation_id,
                    ConversationTurnRow.status.in_(tuple(item.value for item in ACTIVE_TURN_STATUSES)),
                )
            )
            if active is not None:
                raise ConversationRepositoryError("conversation_has_active_turn")
            if row.status != ConversationStatus.DELETING.value:
                row.status = ConversationStatus.DELETING.value
                row.deleting_at = now
                row.updated_at = now
                row.state_version += 1
                session.flush()
            return row

    def active_turn(self, *, auth: AuthContext, conversation_id: str) -> ConversationTurnRow | None:
        self.get_owned(auth=auth, conversation_id=conversation_id)
        with self._database.sessions() as session:
            return session.scalar(
                select(ConversationTurnRow).where(
                    ConversationTurnRow.tenant_id == auth.tenant_id,
                    ConversationTurnRow.user_id == auth.user_id,
                    ConversationTurnRow.conversation_id == conversation_id,
                    ConversationTurnRow.status.in_(tuple(item.value for item in ACTIVE_TURN_STATUSES)),
                )
            )

    def get_turn(
        self, *, auth: AuthContext, conversation_id: str, turn_id: str
    ) -> ConversationTurnRow:
        self.get_owned(auth=auth, conversation_id=conversation_id)
        with self._database.sessions() as session:
            row = session.scalar(
                select(ConversationTurnRow).where(
                    ConversationTurnRow.tenant_id == auth.tenant_id,
                    ConversationTurnRow.user_id == auth.user_id,
                    ConversationTurnRow.conversation_id == conversation_id,
                    ConversationTurnRow.turn_id == turn_id,
                )
            )
            if row is None:
                raise ConversationRepositoryError("clarification_binding_invalid")
            session.expunge(row)
            return row

    def append_clarification_reply(
        self,
        *,
        auth: AuthContext,
        conversation_id: str,
        turn_id: str,
        task_id: str,
        content: str,
        idempotency_key: str,
        request_fingerprint: str,
        task_state_version: int,
        resume_repository,
    ) -> tuple[ConversationTurnRow, ConversationMessageRow, bool]:
        now = datetime.now(UTC)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            session = work.session
            conversation = session.scalar(
                select(ConversationRow)
                .where(
                    ConversationRow.tenant_id == auth.tenant_id,
                    ConversationRow.user_id == auth.user_id,
                    ConversationRow.conversation_id == conversation_id,
                )
                .with_for_update()
            )
            if conversation is None or conversation.status == ConversationStatus.DELETED.value:
                raise ConversationRepositoryError("conversation_not_found")
            if conversation.status == ConversationStatus.DELETING.value:
                raise ConversationRepositoryError("conversation_deleting")
            turn = session.scalar(
                select(ConversationTurnRow)
                .where(
                    ConversationTurnRow.tenant_id == auth.tenant_id,
                    ConversationTurnRow.user_id == auth.user_id,
                    ConversationTurnRow.conversation_id == conversation_id,
                    ConversationTurnRow.turn_id == turn_id,
                )
                .with_for_update()
            )
            if turn is None:
                raise ConversationRepositoryError("clarification_binding_invalid")
            existing = session.scalar(
                select(ConversationMessageRow).where(
                    ConversationMessageRow.tenant_id == auth.tenant_id,
                    ConversationMessageRow.conversation_id == conversation_id,
                    ConversationMessageRow.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_hash != request_fingerprint:
                    raise ConversationRepositoryError("idempotency_conflict")
                return turn, existing, True
            if (
                turn.task_id != task_id
                or turn.status != ConversationTurnStatus.WAITING_FOR_USER.value
                or turn.task_state_version != task_state_version
            ):
                raise ConversationRepositoryError("clarification_binding_invalid")
            resume_repository.enqueue_user_resume_command_in_session(
                session,
                auth=auth,
                task_id=task_id,
                expected_state_version=task_state_version,
                answer=content,
            )
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(ConversationMessageRow.sequence), 0)).where(
                        ConversationMessageRow.tenant_id == auth.tenant_id,
                        ConversationMessageRow.conversation_id == conversation_id,
                    )
                )
                or 0
            ) + 1
            message = ConversationMessageRow(
                message_id=uuid4().hex,
                tenant_id=auth.tenant_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                sequence=sequence,
                role=ConversationMessageRole.USER.value,
                kind="clarification_reply",
                content=content,
                content_hash=request_fingerprint,
                idempotency_key=idempotency_key,
                request_hash=request_fingerprint,
                task_id=task_id,
                evidence_ids=[],
                artifacts=[],
                created_at=now,
            )
            session.add(message)
            turn.status = ConversationTurnStatus.RUNNING.value
            turn.task_state_version = task_state_version + 1
            turn.updated_at = now
            conversation.updated_at = now
            conversation.state_version += 1
            session.flush()
            return turn, message, False

    def latest_assistant_message(
        self, *, auth: AuthContext, conversation_id: str, turn_id: str
    ) -> ConversationMessageRow | None:
        self.get_owned(auth=auth, conversation_id=conversation_id)
        with self._database.sessions() as session:
            return session.scalar(
                select(ConversationMessageRow)
                .where(
                    ConversationMessageRow.tenant_id == auth.tenant_id,
                    ConversationMessageRow.conversation_id == conversation_id,
                    ConversationMessageRow.turn_id == turn_id,
                    ConversationMessageRow.role == ConversationMessageRole.ASSISTANT.value,
                )
                .order_by(ConversationMessageRow.sequence.desc())
                .limit(1)
            )

    def revalidated_citations(
        self, *, auth: AuthContext, message: ConversationMessageRow
    ) -> tuple[object, ...] | None:
        """返回当前 ACL 下的引用；None 表示整条助手事实必须隐藏。"""
        try:
            with self._database.sessions() as session:
                return self._revalidated_citations_in_session(
                    session, auth=auth, message=message
                )
        except Exception:
            return None

    def _revalidated_citations_in_session(
        self, session, *, auth: AuthContext, message: ConversationMessageRow
    ) -> tuple[object, ...] | None:
        """在调用方事务中复用页面展示的证据可见性规则。"""
        if message.role != ConversationMessageRole.ASSISTANT.value:
            return ()
        evidence_ids = tuple(message.evidence_ids or ())
        if not evidence_ids:
            return () if message.kind == "refusal" else None
        if not message.task_id:
            return None
        try:
            return self._evidence.citations_from_ledger(
                session,
                task_id=message.task_id,
                evidence_ids=evidence_ids,
                auth=auth,
            )
        except Exception:
            return None

    def revalidated_artifacts(
        self, *, auth: AuthContext, message: ConversationMessageRow
    ) -> tuple[dict[str, object], ...]:
        """会话仅展示当前 owner 仍可读取的产物摘要，不复制产物正文。"""
        if (
            message.role != ConversationMessageRole.ASSISTANT.value
            or not message.task_id
            or not auth.has_permission(Permission.TASK_READ)
        ):
            return ()
        artifact_ids = tuple(
            item.get("artifact_id")
            for item in (message.artifacts or ())
            if isinstance(item, dict) and isinstance(item.get("artifact_id"), str)
        )
        if not artifact_ids:
            return ()
        with self._database.sessions() as session:
            owner = session.scalar(
                select(AgentTaskRow.task_id).where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.user_id == auth.user_id,
                    AgentTaskRow.task_id == message.task_id,
                )
            )
            if owner is None:
                return ()
            valid_ids = set(
                session.scalars(
                    select(ArtifactRow.artifact_id).where(
                        ArtifactRow.tenant_id == auth.tenant_id,
                        ArtifactRow.task_id == message.task_id,
                        ArtifactRow.artifact_id.in_(artifact_ids),
                        ArtifactRow.status == "ready",
                        ArtifactRow.content_purged_at.is_(None),
                    )
                )
            )
        return tuple(
            dict(item)
            for item in (message.artifacts or ())
            if isinstance(item, dict) and item.get("artifact_id") in valid_ids
        )

    def purge_deleting(self, *, limit: int = 100) -> int:
        """分批清正文并等待现有 task retention 删除结果与 checkpoint。"""
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
        completed = 0
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            session = work.session
            rows = tuple(
                session.scalars(
                    select(ConversationRow)
                    .where(ConversationRow.status == ConversationStatus.DELETING.value)
                    .order_by(ConversationRow.deleting_at, ConversationRow.conversation_id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            now = datetime.now(UTC)
            for conversation in rows:
                session.execute(
                    update(ConversationMessageRow)
                    .where(
                        ConversationMessageRow.tenant_id == conversation.tenant_id,
                        ConversationMessageRow.conversation_id == conversation.conversation_id,
                        ConversationMessageRow.content.is_not(None),
                    )
                    .values(content=None, content_purged_at=now)
                )
                session.execute(
                    update(ConversationTurnRow)
                    .where(
                        ConversationTurnRow.tenant_id == conversation.tenant_id,
                        ConversationTurnRow.conversation_id == conversation.conversation_id,
                    )
                    .values(
                        resolved_input=None,
                        context_snapshot=[],
                        updated_at=now,
                    )
                )
                task_ids = tuple(
                    session.scalars(
                        select(AgentTaskRow.task_id).where(
                            AgentTaskRow.tenant_id == conversation.tenant_id,
                            AgentTaskRow.conversation_id == conversation.conversation_id,
                        )
                    )
                )
                if task_ids:
                    session.execute(
                        update(AgentTaskRow)
                        .where(AgentTaskRow.task_id.in_(task_ids))
                        .values(goal="[conversation content purged]", content_expires_at=now)
                    )
                    session.execute(
                        update(ToolCallRow)
                        .where(
                            ToolCallRow.task_id.in_(task_ids),
                            ToolCallRow.result_ref.is_not(None),
                        )
                        .values(result_expires_at=now)
                    )
                    session.execute(
                        update(ArtifactRow)
                        .where(
                            ArtifactRow.tenant_id == conversation.tenant_id,
                            ArtifactRow.task_id.in_(task_ids),
                            ArtifactRow.content_purged_at.is_(None),
                        )
                        .values(content_expires_at=now)
                    )
                    # 同步快速问答从不创建 LangGraph checkpoint 或任务结果对象；
                    # 直接标记为空，避免删除状态机等待不存在的内容。
                    session.execute(
                        update(AgentTaskRow)
                        .where(
                            AgentTaskRow.task_id.in_(task_ids),
                            AgentTaskRow.task_mode == TaskMode.KNOWLEDGE_QUERY.value,
                        )
                        .values(
                            checkpoint_purged_at=func.coalesce(
                                AgentTaskRow.checkpoint_purged_at, now
                            ),
                            result_purged_at=func.coalesce(
                                AgentTaskRow.result_purged_at, now
                            ),
                        )
                    )
                conversation.title = None
                conversation.summary = None
                conversation.summary_through_sequence = 0
                conversation.memory_hash = None
                conversation.summary_updated_at = None
                conversation.updated_at = now
                conversation.state_version += 1
                pending = session.scalar(
                    select(func.count())
                    .select_from(AgentTaskRow)
                    .where(
                        AgentTaskRow.tenant_id == conversation.tenant_id,
                        AgentTaskRow.conversation_id == conversation.conversation_id,
                        (
                            AgentTaskRow.checkpoint_purged_at.is_(None)
                            | AgentTaskRow.result_purged_at.is_(None)
                        ),
                    )
                )
                if int(pending or 0) == 0:
                    conversation.status = ConversationStatus.DELETED.value
                    conversation.deleted_at = now
                    completed += 1
        return completed

    def mark_expired_for_deletion(self, *, limit: int = 100) -> int:
        if limit < 1:
            raise ValueError("invalid_sweeper_limit")
        changed = 0
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            session = work.session
            now = datetime.now(UTC)
            rows = tuple(
                session.scalars(
                    select(ConversationRow)
                    .where(
                        ConversationRow.status == ConversationStatus.ACTIVE.value,
                        ConversationRow.content_expires_at.is_not(None),
                        ConversationRow.content_expires_at <= now,
                    )
                    .order_by(ConversationRow.content_expires_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for row in rows:
                active = session.scalar(
                    select(ConversationTurnRow.turn_id).where(
                        ConversationTurnRow.tenant_id == row.tenant_id,
                        ConversationTurnRow.conversation_id == row.conversation_id,
                        ConversationTurnRow.status.in_(
                            tuple(item.value for item in ACTIVE_TURN_STATUSES)
                        ),
                    )
                )
                if active is not None:
                    continue
                row.status = ConversationStatus.DELETING.value
                row.deleting_at = now
                row.updated_at = now
                row.state_version += 1
                changed += 1
        return changed
