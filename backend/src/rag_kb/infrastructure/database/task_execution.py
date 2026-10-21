"""通用任务 worker 所需的最小数据库读取端口。"""

from __future__ import annotations

import hashlib
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_kb.application.ports.object_store import ObjectStorePort
from rag_kb.application.ports.task_read import ArtifactSummary
from rag_kb.application.ports.task_worker import TaskWorkerPersistenceError
from rag_kb.domain.evidence import EvidenceSnapshot, canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ArtifactRow,
    GroupMembershipRow,
    PlanRevisionRow,
    PrincipalRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.models import Citation


logger = logging.getLogger(__name__)


class TaskExecutionContextError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def current_task_auth(
    session: Session,
    *,
    task: AgentTaskRow,
    auth_context_ref: str,
    request_id: str,
) -> AuthContext:
    """重建当前服务端身份；与冻结摘要不符即视为身份已变更。"""
    principal = session.scalar(
        select(PrincipalRow)
        .where(
            PrincipalRow.tenant_id == task.tenant_id,
            PrincipalRow.user_id == task.user_id,
        )
        .with_for_update()
    )
    if principal is None or not principal.active:
        raise TaskExecutionContextError("task_principal_unavailable")
    groups = tuple(
        session.scalars(
            select(GroupMembershipRow.group_id).where(
                GroupMembershipRow.tenant_id == task.tenant_id,
                GroupMembershipRow.user_id == task.user_id,
            ).order_by(GroupMembershipRow.group_id).with_for_update()
        )
    )
    values = {
        "tenant_id": task.tenant_id,
        "user_id": task.user_id,
        "roles": sorted(str(item) for item in principal.roles),
        "groups": sorted(str(item) for item in groups),
        "attributes": principal.attributes,
    }
    if canonical_fingerprint("auth_context", values) != auth_context_ref:
        raise TaskExecutionContextError("task_auth_context_changed")
    return AuthContext(
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        roles=frozenset(str(item) for item in principal.roles),
        groups=frozenset(str(item) for item in groups),
        attributes=principal.attributes,
        request_id=request_id,
    )


def _plan_evidence_snapshot(
    row: PlanRevisionRow,
    *,
    require_non_empty: bool,
) -> EvidenceSnapshot:
    raw_ids = row.plan.get("evidence_ids")
    if not isinstance(raw_ids, list):
        raise TaskExecutionContextError("task_plan_evidence_ids_missing")
    try:
        snapshot = canonical_evidence_snapshot(
            raw_ids,
            require_non_empty=require_non_empty,
        )
    except ValueError as exc:
        raise TaskExecutionContextError(str(exc)) from exc
    if snapshot.digest != row.evidence_digest:
        raise TaskExecutionContextError("task_plan_evidence_digest_mismatch")
    return snapshot


class DatabaseTaskAuthContextPort:
    """重建当前服务端主体，并强制匹配创建任务时冻结的身份摘要。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def resolve(self, *, task_id: str, auth_context_ref: str, request_id: str) -> AuthContext:
        with self._database.sessions() as session:
            task = session.get(AgentTaskRow, task_id)
            if task is None:
                raise TaskExecutionContextError("task_not_found")
            if task.auth_context_ref != auth_context_ref:
                raise TaskExecutionContextError("task_auth_context_binding_mismatch")
            return current_task_auth(
                session,
                task=task,
                auth_context_ref=auth_context_ref,
                request_id=request_id,
            )


class DatabaseTaskEvidencePort:
    """回查当前主体、证据账本、ACL 与激活投影；数据库始终是授权真相。"""

    def __init__(
        self,
        database: Database,
        evidence_repository: EvidenceLedgerRepository | None = None,
    ) -> None:
        self._database = database
        self._evidence = evidence_repository or EvidenceLedgerRepository()

    def validate_plan_in_session(
        self,
        session: Session,
        *,
        task: AgentTaskRow,
        plan: PlanRevisionRow,
        request_id: str,
        require_non_empty: bool,
    ) -> tuple[AuthContext, EvidenceSnapshot]:
        if plan.tenant_id != task.tenant_id or plan.task_id != task.task_id:
            raise TaskExecutionContextError("task_plan_binding_invalid")
        auth = current_task_auth(
            session,
            task=task,
            auth_context_ref=task.auth_context_ref,
            request_id=request_id,
        )
        snapshot = _plan_evidence_snapshot(
            plan,
            require_non_empty=require_non_empty,
        )
        self.validate_in_session(
            session,
            task=task,
            auth=auth,
            evidence_ids=snapshot.evidence_ids,
            expected_digest=plan.evidence_digest,
            require_non_empty=require_non_empty,
        )
        return auth, snapshot

    def validate_in_session(
        self,
        session: Session,
        *,
        task: AgentTaskRow,
        auth: AuthContext,
        evidence_ids: tuple[str, ...],
        expected_digest: str | None = None,
        require_non_empty: bool = True,
    ) -> EvidenceSnapshot:
        if task.tenant_id != auth.tenant_id or task.user_id != auth.user_id:
            raise TaskExecutionContextError("evidence_task_binding_invalid")
        try:
            snapshot = canonical_evidence_snapshot(
                evidence_ids,
                require_non_empty=require_non_empty,
            )
        except ValueError as exc:
            raise TaskExecutionContextError(str(exc)) from exc
        if expected_digest is not None and snapshot.digest != expected_digest:
            raise TaskExecutionContextError("evidence_exact_set_mismatch")
        try:
            self._evidence.citations_from_ledger(
                session,
                task_id=task.task_id,
                evidence_ids=snapshot.evidence_ids,
                auth=auth,
            )
        except EvidenceLedgerConflict as exc:
            raise TaskExecutionContextError(str(exc)) from exc
        return snapshot


class DatabaseTaskWorkerSettlement:
    """收敛 TaskWorker 完成期的数据库和对象存储操作。"""

    def __init__(
        self,
        database: Database,
        objects: ObjectStorePort,
        *,
        evidence_repository: EvidenceLedgerRepository | None = None,
    ) -> None:
        self._database = database
        self._objects = objects
        self._evidence = evidence_repository or EvidenceLedgerRepository()

    def revalidate_citations(
        self,
        *,
        task_id: str,
        evidence_ids: tuple[str, ...],
        auth: AuthContext,
    ) -> tuple[Citation, ...]:
        with self._database.sessions() as session:
            try:
                return self._evidence.citations_from_ledger(
                    session,
                    task_id=task_id,
                    evidence_ids=evidence_ids,
                    auth=auth,
                )
            except EvidenceLedgerConflict as exc:
                raise TaskWorkerPersistenceError(
                    "task_evidence_revalidation_failed"
                ) from exc

    def tool_call_count(self, *, tenant_id: str, task_id: str) -> int:
        with self._database.sessions() as session:
            count = session.scalar(
                select(func.count())
                .select_from(ToolCallRow)
                .where(
                    ToolCallRow.tenant_id == tenant_id,
                    ToolCallRow.task_id == task_id,
                )
            )
        return int(count or 0)

    def verify_artifacts(
        self,
        *,
        tenant_id: str,
        task_id: str,
        artifact_ids: tuple[str, ...],
    ) -> tuple[ArtifactSummary, ...]:
        """校验候选声明的每个产物，不把对象路径泄漏给应用层。"""
        with self._database.sessions() as session:
            rows = {
                row.artifact_id: row
                for row in session.scalars(
                    select(ArtifactRow).where(
                        ArtifactRow.tenant_id == tenant_id,
                        ArtifactRow.task_id == task_id,
                        ArtifactRow.artifact_id.in_(artifact_ids),
                    )
                )
            }
        summaries: list[ArtifactSummary] = []
        for artifact_id in artifact_ids:
            row = rows.get(artifact_id)
            if row is None or row.status != "ready":
                raise TaskWorkerPersistenceError("task_artifact_binding_invalid")
            try:
                content = self._objects.read(row.storage_key)
            except (OSError, ValueError):
                raise TaskWorkerPersistenceError("task_artifact_binding_invalid") from None
            digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
            if digest != row.content_hash:
                raise TaskWorkerPersistenceError("task_artifact_binding_invalid")
            summaries.append(
                ArtifactSummary(
                    artifact_id=row.artifact_id,
                    artifact_type=row.artifact_type,
                    display_name=row.display_name or "",
                    file_extension=(
                        "md" if (row.storage_key or "").lower().endswith(".md") else "json"
                    ),
                    content_hash=row.content_hash,
                    status="ready",
                )
            )
        return tuple(summaries)

    def is_task_result_bound(self, *, task_id: str, result_ref: str) -> bool:
        try:
            with self._database.sessions.begin() as session:
                row = session.scalar(
                    select(AgentTaskRow)
                    .where(AgentTaskRow.task_id == task_id)
                    .with_for_update()
                )
                return row is not None and row.result_ref == result_ref
        except Exception:
            # 提交确认丢失时不能猜测未绑定；保留对象交给孤儿扫描收敛。
            logger.error("任务结果绑定状态确认失败 task_id=%s", task_id)
            return True
