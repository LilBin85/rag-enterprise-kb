"""通用任务 worker 所需的最小数据库读取端口。"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_kb.domain.evidence import EvidenceSnapshot, canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.infrastructure.database.evidence_repository import (
    EvidenceLedgerConflict,
    EvidenceLedgerRepository,
)
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    GroupMembershipRow,
    PlanRevisionRow,
    PrincipalRow,
)
from rag_kb.infrastructure.database.session import Database


class TaskExecutionContextError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TaskPlanBinding:
    plan_revision: int
    plan_digest: str
    evidence_digest: str
    evidence_ids: tuple[str, ...]


def _current_task_auth(
    session: Session,
    *,
    task: AgentTaskRow,
    auth_context_ref: str,
    request_id: str,
) -> AuthContext:
    principal = session.get(
        PrincipalRow,
        {"tenant_id": task.tenant_id, "user_id": task.user_id},
    )
    if principal is None or not principal.active:
        raise TaskExecutionContextError("task_principal_unavailable")
    groups = tuple(
        session.scalars(
            select(GroupMembershipRow.group_id).where(
                GroupMembershipRow.tenant_id == task.tenant_id,
                GroupMembershipRow.user_id == task.user_id,
            )
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
            return _current_task_auth(
                session,
                task=task,
                auth_context_ref=auth_context_ref,
                request_id=request_id,
            )


class DatabaseTaskPlanBindingPort:
    def __init__(self, database: Database) -> None:
        self._database = database

    def load(self, *, auth: AuthContext, task_id: str, revision: int) -> TaskPlanBinding:
        with self._database.sessions() as session:
            row = session.scalar(
                select(PlanRevisionRow).where(
                    PlanRevisionRow.tenant_id == auth.tenant_id,
                    PlanRevisionRow.task_id == task_id,
                    PlanRevisionRow.revision == revision,
                )
            )
            if row is None:
                raise TaskExecutionContextError("task_plan_binding_missing")
            snapshot = _plan_evidence_snapshot(row, require_non_empty=False)
            return TaskPlanBinding(
                plan_revision=row.revision,
                plan_digest=row.plan_digest,
                evidence_digest=row.evidence_digest,
                evidence_ids=snapshot.evidence_ids,
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

    def validate_task(
        self,
        *,
        task_id: str,
        auth_context_ref: str,
        evidence_ids: tuple[str, ...],
        expected_digest: str | None = None,
        request_id: str = "task-evidence-revalidation",
        require_non_empty: bool = True,
    ) -> EvidenceSnapshot:
        with self._database.sessions() as session:
            task = session.get(AgentTaskRow, task_id)
            if task is None:
                raise TaskExecutionContextError("task_not_found")
            if task.auth_context_ref != auth_context_ref:
                raise TaskExecutionContextError("task_auth_context_binding_mismatch")
            auth = _current_task_auth(
                session,
                task=task,
                auth_context_ref=auth_context_ref,
                request_id=request_id,
            )
            return self.validate_in_session(
                session,
                task=task,
                auth=auth,
                evidence_ids=evidence_ids,
                expected_digest=expected_digest,
                require_non_empty=require_non_empty,
            )

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
        auth = _current_task_auth(
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
