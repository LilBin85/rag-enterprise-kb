"""持久化批准协议；批准只对一组不可变执行事实有效。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update

from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.infrastructure.database.models import ApprovalRow, AgentTaskRow, PlanRevisionRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskEvidencePort,
    TaskExecutionContextError,
)
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class ApprovalServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def principal_fingerprint(auth: AuthContext) -> str:
    return canonical_fingerprint(
        "principal",
        {
            "tenant_id": auth.tenant_id,
            "user_id": auth.user_id,
            "roles": sorted(auth.roles),
            "groups": sorted(auth.groups),
            "attributes": auth.attributes,
        },
    )


def arguments_fingerprint(arguments: dict[str, object]) -> str:
    return canonical_fingerprint("tool_arguments", arguments)


class ApprovalService:
    def __init__(
        self,
        database: Database,
        *,
        evidence_validator: DatabaseTaskEvidencePort | None = None,
        now=None,
    ) -> None:
        self._database = database
        self._evidence_validator = evidence_validator or DatabaseTaskEvidencePort(database)
        self._now = now or (lambda: datetime.now(UTC))

    def request(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        plan_revision: int,
        plan_digest: str,
        evidence_digest: str,
        evidence_ids: tuple[str, ...],
        tool_name: str,
        tool_version: str,
        arguments: dict[str, object],
        expires_at: datetime,
    ) -> ApprovalBinding:
        self.expire_due()
        now = self._now()
        if expires_at.tzinfo is None or expires_at <= now:
            raise ApprovalServiceError("approval_expiry_invalid")
        arguments_hash = arguments_fingerprint(arguments)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            task = work.session.scalar(
                select(AgentTaskRow).where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.user_id == auth.user_id,
                ).with_for_update()
            )
            plan = work.session.scalar(
                select(PlanRevisionRow).where(
                    PlanRevisionRow.tenant_id == auth.tenant_id,
                    PlanRevisionRow.task_id == task_id,
                    PlanRevisionRow.revision == plan_revision,
                    PlanRevisionRow.plan_digest == plan_digest,
                    PlanRevisionRow.evidence_digest == evidence_digest,
                ).with_for_update()
            )
            if (
                task is None
                or plan is None
                or task.task_mode != "controlled_action"
            ):
                raise ApprovalServiceError("approval_plan_binding_invalid")
            try:
                current_auth, plan_evidence = self._evidence_validator.validate_plan_in_session(
                    work.session,
                    task=task,
                    plan=plan,
                    request_id=auth.request_id,
                    require_non_empty=True,
                )
                requested_evidence = canonical_evidence_snapshot(
                    evidence_ids,
                    require_non_empty=True,
                )
            except (TaskExecutionContextError, ValueError) as exc:
                raise ApprovalServiceError(getattr(exc, "code", str(exc))) from exc
            if (
                requested_evidence != plan_evidence
                or principal_fingerprint(auth) != principal_fingerprint(current_auth)
            ):
                raise ApprovalServiceError("approval_current_binding_invalid")
            principal_hash = principal_fingerprint(current_auth)
            deadline = task.deadline
            if deadline is not None and deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            if task.cancelled or (
                deadline is not None and expires_at > deadline
            ) or expires_at > now + timedelta(minutes=15):
                raise ApprovalServiceError("approval_expiry_invalid")
            existing = work.session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.task_id == task_id,
                    ApprovalRow.plan_revision == plan_revision,
                    ApprovalRow.plan_digest == plan_digest,
                    ApprovalRow.evidence_digest == evidence_digest,
                    ApprovalRow.tool_name == tool_name,
                    ApprovalRow.tool_version == tool_version,
                    ApprovalRow.arguments_hash == arguments_hash,
                    ApprovalRow.principal_hash == principal_hash,
                    ApprovalRow.status.in_(("pending", "approved")),
                )
            )
            if existing is None:
                existing = ApprovalRow(
                    approval_id=uuid4().hex,
                    task_id=task_id,
                    tenant_id=auth.tenant_id,
                    plan_revision=plan_revision,
                    plan_digest=plan_digest,
                    evidence_digest=evidence_digest,
                    tool_name=tool_name,
                    tool_version=tool_version,
                    arguments_hash=arguments_hash,
                    principal_hash=principal_hash,
                    status="pending",
                    expires_at=expires_at,
                    created_at=now,
                )
                work.session.add(existing)
                work.session.flush()
            return self._binding(existing)

    def approve(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        expected_task_id: str | None = None,
        expected_plan_revision: int | None = None,
    ) -> ApprovalBinding:
        if not auth.has_permission(Permission.TASK_APPROVE):
            raise ApprovalServiceError("approver_permission_denied")
        self.expire_due(tenant_id=auth.tenant_id)
        now = self._now()
        invalid_code: str | None = None
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            candidate = work.session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if candidate is None:
                raise ApprovalServiceError("approval_not_found")
            task = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == candidate.task_id,
                )
                .with_for_update()
            )
            row = work.session.scalar(
                select(ApprovalRow)
                .where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
                .with_for_update()
            )
            if row is None:
                raise ApprovalServiceError("approval_not_found")
            if (
                task is None
                or task.cancelled
                or (expected_task_id is not None and row.task_id != expected_task_id)
                or (
                    expected_plan_revision is not None
                    and row.plan_revision != expected_plan_revision
                )
            ):
                raise ApprovalServiceError("approval_plan_binding_invalid")
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now:
                raise ApprovalServiceError("approval_expired")
            if row.status == "expired":
                raise ApprovalServiceError("approval_expired")
            if row.status != "pending":
                raise ApprovalServiceError("approval_not_pending")
            plan = work.session.scalar(
                select(PlanRevisionRow)
                .where(
                    PlanRevisionRow.tenant_id == row.tenant_id,
                    PlanRevisionRow.task_id == row.task_id,
                    PlanRevisionRow.revision == row.plan_revision,
                    PlanRevisionRow.plan_digest == row.plan_digest,
                    PlanRevisionRow.evidence_digest == row.evidence_digest,
                )
                .with_for_update()
            )
            try:
                if task is None or plan is None or task.task_mode != "controlled_action":
                    raise TaskExecutionContextError("approval_current_binding_invalid")
                current_auth, _ = self._evidence_validator.validate_plan_in_session(
                    work.session,
                    task=task,
                    plan=plan,
                    request_id=auth.request_id,
                    require_non_empty=True,
                )
                if principal_fingerprint(current_auth) != row.principal_hash:
                    raise TaskExecutionContextError("approval_principal_changed")
            except TaskExecutionContextError as exc:
                row.status = "rejected"
                invalid_code = exc.code
            if invalid_code is None:
                row.status = "approved"
                row.approved_by = auth.user_id
                row.approved_at = now
            work.session.flush()
            binding = self._binding(row)
        if invalid_code is not None:
            raise ApprovalServiceError(invalid_code)
        return binding

    def approve_for_task_revision(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        plan_revision: int,
    ) -> ApprovalBinding:
        """旧接口只给 task/revision；仅在唯一 pending 批准时安全适配。"""
        self.expire_due(tenant_id=auth.tenant_id)
        with self._database.sessions() as session:
            ids = tuple(
                session.scalars(
                    select(ApprovalRow.approval_id).where(
                        ApprovalRow.tenant_id == auth.tenant_id,
                        ApprovalRow.task_id == task_id,
                        ApprovalRow.plan_revision == plan_revision,
                        ApprovalRow.status == "pending",
                    )
                )
            )
        if len(ids) != 1:
            raise ApprovalServiceError("approval_binding_ambiguous")
        return self.approve(
            auth=auth,
            approval_id=ids[0],
            expected_task_id=task_id,
            expected_plan_revision=plan_revision,
        )

    def validate_in_session(
        self,
        session,
        *,
        auth: AuthContext,
        approval_id: str,
        task_id: str,
        plan_revision: int,
        plan_digest: str,
        evidence_digest: str,
        evidence_ids: tuple[str, ...],
        tool_name: str,
        tool_version: str,
        arguments_hash: str,
    ) -> ApprovalRow:
        row = session.scalar(
            select(ApprovalRow)
            .where(
                ApprovalRow.tenant_id == auth.tenant_id,
                ApprovalRow.approval_id == approval_id,
            )
            .with_for_update()
        )
        if row is None:
            raise ApprovalServiceError("approval_not_found")
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= self._now():
            raise ApprovalServiceError("approval_expired")
        try:
            evidence_snapshot = canonical_evidence_snapshot(
                evidence_ids,
                require_non_empty=True,
            )
        except ValueError as exc:
            raise ApprovalServiceError(str(exc)) from exc
        if evidence_snapshot.digest != evidence_digest:
            raise ApprovalServiceError("approval_evidence_exact_set_invalid")
        expected = (
            task_id,
            plan_revision,
            plan_digest,
            evidence_digest,
            tool_name,
            tool_version,
            arguments_hash,
            principal_fingerprint(auth),
        )
        actual = (
            row.task_id,
            row.plan_revision,
            row.plan_digest,
            row.evidence_digest,
            row.tool_name,
            row.tool_version,
            row.arguments_hash,
            row.principal_hash,
        )
        if actual != expected:
            raise ApprovalServiceError("approval_binding_invalid")
        if row.status != "approved":
            raise ApprovalServiceError("approval_not_approved")
        return row

    def expire_due(self, *, tenant_id: str | None = None) -> int:
        """独立提交过期转换，避免异常回滚令 active 唯一键永久被占用。"""
        statement = update(ApprovalRow).where(
            ApprovalRow.status.in_(("pending", "approved")),
            ApprovalRow.expires_at <= self._now(),
        )
        if tenant_id is not None:
            statement = statement.where(ApprovalRow.tenant_id == tenant_id)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            result = work.session.execute(statement.values(status="expired"))
            return int(result.rowcount or 0)

    @staticmethod
    def _binding(row: ApprovalRow) -> ApprovalBinding:
        expires_at = row.expires_at
        approved_at = row.approved_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if approved_at is not None and approved_at.tzinfo is None:
            approved_at = approved_at.replace(tzinfo=UTC)
        return ApprovalBinding(
            approval_id=row.approval_id,
            task_id=row.task_id,
            tenant_id=row.tenant_id,
            plan_revision=row.plan_revision,
            plan_digest=row.plan_digest,
            tool_name=row.tool_name,
            tool_version=row.tool_version,
            canonical_arguments_hash=row.arguments_hash,
            principal_hash=row.principal_hash,
            evidence_hash=row.evidence_digest,
            expires_at=expires_at,
            approved_by=row.approved_by,
            approved_at=approved_at,
        )
