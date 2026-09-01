"""审批生命周期的数据库原子操作。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from rag_kb.application.ports.approvals import (
    ActionBinding,
    ApprovalCheckpointBinding,
    ApprovalDetail,
    ApprovalLifecycleError,
    ApprovalLifecyclePort,
    ApprovalMutationResult,
    ApprovalRequestResult,
    ApprovalResumeResult,
    ApprovalTransition,
    principal_fingerprint,
)
from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.resume_commands import ApprovalResumeDecision, ResumeCommandKind
from rag_kb.infrastructure.database.models import (
    AgentTaskRow,
    ApprovalRow,
    EvidenceRecordRow,
    PlanRevisionRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskEvidencePort,
    TaskExecutionContextError,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class DatabaseApprovalLifecycle(ApprovalLifecyclePort):
    def __init__(
        self,
        database: Database,
        *,
        evidence_validator: DatabaseTaskEvidencePort | None = None,
        resume_repository: DatabaseTaskRepository | None = None,
        now=None,
    ) -> None:
        self._database = database
        self._now = now or (lambda: datetime.now(UTC))
        self._evidence_validator = evidence_validator or DatabaseTaskEvidencePort(database)
        self._resume_repository = resume_repository or DatabaseTaskRepository(
            database, now=self._now
        )

    def ensure_action_binding(
        self,
        *,
        auth,
        task_id: str,
        tool_name: str,
        tool_version: str,
        arguments_hash: str,
        checkpoint_binding: ApprovalCheckpointBinding,
    ) -> ActionBinding:
        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                task = work.session.scalar(
                    select(AgentTaskRow)
                    .where(
                        AgentTaskRow.tenant_id == auth.tenant_id,
                        AgentTaskRow.task_id == task_id,
                        AgentTaskRow.user_id == auth.user_id,
                    )
                    .with_for_update()
                )
                if task is None or task.task_mode != "controlled_action":
                    raise ApprovalLifecycleError("approval_plan_binding_invalid")
                evidence_ids = tuple(
                    work.session.scalars(
                        select(EvidenceRecordRow.evidence_id)
                        .where(
                            EvidenceRecordRow.tenant_id == auth.tenant_id,
                            EvidenceRecordRow.task_id == task_id,
                        )
                        .order_by(EvidenceRecordRow.evidence_id)
                    )
                )
                snapshot = canonical_evidence_snapshot(
                    evidence_ids, require_non_empty=True
                )
                plan = {
                    "tool_name": tool_name,
                    "tool_version": tool_version,
                    "arguments_hash": arguments_hash,
                    "evidence_ids": list(snapshot.evidence_ids),
                    "checkpoint_binding_hash": checkpoint_binding.checkpoint_binding_hash,
                }
                plan_digest = canonical_fingerprint("task_plan", plan)
                existing = work.session.scalar(
                    select(PlanRevisionRow).where(
                        PlanRevisionRow.tenant_id == auth.tenant_id,
                        PlanRevisionRow.task_id == task_id,
                        PlanRevisionRow.plan_digest == plan_digest,
                        PlanRevisionRow.evidence_digest == snapshot.digest,
                    )
                )
                if existing is not None:
                    return ActionBinding(
                        plan_revision=existing.revision,
                        plan_digest=plan_digest,
                        evidence_digest=snapshot.digest,
                        evidence_ids=snapshot.evidence_ids,
                    )
                revision = int(
                    work.session.scalar(
                        select(func.max(PlanRevisionRow.revision)).where(
                            PlanRevisionRow.tenant_id == auth.tenant_id,
                            PlanRevisionRow.task_id == task_id,
                        )
                    )
                    or 0
                ) + 1
                work.session.add(
                    PlanRevisionRow(
                        plan_revision_id=uuid4().hex,
                        task_id=task_id,
                        tenant_id=auth.tenant_id,
                        revision=revision,
                        schema_version=1,
                        plan=plan,
                        plan_digest=plan_digest,
                        evidence_digest=snapshot.digest,
                        status="ready",
                    )
                )
                work.session.flush()
                return ActionBinding(
                    plan_revision=revision,
                    plan_digest=plan_digest,
                    evidence_digest=snapshot.digest,
                    evidence_ids=snapshot.evidence_ids,
                )
        except IntegrityError:
            # 并发写入同一不可变 plan：按唯一绑定复读复用。
            with self._database.sessions() as session:
                existing = session.scalar(
                    select(PlanRevisionRow).where(
                        PlanRevisionRow.tenant_id == auth.tenant_id,
                        PlanRevisionRow.task_id == task_id,
                        PlanRevisionRow.plan_digest == plan_digest,
                        PlanRevisionRow.evidence_digest == snapshot.digest,
                    )
                )
                if existing is not None:
                    return ActionBinding(
                        plan_revision=existing.revision,
                        plan_digest=plan_digest,
                        evidence_digest=snapshot.digest,
                        evidence_ids=snapshot.evidence_ids,
                    )
            raise

    def request(
        self,
        *,
        auth,
        task_id: str,
        plan_revision: int,
        plan_digest: str,
        evidence_digest: str,
        evidence_ids: tuple[str, ...],
        tool_name: str,
        tool_version: str,
        arguments_hash: str,
        expires_at: datetime,
        checkpoint_binding: ApprovalCheckpointBinding | None,
    ) -> ApprovalRequestResult:
        now = self._now()
        if expires_at.tzinfo is None or expires_at <= now:
            raise ApprovalLifecycleError("approval_expiry_invalid")
        created = False
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            task = work.session.scalar(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.tenant_id == auth.tenant_id,
                    AgentTaskRow.task_id == task_id,
                    AgentTaskRow.user_id == auth.user_id,
                )
                .with_for_update()
            )
            plan = work.session.scalar(
                select(PlanRevisionRow)
                .where(
                    PlanRevisionRow.tenant_id == auth.tenant_id,
                    PlanRevisionRow.task_id == task_id,
                    PlanRevisionRow.revision == plan_revision,
                    PlanRevisionRow.plan_digest == plan_digest,
                    PlanRevisionRow.evidence_digest == evidence_digest,
                )
                .with_for_update()
            )
            if task is None or plan is None or task.task_mode != "controlled_action":
                raise ApprovalLifecycleError("approval_plan_binding_invalid")
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
                raise ApprovalLifecycleError(getattr(exc, "code", str(exc))) from exc
            if (
                requested_evidence != plan_evidence
                or principal_fingerprint(auth) != principal_fingerprint(current_auth)
            ):
                raise ApprovalLifecycleError("approval_current_binding_invalid")
            principal_hash = principal_fingerprint(current_auth)
            deadline = task.deadline
            if deadline is not None and deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            if task.cancelled or (
                deadline is not None and expires_at > deadline
            ) or expires_at > now + timedelta(minutes=15):
                raise ApprovalLifecycleError("approval_expiry_invalid")
            existing = None
            if checkpoint_binding is not None:
                existing = work.session.scalar(
                    select(ApprovalRow).where(
                        ApprovalRow.tenant_id == auth.tenant_id,
                        ApprovalRow.task_id == task_id,
                        ApprovalRow.checkpoint_binding_hash
                        == checkpoint_binding.checkpoint_binding_hash,
                        ApprovalRow.status.in_(("pending", "approved")),
                    )
                )
                if existing is not None and (
                    existing.plan_revision != plan_revision
                    or existing.plan_digest != plan_digest
                    or existing.evidence_digest != evidence_digest
                    or existing.tool_name != tool_name
                    or existing.tool_version != tool_version
                    or existing.arguments_hash != arguments_hash
                    or existing.principal_hash != principal_hash
                    or existing.thread_id != checkpoint_binding.thread_id
                    or existing.checkpoint_ns != checkpoint_binding.checkpoint_ns
                    or existing.checkpoint_id != checkpoint_binding.checkpoint_id
                    or existing.interrupt_id != checkpoint_binding.interrupt_id
                    or existing.tool_call_id != checkpoint_binding.tool_call_id
                ):
                    raise ApprovalLifecycleError("approval_checkpoint_conflict")
            if existing is None:
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
                        ApprovalRow.checkpoint_binding_hash.is_(None),
                    )
                )
                if existing is not None and checkpoint_binding is not None:
                    existing.thread_id = checkpoint_binding.thread_id
                    existing.checkpoint_ns = checkpoint_binding.checkpoint_ns
                    existing.checkpoint_id = checkpoint_binding.checkpoint_id
                    existing.interrupt_id = checkpoint_binding.interrupt_id
                    existing.tool_call_id = checkpoint_binding.tool_call_id
                    existing.checkpoint_binding_hash = (
                        checkpoint_binding.checkpoint_binding_hash
                    )
                    work.session.flush()
            if existing is None:
                created = True
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
                    thread_id=checkpoint_binding.thread_id if checkpoint_binding else None,
                    checkpoint_ns=checkpoint_binding.checkpoint_ns if checkpoint_binding else None,
                    checkpoint_id=checkpoint_binding.checkpoint_id if checkpoint_binding else None,
                    interrupt_id=checkpoint_binding.interrupt_id if checkpoint_binding else None,
                    tool_call_id=checkpoint_binding.tool_call_id if checkpoint_binding else None,
                    checkpoint_binding_hash=(
                        checkpoint_binding.checkpoint_binding_hash
                        if checkpoint_binding
                        else None
                    ),
                )
                work.session.add(existing)
                work.session.flush()
            return ApprovalRequestResult(self._binding(existing), created)

    def approve(
        self,
        *,
        auth,
        approval_id: str,
        expected_task_id: str | None,
        expected_plan_revision: int | None,
        expected_task_state_version: int | None,
    ) -> ApprovalMutationResult:
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            candidate = work.session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if candidate is None:
                raise ApprovalLifecycleError("approval_not_found")
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
                raise ApprovalLifecycleError("approval_not_found")
            if (
                task is None
                or task.cancelled
                or (expected_task_id is not None and row.task_id != expected_task_id)
                or (
                    expected_plan_revision is not None
                    and row.plan_revision != expected_plan_revision
                )
            ):
                raise ApprovalLifecycleError("approval_plan_binding_invalid")
            if auth.user_id == task.user_id:
                raise ApprovalLifecycleError("approval_self_rejected")
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now:
                raise ApprovalLifecycleError("approval_expired")
            if row.status in ("approved", "executed"):
                return ApprovalMutationResult(self._binding(row))
            if row.status == "expired":
                raise ApprovalLifecycleError("approval_expired")
            if row.status == "rejected":
                raise ApprovalLifecycleError("approval_decision_conflict")
            if row.status != "pending":
                raise ApprovalLifecycleError("approval_not_pending")
            self._require_pending_resume_target(
                task,
                row,
                expected_task_state_version=expected_task_state_version,
            )
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
            invalid_code: str | None = None
            try:
                if task.task_mode != "controlled_action" or plan is None:
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
                row.rejected_by = auth.user_id
                row.rejected_at = now
                row.rejection_reason = f"auto:{exc.code}"[:200]
                invalid_code = exc.code
            if invalid_code is None:
                row.status = "approved"
                row.approved_by = auth.user_id
                row.approved_at = now
            work.session.flush()
            return ApprovalMutationResult(
                self._binding(row),
                ApprovalTransition(task_id=row.task_id, created_at=row.created_at),
                invalid_code,
            )

    def decide_and_enqueue(
        self,
        *,
        auth,
        approval_id: str,
        expected_task_state_version: int,
        decision: ApprovalResumeDecision,
        reason: str | None,
    ) -> ApprovalResumeResult:
        transitioned = False
        transition: ApprovalTransition | None = None
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            # SQLite 的写锁必须先于无锁候选读取；PostgreSQL 候选本身不构成锁。
            self._resume_repository._begin_resume_write(work.session)
            candidate_task_id = work.session.scalar(
                select(ApprovalRow.task_id).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if candidate_task_id is None:
                raise ApprovalLifecycleError("approval_not_found")
            task = self._resume_repository._lock_resume_task(
                work.session,
                tenant_id=auth.tenant_id,
                task_id=candidate_task_id,
                skip_locked=False,
            )
            approval = self._resume_repository._lock_resume_approval(
                work.session,
                tenant_id=auth.tenant_id,
                approval_id=approval_id,
                skip_locked=False,
            )
            if approval is None:
                raise ApprovalLifecycleError("approval_not_found")
            if task is None or task.cancelled or approval.task_id != task.task_id:
                raise ApprovalLifecycleError("approval_plan_binding_invalid")
            if auth.user_id == task.user_id:
                raise ApprovalLifecycleError("approval_self_rejected")
            now = self._resume_repository._lease_now(work.session)
            expires_at = approval.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now or approval.status == "expired":
                raise ApprovalLifecycleError("approval_expired")
            command = self._resume_repository.lock_approval_resume_command_in_session(
                work.session,
                tenant_id=auth.tenant_id,
                task_id=task.task_id,
                approval_id=approval_id,
            )
            same_decision = (
                decision is ApprovalResumeDecision.APPROVED
                and approval.status in {"approved", "executed"}
            ) or (
                decision is ApprovalResumeDecision.REJECTED
                and approval.status == "rejected"
            )
            if command is not None:
                if not same_decision or command.decision is not decision:
                    raise ApprovalLifecycleError("approval_decision_conflict")
                if command.expected_task_state_version != expected_task_state_version:
                    raise ApprovalLifecycleError("approval_resume_state_conflict")
                return ApprovalResumeResult(command)
            if not same_decision and approval.status != "pending":
                if approval.status in {"approved", "executed", "rejected"}:
                    raise ApprovalLifecycleError("approval_decision_conflict")
                raise ApprovalLifecycleError("approval_not_pending")
            self._require_pending_resume_target(
                task,
                approval,
                expected_task_state_version=expected_task_state_version,
            )
            self._validate_decision_binding_in_session(
                work.session,
                task=task,
                approval=approval,
                request_id=auth.request_id,
            )
            if approval.status == "pending":
                if decision is ApprovalResumeDecision.APPROVED:
                    approval.status = "approved"
                    approval.approved_by = auth.user_id
                    approval.approved_at = now
                else:
                    approval.status = "rejected"
                    approval.rejected_by = auth.user_id
                    approval.rejected_at = now
                    reason_text = (reason or "rejected_by_approver").strip()[:200]
                    approval.rejection_reason = reason_text or "rejected_by_approver"
                transitioned = True
                transition = ApprovalTransition(
                    task_id=task.task_id,
                    created_at=approval.created_at,
                )
            command = self._resume_repository.enqueue_resume_command_in_session(
                work.session,
                task=task,
                actor_principal_id=auth.user_id,
                actor_kind="user",
                kind=ResumeCommandKind.TOOL_APPROVAL,
                expected_task_state_version=expected_task_state_version,
                payload_body=reason,
                approval=approval,
                decision=decision,
                now=now,
            )
        return ApprovalResumeResult(command, transition if transitioned else None)

    def _validate_decision_binding_in_session(
        self,
        session,
        *,
        task: AgentTaskRow,
        approval: ApprovalRow,
        request_id: str,
    ) -> None:
        plan = session.scalar(
            select(PlanRevisionRow)
            .where(
                PlanRevisionRow.tenant_id == approval.tenant_id,
                PlanRevisionRow.task_id == approval.task_id,
                PlanRevisionRow.revision == approval.plan_revision,
                PlanRevisionRow.plan_digest == approval.plan_digest,
                PlanRevisionRow.evidence_digest == approval.evidence_digest,
            )
            .with_for_update()
        )
        try:
            if plan is None or task.task_mode != "controlled_action":
                raise TaskExecutionContextError("approval_current_binding_invalid")
            current_auth, _ = self._evidence_validator.validate_plan_in_session(
                session,
                task=task,
                plan=plan,
                request_id=request_id,
                require_non_empty=True,
            )
            if principal_fingerprint(current_auth) != approval.principal_hash:
                raise TaskExecutionContextError("approval_principal_changed")
        except TaskExecutionContextError as exc:
            raise ApprovalLifecycleError(exc.code) from exc

    def reject(
        self,
        *,
        auth,
        approval_id: str,
        reason: str | None,
        expected_task_id: str | None,
        expected_task_state_version: int | None,
    ) -> ApprovalMutationResult:
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            self._resume_repository._begin_resume_write(work.session)
            candidate_task_id = work.session.scalar(
                select(ApprovalRow.task_id).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if candidate_task_id is None:
                raise ApprovalLifecycleError("approval_not_found")
            task = self._resume_repository._lock_resume_task(
                work.session,
                tenant_id=auth.tenant_id,
                task_id=candidate_task_id,
                skip_locked=False,
            )
            row = self._resume_repository._lock_resume_approval(
                work.session,
                tenant_id=auth.tenant_id,
                approval_id=approval_id,
                skip_locked=False,
            )
            if row is None:
                raise ApprovalLifecycleError("approval_not_found")
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now or row.status == "expired":
                raise ApprovalLifecycleError("approval_expired")
            if row.status == "rejected":
                return ApprovalMutationResult(self._binding(row))
            if row.status in ("approved", "executed"):
                raise ApprovalLifecycleError("approval_decision_conflict")
            if row.status != "pending":
                raise ApprovalLifecycleError("approval_not_pending")
            if (
                task is None
                or task.cancelled
                or (expected_task_id is not None and row.task_id != expected_task_id)
            ):
                raise ApprovalLifecycleError("approval_plan_binding_invalid")
            self._require_pending_resume_target(
                task,
                row,
                expected_task_state_version=expected_task_state_version,
            )
            row.status = "rejected"
            row.rejected_by = auth.user_id
            row.rejected_at = now
            reason_text = (reason or "rejected_by_approver").strip()[:200]
            row.rejection_reason = reason_text or "rejected_by_approver"
            work.session.flush()
            return ApprovalMutationResult(
                self._binding(row),
                ApprovalTransition(task_id=row.task_id, created_at=row.created_at),
            )

    def get(self, *, tenant_id: str, approval_id: str) -> ApprovalBinding:
        with self._database.sessions() as session:
            row = session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if row is None:
                raise ApprovalLifecycleError("approval_not_found")
            return self._binding(row)

    def get_detail(self, *, tenant_id: str, approval_id: str) -> ApprovalDetail:
        with self._database.sessions() as session:
            result = session.execute(
                select(ApprovalRow, AgentTaskRow.interrupt_summary)
                .join(
                    AgentTaskRow,
                    (AgentTaskRow.tenant_id == ApprovalRow.tenant_id)
                    & (AgentTaskRow.task_id == ApprovalRow.task_id),
                )
                .where(
                    ApprovalRow.tenant_id == tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            ).one_or_none()
            if result is None:
                raise ApprovalLifecycleError("approval_not_found")
            row, summary = result
            return ApprovalDetail(binding=self._binding(row), summary=summary)

    def list(
        self,
        *,
        tenant_id: str,
        status: str | None,
        task_id: str | None,
    ) -> tuple[ApprovalBinding, ...]:
        statement = select(ApprovalRow).where(ApprovalRow.tenant_id == tenant_id)
        if status is not None:
            statement = statement.where(ApprovalRow.status == status)
        if task_id is not None:
            statement = statement.where(ApprovalRow.task_id == task_id)
        statement = statement.order_by(ApprovalRow.created_at, ApprovalRow.approval_id)
        with self._database.sessions() as session:
            return tuple(self._binding(row) for row in session.scalars(statement))

    def list_details(
        self,
        *,
        tenant_id: str,
        status: str | None,
        task_id: str | None,
    ) -> tuple[ApprovalDetail, ...]:
        statement = (
            select(ApprovalRow, AgentTaskRow.interrupt_summary)
            .join(
                AgentTaskRow,
                (AgentTaskRow.tenant_id == ApprovalRow.tenant_id)
                & (AgentTaskRow.task_id == ApprovalRow.task_id),
            )
            .where(ApprovalRow.tenant_id == tenant_id)
        )
        if status is not None:
            statement = statement.where(ApprovalRow.status == status)
        if task_id is not None:
            statement = statement.where(ApprovalRow.task_id == task_id)
        statement = statement.order_by(ApprovalRow.created_at, ApprovalRow.approval_id)
        with self._database.sessions() as session:
            return tuple(
                ApprovalDetail(binding=self._binding(row), summary=summary)
                for row, summary in session.execute(statement)
            )

    def load_action_facts(
        self,
        *,
        auth,
        approval_id: str,
    ) -> tuple[ApprovalBinding, tuple[str, ...]]:
        with self._database.sessions() as session:
            row = session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if row is None:
                raise ApprovalLifecycleError("approval_not_found")
            plan = session.scalar(
                select(PlanRevisionRow).where(
                    PlanRevisionRow.tenant_id == row.tenant_id,
                    PlanRevisionRow.task_id == row.task_id,
                    PlanRevisionRow.revision == row.plan_revision,
                    PlanRevisionRow.plan_digest == row.plan_digest,
                    PlanRevisionRow.evidence_digest == row.evidence_digest,
                )
            )
            if plan is None:
                raise ApprovalLifecycleError("approval_plan_binding_invalid")
            raw_ids = plan.plan.get("evidence_ids") if isinstance(plan.plan, dict) else None
            evidence_ids = tuple(
                str(evidence_id) for evidence_id in raw_ids or () if str(evidence_id)
            )
            return self._binding(row), evidence_ids

    def load_approved_for_replay(
        self,
        *,
        auth,
        task_id: str,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        tool_call_id: str,
    ) -> tuple[ApprovalBinding, tuple[str, ...]] | None:
        # checkpoint_id 可因 LangGraph resume 产生副本；tool_call_id 是稳定 action ID。
        del checkpoint_id
        with self._database.sessions() as session:
            row = session.scalar(
                select(ApprovalRow)
                .where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.task_id == task_id,
                    ApprovalRow.status == "approved",
                    ApprovalRow.thread_id == thread_id,
                    ApprovalRow.checkpoint_ns == checkpoint_ns,
                    ApprovalRow.tool_call_id == tool_call_id,
                )
                .order_by(ApprovalRow.created_at.desc(), ApprovalRow.approval_id.desc())
                .limit(1)
            )
            if row is None:
                return None
            plan = session.scalar(
                select(PlanRevisionRow).where(
                    PlanRevisionRow.tenant_id == row.tenant_id,
                    PlanRevisionRow.task_id == row.task_id,
                    PlanRevisionRow.revision == row.plan_revision,
                    PlanRevisionRow.plan_digest == row.plan_digest,
                    PlanRevisionRow.evidence_digest == row.evidence_digest,
                )
            )
            if plan is None:
                return None
            raw_ids = plan.plan.get("evidence_ids") if isinstance(plan.plan, dict) else None
            evidence_ids = tuple(
                str(evidence_id) for evidence_id in raw_ids or () if str(evidence_id)
            )
            return self._binding(row), evidence_ids

    def expire_due(
        self,
        *,
        tenant_id: str | None,
    ) -> tuple[ApprovalTransition, ...]:
        now = self._now()
        statement = select(ApprovalRow).where(
            ApprovalRow.status.in_(("pending", "approved")),
            ApprovalRow.expires_at <= now,
        )
        if tenant_id is not None:
            statement = statement.where(ApprovalRow.tenant_id == tenant_id)
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            rows = tuple(work.session.scalars(statement.with_for_update()))
            for row in rows:
                row.status = "expired"
            return tuple(
                ApprovalTransition(task_id=row.task_id, created_at=row.created_at)
                for row in rows
            )

    @staticmethod
    def _require_pending_resume_target(
        task: AgentTaskRow,
        row: ApprovalRow,
        *,
        expected_task_state_version: int | None,
    ) -> None:
        if expected_task_state_version is None:
            return
        if task.state_version != expected_task_state_version:
            raise ApprovalLifecycleError("approval_resume_state_conflict")
        if (
            task.status != "waiting_for_approval"
            or task.interrupt_kind != "tool_approval"
            or task.interrupt_ref != row.approval_id
        ):
            raise ApprovalLifecycleError("approval_resume_kind_invalid")

    @staticmethod
    def _binding(row: ApprovalRow) -> ApprovalBinding:
        expires_at = row.expires_at
        approved_at = row.approved_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if approved_at is not None and approved_at.tzinfo is None:
            approved_at = approved_at.replace(tzinfo=UTC)
        rejected_at = row.rejected_at
        if rejected_at is not None and rejected_at.tzinfo is None:
            rejected_at = rejected_at.replace(tzinfo=UTC)
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
            status=row.status,
            thread_id=row.thread_id,
            checkpoint_ns=row.checkpoint_ns,
            checkpoint_id=row.checkpoint_id,
            interrupt_id=row.interrupt_id,
            tool_call_id=row.tool_call_id,
            checkpoint_binding_hash=row.checkpoint_binding_hash,
            rejected_by=row.rejected_by,
            rejected_at=rejected_at,
            rejection_reason=row.rejection_reason,
        )


def consume_approved_execution_in_session(
    session: object,
    *,
    auth,
    approval_id: str,
    task_id: str,
    plan_revision: int,
    plan_digest: str,
    evidence_digest: str,
    evidence_ids: tuple[str, ...],
    tool_name: str,
    tool_version: str,
    arguments_hash: str,
    now: datetime,
) -> str:
    """与 ToolCall running intent 同事务消费一条已批准审批。"""
    row = _execution_approval_row(
        session,
        auth=auth,
        approval_id=approval_id,
        task_id=task_id,
        plan_revision=plan_revision,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        evidence_ids=evidence_ids,
        tool_name=tool_name,
        tool_version=tool_version,
        arguments_hash=arguments_hash,
        now=now,
    )
    row.status = "executed"
    return row.approval_id


def _execution_approval_row(
    session: object,
    *,
    auth,
    approval_id: str,
    task_id: str,
    plan_revision: int,
    plan_digest: str,
    evidence_digest: str,
    evidence_ids: tuple[str, ...],
    tool_name: str,
    tool_version: str,
    arguments_hash: str,
    now: datetime,
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
        raise ApprovalLifecycleError("approval_not_found")
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= now:
        raise ApprovalLifecycleError("approval_expired")
    try:
        evidence_snapshot = canonical_evidence_snapshot(
            evidence_ids,
            require_non_empty=True,
        )
    except ValueError as exc:
        raise ApprovalLifecycleError(str(exc)) from exc
    if evidence_snapshot.digest != evidence_digest:
        raise ApprovalLifecycleError("approval_evidence_exact_set_invalid")
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
        raise ApprovalLifecycleError("approval_binding_invalid")
    if row.status != "approved":
        raise ApprovalLifecycleError("approval_not_approved")
    return row
