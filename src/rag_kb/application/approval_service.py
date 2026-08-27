"""持久化批准协议；批准只对一组不可变执行事实有效。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.evidence import canonical_evidence_snapshot
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.tools import ToolMode
from rag_kb.application.tool_registry import RegisteredTool, ToolRegistry, ToolRegistryError
from rag_kb.infrastructure.database.models import (
    ApprovalRow,
    AgentTaskRow,
    EvidenceRecordRow,
    PlanRevisionRow,
)
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


def checkpoint_binding_fingerprint(
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    interrupt_id: str,
    tool_call_id: str,
) -> str:
    return canonical_fingerprint(
        "approval_checkpoint_binding",
        {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "interrupt_id": interrupt_id,
            "tool_call_id": tool_call_id,
        },
    )


def stable_write_action_key(
    *,
    task_id: str,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    interrupt_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_version: str,
    arguments_hash: str,
    evidence_digest: str,
) -> str:
    """D7 稳定 idempotency key：全部输入是不可变 checkpoint action 事实。

    刻意排除 approval ID、run ID、worker ID 与 lease generation——这些标识
    在重复 decision、Worker 更替或租约续代时都会变化，掺入即破坏
    “重复 resume 命中同一 tool_calls 行”的 exactly-once 收敛。
    """
    return canonical_fingerprint(
        "agent_write_action",
        {
            "task_id": task_id,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "interrupt_id": interrupt_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_version": tool_version,
            "arguments_hash": arguments_hash,
            "evidence_digest": evidence_digest,
        },
    )


@dataclass(frozen=True)
class ApprovalCheckpointBinding:
    """审批与 LangGraph checkpoint action 的服务端绑定事实（批次 1 新增）。

    同一 thread/checkpoint/interrupt/tool_call 的重复同步必须命中同一条
    active approval（uq_approvals_active_checkpoint_binding 兜底）。
    """

    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    interrupt_id: str
    # LangGraph checkpoint action ID，不是 tool_calls 外键。
    tool_call_id: str

    @property
    def checkpoint_binding_hash(self) -> str:
        return checkpoint_binding_fingerprint(
            thread_id=self.thread_id,
            checkpoint_ns=self.checkpoint_ns,
            checkpoint_id=self.checkpoint_id,
            interrupt_id=self.interrupt_id,
            tool_call_id=self.tool_call_id,
        )


@dataclass(frozen=True)
class ActionBinding:
    """服务端从 Evidence Ledger 派生的最小不可变 action binding（PlanRevision 事实）。"""

    plan_revision: int
    plan_digest: str
    evidence_digest: str
    evidence_ids: tuple[str, ...]


class ApprovalService:
    def __init__(
        self,
        database: Database,
        *,
        evidence_validator: DatabaseTaskEvidencePort | None = None,
        registry: ToolRegistry | None = None,
        now=None,
    ) -> None:
        self._database = database
        self._evidence_validator = evidence_validator or DatabaseTaskEvidencePort(database)
        # 信任边界：tool/version 只从注册表白名单派生，拒绝任意工具或伪造版本。
        self._registry = registry or ToolRegistry()
        self._now = now or (lambda: datetime.now(UTC))

    def _require_writable_tool(self, tool_name: str) -> RegisteredTool:
        try:
            registered = self._registry.get(tool_name)
        except ToolRegistryError as exc:
            raise ApprovalServiceError("tool_not_registered") from exc
        if registered.definition.mode is not ToolMode.WRITE:
            raise ApprovalServiceError("approval_tool_mode_invalid")
        return registered

    @staticmethod
    def _require_valid_checkpoint_binding(
        checkpoint_binding: ApprovalCheckpointBinding,
    ) -> None:
        if (
            not isinstance(checkpoint_binding.thread_id, str)
            or not checkpoint_binding.thread_id.strip()
            or not isinstance(checkpoint_binding.checkpoint_ns, str)
            or not isinstance(checkpoint_binding.checkpoint_id, str)
            or not checkpoint_binding.checkpoint_id.strip()
            or not isinstance(checkpoint_binding.interrupt_id, str)
            or not checkpoint_binding.interrupt_id.strip()
            or not isinstance(checkpoint_binding.tool_call_id, str)
            or not checkpoint_binding.tool_call_id.strip()
        ):
            raise ApprovalServiceError("approval_checkpoint_binding_invalid")

    def ensure_action_binding(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        tool_name: str,
        arguments: dict[str, object],
        checkpoint_binding: ApprovalCheckpointBinding,
    ) -> ActionBinding:
        """从当前任务 Evidence Ledger 生成最小、不可变的 action binding（PlanRevision）。

        plan 只保存 hash 引用，不保存原始参数；evidence IDs、digest、principal、
        arguments hash 与 checkpoint binding hash 全部由服务端计算。同一 action
        重复同步按 (plan_digest, evidence_digest) 幂等复用同一 revision。
        """
        registered = self._require_writable_tool(tool_name)
        self._require_valid_checkpoint_binding(checkpoint_binding)
        arguments_hash = arguments_fingerprint(arguments)
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
                    raise ApprovalServiceError("approval_plan_binding_invalid")
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
                    "tool_version": registered.definition.version,
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
        auth: AuthContext,
        task_id: str,
        plan_revision: int,
        plan_digest: str,
        evidence_digest: str,
        evidence_ids: tuple[str, ...],
        tool_name: str,
        tool_version: str | None = None,
        arguments: dict[str, object],
        expires_at: datetime,
        checkpoint_binding: ApprovalCheckpointBinding | None = None,
    ) -> ApprovalBinding:
        registered = self._require_writable_tool(tool_name)
        if tool_version is not None and tool_version != registered.definition.version:
            # 拒绝调用方/模型自报版本；版本唯一事实源是 ToolRegistry。
            raise ApprovalServiceError("approval_tool_version_invalid")
        tool_version = registered.definition.version
        if checkpoint_binding is not None:
            self._require_valid_checkpoint_binding(checkpoint_binding)
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
            existing = None
            if checkpoint_binding is not None:
                # 同一 checkpoint action 的重复同步必须复用同一条 active approval；
                # 绑定事实与十列执行绑定任一不符即冲突，fail-closed。
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
                    raise ApprovalServiceError("approval_checkpoint_conflict")
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
                    # 同一执行绑定的 legacy active approval：补齐 checkpoint 绑定后复用。
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
                    thread_id=(
                        checkpoint_binding.thread_id if checkpoint_binding else None
                    ),
                    checkpoint_ns=(
                        checkpoint_binding.checkpoint_ns if checkpoint_binding else None
                    ),
                    checkpoint_id=(
                        checkpoint_binding.checkpoint_id if checkpoint_binding else None
                    ),
                    interrupt_id=(
                        checkpoint_binding.interrupt_id if checkpoint_binding else None
                    ),
                    tool_call_id=(
                        checkpoint_binding.tool_call_id if checkpoint_binding else None
                    ),
                    checkpoint_binding_hash=(
                        checkpoint_binding.checkpoint_binding_hash
                        if checkpoint_binding
                        else None
                    ),
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
            if auth.user_id == task.user_id:
                # 开放问题 3（已拍板）：发起人不能自批。判定以 agent_tasks.user_id
                # （任务发起人）为准，与 decision 主体（approved_by）同一即拒绝；
                # approval 保持 pending，合格审批者仍可批准。
                raise ApprovalServiceError("approval_self_rejected")
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now:
                raise ApprovalServiceError("approval_expired")
            if row.status in ("approved", "executed"):
                # 幂等 decision：相同 decision 返回当前状态，不产生第二次副作用；
                # 重复 resume 复用同一 approval 与 action key（D6 窗口表第 2 行）。
                return self._binding(row)
            if row.status == "expired":
                raise ApprovalServiceError("approval_expired")
            if row.status == "rejected":
                raise ApprovalServiceError("approval_decision_conflict")
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
                # 审批时发现绑定失效：审批者是触发拒绝的真实主体，
                # 审计三元组必须完整（checkpoint 绑定行受 CHECK 约束）。
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
            binding = self._binding(row)
        if invalid_code is not None:
            raise ApprovalServiceError(invalid_code)
        return binding

    def reject(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        reason: str | None = None,
    ) -> ApprovalBinding:
        """业务拒绝：写完整 reject 审计三元组；相同 decision 幂等，相反冲突。"""
        if not auth.has_permission(Permission.TASK_APPROVE):
            raise ApprovalServiceError("approver_permission_denied")
        self.expire_due(tenant_id=auth.tenant_id)
        now = self._now()
        with UnitOfWork(self._database) as work:
            assert work.session is not None
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
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now or row.status == "expired":
                raise ApprovalServiceError("approval_expired")
            if row.status == "rejected":
                return self._binding(row)
            if row.status in ("approved", "executed"):
                raise ApprovalServiceError("approval_decision_conflict")
            if row.status != "pending":
                raise ApprovalServiceError("approval_not_pending")
            row.status = "rejected"
            row.rejected_by = auth.user_id
            row.rejected_at = now
            reason_text = (reason or "rejected_by_approver").strip()[:200]
            row.rejection_reason = reason_text or "rejected_by_approver"
            work.session.flush()
            return self._binding(row)

    def get(self, *, tenant_id: str, approval_id: str) -> ApprovalBinding:
        with self._database.sessions() as session:
            row = session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if row is None:
                raise ApprovalServiceError("approval_not_found")
            return self._binding(row)

    def list(
        self,
        *,
        tenant_id: str,
        status: str | None = "pending",
        task_id: str | None = None,
    ) -> tuple[ApprovalBinding, ...]:
        statement = select(ApprovalRow).where(
            ApprovalRow.tenant_id == tenant_id,
            ApprovalRow.status == status,
        )
        if task_id is not None:
            statement = statement.where(ApprovalRow.task_id == task_id)
        statement = statement.order_by(ApprovalRow.created_at, ApprovalRow.approval_id)
        with self._database.sessions() as session:
            return tuple(
                self._binding(row) for row in session.scalars(statement)
            )

    def load_action_facts(
        self, *, auth: AuthContext, approval_id: str
    ) -> tuple[ApprovalBinding, tuple[str, ...]]:
        """恢复路径读取审批事实：binding + 服务端 plan 派生的 evidence_ids。"""
        with self._database.sessions() as session:
            row = session.scalar(
                select(ApprovalRow).where(
                    ApprovalRow.tenant_id == auth.tenant_id,
                    ApprovalRow.approval_id == approval_id,
                )
            )
            if row is None:
                raise ApprovalServiceError("approval_not_found")
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
                raise ApprovalServiceError("approval_plan_binding_invalid")
            raw_ids = plan.plan.get("evidence_ids") if isinstance(plan.plan, dict) else None
            evidence_ids = tuple(
                str(evidence_id) for evidence_id in raw_ids or () if str(evidence_id)
            )
            return self._binding(row), evidence_ids

    def load_approved_for_replay(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        tool_call_id: str,
    ) -> tuple[ApprovalBinding, tuple[str, ...]] | None:
        """崩溃重放路径：按 checkpoint 绑定反查已批准的审批事实。

        resume 已消费、业务表 interrupt_ref 已清空时，重放的 tool call
        仍必须携带同一审批事实与 D7 稳定 key。checkpoint_id 不参与匹配：
        LangGraph 的 resume 写入会产生 checkpoint 副本导致 ID 漂移，而
        tool_call_id 是全局唯一的 LangGraph action ID，绑定强度足够。
        查不到返回 None（最终门依 approval_required fail-closed，不猜测）。
        """
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
