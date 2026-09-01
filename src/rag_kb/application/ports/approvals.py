"""审批生命周期的稳定应用契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.resume_commands import ApprovalResumeDecision, ResumeCommand


class ApprovalLifecycleError(RuntimeError):
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
    """稳定 idempotency key 只依赖不可变 checkpoint action 事实。"""
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
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    interrupt_id: str
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
    plan_revision: int
    plan_digest: str
    evidence_digest: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ApprovalDetail:
    binding: ApprovalBinding
    summary: str | None


@dataclass(frozen=True)
class ApprovalTransition:
    task_id: str
    created_at: datetime


@dataclass(frozen=True)
class ApprovalRequestResult:
    binding: ApprovalBinding
    created: bool


@dataclass(frozen=True)
class ApprovalMutationResult:
    binding: ApprovalBinding
    transition: ApprovalTransition | None = None
    invalid_code: str | None = None


@dataclass(frozen=True)
class ApprovalResumeResult:
    command: ResumeCommand
    transition: ApprovalTransition | None = None


class ApprovalLifecyclePort(Protocol):
    def ensure_action_binding(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        tool_name: str,
        tool_version: str,
        arguments_hash: str,
        checkpoint_binding: ApprovalCheckpointBinding,
    ) -> ActionBinding: ...

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
        arguments_hash: str,
        expires_at: datetime,
        checkpoint_binding: ApprovalCheckpointBinding | None,
    ) -> ApprovalRequestResult: ...

    def approve(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        expected_task_id: str | None,
        expected_plan_revision: int | None,
        expected_task_state_version: int | None,
    ) -> ApprovalMutationResult: ...

    def decide_and_enqueue(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        expected_task_state_version: int,
        decision: ApprovalResumeDecision,
        reason: str | None,
    ) -> ApprovalResumeResult: ...

    def reject(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
        reason: str | None,
        expected_task_id: str | None,
        expected_task_state_version: int | None,
    ) -> ApprovalMutationResult: ...

    def get(self, *, tenant_id: str, approval_id: str) -> ApprovalBinding: ...

    def get_detail(self, *, tenant_id: str, approval_id: str) -> ApprovalDetail: ...

    def list(
        self,
        *,
        tenant_id: str,
        status: str | None,
        task_id: str | None,
    ) -> tuple[ApprovalBinding, ...]: ...

    def list_details(
        self,
        *,
        tenant_id: str,
        status: str | None,
        task_id: str | None,
    ) -> tuple[ApprovalDetail, ...]: ...

    def load_action_facts(
        self,
        *,
        auth: AuthContext,
        approval_id: str,
    ) -> tuple[ApprovalBinding, tuple[str, ...]]: ...

    def load_approved_for_replay(
        self,
        *,
        auth: AuthContext,
        task_id: str,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        tool_call_id: str,
    ) -> tuple[ApprovalBinding, tuple[str, ...]] | None: ...

    def expire_due(
        self,
        *,
        tenant_id: str | None,
    ) -> tuple[ApprovalTransition, ...]: ...
