"""受控工具执行持久化的稳定应用契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tools import ToolMode


class ToolExecutionLifecycleError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ToolExecutionRequest:
    auth: AuthContext
    task_id: str
    tool_name: str
    arguments: dict[str, object]
    worker_id: str
    lease_generation: int
    idempotency_key: str | None = None
    approval_id: str | None = None
    # M1 起 READ 工具不携带计划绑定；WRITE（M2）仍要求完整计划三元组。
    plan_revision: int | None = None
    plan_digest: str | None = None
    evidence_digest: str | None = None
    evidence_ids: tuple[str, ...] = ()
    scope_document_id: str | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    status: str
    result_hash: str | None = None
    result_ref: str | None = None
    result: dict[str, object] | None = None
    error_code: str | None = None
    duplicate: bool = False


@dataclass(frozen=True)
class ToolExecutionFacts:
    tool_version: str
    mode: ToolMode
    risk_level: str
    timeout_seconds: int
    arguments_hash: str
    principal_hash: str


@dataclass(frozen=True)
class ToolExecutionReservation:
    tool_call_id: str
    mode: ToolMode
    status: str
    result_hash: str | None = None
    result_ref: str | None = None
    error_code: str | None = None
    duplicate: bool = False
    requires_result_read: bool = False
    historical_principal_hash: str | None = None


@dataclass(frozen=True)
class ToolExecutionRunningCall:
    tool_call_id: str
    mode: ToolMode


@dataclass(frozen=True)
class ToolExecutionCompletion:
    status: str
    result_hash: str | None
    result_ref: str | None
    error_code: str | None
    result_to_discard: str | None = None


class ToolExecutionLifecyclePort(Protocol):
    def current_auth(
        self,
        request: ToolExecutionRequest,
        *,
        principal_hash: str,
    ) -> AuthContext: ...

    def preflight_write(
        self,
        request: ToolExecutionRequest,
        *,
        principal_hash: str,
    ) -> None: ...

    def reserve_or_replay(
        self,
        request: ToolExecutionRequest,
        facts: ToolExecutionFacts,
        *,
        tool_call_id: str,
    ) -> ToolExecutionReservation: ...

    def confirm_replay(
        self,
        request: ToolExecutionRequest,
        reservation: ToolExecutionReservation,
        *,
        principal_hash: str,
    ) -> bool: ...

    def prepare_completion(
        self,
        request: ToolExecutionRequest,
        *,
        tool_call_id: str,
        principal_hash: str,
    ) -> ToolExecutionRunningCall: ...

    def complete(
        self,
        request: ToolExecutionRequest,
        *,
        tool_call_id: str,
        principal_hash: str,
        status: str,
        error_code: str | None,
        result_hash: str | None,
        result_ref: str | None,
        result_size: int | None,
        artifact_id: str | None,
        tool_result_ttl_seconds: int,
    ) -> ToolExecutionCompletion: ...

    def is_result_bound(self, *, tool_call_id: str, result_ref: str) -> bool: ...

    def expire_stale_running(self, *, limit: int) -> int: ...
