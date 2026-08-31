"""M2 审批发现与服务端受控恢复接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from rag_kb.application.approval_service import (
    ApprovalDetail,
    ApprovalService,
    ApprovalServiceError,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskRepositoryError,
)
from rag_kb.web.dependencies import (
    build_approval_service,
    build_task_repository,
    resolve_auth_context,
)
from rag_kb.web.schemas.approvals import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalResponse,
    ApprovalStatus,
)
from rag_kb.web.schemas.tasks import ResumeCommandResponse


router = APIRouter(prefix="/v1/approvals", tags=["approvals"])


def _require_approver(auth: AuthContext) -> None:
    if not auth.has_permission(Permission.TASK_APPROVE):
        raise HTTPException(status_code=403, detail="审批权限不足")


def _response(detail: ApprovalDetail) -> ApprovalResponse:
    binding = detail.binding
    return ApprovalResponse(
        approval_id=binding.approval_id,
        task_id=binding.task_id,
        summary=detail.summary,
        plan_revision=binding.plan_revision,
        plan_digest=binding.plan_digest,
        evidence_hash=binding.evidence_hash,
        canonical_arguments_hash=binding.canonical_arguments_hash,
        checkpoint_binding_hash=binding.checkpoint_binding_hash,
        tool_name=binding.tool_name,
        tool_version=binding.tool_version,
        expires_at=binding.expires_at,
        status=binding.status or "pending",
        approved_by=binding.approved_by,
        approved_at=binding.approved_at,
        rejected_by=binding.rejected_by,
        rejected_at=binding.rejected_at,
        rejection_reason=binding.rejection_reason,
    )


def _approval_error(exc: ApprovalServiceError) -> HTTPException:
    if exc.code in {"approver_permission_denied", "approval_self_rejected"}:
        return HTTPException(status_code=403, detail="审批权限不足")
    if exc.code == "approval_not_found":
        return HTTPException(status_code=404, detail="审批不存在")
    return HTTPException(status_code=409, detail="审批状态冲突")


def _resume_error(exc: TaskRepositoryError) -> HTTPException:
    if exc.code == "task_not_found":
        return HTTPException(status_code=404, detail="审批不存在")
    return HTTPException(status_code=409, detail="审批状态冲突")


@router.get("", response_model=tuple[ApprovalResponse, ...])
def list_approvals(
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    approvals: Annotated[ApprovalService, Depends(build_approval_service)],
    status: Annotated[ApprovalStatus | None, Query()] = "pending",
    task_id: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
) -> tuple[ApprovalResponse, ...]:
    _require_approver(auth)
    return tuple(
        _response(item)
        for item in approvals.list_details(
            tenant_id=auth.tenant_id, status=status, task_id=task_id
        )
    )


@router.get("/{approval_id}", response_model=ApprovalResponse)
def get_approval(
    approval_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    approvals: Annotated[ApprovalService, Depends(build_approval_service)],
) -> ApprovalResponse:
    _require_approver(auth)
    try:
        return _response(
            approvals.get_detail(tenant_id=auth.tenant_id, approval_id=approval_id)
        )
    except ApprovalServiceError as exc:
        raise _approval_error(exc) from None


@router.post(
    "/{approval_id}/decision",
    response_model=ApprovalDecisionResponse,
    status_code=202,
)
def decide_approval(
    approval_id: str,
    request: ApprovalDecisionRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    approvals: Annotated[ApprovalService, Depends(build_approval_service)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> ApprovalDecisionResponse:
    _require_approver(auth)
    try:
        command = approvals.decide_and_enqueue(
            auth=auth,
            approval_id=approval_id,
            expected_task_state_version=request.expected_task_state_version,
            decision=request.decision,
            reason=request.reason,
            repository=repository,
        )
        detail = approvals.get_detail(tenant_id=auth.tenant_id, approval_id=approval_id)
        return ApprovalDecisionResponse(
            **_response(detail).model_dump(),
            resume_command=ResumeCommandResponse(command_id=command.command_id),
        )
    except ApprovalServiceError as exc:
        raise _approval_error(exc) from None
    except TaskRepositoryError as exc:
        raise _resume_error(exc) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="审批决定请求非法") from exc
