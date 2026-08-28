"""M2 审批发现与服务端受控恢复接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from rag_kb.application.approval_service import (
    ApprovalDetail,
    ApprovalService,
    ApprovalServiceError,
)
from rag_kb.application.task_worker import TaskWorker, TaskWorkerError
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.infrastructure.database.task_repository import TaskRepositoryError
from rag_kb.web.dependencies import (
    build_approval_service,
    build_task_resume_worker,
    resolve_auth_context,
)
from rag_kb.web.schemas.approvals import (
    ApprovalDecisionRequest,
    ApprovalResponse,
    ApprovalStatus,
)


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


def _same_decision(*, status: str | None, decision: str) -> bool:
    return (decision == "approve" and status in {"approved", "executed"}) or (
        decision == "reject" and status == "rejected"
    )


def _worker_error(exc: TaskWorkerError) -> HTTPException:
    if exc.code in {"task_worker_closed", "task_mode_not_supported", "task_protocol_mismatch"}:
        return HTTPException(status_code=503, detail="任务 Agent 暂不可用")
    return HTTPException(status_code=409, detail="审批状态冲突")


def _resume_result_error(result: object) -> HTTPException | None:
    status = getattr(result, "status", None)
    code = getattr(result, "error_code", None)
    if status == "stale":
        return HTTPException(status_code=409, detail="审批状态冲突")
    if status != "failed":
        return None
    if isinstance(code, str) and (
        code.startswith("task_interrupt_")
        or code.startswith("task_approval_")
        or code == "task_checkpoint_binding_invalid"
    ):
        return HTTPException(status_code=409, detail="审批状态冲突")
    return HTTPException(status_code=500, detail="任务恢复失败")


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


@router.post("/{approval_id}/decision", response_model=ApprovalResponse)
def decide_approval(
    approval_id: str,
    request: ApprovalDecisionRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    approvals: Annotated[ApprovalService, Depends(build_approval_service)],
    worker: Annotated[TaskWorker, Depends(build_task_resume_worker)],
) -> ApprovalResponse:
    _require_approver(auth)
    try:
        detail = approvals.get_detail(tenant_id=auth.tenant_id, approval_id=approval_id)
        same_decision = _same_decision(
            status=detail.binding.status, decision=request.decision
        )
        try:
            result = worker.resume_for_approval(
                auth=auth,
                task_id=detail.binding.task_id,
                expected_state_version=request.expected_task_state_version,
                approval_id=detail.binding.approval_id,
                decision=request.decision,
                reason=request.reason,
            )
        except TaskRepositoryError as exc:
            if same_decision and exc.code in {
                "task_state_conflict",
                "task_not_waiting_for_approval",
            }:
                return _response(
                    approvals.get_detail(
                        tenant_id=auth.tenant_id, approval_id=approval_id
                    )
                )
            raise
        result_error = _resume_result_error(result)
        if result_error is not None:
            raise result_error
        return _response(
            approvals.get_detail(tenant_id=auth.tenant_id, approval_id=approval_id)
        )
    except ApprovalServiceError as exc:
        raise _approval_error(exc) from None
    except TaskRepositoryError as exc:
        raise _resume_error(exc) from None
    except TaskWorkerError as exc:
        raise _worker_error(exc) from None
