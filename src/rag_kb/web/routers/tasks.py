"""新版通用任务接口：仅委托可恢复任务与批准服务。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from rag_kb.application.approval_service import ApprovalService, ApprovalServiceError
from rag_kb.application.task_query_service import DatabaseTaskQueryService, TaskQueryError
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.domain.tasks import TaskState
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError
from rag_kb.web.dependencies import (
    build_approval_service,
    build_task_query_service,
    build_task_repository,
    resolve_auth_context,
)
from rag_kb.web.schemas.tasks import (
    TaskApprovalRequest,
    TaskApprovalResponse,
    TaskArtifactResponse,
    TaskCancelRequest,
    TaskContinueRequest,
    TaskCreateRequest,
    TaskEventResponse,
    TaskResponse,
)


router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


def _require(auth: AuthContext, permission: Permission) -> None:
    if not auth.has_permission(permission):
        raise HTTPException(status_code=403, detail="任务权限不足")


def _task_response(state: TaskState) -> TaskResponse:
    return TaskResponse(
        task_id=state.task_id,
        goal=state.goal,
        task_mode=state.task_mode,
        status=state.status,
        current_node=state.current_node,
        state_version=state.state_version,
        plan_revision=state.plan_revision,
        current_step=state.current_step,
        pending_question=state.pending_question,
        cancelled=state.cancelled,
    )


def _repository_error(exc: TaskRepositoryError) -> HTTPException:
    if exc.code == "task_not_found":
        return HTTPException(status_code=404, detail="任务不存在")
    if exc.code in {
        "task_state_conflict",
        "task_not_waiting_for_user",
        "task_terminal_state",
        "task_deadline_expired",
    }:
        return HTTPException(status_code=409, detail="任务状态冲突")
    return HTTPException(status_code=409, detail="任务操作冲突")


@router.post("", response_model=TaskResponse, status_code=202)
def create_task(
    request: TaskCreateRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> TaskResponse:
    _require(auth, Permission.TASK_CREATE)
    try:
        return _task_response(
            repository.create_task(auth=auth, goal=request.goal, task_mode=request.task_mode)
        )
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="任务请求非法") from exc


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> TaskResponse:
    _require(auth, Permission.TASK_READ)
    try:
        return _task_response(repository.get_task(auth=auth, task_id=task_id))
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc


@router.post("/{task_id}/continue", response_model=TaskResponse)
def continue_task(
    task_id: str,
    request: TaskContinueRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> TaskResponse:
    _require(auth, Permission.TASK_READ)
    answer_ref = canonical_fingerprint("user_answer", request.answer)
    try:
        return _task_response(
            repository.resume_user(
                auth=auth,
                task_id=task_id,
                expected_state_version=request.expected_state_version,
                answer_ref=answer_ref,
            )
        )
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc


@router.post("/{task_id}/cancel", response_model=TaskResponse)
def cancel_task(
    task_id: str,
    request: TaskCancelRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> TaskResponse:
    _require(auth, Permission.TASK_READ)
    try:
        return _task_response(
            repository.cancel(
                auth=auth,
                task_id=task_id,
                expected_state_version=request.expected_state_version,
            )
        )
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc


@router.get("/{task_id}/events", response_model=tuple[TaskEventResponse, ...])
def get_events(
    task_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DatabaseTaskQueryService, Depends(build_task_query_service)],
) -> tuple[TaskEventResponse, ...]:
    try:
        return tuple(
            TaskEventResponse(
                sequence=item.sequence,
                event_type=item.event_type,
                stage=item.stage,
                status=item.status,
                counts=item.counts,
                token_usage=item.token_usage,
                error_code=item.error_code,
                created_at=item.created_at,
            )
            for item in service.events(auth=auth, task_id=task_id)
        )
    except TaskQueryError as exc:
        if exc.code == "task_read_denied":
            raise HTTPException(status_code=403, detail="任务权限不足") from exc
        raise HTTPException(status_code=404, detail="任务不存在") from exc


@router.post("/{task_id}/plans/{revision}/approval", response_model=TaskApprovalResponse)
def approve_plan(
    task_id: str,
    revision: int,
    request: TaskApprovalRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[ApprovalService, Depends(build_approval_service)],
) -> TaskApprovalResponse:
    try:
        binding = service.approve(
            auth=auth,
            approval_id=request.approval_id,
            expected_task_id=task_id,
            expected_plan_revision=revision,
        )
    except ApprovalServiceError as exc:
        if exc.code == "approver_permission_denied":
            raise HTTPException(status_code=403, detail="任务审批权限不足") from exc
        if exc.code == "approval_not_found":
            raise HTTPException(status_code=404, detail="审批不存在") from exc
        raise HTTPException(status_code=409, detail="审批状态冲突") from exc
    return TaskApprovalResponse(
        approval_id=binding.approval_id,
        task_id=binding.task_id,
        plan_revision=binding.plan_revision,
        status="approved",
        expires_at=binding.expires_at,
        approved_at=binding.approved_at,
    )


@router.get("/{task_id}/artifacts", response_model=tuple[TaskArtifactResponse, ...])
def get_artifacts(
    task_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DatabaseTaskQueryService, Depends(build_task_query_service)],
) -> tuple[TaskArtifactResponse, ...]:
    try:
        return tuple(
            TaskArtifactResponse(
                artifact_id=item.artifact_id,
                artifact_type=item.artifact_type,
                status=item.status,
                created_at=item.created_at,
            )
            for item in service.artifacts(auth=auth, task_id=task_id)
        )
    except TaskQueryError as exc:
        if exc.code == "task_read_denied":
            raise HTTPException(status_code=403, detail="任务权限不足") from exc
        raise HTTPException(status_code=404, detail="任务不存在") from exc
