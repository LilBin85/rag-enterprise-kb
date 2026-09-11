"""M1 Research Agent 任务接口。"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from rag_kb.application.ports.task_read import TaskResult
from rag_kb.application.task_query_service import TaskQueryError, TaskQueryService
from rag_kb.application.task_result_service import TaskResultError, TaskResultService
from rag_kb.application.artifact_content_service import (
    ArtifactContentError,
    ArtifactContentService,
)
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.tasks import TaskState
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError
from rag_kb.web.dependencies import (
    build_task_query_service,
    build_task_repository,
    build_task_result_service,
    build_artifact_content_service,
    resolve_auth_context,
)
from rag_kb.web.schemas.tasks import (
    TaskCancelRequest,
    TaskEventResponse,
    TaskResponse,
)

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


def _require(auth: AuthContext, permission: Permission) -> None:
    if not auth.has_permission(permission):
        raise HTTPException(status_code=403, detail="任务权限不足")


def _task_response(*, state: TaskState, now: datetime) -> TaskResponse:
    return TaskResponse(
        task_id=state.task_id,
        conversation_id=getattr(state, "conversation_id", None),
        turn_id=getattr(state, "turn_id", None),
        goal=state.goal,
        task_mode=state.task_mode,
        status=state.status,
        state_version=state.state_version,
        interrupt_kind=state.interrupt_kind,
        interrupt_summary=state.interrupt_summary,
        result_available=state.result_available(now=now),
        cancelled=state.cancelled,
    )


def _repository_error(exc: TaskRepositoryError) -> HTTPException:
    if exc.code == "task_not_found":
        return HTTPException(status_code=404, detail="任务不存在")
    if exc.code in {"research_agent_unavailable", "task_agent_unavailable"}:
        return HTTPException(status_code=503, detail="任务 Agent 暂不可用")
    if exc.code in {
        "task_state_conflict",
        "task_not_waiting_for_user",
        "task_resume_kind_invalid",
        "task_terminal_state",
        "task_deadline_expired",
        "task_mode_not_supported",
    }:
        return HTTPException(status_code=409, detail="任务状态冲突")
    return HTTPException(status_code=409, detail="任务操作冲突")


@router.get("/{task_id}/artifacts/{artifact_id}/content")
def get_artifact_content(
    task_id: str,
    artifact_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    artifacts: Annotated[ArtifactContentService, Depends(build_artifact_content_service)],
) -> Response:
    try:
        return Response(
            content=artifacts.read(
                auth=auth, task_id=task_id, artifact_id=artifact_id
            )
        )
    except ArtifactContentError as exc:
        if exc.code == "task_read_denied":
            raise HTTPException(status_code=403, detail="任务权限不足") from None
        if exc.code == "artifact_not_found":
            raise HTTPException(status_code=404, detail="产物不存在") from None
        if exc.code == "artifact_content_expired":
            raise HTTPException(status_code=410, detail="产物正文已过期") from None
        if exc.code == "artifact_content_unavailable":
            raise HTTPException(status_code=409, detail="产物正文不可用") from None
        raise HTTPException(status_code=500, detail="产物正文校验失败") from None


@router.get("/{task_id}/result", response_model=TaskResult)
def get_result(
    task_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
    results: Annotated[TaskResultService, Depends(build_task_result_service)],
) -> TaskResult:
    _require(auth, Permission.TASK_READ)
    try:
        state = repository.get_task(auth=auth, task_id=task_id)
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc
    try:
        return results.read(auth=auth, state=state)
    except TaskResultError as exc:
        if exc.code == "task_result_expired":
            raise HTTPException(status_code=410, detail="任务结果已过期") from None
        if exc.code in {"task_result_integrity_failed", "task_result_binding_invalid"}:
            raise HTTPException(status_code=500, detail="任务结果校验失败") from None
        raise HTTPException(status_code=409, detail="任务结果不可用") from None


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
            state=repository.cancel(
                auth=auth,
                task_id=task_id,
                expected_state_version=request.expected_state_version,
            ),
            now=repository.current_time(),
        )
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc


@router.get("/{task_id}/events", response_model=tuple[TaskEventResponse, ...])
def get_events(
    task_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[TaskQueryService, Depends(build_task_query_service)],
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


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> TaskResponse:
    """通用动态任务路由必须最后注册，避免吞掉 result/events 等固定后缀。"""
    _require(auth, Permission.TASK_READ)
    try:
        return _task_response(
            state=repository.get_task(auth=auth, task_id=task_id),
            now=repository.current_time(),
        )
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc
