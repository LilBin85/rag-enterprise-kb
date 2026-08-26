"""M1 Research Agent 任务接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from rag_kb.agent.result import TaskResult
from rag_kb.application.task_query_service import DatabaseTaskQueryService, TaskQueryError
from rag_kb.application.task_result_service import TaskResultError, TaskResultService
from rag_kb.application.task_worker import TaskWorker
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.tasks import TaskState
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError
from rag_kb.web.dependencies import (
    build_task_query_service,
    build_task_repository,
    build_task_result_service,
    build_task_resume_worker,
    resolve_auth_context,
)
from rag_kb.web.schemas.tasks import (
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
        state_version=state.state_version,
        interrupt_kind=state.interrupt_kind,
        interrupt_summary=state.interrupt_summary,
        result_available=bool(state.result_ref and state.result_hash),
        cancelled=state.cancelled,
    )


def _repository_error(exc: TaskRepositoryError) -> HTTPException:
    if exc.code == "task_not_found":
        return HTTPException(status_code=404, detail="任务不存在")
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
    except TaskResultError:
        raise HTTPException(status_code=409, detail="任务结果不可用") from None


@router.post("/{task_id}/continue", response_model=TaskResponse)
def continue_task(
    task_id: str,
    request: TaskContinueRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
    worker: Annotated[TaskWorker, Depends(build_task_resume_worker)],
) -> TaskResponse:
    _require(auth, Permission.TASK_READ)
    try:
        worker.resume_for_user(
            auth=auth,
            task_id=task_id,
            expected_state_version=request.expected_state_version,
            answer=request.answer,
        )
        return _task_response(repository.get_task(auth=auth, task_id=task_id))
    except TaskRepositoryError as exc:
        raise _repository_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="任务继续请求非法") from exc


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
