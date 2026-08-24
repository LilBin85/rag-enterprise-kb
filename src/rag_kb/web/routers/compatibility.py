"""旧 HTTP 契约薄适配；不再导入旧检索器、规划器或专用 SQLite store。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from rag_kb.application.approval_service import ApprovalService, ApprovalServiceError
from rag_kb.application.policy_comparison_compat import CompatibilityPolicyComparisonService
from rag_kb.application.query_service import KnowledgeQueryService
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.domain.tasks import TaskMode, TaskState, TaskStatus
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository, TaskRepositoryError
from rag_kb.models import (
    PlanStatus,
    PolicyComparisonRequest,
    PolicyComparisonResponse,
    PolicyRolloutApprovalRequest,
    PolicyRolloutApprovalResponse,
    PolicyRolloutClarification,
    PolicyRolloutExecutionRequest,
    PolicyRolloutTaskContinueRequest,
    PolicyRolloutTaskCreateRequest,
    PolicyRolloutTaskResponse,
    QuestionResponse,
    Refusal,
    Role,
)
from rag_kb.web.dependencies import (
    build_approval_service,
    build_compatibility_policy_comparison_service,
    build_knowledge_query_service,
    build_task_repository,
    resolve_auth_context,
)


router = APIRouter(tags=["deprecated-compatibility"])


class CompatibilityQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=8_000)
    role: Role


def _deprecated(response: Response, successor: str) -> None:
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Sat, 22 Aug 2027 00:00:00 GMT"
    response.headers["Link"] = f'<{successor}>; rel="successor-version"'


def _role(auth: AuthContext, requested: Role | None = None) -> Role:
    values = [Role(value) for value in auth.roles if value in {item.value for item in Role}]
    if requested is not None and requested not in values and "admin" not in auth.roles:
        raise HTTPException(status_code=403, detail="身份验证失败")
    if requested is not None:
        return requested
    if not values:
        raise HTTPException(status_code=403, detail="身份验证失败")
    return values[0]


@router.post("/questions", response_model=QuestionResponse)
def questions(
    request: CompatibilityQuestionRequest,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[KnowledgeQueryService, Depends(build_knowledge_query_service)],
) -> QuestionResponse:
    _deprecated(response, "/v1/knowledge/query")
    role = _role(auth, request.role)
    result = service.query(request.question, auth)
    return QuestionResponse(
        verdict=result.verdict,
        answer=result.answer,
        refusal=result.refusal,
        citations=list(result.citations),
        role=role,
    )


@router.post("/policy-comparisons", response_model=PolicyComparisonResponse)
def policy_comparisons(
    request: PolicyComparisonRequest,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[
        CompatibilityPolicyComparisonService,
        Depends(build_compatibility_policy_comparison_service),
    ],
) -> PolicyComparisonResponse:
    _deprecated(response, "/v1/tasks")
    role = _role(auth, request.role)
    return service.compare(
        auth=auth,
        document_id=request.document_id,
        from_version=request.from_version,
        to_version=request.to_version,
        role=role,
    )


def _legacy_task(state: TaskState, role: Role) -> PolicyRolloutTaskResponse:
    statuses = {
        TaskStatus.WAITING_FOR_USER: PlanStatus.CLARIFICATION,
        TaskStatus.PLANNING: PlanStatus.PLANNING,
        TaskStatus.RUNNING: PlanStatus.PLANNING,
        TaskStatus.WAITING_FOR_APPROVAL: PlanStatus.PLANNED,
        TaskStatus.SUCCEEDED: PlanStatus.PLANNED,
        TaskStatus.REFUSED: PlanStatus.REFUSED,
        TaskStatus.FAILED: PlanStatus.FAILED,
        TaskStatus.CANCELLED: PlanStatus.FAILED,
        TaskStatus.UNKNOWN: PlanStatus.FAILED,
    }
    clarification = (
        PolicyRolloutClarification(questions=[state.pending_question], answers=[])
        if state.pending_question
        else None
    )
    return PolicyRolloutTaskResponse(
        task_id=state.task_id,
        role=role,
        status=statuses[state.status],
        goal=state.goal,
        plan_versions=[],
        clarification=clarification,
        run_id=state.task_id,
    )


@router.post("/policy-rollout-tasks", response_model=PolicyRolloutTaskResponse, status_code=202)
def create_rollout(
    request: PolicyRolloutTaskCreateRequest,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> PolicyRolloutTaskResponse:
    _deprecated(response, "/v1/tasks")
    role = _role(auth, request.role)
    if not auth.has_permission(Permission.TASK_CREATE):
        raise HTTPException(status_code=403, detail="任务权限不足")
    state = repository.create_task(
        auth=auth, goal=request.goal, task_mode=TaskMode.CONTROLLED_ACTION
    )
    return _legacy_task(state, role)


@router.get("/policy-rollout-tasks/{task_id}", response_model=PolicyRolloutTaskResponse)
def get_rollout(
    task_id: str,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> PolicyRolloutTaskResponse:
    _deprecated(response, f"/v1/tasks/{task_id}")
    try:
        return _legacy_task(repository.get_task(auth=auth, task_id=task_id), _role(auth))
    except TaskRepositoryError as exc:
        raise HTTPException(status_code=404, detail="任务不可访问") from exc


@router.post("/policy-rollout-tasks/{task_id}/continue", response_model=PolicyRolloutTaskResponse)
def continue_rollout(
    task_id: str,
    request: PolicyRolloutTaskContinueRequest,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    repository: Annotated[DatabaseTaskRepository, Depends(build_task_repository)],
) -> PolicyRolloutTaskResponse:
    _deprecated(response, f"/v1/tasks/{task_id}/continue")
    try:
        current = repository.get_task(auth=auth, task_id=task_id)
        answer_ref = canonical_fingerprint("user_answers", request.clarification_answers)
        state = repository.resume_user(
            auth=auth,
            task_id=task_id,
            expected_state_version=current.state_version,
            answer_ref=answer_ref,
        )
        return _legacy_task(state, _role(auth))
    except TaskRepositoryError as exc:
        raise HTTPException(status_code=409, detail="任务不可访问或状态冲突") from exc


@router.post(
    "/policy-rollout-tasks/{task_id}/versions/{version}/approval",
    response_model=PolicyRolloutApprovalResponse,
)
def approve_rollout(
    task_id: str,
    version: int,
    request: PolicyRolloutApprovalRequest,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[ApprovalService, Depends(build_approval_service)],
) -> PolicyRolloutApprovalResponse:
    _deprecated(response, f"/v1/tasks/{task_id}/plans/{version}/approval")
    if request.version != version:
        raise HTTPException(status_code=409, detail="版本不匹配")
    try:
        binding = service.approve_for_task_revision(
            auth=auth, task_id=task_id, plan_revision=version
        )
    except ApprovalServiceError as exc:
        code = 403 if exc.code == "approver_permission_denied" else 409
        raise HTTPException(status_code=code, detail="批准状态冲突") from exc
    return PolicyRolloutApprovalResponse(
        task_id=task_id,
        version=version,
        role=_role(auth),
        digest=binding.plan_digest,
        approved_at=binding.approved_at,
    )


@router.post("/policy-rollout-tasks/{task_id}/versions/{version}/executions")
def execute_rollout_compatibility(
    task_id: str,
    version: int,
    request: PolicyRolloutExecutionRequest,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
) -> None:
    _deprecated(response, f"/v1/tasks/{task_id}")
    _role(auth)
    raise HTTPException(status_code=409, detail="批准动作由新版任务 worker 异步执行")


@router.get("/policy-rollout-tasks/{task_id}/executions")
def get_rollout_execution_compatibility(
    task_id: str,
    response: Response,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
) -> None:
    _deprecated(response, f"/v1/tasks/{task_id}/artifacts")
    _role(auth)
    raise HTTPException(status_code=409, detail="请从新版任务产物接口读取执行结果")
