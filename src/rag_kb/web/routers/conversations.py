"""统一会话 HTTP 入口；业务状态均委托给 ConversationService。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from rag_kb.application.conversation_service import ConversationService, ConversationServiceError
from rag_kb.domain.identity import AuthContext
from rag_kb.web.dependencies import build_conversation_service, resolve_auth_context
from rag_kb.web.schemas.conversations import (
    ConversationCreateRequest, ConversationDeleteResponse, ConversationListResponse,
    ConversationMessageRequest, ConversationMessagesResponse, ConversationSummaryResponse,
    ConversationTurnResponse,
)

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

_ERROR_STATUS = {
    "conversation_not_found": 404, "conversation_deleting": 409,
    "conversation_has_active_turn": 409, "idempotency_conflict": 409,
    "clarification_binding_invalid": 409, "task_state_conflict": 409,
    "controlled_action_explicit_mode_required": 409,
    "controlled_action_unavailable": 503, "task_agent_unavailable": 503,
    "conversation_router_unavailable": 503, "conversation_content_unavailable": 410,
}


def _error(exc: ConversationServiceError) -> HTTPException:
    return HTTPException(status_code=_ERROR_STATUS.get(exc.code, 409), detail={"code": exc.code})


def _invalid(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=422, detail={"code": "conversation_request_invalid"}
    )


@router.post("", response_model=ConversationSummaryResponse, status_code=201)
def create(request: ConversationCreateRequest, auth: Annotated[AuthContext, Depends(resolve_auth_context)], service: Annotated[ConversationService, Depends(build_conversation_service)]) -> ConversationSummaryResponse:
    try:
        return ConversationSummaryResponse.model_validate(service.create(auth=auth, title=request.title).model_dump(mode="json"))
    except ConversationServiceError as exc:
        raise _error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.get("", response_model=ConversationListResponse)
def list_conversations(auth: Annotated[AuthContext, Depends(resolve_auth_context)], service: Annotated[ConversationService, Depends(build_conversation_service)], limit: int = Query(20, ge=1, le=100), cursor: str | None = None) -> ConversationListResponse:
    try:
        return ConversationListResponse.model_validate(service.list(auth=auth, limit=limit, cursor=cursor).model_dump(mode="json"))
    except ConversationServiceError as exc:
        raise _error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.get("/{conversation_id}", response_model=ConversationSummaryResponse)
def read(conversation_id: str, auth: Annotated[AuthContext, Depends(resolve_auth_context)], service: Annotated[ConversationService, Depends(build_conversation_service)]) -> ConversationSummaryResponse:
    try:
        return ConversationSummaryResponse.model_validate(service.read(auth=auth, conversation_id=conversation_id).model_dump(mode="json"))
    except ConversationServiceError as exc:
        raise _error(exc) from exc


@router.get("/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def messages(conversation_id: str, auth: Annotated[AuthContext, Depends(resolve_auth_context)], service: Annotated[ConversationService, Depends(build_conversation_service)], limit: int = Query(100, ge=1, le=100), after_sequence: int = Query(0, ge=0)) -> ConversationMessagesResponse:
    try:
        return ConversationMessagesResponse.model_validate(service.messages(auth=auth, conversation_id=conversation_id, limit=limit, after_sequence=after_sequence).model_dump(mode="json"))
    except ConversationServiceError as exc:
        raise _error(exc) from exc


@router.post("/{conversation_id}/messages", response_model=ConversationTurnResponse)
def submit(conversation_id: str, request: ConversationMessageRequest, auth: Annotated[AuthContext, Depends(resolve_auth_context)], service: Annotated[ConversationService, Depends(build_conversation_service)], idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)], response: Response) -> ConversationTurnResponse:
    try:
        data = request.model_dump()
        result = service.submit(auth=auth, conversation_id=conversation_id, idempotency_key=idempotency_key, **data)
        response.status_code = result.status_code
        return ConversationTurnResponse.model_validate(result.model_dump(mode="json"))
    except ConversationServiceError as exc:
        raise _error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.delete("/{conversation_id}", response_model=ConversationDeleteResponse, status_code=202)
def delete(conversation_id: str, auth: Annotated[AuthContext, Depends(resolve_auth_context)], service: Annotated[ConversationService, Depends(build_conversation_service)]) -> ConversationDeleteResponse:
    try:
        return ConversationDeleteResponse.model_validate(service.delete(auth=auth, conversation_id=conversation_id).model_dump(mode="json"))
    except ConversationServiceError as exc:
        raise _error(exc) from exc
