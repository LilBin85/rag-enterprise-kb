"""版本化知识问答快速路径。"""

from typing import Annotated

from fastapi import APIRouter, Depends

from rag_kb.application.query_service import KnowledgeQueryResult, KnowledgeQueryService
from rag_kb.domain.identity import AuthContext
from rag_kb.web.dependencies import build_knowledge_query_service, resolve_auth_context
from rag_kb.web.schemas.query import KnowledgeQueryRequest


router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


@router.post("/query", response_model=KnowledgeQueryResult)
def query_knowledge(
    request: KnowledgeQueryRequest,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[KnowledgeQueryService, Depends(build_knowledge_query_service)],
) -> KnowledgeQueryResult:
    return service.query(request.question, auth)
