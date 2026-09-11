"""当前身份读取入口。"""

from typing import Annotated

from fastapi import APIRouter, Depends

from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.web.dependencies import resolve_auth_context
from rag_kb.web.schemas.me import MeResponse


router = APIRouter(prefix="/v1", tags=["identity"])


@router.get("/me", response_model=MeResponse)
def read_me(
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
) -> MeResponse:
    return MeResponse(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        roles=tuple(sorted(auth.roles)),
        permissions=tuple(
            permission.value
            for permission in Permission
            if auth.has_permission(permission)
        ),
    )
