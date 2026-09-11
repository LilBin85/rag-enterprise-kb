"""可插拔身份提供器；首版只实现测试和本地演示令牌。"""

import secrets
from collections.abc import Mapping
from typing import Protocol

from rag_kb.domain.identity import AuthContext, Principal


_DEMO_USER_PERMISSIONS = (
    "knowledge:read",
    "document:read",
    "task:create",
    "task:read",
    "artifact:write",
)
_DEMO_APPROVER_PERMISSIONS = ("task:approve",)


def demo_token_principals(
    role_tokens: Mapping[str, str], *, tenant_id: str = "local-demo"
) -> dict[str, Principal]:
    """构造本地演示令牌与主体的唯一映射，供初始化入口复用。"""
    return {
        token: Principal(
            tenant_id=tenant_id,
            user_id=f"demo-{role}",
            roles=frozenset({role}),
            attributes={
                "permissions": list(
                    _DEMO_APPROVER_PERMISSIONS
                    if role == "approver"
                    else _DEMO_USER_PERMISSIONS
                )
            },
        )
        for role, token in role_tokens.items()
        if token
    }


class IdentityProvider(Protocol):
    def authenticate(self, token: str, *, request_id: str) -> AuthContext | None: ...


class DemoTokenIdentityProvider:
    """把本地演示令牌映射为服务端身份，不信任请求体角色。"""

    def __init__(self, token_to_principal: Mapping[str, Principal]) -> None:
        self._entries = tuple(token_to_principal.items())

    @classmethod
    def from_role_tokens(
        cls,
        role_tokens: Mapping[str, str],
        *,
        tenant_id: str = "local-demo",
    ) -> "DemoTokenIdentityProvider":
        return cls(demo_token_principals(role_tokens, tenant_id=tenant_id))

    def authenticate(self, token: str, *, request_id: str) -> AuthContext | None:
        matched: Principal | None = None
        for expected, principal in self._entries:
            if secrets.compare_digest(token, expected):
                matched = principal
        return AuthContext.from_principal(matched, request_id=request_id) if matched else None
