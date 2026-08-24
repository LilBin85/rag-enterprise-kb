from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_kb.domain.identity import ACLGrant, AuthContext, Permission, Principal, SubjectType
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider


def test_demo_provider_builds_server_owned_auth_context() -> None:
    provider = DemoTokenIdentityProvider.from_role_tokens({"engineer": "secret-token"})

    context = provider.authenticate("secret-token", request_id="request-1")

    assert context == AuthContext(
        tenant_id="local-demo",
        user_id="demo-engineer",
        roles=frozenset({"engineer"}),
        attributes={
            "permissions": [
                "knowledge:read",
                "document:read",
                "task:create",
                "task:read",
                "artifact:write",
            ]
        },
        request_id="request-1",
    )
    assert context.acl_subjects == frozenset({"user:demo-engineer", "role:engineer"})


def test_demo_provider_rejects_unknown_token() -> None:
    provider = DemoTokenIdentityProvider.from_role_tokens({"engineer": "secret-token"})

    assert provider.authenticate("unknown", request_id="request-1") is None


def test_inactive_principal_cannot_create_auth_context() -> None:
    principal = Principal(tenant_id="tenant-a", user_id="user-a", active=False)

    with pytest.raises(ValueError, match="身份已停用"):
        AuthContext.from_principal(principal, request_id="request-1")


def test_auth_context_fails_closed_across_tenants() -> None:
    context = AuthContext(tenant_id="tenant-a", user_id="user-a", request_id="request-1")

    with pytest.raises(PermissionError, match="cross_tenant_access_denied"):
        context.require_tenant("tenant-b")


def test_acl_grant_requires_at_least_one_permission() -> None:
    with pytest.raises(ValidationError, match="ACL 授权至少包含一项权限"):
        ACLGrant(
            tenant_id="tenant-a",
            resource_type="document",
            resource_id="document-a",
            subject_type=SubjectType.USER,
            subject_id="user-a",
            permissions=frozenset(),
        )

    grant = ACLGrant(
        tenant_id="tenant-a",
        resource_type="document",
        resource_id="document-a",
        subject_type=SubjectType.USER,
        subject_id="user-a",
        permissions=frozenset({Permission.DOCUMENT_READ}),
    )
    assert Permission.DOCUMENT_READ in grant.permissions
