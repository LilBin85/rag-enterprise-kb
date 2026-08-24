"""租户、身份与访问控制领域模型。

这些模型只表达已经由服务端身份提供器确认的事实。请求体中的角色声明不得直接
构造 :class:`AuthContext`，也不能改变租户、用户组或属性。
"""

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class Permission(StrEnum):
    """首版服务端可授权的稳定权限。"""

    KNOWLEDGE_READ = "knowledge:read"
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_APPROVE = "task:approve"
    ARTIFACT_WRITE = "artifact:write"
    ADMIN = "admin"


class SubjectType(StrEnum):
    USER = "user"
    GROUP = "group"
    ROLE = "role"


class FrozenDomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Tenant(FrozenDomainModel):
    tenant_id: Identifier
    name: Identifier
    active: bool = True


class Principal(FrozenDomainModel):
    tenant_id: Identifier
    user_id: Identifier
    roles: frozenset[Identifier] = Field(default_factory=frozenset)
    groups: frozenset[Identifier] = Field(default_factory=frozenset)
    attributes: dict[str, Any] = Field(default_factory=dict)
    active: bool = True

    @field_validator("attributes")
    @classmethod
    def validate_attribute_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError("身份属性键不得为空")
        return value


class Group(FrozenDomainModel):
    tenant_id: Identifier
    group_id: Identifier
    name: Identifier


class ACLGrant(FrozenDomainModel):
    """一条租户内资源授权；拒绝规则由上层策略显式表达，首版仅允许白名单。"""

    tenant_id: Identifier
    resource_type: Identifier
    resource_id: Identifier
    subject_type: SubjectType
    subject_id: Identifier
    permissions: frozenset[Permission]

    @field_validator("permissions")
    @classmethod
    def require_permissions(cls, value: frozenset[Permission]) -> frozenset[Permission]:
        if not value:
            raise ValueError("ACL 授权至少包含一项权限")
        return value


class AuthContext(FrozenDomainModel):
    """一次请求中不可变、已认证的服务端身份上下文。"""

    tenant_id: Identifier
    user_id: Identifier
    roles: frozenset[Identifier] = Field(default_factory=frozenset)
    groups: frozenset[Identifier] = Field(default_factory=frozenset)
    attributes: dict[str, Any] = Field(default_factory=dict)
    request_id: Identifier

    @classmethod
    def from_principal(cls, principal: Principal, *, request_id: str) -> "AuthContext":
        if not principal.active:
            raise ValueError("身份已停用")
        return cls(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            roles=principal.roles,
            groups=principal.groups,
            attributes=principal.attributes,
            request_id=request_id,
        )

    @property
    def acl_subjects(self) -> frozenset[str]:
        """返回供存储层服务端过滤使用的规范化主体集合。"""
        subjects = {f"user:{self.user_id}"}
        subjects.update(f"group:{group}" for group in self.groups)
        subjects.update(f"role:{role}" for role in self.roles)
        return frozenset(subjects)

    def require_tenant(self, tenant_id: str) -> None:
        if self.tenant_id != tenant_id:
            raise PermissionError("cross_tenant_access_denied")

    def has_permission(self, permission: Permission) -> bool:
        """本地角色和显式权限属性的最小策略；未来身份适配器仍只产出事实。"""
        if "admin" in self.roles:
            return True
        raw_permissions = self.attributes.get("permissions", ())
        if not isinstance(raw_permissions, (list, tuple, set, frozenset)):
            return False
        return str(permission) in {str(value) for value in raw_permissions}
