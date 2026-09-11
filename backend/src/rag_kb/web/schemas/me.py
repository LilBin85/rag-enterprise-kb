"""当前已认证身份的最小公开契约。"""

from pydantic import BaseModel, ConfigDict


class MeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str
    roles: tuple[str, ...]
    permissions: tuple[str, ...]
