"""服务端拥有的 Agent 运行上下文；由 worker 构造并经 context_schema 注入。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode


class AgentRunContext(BaseModel):
    """模型与工具均不可通过请求或工具参数覆盖任何控制字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    task_mode: TaskMode
    worker_id: str = Field(min_length=1, max_length=200)
    lease_generation: int = Field(ge=0)
    run_id: str = Field(min_length=1, max_length=64)
    auth: AuthContext
    # M2 受控写工具的服务端授权事实；批次 1 恒为默认 None（无审批即无执行），
    # 批次 2 的受控 resume 才由 Worker 填充。
    approval_id: str | None = Field(default=None, max_length=64)
    plan_revision: int | None = Field(default=None, ge=1)
    plan_digest: str | None = Field(default=None, max_length=80)
    evidence_digest: str | None = Field(default=None, max_length=80)
    idempotency_key: str | None = Field(default=None, max_length=200)
