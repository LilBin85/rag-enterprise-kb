"""任务事件、结果与受控产物读取的稳定应用契约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskState
from rag_kb.models import Citation


class TaskQueryError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ArtifactContentError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class TaskResultError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TaskEventResult:
    sequence: int
    event_type: str
    stage: str
    status: str
    counts: dict[str, object]
    token_usage: int
    error_code: str | None
    created_at: datetime


@dataclass(frozen=True)
class TaskArtifactResult:
    artifact_id: str
    artifact_type: str
    content_hash: str
    status: str
    created_at: datetime


class ArtifactSummary(BaseModel):
    """任务结果中的产物摘要；不暴露对象路径。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1, max_length=64)
    artifact_type: str = Field(min_length=1, max_length=40)
    display_name: str = Field(default="", max_length=300)
    file_extension: Literal["md", "json"] = "json"
    content_hash: str = Field(min_length=1, max_length=80)
    status: Literal["ready"] = "ready"


class TaskResult(BaseModel):
    """最终可交付结果；引用只能由 Evidence Ledger 复核构造。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    result_kind: Literal["research", "controlled_action"] = "research"
    status: Literal["succeeded", "refused"]
    answer: str = Field(default="", max_length=20_000)
    refusal_code: str | None = Field(default=None, max_length=100)
    evidence_ids: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    artifacts: tuple[ArtifactSummary, ...] = ()
    stop_reason: str = Field(default="", max_length=200)
    created_at: datetime


class TaskQueryPort(Protocol):
    def events(
        self,
        *,
        auth: AuthContext,
        task_id: str,
    ) -> tuple[TaskEventResult, ...] | None: ...

    def artifacts(
        self,
        *,
        auth: AuthContext,
        task_id: str,
    ) -> tuple[TaskArtifactResult, ...] | None: ...


class ArtifactContentPort(Protocol):
    def read(self, *, auth: AuthContext, task_id: str, artifact_id: str) -> bytes: ...


class TaskResultObjectStorePort(Protocol):
    def read_task_result(self, ref: str, expected_hash: str) -> TaskResult: ...


class TaskResultReaderPort(Protocol):
    def read(self, *, auth: AuthContext, state: TaskState) -> TaskResult: ...
