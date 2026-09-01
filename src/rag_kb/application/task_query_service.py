"""任务事件与产物的租户/所有者隔离只读用例。"""

from rag_kb.application.ports.task_read import (
    TaskArtifactResult,
    TaskEventResult,
    TaskQueryError,
    TaskQueryPort,
)
from rag_kb.domain.identity import AuthContext, Permission


class TaskQueryService:
    def __init__(self, reader: TaskQueryPort) -> None:
        self._reader = reader

    def events(self, *, auth: AuthContext, task_id: str) -> tuple[TaskEventResult, ...]:
        self._require_read(auth)
        events = self._reader.events(auth=auth, task_id=task_id)
        if events is None:
            raise TaskQueryError("task_not_found")
        return events

    def artifacts(self, *, auth: AuthContext, task_id: str) -> tuple[TaskArtifactResult, ...]:
        self._require_read(auth)
        artifacts = self._reader.artifacts(auth=auth, task_id=task_id)
        if artifacts is None:
            raise TaskQueryError("task_not_found")
        return artifacts

    @staticmethod
    def _require_read(auth: AuthContext) -> None:
        if not auth.has_permission(Permission.TASK_READ):
            raise TaskQueryError("task_read_denied")
