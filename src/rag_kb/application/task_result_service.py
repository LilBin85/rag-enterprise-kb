"""任务结果读取用例。"""

from rag_kb.application.ports.task_read import (
    TaskResult,
    TaskResultError,
    TaskResultReaderPort,
)
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskState


class TaskResultService:
    def __init__(
        self,
        reader: TaskResultReaderPort,
    ) -> None:
        self._reader = reader

    def read(self, *, auth: AuthContext, state: TaskState) -> TaskResult:
        return self._reader.read(auth=auth, state=state)
