"""受控产物正文读取用例。"""

from rag_kb.application.ports.task_read import (
    ArtifactContentError,
    ArtifactContentPort,
)
from rag_kb.domain.identity import AuthContext, Permission


class ArtifactContentService:
    """只返回已授权且未过期的正文 bytes，不向上层暴露对象引用。"""

    def __init__(self, reader: ArtifactContentPort) -> None:
        self._reader = reader

    def read(self, *, auth: AuthContext, task_id: str, artifact_id: str) -> bytes:
        if not auth.has_permission(Permission.TASK_READ):
            raise ArtifactContentError("task_read_denied")
        return self._reader.read(auth=auth, task_id=task_id, artifact_id=artifact_id)
