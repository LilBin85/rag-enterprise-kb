"""任务事件读取必须保持权限和所有者隐藏语义。"""

from pathlib import Path

import pytest

from rag_kb.application.task_query_service import TaskQueryError, TaskQueryService
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import PrincipalRow, TenantRow
from rag_kb.infrastructure.database.task_read import DatabaseTaskQueryReader
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository


def _auth(user_id: str, *, permissions: list[str] | None = None) -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id=user_id,
        attributes={
            "permissions": permissions if permissions is not None else ["task:create", "task:read"]
        },
        request_id=f"request-{user_id}",
    )


@pytest.fixture()
def service(tmp_path: Path) -> TaskQueryService:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'task-query.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    owner = _auth("owner-a")
    reader = _auth("reader-a")
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="tenant-a"))
        session.flush()
        session.add_all(
            [
                PrincipalRow(
                    tenant_id="tenant-a",
                    user_id=owner.user_id,
                    roles=[],
                    attributes=owner.attributes,
                ),
                PrincipalRow(
                    tenant_id="tenant-a",
                    user_id=reader.user_id,
                    roles=[],
                    attributes=reader.attributes,
                ),
            ]
        )
    DatabaseTaskRepository(database).create_task(
        auth=owner,
        goal="查询事件",
        task_mode=TaskMode.RESEARCH,
        task_id="task-a",
    )
    return TaskQueryService(DatabaseTaskQueryReader(database))


def test_events_keep_permission_and_owner_hidden(service: TaskQueryService) -> None:
    events = service.events(auth=_auth("owner-a"), task_id="task-a")

    assert [event.event_type for event in events] == ["task_created"]
    with pytest.raises(TaskQueryError, match="task_not_found"):
        service.events(auth=_auth("reader-a"), task_id="task-a")
    with pytest.raises(TaskQueryError, match="task_read_denied"):
        service.events(auth=_auth("owner-a", permissions=[]), task_id="task-a")
