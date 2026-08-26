from pathlib import Path

from scripts.init_demo_principals import sync_demo_principals

from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import PrincipalRow, TenantRow
from rag_kb.infrastructure.database.task_execution import DatabaseTaskAuthContextPort
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository


def test_demo_principal_sync_repairs_auth_snapshot_for_task_worker(tmp_path: Path) -> None:
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'demo.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="local-demo", name="旧本地演示", active=False))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id="local-demo",
                user_id="demo-engineer",
                roles=[],
                attributes={},
                active=False,
            )
        )

    synced = sync_demo_principals(database, {"engineer": "demo-token"})
    auth = DemoTokenIdentityProvider.from_role_tokens({"engineer": "demo-token"}).authenticate(
        "demo-token", request_id="request-1"
    )
    assert auth is not None
    task = DatabaseTaskRepository(database).create_task(
        auth=auth,
        goal="验证本地演示身份",
        task_mode=TaskMode.RESEARCH,
        task_id="task-1",
    )

    resolved = DatabaseTaskAuthContextPort(database).resolve(
        task_id=task.task_id,
        auth_context_ref=task.auth_context_ref,
        request_id="worker-1",
    )

    assert synced == ("demo-engineer",)
    assert resolved.roles == auth.roles
    assert resolved.attributes == auth.attributes
