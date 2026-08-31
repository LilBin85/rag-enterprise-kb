"""A2 migration 的 SQLite upgrade/downgrade 对齐检查。"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from rag_kb.infrastructure.database import build_database


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_resume_command_migration_round_trip_on_sqlite(tmp_path: Path) -> None:
    database_path = tmp_path / "resume-command-migration.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = f"sqlite+pysqlite:///{database_path.as_posix()}"

    command.upgrade(config, "20260829_0014")
    command.upgrade(config, "head")

    database = build_database(config.attributes["database_url"])
    inspector = inspect(database.engine)
    assert "task_resume_commands" in inspector.get_table_names()
    assert {
        "interrupt_checkpoint_ns",
        "interrupt_checkpoint_id",
        "interrupt_id",
        "interrupt_approval_id",
        "active_resume_command_id",
    } <= {column["name"] for column in inspector.get_columns("agent_tasks")}
    assert {
        "command_id",
        "payload_body",
        "payload_sha256",
        "claimed_task_state_version",
        "claimed_task_lease_generation",
        "applied_checkpoint_id",
        "body_purged_at",
    } <= {column["name"] for column in inspector.get_columns("task_resume_commands")}
    assert {
        "ix_task_resume_commands_claim",
        "ix_task_resume_commands_recovery",
    } <= {index["name"] for index in inspector.get_indexes("task_resume_commands")}
    assert {"interrupt_complete", "active_resume_command_complete"} <= {
        item["name"].removeprefix("ck_agent_tasks_")
        for item in inspector.get_check_constraints("agent_tasks")
        if item["name"]
    }
    assert {
        "kind_payload_complete",
        "lease_complete",
        "apply_marker_complete",
        "body_purge_complete",
    } <= {
        item["name"].removeprefix("ck_task_resume_commands_")
        for item in inspector.get_check_constraints("task_resume_commands")
        if item["name"]
    }
    approval_fk = next(
        item
        for item in inspector.get_foreign_keys("task_resume_commands")
        if item["referred_table"] == "approvals"
    )
    assert approval_fk["options"].get("ondelete") == "CASCADE"

    command.downgrade(config, "20260829_0014")

    inspector = inspect(database.engine)
    assert "task_resume_commands" not in inspector.get_table_names()
    assert "active_resume_command_id" not in {
        column["name"] for column in inspector.get_columns("agent_tasks")
    }
    database.engine.dispose()
