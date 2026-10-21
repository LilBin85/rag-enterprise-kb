import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import CheckConstraint, Column, Integer, MetaData, Table, inspect, text
from sqlalchemy.exc import IntegrityError

from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.base import NAMING_CONVENTION
from rag_kb.web.dependencies import EXPECTED_DATABASE_REVISION


ROOT_DIR = Path(__file__).resolve().parents[1]
POSTGRES_MIGRATION_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_MIGRATION_URL")
EXPECTED_TABLES = {
    "acl_grants",
    "agent_capability_leases",
    "agent_tasks",
    "approvals",
    "artifacts",
    "chunk_metadata",
    "conversation_messages",
    "conversation_turns",
    "conversations",
    "document_versions",
    "documents",
    "evidence_records",
    "group_memberships",
    "groups",
    "idempotency_records",
    "ingestion_jobs",
    "plan_revisions",
    "principals",
    "task_events",
    "task_resume_commands",
    "tenants",
    "tool_calls",
}


def sqlite_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def test_check_constraint_naming_convention_is_idempotent() -> None:
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    table = Table(
        "widgets",
        metadata,
        Column("value", Integer),
        CheckConstraint("value >= 0", name="value_nonnegative"),
        CheckConstraint("value <= 10", name="ck_widgets_value_bounded"),
    )

    assert {constraint.name for constraint in table.constraints if isinstance(constraint, CheckConstraint)} == {
        "ck_widgets_value_bounded",
        "ck_widgets_value_nonnegative",
    }


def upgrade_sqlite_database(tmp_path: Path):
    database_path = tmp_path / "constraints.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "head")
    return build_database(sqlite_url(database_path))


def _dialect_bool(connection, *, truthy: str, falsy: str) -> str:
    """SQLite 接受整型布尔；PostgreSQL 只接受 TRUE/FALSE 字面量。"""
    return truthy if connection.dialect.name == "postgresql" else falsy


def insert_principal(connection, tenant_id: str, user_id: str) -> None:
    active = _dialect_bool(connection, truthy="TRUE", falsy="1")
    connection.execute(
        text(
            "INSERT INTO tenants (tenant_id, name, active, created_at) "
            f"VALUES (:tenant_id, :name, {active}, CURRENT_TIMESTAMP)"
        ),
        {"tenant_id": tenant_id, "name": tenant_id},
    )
    connection.execute(
        text(
            "INSERT INTO principals (tenant_id, user_id, roles, attributes, active, created_at) "
            f"VALUES (:tenant_id, :user_id, '[]', '{{}}', {active}, CURRENT_TIMESTAMP)"
        ),
        {"tenant_id": tenant_id, "user_id": user_id},
    )


def insert_task(connection, tenant_id: str, task_id: str, user_id: str) -> None:
    """旧 Runtime（0009 及更早）任务行；0011 升级前使用。"""
    connection.execute(
        text(
            "INSERT INTO agent_tasks ("
            "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, "
            "current_plan_revision, current_step, budgets, observation_refs, pending_question, "
            "pending_approval_id, repeated_action_fingerprints, no_new_evidence_count, cancelled, "
            "model_calls, tool_calls, replans, consecutive_failures, token_usage, deadline, "
            "last_error_code, state_version, created_at, updated_at"
            ") VALUES ("
            ":task_id, :tenant_id, :user_id, 'knowledge_query', 'running', 'goal', 'auth-ref', "
            "0, 0, '{}', '[]', NULL, NULL, '[]', 0, 0, 0, 0, 0, 0, 0, NULL, NULL, 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"tenant_id": tenant_id, "task_id": task_id, "user_id": user_id},
    )


def insert_shrunk_task(connection, tenant_id: str, task_id: str, user_id: str) -> None:
    """M1 收缩后的任务行（0011 head schema）。"""
    connection.execute(
        text(
            "INSERT INTO agent_tasks ("
            "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, "
            "budgets, cancelled, state_version, created_at, updated_at"
            ") VALUES ("
            ":task_id, :tenant_id, :user_id, 'knowledge_query', 'running', 'goal', 'auth-ref', "
            "'{}', 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"tenant_id": tenant_id, "task_id": task_id, "user_id": user_id},
    )


def insert_plan_revision(connection) -> None:
    connection.execute(
        text(
            "INSERT INTO plan_revisions ("
            "plan_revision_id, task_id, tenant_id, revision, schema_version, plan, plan_digest, "
            "evidence_digest, status, created_at"
            ") VALUES ('plan-a1', 'task-a', 'tenant-a', 1, 1, '{}', 'plan-digest', "
            "'evidence-digest', 'ready', CURRENT_TIMESTAMP)"
        )
    )


def _sha256(character: str) -> str:
    return "sha256:" + character * 64


def _future_timestamp(dialect: str) -> str:
    if dialect == "postgresql":
        return "CURRENT_TIMESTAMP + INTERVAL '1 day'"
    if dialect == "sqlite":
        return "datetime(CURRENT_TIMESTAMP, '+1 day')"
    raise RuntimeError("unsupported_database_dialect")


def seed_0012_m2_history(connection) -> None:
    """写入仅由 0012 能表达的历史事实，0013 不得补造 M2 绑定。"""
    future = _future_timestamp(connection.dialect.name)
    not_cancelled = _dialect_bool(connection, truthy="FALSE", falsy="0")
    insert_principal(connection, "tenant-m2", "user-m2")
    for task_id, task_mode, status in (
        ("m1-terminal", "research", "succeeded"),
        ("m1-retired", "research", "waiting_for_user"),
        ("controlled-terminal", "controlled_action", "failed"),
    ):
        values = {
            "task_id": task_id,
            "task_mode": task_mode,
            "status": status,
            "content_expires_at": future if task_mode == "research" else "NULL",
            "result_kind": "'research'" if task_id == "m1-terminal" else "NULL",
            "result_ref": "'task-results/m1-terminal/result.json'" if task_id == "m1-terminal" else "NULL",
            "result_hash": ":result_hash" if task_id == "m1-terminal" else "NULL",
            "interrupt_kind": "'user_clarification'" if task_id == "m1-retired" else "NULL",
            "interrupt_summary": "'历史澄清'" if task_id == "m1-retired" else "NULL",
        }
        connection.execute(
            text(
                "INSERT INTO agent_tasks ("
                "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, budgets, "
                "artifact_refs, "
                "cancelled, result_kind, result_ref, result_hash, interrupt_kind, interrupt_summary, "
                "content_expires_at, state_version, created_at, updated_at"
                ") VALUES ("
                ":task_id, 'tenant-m2', 'user-m2', :task_mode, :status, 'goal', 'auth-ref', '{}', "
                ":artifact_refs, "
                f"{not_cancelled}, "
                f"{values['result_kind']}, {values['result_ref']}, {values['result_hash']}, "
                f"{values['interrupt_kind']}, {values['interrupt_summary']}, "
                f"{values['content_expires_at']}, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {
                "task_id": task_id,
                "task_mode": task_mode,
                "status": status,
                "result_hash": _sha256("a"),
                # M1 既有行以非空 artifact_refs 起步；0013 round-trip 不得改动它。
                "artifact_refs": f'["artifacts/history/{task_id}"]',
            },
        )
    connection.execute(
        text(
            "INSERT INTO plan_revisions ("
            "plan_revision_id, task_id, tenant_id, revision, schema_version, plan, plan_digest, "
            "evidence_digest, status, created_at"
            ") VALUES ('m2-plan', 'controlled-terminal', 'tenant-m2', 1, 1, '{}', "
            "'plan-digest', 'evidence-digest', 'ready', CURRENT_TIMESTAMP)"
        )
    )
    approval_rows = (
        ("approval-pending", "pending", None, None),
        ("approval-approved", "approved", "user-m2", "CURRENT_TIMESTAMP"),
        ("approval-rejected", "rejected", None, None),
        ("approval-expired", "expired", None, None),
        ("approval-executed", "executed", "user-m2", "CURRENT_TIMESTAMP"),
    )
    for approval_id, status, approved_by, approved_at in approval_rows:
        approved_at_expression = approved_at or "NULL"
        connection.execute(
            text(
                "INSERT INTO approvals ("
                "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                "arguments_hash, principal_hash, evidence_digest, status, approved_by, approved_at, "
                "expires_at, created_at"
                ") VALUES ("
                ":approval_id, 'controlled-terminal', 'tenant-m2', 1, 'plan-digest', 'write', 'v1', "
                ":arguments_hash, 'principal-hash', 'evidence-digest', :status, :approved_by, "
                f"{approved_at_expression}, {future}, CURRENT_TIMESTAMP)"
            ),
            {
                "approval_id": approval_id,
                "arguments_hash": f"arguments-{approval_id}",
                "status": status,
                "approved_by": approved_by,
            },
        )
    for artifact_id, status in (
        ("artifact-ready", "ready"),
        ("artifact-creating", "creating"),
        ("artifact-unknown", "unknown"),
    ):
        tool_call_id = f"tool-{artifact_id}"
        connection.execute(
            text(
                "INSERT INTO tool_calls ("
                "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, evidence_digest, "
                "tool_name, tool_version, arguments_hash, principal_hash, mode, idempotency_key, result_hash, "
                "result_ref, result_size, result_expires_at, risk_level, status, error_code, created_at, "
                "execution_deadline, verified_at, completed_at"
                ") VALUES ("
                ":tool_call_id, 'controlled-terminal', 'tenant-m2', NULL, NULL, NULL, NULL, "
                "'read_history', 'v1', :arguments_hash, 'principal-hash', 'read', NULL, :result_hash, "
                ":result_ref, 2, "
                f"{future}, 'low', 'succeeded', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {
                "tool_call_id": tool_call_id,
                "arguments_hash": f"arguments-{tool_call_id}",
                "result_hash": _sha256("b"),
                "result_ref": f"tool-results/{tool_call_id}.json",
            },
        )
        connection.execute(
            text(
                "INSERT INTO artifacts ("
                "artifact_id, task_id, tenant_id, tool_call_id, artifact_type, storage_key, content_hash, "
                "status, created_at"
                ") VALUES ("
                ":artifact_id, 'controlled-terminal', 'tenant-m2', :tool_call_id, 'report', "
                ":storage_key, :content_hash, :status, CURRENT_TIMESTAMP)"
            ),
            {
                "artifact_id": artifact_id,
                "tool_call_id": tool_call_id,
                "storage_key": f"artifacts/history/{artifact_id}",
                "content_hash": _sha256("c"),
                "status": status,
            },
        )


def test_database_factory_never_creates_schema_implicitly(tmp_path: Path) -> None:
    database = build_database(sqlite_url(tmp_path / "empty.sqlite3"))

    assert inspect(database.engine).get_table_names() == []


def test_metadata_contains_required_enterprise_entities() -> None:
    assert EXPECTED_TABLES <= set(Base.metadata.tables)


def test_readiness_revision_matches_alembic_head() -> None:
    config = Config(str(ROOT_DIR / "alembic.ini"))
    assert ScriptDirectory.from_config(config).get_current_head() == EXPECTED_DATABASE_REVISION


def test_m1_shrink_and_retention_migration_round_trip(tmp_path: Path) -> None:
    """0011 在 SQLite 上独立完成 upgrade/downgrade round-trip。"""
    database_path = tmp_path / "shrink.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)

    command.upgrade(config, "20260826_0011")

    inspector = inspect(build_database(sqlite_url(database_path)).engine)
    columns = {item["name"] for item in inspector.get_columns("agent_tasks")}
    assert {"checkpoint_purged_at", "result_purged_at"} <= columns
    dropped = {
        "current_node",
        "current_step",
        "current_plan_revision",
        "observation_refs",
        "pending_question",
        "pending_approval_id",
        "pending_action_ref",
        "checkpoint_digest",
        "model_calls",
        "tool_calls",
        "replans",
        "consecutive_failures",
        "token_usage",
        "query_fingerprints",
        "result_fingerprints",
        "evidence_refs",
        "no_new_evidence_count",
        "repeated_action_fingerprints",
    }
    assert not dropped & columns
    check_names = {
        item["name"].removeprefix("ck_agent_tasks_").removeprefix("ck_agent_tasks_")
        for item in inspector.get_check_constraints("agent_tasks")
        if item["name"]
    }
    assert "current_node_valid" not in check_names
    assert {"state_version_positive", "research_result_complete"} <= check_names

    command.downgrade(config, "20260825_0010")

    inspector = inspect(build_database(sqlite_url(database_path)).engine)
    restored = {item["name"] for item in inspector.get_columns("agent_tasks")}
    assert "current_node" in restored and "token_usage" in restored
    assert "checkpoint_purged_at" not in restored

    command.upgrade(config, "20260826_0011")

    inspector = inspect(build_database(sqlite_url(database_path)).engine)
    assert "checkpoint_purged_at" in {
        item["name"] for item in inspector.get_columns("agent_tasks")
    }


def test_m1_fencing_and_content_lifecycle_migration_repairs_history(tmp_path: Path) -> None:
    database_path = tmp_path / "fencing-content.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "20260826_0011")

    database = build_database(sqlite_url(database_path))
    result_hash = "sha256:" + "b" * 64
    with database.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO agent_tasks ("
                "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, "
                "budgets, cancelled, result_kind, result_ref, result_hash, state_version, "
                "result_purged_at, created_at, updated_at"
                ") VALUES ("
                "'research-purged', 'tenant-a', 'user-a', 'research', 'succeeded', 'goal', 'auth-ref', "
                "'{}', 0, 'research', 'task-results/research-purged/result.json', :result_hash, 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"result_hash": result_hash},
        )
        connection.execute(
            text(
                "INSERT INTO agent_tasks ("
                "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, "
                "budgets, cancelled, state_version, created_at, updated_at"
                ") VALUES ("
                "'research-refused-missing', 'tenant-a', 'user-a', 'research', 'refused', "
                "'goal', 'auth-ref', '{}', 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        insert_plan_revision(connection)
        connection.execute(
            text(
                "INSERT INTO tool_calls ("
                "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, "
                "evidence_digest, tool_name, tool_version, arguments_hash, principal_hash, mode, "
                "result_hash, risk_level, status, error_code, created_at, completed_at, "
                "idempotency_key, execution_deadline, verified_at"
                ") VALUES ("
                "'tool-expired', 'task-a', 'tenant-a', NULL, 1, 'plan-digest', 'evidence-digest', "
                "'search', 'v1', 'arguments', 'principal', 'read', :result_hash, 'low', "
                "'failed', 'tool_result_expired', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"result_hash": result_hash},
        )
        connection.execute(
            text(
                "INSERT INTO approvals ("
                "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, "
                "tool_version, arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
                ") VALUES ("
                "'approval-write', 'task-a', 'tenant-a', 1, 'plan-digest', 'write', 'v1', "
                "'arguments-write', 'principal-write', 'evidence-digest', 'rejected', "
                "datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO tool_calls ("
                "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, "
                "evidence_digest, tool_name, tool_version, arguments_hash, principal_hash, mode, "
                "result_hash, risk_level, status, error_code, created_at, completed_at, "
                "idempotency_key, execution_deadline, verified_at"
                ") VALUES ("
                "'tool-write-expired', 'task-a', 'tenant-a', 'approval-write', 1, 'plan-digest', "
                "'evidence-digest', 'write', 'v1', 'arguments-write', 'principal-write', 'write', "
                ":result_hash, 'low', 'unknown', 'tool_result_expired', CURRENT_TIMESTAMP, "
                "CURRENT_TIMESTAMP, 'write-key', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"result_hash": result_hash},
        )

    command.upgrade(config, "head")

    inspector = inspect(build_database(sqlite_url(database_path)).engine)
    assert {item["name"] for item in inspector.get_columns("agent_capability_leases")} == {
        "capability_name",
        "protocol_version",
        "worker_id",
        "lease_expires_at",
        "updated_at",
    }
    assert "ix_agent_tasks_retention" in {
        item["name"] for item in inspector.get_indexes("agent_tasks")
    }
    assert "ix_tool_calls_retention" in {
        item["name"] for item in inspector.get_indexes("tool_calls")
    }
    with database.engine.connect() as connection:
        task = connection.execute(
            text(
                "SELECT result_kind, result_ref, result_hash, content_expires_at "
                "FROM agent_tasks WHERE task_id = 'research-purged'"
            )
        ).one()
        assert task.result_kind == "research"
        assert task.result_ref is None
        assert task.result_hash == result_hash
        assert task.content_expires_at is not None

        tool = connection.execute(
            text(
                "SELECT status, result_ref, result_size, result_expires_at, "
                "result_purged_at, result_hash, error_code "
                "FROM tool_calls WHERE tool_call_id = 'tool-expired'"
            )
        ).one()
        assert tool.status == "succeeded"
        assert tool.result_ref is None
        assert tool.result_size is None
        assert tool.result_expires_at is not None
        assert tool.result_purged_at is not None
        assert tool.result_hash == result_hash
        assert tool.error_code is None
        assert connection.execute(
            text(
                "SELECT status, error_code, result_purged_at "
                "FROM tool_calls WHERE tool_call_id = 'tool-write-expired'"
            )
        ).one() == ("unknown", "tool_result_expired", None)
        refused = connection.execute(
            text(
                "SELECT status, result_kind, result_ref, result_hash, last_error_code, "
                "content_expires_at FROM agent_tasks "
                "WHERE task_id = 'research-refused-missing'"
            )
        ).one()
        assert refused.status == "failed"
        assert refused.result_kind is None
        assert refused.result_ref is None
        assert refused.result_hash is None
        assert refused.last_error_code == "task_result_binding_missing"
        assert refused.content_expires_at is not None
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "UPDATE agent_tasks SET content_expires_at = NULL "
                    "WHERE task_id = 'research-purged'"
                )
            )
        connection.rollback()
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "UPDATE agent_tasks SET result_hash = NULL "
                    "WHERE task_id = 'research-purged'"
                )
            )
        connection.rollback()
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "UPDATE tool_calls SET result_expires_at = NULL "
                    "WHERE tool_call_id = 'tool-expired'"
                )
            )
        connection.rollback()

    command.downgrade(config, "20260826_0011")
    with database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT status FROM agent_tasks WHERE task_id = 'research-purged'")
        ).scalar_one() == "failed"
        assert connection.execute(
            text("SELECT status FROM tool_calls WHERE tool_call_id = 'tool-expired'")
        ).scalar_one() == "failed"

    command.upgrade(config, "head")
    with database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT status FROM tool_calls WHERE tool_call_id = 'tool-expired'")
        ).scalar_one() == "succeeded"


def test_m2_protocol_and_artifact_lifecycle_round_trip_preserves_history(tmp_path: Path) -> None:
    database_path = tmp_path / "m2-protocol.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "20260826_0012")

    database = build_database(sqlite_url(database_path))
    with database.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        seed_0012_m2_history(connection)

    command.upgrade(config, "head")
    inspector = inspect(database.engine)
    assert {
        "required_capability_name",
        "required_protocol_version",
        "interrupt_ref",
    } <= {item["name"] for item in inspector.get_columns("agent_tasks")}
    assert {
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
        "interrupt_id",
        "tool_call_id",
        "checkpoint_binding_hash",
        "rejected_by",
        "rejected_at",
        "rejection_reason",
    } <= {item["name"] for item in inspector.get_columns("approvals")}
    assert {"content_expires_at", "content_purged_at"} <= {
        item["name"] for item in inspector.get_columns("artifacts")
    }
    assert {"ix_agent_tasks_protocol_claim"} <= {
        item["name"] for item in inspector.get_indexes("agent_tasks")
    }
    assert {
        "ix_approvals_checkpoint_binding",
        "uq_approvals_active_checkpoint_binding",
    } <= {item["name"] for item in inspector.get_indexes("approvals")}
    assert "ix_artifacts_content_retention" in {
        item["name"] for item in inspector.get_indexes("artifacts")
    }
    approval_fks = inspector.get_foreign_keys("approvals")
    # approvals.tool_call_id 是 LangGraph checkpoint action ID，不再指向 tool_calls；
    # 否则与 tool_calls -> approvals 的执行侧复合 FK 形成即时循环，
    # 阻断“checkpoint 落盘 -> pending approval -> 批准 -> write ToolCallRow”流程。
    assert not any(item["referred_table"] == "tool_calls" for item in approval_fks)
    assert any(item["referred_table"] == "principals" for item in approval_fks)

    def check_names(table_name: str) -> set[str]:
        return {
            item["name"].removeprefix(f"ck_{table_name}_")
            for item in inspector.get_check_constraints(table_name)
            if item["name"]
        }

    # M2 关键 CHECK 与唯一索引必须真实存在（验收 inspect 断言）。
    assert {"task_protocol_complete", "interrupt_complete"} <= check_names("agent_tasks")
    assert {
        "checkpoint_binding_complete",
        "rejected_audit_complete",
    } <= check_names("approvals")
    assert "content_purge_binding" in check_names("artifacts")
    approval_indexes = {item["name"] for item in inspector.get_indexes("approvals")}
    assert "uq_approvals_active_legacy_binding" in approval_indexes
    assert "uq_approvals_active_checkpoint_binding" in approval_indexes

    with database.engine.connect() as connection:
        terminal = connection.execute(
            text(
                "SELECT status, required_capability_name, required_protocol_version "
                "FROM agent_tasks WHERE task_id = 'm1-terminal'"
            )
        ).one()
        assert terminal == ("succeeded", "research-agent", "m1-fenced-v2")
        # 历史 M1 行的 artifact_refs 在 0013 round-trip 前后保持不变。
        preserved_artifact_refs = connection.execute(
            text(
                "SELECT task_id, artifact_refs FROM agent_tasks "
                "WHERE artifact_refs IS NOT NULL ORDER BY task_id"
            )
        ).all()
        assert {
            task_id: refs
            for task_id, refs in preserved_artifact_refs
        } == {
            "controlled-terminal": '["artifacts/history/controlled-terminal"]',
            "m1-retired": '["artifacts/history/m1-retired"]',
            "m1-terminal": '["artifacts/history/m1-terminal"]',
        }
        retired = connection.execute(
            text(
                "SELECT status, last_error_code, interrupt_kind, interrupt_summary, interrupt_ref, "
                "content_expires_at, required_capability_name, required_protocol_version "
                "FROM agent_tasks WHERE task_id = 'm1-retired'"
            )
        ).one()
        assert retired.status == "failed"
        assert retired.last_error_code == "task_protocol_retired"
        assert retired.interrupt_kind is None
        assert retired.interrupt_summary is None
        assert retired.interrupt_ref is None
        assert retired.content_expires_at is not None
        assert retired.required_capability_name == "research-agent"
        assert retired.required_protocol_version == "m1-fenced-v2"
        assert tuple(
            connection.scalars(
                text("SELECT status FROM approvals ORDER BY approval_id")
            )
        ) == ("approved", "executed", "expired", "pending", "rejected")
        assert tuple(
            connection.execute(
                text(
                    "SELECT status, content_expires_at, content_purged_at FROM artifacts "
                    "ORDER BY artifact_id"
                )
            )
        ) == (
            ("creating", None, None),
            ("ready", None, None),
            ("unknown", None, None),
        )
        assert connection.execute(
            text(
                "SELECT result_ref, result_hash, result_expires_at "
                "FROM tool_calls WHERE tool_call_id = 'tool-artifact-ready'"
            )
        ).one().result_ref == "tool-results/tool-artifact-ready.json"

    command.downgrade(config, "20260826_0012")
    inspector = inspect(database.engine)
    assert "required_protocol_version" not in {
        item["name"] for item in inspector.get_columns("agent_tasks")
    }
    assert "checkpoint_binding_hash" not in {
        item["name"] for item in inspector.get_columns("approvals")
    }
    assert "content_purged_at" not in {
        item["name"] for item in inspector.get_columns("artifacts")
    }

    command.upgrade(config, "head")
    assert EXPECTED_DATABASE_REVISION == "20260913_0018"
    inspector = inspect(database.engine)
    assert "required_protocol_version" in {
        item["name"] for item in inspector.get_columns("agent_tasks")
    }
    assert "scope_document_id" in {
        item["name"] for item in inspector.get_columns("agent_tasks")
    }
    assert "scope_document_id" in {
        item["name"] for item in inspector.get_columns("conversation_turns")
    }
    assert "display_name" in {
        item["name"] for item in inspector.get_columns("artifacts")
    }
    # round-trip 后 artifact_refs 依然原样保留。
    with database.engine.connect() as connection:
        assert connection.execute(
            text(
                "SELECT COUNT(*) FROM agent_tasks "
                "WHERE artifact_refs = '[]' AND task_id LIKE 'm1%'"
            )
        ).scalar_one() == 0

    with database.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO agent_tasks ("
                "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, budgets, "
                "cancelled, required_capability_name, required_protocol_version, state_version, "
                "created_at, updated_at"
                ") VALUES ("
                "'m2-constraint-task', 'tenant-m2', 'user-m2', 'controlled_action', 'failed', "
                "'goal', 'auth-ref', '{}', 0, 'task-agent', 'm2-fenced-v1', 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO plan_revisions ("
                "plan_revision_id, task_id, tenant_id, revision, schema_version, plan, plan_digest, "
                "evidence_digest, status, created_at"
                ") VALUES ('m2-constraint-plan', 'm2-constraint-task', 'tenant-m2', 1, 1, '{}', "
                "'plan-digest', 'evidence-digest', 'ready', CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO tool_calls ("
                "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, evidence_digest, "
                "tool_name, tool_version, arguments_hash, principal_hash, mode, idempotency_key, result_hash, "
                "result_ref, result_size, result_expires_at, risk_level, status, error_code, created_at, "
                "execution_deadline, verified_at, completed_at"
                ") VALUES ("
                "'m2-tool-call', 'm2-constraint-task', 'tenant-m2', NULL, NULL, NULL, NULL, "
                "'read_history', 'v1', 'arguments-m2', 'principal-m2', 'read', NULL, :result_hash, "
                "'tool-results/m2-tool-call.json', 2, datetime(CURRENT_TIMESTAMP, '+1 day'), 'low', "
                "'succeeded', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"result_hash": _sha256("d")},
        )
        insert_principal(connection, "tenant-other", "user-other")
        connection.execute(
            text(
                "INSERT INTO agent_tasks ("
                "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, budgets, "
                "cancelled, required_capability_name, required_protocol_version, state_version, "
                "created_at, updated_at"
                ") VALUES ("
                "'other-task', 'tenant-other', 'user-other', 'controlled_action', 'failed', "
                "'goal', 'auth-ref', '{}', 0, 'task-agent', 'm2-fenced-v1', 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO plan_revisions ("
                "plan_revision_id, task_id, tenant_id, revision, schema_version, plan, plan_digest, "
                "evidence_digest, status, created_at"
                ") VALUES ('other-plan', 'other-task', 'tenant-other', 1, 1, '{}', "
                "'plan-digest', 'evidence-digest', 'ready', CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO tool_calls ("
                "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, evidence_digest, "
                "tool_name, tool_version, arguments_hash, principal_hash, mode, idempotency_key, result_hash, "
                "result_ref, result_size, result_expires_at, risk_level, status, error_code, created_at, "
                "execution_deadline, verified_at, completed_at"
                ") VALUES ("
                "'other-tool-call', 'other-task', 'tenant-other', NULL, NULL, NULL, NULL, "
                "'read_history', 'v1', 'arguments-other', 'principal-other', 'read', NULL, :result_hash, "
                "'tool-results/other-tool-call.json', 2, datetime(CURRENT_TIMESTAMP, '+1 day'), 'low', "
                "'succeeded', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"result_hash": _sha256("e")},
        )

    def assert_constraint_rejected(statement: str, parameters: dict[str, object] | None = None) -> None:
        with database.engine.connect() as connection:
            with pytest.raises(IntegrityError):
                connection.execute(text(statement), parameters or {})
            connection.rollback()

    approval_columns = (
        "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
        "arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
    )
    approval_values = (
        ":approval_id, 'm2-constraint-task', 'tenant-m2', 1, 'plan-digest', 'write', 'v1', "
        ":arguments_hash, 'principal-m2', 'evidence-digest', :status, "
        "datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP"
    )
    assert_constraint_rejected(
        "INSERT INTO approvals ("
        f"{approval_columns}, thread_id"
        ") VALUES ("
        f"{approval_values}, 'thread-m2'"
        ")",
        {
            "approval_id": "missing-checkpoint-binding",
            "arguments_hash": "arguments-missing",
            "status": "pending",
        },
    )
    binding_columns = (
        "thread_id, checkpoint_ns, checkpoint_id, interrupt_id, tool_call_id, checkpoint_binding_hash"
    )
    binding_values = "'thread-m2', '', 'checkpoint-m2', 'interrupt-m2', :tool_call_id, :binding_hash"
    assert_constraint_rejected(
        "INSERT INTO approvals ("
        f"{approval_columns}, {binding_columns}"
        ") VALUES ("
        f"{approval_values}, {binding_values}"
        ")",
        {
            "approval_id": "approved-without-audit",
            "arguments_hash": "arguments-approved",
            "status": "approved",
            "tool_call_id": "m2-tool-call",
            "binding_hash": _sha256("f"),
        },
    )
    assert_constraint_rejected(
        "INSERT INTO approvals ("
        f"{approval_columns}, {binding_columns}"
        ") VALUES ("
        f"{approval_values}, {binding_values}"
        ")",
        {
            "approval_id": "rejected-without-audit",
            "arguments_hash": "arguments-rejected",
            "status": "rejected",
            "tool_call_id": "m2-tool-call",
            "binding_hash": _sha256("g"),
        },
    )

    # M2 HITL 正例：checkpoint 已落盘后先建 pending approval，此时执行侧
    # ToolCallRow 尚不存在；approvals.tool_call_id 只是 checkpoint action ID。
    # （旧断言“approval 指向跨 tenant ToolCallRow 失败”已删除：该关系不再是数据库 FK。）
    with database.engine.connect() as connection:
        connection.execute(
            text(
                "INSERT INTO approvals ("
                f"{approval_columns}, {binding_columns}"
                ") VALUES ("
                f"{approval_values}, {binding_values}"
                ")",
            ),
            {
                "approval_id": "pending-checkpoint-approval",
                "arguments_hash": "arguments-pending",
                "status": "pending",
                "tool_call_id": "checkpoint-action-not-a-row-yet",
                "binding_hash": _sha256("j"),
            },
        )
        connection.rollback()

    # 执行侧复合绑定仍由数据库强制：write ToolCallRow 必须精确命中同一审批。
    assert_constraint_rejected(
        "INSERT INTO tool_calls ("
        "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, evidence_digest, "
        "tool_name, tool_version, arguments_hash, principal_hash, mode, idempotency_key, "
        "risk_level, status, created_at, execution_deadline"
        ") VALUES ("
        "'m2-write-cross-binding', 'm2-constraint-task', 'tenant-m2', 'other-approval', 1, 'plan-digest', "
        "'evidence-digest', 'write', 'v1', 'arguments-m2', 'principal-m2', 'write', 'write-key-cross', "
        "'high', 'proposed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )
    other_approval_values = (
        ":approval_id, 'other-task', 'tenant-other', 1, 'plan-digest', 'write', 'v1', "
        ":arguments_hash, 'principal-other', 'evidence-digest', 'approved', 'user-other', "
        "CURRENT_TIMESTAMP, datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP"
    )
    with database.engine.connect() as connection:
        connection.execute(
            text(
                "INSERT INTO approvals ("
                "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                "arguments_hash, principal_hash, evidence_digest, status, approved_by, approved_at, "
                "expires_at, created_at"
                ") VALUES (" + other_approval_values + ")"
            ),
            {
                "approval_id": "other-approval",
                "arguments_hash": "arguments-other-approval",
            },
        )
        connection.rollback()
    assert_constraint_rejected(
        "INSERT INTO tool_calls ("
        "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, evidence_digest, "
        "tool_name, tool_version, arguments_hash, principal_hash, mode, idempotency_key, "
        "risk_level, status, created_at, execution_deadline"
        ") VALUES ("
        "'m2-write-cross-tenant', 'm2-constraint-task', 'tenant-m2', 'other-approval', 1, 'plan-digest', "
        "'evidence-digest', 'write', 'v1', 'arguments-m2', 'principal-m2', 'write', 'write-key-x', "
        "'high', 'proposed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )
    assert_constraint_rejected(
        "INSERT INTO artifacts ("
        "artifact_id, task_id, tenant_id, tool_call_id, artifact_type, storage_key, content_hash, status, "
        "content_expires_at, content_purged_at, created_at"
        ") VALUES ("
        "'purged-but-readable', 'm2-constraint-task', 'tenant-m2', 'm2-tool-call', 'report', "
        "'artifacts/purged-but-readable', :content_hash, 'ready', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP)",
        {"content_hash": _sha256("i")},
    )


def _prepare_m2_downgrade_database(tmp_path: Path, name: str):
    """0012 历史 + 升到 0013，供 downgrade fail-close 场景复用。"""
    database_path = tmp_path / f"{name}.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "20260826_0012")
    database = build_database(sqlite_url(database_path))
    with database.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        seed_0012_m2_history(connection)
    command.upgrade(config, "head")
    return config, database


@pytest.mark.parametrize("fact_kind", ["m2_protocol", "checkpoint_binding"])
def test_m2_downgrade_refuses_when_m2_facts_present(tmp_path: Path, fact_kind: str) -> None:
    """存在 M2 协议/checkpoint 事实时，0013 downgrade 必须 fail-close。"""
    config, database = _prepare_m2_downgrade_database(tmp_path, fact_kind)
    with database.engine.begin() as connection:
        if fact_kind == "m2_protocol":
            # 非默认 capability/protocol 的任务行本身就是 M2 事实。
            connection.execute(
                text(
                    "INSERT INTO agent_tasks (task_id, tenant_id, user_id, task_mode, status, goal, "
                    "auth_context_ref, budgets, cancelled, required_capability_name, "
                    "required_protocol_version, result_kind, result_ref, result_hash, "
                    "content_expires_at, state_version, created_at, updated_at) VALUES ('dg-task', "
                    "'tenant-a', 'user-a', 'research', 'succeeded', 'goal', 'auth-ref', '{}', 0, "
                    "'task-agent', 'm2-fenced-v1', 'research', 'task-results/dg-task/result.json', "
                    ":result_hash, datetime(CURRENT_TIMESTAMP, '+7 days'), 1, CURRENT_TIMESTAMP, "
                    "CURRENT_TIMESTAMP)"
                ),
                {"result_hash": _sha256("l")},
            )
        else:
            # checkpoint 绑定审批（pending + 完整 binding）也是 M2 事实。
            connection.execute(
                text(
                    "INSERT INTO agent_tasks (task_id, tenant_id, user_id, task_mode, status, goal, "
                    "auth_context_ref, budgets, cancelled, result_kind, result_ref, result_hash, "
                    "content_expires_at, state_version, created_at, updated_at) VALUES ("
                    "'task-a', 'tenant-a', 'user-a', 'research', 'succeeded', 'goal', 'auth-ref', "
                    "'{}', 0, 'research', 'task-results/task-a/result.json', :result_hash, "
                    "datetime(CURRENT_TIMESTAMP, '+7 days'), 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"result_hash": _sha256("m")},
            )
            connection.execute(
                text(
                    "INSERT INTO plan_revisions (plan_revision_id, task_id, tenant_id, revision, "
                    "schema_version, plan, plan_digest, evidence_digest, status, created_at) VALUES ("
                    "'dg-plan', 'task-a', 'tenant-a', 1, 1, '{}', 'plan-digest', 'evidence-digest', "
                    "'ready', CURRENT_TIMESTAMP)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO approvals (approval_id, task_id, tenant_id, plan_revision, plan_digest, "
                    "tool_name, tool_version, arguments_hash, principal_hash, evidence_digest, status, "
                    "expires_at, created_at, thread_id, checkpoint_ns, checkpoint_id, interrupt_id, "
                    "tool_call_id, checkpoint_binding_hash) VALUES ('dg-approval', 'task-a', 'tenant-a', "
                    "1, 'plan-digest', 'write', 'v1', 'arguments', 'principal', 'evidence-digest', "
                    "'pending', datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP, 'thread-a', "
                    "'', 'checkpoint-a', 'interrupt-a', 'action-a', :binding_hash)"
                ),
                {"binding_hash": _sha256("k")},
            )
    with pytest.raises(RuntimeError, match="m2_downgrade_data_present"):
        command.downgrade(config, "20260826_0012")


@pytest.mark.skipif(
    not POSTGRES_MIGRATION_TEST_URL,
    reason="postgres_migration_test_url_unconfigured",
)
def test_m2_protocol_and_artifact_lifecycle_round_trip_on_postgres() -> None:
    """只使用显式空临时库执行 0012 -> 0013 -> 0012 -> 0013。"""
    assert POSTGRES_MIGRATION_TEST_URL is not None
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = POSTGRES_MIGRATION_TEST_URL
    database = build_database(POSTGRES_MIGRATION_TEST_URL)
    is_empty = not inspect(database.engine).get_table_names()
    try:
        if not is_empty:
            pytest.fail("postgres_migration_test_database_must_be_empty")
        command.upgrade(config, "20260826_0012")
        with database.engine.begin() as connection:
            seed_0012_m2_history(connection)
        command.upgrade(config, "head")
        inspector = inspect(database.engine)
        assert "agent_capability_leases" in inspector.get_table_names()
        assert "content_expires_at" in {
            item["name"] for item in inspector.get_columns("agent_tasks")
        }
        assert "result_purged_at" in {
            item["name"] for item in inspector.get_columns("tool_calls")
        }
        assert {"research_result_complete", "research_content_expiry"} <= {
            item["name"].removeprefix("ck_agent_tasks_")
            for item in inspector.get_check_constraints("agent_tasks")
            if item["name"]
        }
        assert {"result_reference_valid", "succeeded_result_reference_complete"} <= {
            item["name"].removeprefix("ck_tool_calls_")
            for item in inspector.get_check_constraints("tool_calls")
            if item["name"]
        }
        assert "ix_agent_tasks_retention" in {
            item["name"] for item in inspector.get_indexes("agent_tasks")
        }
        assert "ix_tool_calls_retention" in {
            item["name"] for item in inspector.get_indexes("tool_calls")
        }
        assert "ix_agent_capability_leases_expiry" in {
            item["name"] for item in inspector.get_indexes("agent_capability_leases")
        }
        assert {
            "required_capability_name",
            "required_protocol_version",
            "interrupt_ref",
        } <= {item["name"] for item in inspector.get_columns("agent_tasks")}
        assert {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "interrupt_id",
            "tool_call_id",
            "checkpoint_binding_hash",
            "rejected_by",
            "rejected_at",
            "rejection_reason",
        } <= {item["name"] for item in inspector.get_columns("approvals")}
        assert {"content_expires_at", "content_purged_at"} <= {
            item["name"] for item in inspector.get_columns("artifacts")
        }
        assert "ix_agent_tasks_protocol_claim" in {
            item["name"] for item in inspector.get_indexes("agent_tasks")
        }
        assert "uq_approvals_active_checkpoint_binding" in {
            item["name"] for item in inspector.get_indexes("approvals")
        }
        assert "ix_artifacts_content_retention" in {
            item["name"] for item in inspector.get_indexes("artifacts")
        }
        assert not any(
            item["referred_table"] == "tool_calls"
            for item in inspector.get_foreign_keys("approvals")
        )
        with database.engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT status, required_capability_name, required_protocol_version "
                    "FROM agent_tasks WHERE task_id = 'm1-terminal'"
                )
            ).one() == ("succeeded", "research-agent", "m1-fenced-v2")
            assert connection.execute(
                text(
                    "SELECT status, last_error_code, interrupt_ref FROM agent_tasks "
                    "WHERE task_id = 'm1-retired'"
                )
            ).one() == ("failed", "task_protocol_retired", None)

        command.downgrade(config, "20260826_0012")
        inspector = inspect(database.engine)
        assert "required_capability_name" not in {
            item["name"] for item in inspector.get_columns("agent_tasks")
        }
        assert "checkpoint_binding_hash" not in {
            item["name"] for item in inspector.get_columns("approvals")
        }
        assert "content_purged_at" not in {
            item["name"] for item in inspector.get_columns("artifacts")
        }

        command.upgrade(config, "head")
        inspector = inspect(database.engine)
        assert "required_protocol_version" in {
            item["name"] for item in inspector.get_columns("agent_tasks")
        }
        assert "checkpoint_binding_hash" in {
            item["name"] for item in inspector.get_columns("approvals")
        }
        assert "content_purged_at" in {
            item["name"] for item in inspector.get_columns("artifacts")
        }
    finally:
        if is_empty:
            with database.engine.begin() as connection:
                for table_name in (
                    "artifacts",
                    "tool_calls",
                    "approvals",
                    "plan_revisions",
                    "agent_tasks",
                    "principals",
                    "tenants",
                ):
                    connection.execute(
                        text(f"DELETE FROM {table_name} WHERE tenant_id = :tenant"),
                        {"tenant": "tenant-m2"},
                    )
            command.downgrade(config, "base")
        database.engine.dispose()


def test_alembic_upgrade_and_downgrade_on_sqlite_adapter(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)

    command.upgrade(config, "head")

    database = build_database(sqlite_url(database_path))
    inspector = inspect(database.engine)
    assert EXPECTED_TABLES <= set(inspector.get_table_names())
    assert "ix_agent_tasks_claim" in {
        index["name"] for index in inspector.get_indexes("agent_tasks")
    }
    assert "ix_ingestion_jobs_expired" in {
        index["name"] for index in inspector.get_indexes("ingestion_jobs")
    }
    assert "ix_tool_calls_sweeper" in {
        index["name"] for index in inspector.get_indexes("tool_calls")
    }
    command.check(config)

    command.downgrade(config, "base")

    assert inspect(database.engine).get_table_names() == ["alembic_version"]


def test_conversation_downgrade_refuses_persisted_conversation_data(tmp_path: Path) -> None:
    database_path = tmp_path / "conversation-downgrade.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "head")
    database = build_database(sqlite_url(database_path))
    with database.engine.begin() as connection:
        insert_principal(connection, "tenant-conversation", "user-conversation")
        connection.execute(
            text(
                "INSERT INTO conversations (conversation_id, tenant_id, user_id, status, "
                "state_version, summary_through_sequence, memory_version, created_at, updated_at) "
                "VALUES ('conversation-a', 'tenant-conversation', 'user-conversation', 'active', "
                "1, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )

    with pytest.raises(RuntimeError, match="conversation_downgrade_data_present"):
        command.downgrade(config, "20260831_0015")


def test_observability_m0_migration_preserves_legacy_correlation_id(tmp_path: Path) -> None:
    database_path = tmp_path / "observability-m0.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "20260827_0013")
    database = build_database(sqlite_url(database_path))

    with database.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-observe", "user-observe")
        connection.execute(
            text(
                "INSERT INTO agent_tasks ("
                "task_id, tenant_id, user_id, task_mode, status, goal, auth_context_ref, budgets, "
                "cancelled, required_capability_name, required_protocol_version, state_version, "
                "created_at, updated_at"
                ") VALUES ("
                "'task-observe', 'tenant-observe', 'user-observe', 'controlled_action', 'failed', "
                "'goal', 'auth-ref', '{}', 0, 'task-agent', 'm2-fenced-v1', 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO task_events ("
                "event_id, task_id, tenant_id, sequence, event_type, trace_id, run_id, stage, "
                "status, counts, created_at"
                ") VALUES ("
                "'event-observe', 'task-observe', 'tenant-observe', 0, 'task_created', "
                "'legacy-request-id', 'legacy-run-id', 'planning', 'planning', '{}', CURRENT_TIMESTAMP)"
            )
        )

    command.upgrade(config, "head")

    inspector = inspect(database.engine)
    event_columns = {item["name"]: item for item in inspector.get_columns("task_events")}
    assert {"correlation_id", "otel_trace_id", "otel_span_id"} <= set(event_columns)
    assert event_columns["run_id"]["nullable"] is True
    assert {"pending_traceparent", "pending_tracestate"} <= {
        item["name"] for item in inspector.get_columns("agent_tasks")
    }
    assert {"pending_traceparent", "pending_tracestate"} <= {
        item["name"] for item in inspector.get_columns("ingestion_jobs")
    }
    with database.engine.connect() as connection:
        assert connection.execute(
            text(
                "SELECT correlation_id, run_id, otel_trace_id, otel_span_id "
                "FROM task_events WHERE event_id = 'event-observe'"
            )
        ).one() == ("legacy-request-id", "legacy-run-id", None, None)

    command.downgrade(config, "20260827_0013")
    with database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT trace_id, run_id FROM task_events WHERE event_id = 'event-observe'")
        ).one() == ("legacy-request-id", "legacy-run-id")


def test_m1_migration_closes_legacy_runtime_records(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-runtime.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "20260824_0009")

    database = build_database(sqlite_url(database_path))
    result_hash = "sha256:" + "a" * 64
    with database.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        insert_task(connection, "tenant-a", "task-a", "user-a")
        connection.execute(
            text(
                "UPDATE agent_tasks SET task_mode = 'research', status = 'succeeded' "
                "WHERE task_id = 'task-a'"
            )
        )
        insert_plan_revision(connection)
        connection.execute(
            text(
                "INSERT INTO tool_calls ("
                "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, "
                "evidence_digest, tool_name, tool_version, arguments_hash, principal_hash, mode, "
                "result_hash, risk_level, status, error_code, created_at, completed_at, "
                "idempotency_key, execution_deadline, verified_at"
                ") VALUES ("
                "'tool-a', 'task-a', 'tenant-a', NULL, 1, 'plan-digest', 'evidence-digest', "
                "'search', 'v1', 'arguments', 'principal', 'read', :result_hash, 'low', "
                "'succeeded', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"result_hash": result_hash},
        )
        insert_task(connection, "tenant-a", "task-wait", "user-a")
        connection.execute(
            text(
                "UPDATE agent_tasks SET status = 'waiting_for_user', "
                "pending_question = 'Need a scope.' WHERE task_id = 'task-wait'"
            )
        )

    command.upgrade(config, "head")

    with database.engine.connect() as connection:
        assert connection.execute(
            text(
                "SELECT status, result_ref, result_hash, last_error_code "
                "FROM agent_tasks WHERE task_id = 'task-a'"
            )
        ).one() == ("failed", None, None, "task_runtime_replaced")
        assert connection.execute(
            text(
                "SELECT status, interrupt_kind, interrupt_summary "
                "FROM agent_tasks WHERE task_id = 'task-wait'"
            )
        ).one() == ("failed", None, None)
        assert connection.execute(
            text(
                "SELECT status, result_ref, result_hash, error_code "
                "FROM tool_calls WHERE tool_call_id = 'tool-a'"
            )
        ).one() == ("unknown", None, result_hash, "task_runtime_replaced")


def test_sqlite_constraints_reject_cross_tenant_and_duplicate_state(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        insert_principal(connection, "tenant-b", "user-b")
        connection.execute(
            text(
                "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                "VALUES ('tenant-a', 'document-a', 'title', 'department', 'active', 'user-a', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_versions ("
                "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, desired_projection_generation, created_at"
                    ") VALUES ('version-a1', 'tenant-a', 'document-a', 1, 'active', 'source', 'file', 'hash-a1', 'index', 1, CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO chunk_metadata ("
                    "chunk_id, version_id, tenant_id, document_id, document_version, content_hash, "
                    "source_locator, source_file_name, index_version"
                    ") VALUES ('cross-tenant-chunk', 'version-a1', 'tenant-b', 'document-a', 1, 'hash', 'section', 'file', 'index')"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-a", "user-a")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        insert_plan_revision(connection)
        connection.execute(
            text(
                "INSERT INTO approvals ("
                "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                "arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
                ") VALUES ('approval-a1', 'task-a', 'tenant-a', 1, 'plan-digest', 'write', 'v1', "
                "'arguments', 'principal', 'evidence-digest', 'pending', datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO approvals ("
                    "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                    "arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
                    ") VALUES ('approval-a2', 'task-a', 'tenant-a', 1, 'plan-digest', 'write', 'v1', "
                    "'arguments', 'principal', 'evidence-digest', 'approved', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-a", "user-a")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        insert_plan_revision(connection)
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO tool_calls ("
                    "tool_call_id, task_id, tenant_id, approval_id, plan_revision, plan_digest, "
                    "evidence_digest, tool_name, tool_version, arguments_hash, principal_hash, mode, "
                    "risk_level, status, created_at"
                    ") VALUES ('tool-missing-approval', 'task-a', 'tenant-a', NULL, 1, 'plan-digest', "
                    "'evidence-digest', 'write', 'v1', 'arguments', 'principal', 'write', "
                    "'low', 'pending', CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        # 回滚会撤销前置数据；重建最小事实后继续验证任务子表约束。
        insert_principal(connection, "tenant-a", "user-a")
        insert_principal(connection, "tenant-b", "user-b")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO task_events ("
                "event_id, task_id, tenant_id, sequence, event_type, correlation_id, run_id, stage, status, counts, created_at"
                ") VALUES ('event-a1', 'task-a', 'tenant-a', 1, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO task_events ("
                    "event_id, task_id, tenant_id, sequence, event_type, correlation_id, run_id, stage, status, counts, created_at"
                    ") VALUES ('event-cross-tenant', 'task-a', 'tenant-b', 2, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-a", "user-a")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO task_events ("
                "event_id, task_id, tenant_id, sequence, event_type, correlation_id, run_id, stage, status, counts, created_at"
                ") VALUES ('event-a1', 'task-a', 'tenant-a', 1, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO task_events ("
                    "event_id, task_id, tenant_id, sequence, event_type, correlation_id, run_id, stage, status, counts, created_at"
                    ") VALUES ('event-a2', 'task-a', 'tenant-a', 1, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                "VALUES ('tenant-a', 'document-a', 'title', 'department', 'active', 'user-a', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_versions ("
                "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, desired_projection_generation, created_at"
                    ") VALUES ('version-a1', 'tenant-a', 'document-a', 1, 'active', 'source', 'file', 'hash-a1', 'index', 1, CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO document_versions ("
                    "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, created_at"
                    ") VALUES ('version-a2', 'tenant-a', 'document-a', 2, 'active', 'source', 'file', 'hash-a2', 'index', CURRENT_TIMESTAMP)"
                )
            )


def test_approval_and_write_tool_binding_constraints(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        insert_plan_revision(connection)
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO approvals ("
                    "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                    "arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
                    ") VALUES ('approval-invalid', 'task-a', 'tenant-a', 1, 'wrong-plan', 'write', 'v1', "
                    "'arguments', 'principal', 'evidence-digest', 'pending', datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-a", "user-a")
        insert_shrunk_task(connection, "tenant-a", "task-a", "user-a")
        insert_plan_revision(connection)
        connection.execute(
            text(
                "INSERT INTO approvals ("
                "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                "arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
                ") VALUES ('approval-a1', 'task-a', 'tenant-a', 1, 'plan-digest', 'write', 'v1', "
                "'arguments', 'principal', 'evidence-digest', 'pending', datetime(CURRENT_TIMESTAMP, '+1 day'), CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO approvals ("
                    "approval_id, task_id, tenant_id, plan_revision, plan_digest, tool_name, tool_version, "
                    "arguments_hash, principal_hash, evidence_digest, status, expires_at, created_at"
                    ") VALUES ('approval-a2', 'task-a', 'tenant-a', 1, 'plan-digest', 'write', 'v1', "
                    "'arguments', 'principal', 'evidence-digest', 'approved', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


def test_ingestion_lease_generation_defaults_and_rejects_negative_value(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                "VALUES ('tenant-a', 'document-a', 'title', 'department', 'pending', 'user-a', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_versions ("
                "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                "source_file_name, content_hash, index_version, created_at"
                ") VALUES ('version-a1', 'tenant-a', 'document-a', 1, 'pending', 'source', 'file', 'hash', 'index', CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO ingestion_jobs (job_id, tenant_id, version_id, status, attempts, available_at, created_at, updated_at) "
                "VALUES ('job-a1', 'tenant-a', 'version-a1', 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        assert connection.scalar(
            text("SELECT lease_generation FROM ingestion_jobs WHERE job_id = 'job-a1'")
        ) == 0
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "UPDATE ingestion_jobs SET lease_generation = -1 WHERE job_id = 'job-a1'"
                )
            )


def test_idempotency_records_are_isolated_by_subject_and_tenant(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO principals (tenant_id, user_id, roles, attributes, active, created_at) "
                "VALUES ('tenant-a', 'user-b', '[]', '{}', 1, CURRENT_TIMESTAMP)"
            )
        )
        for record_id, user_id in (("idempotency-a", "user-a"), ("idempotency-b", "user-b")):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, status, created_at, updated_at"
                    ") VALUES (:record_id, 'tenant-a', :user_id, 'upload', 'same-key', 'hash', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"record_id": record_id, "user_id": user_id},
            )
        assert connection.scalar(text("SELECT COUNT(*) FROM idempotency_records")) == 2
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, status, created_at, updated_at"
                    ") VALUES ('idempotency-duplicate', 'tenant-a', 'user-a', 'upload', 'same-key', 'hash', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-b", "user-b")
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, status, created_at, updated_at"
                    ") VALUES ('idempotency-cross-tenant', 'tenant-b', 'user-a', 'upload', 'key', 'hash', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()


def test_document_versions_reject_duplicate_tenant_object_key(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                "VALUES ('tenant-a', 'document-a', 'title', 'department', 'pending', 'user-a', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_versions ("
                "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                "source_file_name, content_hash, index_version, created_at"
                ") VALUES ('version-a1', 'tenant-a', 'document-a', 1, 'pending', 'object-a', 'file', 'hash-a1', 'index', CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO document_versions ("
                    "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, created_at"
                    ") VALUES ('version-a2', 'tenant-a', 'document-a', 2, 'pending', 'object-a', 'file', 'hash-a2', 'index', CURRENT_TIMESTAMP)"
                )
            )


def test_idempotency_result_references_are_tenant_scoped(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        for tenant_id, user_id, document_id, version_id, job_id in (
            ("tenant-a", "user-a", "document-a", "version-a1", "job-a1"),
            ("tenant-b", "user-b", "document-b", "version-b1", "job-b1"),
        ):
            insert_principal(connection, tenant_id, user_id)
            connection.execute(
                text(
                    "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                    "VALUES (:tenant_id, :document_id, 'title', 'department', 'pending', :user_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"tenant_id": tenant_id, "document_id": document_id, "user_id": user_id},
            )
            connection.execute(
                text(
                    "INSERT INTO document_versions ("
                    "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, created_at"
                    ") VALUES (:version_id, :tenant_id, :document_id, 1, 'pending', :source_object_key, 'file', 'hash', 'index', CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "version_id": version_id,
                    "source_object_key": f"object-{tenant_id}",
                },
            )
            connection.execute(
                text(
                    "INSERT INTO ingestion_jobs (job_id, tenant_id, version_id, status, attempts, available_at, created_at, updated_at) "
                    "VALUES (:job_id, :tenant_id, :version_id, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"tenant_id": tenant_id, "version_id": version_id, "job_id": job_id},
            )

        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, result_version_id, status, created_at, updated_at"
                    ") VALUES ('cross-version', 'tenant-a', 'user-a', 'document_upload', 'key-a', 'hash', 'version-b1', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


        connection.rollback()

        # 回滚会撤销初始化数据；重建后验证摄取任务结果也遵守同一租户边界。
        for tenant_id, user_id, document_id, version_id, job_id in (
            ("tenant-a", "user-a", "document-a", "version-a1", "job-a1"),
            ("tenant-b", "user-b", "document-b", "version-b1", "job-b1"),
        ):
            insert_principal(connection, tenant_id, user_id)
            connection.execute(
                text(
                    "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                    "VALUES (:tenant_id, :document_id, 'title', 'department', 'pending', :user_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"tenant_id": tenant_id, "document_id": document_id, "user_id": user_id},
            )
            connection.execute(
                text(
                    "INSERT INTO document_versions ("
                    "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, created_at"
                    ") VALUES (:version_id, :tenant_id, :document_id, 1, 'pending', :source_object_key, 'file', 'hash', 'index', CURRENT_TIMESTAMP)"
                ),
                {
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "version_id": version_id,
                    "source_object_key": f"object-{tenant_id}",
                },
            )
            connection.execute(
                text(
                    "INSERT INTO ingestion_jobs (job_id, tenant_id, version_id, status, attempts, available_at, created_at, updated_at) "
                    "VALUES (:job_id, :tenant_id, :version_id, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"tenant_id": tenant_id, "version_id": version_id, "job_id": job_id},
            )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, result_job_id, status, created_at, updated_at"
                    ") VALUES ('cross-job', 'tenant-a', 'user-a', 'document_upload', 'key-b', 'hash', 'job-b1', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


def test_idempotency_result_pair_must_reference_one_ingestion_job(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        for document_id, version_id, job_id in (
            ("document-a", "version-a1", "job-a1"),
            ("document-b", "version-a2", "job-a2"),
        ):
            connection.execute(
                text(
                    "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                    "VALUES ('tenant-a', :document_id, 'title', 'department', 'pending', 'user-a', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"document_id": document_id},
            )
            connection.execute(
                text(
                    "INSERT INTO document_versions ("
                    "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                    "source_file_name, content_hash, index_version, created_at"
                    ") VALUES (:version_id, 'tenant-a', :document_id, 1, 'pending', :object_key, 'file', 'hash', 'index', CURRENT_TIMESTAMP)"
                ),
                {
                    "document_id": document_id,
                    "version_id": version_id,
                    "object_key": f"object-{version_id}",
                },
            )
            connection.execute(
                text(
                    "INSERT INTO ingestion_jobs (job_id, tenant_id, version_id, status, attempts, available_at, created_at, updated_at) "
                    "VALUES (:job_id, 'tenant-a', :version_id, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"version_id": version_id, "job_id": job_id},
            )

        # 空占位保持合法；业务实体成功落库后才在同一事务内回填三列。
        connection.execute(
            text(
                "INSERT INTO idempotency_records ("
                "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, status, created_at, updated_at"
                ") VALUES ('placeholder', 'tenant-a', 'user-a', 'document_upload', 'placeholder-key', 'hash', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, result_reference, status, created_at, updated_at"
                    ") VALUES ('partial-result', 'tenant-a', 'user-a', 'document_upload', 'partial-key', 'hash', 'reference-only', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO idempotency_records ("
                    "record_id, tenant_id, user_id, scope, idempotency_key, request_hash, result_reference, result_version_id, result_job_id, status, created_at, updated_at"
                    ") VALUES ('mismatched-result', 'tenant-a', 'user-a', 'document_upload', 'mismatch-key', 'hash', 'version-a1:job-a2', 'version-a1', 'job-a2', 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


def test_index_receipt_and_acl_grant_constraints(tmp_path: Path) -> None:
    database = upgrade_sqlite_database(tmp_path)

    with database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        insert_principal(connection, "tenant-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO documents (tenant_id, document_id, title, department, status, created_by, created_at, updated_at) "
                "VALUES ('tenant-a', 'document-a', 'title', 'department', 'pending', 'user-a', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_versions ("
                "version_id, tenant_id, document_id, version, lifecycle_status, source_object_key, "
                "source_file_name, content_hash, index_version, created_at"
                ") VALUES ('version-a1', 'tenant-a', 'document-a', 1, 'pending', 'object-a', 'file', 'hash', 'index', CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO ingestion_jobs ("
                "job_id, tenant_id, version_id, status, attempts, available_at, indexed_chunk_count, "
                "index_manifest_digest, index_completed_at, created_at, updated_at"
                ") VALUES ('job-a1', 'tenant-a', 'version-a1', 'pending', 0, CURRENT_TIMESTAMP, 3, "
                "'manifest-digest', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "UPDATE ingestion_jobs SET indexed_chunk_count = 0 WHERE job_id = 'job-a1'"
                )
            )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "UPDATE ingestion_jobs SET index_manifest_digest = NULL WHERE job_id = 'job-a1'"
                )
            )

        connection.execute(
            text(
                "INSERT INTO acl_grants ("
                "grant_id, tenant_id, resource_type, resource_id, subject_type, subject_id, permissions, created_at"
                ") VALUES ('grant-a1', 'tenant-a', 'document', 'document-a', 'user', 'user-a', '[]', CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO acl_grants ("
                    "grant_id, tenant_id, resource_type, resource_id, subject_type, subject_id, permissions, created_at"
                    ") VALUES ('grant-a2', 'tenant-a', 'document', 'document-a', 'user', 'user-a', '[]', CURRENT_TIMESTAMP)"
                )
            )
