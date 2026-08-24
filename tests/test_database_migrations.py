from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from rag_kb.infrastructure.database import Base, build_database
from rag_kb.web.dependencies import EXPECTED_DATABASE_REVISION


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {
    "acl_grants",
    "agent_tasks",
    "approvals",
    "artifacts",
    "chunk_metadata",
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
    "tenants",
    "tool_calls",
}


def sqlite_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def upgrade_sqlite_database(tmp_path: Path):
    database_path = tmp_path / "constraints.sqlite3"
    config = Config(str(ROOT_DIR / "alembic.ini"))
    config.attributes["database_url"] = sqlite_url(database_path)
    command.upgrade(config, "head")
    return build_database(sqlite_url(database_path))


def insert_principal(connection, tenant_id: str, user_id: str) -> None:
    connection.execute(
        text(
            "INSERT INTO tenants (tenant_id, name, active, created_at) "
            "VALUES (:tenant_id, :name, 1, CURRENT_TIMESTAMP)"
        ),
        {"tenant_id": tenant_id, "name": tenant_id},
    )
    connection.execute(
        text(
            "INSERT INTO principals (tenant_id, user_id, roles, attributes, active, created_at) "
            "VALUES (:tenant_id, :user_id, '[]', '{}', 1, CURRENT_TIMESTAMP)"
        ),
        {"tenant_id": tenant_id, "user_id": user_id},
    )


def insert_task(connection, tenant_id: str, task_id: str, user_id: str) -> None:
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


def test_database_factory_never_creates_schema_implicitly(tmp_path: Path) -> None:
    database = build_database(sqlite_url(tmp_path / "empty.sqlite3"))

    assert inspect(database.engine).get_table_names() == []


def test_metadata_contains_required_enterprise_entities() -> None:
    assert EXPECTED_TABLES <= set(Base.metadata.tables)


def test_readiness_revision_matches_alembic_head() -> None:
    config = Config(str(ROOT_DIR / "alembic.ini"))
    assert ScriptDirectory.from_config(config).get_current_head() == EXPECTED_DATABASE_REVISION


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
        insert_task(connection, "tenant-a", "task-a", "user-a")
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
        insert_task(connection, "tenant-a", "task-a", "user-a")
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
        insert_task(connection, "tenant-a", "task-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO task_events ("
                "event_id, task_id, tenant_id, sequence, event_type, trace_id, run_id, stage, status, counts, created_at"
                ") VALUES ('event-a1', 'task-a', 'tenant-a', 1, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO task_events ("
                    "event_id, task_id, tenant_id, sequence, event_type, trace_id, run_id, stage, status, counts, created_at"
                    ") VALUES ('event-cross-tenant', 'task-a', 'tenant-b', 2, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
                )
            )
        connection.rollback()

        insert_principal(connection, "tenant-a", "user-a")
        insert_task(connection, "tenant-a", "task-a", "user-a")
        connection.execute(
            text(
                "INSERT INTO task_events ("
                "event_id, task_id, tenant_id, sequence, event_type, trace_id, run_id, stage, status, counts, created_at"
                ") VALUES ('event-a1', 'task-a', 'tenant-a', 1, 'event', 'trace', 'run', 'stage', 'ok', '{}', CURRENT_TIMESTAMP)"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO task_events ("
                    "event_id, task_id, tenant_id, sequence, event_type, trace_id, run_id, stage, status, counts, created_at"
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
        insert_task(connection, "tenant-a", "task-a", "user-a")
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
        insert_task(connection, "tenant-a", "task-a", "user-a")
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
