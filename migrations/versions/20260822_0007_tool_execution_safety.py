"""为工具调用、审批和产物加入幂等与终态安全约束。

Revision ID: 20260822_0007
Revises: 20260822_0006
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0007"
down_revision = "20260822_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 先加入可空键，随后为存量 write 调用安全生成可追溯且全局唯一的 legacy 键。
    with op.batch_alter_table("tool_calls") as batch_op:
        batch_op.add_column(sa.Column("idempotency_key", sa.String(200), nullable=True))
        batch_op.add_column(
            sa.Column(
                "execution_deadline", sa.DateTime(timezone=True), nullable=True,
                server_default=sa.func.now(),
            )
        )
        batch_op.add_column(sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True))

    op.execute(
        sa.text(
            "UPDATE tool_calls "
            "SET idempotency_key = 'legacy:' || tool_call_id "
            "WHERE mode = 'write' AND (idempotency_key IS NULL OR length(trim(idempotency_key)) = 0)"
        )
    )
    # 不会伪造成功绑定：历史 succeeded 缺少回执一律降为 unknown。
    op.execute(
        sa.text(
            "UPDATE tool_calls SET status = 'unknown' "
            "WHERE status NOT IN ('proposed', 'running', 'succeeded', 'failed', 'unknown') "
            "OR (status = 'succeeded' AND "
            "(result_hash IS NULL OR length(trim(result_hash)) = 0 OR completed_at IS NULL))"
        )
    )
    op.execute(
        sa.text(
            "UPDATE tool_calls SET risk_level = 'high' "
            "WHERE risk_level NOT IN ('low', 'medium', 'high')"
        )
    )
    op.execute(
        sa.text(
            "UPDATE tool_calls SET execution_deadline = COALESCE(completed_at, created_at) "
            "WHERE execution_deadline IS NULL"
        )
    )
    op.execute(
        sa.text(
            "UPDATE tool_calls SET verified_at = completed_at "
            "WHERE status = 'succeeded' AND verified_at IS NULL"
        )
    )
    op.execute(
        sa.text(
            "UPDATE tool_calls SET completed_at = COALESCE(completed_at, created_at), "
            "error_code = COALESCE(NULLIF(trim(error_code), ''), 'legacy_outcome_unknown') "
            "WHERE status IN ('failed', 'unknown')"
        )
    )

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        op.execute(sa.text("UPDATE approvals SET status = 'expired', expires_at = datetime(created_at, '+1 second') WHERE expires_at <= created_at"))
    else:
        op.execute(sa.text("UPDATE approvals SET status = 'expired', expires_at = created_at + INTERVAL '1 second' WHERE expires_at <= created_at"))

    # Approval 不存在 unknown 合法终态；未知或不完整批准按过期失败关闭，绝不补造批准人。
    op.execute(
        sa.text(
            "UPDATE approvals SET status = 'expired' "
            "WHERE status NOT IN ('pending', 'approved', 'rejected', 'expired', 'executed') "
            "OR (status = 'approved' AND (approved_by IS NULL OR approved_at IS NULL)) "
            "OR (status = 'pending' AND (approved_by IS NOT NULL OR approved_at IS NOT NULL))"
        )
    )
    op.execute(
        sa.text(
            "UPDATE artifacts SET status = 'unknown' "
            "WHERE status NOT IN ('creating', 'ready', 'failed', 'unknown') "
            "OR (status = 'ready' AND (content_hash IS NULL OR length(trim(content_hash)) = 0))"
        )
    )

    # batch 模式使 SQLite 与 PostgreSQL 的唯一键、CHECK 语义可重复执行且保持一致。
    with op.batch_alter_table("tool_calls") as batch_op:
        batch_op.alter_column("execution_deadline", nullable=False)
        batch_op.create_unique_constraint(
            "uq_tool_calls_task_tool_idempotency",
            ["tenant_id", "task_id", "tool_name", "idempotency_key"],
        )
        batch_op.create_check_constraint(
            "write_requires_idempotency_key",
            "mode = 'read' OR (idempotency_key IS NOT NULL AND length(trim(idempotency_key)) > 0)",
        )
        batch_op.create_check_constraint(
            "status_valid",
            "status IN ('proposed', 'running', 'succeeded', 'failed', 'unknown')",
        )
        batch_op.create_check_constraint(
            "risk_level_valid", "risk_level IN ('low', 'medium', 'high')"
        )
        batch_op.create_check_constraint(
            "succeeded_result_complete",
            "status <> 'succeeded' OR (result_hash IS NOT NULL AND length(trim(result_hash)) > 0 "
            "AND verified_at IS NOT NULL AND completed_at IS NOT NULL AND error_code IS NULL)",
        )
        batch_op.create_check_constraint(
            "running_not_completed", "status <> 'running' OR (completed_at IS NULL "
            "AND verified_at IS NULL AND result_hash IS NULL AND error_code IS NULL)"
        )
        batch_op.create_check_constraint(
            "running_has_deadline", "status <> 'running' OR execution_deadline IS NOT NULL"
        )
        batch_op.create_check_constraint(
            "failure_terminal_complete", "status NOT IN ('failed', 'unknown') OR "
            "(completed_at IS NOT NULL AND error_code IS NOT NULL AND length(trim(error_code)) > 0)"
        )
        batch_op.create_index("ix_tool_calls_task_status", ["tenant_id", "task_id", "status"])
        batch_op.create_index(
            "uq_tool_calls_consumed_approval",
            ["tenant_id", "approval_id"],
            unique=True,
            sqlite_where=sa.text("approval_id IS NOT NULL"),
            postgresql_where=sa.text("approval_id IS NOT NULL"),
        )

    with op.batch_alter_table("approvals") as batch_op:
        batch_op.create_check_constraint(
            "status_valid",
            "status IN ('pending', 'approved', 'rejected', 'expired', 'executed')",
        )
        batch_op.create_check_constraint(
            "approved_binding_complete",
            "status NOT IN ('approved', 'executed') OR "
            "(approved_by IS NOT NULL AND approved_at IS NOT NULL)",
        )
        batch_op.create_check_constraint(
            "pending_binding_empty",
            "status <> 'pending' OR (approved_by IS NULL AND approved_at IS NULL)",
        )
        batch_op.create_check_constraint(
            "expires_after_created", "expires_at > created_at"
        )

    with op.batch_alter_table("artifacts") as batch_op:
        batch_op.create_unique_constraint(
            "uq_artifacts_tenant_storage_key", ["tenant_id", "storage_key"]
        )
        batch_op.create_check_constraint(
            "status_valid", "status IN ('creating', 'ready', 'failed', 'unknown')"
        )
        batch_op.create_check_constraint(
            "ready_content_hash_complete",
            "status <> 'ready' OR (content_hash IS NOT NULL AND length(content_hash) = 71 "
            "AND substr(content_hash, 1, 7) = 'sha256:')",
        )
        batch_op.create_check_constraint(
            "artifact_type_valid",
            "artifact_type IN ('report', 'work_item_draft', 'checklist', 'notification_draft')",
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    artifact_checks = {item["name"] for item in inspector.get_check_constraints("artifacts")}
    with op.batch_alter_table("artifacts") as batch_op:
        if "ck_artifacts_artifact_type_valid" in artifact_checks:
            batch_op.drop_constraint("artifact_type_valid", type_="check")
        batch_op.drop_constraint("ready_content_hash_complete", type_="check")
        batch_op.drop_constraint("status_valid", type_="check")
        batch_op.drop_constraint("uq_artifacts_tenant_storage_key", type_="unique")

    with op.batch_alter_table("approvals") as batch_op:
        batch_op.drop_constraint("expires_after_created", type_="check")
        batch_op.drop_constraint("pending_binding_empty", type_="check")
        batch_op.drop_constraint("approved_binding_complete", type_="check")
        batch_op.drop_constraint("status_valid", type_="check")

    inspector = sa.inspect(op.get_bind())
    tool_checks = {item["name"] for item in inspector.get_check_constraints("tool_calls")}
    tool_columns = {item["name"] for item in inspector.get_columns("tool_calls")}
    with op.batch_alter_table("tool_calls") as batch_op:
        batch_op.drop_index("uq_tool_calls_consumed_approval")
        batch_op.drop_index("ix_tool_calls_task_status")
        if "ck_tool_calls_failure_terminal_complete" in tool_checks:
            batch_op.drop_constraint("failure_terminal_complete", type_="check")
        if "ck_tool_calls_running_has_deadline" in tool_checks:
            batch_op.drop_constraint("running_has_deadline", type_="check")
        batch_op.drop_constraint("running_not_completed", type_="check")
        batch_op.drop_constraint("succeeded_result_complete", type_="check")
        batch_op.drop_constraint("risk_level_valid", type_="check")
        batch_op.drop_constraint("status_valid", type_="check")
        batch_op.drop_constraint("write_requires_idempotency_key", type_="check")
        batch_op.drop_constraint("uq_tool_calls_task_tool_idempotency", type_="unique")
        if "execution_deadline" in tool_columns:
            batch_op.drop_column("execution_deadline")
        if "verified_at" in tool_columns:
            batch_op.drop_column("verified_at")
        batch_op.drop_column("idempotency_key")
