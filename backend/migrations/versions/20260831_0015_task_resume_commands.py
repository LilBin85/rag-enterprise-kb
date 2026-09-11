"""耐久恢复命令 inbox 与 task 侧低敏 resume binding。"""

from alembic import op
import sqlalchemy as sa


revision = "20260831_0015"
down_revision = "20260829_0014"
branch_labels = None
depends_on = None


_OLD_INTERRUPT_COMPLETE = (
    "(status = 'waiting_for_user' AND interrupt_kind = 'user_clarification' "
    "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0 "
    "AND interrupt_ref IS NULL) "
    "OR (status = 'waiting_for_approval' AND interrupt_kind = 'tool_approval' "
    "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0 "
    "AND interrupt_ref IS NOT NULL AND length(trim(interrupt_ref)) > 0) "
    "OR (status NOT IN ('waiting_for_user', 'waiting_for_approval') "
    "AND interrupt_kind IS NULL AND interrupt_summary IS NULL AND interrupt_ref IS NULL)"
)
_TRANSITIONAL_INTERRUPT_COMPLETE = (
    "(status = 'waiting_for_user' AND interrupt_kind = 'user_clarification' "
    "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0 "
    "AND interrupt_ref IS NULL AND ((interrupt_checkpoint_ns IS NULL "
    "AND interrupt_checkpoint_id IS NULL AND interrupt_id IS NULL "
    "AND interrupt_approval_id IS NULL) OR (interrupt_checkpoint_ns IS NOT NULL "
    "AND interrupt_checkpoint_id IS NOT NULL AND length(trim(interrupt_checkpoint_id)) > 0 "
    "AND interrupt_id IS NOT NULL AND length(trim(interrupt_id)) > 0 "
    "AND interrupt_approval_id IS NULL))) "
    "OR (status = 'waiting_for_approval' AND interrupt_kind = 'tool_approval' "
    "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0 "
    "AND interrupt_ref IS NOT NULL AND length(trim(interrupt_ref)) > 0 "
    "AND ((interrupt_checkpoint_ns IS NULL AND interrupt_checkpoint_id IS NULL "
    "AND interrupt_id IS NULL AND interrupt_approval_id IS NULL) "
    "OR (interrupt_checkpoint_ns IS NOT NULL "
    "AND interrupt_checkpoint_id IS NOT NULL AND length(trim(interrupt_checkpoint_id)) > 0 "
    "AND interrupt_id IS NOT NULL AND length(trim(interrupt_id)) > 0 "
    "AND interrupt_approval_id IS NOT NULL AND length(trim(interrupt_approval_id)) > 0))) "
    "OR (status NOT IN ('waiting_for_user', 'waiting_for_approval') "
    "AND interrupt_kind IS NULL AND interrupt_summary IS NULL AND interrupt_ref IS NULL "
    "AND interrupt_checkpoint_ns IS NULL AND interrupt_checkpoint_id IS NULL "
    "AND interrupt_id IS NULL AND interrupt_approval_id IS NULL)"
)


def _constraint_key(table_name: str, name: str) -> str:
    prefix = f"ck_{table_name}_"
    while name.startswith(prefix):
        name = name[len(prefix) :]
    return name


def _drop_checks(batch, table_name: str, names: set[str]) -> None:
    for item in sa.inspect(op.get_bind()).get_check_constraints(table_name):
        name = str(item.get("name") or "")
        if name and _constraint_key(table_name, name) in names:
            batch.drop_constraint(_constraint_key(table_name, name), type_="check")


def _assert_downgrade_safe(bind) -> None:
    if bind.scalar(sa.text("SELECT 1 FROM task_resume_commands LIMIT 1")) is not None:
        raise RuntimeError("task_resume_commands_downgrade_data_present")
    if bind.scalar(
        sa.text(
            "SELECT 1 FROM agent_tasks WHERE active_resume_command_id IS NOT NULL "
            "OR interrupt_checkpoint_ns IS NOT NULL OR interrupt_checkpoint_id IS NOT NULL "
            "OR interrupt_id IS NOT NULL OR interrupt_approval_id IS NOT NULL LIMIT 1"
        )
    ) is not None:
        raise RuntimeError("task_resume_binding_downgrade_data_present")


def upgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(sa.Column("interrupt_checkpoint_ns", sa.String(200), nullable=True))
        batch.add_column(sa.Column("interrupt_checkpoint_id", sa.String(150), nullable=True))
        batch.add_column(sa.Column("interrupt_id", sa.String(100), nullable=True))
        batch.add_column(sa.Column("interrupt_approval_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("active_resume_command_id", sa.String(64), nullable=True))
        _drop_checks(batch, "agent_tasks", {"interrupt_complete"})
        batch.create_check_constraint("interrupt_complete", _TRANSITIONAL_INTERRUPT_COMPLETE)
        batch.create_check_constraint(
            "active_resume_command_complete",
            "active_resume_command_id IS NULL OR "
            "(status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
        )

    op.create_table(
        "task_resume_commands",
        sa.Column("command_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("actor_principal_id", sa.String(200), nullable=False),
        sa.Column("actor_kind", sa.String(32), nullable=False),
        sa.Column("expected_task_state_version", sa.Integer(), nullable=False),
        sa.Column("expected_interrupt_checkpoint_ns", sa.String(200), nullable=False),
        sa.Column("expected_interrupt_checkpoint_id", sa.String(150), nullable=False),
        sa.Column("expected_interrupt_id", sa.String(100), nullable=False),
        sa.Column("expected_protocol_version", sa.String(64), nullable=False),
        sa.Column("approval_id", sa.String(64), nullable=True),
        sa.Column("decision", sa.String(32), nullable=True),
        sa.Column("payload_body", sa.Text(), nullable=True),
        sa.Column("payload_sha256", sa.String(80), nullable=False),
        sa.Column("resume_target_key", sa.String(80), nullable=False),
        sa.Column("idempotency_key", sa.String(80), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_owner", sa.String(200), nullable=True),
        sa.Column("lease_generation", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_task_state_version", sa.Integer(), nullable=True),
        sa.Column("claimed_task_lease_generation", sa.BigInteger(), nullable=True),
        sa.Column("applied_checkpoint_id", sa.String(150), nullable=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(100), nullable=True),
        sa.Column("body_purged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            name="fk_task_resume_commands_tenant_id_agent_tasks",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "actor_principal_id"],
            ["principals.tenant_id", "principals.user_id"],
            name="fk_task_resume_commands_tenant_id_principals",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "approval_id"],
            ["approvals.tenant_id", "approvals.approval_id"],
            name="fk_task_resume_commands_tenant_id_approvals",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_task_resume_commands_tenant_idempotency"
        ),
        sa.UniqueConstraint(
            "tenant_id", "resume_target_key", name="uq_task_resume_commands_tenant_target"
        ),
        sa.CheckConstraint("kind IN ('user_clarification', 'tool_approval')", name="kind_valid"),
        sa.CheckConstraint("actor_kind IN ('user', 'service')", name="actor_kind_valid"),
        sa.CheckConstraint("expected_task_state_version > 0", name="expected_state_version_positive"),
        sa.CheckConstraint("attempt_count >= 0", name="attempt_count_nonnegative"),
        sa.CheckConstraint("lease_generation >= 0", name="lease_generation_nonnegative"),
        sa.CheckConstraint("expires_at > created_at", name="expires_after_created"),
        sa.CheckConstraint("next_attempt_at <= expires_at", name="next_attempt_before_expiry"),
        sa.CheckConstraint(
            "(kind = 'user_clarification' AND approval_id IS NULL AND decision IS NULL "
            "AND ((status IN ('pending', 'claimed') AND payload_body IS NOT NULL "
            "AND length(trim(payload_body)) > 0) OR (status IN ('consumed', 'failed', 'expired') "
            "AND payload_body IS NULL))) OR (kind = 'tool_approval' "
            "AND approval_id IS NOT NULL AND length(trim(approval_id)) > 0 "
            "AND decision IN ('approved', 'rejected') "
            "AND (payload_body IS NULL OR length(trim(payload_body)) > 0))",
            name="kind_payload_complete",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND lease_owner IS NULL AND lease_expires_at IS NULL "
            "AND claimed_at IS NULL AND claimed_task_state_version IS NULL "
            "AND claimed_task_lease_generation IS NULL) OR (status = 'claimed' "
            "AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL AND claimed_at IS NOT NULL "
            "AND claimed_task_state_version IS NOT NULL AND claimed_task_state_version > 0 "
            "AND claimed_task_lease_generation IS NOT NULL AND claimed_task_lease_generation > 0) "
            "OR (status IN ('consumed', 'failed', 'expired') AND lease_owner IS NULL "
            "AND lease_expires_at IS NULL)",
            name="lease_complete",
        ),
        sa.CheckConstraint(
            "(applied_checkpoint_id IS NULL AND applied_at IS NULL) OR "
            "(applied_checkpoint_id IS NOT NULL AND length(trim(applied_checkpoint_id)) > 0 "
            "AND applied_at IS NOT NULL)",
            name="apply_marker_complete",
        ),
        sa.CheckConstraint(
            "applied_checkpoint_id IS NULL OR status IN ('claimed', 'consumed')",
            name="apply_marker_status_valid",
        ),
        sa.CheckConstraint(
            "(status = 'consumed' AND consumed_at IS NOT NULL AND failed_at IS NULL "
            "AND expired_at IS NULL) OR (status = 'failed' AND consumed_at IS NULL "
            "AND failed_at IS NOT NULL AND expired_at IS NULL) OR (status = 'expired' "
            "AND consumed_at IS NULL AND failed_at IS NULL AND expired_at IS NOT NULL) "
            "OR (status IN ('pending', 'claimed') AND consumed_at IS NULL "
            "AND failed_at IS NULL AND expired_at IS NULL)",
            name="terminal_timestamp_complete",
        ),
        sa.CheckConstraint(
            "status IN ('consumed', 'failed', 'expired') OR body_purged_at IS NULL",
            name="body_purge_status_valid",
        ),
        sa.CheckConstraint(
            "(status IN ('consumed', 'failed', 'expired') AND payload_body IS NULL "
            "AND body_purged_at IS NOT NULL) OR (status IN ('pending', 'claimed') "
            "AND body_purged_at IS NULL)",
            name="body_purge_complete",
        ),
        sa.CheckConstraint(
            "(status IN ('pending', 'claimed') AND failure_code IS NULL) OR "
            "(status IN ('consumed', 'failed', 'expired') AND (failure_code IS NULL "
            "OR length(trim(failure_code)) > 0))",
            name="failure_code_valid",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'claimed', 'consumed', 'failed', 'expired')",
            name="status_valid",
        ),
    )
    op.create_index(
        "ix_task_resume_commands_claim",
        "task_resume_commands",
        ["status", "next_attempt_at", "created_at"],
    )
    op.create_index(
        "ix_task_resume_commands_recovery",
        "task_resume_commands",
        ["status", "lease_expires_at", "created_at"],
    )
    op.create_index(
        "ix_task_resume_commands_task_status",
        "task_resume_commands",
        ["tenant_id", "task_id", "status"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    _assert_downgrade_safe(bind)

    op.drop_index("ix_task_resume_commands_task_status", table_name="task_resume_commands")
    op.drop_index("ix_task_resume_commands_recovery", table_name="task_resume_commands")
    op.drop_index("ix_task_resume_commands_claim", table_name="task_resume_commands")
    op.drop_table("task_resume_commands")

    with op.batch_alter_table("agent_tasks") as batch:
        _drop_checks(
            batch,
            "agent_tasks",
            {"interrupt_complete", "active_resume_command_complete"},
        )
        batch.drop_column("active_resume_command_id")
        batch.drop_column("interrupt_approval_id")
        batch.drop_column("interrupt_id")
        batch.drop_column("interrupt_checkpoint_id")
        batch.drop_column("interrupt_checkpoint_ns")
        batch.create_check_constraint("interrupt_complete", _OLD_INTERRUPT_COMPLETE)
