"""M2 协议事实、审批 checkpoint 绑定与产物正文生命周期。"""

from alembic import op
import sqlalchemy as sa


revision = "20260827_0013"
down_revision = "20260826_0012"
branch_labels = None
depends_on = None


_M1_CAPABILITY_NAME = "research-agent"
_M1_PROTOCOL_VERSION = "m1-fenced-v2"
_TASK_PROTOCOL_COMPLETE = (
    "length(trim(required_capability_name)) > 0 "
    "AND length(trim(required_protocol_version)) > 0"
)
_INTERRUPT_COMPLETE = (
    "(status = 'waiting_for_user' AND interrupt_kind = 'user_clarification' "
    "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0 "
    "AND interrupt_ref IS NULL) "
    "OR (status = 'waiting_for_approval' AND interrupt_kind = 'tool_approval' "
    "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0 "
    "AND interrupt_ref IS NOT NULL AND length(trim(interrupt_ref)) > 0) "
    "OR (status NOT IN ('waiting_for_user', 'waiting_for_approval') "
    "AND interrupt_kind IS NULL AND interrupt_summary IS NULL AND interrupt_ref IS NULL)"
)
_APPROVAL_CHECKPOINT_BINDING_COMPLETE = (
    "(thread_id IS NULL AND checkpoint_ns IS NULL AND checkpoint_id IS NULL "
    "AND interrupt_id IS NULL AND tool_call_id IS NULL AND checkpoint_binding_hash IS NULL) "
    "OR (thread_id IS NOT NULL AND length(trim(thread_id)) > 0 "
    "AND checkpoint_ns IS NOT NULL AND checkpoint_id IS NOT NULL "
    "AND length(trim(checkpoint_id)) > 0 AND interrupt_id IS NOT NULL "
    "AND length(trim(interrupt_id)) > 0 AND tool_call_id IS NOT NULL "
    "AND length(trim(tool_call_id)) > 0 AND checkpoint_binding_hash IS NOT NULL "
    "AND length(trim(checkpoint_binding_hash)) > 0)"
)
_APPROVAL_REJECT_AUDIT_COMPLETE = (
    "status <> 'rejected' OR checkpoint_binding_hash IS NULL "
    "OR (rejected_by IS NOT NULL AND rejected_at IS NOT NULL "
    "AND rejection_reason IS NOT NULL AND length(trim(rejection_reason)) > 0)"
)
_ARTIFACT_CONTENT_PURGE_BINDING = (
    "(content_purged_at IS NULL AND storage_key IS NOT NULL) "
    "OR (content_purged_at IS NOT NULL AND storage_key IS NULL "
    "AND content_expires_at IS NOT NULL)"
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


def _retire_m1_nonterminal_tasks(bind) -> None:
    if bind.dialect.name == "postgresql":
        content_expiry = "CURRENT_TIMESTAMP + INTERVAL '7 days'"
    elif bind.dialect.name == "sqlite":
        content_expiry = "datetime(CURRENT_TIMESTAMP, '+7 days')"
    else:
        raise RuntimeError("unsupported_database_dialect")
    bind.execute(
        sa.text(
            "UPDATE agent_tasks SET status = 'failed', cancelled = FALSE, "
            "last_error_code = 'task_protocol_retired', interrupt_kind = NULL, "
            "interrupt_summary = NULL, interrupt_ref = NULL, lease_owner = NULL, "
            "lease_expires_at = NULL, state_version = state_version + 1, "
            "content_expires_at = CASE WHEN task_mode = 'research' "
            f"THEN COALESCE(content_expires_at, {content_expiry}) ELSE content_expires_at END, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE status IN ('waiting_for_user', 'planning', 'running', 'waiting_for_approval')"
        )
    )


def _assert_downgrade_safe(bind) -> None:
    has_m2_facts = bind.scalar(
        sa.text(
            "SELECT 1 FROM agent_tasks "
            "WHERE required_capability_name <> :capability "
            "OR required_protocol_version <> :protocol "
            "OR interrupt_ref IS NOT NULL "
            "UNION ALL "
            "SELECT 1 FROM approvals "
            "WHERE thread_id IS NOT NULL OR checkpoint_ns IS NOT NULL "
            "OR checkpoint_id IS NOT NULL OR interrupt_id IS NOT NULL "
            "OR tool_call_id IS NOT NULL OR checkpoint_binding_hash IS NOT NULL "
            "OR rejected_by IS NOT NULL OR rejected_at IS NOT NULL OR rejection_reason IS NOT NULL "
            "UNION ALL "
            "SELECT 1 FROM artifacts "
            "WHERE content_expires_at IS NOT NULL OR content_purged_at IS NOT NULL "
            "OR storage_key IS NULL LIMIT 1"
        ),
        {"capability": _M1_CAPABILITY_NAME, "protocol": _M1_PROTOCOL_VERSION},
    )
    if has_m2_facts is not None:
        raise RuntimeError("m2_downgrade_data_present")


def upgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(
            sa.Column(
                "required_capability_name",
                sa.String(64),
                nullable=False,
                server_default=sa.text(f"'{_M1_CAPABILITY_NAME}'"),
            )
        )
        batch.add_column(
            sa.Column(
                "required_protocol_version",
                sa.String(64),
                nullable=False,
                server_default=sa.text(f"'{_M1_PROTOCOL_VERSION}'"),
            )
        )
        batch.add_column(sa.Column("interrupt_ref", sa.String(200), nullable=True))

    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE agent_tasks SET required_capability_name = :capability, "
            "required_protocol_version = :protocol"
        ),
        {"capability": _M1_CAPABILITY_NAME, "protocol": _M1_PROTOCOL_VERSION},
    )
    _retire_m1_nonterminal_tasks(bind)
    with op.batch_alter_table("agent_tasks") as batch:
        _drop_checks(batch, "agent_tasks", {"interrupt_complete"})
        batch.create_check_constraint("task_protocol_complete", _TASK_PROTOCOL_COMPLETE)
        batch.create_check_constraint("interrupt_complete", _INTERRUPT_COMPLETE)
    op.create_index(
        "ix_agent_tasks_protocol_claim",
        "agent_tasks",
        [
            "required_capability_name",
            "required_protocol_version",
            "status",
            "available_at",
            "created_at",
        ],
    )

    op.drop_index("uq_approvals_active_binding", table_name="approvals")
    with op.batch_alter_table("approvals") as batch:
        batch.add_column(sa.Column("thread_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("checkpoint_ns", sa.String(200), nullable=True))
        batch.add_column(sa.Column("checkpoint_id", sa.String(150), nullable=True))
        batch.add_column(sa.Column("interrupt_id", sa.String(100), nullable=True))
        batch.add_column(sa.Column("tool_call_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("checkpoint_binding_hash", sa.String(80), nullable=True))
        batch.add_column(sa.Column("rejected_by", sa.String(200), nullable=True))
        batch.add_column(sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("rejection_reason", sa.Text(), nullable=True))
        batch.create_foreign_key(
            "fk_approvals_rejected_by_principal",
            "principals",
            ["tenant_id", "rejected_by"],
            ["tenant_id", "user_id"],
            ondelete="RESTRICT",
        )
        batch.create_check_constraint(
            "checkpoint_binding_complete", _APPROVAL_CHECKPOINT_BINDING_COMPLETE
        )
        batch.create_check_constraint(
            "rejected_audit_complete", _APPROVAL_REJECT_AUDIT_COMPLETE
        )
    op.create_index(
        "ix_approvals_checkpoint_binding",
        "approvals",
        ["tenant_id", "task_id", "checkpoint_binding_hash", "status"],
    )
    op.create_index(
        "uq_approvals_active_legacy_binding",
        "approvals",
        [
            "tenant_id",
            "task_id",
            "plan_revision",
            "plan_digest",
            "evidence_digest",
            "tool_name",
            "tool_version",
            "arguments_hash",
            "principal_hash",
        ],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'approved') AND checkpoint_binding_hash IS NULL"),
        postgresql_where=sa.text("status IN ('pending', 'approved') AND checkpoint_binding_hash IS NULL"),
    )
    op.create_index(
        "uq_approvals_active_checkpoint_binding",
        "approvals",
        [
            "tenant_id",
            "task_id",
            "checkpoint_binding_hash",
            "tool_name",
            "tool_version",
            "arguments_hash",
            "evidence_digest",
            "principal_hash",
        ],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'approved') AND checkpoint_binding_hash IS NOT NULL"),
        postgresql_where=sa.text("status IN ('pending', 'approved') AND checkpoint_binding_hash IS NOT NULL"),
    )

    with op.batch_alter_table("artifacts") as batch:
        batch.add_column(sa.Column("content_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("content_purged_at", sa.DateTime(timezone=True), nullable=True))
        batch.alter_column("storage_key", existing_type=sa.String(500), nullable=True)
        batch.create_check_constraint(
            "content_purge_binding", _ARTIFACT_CONTENT_PURGE_BINDING
        )
    op.create_index(
        "ix_artifacts_content_retention",
        "artifacts",
        ["status", "content_expires_at", "content_purged_at"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    _assert_downgrade_safe(bind)

    op.drop_index("ix_artifacts_content_retention", table_name="artifacts")
    with op.batch_alter_table("artifacts") as batch:
        _drop_checks(batch, "artifacts", {"content_purge_binding"})
        batch.alter_column("storage_key", existing_type=sa.String(500), nullable=False)
        batch.drop_column("content_purged_at")
        batch.drop_column("content_expires_at")

    op.drop_index("uq_approvals_active_checkpoint_binding", table_name="approvals")
    op.drop_index("uq_approvals_active_legacy_binding", table_name="approvals")
    op.drop_index("ix_approvals_checkpoint_binding", table_name="approvals")
    with op.batch_alter_table("approvals") as batch:
        _drop_checks(
            batch,
            "approvals",
            {"checkpoint_binding_complete", "rejected_audit_complete"},
        )
        batch.drop_constraint("fk_approvals_rejected_by_principal", type_="foreignkey")
        batch.drop_column("rejection_reason")
        batch.drop_column("rejected_at")
        batch.drop_column("rejected_by")
        batch.drop_column("checkpoint_binding_hash")
        batch.drop_column("tool_call_id")
        batch.drop_column("interrupt_id")
        batch.drop_column("checkpoint_id")
        batch.drop_column("checkpoint_ns")
        batch.drop_column("thread_id")
    op.create_index(
        "uq_approvals_active_binding",
        "approvals",
        [
            "tenant_id",
            "task_id",
            "plan_revision",
            "plan_digest",
            "evidence_digest",
            "tool_name",
            "tool_version",
            "arguments_hash",
            "principal_hash",
        ],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'approved')"),
        postgresql_where=sa.text("status IN ('pending', 'approved')"),
    )

    op.drop_index("ix_agent_tasks_protocol_claim", table_name="agent_tasks")
    with op.batch_alter_table("agent_tasks") as batch:
        _drop_checks(batch, "agent_tasks", {"task_protocol_complete", "interrupt_complete"})
        batch.drop_column("interrupt_ref")
        batch.drop_column("required_protocol_version")
        batch.drop_column("required_capability_name")
        batch.create_check_constraint(
            "interrupt_complete",
            "status NOT IN ('waiting_for_user', 'waiting_for_approval') OR "
            "(interrupt_kind IS NOT NULL AND length(trim(interrupt_kind)) > 0 "
            "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0)",
        )
