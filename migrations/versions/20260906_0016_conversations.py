"""统一会话入口与持久记忆的关系事实。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260906_0016"
down_revision = "20260831_0015"
branch_labels = None
depends_on = None


JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
_ACTIVE_TURN_STATUSES = "'routing', 'planning', 'running', 'waiting_for_user', 'waiting_for_approval'"


def _assert_downgrade_safe(bind) -> None:
    for table_name in ("conversation_messages", "conversation_turns", "conversations"):
        if bind.scalar(sa.text(f"SELECT 1 FROM {table_name} LIMIT 1")) is not None:
            raise RuntimeError("conversation_downgrade_data_present")
    if bind.scalar(
        sa.text(
            "SELECT 1 FROM agent_tasks WHERE conversation_id IS NOT NULL "
            "OR conversation_turn_id IS NOT NULL LIMIT 1"
        )
    ) is not None:
        raise RuntimeError("conversation_task_binding_downgrade_data_present")
    if bind.scalar(
        sa.text("SELECT 1 FROM tenants WHERE conversation_retention_days IS NOT NULL LIMIT 1")
    ) is not None:
        raise RuntimeError("conversation_retention_downgrade_data_present")


def upgrade() -> None:
    with op.batch_alter_table("tenants") as batch:
        batch.add_column(sa.Column("conversation_retention_days", sa.Integer(), nullable=True))
        batch.create_check_constraint(
            "conversation_retention_days_positive",
            "conversation_retention_days IS NULL OR conversation_retention_days > 0",
        )

    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_through_sequence", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("memory_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("memory_hash", sa.String(80), nullable=True),
        sa.Column("summary_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleting_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("tenant_id", "conversation_id", name="uq_conversations_tenant_conversation"),
        sa.CheckConstraint("status IN ('active', 'deleting', 'deleted')", name="status_valid"),
        sa.CheckConstraint("summary_through_sequence >= 0", name="summary_sequence_nonnegative"),
        sa.CheckConstraint("memory_version >= 0", name="memory_version_nonnegative"),
        sa.CheckConstraint("state_version > 0", name="state_version_positive"),
        sa.CheckConstraint(
            "(status = 'active' AND deleting_at IS NULL AND deleted_at IS NULL) OR "
            "(status = 'deleting' AND deleting_at IS NOT NULL AND deleted_at IS NULL) OR "
            "(status = 'deleted' AND deleting_at IS NOT NULL AND deleted_at IS NOT NULL)",
            name="deletion_binding_complete",
        ),
    )
    op.create_index(
        "ix_conversations_owner_updated",
        "conversations",
        ["tenant_id", "user_id", "status", "updated_at"],
    )
    op.create_index(
        "ix_conversations_content_retention",
        "conversations",
        ["status", "content_expires_at"],
    )

    op.create_table(
        "conversation_turns",
        sa.Column("turn_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("submission_kind", sa.String(16), nullable=False),
        sa.Column("requested_mode", sa.String(32), nullable=False),
        sa.Column("selected_mode", sa.String(32), nullable=True),
        sa.Column("route_reason_code", sa.String(100), nullable=True),
        sa.Column("resolved_input", sa.Text(), nullable=True),
        sa.Column("resolved_input_hash", sa.String(80), nullable=True),
        sa.Column("task_id", sa.String(64), nullable=True),
        sa.Column("task_state_version", sa.Integer(), nullable=True),
        sa.Column("context_hash", sa.String(80), nullable=True),
        sa.Column("context_snapshot", JSON_TYPE, nullable=False, server_default=sa.text("'[]'")),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("request_hash", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("tenant_id", "conversation_id", "turn_id", name="uq_conversation_turns_tenant_conversation_turn"),
        sa.UniqueConstraint("tenant_id", "conversation_id", "sequence", name="uq_conversation_turns_sequence"),
        sa.UniqueConstraint("tenant_id", "user_id", "conversation_id", "idempotency_key", name="uq_conversation_turns_idempotency"),
        sa.UniqueConstraint("tenant_id", "task_id", name="uq_conversation_turns_task"),
        sa.CheckConstraint("sequence > 0", name="sequence_positive"),
        sa.CheckConstraint("task_state_version IS NULL OR task_state_version > 0", name="task_state_version_positive"),
        sa.CheckConstraint("submission_kind IN ('new', 'clarification')", name="submission_kind_valid"),
        sa.CheckConstraint("requested_mode IN ('auto', 'answer', 'research', 'controlled_action')", name="requested_mode_valid"),
        sa.CheckConstraint("selected_mode IS NULL OR selected_mode IN ('answer', 'research', 'controlled_action')", name="selected_mode_valid"),
        sa.CheckConstraint(
            "(status = 'routing' AND selected_mode IS NULL AND task_id IS NULL) OR "
            "(status <> 'routing' AND selected_mode IS NOT NULL)",
            name="routing_binding_complete",
        ),
        sa.CheckConstraint(
            "status IN ('routing', 'planning', 'running', 'waiting_for_user', "
            "'waiting_for_approval', 'succeeded', 'refused', 'failed', 'cancelled')",
            name="status_valid",
        ),
    )
    op.create_index(
        "uq_conversation_turns_one_active",
        "conversation_turns",
        ["tenant_id", "conversation_id"],
        unique=True,
        sqlite_where=sa.text(f"status IN ({_ACTIVE_TURN_STATUSES})"),
        postgresql_where=sa.text(f"status IN ({_ACTIVE_TURN_STATUSES})"),
    )
    op.create_index(
        "ix_conversation_turns_owner_updated",
        "conversation_turns",
        ["tenant_id", "conversation_id", "updated_at"],
    )

    op.create_table(
        "conversation_messages",
        sa.Column("message_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("turn_id", sa.String(64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(80), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=True),
        sa.Column("request_hash", sa.String(80), nullable=True),
        sa.Column("task_id", sa.String(64), nullable=True),
        sa.Column("evidence_ids", JSON_TYPE, nullable=False, server_default=sa.text("'[]'")),
        sa.Column("artifacts", JSON_TYPE, nullable=False, server_default=sa.text("'[]'")),
        sa.Column("redacted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_purged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "conversation_id", "turn_id"],
            ["conversation_turns.tenant_id", "conversation_turns.conversation_id", "conversation_turns.turn_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("tenant_id", "conversation_id", "sequence", name="uq_conversation_messages_sequence"),
        sa.UniqueConstraint("tenant_id", "conversation_id", "idempotency_key", name="uq_conversation_messages_idempotency"),
        sa.CheckConstraint("sequence > 0", name="sequence_positive"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="role_valid"),
        sa.CheckConstraint(
            "(content IS NOT NULL AND content_purged_at IS NULL) OR "
            "(content IS NULL AND content_purged_at IS NOT NULL)",
            name="content_purge_binding_complete",
        ),
    )
    op.create_index(
        "ix_conversation_messages_page",
        "conversation_messages",
        ["tenant_id", "conversation_id", "sequence"],
    )

    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(sa.Column("conversation_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("conversation_turn_id", sa.String(64), nullable=True))
        batch.create_check_constraint(
            "conversation_binding_complete",
            "(conversation_id IS NULL AND conversation_turn_id IS NULL) OR "
            "(conversation_id IS NOT NULL AND conversation_turn_id IS NOT NULL)",
        )
        batch.create_foreign_key(
            "fk_agent_tasks_conversation_turn",
            "conversation_turns",
            ["tenant_id", "conversation_id", "conversation_turn_id"],
            ["tenant_id", "conversation_id", "turn_id"],
            ondelete="RESTRICT",
        )
        batch.create_index(
            "ix_agent_tasks_conversation_turn",
            ["tenant_id", "conversation_id", "conversation_turn_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    _assert_downgrade_safe(bind)

    with op.batch_alter_table("agent_tasks") as batch:
        batch.drop_index("ix_agent_tasks_conversation_turn")
        batch.drop_constraint("fk_agent_tasks_conversation_turn", type_="foreignkey")
        batch.drop_constraint("conversation_binding_complete", type_="check")
        batch.drop_column("conversation_turn_id")
        batch.drop_column("conversation_id")

    op.drop_index("ix_conversation_messages_page", table_name="conversation_messages")
    op.drop_table("conversation_messages")
    op.drop_index("ix_conversation_turns_owner_updated", table_name="conversation_turns")
    op.drop_index("uq_conversation_turns_one_active", table_name="conversation_turns")
    op.drop_table("conversation_turns")
    op.drop_index("ix_conversations_content_retention", table_name="conversations")
    op.drop_index("ix_conversations_owner_updated", table_name="conversations")
    op.drop_table("conversations")

    with op.batch_alter_table("tenants") as batch:
        batch.drop_constraint("conversation_retention_days_positive", type_="check")
        batch.drop_column("conversation_retention_days")
