"""为任务 checkpoint 增加不可变 pending action 对象绑定。

Revision ID: 20260822_0008
Revises: 20260822_0007
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0008"
down_revision = "20260822_0007"
branch_labels = None
depends_on = None


_PENDING_ACTION_COMPLETE = (
    "(pending_action_ref IS NULL AND pending_action_hash IS NULL "
    "AND pending_action_kind IS NULL AND pending_action_expires_at IS NULL) OR "
    "(pending_action_ref IS NOT NULL AND length(trim(pending_action_ref)) > 0 "
    "AND pending_action_hash IS NOT NULL AND length(pending_action_hash) = 71 "
    "AND substr(pending_action_hash, 1, 7) = 'sha256:' "
    "AND pending_action_kind IS NOT NULL AND pending_action_expires_at IS NOT NULL)"
)


def upgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch_op:
        batch_op.add_column(sa.Column("pending_action_ref", sa.String(500), nullable=True))
        batch_op.add_column(sa.Column("pending_action_hash", sa.String(80), nullable=True))
        batch_op.add_column(sa.Column("pending_action_kind", sa.String(32), nullable=True))
        batch_op.add_column(
            sa.Column(
                "pending_action_expires_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )

    # 旧运行时没有动作正文，不能为存量执行节点伪造对象或批准绑定。
    # 只读动作安全失败；可能已越过写执行边界的任务标记 unknown，禁止重放。
    op.execute(
        sa.text(
            "UPDATE agent_tasks SET "
            "status = CASE "
            "WHEN current_node = 'execute_approved_action' THEN 'unknown' "
            "ELSE 'failed' END, "
            "current_node = 'terminal', pending_approval_id = NULL, "
            "lease_owner = NULL, lease_expires_at = NULL, "
            "last_error_code = 'legacy_pending_action_unrecoverable', "
            "state_version = state_version + 1, checkpoint_digest = NULL, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE status NOT IN ('succeeded', 'refused', 'failed', 'cancelled', 'unknown') "
            "AND (current_node IN ('execute_read_tool', 'await_approval', "
            "'execute_approved_action') OR status = 'waiting_for_approval' "
            "OR pending_approval_id IS NOT NULL)"
        )
    )
    op.execute(
        sa.text(
            "UPDATE agent_tasks SET pending_approval_id = NULL "
            "WHERE status IN ('succeeded', 'refused', 'failed', 'cancelled', 'unknown') "
            "AND pending_approval_id IS NOT NULL"
        )
    )

    with op.batch_alter_table("agent_tasks") as batch_op:
        batch_op.create_check_constraint(
            "pending_action_complete", _PENDING_ACTION_COMPLETE
        )
        batch_op.create_check_constraint(
            "pending_action_kind_valid",
            "pending_action_kind IS NULL OR "
            "pending_action_kind IN ('search', 'call_tool', 'propose_action')",
        )
        batch_op.create_check_constraint(
            "pending_action_node_complete",
            "current_node NOT IN ('execute_read_tool', 'await_approval', "
            "'execute_approved_action') OR pending_action_ref IS NOT NULL",
        )
        batch_op.create_check_constraint(
            "waiting_approval_action_valid",
            "status <> 'waiting_for_approval' "
            "OR pending_action_kind = 'propose_action'",
        )
        batch_op.create_check_constraint(
            "pending_approval_action_valid",
            "pending_approval_id IS NULL OR pending_action_kind = 'propose_action'",
        )
        batch_op.create_index(
            "ix_agent_tasks_pending_action_expiry",
            ["pending_action_expires_at"],
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch_op:
        batch_op.drop_index("ix_agent_tasks_pending_action_expiry")
        batch_op.drop_constraint("pending_approval_action_valid", type_="check")
        batch_op.drop_constraint("waiting_approval_action_valid", type_="check")
        batch_op.drop_constraint("pending_action_node_complete", type_="check")
        batch_op.drop_constraint("pending_action_kind_valid", type_="check")
        batch_op.drop_constraint("pending_action_complete", type_="check")
        batch_op.drop_column("pending_action_expires_at")
        batch_op.drop_column("pending_action_kind")
        batch_op.drop_column("pending_action_hash")
        batch_op.drop_column("pending_action_ref")

