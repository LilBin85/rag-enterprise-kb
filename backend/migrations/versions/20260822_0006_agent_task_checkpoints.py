"""为 AgentTask 增加独立运行租约与可恢复 checkpoint 状态。

Revision ID: 20260822_0006
Revises: 20260822_0005
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260822_0006"
down_revision = "20260822_0005"
branch_labels = None
depends_on = None


# SQLite 使用 JSON；生产 PostgreSQL 使用与 ORM 一致的 JSONB。
JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

_STATUS_VALID = (
    "status IN ('waiting_for_user', 'planning', 'running', 'waiting_for_approval', "
    "'succeeded', 'refused', 'failed', 'cancelled', 'unknown')"
)
_CURRENT_NODE_VALID = (
    "current_node IN ('load_task', 'classify_goal', 'check_clarification', 'update_plan', "
    "'select_action', 'execute_read_tool', 'record_evidence', 'evaluate_evidence', "
    "'await_approval', 'execute_approved_action', 'verify_action', 'generate_result', "
    "'terminal')"
)
_LEASE_COMPLETE = (
    "(lease_owner IS NULL AND lease_expires_at IS NULL) "
    "OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)"
)
_CANCELLED_STATUS_CONSISTENT = (
    "(cancelled AND status = 'cancelled') "
    "OR (NOT cancelled AND status <> 'cancelled')"
)


def upgrade() -> None:
    # 历史终态统一为 Stage 4 的 succeeded；约束加入前同步旧 cancelled 标志，
    # 防止无租约的历史记录因命名迁移而被误判为运行中。
    op.execute(sa.text("UPDATE agent_tasks SET status = 'succeeded' WHERE status = 'completed'"))
    op.execute(
        sa.text(
            "UPDATE agent_tasks "
            "SET cancelled = CASE WHEN status = 'cancelled' THEN TRUE ELSE FALSE END"
        )
    )

    # batch 模式保证 SQLite 重建表时保留既有复合外键和命名约束；PostgreSQL 使用 ALTER。
    with op.batch_alter_table("agent_tasks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "current_node",
                sa.String(80),
                nullable=False,
                server_default=sa.text("'load_task'"),
            )
        )
        batch_op.add_column(sa.Column("lease_owner", sa.String(200), nullable=True))
        batch_op.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(
            sa.Column("lease_generation", sa.BigInteger(), nullable=False, server_default=sa.text("0"))
        )
        batch_op.add_column(
            sa.Column(
                "available_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            )
        )
        batch_op.add_column(
            sa.Column("evidence_refs", JSON_TYPE, nullable=False, server_default=sa.text("'[]'"))
        )
        batch_op.add_column(
            sa.Column("artifact_refs", JSON_TYPE, nullable=False, server_default=sa.text("'[]'"))
        )
        batch_op.add_column(
            sa.Column("query_fingerprints", JSON_TYPE, nullable=False, server_default=sa.text("'[]'"))
        )
        batch_op.add_column(
            sa.Column("result_fingerprints", JSON_TYPE, nullable=False, server_default=sa.text("'[]'"))
        )
        batch_op.add_column(sa.Column("checkpoint_digest", sa.String(80), nullable=True))
        batch_op.create_check_constraint("ck_agent_tasks_status_valid", _STATUS_VALID)
        batch_op.create_check_constraint("ck_agent_tasks_current_node_valid", _CURRENT_NODE_VALID)
        batch_op.create_check_constraint(
            "ck_agent_tasks_lease_generation_nonnegative", "lease_generation >= 0"
        )
        batch_op.create_check_constraint("ck_agent_tasks_lease_complete", _LEASE_COMPLETE)
        batch_op.create_check_constraint(
            "ck_agent_tasks_lease_status_valid",
            "lease_owner IS NULL OR status IN ('planning', 'running')",
        )
        batch_op.create_check_constraint(
            "ck_agent_tasks_cancelled_status_consistent", _CANCELLED_STATUS_CONSISTENT
        )
        batch_op.create_check_constraint(
            "ck_agent_tasks_waiting_question_complete",
            "status <> 'waiting_for_user' "
            "OR (pending_question IS NOT NULL AND length(trim(pending_question)) > 0)",
        )
        batch_op.create_check_constraint(
            "ck_agent_tasks_waiting_approval_complete",
            "status <> 'waiting_for_approval' "
            "OR (pending_approval_id IS NOT NULL AND length(trim(pending_approval_id)) > 0)",
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch_op:
        batch_op.drop_constraint("ck_agent_tasks_waiting_approval_complete", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_waiting_question_complete", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_cancelled_status_consistent", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_lease_status_valid", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_lease_complete", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_lease_generation_nonnegative", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_current_node_valid", type_="check")
        batch_op.drop_constraint("ck_agent_tasks_status_valid", type_="check")
        batch_op.drop_column("checkpoint_digest")
        batch_op.drop_column("result_fingerprints")
        batch_op.drop_column("query_fingerprints")
        batch_op.drop_column("artifact_refs")
        batch_op.drop_column("evidence_refs")
        batch_op.drop_column("available_at")
        batch_op.drop_column("lease_generation")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("lease_owner")
        batch_op.drop_column("current_node")
