"""为全局任务领取和过期清理补充复合索引。

Revision ID: 20260824_0009
Revises: 20260822_0008
Create Date: 2026-08-24
"""

from alembic import op


revision = "20260824_0009"
down_revision = "20260822_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_agent_tasks_claim",
        "agent_tasks",
        ["status", "available_at", "created_at"],
    )
    op.create_index(
        "ix_ingestion_jobs_expired",
        "ingestion_jobs",
        ["status", "lease_expires_at"],
    )
    op.create_index(
        "ix_tool_calls_sweeper",
        "tool_calls",
        ["status", "execution_deadline"],
    )


def downgrade() -> None:
    op.drop_index("ix_tool_calls_sweeper", table_name="tool_calls")
    op.drop_index("ix_ingestion_jobs_expired", table_name="ingestion_jobs")
    op.drop_index("ix_agent_tasks_claim", table_name="agent_tasks")
