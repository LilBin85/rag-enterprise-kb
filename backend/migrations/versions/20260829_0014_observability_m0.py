"""M0 关联 ID 与异步 Trace Link 持久化上下文。"""

from alembic import op
import sqlalchemy as sa


revision = "20260829_0014"
down_revision = "20260827_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("task_events") as batch:
        batch.alter_column(
            "trace_id",
            new_column_name="correlation_id",
            existing_type=sa.String(64),
            existing_nullable=False,
        )
        batch.add_column(sa.Column("otel_trace_id", sa.String(32), nullable=True))
        batch.add_column(sa.Column("otel_span_id", sa.String(16), nullable=True))
        # 创建、取消和 HTTP 恢复不是 Worker run，不能伪造 run_id。
        batch.alter_column("run_id", existing_type=sa.String(64), nullable=True)

    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(sa.Column("pending_traceparent", sa.String(55), nullable=True))
        batch.add_column(sa.Column("pending_tracestate", sa.Text(), nullable=True))

    with op.batch_alter_table("ingestion_jobs") as batch:
        batch.add_column(sa.Column("pending_traceparent", sa.String(55), nullable=True))
        batch.add_column(sa.Column("pending_tracestate", sa.Text(), nullable=True))


def downgrade() -> None:
    # 旧 schema 要求非空 run_id；回退到旧混合语义但不丢失业务关联值。
    op.execute(
        sa.text(
            "UPDATE task_events SET run_id = correlation_id WHERE run_id IS NULL"
        )
    )

    with op.batch_alter_table("ingestion_jobs") as batch:
        batch.drop_column("pending_tracestate")
        batch.drop_column("pending_traceparent")

    with op.batch_alter_table("agent_tasks") as batch:
        batch.drop_column("pending_tracestate")
        batch.drop_column("pending_traceparent")

    with op.batch_alter_table("task_events") as batch:
        batch.alter_column("run_id", existing_type=sa.String(64), nullable=False)
        batch.drop_column("otel_span_id")
        batch.drop_column("otel_trace_id")
        batch.alter_column(
            "correlation_id",
            new_column_name="trace_id",
            existing_type=sa.String(64),
            existing_nullable=False,
        )
