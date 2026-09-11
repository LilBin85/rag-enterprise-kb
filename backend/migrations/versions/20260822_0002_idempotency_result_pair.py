"""约束幂等结果引用到同一摄取任务。

Revision ID: 20260822_0002
Revises: 20260822_0001
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0002"
down_revision = "20260822_0001"
branch_labels = None
depends_on = None


_RESULT_BINDING_COMPLETE = (
    "(result_reference IS NULL AND result_version_id IS NULL AND result_job_id IS NULL) "
    "OR (result_reference IS NOT NULL AND result_version_id IS NOT NULL AND result_job_id IS NOT NULL)"
)


def upgrade() -> None:
    # batch 模式会在 SQLite 重建表，在 PostgreSQL 上仍保持常规 ALTER 语义。
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.create_unique_constraint(
            "uq_ingestion_jobs_tenant_version_job", ["tenant_id", "version_id", "job_id"]
        )

    with op.batch_alter_table("idempotency_records") as batch_op:
        batch_op.create_foreign_key(
            "fk_idempotency_records_result_job_pair",
            "ingestion_jobs",
            ["tenant_id", "result_version_id", "result_job_id"],
            ["tenant_id", "version_id", "job_id"],
            ondelete="RESTRICT",
        )
        batch_op.create_check_constraint(
            "ck_idempotency_records_result_binding_complete", _RESULT_BINDING_COMPLETE
        )


def downgrade() -> None:
    with op.batch_alter_table("idempotency_records") as batch_op:
        batch_op.drop_constraint("ck_idempotency_records_result_binding_complete", type_="check")
        batch_op.drop_constraint("fk_idempotency_records_result_job_pair", type_="foreignkey")

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("uq_ingestion_jobs_tenant_version_job", type_="unique")
