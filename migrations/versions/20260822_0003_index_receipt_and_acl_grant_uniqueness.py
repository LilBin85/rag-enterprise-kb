"""持久化索引回执并令 ACL 授权可幂等创建。

Revision ID: 20260822_0003
Revises: 20260822_0002
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0003"
down_revision = "20260822_0002"
branch_labels = None
depends_on = None


_INDEX_RECEIPT_COMPLETE = (
    "(indexed_chunk_count IS NULL AND index_manifest_digest IS NULL AND index_completed_at IS NULL) "
    "OR (indexed_chunk_count IS NOT NULL AND indexed_chunk_count > 0 "
    "AND index_manifest_digest IS NOT NULL AND index_completed_at IS NOT NULL)"
)


def upgrade() -> None:
    # batch 模式会在 SQLite 重建表，在 PostgreSQL 上仍保持常规 ALTER 语义。
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.add_column(sa.Column("indexed_chunk_count", sa.Integer()))
        batch_op.add_column(sa.Column("index_manifest_digest", sa.String(80)))
        batch_op.add_column(sa.Column("index_completed_at", sa.DateTime(timezone=True)))
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_index_receipt_complete", _INDEX_RECEIPT_COMPLETE
        )

    with op.batch_alter_table("acl_grants") as batch_op:
        batch_op.create_unique_constraint(
            "uq_acl_grants_resource_subject",
            ["tenant_id", "resource_type", "resource_id", "subject_type", "subject_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("acl_grants") as batch_op:
        batch_op.drop_constraint("uq_acl_grants_resource_subject", type_="unique")

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint("ck_ingestion_jobs_index_receipt_complete", type_="check")
        batch_op.drop_column("index_completed_at")
        batch_op.drop_column("index_manifest_digest")
        batch_op.drop_column("indexed_chunk_count")
