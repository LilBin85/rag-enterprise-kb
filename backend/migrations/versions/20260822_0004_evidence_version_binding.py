"""将证据账本绑定到精确文档版本和块元数据。

Revision ID: 20260822_0004
Revises: 20260822_0003
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0004"
down_revision = "20260822_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chunk_metadata") as batch_op:
        batch_op.create_unique_constraint(
            "uq_chunk_metadata_tenant_version_chunk",
            ["tenant_id", "version_id", "chunk_id"],
        )

    with op.batch_alter_table("evidence_records") as batch_op:
        batch_op.add_column(sa.Column("version_id", sa.String(64), nullable=True))

    op.execute(
        sa.text(
            """
            UPDATE evidence_records
            SET version_id = (
                SELECT chunk_metadata.version_id
                FROM chunk_metadata
                WHERE chunk_metadata.tenant_id = evidence_records.tenant_id
                  AND chunk_metadata.document_id = evidence_records.document_id
                  AND chunk_metadata.document_version = evidence_records.document_version
                  AND chunk_metadata.chunk_id = evidence_records.chunk_id
                  AND chunk_metadata.content_hash = evidence_records.content_hash
            )
            """
        )
    )

    with op.batch_alter_table("evidence_records") as batch_op:
        batch_op.alter_column("version_id", existing_type=sa.String(64), nullable=False)
        batch_op.create_foreign_key(
            "fk_evidence_records_chunk_metadata",
            "chunk_metadata",
            ["tenant_id", "version_id", "chunk_id"],
            ["tenant_id", "version_id", "chunk_id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    with op.batch_alter_table("evidence_records") as batch_op:
        batch_op.drop_constraint("fk_evidence_records_chunk_metadata", type_="foreignkey")
        batch_op.drop_column("version_id")

    with op.batch_alter_table("chunk_metadata") as batch_op:
        batch_op.drop_constraint("uq_chunk_metadata_tenant_version_chunk", type_="unique")
