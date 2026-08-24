"""为 PostgreSQL/Qdrant 投影增加持久世代和周期收敛租约。

Revision ID: 20260822_0005
Revises: 20260822_0004
Create Date: 2026-08-22
"""

from alembic import op
import sqlalchemy as sa


revision = "20260822_0005"
down_revision = "20260822_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    missing_active_receipts = op.get_bind().execute(
        sa.text(
            """
            SELECT COUNT(*)
            FROM document_versions
            LEFT JOIN ingestion_jobs
              ON ingestion_jobs.tenant_id = document_versions.tenant_id
             AND ingestion_jobs.version_id = document_versions.version_id
            WHERE document_versions.lifecycle_status = 'active'
              AND (
                  ingestion_jobs.job_id IS NULL
                  OR ingestion_jobs.indexed_chunk_count IS NULL
                  OR ingestion_jobs.index_manifest_digest IS NULL
                  OR ingestion_jobs.index_completed_at IS NULL
              )
            """
        )
    ).scalar_one()
    if missing_active_receipts:
        raise RuntimeError("projection_migration_active_receipt_missing")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("projection_generation", sa.BigInteger(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("applied_projection_generation", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("applied_projection_version_id", sa.String(64), nullable=True))
        batch_op.add_column(
            sa.Column(
                "projection_reconcile_after",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            )
        )
        batch_op.add_column(sa.Column("projection_reconcile_owner", sa.String(200), nullable=True))
        batch_op.add_column(
            sa.Column("projection_reconcile_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(sa.Column("projection_reconcile_error_code", sa.String(100), nullable=True))
        batch_op.add_column(
            sa.Column("projection_verified_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "projection_reconcile_lease_generation",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.create_check_constraint(
            "ck_documents_projection_generation_nonnegative", "projection_generation >= 0"
        )
        batch_op.create_check_constraint(
            "ck_documents_applied_projection_generation_positive",
            "applied_projection_generation IS NULL OR applied_projection_generation > 0",
        )
        batch_op.create_check_constraint(
            "ck_documents_applied_projection_binding_complete",
            "(applied_projection_generation IS NULL AND applied_projection_version_id IS NULL) "
            "OR (applied_projection_generation IS NOT NULL AND applied_projection_version_id IS NOT NULL)",
        )
        batch_op.create_check_constraint(
            "ck_documents_applied_projection_not_ahead",
            "applied_projection_generation IS NULL "
            "OR applied_projection_generation <= projection_generation",
        )
        batch_op.create_check_constraint(
            "ck_documents_projection_reconcile_lease_generation_nonnegative",
            "projection_reconcile_lease_generation >= 0",
        )
        batch_op.create_check_constraint(
            "ck_documents_projection_reconcile_lease_complete",
            "(projection_reconcile_owner IS NULL AND projection_reconcile_expires_at IS NULL) "
            "OR (projection_reconcile_owner IS NOT NULL AND projection_reconcile_expires_at IS NOT NULL)",
        )
        batch_op.create_index(
            "ix_documents_projection_reconcile", ["projection_reconcile_after"], unique=False
        )

    with op.batch_alter_table("document_versions") as batch_op:
        batch_op.add_column(sa.Column("desired_projection_generation", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("applied_projection_generation", sa.BigInteger(), nullable=True))
        batch_op.create_check_constraint(
            "ck_document_versions_desired_projection_generation_positive",
            "desired_projection_generation IS NULL OR desired_projection_generation > 0",
        )
        batch_op.create_check_constraint(
            "ck_document_versions_applied_projection_generation_positive",
            "applied_projection_generation IS NULL OR applied_projection_generation > 0",
        )
        batch_op.create_check_constraint(
            "ck_document_versions_applied_projection_not_ahead",
            "applied_projection_generation IS NULL "
            "OR (desired_projection_generation IS NOT NULL "
            "AND applied_projection_generation <= desired_projection_generation)",
        )

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.add_column(sa.Column("projection_generation", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("projection_lease_generation", sa.BigInteger(), nullable=True))
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_projection_generation_positive",
            "projection_generation IS NULL OR projection_generation > 0",
        )
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_projection_lease_generation_positive",
            "projection_lease_generation IS NULL OR projection_lease_generation > 0",
        )
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_projection_lease_binding_complete",
            "(projection_generation IS NULL AND projection_lease_generation IS NULL) "
            "OR (projection_generation IS NOT NULL AND projection_lease_generation IS NOT NULL)",
        )

    # 存量业务版本号只用于初始化单调世代；不回填 applied，避免把没有世代字段
    # 的旧 Qdrant 点伪报为已验证。Reconciler 收敛前检索会失败关闭。
    op.execute(
        sa.text(
            """
            UPDATE documents
            SET projection_generation = (
                SELECT MAX(document_versions.version)
                FROM document_versions
                WHERE document_versions.tenant_id = documents.tenant_id
                  AND document_versions.document_id = documents.document_id
                  AND document_versions.lifecycle_status IN ('active', 'superseded')
            )
            WHERE EXISTS (
                SELECT 1
                FROM document_versions
                WHERE document_versions.tenant_id = documents.tenant_id
                  AND document_versions.document_id = documents.document_id
                  AND document_versions.lifecycle_status IN ('active', 'superseded')
            )
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE document_versions
            SET desired_projection_generation = version
            WHERE lifecycle_status IN ('active', 'superseded')
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE ingestion_jobs
            SET projection_generation = (
                    SELECT document_versions.version
                    FROM document_versions
                    WHERE document_versions.tenant_id = ingestion_jobs.tenant_id
                      AND document_versions.version_id = ingestion_jobs.version_id
                ),
                projection_lease_generation = CASE
                    WHEN lease_generation > 0 THEN lease_generation
                    ELSE 1
                END
            WHERE indexed_chunk_count IS NOT NULL
              AND index_manifest_digest IS NOT NULL
              AND index_completed_at IS NOT NULL
              AND EXISTS (
                  SELECT 1
                  FROM document_versions
                  WHERE document_versions.tenant_id = ingestion_jobs.tenant_id
                    AND document_versions.version_id = ingestion_jobs.version_id
                    AND document_versions.lifecycle_status IN (
                        'active', 'superseded', 'failed', 'withdrawn'
                    )
              )
            """
        )
    )

    with op.batch_alter_table("document_versions") as batch_op:
        batch_op.create_check_constraint(
            "ck_document_versions_publishable_projection_generation",
            "lifecycle_status NOT IN ('active', 'superseded') "
            "OR desired_projection_generation IS NOT NULL",
        )
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.create_check_constraint(
            "ck_ingestion_jobs_succeeded_projection_complete",
            "status != 'succeeded' OR (projection_generation IS NOT NULL "
            "AND projection_lease_generation IS NOT NULL)",
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_constraint(
            "ck_ingestion_jobs_succeeded_projection_complete", type_="check"
        )
        batch_op.drop_constraint(
            "ck_ingestion_jobs_projection_lease_binding_complete", type_="check"
        )
        batch_op.drop_constraint(
            "ck_ingestion_jobs_projection_lease_generation_positive", type_="check"
        )
        batch_op.drop_constraint("ck_ingestion_jobs_projection_generation_positive", type_="check")
        batch_op.drop_column("projection_lease_generation")
        batch_op.drop_column("projection_generation")

    with op.batch_alter_table("document_versions") as batch_op:
        batch_op.drop_constraint(
            "ck_document_versions_publishable_projection_generation", type_="check"
        )
        batch_op.drop_constraint(
            "ck_document_versions_applied_projection_not_ahead", type_="check"
        )
        batch_op.drop_constraint(
            "ck_document_versions_applied_projection_generation_positive", type_="check"
        )
        batch_op.drop_constraint(
            "ck_document_versions_desired_projection_generation_positive", type_="check"
        )
        batch_op.drop_column("applied_projection_generation")
        batch_op.drop_column("desired_projection_generation")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_index("ix_documents_projection_reconcile")
        batch_op.drop_constraint(
            "ck_documents_projection_reconcile_lease_complete", type_="check"
        )
        batch_op.drop_constraint(
            "ck_documents_projection_reconcile_lease_generation_nonnegative", type_="check"
        )
        batch_op.drop_constraint("ck_documents_applied_projection_not_ahead", type_="check")
        batch_op.drop_constraint(
            "ck_documents_applied_projection_binding_complete", type_="check"
        )
        batch_op.drop_constraint(
            "ck_documents_applied_projection_generation_positive", type_="check"
        )
        batch_op.drop_constraint("ck_documents_projection_generation_nonnegative", type_="check")
        batch_op.drop_column("projection_reconcile_lease_generation")
        batch_op.drop_column("projection_reconcile_expires_at")
        batch_op.drop_column("projection_verified_at")
        batch_op.drop_column("projection_reconcile_error_code")
        batch_op.drop_column("projection_reconcile_owner")
        batch_op.drop_column("projection_reconcile_after")
        batch_op.drop_column("applied_projection_version_id")
        batch_op.drop_column("applied_projection_generation")
        batch_op.drop_column("projection_generation")
