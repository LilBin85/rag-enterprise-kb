"""建立企业身份、知识、任务、证据、审批与产物事实表。

Revision ID: 20260822_0001
Revises:
Create Date: 2026-08-22
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260822_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# SQLite 等适配器使用通用 JSON；生产 PostgreSQL 自动使用 JSONB。
JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.String(200), primary_key=True),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "principals",
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("roles", JSON_TYPE, nullable=False),
        sa.Column("attributes", JSON_TYPE, nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "user_id"),
    )
    op.create_table(
        "groups",
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("group_id", sa.String(200), nullable=False),
        sa.Column("name", sa.String(300), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "group_id"),
    )
    op.create_table(
        "group_memberships",
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("group_id", sa.String(200), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "group_id"],
            ["groups.tenant_id", "groups.group_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("tenant_id", "group_id", "user_id"),
    )
    op.create_table(
        "acl_grants",
        sa.Column("grant_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("resource_type", sa.String(80), nullable=False),
        sa.Column("resource_id", sa.String(200), nullable=False),
        sa.Column("subject_type", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(200), nullable=False),
        sa.Column("permissions", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_acl_grants_resource",
        "acl_grants",
        ["tenant_id", "resource_type", "resource_id"],
    )
    op.create_table(
        "documents",
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("document_id", sa.String(200), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("department", sa.String(200), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_by", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "created_by"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'active', 'failed', 'withdrawn')",
            name="ck_documents_status_valid",
        ),
        sa.PrimaryKeyConstraint("tenant_id", "document_id"),
    )
    op.create_table(
        "document_versions",
        sa.Column("version_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("document_id", sa.String(200), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("lifecycle_status", sa.String(32), nullable=False),
        sa.Column("source_object_key", sa.String(500), nullable=False),
        sa.Column("source_file_name", sa.String(500), nullable=False),
        sa.Column("content_hash", sa.String(80), nullable=False),
        sa.Column("index_version", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.Column("superseded_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.document_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "tenant_id", "document_id", "version", name="uq_document_version_number"
        ),
        sa.UniqueConstraint("tenant_id", "version_id", name="uq_document_versions_tenant_version"),
        sa.UniqueConstraint(
            "tenant_id", "source_object_key", name="uq_document_versions_tenant_source_object"
        ),
        sa.CheckConstraint("version > 0", name="ck_document_versions_version_positive"),
        sa.CheckConstraint(
            "lifecycle_status IN ('awaiting_object', 'pending', 'indexing', 'active', 'superseded', 'failed', 'withdrawn')",
            name="ck_document_versions_lifecycle_status_valid",
        ),
    )
    op.create_index(
        "ix_document_versions_active",
        "document_versions",
        ["tenant_id", "document_id", "lifecycle_status"],
    )
    op.create_index(
        "uq_document_versions_one_active",
        "document_versions",
        ["tenant_id", "document_id"],
        unique=True,
        sqlite_where=sa.text("lifecycle_status = 'active'"),
        postgresql_where=sa.text("lifecycle_status = 'active'"),
    )
    op.create_table(
        "chunk_metadata",
        sa.Column("chunk_id", sa.String(300), primary_key=True),
        sa.Column("version_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("document_id", sa.String(200), nullable=False),
        sa.Column("document_version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(80), nullable=False),
        sa.Column("source_locator", sa.String(500), nullable=False),
        sa.Column("source_file_name", sa.String(500), nullable=False),
        sa.Column("index_version", sa.String(80), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("document_version > 0", name="ck_chunk_metadata_document_version_positive"),
    )
    op.create_index(
        "ix_chunk_metadata_document",
        "chunk_metadata",
        ["tenant_id", "document_id", "document_version"],
    )
    op.create_table(
        "ingestion_jobs",
        sa.Column("job_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("version_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_owner", sa.String(200)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("lease_generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error_code", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_ingestion_jobs_attempts_nonnegative"),
        sa.CheckConstraint("lease_generation >= 0", name="ck_ingestion_jobs_lease_generation_nonnegative"),
        sa.CheckConstraint(
            "status IN ('awaiting_object', 'pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_ingestion_jobs_status_valid",
        ),
        sa.UniqueConstraint("tenant_id", "version_id", name="uq_ingestion_jobs_tenant_version"),
        sa.UniqueConstraint("tenant_id", "job_id", name="uq_ingestion_jobs_tenant_job"),
    )
    op.create_index("ix_ingestion_jobs_claim", "ingestion_jobs", ["status", "available_at"])
    op.create_table(
        "agent_tasks",
        sa.Column("task_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("task_mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("auth_context_ref", sa.String(500), nullable=False),
        sa.Column("current_plan_revision", sa.Integer(), nullable=False),
        sa.Column("current_step", sa.Integer(), nullable=False),
        sa.Column("budgets", JSON_TYPE, nullable=False),
        sa.Column("observation_refs", JSON_TYPE, nullable=False),
        sa.Column("pending_question", sa.Text()),
        sa.Column("pending_approval_id", sa.String(64)),
        sa.Column("repeated_action_fingerprints", JSON_TYPE, nullable=False),
        sa.Column("no_new_evidence_count", sa.Integer(), nullable=False),
        sa.Column("cancelled", sa.Boolean(), nullable=False),
        sa.Column("model_calls", sa.Integer(), nullable=False),
        sa.Column("tool_calls", sa.Integer(), nullable=False),
        sa.Column("replans", sa.Integer(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("token_usage", sa.Integer(), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True)),
        sa.Column("last_error_code", sa.String(100)),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("tenant_id", "task_id", name="uq_agent_tasks_tenant_task"),
        sa.CheckConstraint("current_plan_revision >= 0", name="ck_agent_tasks_plan_revision_nonnegative"),
        sa.CheckConstraint("current_step >= 0", name="ck_agent_tasks_current_step_nonnegative"),
        sa.CheckConstraint("model_calls >= 0", name="ck_agent_tasks_model_calls_nonnegative"),
        sa.CheckConstraint("tool_calls >= 0", name="ck_agent_tasks_tool_calls_nonnegative"),
        sa.CheckConstraint("replans >= 0", name="ck_agent_tasks_replans_nonnegative"),
        sa.CheckConstraint("consecutive_failures >= 0", name="ck_agent_tasks_failures_nonnegative"),
        sa.CheckConstraint("token_usage >= 0", name="ck_agent_tasks_token_usage_nonnegative"),
        sa.CheckConstraint("no_new_evidence_count >= 0", name="ck_agent_tasks_no_evidence_nonnegative"),
        sa.CheckConstraint("state_version > 0", name="ck_agent_tasks_state_version_positive"),
    )
    op.create_index(
        "ix_agent_tasks_owner_status", "agent_tasks", ["tenant_id", "user_id", "status"]
    )
    op.create_table(
        "task_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("stage", sa.String(80), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("tool_name", sa.String(100)),
        sa.Column("counts", JSON_TYPE, nullable=False),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column("token_usage", sa.Integer()),
        sa.Column("error_code", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("tenant_id", "task_id", "sequence", name="uq_task_events_tenant_task_sequence"),
        sa.CheckConstraint("sequence >= 0", name="ck_task_events_sequence_nonnegative"),
        sa.CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_task_events_duration_nonnegative"),
        sa.CheckConstraint("token_usage IS NULL OR token_usage >= 0", name="ck_task_events_token_usage_nonnegative"),
    )
    op.create_index("ix_task_events_task_sequence", "task_events", ["tenant_id", "task_id", "sequence"])
    op.create_table(
        "plan_revisions",
        sa.Column("plan_revision_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("plan", JSON_TYPE, nullable=False),
        sa.Column("plan_digest", sa.String(80), nullable=False),
        sa.Column("evidence_digest", sa.String(80), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("tenant_id", "task_id", "revision", name="uq_plan_revision_number"),
        sa.UniqueConstraint(
            "tenant_id", "task_id", "revision", "plan_digest", "evidence_digest",
            name="uq_plan_revision_binding",
        ),
        sa.CheckConstraint("revision > 0", name="ck_plan_revisions_revision_positive"),
        sa.CheckConstraint("schema_version > 0", name="ck_plan_revisions_schema_version_positive"),
    )
    op.create_table(
        "evidence_records",
        sa.Column("evidence_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("query_fingerprint", sa.String(80), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("document_id", sa.String(200), nullable=False),
        sa.Column("document_version", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(300), nullable=False),
        sa.Column("content_hash", sa.String(80), nullable=False),
        sa.Column("source_locator", sa.String(500), nullable=False),
        sa.Column("permission_granted", sa.Boolean(), nullable=False),
        sa.Column("supported_claim_ids", JSON_TYPE, nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("document_version > 0", name="ck_evidence_records_document_version_positive"),
    )
    op.create_index(
        "ix_evidence_records_task_query",
        "evidence_records",
        ["tenant_id", "task_id", "query_fingerprint"],
    )
    op.create_table(
        "approvals",
        sa.Column("approval_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("plan_revision", sa.Integer(), nullable=False),
        sa.Column("plan_digest", sa.String(80), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("tool_version", sa.String(80), nullable=False),
        sa.Column("arguments_hash", sa.String(80), nullable=False),
        sa.Column("principal_hash", sa.String(80), nullable=False),
        sa.Column("evidence_digest", sa.String(80), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("approved_by", sa.String(200)),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id", "plan_revision", "plan_digest", "evidence_digest"],
            [
                "plan_revisions.tenant_id",
                "plan_revisions.task_id",
                "plan_revisions.revision",
                "plan_revisions.plan_digest",
                "plan_revisions.evidence_digest",
            ],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "approved_by"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("tenant_id", "approval_id", name="uq_approvals_tenant_approval"),
        sa.UniqueConstraint(
            "tenant_id", "approval_id", "task_id", "plan_revision", "plan_digest",
            "evidence_digest", "tool_name", "tool_version", "arguments_hash", "principal_hash",
            name="uq_approvals_execution_binding",
        ),
        sa.CheckConstraint("plan_revision > 0", name="ck_approvals_plan_revision_positive"),
    )
    op.create_index("ix_approvals_task_status", "approvals", ["tenant_id", "task_id", "status"])
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
    op.create_table(
        "tool_calls",
        sa.Column("tool_call_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("approval_id", sa.String(64)),
        sa.Column("plan_revision", sa.Integer(), nullable=False),
        sa.Column("plan_digest", sa.String(80), nullable=False),
        sa.Column("evidence_digest", sa.String(80), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("tool_version", sa.String(80), nullable=False),
        sa.Column("arguments_hash", sa.String(80), nullable=False),
        sa.Column("principal_hash", sa.String(80), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("result_hash", sa.String(80)),
        sa.Column("risk_level", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_code", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id", "plan_revision", "plan_digest", "evidence_digest"],
            [
                "plan_revisions.tenant_id",
                "plan_revisions.task_id",
                "plan_revisions.revision",
                "plan_revisions.plan_digest",
                "plan_revisions.evidence_digest",
            ],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            [
                "tenant_id", "approval_id", "task_id", "plan_revision", "plan_digest",
                "evidence_digest", "tool_name", "tool_version", "arguments_hash", "principal_hash",
            ],
            [
                "approvals.tenant_id", "approvals.approval_id", "approvals.task_id",
                "approvals.plan_revision", "approvals.plan_digest", "approvals.evidence_digest",
                "approvals.tool_name", "approvals.tool_version", "approvals.arguments_hash",
                "approvals.principal_hash",
            ],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("plan_revision > 0", name="ck_tool_calls_plan_revision_positive"),
        sa.CheckConstraint("mode IN ('read', 'write')", name="ck_tool_calls_mode_valid"),
        sa.CheckConstraint(
            "mode = 'read' OR approval_id IS NOT NULL",
            name="ck_tool_calls_write_requires_approval",
        ),
        sa.UniqueConstraint(
            "tenant_id", "task_id", "tool_call_id", name="uq_tool_calls_tenant_task_call"
        ),
    )
    op.create_index("ix_tool_calls_task", "tool_calls", ["tenant_id", "task_id", "created_at"])
    op.create_table(
        "artifacts",
        sa.Column("artifact_id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("tool_call_id", sa.String(64), nullable=False),
        sa.Column("artifact_type", sa.String(80), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False),
        sa.Column("content_hash", sa.String(80), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "task_id", "tool_call_id"],
            ["tool_calls.tenant_id", "tool_calls.task_id", "tool_calls.tool_call_id"],
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_artifacts_task", "artifacts", ["tenant_id", "task_id", "created_at"])
    op.create_table(
        "idempotency_records",
        sa.Column("record_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(200), nullable=False),
        sa.Column("user_id", sa.String(200), nullable=False),
        sa.Column("scope", sa.String(100), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("request_hash", sa.String(80), nullable=False),
        sa.Column("result_version_id", sa.String(64)),
        sa.Column("result_job_id", sa.String(64)),
        sa.Column("result_reference", sa.String(500)),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "result_version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "result_job_id"],
            ["ingestion_jobs.tenant_id", "ingestion_jobs.job_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "tenant_id", "scope", "user_id", "idempotency_key",
            name="uq_idempotency_scope_user_key",
        ),
        sa.CheckConstraint(
            "status IN ('in_progress', 'succeeded', 'failed', 'unknown')",
            name="ck_idempotency_records_status_valid",
        ),
    )


def downgrade() -> None:
    op.drop_table("idempotency_records")
    op.drop_index("ix_artifacts_task", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index("ix_tool_calls_task", table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index("uq_approvals_active_binding", table_name="approvals")
    op.drop_index("ix_approvals_task_status", table_name="approvals")
    op.drop_table("approvals")
    op.drop_index("ix_evidence_records_task_query", table_name="evidence_records")
    op.drop_table("evidence_records")
    op.drop_table("plan_revisions")
    op.drop_index("ix_task_events_task_sequence", table_name="task_events")
    op.drop_table("task_events")
    op.drop_index("ix_agent_tasks_owner_status", table_name="agent_tasks")
    op.drop_table("agent_tasks")
    op.drop_index("ix_ingestion_jobs_claim", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_chunk_metadata_document", table_name="chunk_metadata")
    op.drop_table("chunk_metadata")
    op.drop_index("uq_document_versions_one_active", table_name="document_versions")
    op.drop_index("ix_document_versions_active", table_name="document_versions")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_index("ix_acl_grants_resource", table_name="acl_grants")
    op.drop_table("acl_grants")
    op.drop_table("group_memberships")
    op.drop_table("groups")
    op.drop_table("principals")
    op.drop_table("tenants")
