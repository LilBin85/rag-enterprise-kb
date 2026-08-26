"""企业知识任务智能体的关系数据库事实模型。

所有业务表都显式带 ``tenant_id``。Qdrant 点只是这些事实的检索投影，不参与
版本号分配、任务状态、审批或产物幂等性的最终判定。
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from rag_kb.infrastructure.database.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


# SQLite 等开发适配器使用通用 JSON；生产 PostgreSQL 自动使用 JSONB。
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class TenantRow(Base):
    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class PrincipalRow(Base):
    __tablename__ = "principals"

    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    roles: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class GroupRow(Base):
    __tablename__ = "groups"

    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), primary_key=True
    )
    group_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)


class GroupMembershipRow(Base):
    __tablename__ = "group_memberships"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "group_id"],
            ["groups.tenant_id", "groups.group_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="CASCADE",
        ),
    )

    tenant_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    group_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(200), primary_key=True)


class ACLGrantRow(Base):
    __tablename__ = "acl_grants"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "resource_type",
            "resource_id",
            "subject_type",
            "subject_id",
            name="uq_acl_grants_resource_subject",
        ),
        Index("ix_acl_grants_resource", "tenant_id", "resource_type", "resource_id"),
    )

    grant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(200), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(200), nullable=False)
    permissions: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "created_by"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('pending', 'active', 'failed', 'withdrawn')",
            name="status_valid",
        ),
        CheckConstraint("projection_generation >= 0", name="projection_generation_nonnegative"),
        CheckConstraint(
            "applied_projection_generation IS NULL OR applied_projection_generation > 0",
            name="applied_projection_generation_positive",
        ),
        CheckConstraint(
            "(applied_projection_generation IS NULL AND applied_projection_version_id IS NULL) "
            "OR (applied_projection_generation IS NOT NULL AND applied_projection_version_id IS NOT NULL)",
            name="applied_projection_binding_complete",
        ),
        CheckConstraint(
            "applied_projection_generation IS NULL "
            "OR applied_projection_generation <= projection_generation",
            name="applied_projection_not_ahead",
        ),
        CheckConstraint(
            "projection_reconcile_lease_generation >= 0",
            name="projection_reconcile_lease_generation_nonnegative",
        ),
        CheckConstraint(
            "(projection_reconcile_owner IS NULL AND projection_reconcile_expires_at IS NULL) "
            "OR (projection_reconcile_owner IS NOT NULL AND projection_reconcile_expires_at IS NOT NULL)",
            name="projection_reconcile_lease_complete",
        ),
        Index("ix_documents_projection_reconcile", "projection_reconcile_after"),
    )

    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    department: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    projection_generation: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    applied_projection_generation: Mapped[int | None] = mapped_column(BigInteger)
    applied_projection_version_id: Mapped[str | None] = mapped_column(String(64))
    projection_reconcile_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    projection_reconcile_owner: Mapped[str | None] = mapped_column(String(200))
    projection_reconcile_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    projection_reconcile_error_code: Mapped[str | None] = mapped_column(String(100))
    projection_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    projection_reconcile_lease_generation: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    created_by: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class DocumentVersionRow(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.document_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("tenant_id", "document_id", "version", name="uq_document_version_number"),
        UniqueConstraint("tenant_id", "version_id", name="uq_document_versions_tenant_version"),
        UniqueConstraint(
            "tenant_id", "source_object_key", name="uq_document_versions_tenant_source_object"
        ),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint(
            "desired_projection_generation IS NULL OR desired_projection_generation > 0",
            name="desired_projection_generation_positive",
        ),
        CheckConstraint(
            "applied_projection_generation IS NULL OR applied_projection_generation > 0",
            name="applied_projection_generation_positive",
        ),
        CheckConstraint(
            "applied_projection_generation IS NULL "
            "OR (desired_projection_generation IS NOT NULL "
            "AND applied_projection_generation <= desired_projection_generation)",
            name="applied_projection_not_ahead",
        ),
        CheckConstraint(
            "lifecycle_status NOT IN ('active', 'superseded') "
            "OR desired_projection_generation IS NOT NULL",
            name="publishable_projection_generation",
        ),
        CheckConstraint(
            "lifecycle_status IN ('awaiting_object', 'pending', 'indexing', 'active', 'superseded', 'failed', 'withdrawn')",
            name="lifecycle_status_valid",
        ),
        Index("ix_document_versions_active", "tenant_id", "document_id", "lifecycle_status"),
        Index(
            "uq_document_versions_one_active",
            "tenant_id",
            "document_id",
            unique=True,
            sqlite_where=text("lifecycle_status = 'active'"),
            postgresql_where=text("lifecycle_status = 'active'"),
        ),
    )

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    document_id: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    lifecycle_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    source_object_key: Mapped[str] = mapped_column(String(500), nullable=False)
    source_file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    index_version: Mapped[str] = mapped_column(String(80), nullable=False)
    desired_projection_generation: Mapped[int | None] = mapped_column(BigInteger)
    applied_projection_generation: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChunkMetadataRow(Base):
    __tablename__ = "chunk_metadata"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("document_version > 0", name="document_version_positive"),
        UniqueConstraint(
            "tenant_id", "version_id", "chunk_id", name="uq_chunk_metadata_tenant_version_chunk"
        ),
        Index("ix_chunk_metadata_document", "tenant_id", "document_id", "document_version"),
    )

    chunk_id: Mapped[str] = mapped_column(String(300), primary_key=True)
    version_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    document_id: Mapped[str] = mapped_column(String(200), nullable=False)
    document_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    source_locator: Mapped[str] = mapped_column(String(500), nullable=False)
    source_file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    index_version: Mapped[str] = mapped_column(String(80), nullable=False)


class IngestionJobRow(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("attempts >= 0", name="attempts_nonnegative"),
        CheckConstraint("lease_generation >= 0", name="lease_generation_nonnegative"),
        CheckConstraint(
            "projection_generation IS NULL OR projection_generation > 0",
            name="projection_generation_positive",
        ),
        CheckConstraint(
            "projection_lease_generation IS NULL OR projection_lease_generation > 0",
            name="projection_lease_generation_positive",
        ),
        CheckConstraint(
            "(projection_generation IS NULL AND projection_lease_generation IS NULL) "
            "OR (projection_generation IS NOT NULL AND projection_lease_generation IS NOT NULL)",
            name="projection_lease_binding_complete",
        ),
        CheckConstraint(
            "status != 'succeeded' OR (projection_generation IS NOT NULL "
            "AND projection_lease_generation IS NOT NULL)",
            name="succeeded_projection_complete",
        ),
        CheckConstraint(
            "status IN ('awaiting_object', 'pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="status_valid",
        ),
        UniqueConstraint("tenant_id", "version_id", name="uq_ingestion_jobs_tenant_version"),
        UniqueConstraint("tenant_id", "job_id", name="uq_ingestion_jobs_tenant_job"),
        UniqueConstraint(
            "tenant_id", "version_id", "job_id", name="uq_ingestion_jobs_tenant_version_job"
        ),
        CheckConstraint(
            "(indexed_chunk_count IS NULL AND index_manifest_digest IS NULL AND index_completed_at IS NULL) "
            "OR (indexed_chunk_count IS NOT NULL AND indexed_chunk_count > 0 "
            "AND index_manifest_digest IS NOT NULL AND index_completed_at IS NOT NULL)",
            name="index_receipt_complete",
        ),
        Index("ix_ingestion_jobs_claim", "status", "available_at"),
        Index("ix_ingestion_jobs_expired", "status", "lease_expires_at"),
    )

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    version_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    lease_owner: Mapped[str | None] = mapped_column(String(200))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    projection_generation: Mapped[int | None] = mapped_column(BigInteger)
    projection_lease_generation: Mapped[int | None] = mapped_column(BigInteger)
    error_code: Mapped[str | None] = mapped_column(String(100))
    indexed_chunk_count: Mapped[int | None] = mapped_column(Integer)
    index_manifest_digest: Mapped[str | None] = mapped_column(String(80))
    index_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class AgentTaskRow(Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        UniqueConstraint("tenant_id", "task_id", name="uq_agent_tasks_tenant_task"),
        CheckConstraint("state_version > 0", name="state_version_positive"),
        CheckConstraint(
            "status IN ('waiting_for_user', 'planning', 'running', 'waiting_for_approval', "
            "'succeeded', 'refused', 'failed', 'cancelled', 'unknown')",
            name="status_valid",
        ),
        CheckConstraint("lease_generation >= 0", name="lease_generation_nonnegative"),
        CheckConstraint(
            "(lease_owner IS NULL AND lease_expires_at IS NULL) "
            "OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="lease_complete",
        ),
        CheckConstraint(
            "lease_owner IS NULL OR status IN ('planning', 'running')",
            name="lease_status_valid",
        ),
        CheckConstraint(
            "(cancelled AND status = 'cancelled') "
            "OR (NOT cancelled AND status <> 'cancelled')",
            name="cancelled_status_consistent",
        ),
        CheckConstraint(
            "status NOT IN ('waiting_for_user', 'waiting_for_approval') OR "
            "(interrupt_kind IS NOT NULL AND length(trim(interrupt_kind)) > 0 "
            "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0)",
            name="interrupt_complete",
        ),
        CheckConstraint(
            "CASE WHEN task_mode <> 'research' OR status NOT IN ('succeeded', 'refused') "
            "THEN 1 WHEN result_purged_at IS NULL THEN CASE WHEN result_kind = 'research' "
            "AND result_ref IS NOT NULL AND length(trim(result_ref)) > 0 "
            "AND result_hash IS NOT NULL AND length(trim(result_hash)) > 0 THEN 1 ELSE 0 END "
            "ELSE CASE WHEN result_kind = 'research' AND result_ref IS NULL "
            "AND result_hash IS NOT NULL AND length(trim(result_hash)) > 0 THEN 1 ELSE 0 END END = 1",
            name="research_result_complete",
        ),
        CheckConstraint(
            "CASE WHEN task_mode = 'research' AND status IN "
            "('succeeded', 'refused', 'failed', 'cancelled', 'unknown') "
            "THEN CASE WHEN content_expires_at IS NOT NULL THEN 1 ELSE 0 END ELSE 1 END = 1",
            name="research_content_expiry",
        ),
        Index("ix_agent_tasks_owner_status", "tenant_id", "user_id", "status"),
        Index(
            "ix_agent_tasks_claim",
            "status",
            "available_at",
            "created_at",
        ),
        Index(
            "ix_agent_tasks_retention",
            "task_mode",
            "status",
            "content_expires_at",
        ),
    )

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(200), nullable=False)
    task_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    auth_context_ref: Mapped[str] = mapped_column(String(500), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(200))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_generation: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    budgets: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    artifact_refs: Mapped[list[str]] = mapped_column(
        JSON_TYPE, nullable=False, default=list, server_default=text("'[]'")
    )
    cancelled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    # M1 业务字段：中断摘要与最终结果引用；不保存消息、Todo 或 interrupt 载荷。
    interrupt_kind: Mapped[str | None] = mapped_column(String(32))
    interrupt_summary: Mapped[str | None] = mapped_column(Text)
    result_kind: Mapped[str | None] = mapped_column(String(32))
    result_ref: Mapped[str | None] = mapped_column(String(500))
    result_hash: Mapped[str | None] = mapped_column(String(80))
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # 保留策略执行标记（M1_FIX_PLAN 修复 5）：幂等重扫依据。
    checkpoint_purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class AgentCapabilityLeaseRow(Base):
    __tablename__ = "agent_capability_leases"
    __table_args__ = (
        Index(
            "ix_agent_capability_leases_expiry",
            "capability_name",
            "protocol_version",
            "lease_expires_at",
        ),
    )

    capability_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TaskEventRow(Base):
    __tablename__ = "task_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("tenant_id", "task_id", "sequence", name="uq_task_events_tenant_task_sequence"),
        CheckConstraint("sequence >= 0", name="sequence_nonnegative"),
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="duration_nonnegative"),
        CheckConstraint("token_usage IS NULL OR token_usage >= 0", name="token_usage_nonnegative"),
        Index("ix_task_events_task_sequence", "tenant_id", "task_id", "sequence"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(100))
    counts: Mapped[dict[str, int]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    token_usage: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class PlanRevisionRow(Base):
    __tablename__ = "plan_revisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("tenant_id", "task_id", "revision", name="uq_plan_revision_number"),
        UniqueConstraint(
            "tenant_id", "task_id", "revision", "plan_digest", "evidence_digest",
            name="uq_plan_revision_binding",
        ),
        CheckConstraint("revision > 0", name="revision_positive"),
        CheckConstraint("schema_version > 0", name="schema_version_positive"),
    )

    plan_revision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    plan_digest: Mapped[str] = mapped_column(String(80), nullable=False)
    evidence_digest: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ToolCallRow(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
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
        CheckConstraint("plan_revision > 0", name="plan_revision_positive"),
        CheckConstraint("mode IN ('read', 'write')", name="mode_valid"),
        CheckConstraint("mode = 'read' OR approval_id IS NOT NULL", name="write_requires_approval"),
        CheckConstraint(
            "mode = 'read' OR (idempotency_key IS NOT NULL AND length(trim(idempotency_key)) > 0)",
            name="write_requires_idempotency_key",
        ),
        CheckConstraint(
            "status IN ('proposed', 'running', 'succeeded', 'failed', 'unknown')",
            name="status_valid",
        ),
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="risk_level_valid"),
        CheckConstraint(
            "status <> 'succeeded' "
            "OR (result_hash IS NOT NULL AND length(trim(result_hash)) > 0 "
            "AND verified_at IS NOT NULL AND completed_at IS NOT NULL AND error_code IS NULL)",
            name="succeeded_result_complete",
        ),
        CheckConstraint(
            "status <> 'running' OR (completed_at IS NULL AND verified_at IS NULL "
            "AND result_hash IS NULL AND error_code IS NULL)",
            name="running_not_completed",
        ),
        CheckConstraint(
            "status <> 'running' OR execution_deadline IS NOT NULL",
            name="running_has_deadline",
        ),
        CheckConstraint(
            "status NOT IN ('failed', 'unknown') OR "
            "(completed_at IS NOT NULL AND error_code IS NOT NULL AND length(trim(error_code)) > 0)",
            name="failure_terminal_complete",
        ),
        CheckConstraint(
            "CASE WHEN result_purged_at IS NOT NULL THEN CASE WHEN status = 'succeeded' "
            "AND result_ref IS NULL AND result_size IS NULL AND result_expires_at IS NOT NULL "
            "THEN 1 ELSE 0 END WHEN result_ref IS NULL THEN CASE WHEN result_size IS NULL "
            "AND result_expires_at IS NULL THEN 1 ELSE 0 END ELSE CASE WHEN "
            "length(trim(result_ref)) > 0 AND result_size IS NOT NULL AND result_size >= 0 "
            "AND result_expires_at IS NOT NULL THEN 1 ELSE 0 END END = 1",
            name="result_reference_valid",
        ),
        CheckConstraint(
            "CASE WHEN status <> 'succeeded' THEN 1 WHEN result_purged_at IS NULL "
            "THEN CASE WHEN result_ref IS NOT NULL AND length(trim(result_ref)) > 0 "
            "AND result_size IS NOT NULL AND result_expires_at IS NOT NULL THEN 1 ELSE 0 END "
            "ELSE CASE WHEN result_ref IS NULL AND result_size IS NULL "
            "AND result_expires_at IS NOT NULL THEN 1 ELSE 0 END END = 1",
            name="succeeded_result_reference_complete",
        ),
        UniqueConstraint("tenant_id", "task_id", "tool_call_id", name="uq_tool_calls_tenant_task_call"),
        UniqueConstraint(
            "tenant_id", "task_id", "tool_name", "idempotency_key",
            name="uq_tool_calls_task_tool_idempotency",
        ),
        Index("ix_tool_calls_task", "tenant_id", "task_id", "created_at"),
        Index("ix_tool_calls_task_status", "tenant_id", "task_id", "status"),
        Index("ix_tool_calls_sweeper", "status", "execution_deadline"),
        Index(
            "ix_tool_calls_retention",
            "result_expires_at",
            "result_purged_at",
        ),
        Index(
            "uq_tool_calls_consumed_approval",
            "tenant_id",
            "approval_id",
            unique=True,
            sqlite_where=text("approval_id IS NOT NULL"),
            postgresql_where=text("approval_id IS NOT NULL"),
        ),
    )

    tool_call_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    approval_id: Mapped[str | None] = mapped_column(String(64))
    # M1 起 READ 工具不再绑定计划修订；WRITE（M2）仍要求完整计划绑定。
    plan_revision: Mapped[int | None] = mapped_column(Integer)
    plan_digest: Mapped[str | None] = mapped_column(String(80))
    evidence_digest: Mapped[str | None] = mapped_column(String(80))
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(80), nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    principal_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    result_hash: Mapped[str | None] = mapped_column(String(80))
    # M1 可恢复结果引用：完整结果在服务端短期对象中，hash 用于完整性校验。
    result_ref: Mapped[str | None] = mapped_column(String(500))
    result_size: Mapped[int | None] = mapped_column(Integer)
    result_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    execution_deadline: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvidenceRecordRow(Base):
    __tablename__ = "evidence_records"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "version_id", "chunk_id"],
            [
                "chunk_metadata.tenant_id",
                "chunk_metadata.version_id",
                "chunk_metadata.chunk_id",
            ],
            ondelete="RESTRICT",
        ),
        CheckConstraint("document_version > 0", name="document_version_positive"),
        CheckConstraint(
            "evidence_scope IN ('retrieval', 'version_comparison')",
            name="evidence_scope_valid",
        ),
        Index("ix_evidence_records_task_query", "tenant_id", "task_id", "query_fingerprint"),
    )

    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_fingerprint: Mapped[str] = mapped_column(String(80), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    document_id: Mapped[str] = mapped_column(String(200), nullable=False)
    version_id: Mapped[str] = mapped_column(String(64), nullable=False)
    document_version: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(300), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    source_locator: Mapped[str] = mapped_column(String(500), nullable=False)
    evidence_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default="retrieval", server_default=text("'retrieval'")
    )
    permission_granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    supported_claim_ids: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ApprovalRow(Base):
    __tablename__ = "approvals"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
            ["tenant_id", "approved_by"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        UniqueConstraint("tenant_id", "approval_id", name="uq_approvals_tenant_approval"),
        UniqueConstraint(
            "tenant_id", "approval_id", "task_id", "plan_revision", "plan_digest",
            "evidence_digest", "tool_name", "tool_version", "arguments_hash", "principal_hash",
            name="uq_approvals_execution_binding",
        ),
        CheckConstraint("plan_revision > 0", name="plan_revision_positive"),
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'expired', 'executed')",
            name="status_valid",
        ),
        CheckConstraint(
            "status NOT IN ('approved', 'executed') OR "
            "(approved_by IS NOT NULL AND approved_at IS NOT NULL)",
            name="approved_binding_complete",
        ),
        CheckConstraint(
            "status <> 'pending' OR (approved_by IS NULL AND approved_at IS NULL)",
            name="pending_binding_empty",
        ),
        CheckConstraint("expires_at > created_at", name="expires_after_created"),
        Index("ix_approvals_task_status", "tenant_id", "task_id", "status"),
        Index(
            "uq_approvals_active_binding",
            "tenant_id",
            "task_id",
            "plan_revision",
            "plan_digest",
            "evidence_digest",
            "tool_name",
            "tool_version",
            "arguments_hash",
            "principal_hash",
            unique=True,
            sqlite_where=text("status IN ('pending', 'approved')"),
            postgresql_where=text("status IN ('pending', 'approved')"),
        ),
    )

    approval_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    plan_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_digest: Mapped[str] = mapped_column(String(80), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(80), nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    principal_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    evidence_digest: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String(200))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            ["agent_tasks.tenant_id", "agent_tasks.task_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "task_id", "tool_call_id"],
            ["tool_calls.tenant_id", "tool_calls.task_id", "tool_calls.tool_call_id"],
            ondelete="RESTRICT",
        ),
        UniqueConstraint("tenant_id", "storage_key", name="uq_artifacts_tenant_storage_key"),
        CheckConstraint(
            "status IN ('creating', 'ready', 'failed', 'unknown')",
            name="status_valid",
        ),
        CheckConstraint(
            "status <> 'ready' OR (content_hash IS NOT NULL AND length(content_hash) = 71 "
            "AND substr(content_hash, 1, 7) = 'sha256:')",
            name="ready_content_hash_complete",
        ),
        CheckConstraint(
            "artifact_type IN ('report', 'work_item_draft', 'checklist', 'notification_draft')",
            name="artifact_type_valid",
        ),
        Index("ix_artifacts_task", "tenant_id", "task_id", "created_at"),
    )

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(80), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class IdempotencyRecordRow(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["principals.tenant_id", "principals.user_id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "result_version_id"],
            ["document_versions.tenant_id", "document_versions.version_id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "result_job_id"],
            ["ingestion_jobs.tenant_id", "ingestion_jobs.job_id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "result_version_id", "result_job_id"],
            [
                "ingestion_jobs.tenant_id",
                "ingestion_jobs.version_id",
                "ingestion_jobs.job_id",
            ],
            name="fk_idempotency_records_result_job_pair",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "scope",
            "user_id",
            "idempotency_key",
            name="uq_idempotency_scope_user_key",
        ),
        CheckConstraint(
            "status IN ('in_progress', 'succeeded', 'failed', 'unknown')",
            name="status_valid",
        ),
        CheckConstraint(
            "(result_reference IS NULL AND result_version_id IS NULL AND result_job_id IS NULL) "
            "OR (result_reference IS NOT NULL AND result_version_id IS NOT NULL AND result_job_id IS NOT NULL)",
            name="result_binding_complete",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(200), ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(200), nullable=False)
    scope: Mapped[str] = mapped_column(String(100), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    result_version_id: Mapped[str | None] = mapped_column(String(64))
    result_job_id: Mapped[str | None] = mapped_column(String(64))
    result_reference: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
