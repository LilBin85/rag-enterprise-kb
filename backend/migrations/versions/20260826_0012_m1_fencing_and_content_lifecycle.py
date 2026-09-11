"""M1 二次修复：能力租约与执行正文生命周期。"""

from alembic import op
import sqlalchemy as sa


revision = "20260826_0012"
down_revision = "20260826_0011"
branch_labels = None
depends_on = None


_RESEARCH_RESULT_COMPLETE = (
    "CASE WHEN task_mode <> 'research' OR status NOT IN ('succeeded', 'refused') "
    "THEN 1 WHEN result_purged_at IS NULL THEN CASE WHEN result_kind = 'research' "
    "AND result_ref IS NOT NULL AND length(trim(result_ref)) > 0 "
    "AND result_hash IS NOT NULL AND length(trim(result_hash)) > 0 THEN 1 ELSE 0 END "
    "ELSE CASE WHEN result_kind = 'research' AND result_ref IS NULL "
    "AND result_hash IS NOT NULL AND length(trim(result_hash)) > 0 THEN 1 ELSE 0 END END = 1"
)
_RESEARCH_CONTENT_EXPIRY = (
    "CASE WHEN task_mode = 'research' AND status IN "
    "('succeeded', 'refused', 'failed', 'cancelled', 'unknown') "
    "THEN CASE WHEN content_expires_at IS NOT NULL THEN 1 ELSE 0 END ELSE 1 END = 1"
)
_TOOL_RESULT_REFERENCE_VALID = (
    "CASE WHEN result_purged_at IS NOT NULL THEN CASE WHEN status = 'succeeded' "
    "AND result_ref IS NULL AND result_size IS NULL AND result_expires_at IS NOT NULL "
    "THEN 1 ELSE 0 END WHEN result_ref IS NULL THEN CASE WHEN result_size IS NULL "
    "AND result_expires_at IS NULL THEN 1 ELSE 0 END ELSE CASE WHEN "
    "length(trim(result_ref)) > 0 AND result_size IS NOT NULL AND result_size >= 0 "
    "AND result_expires_at IS NOT NULL THEN 1 ELSE 0 END END = 1"
)
_TOOL_SUCCEEDED_RESULT_REFERENCE_COMPLETE = (
    "CASE WHEN status <> 'succeeded' THEN 1 WHEN result_purged_at IS NULL "
    "THEN CASE WHEN result_ref IS NOT NULL AND length(trim(result_ref)) > 0 "
    "AND result_size IS NOT NULL AND result_expires_at IS NOT NULL THEN 1 ELSE 0 END "
    "ELSE CASE WHEN result_ref IS NULL AND result_size IS NULL "
    "AND result_expires_at IS NOT NULL THEN 1 ELSE 0 END END = 1"
)


def _constraint_key(table_name: str, name: str) -> str:
    """去掉 SQLAlchemy 命名约定可能重复添加的表前缀。"""
    prefix = f"ck_{table_name}_"
    while name.startswith(prefix):
        name = name[len(prefix) :]
    return name


def _drop_checks(batch, table_name: str, names: set[str]) -> None:
    for item in sa.inspect(op.get_bind()).get_check_constraints(table_name):
        name = str(item.get("name") or "")
        if name and _constraint_key(table_name, name) in names:
            batch.drop_constraint(_constraint_key(table_name, name), type_="check")


def _backfill_content_expiry(bind) -> None:
    if bind.dialect.name == "postgresql":
        expression = "updated_at + INTERVAL '7 days'"
    elif bind.dialect.name == "sqlite":
        expression = "datetime(updated_at, '+7 days')"
    else:
        raise RuntimeError("unsupported_database_dialect")
    bind.execute(
        sa.text(
            "UPDATE agent_tasks SET content_expires_at = "
            f"{expression} WHERE content_expires_at IS NULL "
            "AND task_mode = 'research' "
            "AND status IN ('succeeded', 'refused', 'failed', 'cancelled', 'unknown')"
        )
    )


def _normalize_legacy_research_results(bind) -> None:
    """旧 0011 允许没有结果绑定的 refused/succeeded 行；0012 失败关闭。"""
    bind.execute(
        sa.text(
            "UPDATE agent_tasks SET status = 'failed', result_kind = NULL, "
            "result_ref = NULL, result_hash = NULL, "
            "last_error_code = 'task_result_binding_missing' "
            "WHERE task_mode = 'research' AND status IN ('succeeded', 'refused') "
            "AND (result_kind IS NULL OR result_kind <> 'research' "
            "OR result_hash IS NULL OR length(trim(result_hash)) = 0 "
            "OR (result_purged_at IS NULL "
            "AND (result_ref IS NULL OR length(trim(result_ref)) = 0)))"
        )
    )


def _repair_legacy_purged_read_tool_results(bind) -> None:
    """仅恢复第一次 GC 错误改写的 READ 审计行，WRITE unknown 不可推断为成功。

    旧 GC 已丢弃准确到期时间；completed_at 是可验证的保守下界，避免再次丢失
    result_expires_at 审计字段。
    """
    bind.execute(
        sa.text(
            "UPDATE tool_calls SET status = 'succeeded', error_code = NULL, "
            "result_ref = NULL, result_size = NULL, "
            "result_expires_at = COALESCE(result_expires_at, completed_at), "
            "result_purged_at = COALESCE(result_purged_at, CURRENT_TIMESTAMP) "
            "WHERE mode = 'read' AND status = 'failed' "
            "AND error_code = 'tool_result_expired' AND result_ref IS NULL "
            "AND result_size IS NULL AND result_expires_at IS NULL "
            "AND result_hash IS NOT NULL AND length(trim(result_hash)) > 0 "
            "AND verified_at IS NOT NULL AND completed_at IS NOT NULL"
        )
    )


def upgrade() -> None:
    op.create_table(
        "agent_capability_leases",
        sa.Column("capability_name", sa.String(64), nullable=False),
        sa.Column("protocol_version", sa.String(64), nullable=False),
        sa.Column("worker_id", sa.String(200), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "capability_name",
            "protocol_version",
            "worker_id",
            name="pk_agent_capability_leases",
        ),
    )
    op.create_index(
        "ix_agent_capability_leases_expiry",
        "agent_capability_leases",
        ["capability_name", "protocol_version", "lease_expires_at"],
    )

    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(
            sa.Column("content_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
    with op.batch_alter_table("tool_calls") as batch:
        batch.add_column(sa.Column("result_purged_at", sa.DateTime(timezone=True), nullable=True))

    # 先拆除旧约束，历史 GC 状态才能在后续修复中从 failed/unknown 恢复为 succeeded。
    with op.batch_alter_table("agent_tasks") as batch:
        _drop_checks(batch, "agent_tasks", {"research_result_complete"})
    with op.batch_alter_table("tool_calls") as batch:
        _drop_checks(
            batch,
            "tool_calls",
            {
                "result_reference_valid",
                "succeeded_result_reference_complete",
                "tool_result_reference_valid",
                "tool_succeeded_result_reference_complete",
            },
        )

    bind = op.get_bind()
    _backfill_content_expiry(bind)
    _normalize_legacy_research_results(bind)
    bind.execute(
        sa.text(
            "UPDATE agent_tasks SET result_ref = NULL "
            "WHERE result_purged_at IS NOT NULL AND result_ref IS NOT NULL"
        )
    )
    _repair_legacy_purged_read_tool_results(bind)

    with op.batch_alter_table("agent_tasks") as batch:
        batch.create_check_constraint("research_result_complete", _RESEARCH_RESULT_COMPLETE)
        batch.create_check_constraint("research_content_expiry", _RESEARCH_CONTENT_EXPIRY)
    with op.batch_alter_table("tool_calls") as batch:
        batch.create_check_constraint("result_reference_valid", _TOOL_RESULT_REFERENCE_VALID)
        batch.create_check_constraint(
            "succeeded_result_reference_complete",
            _TOOL_SUCCEEDED_RESULT_REFERENCE_COMPLETE,
        )

    op.create_index(
        "ix_agent_tasks_retention",
        "agent_tasks",
        ["task_mode", "status", "content_expires_at"],
    )
    op.create_index(
        "ix_tool_calls_retention",
        "tool_calls",
        ["result_expires_at", "result_purged_at"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    # 0011 无法表达“成功但正文已清理”。显式降为其旧的 fail-closed 形态，
    # 正文不能恢复，重新升级后也不应伪造成功结果。
    with op.batch_alter_table("agent_tasks") as batch:
        _drop_checks(
            batch,
            "agent_tasks",
            {"research_result_complete", "research_content_expiry"},
        )
    with op.batch_alter_table("tool_calls") as batch:
        _drop_checks(
            batch,
            "tool_calls",
            {"result_reference_valid", "succeeded_result_reference_complete"},
        )
    bind.execute(
        sa.text(
            "UPDATE agent_tasks SET status = 'failed', "
            "last_error_code = 'task_result_expired' "
            "WHERE task_mode = 'research' AND status = 'succeeded' "
            "AND result_purged_at IS NOT NULL AND result_ref IS NULL"
        )
    )
    bind.execute(
        sa.text(
            "UPDATE tool_calls SET status = CASE WHEN mode = 'read' THEN 'failed' "
            "ELSE 'unknown' END, error_code = 'tool_result_expired', "
            "result_expires_at = NULL "
            "WHERE status = 'succeeded' AND result_purged_at IS NOT NULL "
            "AND result_ref IS NULL"
        )
    )

    op.drop_index("ix_tool_calls_retention", table_name="tool_calls")
    op.drop_index("ix_agent_tasks_retention", table_name="agent_tasks")
    op.drop_index(
        "ix_agent_capability_leases_expiry",
        table_name="agent_capability_leases",
    )
    op.drop_table("agent_capability_leases")

    with op.batch_alter_table("agent_tasks") as batch:
        batch.drop_column("content_expires_at")
        batch.create_check_constraint(
            "research_result_complete",
            "status <> 'succeeded' OR task_mode <> 'research' OR "
            "(result_kind = 'research' AND result_ref IS NOT NULL "
            "AND length(trim(result_ref)) > 0 AND result_hash IS NOT NULL "
            "AND length(trim(result_hash)) > 0)",
        )
    with op.batch_alter_table("tool_calls") as batch:
        batch.drop_column("result_purged_at")
        batch.create_check_constraint(
            "tool_result_reference_valid",
            "(result_ref IS NULL OR length(trim(result_ref)) > 0) "
            "AND (result_size IS NULL OR result_size >= 0) "
            "AND (result_expires_at IS NULL OR result_ref IS NOT NULL)",
        )
        batch.create_check_constraint(
            "tool_succeeded_result_reference_complete",
            "status <> 'succeeded' OR "
            "(result_ref IS NOT NULL AND length(trim(result_ref)) > 0 "
            "AND result_size IS NOT NULL AND result_expires_at IS NOT NULL)",
        )
