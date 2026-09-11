"""M1：收缩任务业务状态，READ 工具可恢复结果引用与计划解耦。

Revision ID: 20260825_0010
Revises: 20260824_0009
Create Date: 2026-08-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260825_0010"
down_revision = "20260824_0009"
branch_labels = None
depends_on = None


_LEGACY_TASK_MARKERS = (
    "pending_question",
    "pending_approval_id",
    "pending_action_",
)


def _legacy_check_names(table_name: str) -> tuple[str, ...]:
    return tuple(
        item["name"]
        for item in sa.inspect(op.get_bind()).get_check_constraints(table_name)
        if item.get("name")
        and any(marker in str(item.get("sqltext", "")) for marker in _LEGACY_TASK_MARKERS)
    )


def _constraint_key(table_name: str, name: str) -> str:
    return name.removeprefix(f"ck_{table_name}_")


def upgrade() -> None:
    # 旧 Runtime 任务没有可恢复的 LangGraph checkpoint，显式终止而不映射状态。
    op.execute(
        sa.text(
            "UPDATE agent_tasks SET "
            "status = 'failed', current_node = 'terminal', "
            "pending_question = NULL, pending_approval_id = NULL, "
            "pending_action_ref = NULL, pending_action_hash = NULL, "
            "pending_action_kind = NULL, pending_action_expires_at = NULL, "
            "lease_owner = NULL, lease_expires_at = NULL, "
            "last_error_code = 'task_runtime_replaced', "
            "state_version = state_version + 1, updated_at = CURRENT_TIMESTAMP "
            "WHERE status NOT IN ('succeeded', 'refused', 'failed', 'cancelled', 'unknown')"
        )
    )
    # 旧 research 成功记录没有 M1 TaskResult，不能继续声明为可读取的成功结果。
    op.execute(
        sa.text(
            "UPDATE agent_tasks SET "
            "status = 'failed', current_node = 'terminal', "
            "pending_question = NULL, pending_approval_id = NULL, "
            "pending_action_ref = NULL, pending_action_hash = NULL, "
            "pending_action_kind = NULL, pending_action_expires_at = NULL, "
            "lease_owner = NULL, lease_expires_at = NULL, "
            "last_error_code = 'task_runtime_replaced', "
            "state_version = state_version + 1, updated_at = CURRENT_TIMESTAMP "
            "WHERE task_mode = 'research' AND status = 'succeeded'"
        )
    )
    # 旧成功调用只有 hash，没有可幂等回读的结果对象，按 UNKNOWN fail-closed。
    op.execute(
        sa.text(
            "UPDATE tool_calls SET status = 'unknown', "
            "completed_at = COALESCE(completed_at, created_at), "
            "error_code = 'task_runtime_replaced' "
            "WHERE status = 'succeeded'"
        )
    )
    task_checks = _legacy_check_names("agent_tasks")
    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(sa.Column("interrupt_kind", sa.String(32), nullable=True))
        batch.add_column(sa.Column("interrupt_summary", sa.Text(), nullable=True))
        batch.add_column(sa.Column("result_kind", sa.String(32), nullable=True))
        batch.add_column(sa.Column("result_ref", sa.String(500), nullable=True))
        batch.add_column(sa.Column("result_hash", sa.String(80), nullable=True))
        for name in task_checks:
            batch.drop_constraint(_constraint_key("agent_tasks", name), type_="check")
        batch.create_check_constraint(
            "interrupt_complete",
            "status NOT IN ('waiting_for_user', 'waiting_for_approval') OR "
            "(interrupt_kind IS NOT NULL AND length(trim(interrupt_kind)) > 0 "
            "AND interrupt_summary IS NOT NULL AND length(trim(interrupt_summary)) > 0)",
        )
        batch.create_check_constraint(
            "research_result_complete",
            "status <> 'succeeded' OR task_mode <> 'research' OR "
            "(result_kind = 'research' AND result_ref IS NOT NULL "
            "AND length(trim(result_ref)) > 0 AND result_hash IS NOT NULL "
            "AND length(trim(result_hash)) > 0)",
        )
        indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes("agent_tasks")}
        if "ix_agent_tasks_pending_action_expiry" in indexes:
            batch.drop_index("ix_agent_tasks_pending_action_expiry")

    with op.batch_alter_table("tool_calls") as batch:
        # READ 工具不再要求计划绑定；复合外键含 NULL 时按 MATCH SIMPLE 跳过校验。
        batch.alter_column("plan_revision", existing_type=sa.Integer(), nullable=True)
        batch.alter_column("plan_digest", existing_type=sa.String(80), nullable=True)
        batch.alter_column("evidence_digest", existing_type=sa.String(80), nullable=True)
        batch.add_column(sa.Column("result_ref", sa.String(500), nullable=True))
        batch.add_column(sa.Column("result_size", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("result_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
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

    with op.batch_alter_table("evidence_records") as batch:
        batch.add_column(
            sa.Column(
                "evidence_scope",
                sa.String(32),
                nullable=False,
                server_default=sa.text("'retrieval'"),
            )
        )
        batch.create_check_constraint(
            "evidence_scope_valid",
            "evidence_scope IN ('retrieval', 'version_comparison')",
        )


def downgrade() -> None:
    with op.batch_alter_table("evidence_records") as batch:
        batch.drop_constraint("evidence_scope_valid", type_="check")
        batch.drop_column("evidence_scope")

    with op.batch_alter_table("tool_calls") as batch:
        batch.drop_constraint("tool_succeeded_result_reference_complete", type_="check")
        batch.drop_constraint("tool_result_reference_valid", type_="check")
        batch.drop_column("result_expires_at")
        batch.drop_column("result_size")
        batch.drop_column("result_ref")
        batch.alter_column("evidence_digest", existing_type=sa.String(80), nullable=False)
        batch.alter_column("plan_digest", existing_type=sa.String(80), nullable=False)
        batch.alter_column("plan_revision", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("agent_tasks") as batch:
        batch.drop_constraint("research_result_complete", type_="check")
        batch.drop_constraint("interrupt_complete", type_="check")
        batch.create_check_constraint(
            "ck_agent_tasks_waiting_question_complete",
            "status <> 'waiting_for_user' "
            "OR (pending_question IS NOT NULL AND length(trim(pending_question)) > 0)",
        )
        batch.create_check_constraint(
            "ck_agent_tasks_waiting_approval_complete",
            "status <> 'waiting_for_approval' "
            "OR (pending_approval_id IS NOT NULL AND length(trim(pending_approval_id)) > 0)",
        )
        batch.create_check_constraint(
            "pending_action_complete",
            "(pending_action_ref IS NULL AND pending_action_hash IS NULL "
            "AND pending_action_kind IS NULL AND pending_action_expires_at IS NULL) OR "
            "(pending_action_ref IS NOT NULL AND length(trim(pending_action_ref)) > 0 "
            "AND pending_action_hash IS NOT NULL AND length(pending_action_hash) = 71 "
            "AND substr(pending_action_hash, 1, 7) = 'sha256:' "
            "AND pending_action_kind IS NOT NULL AND pending_action_expires_at IS NOT NULL)",
        )
        batch.create_check_constraint(
            "pending_action_kind_valid",
            "pending_action_kind IS NULL OR "
            "pending_action_kind IN ('search', 'call_tool', 'propose_action')",
        )
        batch.create_check_constraint(
            "pending_action_node_complete",
            "current_node NOT IN ('execute_read_tool', 'await_approval', "
            "'execute_approved_action') OR pending_action_ref IS NOT NULL",
        )
        batch.create_check_constraint(
            "waiting_approval_action_valid",
            "status <> 'waiting_for_approval' "
            "OR pending_action_kind = 'propose_action'",
        )
        batch.create_check_constraint(
            "pending_approval_action_valid",
            "pending_approval_id IS NULL OR pending_action_kind = 'propose_action'",
        )
        batch.create_index(
            "ix_agent_tasks_pending_action_expiry", ["pending_action_expires_at"]
        )
        batch.drop_column("result_hash")
        batch.drop_column("result_ref")
        batch.drop_column("result_kind")
        batch.drop_column("interrupt_summary")
        batch.drop_column("interrupt_kind")
