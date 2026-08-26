"""M1 收口：任务线程/结果保留策略标记 + agent_tasks 物理收缩（方案 A）。

新增 checkpoint_purged_at / result_purged_at 作为幂等重扫依据；
删除旧 Runtime 遗留的全部执行字段与关联约束（M1_FIX_PLAN 修复 5、修复 8）。

Revision ID: 20260826_0011
Revises: 20260825_0010
Create Date: 2026-08-26
"""

from alembic import op
import sqlalchemy as sa


revision = "20260826_0011"
down_revision = "20260825_0010"
branch_labels = None
depends_on = None


# 方案 A 删列清单：实施前已全仓 grep 确认无生产读写引用。
_DROPPED_TASK_COLUMNS: tuple[tuple[str, sa.types.TypeEngine], ...] = (
    ("current_node", sa.String(80)),
    ("current_plan_revision", sa.Integer()),
    ("current_step", sa.Integer()),
    ("observation_refs", sa.JSON()),
    ("evidence_refs", sa.JSON()),
    ("query_fingerprints", sa.JSON()),
    ("result_fingerprints", sa.JSON()),
    ("checkpoint_digest", sa.String(80)),
    ("pending_question", sa.Text()),
    ("pending_approval_id", sa.String(64)),
    ("pending_action_ref", sa.String(500)),
    ("pending_action_hash", sa.String(80)),
    ("pending_action_kind", sa.String(32)),
    ("pending_action_expires_at", sa.DateTime(timezone=True)),
    ("repeated_action_fingerprints", sa.JSON()),
    ("no_new_evidence_count", sa.Integer()),
    ("model_calls", sa.Integer()),
    ("tool_calls", sa.Integer()),
    ("replans", sa.Integer()),
    ("consecutive_failures", sa.Integer()),
    ("token_usage", sa.Integer()),
)

_DROPPED_CHECK_CONSTRAINTS = (
    "plan_revision_nonnegative",
    "current_step_nonnegative",
    "model_calls_nonnegative",
    "tool_calls_nonnegative",
    "replans_nonnegative",
    "failures_nonnegative",
    "token_usage_nonnegative",
    "no_evidence_nonnegative",
    "current_node_valid",
)


def _existing_check_constraints() -> dict[str, str]:
    """返回 {去前缀名: 数据库实际名}；历史约束可能带双重 ck_<table>_ 前缀。"""
    result: dict[str, str] = {}
    for item in sa.inspect(op.get_bind()).get_check_constraints("agent_tasks"):
        name = str(item.get("name") or "")
        if not name:
            continue
        key = name
        while key.startswith("ck_agent_tasks_"):
            key = key[len("ck_agent_tasks_"):]
        result[key] = name
    return result


def _constraint_key(name: str) -> str:
    """batch 会按命名约定补一次 ck_agent_tasks_ 前缀，故先剥离既有前缀。"""
    return name.removeprefix("ck_agent_tasks_")


def upgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(
            sa.Column("checkpoint_purged_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column("result_purged_at", sa.DateTime(timezone=True), nullable=True)
        )
        existing = _existing_check_constraints()
        for name in _DROPPED_CHECK_CONSTRAINTS:
            actual = existing.get(name)
            if actual is not None:
                batch.drop_constraint(_constraint_key(actual), type_="check")
        for column, _column_type in _DROPPED_TASK_COLUMNS:
            batch.drop_column(column)


def downgrade() -> None:
    # 尽量 nullable 重建；历史数据不可恢复，仅恢复物理结构以兼容旧代码回滚。
    # 历史约束名带双重前缀（约定 + 原始迁移显式名），此处显式还原。
    with op.batch_alter_table("agent_tasks") as batch:
        for column, column_type in reversed(_DROPPED_TASK_COLUMNS):
            batch.add_column(sa.Column(column, column_type, nullable=True))
        for name, expression in (
            ("plan_revision_nonnegative", "current_plan_revision >= 0"),
            ("current_step_nonnegative", "current_step >= 0"),
            ("model_calls_nonnegative", "model_calls >= 0"),
            ("tool_calls_nonnegative", "tool_calls >= 0"),
            ("replans_nonnegative", "replans >= 0"),
            ("failures_nonnegative", "consecutive_failures >= 0"),
            ("token_usage_nonnegative", "token_usage >= 0"),
            ("no_evidence_nonnegative", "no_new_evidence_count >= 0"),
        ):
            batch.create_check_constraint(
                f"ck_agent_tasks_{name}", expression
            )
        batch.create_check_constraint(
            "ck_agent_tasks_current_node_valid",
            "current_node IN ('load_task', 'classify_goal', 'check_clarification', "
            "'update_plan', 'select_action', 'execute_read_tool', 'record_evidence', "
            "'evaluate_evidence', 'await_approval', 'execute_approved_action', "
            "'verify_action', 'generate_result', 'terminal')",
        )
        batch.drop_column("result_purged_at")
        batch.drop_column("checkpoint_purged_at")
