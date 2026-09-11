"""为会话轮次与耐久任务保存显式文档检索范围。"""

from alembic import op
import sqlalchemy as sa


revision = "20260911_0017"
down_revision = "20260906_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("conversation_turns") as batch:
        batch.add_column(sa.Column("scope_document_id", sa.String(200), nullable=True))
    with op.batch_alter_table("agent_tasks") as batch:
        batch.add_column(sa.Column("scope_document_id", sa.String(200), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("agent_tasks") as batch:
        batch.drop_column("scope_document_id")
    with op.batch_alter_table("conversation_turns") as batch:
        batch.drop_column("scope_document_id")
