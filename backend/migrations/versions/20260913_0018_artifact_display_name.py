"""为受控产物增加可读展示名称。"""

from alembic import op
import sqlalchemy as sa


revision = "20260913_0018"
down_revision = "20260911_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("artifacts") as batch:
        batch.add_column(sa.Column("display_name", sa.String(300), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("artifacts") as batch:
        batch.drop_column("display_name")
