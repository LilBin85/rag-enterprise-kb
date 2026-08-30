"""SQLAlchemy 2.x 声明基类。"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


def _check_constraint_name(constraint: object, table: object) -> str:
    """为显式检查约束名补全前缀，并避免 Alembic 重复套用约定。"""
    name = str(getattr(constraint, "name", "") or "")
    table_name = str(getattr(table, "name", "") or "")
    if not name or not table_name:
        raise ValueError("check_constraint_name_required")
    prefix = f"ck_{table_name}_"
    return name if name.startswith(prefix) else f"{prefix}{name}"


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "%(_check_constraint_name)s",
    "_check_constraint_name": _check_constraint_name,
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
