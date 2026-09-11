"""Alembic 迁移环境；业务请求不得调用本模块。"""

from __future__ import annotations

from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database import models as _models  # noqa: F401


config = context.config
if config.config_file_name is not None:
    # 迁移命令可能与测试或管理进程同进程运行，不能禁用应用已有安全日志器。
    fileConfig(config.config_file_name, disable_existing_loggers=False)

database_url = (
    config.attributes.get("database_url")
    or os.environ.get("DATABASE_URL")
    # 保留手动 Alembic 命令的既有环境变量；Compose 使用应用统一配置。
    or os.environ.get("RAG_KB_DATABASE_URL")
)
if not database_url:
    raise RuntimeError("migration_database_url_required")
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
