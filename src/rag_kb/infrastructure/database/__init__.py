"""业务数据库模型与显式会话工厂。"""

from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.session import Database, build_database
from rag_kb.infrastructure.database import models as _models  # noqa: F401

__all__ = ["Base", "Database", "build_database"]
