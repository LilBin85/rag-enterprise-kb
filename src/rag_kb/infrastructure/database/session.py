"""数据库引擎与会话工厂；本模块绝不隐式创建或升级 schema。"""

from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True)
class Database:
    engine: Engine
    sessions: sessionmaker[Session]


def build_database(database_url: str, *, echo: bool = False) -> Database:
    backend = make_url(database_url).get_backend_name()
    engine_options: dict[str, object] = {"echo": echo, "pool_pre_ping": True}
    if backend == "postgresql":
        # 幂等冲突败者会在下一条语句读取胜者提交的结果，协议明确依赖
        # READ COMMITTED 的逐语句快照，不能继承部署侧更高隔离级别。
        engine_options["isolation_level"] = "READ COMMITTED"
    engine = create_engine(database_url, **engine_options)
    if engine.dialect.name == "sqlite":
        # SQLite 默认不执行外键约束；开发与测试必须显式对齐 PostgreSQL 语义。
        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                # SQLite 只作为开发/测试适配器。并发幂等写入由数据库在这个
                # 有界窗口内串行化；超时后明确失败，不能无限等待或忙循环。
                cursor.execute("PRAGMA busy_timeout=10000")
            finally:
                cursor.close()
    return Database(
        engine=engine,
        sessions=sessionmaker(bind=engine, expire_on_commit=False),
    )
