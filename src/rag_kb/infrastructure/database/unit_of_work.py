"""显式事务工作单元。"""

from types import TracebackType

from sqlalchemy.orm import Session

from rag_kb.infrastructure.database.session import Database


class UnitOfWork:
    def __init__(self, database: Database) -> None:
        self._database = database
        self.session: Session | None = None

    def __enter__(self) -> "UnitOfWork":
        self.session = self._database.sessions()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()
            self.session = None
