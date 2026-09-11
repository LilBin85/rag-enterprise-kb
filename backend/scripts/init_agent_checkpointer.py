"""初始化 LangGraph PostgreSQL checkpointer schema。"""

from __future__ import annotations

import logging

from rag_kb.agent.checkpointer import open_postgres_checkpointer
from rag_kb.config import get_settings


logger = logging.getLogger(__name__)


def run() -> int:
    checkpointer, connection = open_postgres_checkpointer(get_settings().database_url)
    try:
        checkpointer.setup()
    finally:
        connection.close()
    logger.info("LangGraph checkpointer schema 已初始化")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        return run()
    except (OSError, ValueError) as exc:
        logger.error("LangGraph checkpointer 初始化失败 error_code=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
