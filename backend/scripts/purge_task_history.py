"""一次性清理：把超过保留期的终态任务行物理删除（含 SQL 级联与外部正文）。

先收敛内容 sweep（工具结果/任务 checkpoint+结果/产物正文），再执行行级删除；
行删除由 FK 级联清空 approvals/artifacts/evidence/plan_revisions/events/
resume_commands/tool_calls。绑定到仍活跃会话的任务不会删除——请先删除对应会话。

手动运行: python scripts/purge_task_history.py [--all]
  --all  行保留期按 0 处理（内容已清空即删除），用于本地演示环境重置。
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta

from rag_kb.agent.checkpointer import open_postgres_checkpointer
from rag_kb.agent.result import AgentResultStore
from rag_kb.application.task_retention_service import TaskRetentionService
from rag_kb.config import get_settings
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.conversation_repository import (
    DatabaseConversationRepository,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.task_retention import (
    DatabaseTaskRetentionLifecycle,
)
from rag_kb.infrastructure.object_store.local import LocalObjectStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("purge-task-history")


def build_service(*, task_row_retention: timedelta):
    settings = get_settings()
    database = build_database(settings.database_url)
    objects = LocalObjectStore(settings.object_store_root)
    conversations = DatabaseConversationRepository(database)
    checkpointer, connection = open_postgres_checkpointer(settings.database_url)
    service = TaskRetentionService(
        lifecycle=DatabaseTaskRetentionLifecycle(
            database, DatabaseTaskRepository(database)
        ),
        objects=objects,
        results=AgentResultStore(objects),
        checkpointer=checkpointer,
        orphan_grace_seconds=settings.result_object_orphan_grace_seconds,
        close=connection.close,
        conversation_purger=conversations.purge_deleting,
        task_row_retention=task_row_retention,
    )
    return service, database


def _drain(sweep, *, limit: int = 500) -> int:
    total = 0
    while (purged := sweep(limit=limit)) > 0:
        total += purged
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="物理清理终态任务历史")
    parser.add_argument(
        "--all",
        action="store_true",
        help="行保留期按 0 处理；默认沿用 TASK_ROW_RETENTION_DAYS（未配置视为 0）",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    if arguments.all or settings.task_row_retention_days is None:
        retention_window = timedelta(0)
    else:
        retention_window = timedelta(days=settings.task_row_retention_days)

    service, database = build_service(task_row_retention=retention_window)
    try:
        logger.info("工具结果正文清理: %s", _drain(service.purge_expired_tool_results))
        logger.info("任务 checkpoint/结果/产物清理: %s", _drain(service.purge_finished_task_content))
        logger.info("会话空壳行清理: %s", _drain(service.purge_conversation_content))
        logger.info("任务行删除: %s", _drain(service.purge_expired_task_rows))
        # 未能删除的终态任务行只剩一种原因：绑定在仍活跃的会话上。
        from sqlalchemy import func, select

        from rag_kb.domain.tasks import TaskStatus

        from rag_kb.infrastructure.database.models import AgentTaskRow

        with database.sessions() as session:
            remaining = session.scalar(
                select(func.count())
                .select_from(AgentTaskRow)
                .where(
                    AgentTaskRow.status.in_(
                        (
                            TaskStatus.SUCCEEDED.value,
                            TaskStatus.REFUSED.value,
                            TaskStatus.FAILED.value,
                            TaskStatus.CANCELLED.value,
                        )
                    )
                )
            )
        if remaining:
            logger.warning(
                "保留 %s 条终态任务行（绑定在活跃会话；删除会话后重跑本脚本即可收敛）",
                remaining,
            )
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
