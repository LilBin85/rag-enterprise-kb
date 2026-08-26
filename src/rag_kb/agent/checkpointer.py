"""LangGraph PostgreSQL checkpointer 的受控连接与初始化。"""

from __future__ import annotations

from psycopg import Connection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


_TASK_CHECKPOINT_MSGPACK_TYPES = (("rag_kb.agent.result", "TaskResultCandidate"),)


def postgres_checkpoint_dsn(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.removeprefix("postgresql+psycopg://")
    if database_url.startswith("postgresql://"):
        return database_url
    raise ValueError("task_agent_postgresql_required")


def task_checkpoint_serializer() -> JsonPlusSerializer:
    """只允许 M1 checkpoint 实际需要的项目类型，绝不回退到 pickle。"""
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=_TASK_CHECKPOINT_MSGPACK_TYPES,
    )


def open_postgres_checkpointer(database_url: str) -> tuple[PostgresSaver, Connection]:
    connection = Connection.connect(
        postgres_checkpoint_dsn(database_url),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    return PostgresSaver(connection, serde=task_checkpoint_serializer()), connection
