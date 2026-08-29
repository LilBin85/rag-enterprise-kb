"""LangGraph PostgreSQL checkpointer 的受控连接与初始化。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

from rag_kb.config import get_settings
from rag_kb.observability import (
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)

# LangGraph 在导入 _msgpack 时冻结严格模式开关。
get_settings().prepare_runtime_env()

from psycopg import Connection
from psycopg.rows import dict_row
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED


_TASK_CHECKPOINT_MSGPACK_TYPES = (("rag_kb.agent.result", "TaskResultCandidate"),)
_CHECKPOINT_SAVER_VERSION = "3.1.2"
_CHECKPOINT_STALE_CODE = "task_checkpoint_lease_stale"


class TaskCheckpointError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FencedPostgresSaver(BaseCheckpointSaver):
    """在同一 PostgreSQL 事务内把 checkpoint 写入与任务租约串行化。"""

    def __init__(self, saver: PostgresSaver, connection: Connection) -> None:
        self._validate_saver_contract(saver, connection)
        super().__init__(serde=saver.serde)
        self._saver = saver
        self._connection = connection
        # PostgresSaver 的写入也会持有此锁；RLock 使外层 fence 可安全重入。
        self._lock = RLock()
        self._saver.lock = self._lock

    @staticmethod
    def _validate_saver_contract(saver: PostgresSaver, connection: Connection) -> None:
        try:
            installed = version("langgraph-checkpoint-postgres")
        except PackageNotFoundError as exc:
            raise RuntimeError("task_checkpoint_saver_version_unsupported") from exc
        if installed != _CHECKPOINT_SAVER_VERSION or saver.conn is not connection:
            raise RuntimeError("task_checkpoint_saver_version_unsupported")
        if not all(hasattr(saver, name) for name in ("_cursor", "put", "put_writes")):
            raise RuntimeError("task_checkpoint_saver_version_unsupported")

    @staticmethod
    def _fencing_fields(
        config: dict[str, Any],
    ) -> tuple[str, str, str, int, str | None, str | None]:
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            raise TaskCheckpointError("task_checkpoint_fencing_config_invalid")
        task_id = configurable.get("thread_id")
        tenant_id = configurable.get("checkpoint_tenant_id")
        worker_id = configurable.get("checkpoint_worker_id")
        generation = configurable.get("checkpoint_lease_generation")
        capability_name = configurable.get("checkpoint_required_capability_name")
        protocol_version = configurable.get("checkpoint_required_protocol_version")
        if (
            not isinstance(task_id, str)
            or not task_id
            or len(task_id) > 64
            or not isinstance(tenant_id, str)
            or not tenant_id
            or len(tenant_id) > 200
            or not isinstance(worker_id, str)
            or not worker_id
            or len(worker_id) > 200
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise TaskCheckpointError("task_checkpoint_fencing_config_invalid")
        if (
            not isinstance(capability_name, str)
            or not capability_name
            or len(capability_name) > 64
            or not isinstance(protocol_version, str)
            or not protocol_version
            or len(protocol_version) > 64
        ):
            capability_name = None
            protocol_version = None
        return task_id, tenant_id, worker_id, generation, capability_name, protocol_version

    def _verify_lease_in_transaction(
        self, config: dict[str, Any]
    ) -> tuple[str, str, str, int, str | None, str | None]:
        fields = self._fencing_fields(config)
        task_id, tenant_id, worker_id, generation, capability_name, protocol_version = fields
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT task_id, required_capability_name, required_protocol_version
                FROM agent_tasks
                WHERE task_id = %s
                  AND tenant_id = %s
                  AND lease_owner = %s
                  AND lease_generation = %s
                  AND lease_expires_at > clock_timestamp()
                  AND cancelled = FALSE
                  AND status IN ('planning', 'running')
                FOR UPDATE
                """,
                (task_id, tenant_id, worker_id, generation),
            )
            task = cursor.fetchone()
            if task is None:
                raise TaskCheckpointError(_CHECKPOINT_STALE_CODE)
        required_capability_name = task["required_capability_name"]
        required_protocol_version = task["required_protocol_version"]
        if capability_name is None or protocol_version is None:
            if (
                required_capability_name != "research-agent"
                or required_protocol_version != "m1-fenced-v2"
            ):
                raise TaskCheckpointError(_CHECKPOINT_STALE_CODE)
            return task_id, tenant_id, worker_id, generation, None, None
        if (
            capability_name != required_capability_name
            or protocol_version != required_protocol_version
        ):
            raise TaskCheckpointError(_CHECKPOINT_STALE_CODE)
        return fields

    @staticmethod
    def _merge_next_config(
        config: dict[str, Any],
        saved: dict[str, Any],
        fields: tuple[str, str, str, int, str | None, str | None],
    ) -> dict[str, Any]:
        task_id, tenant_id, worker_id, generation, capability_name, protocol_version = fields
        configurable = dict(config.get("configurable", {}))
        configurable.update(saved.get("configurable", {}))
        configurable.update(
            {
                "thread_id": task_id,
                "checkpoint_tenant_id": tenant_id,
                "checkpoint_worker_id": worker_id,
                "checkpoint_lease_generation": generation,
            }
        )
        if capability_name is not None and protocol_version is not None:
            configurable.update(
                {
                    "checkpoint_required_capability_name": capability_name,
                    "checkpoint_required_protocol_version": protocol_version,
                }
            )
        return {**config, **saved, "configurable": configurable}

    def put(self, config, checkpoint, metadata, new_versions):
        def write():
            with self._lock, self._connection.transaction():
                fields = self._verify_lease_in_transaction(config)
                saved = self._saver.put(config, checkpoint, metadata, new_versions)
            return self._merge_next_config(config, saved, fields)

        return self._observe_write("checkpoint.put", write)

    def put_writes(
        self,
        config,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        def write() -> None:
            with self._lock, self._connection.transaction():
                self._verify_lease_in_transaction(config)
                self._saver.put_writes(config, writes, task_id, task_path)

        self._observe_write("checkpoint.put_writes", write)

    @staticmethod
    def _observe_write(operation: str, write: Callable[[], Any]) -> Any:
        started_at = perf_counter()
        error_code: str | None = None
        with observe_span(
            "checkpoint.write",
            attributes={"component": "checkpointer", "operation": operation},
        ) as span:
            try:
                result = write()
            except Exception as exc:
                error_code = stable_error_code(exc)
                category = error_category_for(error_code)
                finish_observed_span(
                    span,
                    outcome="failure",
                    error_category=category,
                    error_code=error_code,
                )
                telemetry = current_observability()
                if telemetry is not None:
                    telemetry.record_checkpoint_write(outcome="failure")
                    if isinstance(exc, TaskCheckpointError) and (
                        "lease_stale" in error_code or "fenc" in error_code
                    ):
                        telemetry.record_security_boundary_violation(kind="fencing")
                log_event(
                    "checkpoint.write.failed",
                    level=40,
                    component="checkpointer",
                    operation=operation,
                    outcome="failure",
                    error_category=category,
                    error_code=error_code,
                    duration_ms=(perf_counter() - started_at) * 1_000,
                )
                raise
            duration_seconds = perf_counter() - started_at
            finish_observed_span(
                span,
                outcome="succeeded",
                attributes={"duration_ms": duration_seconds * 1_000},
            )
            telemetry = current_observability()
            if telemetry is not None:
                telemetry.record_checkpoint_write(outcome="succeeded")
            return result

    def setup(self) -> None:
        self._saver.setup()

    def get_tuple(self, config):
        return self._saver.get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None) -> Iterator:
        return self._saver.list(config, filter=filter, before=before, limit=limit)

    def get_delta_channel_history(self, *, config, channels):
        return self._saver.get_delta_channel_history(config=config, channels=channels)

    def delete_thread(self, thread_id: str) -> None:
        self._saver.delete_thread(thread_id)


def postgres_checkpoint_dsn(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.removeprefix("postgresql+psycopg://")
    if database_url.startswith("postgresql://"):
        return database_url
    raise ValueError("task_agent_postgresql_required")


def task_checkpoint_serializer() -> JsonPlusSerializer:
    """只允许任务 checkpoint 实际需要的项目类型，绝不回退到 pickle。

    安全边界由构造参数决定（无 pickle、精确 allowlist）；部署侧再以
    ``LANGGRAPH_STRICT_MSGPACK=true`` 与 Settings 校验双保险。
    """
    if not STRICT_MSGPACK_ENABLED:
        # 异常集成路径（先导入 LangGraph、后导入 rag_kb）会使 strict 常量
        # 以 False 冻结；此时序列化边界无法保证，必须 fail-close。
        raise ValueError("langgraph_strict_msgpack_required")
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=_TASK_CHECKPOINT_MSGPACK_TYPES,
    )


def open_postgres_checkpointer(database_url: str) -> tuple[FencedPostgresSaver, Connection]:
    connection = Connection.connect(
        postgres_checkpoint_dsn(database_url),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    saver = PostgresSaver(connection, serde=task_checkpoint_serializer())
    return FencedPostgresSaver(saver, connection), connection
