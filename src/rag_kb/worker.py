"""生产摄取工作器组合根；只在领取到任务后解析、加载模型并写 Qdrant。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from http.server import ThreadingHTTPServer
import logging
import socket
import threading
import time
from typing import Any
from uuid import uuid4

from prometheus_client.exposition import MetricsHandler
from qdrant_client import QdrantClient

from rag_kb.agent.checkpointer import open_postgres_checkpointer
from rag_kb.agent.factory import (
    build_controlled_action_agent,
    build_research_agent,
)
from rag_kb.agent.result import AgentResultStore
from rag_kb.agent.tools import (
    build_controlled_action_tools,
    build_research_tools,
)
from rag_kb.application.ingestion_worker import (
    DocumentRegistryIngestionJobPort,
    IngestionWorker,
)
from rag_kb.application.projection_reconciler import ProjectionReconciler
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.task_retention_service import TaskRetentionService
from rag_kb.application.task_worker import TaskWorker
from rag_kb.application.tool_execution_service import ToolExecutionService
from rag_kb.application.upload_reconciler import UploadReservationReconciler
from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.config import get_settings, resolve_retrieval_limits, task_budgets_from_settings
from rag_kb.domain.tasks import TaskBudgets
from rag_kb.embedder import BgeM3Embedder
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskAuthContextPort,
    DatabaseTaskEvidencePort,
)
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
)
from rag_kb.infrastructure.database.index_projection_metadata import (
    DatabaseDocumentIndexMetadataPort,
)
from rag_kb.infrastructure.database.projection_repository import (
    DatabaseProjectionReconcilePort,
)
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.vectorstore.qdrant_document_index import QdrantDocumentIndex
from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import QdrantAuthorizedSearch
from rag_kb.infrastructure.tools.factory import build_tool_registry
from rag_kb.observability import (
    MetadataOnlyLlmCallback,
    Observability,
    current_observability,
    error_category_for,
    finish_observed_span,
    log_event,
    observe_span,
    stable_error_code,
)


_RECOVERY_BATCH_SIZE = 25
_RECOVERY_INTERVAL_SECONDS = 30.0
_RECOVERY_TIME_BUDGET_SECONDS = 1.0
_UPLOAD_RESERVATION_GRACE = timedelta(seconds=30)
_QUEUE_SNAPSHOT_LIMIT = 1_000


class TaskAgentConfigurationError(RuntimeError):
    pass


class _WorkerMetricsServer:
    """Worker 私有 registry 的最小内部 HTTP 服务。"""

    def __init__(self, telemetry: Observability, *, port: int) -> None:
        self._telemetry = telemetry
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def port(self) -> int:
        server = self._server
        return int(server.server_address[1]) if server is not None else self._port

    def start(self) -> bool:
        with self._lock:
            if self._server is not None:
                return True
            base_handler = MetricsHandler.factory(self._telemetry.registry)

            class _MetricsHandler(base_handler):
                def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
                    if self.path != "/metrics":
                        self.send_error(404)
                        return
                    super().do_GET()

                def log_message(self, _format: str, *_args: object) -> None:
                    return

            try:
                server = ThreadingHTTPServer(("0.0.0.0", self._port), _MetricsHandler)
            except OSError:
                log_event(
                    "observability.metrics_server.unavailable",
                    level=logging.WARNING,
                    component="worker",
                    operation="metrics.server",
                    outcome="failure",
                    error_code="worker_metrics_server_unavailable",
                )
                return False
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
                name="rag-kb-worker-metrics",
            )
            self._server = server
            self._thread = thread
            thread.start()
            return True

    def close(self) -> None:
        with self._lock:
            server, thread = self._server, self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join()


class _CapabilityLeaseHeartbeat:
    """独立于任务 invoke 的 M2 task capability 租约心跳。"""

    def __init__(
        self,
        repository: DatabaseTaskRepository,
        *,
        worker_id: str,
        capability_name: str,
        protocol_version: str,
        lease_seconds: int,
        telemetry: Observability | None = None,
    ) -> None:
        self._repository = repository
        self._worker_id = worker_id
        self._capability_name = capability_name
        self._protocol_version = protocol_version
        self._lease_seconds = lease_seconds
        self._telemetry = telemetry
        self._interval = lease_seconds / 3
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._repository.register_capability(
            capability_name=self._capability_name,
            protocol_version=self._protocol_version,
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if self._telemetry is not None:
            self._telemetry.set_worker_capability_available(
                worker_kind="task",
                capability=self._capability_name,
                protocol=self._protocol_version,
                available=True,
            )
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="rag-kb-capability-lease",
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._repository.renew_capability(
                    capability_name=self._capability_name,
                    protocol_version=self._protocol_version,
                    worker_id=self._worker_id,
                    lease_seconds=self._lease_seconds,
                )
                if self._telemetry is not None:
                    self._telemetry.set_worker_capability_available(
                        worker_kind="task",
                        capability=self._capability_name,
                        protocol=self._protocol_version,
                        available=True,
                    )
            except Exception as exc:
                code = stable_error_code(exc)
                if self._telemetry is not None:
                    self._telemetry.set_worker_capability_available(
                        worker_kind="task",
                        capability=self._capability_name,
                        protocol=self._protocol_version,
                        available=False,
                    )
                    self._telemetry.record_lease_failure(
                        kind="capability",
                        operation="capability.lease.renew",
                        error_code=code,
                    )
                else:
                    log_event(
                        "lease.renew.failed",
                        level=logging.ERROR,
                        component="worker",
                        operation="capability.lease.renew",
                        outcome="failure",
                        error_category="lease",
                        error_code=code,
                    )
                self._stop.set()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
            self._thread = None
        try:
            self._repository.unregister_capability(
                capability_name=self._capability_name,
                protocol_version=self._protocol_version,
                worker_id=self._worker_id,
            )
        except Exception:
            log_event(
                "capability.lease.unregister.failed",
                level=logging.ERROR,
                component="worker",
                operation="capability.lease.unregister",
                outcome="failure",
                error_code="capability_unregister_failed",
            )
        finally:
            if self._telemetry is not None:
                self._telemetry.set_worker_capability_available(
                    worker_kind="task",
                    capability=self._capability_name,
                    protocol=self._protocol_version,
                    available=False,
                )


def _build_document_indexing_runtime():
    """构造 ingestion/projection 共用的解析和索引依赖。"""
    settings = get_settings()
    database = build_database(settings.database_url)
    objects = LocalObjectStore(settings.object_store_root)
    parser = LocalDocumentParser(objects, max_chars=settings.chunk_max_chars)
    index = QdrantDocumentIndex(
        QdrantClient(url=settings.qdrant_url),
        BgeM3Embedder(),
        DatabaseDocumentIndexMetadataPort(database),
        collection=settings.qdrant_v2_collection,
    )
    return database, parser, index


def build_ingestion_worker(*, worker_id: str | None = None) -> IngestionWorker:
    """构造单一摄取 worker，不初始化 projection runtime。"""
    database, parser, index = _build_document_indexing_runtime()
    identity = worker_id or f"{socket.gethostname()}-{uuid4().hex[:12]}"
    return IngestionWorker(
        DocumentRegistryIngestionJobPort(database),
        parser,
        index,
        worker_id=identity,
    )


def build_projection_reconciler(*, worker_id: str | None = None) -> ProjectionReconciler:
    """构造单一投影收敛器，不初始化 ingestion runtime。"""
    database, parser, index = _build_document_indexing_runtime()
    identity = worker_id or f"{socket.gethostname()}-{uuid4().hex[:12]}"
    return ProjectionReconciler(
        DatabaseProjectionReconcilePort(database),
        parser,
        index,
        worker_id=f"{identity}-projection-reconciler",
    )


def build_upload_reconciler() -> UploadReservationReconciler:
    settings = get_settings()
    return UploadReservationReconciler(
        build_database(settings.database_url),
        LocalObjectStore(settings.object_store_root),
    )


def _build_research_model(settings):
    if not settings.deepseek_api_key:
        raise TaskAgentConfigurationError("task_agent_api_key_missing")
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=settings.agent_timeout_seconds,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
        callbacks=[
            MetadataOnlyLlmCallback(
                provider="deepseek",
                model_name=settings.deepseek_model,
                operation="agent.generate",
            )
        ],
    )


def build_task_worker(
    *, worker_id: str | None = None, publish_capability: bool = True
) -> TaskWorker:
    """构造 M2 protocol Research Agent worker 和 PostgreSQL checkpointer。"""
    settings = get_settings()
    telemetry = current_observability()
    database = build_database(settings.database_url)
    objects = LocalObjectStore(settings.object_store_root)
    embedder = BgeM3Embedder()
    vectors = QdrantAuthorizedSearch(
        QdrantClient(url=settings.qdrant_url),
        embedder,
        collection=settings.qdrant_v2_collection,
    )
    candidate_k, context_k = resolve_retrieval_limits(settings)
    retriever = EnterpriseRetriever(
        DatabaseAuthorizationPort(database),
        vectors,
        candidate_k=candidate_k,
        context_k=context_k,
        context_token_budget=settings.retrieval_context_token_budget,
        rrf_k=settings.retrieval_rrf_k,
        hybrid_enabled=settings.enable_hybrid_retrieval,
        dense_score_threshold=settings.retrieval_score_threshold,
    )
    registry: ToolRegistry = build_tool_registry(
        database=database,
        artifact_root=settings.object_store_root,
        retriever=retriever,
        chunk_max_chars=settings.chunk_max_chars,
    )
    identity = worker_id or f"{socket.gethostname()}-task-{uuid4().hex[:12]}"
    evidence_validator = DatabaseTaskEvidencePort(database)
    approvals = ApprovalService(database, evidence_validator=evidence_validator)
    tools = ToolExecutionService(
        database,
        registry,
        approvals,
        evidence_validator=evidence_validator,
        tool_results=objects,
    )
    try:
        checkpointer, connection = open_postgres_checkpointer(settings.database_url)
    except ValueError as exc:
        tools.close()
        raise TaskAgentConfigurationError(str(exc)) from exc
    try:
        model = _build_research_model(settings)
        research_tools = build_research_tools(tools)
        controlled_tools = build_controlled_action_tools(tools)
        research_agents: dict[tuple[int, int], Any] = {}
        controlled_agents: dict[tuple[int, int], Any] = {}

        def agent_for(budgets: TaskBudgets) -> Any:
            key = (budgets.max_model_calls, budgets.max_tool_calls)
            if key not in research_agents:
                research_agents[key] = build_research_agent(
                    model,
                    research_tools,
                    checkpointer,
                    max_model_calls=budgets.max_model_calls,
                    max_tool_calls=budgets.max_tool_calls,
                )
            return research_agents[key]

        def controlled_agent_for(budgets: TaskBudgets) -> Any:
            key = (budgets.max_model_calls, budgets.max_tool_calls)
            if key not in controlled_agents:
                controlled_agents[key] = build_controlled_action_agent(
                    model,
                    controlled_tools,
                    checkpointer,
                    max_model_calls=budgets.max_model_calls,
                    max_tool_calls=budgets.max_tool_calls,
                )
            return controlled_agents[key]

        # Capability 仅代表默认预算 Agent 已经可用；其他持久预算按需编译并缓存。
        # 受控 Agent 与 Research Agent 共用已验证的 model/checkpointer，
        # 首个受控任务时惰性编译（编译失败由 worker 错误路径 fail-closed）。
        agent_for(task_budgets_from_settings(settings))
    except Exception:
        tools.close()
        connection.close()
        raise
    repository = DatabaseTaskRepository(
        database,
        task_capability_name=settings.task_agent_capability_name,
        task_capability_protocol_version=settings.task_agent_protocol_version,
    )
    capability: _CapabilityLeaseHeartbeat | None = None

    def close() -> None:
        try:
            if capability is not None:
                capability.close()
        finally:
            connection.close()

    worker = TaskWorker(
        repository=repository,
        auth_contexts=DatabaseTaskAuthContextPort(database),
        agent_for=agent_for,
        controlled_agent_for=controlled_agent_for,
        results=AgentResultStore(objects),
        evidence=EvidenceLedgerRepository(),
        tools=tools,
        database=database,
        worker_id=identity,
        capability_name=settings.task_agent_capability_name,
        protocol_version=settings.task_agent_protocol_version,
        approvals=approvals,
        objects=objects,
        close=close,
    )
    if publish_capability:
        try:
            capability = _CapabilityLeaseHeartbeat(
                repository,
                worker_id=identity,
                capability_name=settings.task_agent_capability_name,
                protocol_version=settings.task_agent_protocol_version,
                lease_seconds=settings.agent_capability_lease_seconds,
                telemetry=telemetry,
            )
            capability.start()
        except Exception as exc:
            worker.close()
            raise TaskAgentConfigurationError("task_agent_capability_registration_failed") from exc
        except BaseException:
            try:
                worker.close()
            except Exception:
                log_event(
                    "capability.lease.startup_cleanup.failed",
                    level=logging.ERROR,
                    component="worker",
                    operation="capability.lease.startup",
                    outcome="failure",
                    error_code="capability_startup_cleanup_failed",
                )
            raise
    return worker


def build_task_retention() -> TaskRetentionService:
    """构造独立周期保留服务，不与正在执行的 Agent 共享 checkpointer 连接。"""
    settings = get_settings()
    database = build_database(settings.database_url)
    objects = LocalObjectStore(settings.object_store_root)
    try:
        checkpointer, connection = open_postgres_checkpointer(settings.database_url)
    except ValueError as exc:
        raise TaskAgentConfigurationError(str(exc)) from exc
    try:
        return TaskRetentionService(
            database=database,
            repository=DatabaseTaskRepository(database),
            objects=objects,
            results=AgentResultStore(objects),
            checkpointer=checkpointer,
            orphan_grace_seconds=settings.result_object_orphan_grace_seconds,
            close=connection.close,
        )
    except Exception:
        connection.close()
        raise


def _build_task_queue_snapshot_port() -> DatabaseTaskRepository:
    settings = get_settings()
    database = build_database(settings.database_url)
    return DatabaseTaskRepository(
        database,
        task_capability_name=settings.task_agent_capability_name,
        task_capability_protocol_version=settings.task_agent_protocol_version,
    )


def _build_ingestion_queue_snapshot_port() -> DocumentRegistryIngestionJobPort:
    settings = get_settings()
    return DocumentRegistryIngestionJobPort(build_database(settings.database_url))


def _finish_maintenance_span(
    span: object,
    *,
    worker_kind: str,
    operation: str,
    started_at: float,
    error_code: str | None,
) -> None:
    duration_seconds = time.perf_counter() - started_at
    outcome = "failure" if error_code is not None else "succeeded"
    category = error_category_for(error_code) if error_code is not None else "none"
    finish_observed_span(
        span,
        outcome=outcome,
        error_category=category,
        error_code=error_code,
        attributes={"duration_ms": duration_seconds * 1_000},
    )
    log_event(
        "worker.maintenance.completed",
        level=logging.ERROR if error_code is not None else logging.INFO,
        component="worker",
        worker_kind=worker_kind,
        operation=operation,
        outcome=outcome,
        error_category=category,
        error_code=error_code,
        duration_ms=duration_seconds * 1_000,
    )


def _run_maintenance_actions(
    *,
    worker_kind: str,
    operation: str,
    actions,
    telemetry: Observability | None,
) -> None:
    started_at = time.perf_counter()
    error_code: str | None = None
    span_operation = f"worker.{worker_kind}.{operation}"
    with observe_span(
        span_operation,
        attributes={
            "component": "worker",
            "worker_kind": worker_kind,
            "operation": span_operation,
        },
    ) as span:
        try:
            for kind, action in actions:
                try:
                    count = action()
                except Exception as exc:
                    error_code = error_code or "runtime_sweep_failed"
                    if telemetry is not None:
                        telemetry.record_recovery_action(
                            worker_kind=worker_kind,
                            kind=kind,
                            outcome="failure",
                        )
                        telemetry.record_worker_error(
                            worker_kind=worker_kind,
                            operation=operation,
                            error_category=error_category_for(stable_error_code(exc)),
                        )
                else:
                    if telemetry is not None:
                        telemetry.record_recovery_action(
                            worker_kind=worker_kind,
                            kind=kind,
                            outcome="succeeded",
                            count=count,
                        )
        finally:
            _finish_maintenance_span(
                span,
                worker_kind=worker_kind,
                operation=span_operation,
                started_at=started_at,
                error_code=error_code,
            )


def _run_task_maintenance(
    *,
    task_worker,
    task_retention,
    telemetry: Observability | None,
) -> None:
    """仅由 task Worker 执行 task/tool/query 超时和 task retention。"""
    _run_maintenance_actions(
        worker_kind="task",
        operation="maintenance",
        actions=(
            (
                "stale_tool_call",
                lambda: task_worker.expire_stale_tool_calls(limit=_RECOVERY_BATCH_SIZE),
            ),
            (
                "stale_knowledge_query",
                lambda: task_worker.expire_stale_knowledge_queries(
                    limit=_RECOVERY_BATCH_SIZE
                ),
            ),
            (
                "waiting_task",
                lambda: task_retention.expire_waiting_tasks(limit=_RECOVERY_BATCH_SIZE),
            ),
            (
                "tool_result_retention",
                lambda: task_retention.purge_expired_tool_results(
                    limit=_RECOVERY_BATCH_SIZE
                ),
            ),
            (
                "task_content_retention",
                lambda: task_retention.purge_finished_task_content(
                    limit=_RECOVERY_BATCH_SIZE
                ),
            ),
            (
                "orphan_result",
                lambda: task_retention.purge_orphaned_result_objects(
                    limit=_RECOVERY_BATCH_SIZE
                ),
            ),
        ),
        telemetry=telemetry,
    )


def _run_ingestion_recovery(
    *, upload_reconciler, telemetry: Observability | None
) -> None:
    """仅由 ingestion Worker 修复上传预留。"""
    started_at = time.perf_counter()
    error_code: str | None = None
    operation = "worker.ingestion.maintenance"
    with observe_span(
        operation,
        attributes={
            "component": "worker",
            "worker_kind": "ingestion",
            "operation": operation,
        },
    ) as span:
        try:
            try:
                upload_result = upload_reconciler.reconcile(
                    older_than=datetime.now(UTC) - _UPLOAD_RESERVATION_GRACE,
                    limit=_RECOVERY_BATCH_SIZE,
                    time_budget_seconds=_RECOVERY_TIME_BUDGET_SECONDS,
                )
            except Exception as exc:
                error_code = "upload_reconciliation_failed"
                if telemetry is not None:
                    telemetry.record_recovery_action(
                        worker_kind="ingestion",
                        kind="upload_reservation",
                        outcome="failure",
                    )
                    telemetry.record_worker_error(
                        worker_kind="ingestion",
                        operation="maintenance",
                        error_category=error_category_for(stable_error_code(exc)),
                    )
            else:
                if telemetry is not None:
                    for outcome, count in (
                        ("recovered", upload_result.recovered),
                        ("missing", upload_result.missing_objects),
                        ("mismatched", upload_result.content_mismatches),
                        ("failure", upload_result.failed),
                    ):
                        telemetry.record_recovery_action(
                            worker_kind="ingestion",
                            kind="upload_reservation",
                            outcome=outcome,
                            count=count,
                        )
                    for _ in range(upload_result.content_mismatches):
                        telemetry.record_security_boundary_violation(kind="content_hash")
                if upload_result.failed:
                    error_code = "upload_reconciliation_failed"
                    if telemetry is not None:
                        telemetry.record_worker_error(
                            worker_kind="ingestion",
                            operation="maintenance",
                            error_category="storage",
                            count=upload_result.failed,
                        )
        finally:
            _finish_maintenance_span(
                span,
                worker_kind="ingestion",
                operation=operation,
                started_at=started_at,
                error_code=error_code,
            )


def _record_task_queue_snapshot(
    *,
    task_queue,
    telemetry: Observability,
    capability_name: str,
    protocol_version: str,
) -> None:
    """仅由 task Worker 写 task queue 与 capability 快照。"""

    def snapshot() -> int:
        for item in task_queue.queue_snapshot(
            capability_name=capability_name,
            protocol_version=protocol_version,
            limit=_QUEUE_SNAPSHOT_LIMIT,
        ):
            telemetry.set_task_queue_snapshot(
                worker_kind="task",
                mode=item.task_mode,
                status=item.status,
                depth=item.depth,
                oldest_age_seconds=item.oldest_age_seconds,
            )
        telemetry.set_worker_capability_available(
            worker_kind="task",
            capability=capability_name,
            protocol=protocol_version,
            available=task_queue.has_active_capability(
                capability_name=capability_name,
                protocol_version=protocol_version,
            ),
        )
        return 0

    _run_maintenance_actions(
        worker_kind="task",
        operation="queue_snapshot",
        actions=(("queue_snapshot", snapshot),),
        telemetry=telemetry,
    )


def _record_ingestion_queue_snapshot(
    *,
    ingestion_queue,
    telemetry: Observability,
    ingestion_max_attempts: int,
) -> None:
    """仅由 ingestion Worker 写 ingestion queue 快照。"""

    def snapshot() -> int:
        for item in ingestion_queue.queue_snapshot(
            max_attempts=ingestion_max_attempts,
            limit=_QUEUE_SNAPSHOT_LIMIT,
        ):
            telemetry.set_ingestion_queue_snapshot(
                worker_kind="ingestion",
                status=item.status,
                depth=item.depth,
                oldest_age_seconds=item.oldest_age_seconds,
            )
        return 0

    _run_maintenance_actions(
        worker_kind="ingestion",
        operation="queue_snapshot",
        actions=(("queue_snapshot", snapshot),),
        telemetry=telemetry,
    )


def run(*, queue: str, watch: bool, poll_seconds: float, max_jobs: int) -> int:
    if poll_seconds <= 0 or max_jobs <= 0:
        raise ValueError("invalid_worker_runtime_limits")
    if queue not in {"task", "ingestion", "projection"}:
        raise ValueError("invalid_worker_queue")
    telemetry = current_observability()
    settings = get_settings()
    task_queue = None
    ingestion_queue = None
    task_worker = None
    task_retention = None
    upload_reconciler = None
    try:
        if queue == "task":
            # Agent 未配置即 fail-fast：宁可进程退出由重启策略兜底，
            # 也不让任务接口产生“已接受永不执行”的任务。
            task_worker = build_task_worker()
            task_retention = build_task_retention()
            current_worker = task_worker
            if telemetry is not None and telemetry.metrics_enabled:
                task_queue = _build_task_queue_snapshot_port()
        elif queue == "ingestion":
            current_worker = build_ingestion_worker()
            upload_reconciler = build_upload_reconciler()
            if telemetry is not None and telemetry.metrics_enabled:
                ingestion_queue = _build_ingestion_queue_snapshot_port()
        else:
            current_worker = build_projection_reconciler()

        processed = 0
        next_recovery_at = 0.0
        while True:
            now = time.monotonic()
            if now >= next_recovery_at:
                if queue == "task":
                    assert task_worker is not None and task_retention is not None
                    _run_task_maintenance(
                        task_worker=task_worker,
                        task_retention=task_retention,
                        telemetry=telemetry,
                    )
                    if task_queue is not None and telemetry is not None:
                        _record_task_queue_snapshot(
                            task_queue=task_queue,
                            telemetry=telemetry,
                            capability_name=settings.task_agent_capability_name,
                            protocol_version=settings.task_agent_protocol_version,
                        )
                elif queue == "ingestion":
                    assert upload_reconciler is not None
                    _run_ingestion_recovery(
                        upload_reconciler=upload_reconciler,
                        telemetry=telemetry,
                    )
                    if ingestion_queue is not None and telemetry is not None:
                        _record_ingestion_queue_snapshot(
                            ingestion_queue=ingestion_queue,
                            telemetry=telemetry,
                            ingestion_max_attempts=current_worker.max_attempts,
                        )
                next_recovery_at = time.monotonic() + _RECOVERY_INTERVAL_SECONDS

            try:
                # Projection repair 由 projection Worker 自己的 run_once 独占。
                result = current_worker.run_once()
            except Exception as exc:
                if telemetry is not None:
                    telemetry.record_worker_error(
                        worker_kind=queue,
                        operation="run_once",
                        error_category=error_category_for(stable_error_code(exc)),
                    )
                raise
            if telemetry is not None:
                telemetry.set_worker_last_loop(
                    worker_kind=queue,
                    timestamp_seconds=time.time(),
                )
                if result.error_code is not None:
                    telemetry.record_worker_error(
                        worker_kind=queue,
                        operation="run_once",
                        error_category=error_category_for(result.error_code),
                    )
            if result.status == "idle":
                if not watch:
                    return 0
                time.sleep(poll_seconds)
                continue

            processed += 1
            if processed >= max_jobs and not watch:
                return 0
    finally:
        try:
            if task_worker is not None:
                task_worker.close()
        finally:
            if task_retention is not None:
                task_retention.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="企业知识库异步摄取工作器")
    parser.add_argument(
        "--queue",
        choices=("task", "ingestion", "projection"),
        required=True,
        help="仅运行指定后台队列",
    )
    parser.add_argument("--watch", action="store_true", help="空闲后持续轮询")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-jobs", type=int, default=100)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    telemetry = Observability(
        enabled=settings.observability_enabled,
        metrics_enabled=settings.metrics_enabled,
        service_name=f"rag-kb-worker-{args.queue}",
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        trace_sample_ratio=settings.otel_trace_sample_ratio,
    )
    from rag_kb.observability import configure_json_logging

    configure_json_logging(service=f"rag-kb-worker-{args.queue}", level=settings.log_level)
    telemetry.start()
    metrics_server = (
        _WorkerMetricsServer(telemetry, port=settings.worker_metrics_port)
        if settings.metrics_enabled
        else None
    )
    if metrics_server is not None:
        metrics_server.start()
    try:
        with telemetry.activate():
            return run(
                queue=args.queue,
                watch=args.watch,
                poll_seconds=args.poll_seconds,
                max_jobs=args.max_jobs,
            )
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            if metrics_server is not None:
                metrics_server.close()
        finally:
            telemetry.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
