"""生产摄取工作器组合根；只在领取到任务后解析、加载模型并写 Qdrant。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import logging
import socket
import threading
import time
from typing import Any
from uuid import uuid4

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
    log_event,
    stable_error_code,
)


logger = logging.getLogger(__name__)

_RECOVERY_BATCH_SIZE = 25
_RECOVERY_INTERVAL_SECONDS = 30.0
_RECOVERY_TIME_BUDGET_SECONDS = 1.0
_UPLOAD_RESERVATION_GRACE = timedelta(seconds=30)


class TaskAgentConfigurationError(RuntimeError):
    pass


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
            except Exception as exc:
                code = stable_error_code(exc)
                if self._telemetry is not None:
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
            logger.error("Research Agent capability 注销失败")


def build_workers(
    *, worker_id: str | None = None
) -> tuple[IngestionWorker, ProjectionReconciler]:
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
    identity = worker_id or f"{socket.gethostname()}-{uuid4().hex[:12]}"
    ingestion = IngestionWorker(
        DocumentRegistryIngestionJobPort(database),
        parser,
        index,
        worker_id=identity,
    )
    reconciler = ProjectionReconciler(
        DatabaseProjectionReconcilePort(database),
        parser,
        index,
        worker_id=f"{identity}-projection-reconciler",
    )
    return ingestion, reconciler


def build_ingestion_worker(*, worker_id: str | None = None) -> IngestionWorker:
    """保留单元测试/一次性摄取调用兼容入口。"""
    return build_workers(worker_id=worker_id)[0]


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
                logger.error("Research Agent capability 启动清理失败")
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


def run(*, watch: bool, poll_seconds: float, max_jobs: int, include_tasks: bool = False) -> int:
    if poll_seconds <= 0 or max_jobs <= 0:
        raise ValueError("invalid_worker_runtime_limits")
    worker, reconciler = build_workers()
    upload_reconciler = build_upload_reconciler()
    task_worker = None
    task_retention = None
    try:
        if include_tasks:
            # Agent 未配置即 fail-fast：宁可进程退出由重启策略兜底，
            # 也不让任务接口产生“已接受永不执行”的任务。
            task_worker = build_task_worker()
            task_retention = build_task_retention()
        queues = [("ingestion", worker), ("projection", reconciler)]
        if task_worker is not None:
            queues.insert(0, ("task", task_worker))
        processed = 0
        queue_index = 0
        idle_queues = 0
        next_recovery_at = 0.0
        while True:
            now = time.monotonic()
            if now >= next_recovery_at:
                if task_worker is not None:
                    try:
                        assert task_retention is not None
                        expired_tools = task_worker.expire_stale_tool_calls(
                            limit=_RECOVERY_BATCH_SIZE
                        )
                        expired_queries = task_worker.expire_stale_knowledge_queries(
                            limit=_RECOVERY_BATCH_SIZE
                        )
                        expired_waiting = task_retention.expire_waiting_tasks(
                            limit=_RECOVERY_BATCH_SIZE
                        )
                        purged_results = task_retention.purge_expired_tool_results(
                            limit=_RECOVERY_BATCH_SIZE
                        )
                        purged_content = task_retention.purge_finished_task_content(
                            limit=_RECOVERY_BATCH_SIZE
                        )
                        purged_orphans = task_retention.purge_orphaned_result_objects(
                            limit=_RECOVERY_BATCH_SIZE
                        )
                        if (
                            expired_tools
                            or expired_queries
                            or expired_waiting
                            or purged_results
                            or purged_content
                            or purged_orphans
                        ):
                            logger.info(
                                "过期运行恢复结束 tools=%s knowledge_queries=%s "
                                "waiting_tasks=%s purged_tool_results=%s "
                                "purged_task_content=%s purged_orphans=%s",
                                expired_tools,
                                expired_queries,
                                expired_waiting,
                                purged_results,
                                purged_content,
                                purged_orphans,
                            )
                    except Exception:
                        logger.error("运行恢复失败 error_code=runtime_sweep_failed")
                try:
                    upload_result = upload_reconciler.reconcile(
                        older_than=datetime.now(UTC) - _UPLOAD_RESERVATION_GRACE,
                        limit=_RECOVERY_BATCH_SIZE,
                        time_budget_seconds=_RECOVERY_TIME_BUDGET_SECONDS,
                    )
                    if upload_result.scanned:
                        logger.info(
                            "上传预留恢复结束 scanned=%s recovered=%s missing=%s mismatched=%s failed=%s",
                            upload_result.scanned,
                            upload_result.recovered,
                            upload_result.missing_objects,
                            upload_result.content_mismatches,
                            upload_result.failed,
                        )
                except Exception:
                    logger.error("上传预留恢复失败 error_code=upload_reconciliation_failed")
                next_recovery_at = time.monotonic() + _RECOVERY_INTERVAL_SECONDS

            queue_name, current_worker = queues[queue_index]
            queue_index = (queue_index + 1) % len(queues)
            result = current_worker.run_once()
            if result.status == "idle":
                idle_queues += 1
                if idle_queues < len(queues):
                    continue
                idle_queues = 0
                if not watch:
                    return 0
                time.sleep(poll_seconds)
                continue

            idle_queues = 0
            processed += 1
            if queue_name == "task":
                logger.info(
                    "通用任务结束 status=%s task_id=%s error_code=%s",
                    result.status,
                    result.task_id,
                    result.error_code,
                )
            elif queue_name == "projection":
                logger.info(
                    "投影收敛结束 status=%s document_id=%s",
                    result.status,
                    result.document_id,
                )
            else:
                logger.info("摄取任务结束 status=%s job_id=%s", result.status, result.job_id)
            if processed >= max_jobs and not watch:
                return 0
    finally:
        try:
            if task_worker is not None:
                task_worker.close()
        finally:
            if task_retention is not None:
                task_retention.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="企业知识库异步摄取工作器")
    parser.add_argument("--watch", action="store_true", help="空闲后持续轮询")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-jobs", type=int, default=100)
    parser.add_argument("--include-tasks", action="store_true", help="同时执行通用任务队列")
    args = parser.parse_args()
    settings = get_settings()
    telemetry = Observability(
        enabled=settings.observability_enabled,
        metrics_enabled=settings.metrics_enabled,
        service_name="rag-kb-worker",
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        trace_sample_ratio=settings.otel_trace_sample_ratio,
    )
    from rag_kb.observability import configure_json_logging

    configure_json_logging(service="rag-kb-worker", level=settings.log_level)
    telemetry.start()
    try:
        with telemetry.activate():
            return run(
                watch=args.watch,
                poll_seconds=args.poll_seconds,
                max_jobs=args.max_jobs,
                include_tasks=args.include_tasks,
            )
    except KeyboardInterrupt:
        return 0
    finally:
        telemetry.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
