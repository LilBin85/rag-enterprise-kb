"""生产摄取工作器组合根；只在领取到任务后解析、加载模型并写 Qdrant。"""

from __future__ import annotations

import argparse
import logging
import socket
import time
from uuid import uuid4

from qdrant_client import QdrantClient

from rag_kb.application.ingestion_worker import (
    DocumentRegistryIngestionJobPort,
    IngestionWorker,
)
from rag_kb.application.projection_reconciler import ProjectionReconciler
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.pending_action_store import PendingActionStore
from rag_kb.application.task_worker import TaskWorker
from rag_kb.application.tool_execution_service import ToolExecutionService
from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.config import get_settings, resolve_retrieval_limits
from rag_kb.embedder import BgeM3Embedder
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.database.task_execution import (
    DatabaseTaskAuthContextPort,
    DatabaseTaskEvidencePort,
    DatabaseTaskPlanBindingPort,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
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
from rag_kb.infrastructure.llm.task_planner import DeepSeekTaskPlanner, TaskPlannerModelError
from rag_kb.infrastructure.tools.factory import build_tool_registry


logger = logging.getLogger(__name__)


class _UnavailableTaskPlanner:
    """模型配置缺失时仅关闭任务本身，绝不让摄取循环停摆。"""

    def __init__(self, code: str) -> None:
        self._code = code

    def next_action(self, state, observations=()):
        raise TaskPlannerModelError(self._code)


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


def build_task_worker(*, worker_id: str | None = None, planner=None) -> TaskWorker:
    """构造独立通用任务 worker；只使用固定注册表与持久 pending action。"""
    settings = get_settings()
    database = build_database(settings.database_url)
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
    )
    identity = worker_id or f"{socket.gethostname()}-task-{uuid4().hex[:12]}"
    evidence_validator = DatabaseTaskEvidencePort(database)
    approvals = ApprovalService(database, evidence_validator=evidence_validator)
    return TaskWorker(
        DatabaseTaskRepository(database),
        planner or DeepSeekTaskPlanner(settings, registry),
        ToolExecutionService(
            database,
            registry,
            approvals,
            evidence_validator=evidence_validator,
        ),
        registry,
        approvals,
        DatabaseTaskAuthContextPort(database),
        DatabaseTaskPlanBindingPort(database),
        evidence_validator,
        PendingActionStore(LocalObjectStore(settings.object_store_root)),
        worker_id=identity,
    )


def run(*, watch: bool, poll_seconds: float, max_jobs: int, include_tasks: bool = False) -> int:
    if poll_seconds <= 0 or max_jobs <= 0:
        raise ValueError("invalid_worker_runtime_limits")
    worker, reconciler = build_workers()
    task_worker = None
    if include_tasks:
        try:
            task_worker = build_task_worker()
        except TaskPlannerModelError as exc:
            # 模型不可用时，领取到的任务会由 TaskWorker 终止为安全失败；摄取仍照常运行。
            logger.error("通用任务模型不可用 error_code=%s", exc.code)
            task_worker = build_task_worker(planner=_UnavailableTaskPlanner(exc.code))
    processed = 0
    while True:
        if task_worker is not None:
            task_result = task_worker.run_once()
            if task_result.status != "idle":
                processed += 1
                logger.info(
                    "通用任务结束 status=%s task_id=%s error_code=%s",
                    task_result.status,
                    task_result.task_id,
                    task_result.error_code,
                )
                if processed >= max_jobs and not watch:
                    return 0
                continue
        result = worker.run_once()
        if result.status == "idle":
            repair = reconciler.run_once()
            if repair.status != "idle":
                processed += 1
                logger.info(
                    "投影收敛结束 status=%s document_id=%s",
                    repair.status,
                    repair.document_id,
                )
                if processed >= max_jobs and not watch:
                    return 0
                continue
            if not watch:
                return 0
            time.sleep(poll_seconds)
            continue
        processed += 1
        logger.info("摄取任务结束 status=%s job_id=%s", result.status, result.job_id)
        if processed >= max_jobs and not watch:
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="企业知识库异步摄取工作器")
    parser.add_argument("--watch", action="store_true", help="空闲后持续轮询")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-jobs", type=int, default=100)
    parser.add_argument("--include-tasks", action="store_true", help="同时执行通用任务队列")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        return run(
            watch=args.watch,
            poll_seconds=args.poll_seconds,
            max_jobs=args.max_jobs,
            include_tasks=args.include_tasks,
        )
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
