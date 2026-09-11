"""真实 PostgreSQL + Qdrant 的投影世代延迟写与最终收敛故障注入。"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

import pytest
from qdrant_client.http import models
from sqlalchemy import text

from rag_kb.application.document_service import DocumentService
from rag_kb.application.ingestion_worker import IngestionWorker
from rag_kb.application.ports.indexing import ClaimedIngestionJob, DocumentIndexError
from rag_kb.application.projection_reconciler import ProjectionReconciler
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.indexing import IndexReceipt, ParsedDocument
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.document_lifecycle import (
    DatabaseDocumentLifecycle,
    DocumentRegistryIngestionJobPort,
)
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.database.index_projection_metadata import (
    DatabaseDocumentIndexMetadataPort,
)
from rag_kb.infrastructure.database.models import (
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
)
from rag_kb.infrastructure.database.projection_repository import (
    DatabaseProjectionReconcilePort,
)
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import (
    QdrantAuthorizedSearch,
)
from rag_kb.infrastructure.vectorstore.qdrant_document_index import QdrantDocumentIndex


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


class DeterministicEmbedder:
    """固定二维向量，让故障测试只观察权限与投影世代。"""

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class CapturingDocumentIndex(QdrantDocumentIndex):
    """保留生产适配器实际收到的已分配世代任务和回执。"""

    indexed_job: ClaimedIngestionJob | None = None
    receipt: IndexReceipt | None = None

    def index(
        self,
        job: ClaimedIngestionJob,
        parsed_document: ParsedDocument,
    ) -> IndexReceipt:
        receipt = super().index(job, parsed_document)
        self.indexed_job = job
        self.receipt = receipt
        return receipt


class BlockingSetPayloadClient:
    """只阻塞下一次 set_payload，并把其余调用原样转给真实客户端。"""

    def __init__(self, client) -> None:
        self._client = client
        self._guard = Lock()
        self._armed = False
        self.entered = Event()
        self.release = Event()

    def arm(self) -> None:
        with self._guard:
            self._armed = True
            self.entered.clear()
            self.release.clear()

    def set_payload(self, *args, **kwargs):
        with self._guard:
            should_block = self._armed
            self._armed = False
        if should_block:
            self.entered.set()
            if not self.release.wait(timeout=30):
                raise TimeoutError("delayed_old_publish_not_released")
        return self._client.set_payload(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._client, name)


def _auth(tenant_id: str, user_id: str) -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        attributes={
            "permissions": ["document:write", "document:read", "knowledge:read"]
        },
        request_id=f"projection-convergence-{user_id}",
    )


def _document_payloads(client, collection: str, tenant_id: str, document_id: str) -> list[dict]:
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
        ]
    )
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        payloads.extend(dict(point.payload or {}) for point in points)
        if offset is None:
            return payloads


def _run_ingestion(
    *,
    database,
    parser: LocalDocumentParser,
    index: QdrantDocumentIndex,
    worker_id: str,
    expected_job_id: str,
) -> None:
    result = IngestionWorker(
        DocumentRegistryIngestionJobPort(database),
        parser,
        index,
        worker_id=worker_id,
        lease_seconds=60,
    ).run_once()
    assert result.status == "succeeded", result.error_code
    assert result.job_id == expected_job_id


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_delayed_old_publish_is_exactly_filtered_and_reconciler_converges(
    tmp_path: Path,
) -> None:
    """旧 G1 写晚于新 G2 applied；SQL/Qdrant 只放行 G2，审计最终收敛到 G3。"""

    assert POSTGRES_TEST_URL is not None
    client = None
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=5)
        client.get_collections()
    except Exception:
        if client is not None:
            client.close()
        pytest.skip("qdrant_unavailable")

    suffix = uuid4().hex
    tenant_id = f"projection-convergence-{suffix}"
    user_id = f"owner-{suffix}"
    document_id = f"document-{suffix}"
    collection = f"projection_convergence_{suffix}"
    # Windows 对传统路径长度仍有限制；存储键本身已包含唯一租户/版本哈希。
    object_root = tmp_path / "o"
    database = build_database(POSTGRES_TEST_URL)
    delayed_client = BlockingSetPayloadClient(client)
    executor: ThreadPoolExecutor | None = None
    old_publish: Future[None] | None = None

    try:
        assert database.engine.dialect.name == "postgresql"
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO tenants (tenant_id, name, active, created_at) "
                    "VALUES (:tenant, :name, TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant": tenant_id, "name": tenant_id},
            )
            connection.execute(
                text(
                    "INSERT INTO principals "
                    "(tenant_id, user_id, roles, attributes, active, created_at) "
                    "VALUES (:tenant, :user, CAST('[]' AS jsonb), "
                    "CAST('{}' AS jsonb), TRUE, CURRENT_TIMESTAMP)"
                ),
                {"tenant": tenant_id, "user": user_id},
            )

        auth = _auth(tenant_id, user_id)
        object_store = LocalObjectStore(object_root)
        parser = LocalDocumentParser(object_store)
        service = DocumentService(DatabaseDocumentLifecycle(database), object_store)
        metadata = DatabaseDocumentIndexMetadataPort(database)
        embedder = DeterministicEmbedder()

        first = service.register_upload(
            document_id=document_id,
            title="Projection convergence policy",
            department="security",
            source_file_name="policy.md",
            content=b"# Policy v1\n\nold generation travel rule",
            idempotency_key=f"upload-v1-{suffix}",
            auth=auth,
        )
        first_index = CapturingDocumentIndex(
            client,
            embedder,
            metadata,
            collection=collection,
        )
        _run_ingestion(
            database=database,
            parser=parser,
            index=first_index,
            worker_id=f"worker-v1-{suffix}",
            expected_job_id=first.ingestion_job_id,
        )
        old_job = first_index.indexed_job
        old_receipt = first_index.receipt
        assert old_job is not None and old_receipt is not None
        assert old_job.projection_generation == old_receipt.projection_generation

        delayed_index = QdrantDocumentIndex(
            delayed_client,
            embedder,
            metadata,
            collection=collection,
        )
        delayed_client.arm()
        executor = ThreadPoolExecutor(max_workers=1)
        old_publish = executor.submit(
            delayed_index.publish,
            old_job,
            old_receipt,
            lifecycle_status="active",
        )
        assert delayed_client.entered.wait(timeout=10), "old_publish_did_not_reach_set_payload"

        second = service.register_upload(
            document_id=document_id,
            title="Projection convergence policy",
            department="security",
            source_file_name="policy.md",
            content=b"# Policy v2\n\nnew generation travel rule",
            idempotency_key=f"upload-v2-{suffix}",
            auth=auth,
            expect_existing=True,
        )
        second_index = CapturingDocumentIndex(
            client,
            embedder,
            metadata,
            collection=collection,
        )
        _run_ingestion(
            database=database,
            parser=parser,
            index=second_index,
            worker_id=f"worker-v2-{suffix}",
            expected_job_id=second.ingestion_job_id,
        )
        new_job = second_index.indexed_job
        assert new_job is not None and new_job.projection_generation is not None
        new_generation = new_job.projection_generation
        assert old_receipt.projection_generation < new_generation

        with database.sessions() as session:
            document = session.get(DocumentRow, (tenant_id, document_id))
            version = session.get(DocumentVersionRow, second.version_id)
            job = session.get(IngestionJobRow, second.ingestion_job_id)
            assert document is not None and version is not None and job is not None
            assert document.applied_projection_version_id == second.version_id
            assert document.applied_projection_generation == new_generation
            assert version.applied_projection_generation == new_generation
            assert job.projection_generation == new_generation
            assert job.status == "succeeded"

        delayed_client.release.set()
        with pytest.raises(DocumentIndexError, match="publish_verification_failed"):
            old_publish.result(timeout=30)

        payloads_after_late_write = _document_payloads(
            client,
            collection,
            tenant_id,
            document_id,
        )
        assert not any(
            payload.get("projection_generation") == old_receipt.projection_generation
            and payload.get("current") is True
            for payload in payloads_after_late_write
        ), "retired points must not be recreated by a delayed payload update"
        new_payloads = [
            payload
            for payload in payloads_after_late_write
            if payload.get("version_id") == second.version_id
            and payload.get("projection_generation") == new_generation
        ]
        assert new_payloads
        assert all(
            payload.get("lifecycle_status") == "active" and payload.get("current") is True
            for payload in new_payloads
        ), "the delayed old write downgraded the newer generation"

        authorization = DatabaseAuthorizationPort(database)
        authorized = authorization.authorized_active_versions(auth)
        assert authorized.version_ids == frozenset({second.version_id})
        assert authorized.projection_generations == frozenset(
            {(second.version_id, new_generation)}
        )

        vectors = QdrantAuthorizedSearch(client, embedder, collection=collection)
        candidates = vectors.search_dense(
            "travel rule",
            tenant_id=tenant_id,
            version_ids=authorized.version_ids,
            projection_generations=authorized.projection_generations,
            allowed_principals=auth.acl_subjects,
            limit=10,
        )
        assert candidates
        assert {
            (candidate.version_id, candidate.projection_generation)
            for candidate in candidates
        } == {(second.version_id, new_generation)}

        # 注入已绑定精确世代异常，确保审计只在确认不一致后才重建。
        client.set_payload(
            collection_name=collection,
            payload={"current": False},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id", match=models.MatchValue(value=tenant_id)
                    ),
                    models.FieldCondition(
                        key="version_id", match=models.MatchValue(value=second.version_id)
                    ),
                    models.FieldCondition(
                        key="projection_generation",
                        match=models.MatchValue(value=new_generation),
                    ),
                ]
            ),
            wait=True,
        )

        reconcile_now = datetime(1970, 1, 2, tzinfo=UTC)
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE documents SET projection_reconcile_after = :due "
                    "WHERE tenant_id = :tenant AND document_id = :document"
                ),
                {
                    "due": reconcile_now - timedelta(seconds=1),
                    "tenant": tenant_id,
                    "document": document_id,
                },
            )
            due_documents = connection.execute(
                text(
                    "SELECT tenant_id, document_id FROM documents "
                    "WHERE projection_reconcile_after <= :now"
                ),
                {"now": reconcile_now},
            ).all()
        assert [(row.tenant_id, row.document_id) for row in due_documents] == [
            (tenant_id, document_id)
        ], (
            "refusing to let the test reconciler claim a non-random document"
        )

        reconcile_index = CapturingDocumentIndex(
            client,
            embedder,
            metadata,
            collection=collection,
        )
        reconcile_result = ProjectionReconciler(
            DatabaseProjectionReconcilePort(database, now=lambda: reconcile_now),
            parser,
            reconcile_index,
            worker_id=f"reconciler-{suffix}",
            lease_seconds=60,
            audit_interval=timedelta(minutes=5),
        ).run_once()
        assert reconcile_result.status == "succeeded", reconcile_result.error_code
        reconcile_job = reconcile_index.indexed_job
        assert reconcile_job is not None and reconcile_job.projection_generation is not None
        converged_generation = reconcile_job.projection_generation
        assert converged_generation > new_generation

        converged_scope = authorization.authorized_active_versions(auth)
        assert converged_scope.version_ids == frozenset({second.version_id})
        assert converged_scope.projection_generations == frozenset(
            {(second.version_id, converged_generation)}
        )
        converged_candidates = vectors.search_dense(
            "travel rule",
            tenant_id=tenant_id,
            version_ids=converged_scope.version_ids,
            projection_generations=converged_scope.projection_generations,
            allowed_principals=auth.acl_subjects,
            limit=10,
        )
        assert converged_candidates
        assert {
            (candidate.version_id, candidate.projection_generation)
            for candidate in converged_candidates
        } == {(second.version_id, converged_generation)}

        converged_payloads = _document_payloads(client, collection, tenant_id, document_id)
        current_payloads = [
            payload for payload in converged_payloads if payload.get("current") is True
        ]
        assert current_payloads
        assert {
            (
                payload.get("version_id"),
                payload.get("projection_generation"),
                payload.get("lifecycle_status"),
            )
            for payload in current_payloads
        } == {(second.version_id, converged_generation, "active")}
        assert all(
            payload.get("current") is False
            for payload in converged_payloads
            if payload.get("projection_generation") < converged_generation
        )
    finally:
        delayed_client.release.set()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
        finally:
            try:
                with database.engine.begin() as connection:
                    connection.execute(
                        text("DELETE FROM tenants WHERE tenant_id = :tenant"),
                        {"tenant": tenant_id},
                    )
            finally:
                database.engine.dispose()
                client.close()
                shutil.rmtree(object_root, ignore_errors=True)
