"""PostgreSQL/SQLite 语义一致的文档注册表与摄取任务仓储。"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from uuid import uuid4

from sqlalchemy import Select, and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from rag_kb.application.ports.documents import (
    DocumentDeletionReservation,
    DocumentRegistryConflict,
    DocumentSnapshot,
    IdempotencyConflict,
    PendingObjectReservation,
    RegisteredUpload,
)
from rag_kb.application.ports.indexing import IngestionQueueSnapshot
from rag_kb.domain.indexing import IndexReceipt, ParsedDocument, compute_chunk_manifest_digest
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IdempotencyRecordRow,
    IngestionJobRow,
    PrincipalRow,
)
from rag_kb.observability import current_trace_headers


class DocumentRegistryRepository:
    def principal_exists(self, session: Session, *, tenant_id: str, user_id: str) -> bool:
        return session.scalar(
            select(PrincipalRow.user_id).where(
                PrincipalRow.tenant_id == tenant_id,
                PrincipalRow.user_id == user_id,
                PrincipalRow.active.is_(True),
            )
        ) is not None

    def reserve_pending_upload(
        self,
        session: Session,
        *,
        tenant_id: str,
        user_id: str,
        document_id: str,
        title: str,
        department: str,
        source_file_name: str,
        content_hash: str,
        idempotency_key: str,
        request_hash: str,
        expect_existing: bool,
        allow_existing_recovery: bool = False,
        index_version: str = "kb-v1",
    ) -> tuple[RegisteredUpload, bool]:
        scope = "document_upload"
        version_id = uuid4().hex
        job_id = uuid4().hex
        reference = f"{version_id}:{job_id}"
        claimed_record_id = self._claim_idempotency_key(
            session,
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )
        if claimed_record_id is None:
            existing_idempotency = session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.tenant_id == tenant_id,
                    IdempotencyRecordRow.user_id == user_id,
                    IdempotencyRecordRow.scope == scope,
                    IdempotencyRecordRow.idempotency_key == idempotency_key,
                )
            )
            if existing_idempotency is None:
                raise IdempotencyConflict("idempotency_claim_lost")
            if existing_idempotency.request_hash != request_hash:
                raise IdempotencyConflict("idempotency_key_reused_with_different_request")
            if not existing_idempotency.result_reference:
                raise IdempotencyConflict("idempotency_result_unknown")
            if existing_idempotency.status not in {"in_progress", "succeeded"}:
                raise IdempotencyConflict("idempotency_request_not_resumable")
            reservation = self._load_reference(
                session,
                tenant_id,
                existing_idempotency.result_reference,
                version_id=existing_idempotency.result_version_id,
                job_id=existing_idempotency.result_job_id,
            )
            return reservation, existing_idempotency.status == "in_progress"

        document = session.scalar(
            select(DocumentRow)
            .where(DocumentRow.tenant_id == tenant_id, DocumentRow.document_id == document_id)
            .with_for_update()
        )
        if document is None and expect_existing:
            raise DocumentRegistryConflict("document_not_found")
        if document is not None and not expect_existing:
            existing = self._latest_upload_for_document(
                session,
                tenant_id=tenant_id,
                document_id=document_id,
            )
            if existing is not None and allow_existing_recovery:
                latest_version, latest_job = existing
                same_upload = (
                    document.title == title
                    and document.department == department
                    and latest_version.source_file_name == source_file_name
                    and latest_version.content_hash == content_hash
                )
                existing_result = self._registered_upload_from_rows(
                    latest_version,
                    latest_job,
                    outcome="existing_recovered" if same_upload else "existing_conflict",
                )
                if same_upload:
                    self._bind_existing_result(
                        session,
                        record_id=claimed_record_id,
                        upload=latest_version,
                        job=latest_job,
                    )
                    return existing_result, False
                raise DocumentRegistryConflict(
                    "document_already_exists",
                    recovery=existing_result,
                )
            raise DocumentRegistryConflict("document_already_exists")
        if document is None:
            now = datetime.now(UTC)
            values = {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "title": title,
                "department": department,
                "status": "pending",
                "created_by": user_id,
                "created_at": now,
                "updated_at": now,
            }
            dialect = session.get_bind().dialect.name
            if dialect == "postgresql":
                statement = postgresql_insert(DocumentRow).values(**values)
            elif dialect == "sqlite":
                statement = sqlite_insert(DocumentRow).values(**values)
            else:
                raise RuntimeError("unsupported_database_dialect")
            inserted_document = session.execute(
                statement.on_conflict_do_nothing(
                    index_elements=["tenant_id", "document_id"]
                ).returning(DocumentRow.document_id)
            ).scalar_one_or_none()
            if inserted_document is None:
                # 空隙无法通过 SELECT FOR UPDATE 锁定；原子插入把并发唯一冲突
                # 转换成稳定领域错误，避免向 API 泄露驱动层 IntegrityError。
                raise DocumentRegistryConflict("document_already_exists")
            grant_values = {
                "grant_id": uuid4().hex,
                "tenant_id": tenant_id,
                "resource_type": "document",
                "resource_id": document_id,
                "subject_type": "user",
                "subject_id": user_id,
                "permissions": ["knowledge:read", "document:read", "document:write"],
                "created_at": now,
            }
            if dialect == "postgresql":
                grant_statement = postgresql_insert(ACLGrantRow).values(**grant_values)
            else:
                grant_statement = sqlite_insert(ACLGrantRow).values(**grant_values)
            session.execute(
                grant_statement.on_conflict_do_nothing(
                    index_elements=[
                        "tenant_id",
                        "resource_type",
                        "resource_id",
                        "subject_type",
                        "subject_id",
                    ]
                )
            )

        current_version = session.scalar(
            select(func.max(DocumentVersionRow.version)).where(
                DocumentVersionRow.tenant_id == tenant_id,
                DocumentVersionRow.document_id == document_id,
            )
        )
        version = int(current_version or 0) + 1
        suffix = PurePosixPath(source_file_name).suffix.lower()
        # 128 位截断摘要在隐藏原始标识的同时，把 Windows 本地对象存储路径
        # 控制在传统 MAX_PATH 限制内；版本 UUID 继续提供对象级唯一性。
        tenant_segment = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:32]
        document_segment = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:32]
        source_object_key = (
            f"t/{tenant_segment}/d/{document_segment}/v/{version_id}/source{suffix}"
        )
        pending_traceparent, pending_tracestate = current_trace_headers()
        session.add(
            DocumentVersionRow(
                version_id=version_id,
                tenant_id=tenant_id,
                document_id=document_id,
                version=version,
                lifecycle_status="awaiting_object",
                source_object_key=source_object_key,
                source_file_name=source_file_name,
                content_hash=content_hash,
                index_version=index_version,
            )
        )
        session.add(
            IngestionJobRow(
                job_id=job_id,
                tenant_id=tenant_id,
                version_id=version_id,
                status="awaiting_object",
                available_at=datetime.now(UTC),
                pending_traceparent=pending_traceparent,
                pending_tracestate=pending_tracestate,
            )
        )
        # 幂等占位必须先于业务实体原子争抢，但显式结果外键只能在实体落库后回填。
        # 两次 flush 仍处于同一事务中，因此其他会话不可能观察到半成品绑定。
        session.flush()
        updated_rows = session.execute(
            update(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.record_id == claimed_record_id)
            .values(
                result_reference=reference,
                result_version_id=version_id,
                result_job_id=job_id,
                updated_at=datetime.now(UTC),
            )
        ).rowcount
        if updated_rows != 1:
            raise IdempotencyConflict("idempotency_claim_lost")
        return RegisteredUpload(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            version=version,
            job_id=job_id,
            source_object_key=source_object_key,
            lifecycle_status="awaiting_object",
            ingestion_status="awaiting_object",
        ), True

    @staticmethod
    def _claim_idempotency_key(
        session: Session,
        *,
        tenant_id: str,
        user_id: str,
        scope: str,
        idempotency_key: str,
        request_hash: str,
    ) -> str | None:
        now = datetime.now(UTC)
        values = {
            "record_id": uuid4().hex,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "scope": scope,
            "idempotency_key": idempotency_key,
            "request_hash": request_hash,
            "status": "in_progress",
            "created_at": now,
            "updated_at": now,
        }
        dialect = session.get_bind().dialect.name
        conflict_columns = ["tenant_id", "scope", "user_id", "idempotency_key"]
        if dialect == "postgresql":
            statement = postgresql_insert(IdempotencyRecordRow).values(**values)
        elif dialect == "sqlite":
            statement = sqlite_insert(IdempotencyRecordRow).values(**values)
        else:
            raise RuntimeError("unsupported_database_dialect")
        return session.execute(
            statement.on_conflict_do_nothing(index_elements=conflict_columns).returning(
                IdempotencyRecordRow.record_id
            )
        ).scalar_one_or_none()

    def mark_object_ready(
        self,
        session: Session,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> RegisteredUpload:
        idempotency = session.scalar(
            select(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.tenant_id == tenant_id,
                IdempotencyRecordRow.user_id == user_id,
                IdempotencyRecordRow.scope == "document_upload",
                IdempotencyRecordRow.idempotency_key == idempotency_key,
            )
            .with_for_update()
        )
        if idempotency is None or idempotency.request_hash != request_hash:
            raise IdempotencyConflict("idempotency_binding_mismatch")
        if not idempotency.result_reference:
            raise IdempotencyConflict("idempotency_result_unknown")
        reservation = self._load_reference(
            session,
            tenant_id,
            idempotency.result_reference,
            version_id=idempotency.result_version_id,
            job_id=idempotency.result_job_id,
        )
        if idempotency.status == "succeeded":
            return reservation
        if idempotency.status != "in_progress":
            raise IdempotencyConflict("idempotency_request_not_resumable")
        version = self.get_version(session, tenant_id, reservation.version_id)
        job = self.get_job(session, tenant_id, reservation.job_id)
        if version is None or job is None:
            raise IdempotencyConflict("idempotency_result_missing")
        if version.lifecycle_status != "awaiting_object" or job.status != "awaiting_object":
            # PostgreSQL 的行锁会串行化这里；SQLite 忽略 FOR UPDATE，因而并发
            # 同键请求可能缓存 in_progress，却已在后续查询看到另一事务推进后的
            # pending 状态。强制取得新快照，只接受完整 succeeded 绑定。
            session.expire_all()
            refreshed = session.scalar(
                select(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.tenant_id == tenant_id,
                    IdempotencyRecordRow.user_id == user_id,
                    IdempotencyRecordRow.scope == "document_upload",
                    IdempotencyRecordRow.idempotency_key == idempotency_key,
                )
            )
            if (
                refreshed is not None
                and refreshed.request_hash == request_hash
                and refreshed.status == "succeeded"
                and refreshed.result_reference
            ):
                return self._load_reference(
                    session,
                    tenant_id,
                    refreshed.result_reference,
                    version_id=refreshed.result_version_id,
                    job_id=refreshed.result_job_id,
                )
            raise DocumentRegistryConflict("upload_reservation_state_invalid")
        now = datetime.now(UTC)
        version.lifecycle_status = "pending"
        job.status = "pending"
        job.available_at = now
        job.updated_at = now
        idempotency.status = "succeeded"
        idempotency.updated_at = now
        session.flush()
        return RegisteredUpload(
            tenant_id=tenant_id,
            document_id=version.document_id,
            version_id=version.version_id,
            version=version.version,
            job_id=job.job_id,
            source_object_key=version.source_object_key,
            lifecycle_status=version.lifecycle_status,
            ingestion_status=job.status,
        )

    def _load_reference(
        self,
        session: Session,
        tenant_id: str,
        reference: str,
        *,
        version_id: str | None = None,
        job_id: str | None = None,
        outcome: str = "created",
    ) -> RegisteredUpload:
        try:
            reference_version_id, reference_job_id = reference.split(":", 1)
        except ValueError as exc:
            raise IdempotencyConflict("invalid_idempotency_result_reference") from exc
        if (
            version_id is None
            or job_id is None
            or version_id != reference_version_id
            or job_id != reference_job_id
        ):
            raise IdempotencyConflict("idempotency_result_binding_mismatch")
        version = session.scalar(
            select(DocumentVersionRow).where(
                DocumentVersionRow.tenant_id == tenant_id,
                DocumentVersionRow.version_id == version_id,
            )
        )
        job = session.scalar(
            select(IngestionJobRow).where(
                IngestionJobRow.tenant_id == tenant_id,
                IngestionJobRow.job_id == job_id,
                IngestionJobRow.version_id == version_id,
            )
        )
        if version is None or job is None:
            raise IdempotencyConflict("idempotency_result_missing")
        return RegisteredUpload(
            tenant_id=tenant_id,
            document_id=version.document_id,
            version_id=version.version_id,
            version=version.version,
            job_id=job.job_id,
            source_object_key=version.source_object_key,
            lifecycle_status=version.lifecycle_status,
            ingestion_status=job.status,
            outcome=outcome,
        )

    @staticmethod
    def _registered_upload_from_rows(
        version: DocumentVersionRow,
        job: IngestionJobRow,
        *,
        outcome: str,
    ) -> RegisteredUpload:
        return RegisteredUpload(
            tenant_id=version.tenant_id,
            document_id=version.document_id,
            version_id=version.version_id,
            version=version.version,
            job_id=job.job_id,
            source_object_key=version.source_object_key,
            lifecycle_status=version.lifecycle_status,
            ingestion_status=job.status,
            outcome=outcome,
        )

    @staticmethod
    def _bind_existing_result(
        session: Session,
        *,
        record_id: str,
        upload: DocumentVersionRow,
        job: IngestionJobRow,
    ) -> None:
        reference = f"{upload.version_id}:{job.job_id}"
        updated_rows = session.execute(
            update(IdempotencyRecordRow)
            .where(IdempotencyRecordRow.record_id == record_id)
            .values(
                result_reference=reference,
                result_version_id=upload.version_id,
                result_job_id=job.job_id,
                status="succeeded",
                updated_at=datetime.now(UTC),
            )
        ).rowcount
        if updated_rows != 1:
            raise IdempotencyConflict("idempotency_claim_lost")

    @staticmethod
    def _latest_upload_for_document(
        session: Session,
        *,
        tenant_id: str,
        document_id: str,
    ) -> tuple[DocumentVersionRow, IngestionJobRow] | None:
        version = session.scalar(
            select(DocumentVersionRow)
            .where(
                DocumentVersionRow.tenant_id == tenant_id,
                DocumentVersionRow.document_id == document_id,
            )
            .order_by(DocumentVersionRow.version.desc())
            .limit(1)
        )
        if version is None:
            return None
        job = session.scalar(
            select(IngestionJobRow).where(
                IngestionJobRow.tenant_id == tenant_id,
                IngestionJobRow.version_id == version.version_id,
            )
        )
        if job is None:
            return None
        return version, job

    def list_pending_object_reservations(
        self,
        session: Session,
        *,
        older_than: datetime,
        limit: int = 100,
    ) -> tuple[PendingObjectReservation, ...]:
        if limit <= 0 or limit > 1000:
            raise ValueError("invalid_reconciliation_limit")
        rows = session.execute(
            select(IdempotencyRecordRow, DocumentVersionRow)
            .join(
                DocumentVersionRow,
                and_(
                    DocumentVersionRow.tenant_id == IdempotencyRecordRow.tenant_id,
                    DocumentVersionRow.version_id == IdempotencyRecordRow.result_version_id,
                ),
            )
            .join(
                IngestionJobRow,
                and_(
                    IngestionJobRow.tenant_id == IdempotencyRecordRow.tenant_id,
                    IngestionJobRow.job_id == IdempotencyRecordRow.result_job_id,
                    IngestionJobRow.version_id == IdempotencyRecordRow.result_version_id,
                ),
            )
            .where(
                IdempotencyRecordRow.scope == "document_upload",
                IdempotencyRecordRow.status == "in_progress",
                IdempotencyRecordRow.updated_at <= older_than,
                DocumentVersionRow.lifecycle_status == "awaiting_object",
                IngestionJobRow.status == "awaiting_object",
            )
            .order_by(IdempotencyRecordRow.updated_at, IdempotencyRecordRow.record_id)
            .limit(limit)
        ).all()
        return tuple(
            PendingObjectReservation(
                tenant_id=idempotency.tenant_id,
                user_id=idempotency.user_id,
                idempotency_key=idempotency.idempotency_key,
                request_hash=idempotency.request_hash,
                source_object_key=version.source_object_key,
                content_hash=version.content_hash,
            )
            for idempotency, version in rows
        )

    def defer_pending_object_reservation(
        self,
        session: Session,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> None:
        """把本次未恢复记录移到队尾，避免坏满批次永久饿死后续记录。"""
        updated = session.execute(
            update(IdempotencyRecordRow)
            .where(
                IdempotencyRecordRow.tenant_id == tenant_id,
                IdempotencyRecordRow.user_id == user_id,
                IdempotencyRecordRow.scope == "document_upload",
                IdempotencyRecordRow.idempotency_key == idempotency_key,
                IdempotencyRecordRow.request_hash == request_hash,
                IdempotencyRecordRow.status == "in_progress",
            )
            .values(updated_at=datetime.now(UTC))
        ).rowcount
        if updated != 1:
            raise IdempotencyConflict("idempotency_binding_mismatch")

    def get_version(self, session: Session, tenant_id: str, version_id: str) -> DocumentVersionRow | None:
        return session.scalar(
            select(DocumentVersionRow).where(
                DocumentVersionRow.tenant_id == tenant_id,
                DocumentVersionRow.version_id == version_id,
            )
        )

    def get_document(
        self,
        session: Session,
        tenant_id: str,
        document_id: str,
    ) -> DocumentSnapshot | None:
        document = session.scalar(
            select(DocumentRow).where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.document_id == document_id,
            )
        )
        if document is None:
            return None
        versions = tuple(
            (row.version_id, row.version, row.lifecycle_status)
            for row in session.scalars(
                select(DocumentVersionRow)
                .where(
                    DocumentVersionRow.tenant_id == tenant_id,
                    DocumentVersionRow.document_id == document_id,
                )
                .order_by(DocumentVersionRow.version)
            )
        )
        return DocumentSnapshot(
            tenant_id=tenant_id,
            document_id=document.document_id,
            title=document.title,
            department=document.department,
            status=document.status,
            versions=versions,
        )

    def get_projection_binding(
        self,
        session: Session,
        *,
        tenant_id: str,
        document_id: str,
    ) -> tuple[int | None, str | None]:
        row = session.execute(
            select(
                DocumentRow.applied_projection_generation,
                DocumentRow.applied_projection_version_id,
            ).where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.document_id == document_id,
            )
        ).one_or_none()
        if row is None:
            raise DocumentRegistryConflict("projection_document_missing")
        return row.applied_projection_generation, row.applied_projection_version_id

    def prepare_document_deletion(
        self,
        session: Session,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentDeletionReservation | None:
        """撤回文档并取消尚未运行的任务，为跨存储物理删除建立稳定快照。"""
        document = session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.document_id == document_id,
            )
            .with_for_update()
        )
        if document is None:
            return None
        versions = tuple(
            session.scalars(
                select(DocumentVersionRow)
                .where(
                    DocumentVersionRow.tenant_id == tenant_id,
                    DocumentVersionRow.document_id == document_id,
                )
                .with_for_update()
            )
        )
        version_ids = tuple(version.version_id for version in versions)
        jobs = tuple(
            session.scalars(
                select(IngestionJobRow)
                .where(
                    IngestionJobRow.tenant_id == tenant_id,
                    IngestionJobRow.version_id.in_(version_ids),
                )
                .with_for_update()
            )
        ) if version_ids else ()
        if any(job.status == "running" for job in jobs):
            raise DocumentRegistryConflict("document_ingestion_running")
        now = datetime.now(UTC)
        reconcile_expires_at = document.projection_reconcile_expires_at
        if reconcile_expires_at is not None and reconcile_expires_at.tzinfo is None:
            reconcile_expires_at = reconcile_expires_at.replace(tzinfo=UTC)
        if (
            document.projection_reconcile_owner is not None
            and reconcile_expires_at is not None
            and reconcile_expires_at > now
        ):
            raise DocumentRegistryConflict("document_projection_reconcile_running")
        for version in versions:
            version.lifecycle_status = "withdrawn"
        for job in jobs:
            if job.status in {"awaiting_object", "pending"}:
                job.status = "cancelled"
                job.lease_owner = None
                job.lease_expires_at = None
                job.updated_at = now
        document.status = "withdrawn"
        document.updated_at = now
        session.flush()
        return DocumentDeletionReservation(
            tenant_id=tenant_id,
            document_id=document_id,
            source_object_keys=tuple(version.source_object_key for version in versions),
            version_count=len(versions),
            job_count=len(jobs),
        )

    def finalize_document_deletion(
        self,
        session: Session,
        *,
        tenant_id: str,
        document_id: str,
    ) -> bool:
        """删除已撤回文档的关系数据；调用方必须先清除外部向量和对象。"""
        document = session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.document_id == document_id,
            )
            .with_for_update()
        )
        if document is None:
            return False
        version_ids = tuple(
            session.scalars(
                select(DocumentVersionRow.version_id).where(
                    DocumentVersionRow.tenant_id == tenant_id,
                    DocumentVersionRow.document_id == document_id,
                )
            )
        )
        job_ids = tuple(
            session.scalars(
                select(IngestionJobRow.job_id).where(
                    IngestionJobRow.tenant_id == tenant_id,
                    IngestionJobRow.version_id.in_(version_ids),
                )
            )
        ) if version_ids else ()
        if version_ids or job_ids:
            session.execute(
                delete(IdempotencyRecordRow).where(
                    IdempotencyRecordRow.tenant_id == tenant_id,
                    (IdempotencyRecordRow.result_version_id.in_(version_ids))
                    | (IdempotencyRecordRow.result_job_id.in_(job_ids)),
                )
            )
        # 证据记录以分块为外键（RESTRICT）。删除源文档时，历史问答中
        # 指向该文档的证据也不再有效，必须先一并移除，才能删除分块。
        session.execute(
            delete(EvidenceRecordRow).where(
                EvidenceRecordRow.tenant_id == tenant_id,
                EvidenceRecordRow.document_id == document_id,
            )
        )
        session.execute(
            delete(ACLGrantRow).where(
                ACLGrantRow.tenant_id == tenant_id,
                ACLGrantRow.resource_type == "document",
                ACLGrantRow.resource_id == document_id,
            )
        )
        session.delete(document)
        session.flush()
        return True

    def get_job(self, session: Session, tenant_id: str, job_id: str) -> IngestionJobRow | None:
        return session.scalar(
            select(IngestionJobRow).where(
                IngestionJobRow.tenant_id == tenant_id,
                IngestionJobRow.job_id == job_id,
            )
        )

    def list_jobs_for_document(
        self,
        session: Session,
        *,
        tenant_id: str,
        document_id: str,
        limit: int = 100,
    ) -> tuple[IngestionJobRow, ...]:
        if limit <= 0 or limit > 1000:
            raise ValueError("invalid_ingestion_job_limit")
        return tuple(
            session.scalars(
                select(IngestionJobRow)
                .join(
                    DocumentVersionRow,
                    and_(
                        DocumentVersionRow.tenant_id == IngestionJobRow.tenant_id,
                        DocumentVersionRow.version_id == IngestionJobRow.version_id,
                    ),
                )
                .where(
                    IngestionJobRow.tenant_id == tenant_id,
                    DocumentVersionRow.document_id == document_id,
                )
                .order_by(IngestionJobRow.created_at.desc(), IngestionJobRow.job_id.desc())
                .limit(limit)
            )
        )

    def claim_next_job(
        self,
        session: Session,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        max_attempts: int = 3,
    ) -> IngestionJobRow | None:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        claimable_conditions = self._claimable_job_conditions(
            now=now, max_attempts=max_attempts
        )
        if session.get_bind().dialect.name == "sqlite":
            candidate = (
                select(IngestionJobRow.job_id)
                .where(*claimable_conditions)
                .order_by(IngestionJobRow.created_at, IngestionJobRow.job_id)
                .limit(1)
                .scalar_subquery()
            )
            job_id = session.execute(
                update(IngestionJobRow)
                .where(
                    IngestionJobRow.job_id == candidate,
                    *claimable_conditions,
                )
                .values(
                    status="running",
                    attempts=IngestionJobRow.attempts + 1,
                    lease_owner=worker_id,
                    lease_generation=IngestionJobRow.lease_generation + 1,
                    lease_expires_at=lease_expires_at,
                    updated_at=now,
                )
                .returning(IngestionJobRow.job_id)
            ).scalar_one_or_none()
            if job_id is None:
                return None
            job = session.get(IngestionJobRow, job_id)
            if job is None:
                raise DocumentRegistryConflict("ingestion_job_disappeared")
            return job
        statement: Select[tuple[IngestionJobRow]] = (
            select(IngestionJobRow)
            .where(*claimable_conditions)
            .order_by(IngestionJobRow.created_at, IngestionJobRow.job_id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = session.scalar(statement)
        if job is None:
            return None
        job.status = "running"
        job.attempts += 1
        job.lease_owner = worker_id
        job.lease_generation += 1
        job.lease_expires_at = lease_expires_at
        job.updated_at = now
        session.flush()
        return job

    @staticmethod
    def _claimable_job_conditions(*, now: datetime, max_attempts: int):
        return (
            IngestionJobRow.status == "pending",
            IngestionJobRow.available_at <= now,
            or_(
                IngestionJobRow.attempts < max_attempts,
                IngestionJobRow.error_code == "ingestion_projection_cleanup_failed",
            ),
        )

    def queue_snapshot(
        self,
        session: Session,
        *,
        max_attempts: int,
        now: datetime,
        limit: int,
    ) -> tuple[IngestionQueueSnapshot, ...]:
        if max_attempts <= 0 or limit <= 0:
            raise ValueError("invalid_ingestion_queue_snapshot_limits")
        snapshots: list[IngestionQueueSnapshot] = []
        for status in ("awaiting_object", "pending", "running"):
            conditions = (
                self._claimable_job_conditions(now=now, max_attempts=max_attempts)
                if status == "pending"
                else (IngestionJobRow.status == status,)
            )
            # ponytail: depth is capped to keep the periodic snapshot index-bounded;
            # add a saturation metric if an exact value above this limit becomes necessary.
            available = tuple(
                session.scalars(
                    select(IngestionJobRow.available_at)
                    .where(*conditions)
                    .order_by(IngestionJobRow.available_at, IngestionJobRow.job_id)
                    .limit(limit)
                )
            )
            oldest = available[0] if available else None
            if oldest is not None and oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=UTC)
            age_seconds = (
                max(0.0, (now - oldest).total_seconds()) if oldest is not None else 0.0
            )
            snapshots.append(
                IngestionQueueSnapshot(
                    status=status,
                    depth=len(available),
                    oldest_age_seconds=age_seconds,
                )
            )
        return tuple(snapshots)

    def recover_expired_jobs(
        self,
        session: Session,
        *,
        now: datetime,
        max_attempts: int,
        batch_size: int = 100,
    ) -> None:
        expired_jobs = tuple(
            session.scalars(
                select(IngestionJobRow)
                .where(
                    IngestionJobRow.status == "running",
                    IngestionJobRow.lease_expires_at.is_not(None),
                    IngestionJobRow.lease_expires_at <= now,
                )
                .order_by(IngestionJobRow.lease_expires_at, IngestionJobRow.job_id)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
        )
        for job in expired_jobs:
            if job.attempts >= max_attempts:
                if job.projection_generation is None:
                    self._mark_job_failed(
                        session,
                        job,
                        error_code="ingestion_retry_exhausted",
                        now=now,
                    )
                    continue
                job.status = "pending"
                job.error_code = "ingestion_projection_cleanup_failed"
                job.lease_owner = None
                job.lease_expires_at = None
                job.available_at = now
                job.updated_at = now
            else:
                job.status = "pending"
                job.lease_owner = None
                job.lease_expires_at = None
                job.available_at = now
                job.updated_at = now

    def lock_current_lease(
        self,
        session: Session,
        *,
        tenant_id: str,
        job_id: str,
        worker_id: str,
        lease_generation: int,
    ) -> IngestionJobRow:
        now = datetime.now(UTC)
        job = session.scalar(
            select(IngestionJobRow)
            .where(
                IngestionJobRow.tenant_id == tenant_id,
                IngestionJobRow.job_id == job_id,
                IngestionJobRow.status == "running",
                IngestionJobRow.lease_owner == worker_id,
                IngestionJobRow.lease_generation == lease_generation,
                IngestionJobRow.lease_expires_at.is_not(None),
                IngestionJobRow.lease_expires_at > now,
            )
            .with_for_update()
        )
        if job is None:
            raise DocumentRegistryConflict("ingestion_lease_stale")
        return job

    def renew_job_lease(
        self,
        session: Session,
        *,
        tenant_id: str,
        job_id: str,
        worker_id: str,
        lease_generation: int,
        lease_seconds: int,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("invalid_ingestion_lease_duration")
        job = self.lock_current_lease(
            session,
            tenant_id=tenant_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
        )
        now = datetime.now(UTC)
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.updated_at = now

    def replace_chunk_manifest(
        self,
        session: Session,
        *,
        job_id: str,
        worker_id: str,
        lease_generation: int,
        parsed: ParsedDocument,
    ) -> None:
        job = self.lock_current_lease(
            session,
            tenant_id=parsed.tenant_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
        )
        version = self.get_version(session, parsed.tenant_id, job.version_id)
        if (
            version is None
            or parsed.version_id != version.version_id
            or parsed.document_id != version.document_id
            or parsed.document_version != version.version
            or parsed.source_file_name != version.source_file_name
            or not parsed.chunks
            or compute_chunk_manifest_digest(parsed.chunks) != parsed.manifest_digest
        ):
            raise DocumentRegistryConflict("chunk_manifest_binding_invalid")
        session.execute(
            delete(ChunkMetadataRow).where(
                ChunkMetadataRow.tenant_id == parsed.tenant_id,
                ChunkMetadataRow.version_id == parsed.version_id,
            )
        )
        session.add_all(
            ChunkMetadataRow(
                chunk_id=chunk.chunk_id,
                version_id=version.version_id,
                tenant_id=version.tenant_id,
                document_id=version.document_id,
                document_version=version.version,
                content_hash=chunk.content_hash,
                source_locator=chunk.source_locator,
                source_file_name=version.source_file_name,
                index_version=version.index_version,
            )
            for chunk in parsed.chunks
        )
        session.flush()

    def allocate_projection_generation(
        self,
        session: Session,
        *,
        tenant_id: str,
        job_id: str,
        worker_id: str,
        lease_generation: int,
    ) -> int:
        """按当前摄取租约分配同文档单调递增的投影世代。

        同一租约内重复调用返回同一世代；租约代次变化时必须分配新世代，确保
        延迟到达的旧 Qdrant 请求只能写入旧 point ID。
        """
        job = self.lock_current_lease(
            session,
            tenant_id=tenant_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
        )
        if (
            job.projection_generation is not None
            and job.projection_lease_generation == lease_generation
        ):
            return job.projection_generation
        version = session.scalar(
            select(DocumentVersionRow)
            .where(
                DocumentVersionRow.tenant_id == tenant_id,
                DocumentVersionRow.version_id == job.version_id,
            )
            .with_for_update()
        )
        if version is None:
            raise DocumentRegistryConflict("projection_version_missing")
        if version.lifecycle_status in {"active", "superseded"}:
            raise DocumentRegistryConflict("projection_generation_not_allocatable")
        document = session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.document_id == version.document_id,
            )
            .with_for_update()
        )
        if document is None:
            raise DocumentRegistryConflict("projection_document_missing")
        generation = document.projection_generation + 1
        document.projection_generation = generation
        document.projection_reconcile_after = datetime.now(UTC)
        version.desired_projection_generation = generation
        job.projection_generation = generation
        job.projection_lease_generation = lease_generation
        session.flush()
        return generation

    def _verify_index_receipt(
        self,
        session: Session,
        *,
        job: IngestionJobRow,
        version: DocumentVersionRow,
        receipt: IndexReceipt,
    ) -> None:
        chunks = tuple(
            session.scalars(
                select(ChunkMetadataRow)
                .where(
                    ChunkMetadataRow.tenant_id == version.tenant_id,
                    ChunkMetadataRow.version_id == version.version_id,
                )
                .order_by(ChunkMetadataRow.chunk_id)
            )
        )
        stored_digest = compute_chunk_manifest_digest(chunks)
        if (
            receipt.tenant_id != version.tenant_id
            or receipt.job_id != job.job_id
            or receipt.version_id != version.version_id
            or receipt.source_content_hash != version.content_hash
            or receipt.index_version != version.index_version
            or receipt.projection_generation != job.projection_generation
            or receipt.projection_generation != version.desired_projection_generation
            or receipt.chunk_count != len(chunks)
            or receipt.manifest_digest != stored_digest
            or not chunks
        ):
            raise DocumentRegistryConflict("index_receipt_binding_invalid")

    def activate_version(
        self,
        session: Session,
        *,
        tenant_id: str,
        job_id: str,
        worker_id: str,
        lease_generation: int,
        receipt: IndexReceipt,
    ) -> RegisteredUpload:
        job = self.lock_current_lease(
            session,
            tenant_id=tenant_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
        )
        version = self.get_version(session, tenant_id, job.version_id)
        if version is None:
            raise DocumentRegistryConflict("document_version_not_activatable")
        self._verify_index_receipt(session, job=job, version=version, receipt=receipt)
        if version.lifecycle_status in {"active", "superseded"}:
            if (
                job.indexed_chunk_count == receipt.chunk_count
                and job.index_manifest_digest == receipt.manifest_digest
                and job.index_completed_at is not None
            ):
                return RegisteredUpload(
                    tenant_id=tenant_id,
                    document_id=version.document_id,
                    version_id=version.version_id,
                    version=version.version,
                    job_id=job.job_id,
                    source_object_key=version.source_object_key,
                    lifecycle_status=version.lifecycle_status,
                    ingestion_status=job.status,
                )
            raise DocumentRegistryConflict("index_receipt_binding_invalid")
        if version.lifecycle_status not in {"pending", "indexing"}:
            raise DocumentRegistryConflict("document_version_not_activatable")
        now = datetime.now(UTC)
        document = session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.tenant_id == tenant_id,
                DocumentRow.document_id == version.document_id,
            )
            .with_for_update()
        )
        if document is None:
            raise DocumentRegistryConflict("document_not_found")
        active_version = session.scalar(
            select(DocumentVersionRow)
            .where(
                DocumentVersionRow.tenant_id == tenant_id,
                DocumentVersionRow.document_id == version.document_id,
                DocumentVersionRow.lifecycle_status == "active",
            )
            .with_for_update()
        )
        active_generation = (
            active_version.desired_projection_generation
            if active_version is not None
            else None
        )
        if (
            active_version is not None
            and active_version.version <= version.version
            and active_generation is not None
            and receipt.projection_generation <= active_generation
        ):
            raise DocumentRegistryConflict("projection_generation_stale")
        if active_version is not None and active_version.version > version.version:
            # 旧版本迟到完成只能进入 superseded，不能把活动版本倒退。
            version.lifecycle_status = "superseded"
            version.superseded_at = now
        else:
            if active_version is not None:
                active_version.lifecycle_status = "superseded"
                active_version.superseded_at = now
                # 条件唯一索引要求同一时刻至多一条 active。先落库旧版本的降级，
                # 再提升新版本，避免 ORM 批量更新顺序在 SQLite/PostgreSQL 上触发瞬时冲突。
                session.flush()
            version.lifecycle_status = "active"
            version.activated_at = now
            document.status = "active"
            document.updated_at = now
        job.indexed_chunk_count = receipt.chunk_count
        job.index_manifest_digest = receipt.manifest_digest
        job.index_completed_at = receipt.verified_at
        job.updated_at = now
        return RegisteredUpload(
            tenant_id=tenant_id,
            document_id=version.document_id,
            version_id=version.version_id,
            version=version.version,
            job_id=job.job_id,
            source_object_key=version.source_object_key,
            lifecycle_status=version.lifecycle_status,
            ingestion_status=job.status,
        )

    def complete_projection(
        self,
        session: Session,
        *,
        tenant_id: str,
        job_id: str,
        worker_id: str,
        lease_generation: int,
        receipt: IndexReceipt,
        expected_lifecycle_status: str | None = None,
        expected_applied_generation: int | None = None,
        expected_applied_version_id: str | None = None,
    ) -> None:
        job = self.lock_current_lease(
            session,
            tenant_id=tenant_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
        )
        version = self.get_version(session, tenant_id, job.version_id)
        if (
            version is None
            or version.lifecycle_status not in {"active", "superseded"}
            or (
                expected_lifecycle_status is not None
                and version.lifecycle_status != expected_lifecycle_status
            )
            or receipt.tenant_id != tenant_id
            or receipt.job_id != job.job_id
            or receipt.version_id != job.version_id
            or receipt.projection_generation != job.projection_generation
            or receipt.projection_generation != version.desired_projection_generation
            or job.indexed_chunk_count != receipt.chunk_count
            or job.index_manifest_digest != receipt.manifest_digest
            or job.index_completed_at is None
        ):
            raise DocumentRegistryConflict("projection_completion_binding_invalid")
        now = datetime.now(UTC)
        if version.lifecycle_status == "active":
            document = session.scalar(
                select(DocumentRow)
                .where(
                    DocumentRow.tenant_id == tenant_id,
                    DocumentRow.document_id == version.document_id,
                )
                .with_for_update()
            )
            if (
                document is None
                or document.projection_generation < receipt.projection_generation
                or version.desired_projection_generation != receipt.projection_generation
                or document.applied_projection_generation != expected_applied_generation
                or document.applied_projection_version_id != expected_applied_version_id
            ):
                raise DocumentRegistryConflict("projection_generation_stale")
            version.applied_projection_generation = receipt.projection_generation
            document.applied_projection_generation = receipt.projection_generation
            document.applied_projection_version_id = version.version_id
            document.projection_verified_at = now
            document.projection_reconcile_error_code = None
            # 即使已收敛也保留周期审计；它是处理网络中延迟旧请求的最终保障。
            document.projection_reconcile_after = now + timedelta(minutes=5)
        job.status = "succeeded"
        job.lease_owner = None
        job.lease_expires_at = None
        job.updated_at = now

    def fail_job(
        self,
        session: Session,
        *,
        tenant_id: str,
        job_id: str,
        worker_id: str,
        lease_generation: int,
        error_code: str,
    ) -> None:
        job = self.lock_current_lease(
            session,
            tenant_id=tenant_id,
            job_id=job_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
        )
        self._mark_job_failed(
            session,
            job,
            error_code=error_code,
            now=datetime.now(UTC),
        )

    def _mark_job_failed(
        self,
        session: Session,
        job: IngestionJobRow,
        *,
        error_code: str,
        now: datetime,
    ) -> None:
        version = self.get_version(session, job.tenant_id, job.version_id)
        if version is not None:
            # 数据库版本已激活后，失败仅表示索引投影发布未完成；不得把业务
            # 真相从 active/superseded 倒退为 failed。
            if version.lifecycle_status not in {"active", "superseded"}:
                version.lifecycle_status = "failed"
            document = session.scalar(
                select(DocumentRow)
                .where(
                    DocumentRow.tenant_id == job.tenant_id,
                    DocumentRow.document_id == version.document_id,
                )
                .with_for_update()
            )
            has_active_version = session.scalar(
                select(DocumentVersionRow.version_id).where(
                    DocumentVersionRow.tenant_id == job.tenant_id,
                    DocumentVersionRow.document_id == version.document_id,
                    DocumentVersionRow.lifecycle_status == "active",
                )
            )
            if (
                document is not None
                and has_active_version is None
                and version.lifecycle_status == "failed"
            ):
                document.status = "failed"
                document.updated_at = now
        job.status = "failed"
        job.error_code = error_code
        job.lease_owner = None
        job.lease_expires_at = None
        job.updated_at = now
