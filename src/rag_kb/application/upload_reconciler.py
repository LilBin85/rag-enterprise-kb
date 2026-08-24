"""修复“对象已发布、数据库尚未推进”的摄取预留。"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
from time import monotonic

from rag_kb.infrastructure.database.document_repository import DocumentRegistryRepository
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.object_store.local import LocalObjectStore


@dataclass(frozen=True)
class ReconciliationResult:
    scanned: int
    recovered: int
    missing_objects: int
    content_mismatches: int
    failed: int = 0


class UploadReservationReconciler:
    """只推进内容哈希匹配的对象；缺失或冲突对象保留给告警/同键重放。"""

    def __init__(
        self,
        database: Database,
        object_store: LocalObjectStore,
        repository: DocumentRegistryRepository | None = None,
    ) -> None:
        self._database = database
        self._object_store = object_store
        self._repository = repository or DocumentRegistryRepository()

    def reconcile(
        self,
        *,
        older_than: datetime,
        limit: int = 100,
        time_budget_seconds: float | None = None,
    ) -> ReconciliationResult:
        if time_budget_seconds is not None and time_budget_seconds <= 0:
            raise ValueError("invalid_reconciliation_time_budget")
        deadline = (
            monotonic() + time_budget_seconds
            if time_budget_seconds is not None
            else None
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            reservations = self._repository.list_pending_object_reservations(
                work.session,
                older_than=older_than,
                limit=limit,
            )

        recovered = 0
        missing = 0
        mismatched = 0
        failed = 0
        scanned = 0
        for reservation in reservations:
            if deadline is not None and monotonic() >= deadline:
                break
            scanned += 1
            try:
                content = self._object_store.read(reservation.source_object_key)
            except FileNotFoundError:
                missing += 1
                failed += int(not self._defer_safely(reservation))
                continue
            except Exception:
                failed += 1
                self._defer_safely(reservation)
                continue
            try:
                actual_hash = "sha256:" + hashlib.sha256(content).hexdigest()
                if not hmac.compare_digest(actual_hash, reservation.content_hash):
                    mismatched += 1
                    failed += int(not self._defer_safely(reservation))
                    continue
                with UnitOfWork(self._database) as work:
                    assert work.session is not None
                    self._repository.mark_object_ready(
                        work.session,
                        tenant_id=reservation.tenant_id,
                        user_id=reservation.user_id,
                        idempotency_key=reservation.idempotency_key,
                        request_hash=reservation.request_hash,
                    )
                recovered += 1
            except Exception:
                # 单条损坏绑定不能阻塞同批次中的其他上传预留。
                failed += 1
                self._defer_safely(reservation)

        return ReconciliationResult(
            scanned=scanned,
            recovered=recovered,
            missing_objects=missing,
            content_mismatches=mismatched,
            failed=failed,
        )

    def _defer_safely(self, reservation) -> bool:
        try:
            with UnitOfWork(self._database) as work:
                assert work.session is not None
                self._repository.defer_pending_object_reservation(
                    work.session,
                    tenant_id=reservation.tenant_id,
                    user_id=reservation.user_id,
                    idempotency_key=reservation.idempotency_key,
                    request_hash=reservation.request_hash,
                )
        except Exception:
            return False
        return True
