"""四个只读工具适配器；全部经租户、任务和文档 ACL 重新校验。"""

from __future__ import annotations

from sqlalchemy import select

from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.domain.identity import Permission
from rag_kb.domain.task_runtime import canonical_fingerprint
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import ChunkMetadataRow, DocumentRow, DocumentVersionRow
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork


class DatabaseReadToolAdapter:
    def __init__(
        self,
        database: Database,
        *,
        operation: str,
        retriever: EnterpriseRetriever | None = None,
        acl: ACLRepository | None = None,
        evidence: EvidenceLedgerRepository | None = None,
    ) -> None:
        if operation not in {
            "search_knowledge",
            "fetch_evidence",
            "list_document_versions",
            "compare_document_versions",
        }:
            raise ValueError("unsupported_read_tool")
        if operation == "search_knowledge" and retriever is None:
            raise ValueError("search_retriever_required")
        self._database = database
        self._operation = operation
        self._retriever = retriever
        self._acl = acl or ACLRepository()
        self._evidence = evidence or EvidenceLedgerRepository(self._acl)

    def execute(self, arguments: dict[str, object], context: ToolExecutionContext) -> dict[str, object]:
        if self._operation == "search_knowledge":
            return self._search(arguments, context)
        if self._operation == "fetch_evidence":
            return self._fetch(arguments, context)
        if self._operation == "list_document_versions":
            return self._versions(arguments, context)
        return self._compare(arguments, context)

    def verify(self, result: dict[str, object], context: ToolExecutionContext) -> bool:
        reference = result.get("result_reference")
        evidence_ids = result.get("evidence_ids")
        items = result.get("items")
        if not isinstance(reference, str) or not isinstance(evidence_ids, list) or not isinstance(items, list):
            return False
        if reference != self._reference(evidence_ids, items):
            return False
        if evidence_ids:
            try:
                with self._database.sessions() as session:
                    citations = self._evidence.citations_from_ledger(
                        session,
                        task_id=context.task_id,
                        evidence_ids=tuple(str(value) for value in evidence_ids),
                        auth=context.auth,
                    )
                return len(citations) == len(evidence_ids)
            except (PermissionError, RuntimeError):
                return False
        return True

    def _search(self, arguments, context):
        assert self._retriever is not None
        retrieval = self._retriever.search(str(arguments["query"]), context.auth)
        limit = int(arguments.get("limit", len(retrieval.candidates)))
        candidates = retrieval.candidates[:limit]
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            records = self._evidence.record_candidates(
                work.session,
                task_id=context.task_id,
                query_fingerprint=retrieval.query_fingerprint,
                auth=context.auth,
                candidates=candidates,
            )
        items = [candidate.model_dump(mode="json") for candidate in candidates]
        return self._pack([record.evidence_id for record in records], items)

    def _fetch(self, arguments, context):
        evidence_ids = [str(value) for value in arguments["evidence_ids"]]
        with self._database.sessions() as session:
            citations = self._evidence.citations_from_ledger(
                session,
                task_id=context.task_id,
                evidence_ids=tuple(evidence_ids),
                auth=context.auth,
            )
        return self._pack(evidence_ids, [item.model_dump(mode="json") for item in citations])

    def _versions(self, arguments, context):
        document_id = str(arguments["document_id"])
        with self._database.sessions() as session:
            self._require_document(session, context, document_id)
            rows = tuple(
                session.scalars(
                    select(DocumentVersionRow)
                    .where(
                        DocumentVersionRow.tenant_id == context.auth.tenant_id,
                        DocumentVersionRow.document_id == document_id,
                    )
                    .order_by(DocumentVersionRow.version)
                )
            )
        items = [
            {
                "version_id": row.version_id,
                "version": row.version,
                "lifecycle_status": row.lifecycle_status,
                "content_hash": row.content_hash,
                "source_file_name": row.source_file_name,
            }
            for row in rows
        ]
        return self._pack([], items)

    def _compare(self, arguments, context):
        document_id = str(arguments["document_id"])
        versions = (int(arguments["left_version"]), int(arguments["right_version"]))
        items: list[dict[str, object]] = []
        with self._database.sessions() as session:
            self._require_document(session, context, document_id)
            for version_number in versions:
                version = session.scalar(
                    select(DocumentVersionRow).where(
                        DocumentVersionRow.tenant_id == context.auth.tenant_id,
                        DocumentVersionRow.document_id == document_id,
                        DocumentVersionRow.version == version_number,
                    )
                )
                if version is None:
                    raise RuntimeError("document_version_not_found")
                chunks = tuple(
                    session.scalars(
                        select(ChunkMetadataRow)
                        .where(
                            ChunkMetadataRow.tenant_id == context.auth.tenant_id,
                            ChunkMetadataRow.version_id == version.version_id,
                        )
                        .order_by(ChunkMetadataRow.chunk_id)
                    )
                )
                items.append(
                    {
                        "version": version.version,
                        "version_id": version.version_id,
                        "content_hash": version.content_hash,
                        "chunks": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "content_hash": chunk.content_hash,
                                "source_locator": chunk.source_locator,
                                "source_file_name": chunk.source_file_name,
                            }
                            for chunk in chunks
                        ],
                    }
                )
        return self._pack([], items)

    def _require_document(self, session, context, document_id: str) -> None:
        document = session.scalar(
            select(DocumentRow).where(
                DocumentRow.tenant_id == context.auth.tenant_id,
                DocumentRow.document_id == document_id,
            )
        )
        if document is None or not self._acl.document_is_allowed(
            session,
            auth=context.auth,
            document_id=document_id,
            permission=Permission.DOCUMENT_READ,
        ):
            raise PermissionError("document_read_denied")

    @staticmethod
    def _reference(evidence_ids: list[object], items: list[object]) -> str:
        return canonical_fingerprint("tool_read_result", {"evidence_ids": evidence_ids, "items": items})

    @classmethod
    def _pack(cls, evidence_ids: list[object], items: list[object]) -> dict[str, object]:
        return {
            "result_reference": cls._reference(evidence_ids, items),
            "evidence_ids": evidence_ids,
            "items": items,
        }
