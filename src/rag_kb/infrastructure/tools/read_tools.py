"""四个只读工具适配器；全部经租户、任务和文档 ACL 重新校验。"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from sqlalchemy import select

from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.domain.identity import Permission
from rag_kb.domain.fingerprints import canonical_fingerprint
from rag_kb.domain.indexing import ParsedChunk, ParsedDocument
from rag_kb.domain.retrieval import RetrievalCandidate
from rag_kb.infrastructure.database.acl_repository import ACLRepository
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.models import (
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser


_MAX_COMPARISON_CHUNKS = 20


class ReadToolError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _ComparisonSource:
    tenant_id: str
    document_id: str
    version_id: str
    version: int
    source_object_key: str
    source_file_name: str
    content_hash: str
    projection_generation: int
    chunk_count: int
    manifest_digest: str
    metadata: tuple[ChunkMetadataRow, ...]


class DatabaseReadToolAdapter:
    def __init__(
        self,
        database: Database,
        *,
        operation: str,
        retriever: EnterpriseRetriever | None = None,
        acl: ACLRepository | None = None,
        evidence: EvidenceLedgerRepository | None = None,
        source_parser: LocalDocumentParser | None = None,
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
        if operation == "compare_document_versions" and source_parser is None:
            raise ValueError("compare_parser_required")
        self._database = database
        self._operation = operation
        self._retriever = retriever
        self._acl = acl or ACLRepository()
        self._evidence = evidence or EvidenceLedgerRepository(self._acl)
        self._source_parser = source_parser

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
        left_version = int(arguments["left_version"])
        right_version = int(arguments["right_version"])
        if left_version >= right_version:
            raise ReadToolError("document_version_order_invalid")
        assert self._source_parser is not None
        with self._database.sessions() as session:
            self._require_document(
                session, context, document_id, permission=Permission.KNOWLEDGE_READ
            )
            left = self._comparison_source(session, context, document_id, left_version)
            right = self._comparison_source(session, context, document_id, right_version)
        left_document = self._replay_source(left)
        right_document = self._replay_source(right)
        changes, selected = self._comparison_changes(left_document, right_document)
        selected_sources = tuple(
            (left if side == "left" else right, chunk) for side, chunk in selected
        )
        if len(selected_sources) > _MAX_COMPARISON_CHUNKS:
            raise ReadToolError("comparison_scope_too_large")
        candidates = tuple(
            self._comparison_candidate(source, chunk) for source, chunk in selected_sources
        )
        query_fingerprint = canonical_fingerprint(
            "version_comparison",
            {
                "document_id": document_id,
                "left_version_id": left.version_id,
                "right_version_id": right.version_id,
            },
        )
        with UnitOfWork(self._database) as work:
            assert work.session is not None
            records = self._evidence.record_candidates(
                work.session,
                task_id=context.task_id,
                query_fingerprint=query_fingerprint,
                auth=context.auth,
                candidates=candidates,
                evidence_scope="version_comparison",
            )
        evidence_by_chunk = {
            (candidate.version_id, candidate.chunk_id): record.evidence_id
            for candidate, record in zip(candidates, records, strict=True)
        }
        return self._pack(
            [record.evidence_id for record in records],
            [
                {
                    "document_id": document_id,
                    "left_version": left.version,
                    "right_version": right.version,
                    "changes": [
                        {
                            "kind": kind,
                            "left": [
                                self._comparison_chunk(left, chunk, evidence_by_chunk)
                                for chunk in left_chunks
                            ],
                            "right": [
                                self._comparison_chunk(right, chunk, evidence_by_chunk)
                                for chunk in right_chunks
                            ],
                        }
                        for kind, left_chunks, right_chunks in changes
                    ],
                }
            ],
        )

    def _comparison_source(self, session, context, document_id: str, version_number: int) -> _ComparisonSource:
        version = session.scalar(
            select(DocumentVersionRow).where(
                DocumentVersionRow.tenant_id == context.auth.tenant_id,
                DocumentVersionRow.document_id == document_id,
                DocumentVersionRow.version == version_number,
            )
        )
        if version is None:
            raise ReadToolError("document_version_not_found")
        if version.lifecycle_status not in {"active", "superseded"}:
            raise ReadToolError("document_version_not_comparable")
        job = session.scalar(
            select(IngestionJobRow).where(
                IngestionJobRow.tenant_id == context.auth.tenant_id,
                IngestionJobRow.version_id == version.version_id,
            )
        )
        if (
            job is None
            or job.status != "succeeded"
            or version.applied_projection_generation is None
            or job.projection_generation != version.applied_projection_generation
            or job.indexed_chunk_count is None
            or job.index_manifest_digest is None
            or job.index_completed_at is None
        ):
            raise ReadToolError("comparison_version_not_indexed")
        metadata = tuple(
            session.scalars(
                select(ChunkMetadataRow)
                .where(
                    ChunkMetadataRow.tenant_id == context.auth.tenant_id,
                    ChunkMetadataRow.version_id == version.version_id,
                )
                .order_by(ChunkMetadataRow.chunk_id)
            )
        )
        if len(metadata) != job.indexed_chunk_count:
            raise ReadToolError("comparison_manifest_invalid")
        return _ComparisonSource(
            tenant_id=context.auth.tenant_id,
            document_id=document_id,
            version_id=version.version_id,
            version=version.version,
            source_object_key=version.source_object_key,
            source_file_name=version.source_file_name,
            content_hash=version.content_hash,
            projection_generation=version.applied_projection_generation,
            chunk_count=job.indexed_chunk_count,
            manifest_digest=job.index_manifest_digest,
            metadata=metadata,
        )

    def _replay_source(self, source: _ComparisonSource) -> ParsedDocument:
        document = self._source_parser.parse_version(
            tenant_id=source.tenant_id,
            document_id=source.document_id,
            version_id=source.version_id,
            document_version=source.version,
            source_object_key=source.source_object_key,
            source_file_name=source.source_file_name,
            content_hash=source.content_hash,
        )
        actual = tuple(
            (chunk.chunk_id, chunk.content_hash, chunk.source_locator, source.source_file_name)
            for chunk in document.chunks
        )
        expected = tuple(
            (chunk.chunk_id, chunk.content_hash, chunk.source_locator, chunk.source_file_name)
            for chunk in source.metadata
        )
        if (
            len(document.chunks) != source.chunk_count
            or document.manifest_digest != source.manifest_digest
            or actual != expected
        ):
            raise ReadToolError("comparison_manifest_invalid")
        return document

    @staticmethod
    def _comparison_changes(
        left: ParsedDocument,
        right: ParsedDocument,
    ) -> tuple[
        list[tuple[str, tuple[ParsedChunk, ...], tuple[ParsedChunk, ...]]],
        list[tuple[str, ParsedChunk]],
    ]:
        changes = [
            (kind, left.chunks[left_start:left_end], right.chunks[right_start:right_end])
            for kind, left_start, left_end, right_start, right_end in SequenceMatcher(
                a=[chunk.content_hash for chunk in left.chunks],
                b=[chunk.content_hash for chunk in right.chunks],
                autojunk=False,
            ).get_opcodes()
            if kind != "equal"
        ]
        if not changes:
            changes = [("unchanged", left.chunks[:1], right.chunks[:1])]
        selected: list[tuple[str, ParsedChunk]] = []
        seen: set[tuple[str, str]] = set()
        for kind, left_chunks, right_chunks in changes:
            for side, chunks in (("left", left_chunks), ("right", right_chunks)):
                for chunk in chunks:
                    key = (side, chunk.chunk_id)
                    if key not in seen:
                        seen.add(key)
                        selected.append((side, chunk))
        return changes, selected

    @staticmethod
    def _comparison_candidate(source: _ComparisonSource, chunk: ParsedChunk) -> RetrievalCandidate:
        return RetrievalCandidate(
            tenant_id=source.tenant_id,
            document_id=source.document_id,
            version_id=source.version_id,
            document_version=source.version,
            projection_generation=source.projection_generation,
            chunk_id=chunk.chunk_id,
            content_hash=chunk.content_hash,
            source_locator=chunk.source_locator,
            source_file_name=source.source_file_name,
            fused_score=1.0,
            text=chunk.text,
        )

    @staticmethod
    def _comparison_chunk(
        source: _ComparisonSource,
        chunk: ParsedChunk,
        evidence_by_chunk: dict[tuple[str, str], str],
    ) -> dict[str, object]:
        return {
            "version": source.version,
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "source_locator": chunk.source_locator,
            "source_file_name": source.source_file_name,
            "text": chunk.text,
            "evidence_id": evidence_by_chunk[(source.version_id, chunk.chunk_id)],
        }

    def _require_document(
        self,
        session,
        context,
        document_id: str,
        *,
        permission: Permission = Permission.DOCUMENT_READ,
    ) -> None:
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
            permission=permission,
        ):
            # 带稳定错误码抛出，避免被压成 tool_execution_failed_permissionerror。
            raise ReadToolError("document_read_denied")

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
