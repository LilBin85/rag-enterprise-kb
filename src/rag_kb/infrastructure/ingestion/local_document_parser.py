"""从本地对象存储解析文档并生成确定性 chunk manifest。"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path

from rag_kb.application.ports.indexing import ClaimedIngestionJob, DocumentParserPort
from rag_kb.application.ports.object_store import ObjectStorePort
from rag_kb.domain.indexing import (
    ParsedChunk,
    ParsedDocument,
    compute_chunk_manifest_digest,
)
from rag_kb.ingestion import ParsedItem, _split_long_paragraph, parse_document, parse_markdown


class DocumentParseError(RuntimeError):
    """仅公开固定安全错误码，禁止将原文或底层异常返回给调用方。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _content_hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _chunk_items(items: list[ParsedItem], *, version_id: str, max_chars: int) -> tuple[ParsedChunk, ...]:
    sections: list[tuple[str, int | None, list[str]]] = []
    heading = "正文"
    page: int | None = None
    paragraphs: list[str] = []

    def flush() -> None:
        nonlocal paragraphs
        if paragraphs:
            sections.append((heading, page, paragraphs))
        paragraphs = []

    for item in items:
        text = item.text.strip()
        if not text or item.kind == "title":
            continue
        if item.kind == "heading":
            flush()
            heading = text
            page = item.page
        elif item.kind == "text":
            paragraphs.append(text)
    flush()

    chunks: list[ParsedChunk] = []
    for section_heading, section_page, section_paragraphs in sections:
        locator = (
            f"第 {section_page} 页 / {section_heading}"
            if section_page is not None
            else section_heading
        )
        buffer = ""
        for paragraph in section_paragraphs:
            for piece in _split_long_paragraph(paragraph, max_chars):
                if buffer and len(buffer) + len(piece) + 1 > max_chars:
                    ordinal = len(chunks) + 1
                    chunks.append(
                        ParsedChunk(
                            chunk_id=f"{version_id}:chunk-{ordinal:06d}",
                            ordinal=ordinal,
                            text=buffer,
                            content_hash=_content_hash(buffer.encode("utf-8")),
                            source_locator=locator,
                        )
                    )
                    buffer = piece
                else:
                    buffer = f"{buffer}\n{piece}" if buffer else piece
        if buffer:
            ordinal = len(chunks) + 1
            chunks.append(
                ParsedChunk(
                    chunk_id=f"{version_id}:chunk-{ordinal:06d}",
                    ordinal=ordinal,
                    text=buffer,
                    content_hash=_content_hash(buffer.encode("utf-8")),
                    source_locator=locator,
                )
            )
    return tuple(chunks)


class LocalDocumentParser(DocumentParserPort):
    """对象存储读取、原始哈希验证、格式解析与确定性切块的适配器。"""

    def __init__(
        self,
        object_store: ObjectStorePort,
        *,
        max_chars: int = 800,
        document_parser: Callable[[Path], list[ParsedItem]] = parse_document,
        temporary_root: Path | None = None,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("invalid_chunk_max_chars")
        self._object_store = object_store
        self._max_chars = max_chars
        self._document_parser = document_parser
        self._temporary_root = temporary_root

    def parse(self, job: ClaimedIngestionJob) -> ParsedDocument:
        return self.parse_version(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            version_id=job.version_id,
            document_version=job.version,
            source_object_key=job.source_object_key,
            source_file_name=job.source_file_name,
            content_hash=job.content_hash,
        )

    def parse_version(
        self,
        *,
        tenant_id: str,
        document_id: str,
        version_id: str,
        document_version: int,
        source_object_key: str,
        source_file_name: str,
        content_hash: str,
    ) -> ParsedDocument:
        """重放已保存版本的原文，并用摄取时的相同规则重建 chunk manifest。"""
        try:
            raw = self._object_store.read(source_object_key)
        except (OSError, ValueError) as exc:
            raise DocumentParseError("source_object_unavailable") from exc
        if _content_hash(raw) != content_hash:
            raise DocumentParseError("source_content_hash_mismatch")

        suffix = Path(source_file_name).suffix.lower()
        if suffix == ".md":
            try:
                items = parse_markdown(raw.decode("utf-8", errors="strict"))
            except UnicodeDecodeError as exc:
                raise DocumentParseError("markdown_invalid_utf8") from exc
        elif suffix == ".pdf":
            items = self._parse_pdf(raw)
        else:
            raise DocumentParseError("unsupported_source_format")

        chunks = _chunk_items(items, version_id=version_id, max_chars=self._max_chars)
        if not chunks:
            raise DocumentParseError("document_parse_empty")
        return ParsedDocument(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            document_version=document_version,
            source_file_name=source_file_name,
            chunks=chunks,
            manifest_digest=compute_chunk_manifest_digest(chunks),
        )

    def _parse_pdf(self, raw: bytes) -> list[ParsedItem]:
        try:
            with tempfile.TemporaryDirectory(dir=self._temporary_root) as temporary_directory:
                temporary_path = Path(temporary_directory) / "source.pdf"
                temporary_path.write_bytes(raw)
                return self._document_parser(temporary_path)
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("pdf_parse_failed") from exc
