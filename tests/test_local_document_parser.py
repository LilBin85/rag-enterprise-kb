from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_kb.application.ingestion_worker import ClaimedIngestionJob
from rag_kb.domain.indexing import ParsedChunk, compute_chunk_manifest_digest
from rag_kb.ingestion import ParsedItem
from rag_kb.infrastructure.ingestion.local_document_parser import (
    DocumentParseError,
    LocalDocumentParser,
)
from rag_kb.infrastructure.object_store import LocalObjectStore


def _hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _job(*, source_file_name: str = "source.md", content_hash: str, document_id: str = "document-a") -> ClaimedIngestionJob:
    return ClaimedIngestionJob(
        tenant_id="tenant-a",
        job_id="job-a",
        version_id="version-a",
        document_id=document_id,
        version=1,
        source_object_key="source-file",
        source_file_name=source_file_name,
        content_hash=content_hash,
        index_version="kb-v1",
        attempts=1,
        lease_owner="worker-a",
        lease_generation=1,
    )


def _parser(tmp_path: Path, raw: bytes, **kwargs) -> LocalDocumentParser:
    store = LocalObjectStore(tmp_path / "objects")
    store.put_exclusive("source-file", raw)
    return LocalDocumentParser(store, **kwargs)


def test_parser_rejects_mismatched_source_hash_without_exposing_content(tmp_path: Path) -> None:
    raw = b"## Section\nsecret body"
    parser = _parser(tmp_path, raw)

    with pytest.raises(DocumentParseError, match="source_content_hash_mismatch") as error:
        parser.parse(_job(content_hash="sha256:" + "0" * 64))

    assert "secret body" not in str(error.value)


def test_markdown_requires_strict_utf8(tmp_path: Path) -> None:
    raw = b"## heading\n\xff"
    parser = _parser(tmp_path, raw)

    with pytest.raises(DocumentParseError, match="markdown_invalid_utf8"):
        parser.parse(_job(content_hash=_hash(raw)))


def test_parser_rejects_documents_without_content_chunks(tmp_path: Path) -> None:
    raw = "# title only\n\n".encode("utf-8")
    parser = _parser(tmp_path, raw)

    with pytest.raises(DocumentParseError, match="document_parse_empty"):
        parser.parse(_job(content_hash=_hash(raw)))


def test_manifest_and_chunk_ids_are_deterministic_and_document_independent(tmp_path: Path) -> None:
    raw = "# title\n## First\nalpha paragraph\n## Second\nbeta paragraph".encode("utf-8")
    parser = _parser(tmp_path, raw, max_chars=20)

    first = parser.parse(_job(content_hash=_hash(raw), document_id="document-a"))
    second = parser.parse(_job(content_hash=_hash(raw), document_id="document-b"))

    assert first.chunks == second.chunks
    assert first.manifest_digest == second.manifest_digest
    assert all(chunk.chunk_id.startswith("version-a:chunk-") for chunk in first.chunks)
    assert all("document-" not in chunk.chunk_id for chunk in first.chunks)
    assert compute_chunk_manifest_digest(tuple(reversed(first.chunks))) == first.manifest_digest


def test_pdf_parser_uses_controlled_temp_file_and_cleans_it(tmp_path: Path) -> None:
    raw = b"%PDF-test"
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    observed_paths: list[Path] = []

    def fake_parse_document(path: Path) -> list[ParsedItem]:
        observed_paths.append(path)
        assert path.read_bytes() == raw
        return [ParsedItem(kind="heading", text="Section", page=1), ParsedItem(kind="text", text="body", page=1)]

    parser = _parser(
        tmp_path,
        raw,
        document_parser=fake_parse_document,
        temporary_root=temporary_root,
    )
    parsed = parser.parse(_job(source_file_name="source.pdf", content_hash=_hash(raw)))

    assert parsed.chunks[0].source_locator == "第 1 页 / Section"
    assert observed_paths and not observed_paths[0].exists()
    assert list(temporary_root.iterdir()) == []


def test_manifest_uses_chunk_metadata_not_input_order() -> None:
    first = ParsedChunk("version-a:chunk-000001", 1, "alpha", _hash(b"alpha"), "First")
    second = ParsedChunk("version-a:chunk-000002", 2, "beta", _hash(b"beta"), "Second")

    assert compute_chunk_manifest_digest((first, second)) == compute_chunk_manifest_digest((second, first))


def test_manifest_accepts_database_metadata_without_ordinal() -> None:
    @dataclass(frozen=True)
    class DatabaseChunkMetadata:
        chunk_id: str
        content_hash: str
        source_locator: str

    first = DatabaseChunkMetadata("version-a:chunk-000002", _hash(b"beta"), "Second")
    second = DatabaseChunkMetadata("version-a:chunk-000001", _hash(b"alpha"), "First")

    assert compute_chunk_manifest_digest((first, second)) == compute_chunk_manifest_digest((second, first))
