"""与存储实现无关的解析块及索引回执契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ParsedChunk:
    """解析阶段的最小块，不包含 ACL、文档生命周期或向量存储字段。"""

    chunk_id: str
    ordinal: int
    text: str
    content_hash: str
    source_locator: str

    def __post_init__(self) -> None:
        if self.ordinal <= 0:
            raise ValueError("invalid_chunk_ordinal")
        if not self.text:
            raise ValueError("empty_chunk_text")
        if not self.content_hash.startswith("sha256:"):
            raise ValueError("invalid_chunk_content_hash")


class ChunkManifestEntry(Protocol):
    """解析块与数据库元数据行共享的 manifest 输入契约。"""

    chunk_id: str
    content_hash: str
    source_locator: str


def compute_chunk_manifest_digest(chunks: Sequence[ChunkManifestEntry]) -> str:
    """对稳定排序后的块元数据哈希生成 manifest 摘要。

    ``chunk_id`` 已编码 ordinal，因此摘要只使用数据库也持久化的三项元数据；
    解析结果与稍后回读的 ``ChunkMetadataRow`` 必须产生完全相同的摘要。
    """
    metadata_hashes = []
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        metadata = {
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "source_locator": chunk.source_locator,
        }
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        metadata_hashes.append(_sha256(encoded))
    manifest = json.dumps(metadata_hashes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256(manifest)


@dataclass(frozen=True)
class ParsedDocument:
    tenant_id: str
    document_id: str
    version_id: str
    document_version: int
    source_file_name: str
    chunks: tuple[ParsedChunk, ...]
    manifest_digest: str

    def __post_init__(self) -> None:
        if self.document_version <= 0:
            raise ValueError("invalid_document_version")
        if not self.chunks:
            raise ValueError("empty_parsed_document")
        if self.manifest_digest != compute_chunk_manifest_digest(self.chunks):
            raise ValueError("invalid_chunk_manifest_digest")


@dataclass(frozen=True)
class IndexReceipt:
    """索引端确认已写入的确定性回执。"""

    tenant_id: str
    job_id: str
    version_id: str
    source_content_hash: str
    index_version: str
    projection_generation: int
    chunk_count: int
    manifest_digest: str
    verified_at: datetime

    def __post_init__(self) -> None:
        if self.projection_generation <= 0:
            raise ValueError("invalid_projection_generation")
        if self.chunk_count <= 0:
            raise ValueError("invalid_indexed_chunk_count")
        if not self.source_content_hash.startswith("sha256:"):
            raise ValueError("invalid_source_content_hash")
        if not self.manifest_digest.startswith("sha256:"):
            raise ValueError("invalid_manifest_digest")
        if self.verified_at.tzinfo is None or self.verified_at.utcoffset() is None:
            raise ValueError("index_receipt_time_not_aware")
