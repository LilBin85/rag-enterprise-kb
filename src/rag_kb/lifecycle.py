"""文档版本生命周期(蓝图 6.3):新增 / 更新 / 撤回 / 删除。

已知 MVP 简化:更新路径先写新版本再改写旧版本状态,两步之间失败需人工核对
(演示脚本不覆盖该失败路径;引入事务/任务队列属第 15 节触发器范围)。
"""

from pathlib import Path

from rag_kb.embedder import Embedder
from rag_kb.ingestion import chunk_items, parse_document
from rag_kb.models import ChunkStatus, DocumentSpec
from rag_kb.store import ChunkStore


class DocumentConflictError(RuntimeError):
    """同 document_id + version 已存在 active 块时拒绝重复新增。"""


class DocumentNotFoundError(RuntimeError):
    """对不存在的文档执行更新。"""


class DocumentLifecycle:
    def __init__(self, store: ChunkStore, embedder: Embedder, upload_dir: Path, max_chars: int = 800) -> None:
        self.store = store
        self.embedder = embedder
        self.upload_dir = upload_dir
        self.max_chars = max_chars

    def save_upload(self, document_id: str, filename: str, content: bytes) -> Path:
        """保存上传原件(可追溯),返回落盘路径。"""
        dest = self.upload_dir / document_id / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest

    def add_document(self, spec: DocumentSpec, file_path: Path) -> dict:
        """新增:写入首版 active 块。同版本已有 active 块时拒绝,防止误覆盖。"""
        existing = self.store.scroll_document(spec.document_id)
        if any(
            c["document_version"] == spec.version and c["status"] == str(ChunkStatus.ACTIVE)
            for c in existing
        ):
            raise DocumentConflictError(f"{spec.document_id} v{spec.version} 已存在 active 块")
        return self._ingest(spec, file_path)

    def update_document(self, document_id: str, spec: DocumentSpec, file_path: Path) -> dict:
        """更新:写入更高版本 active 块,原最高版本改 superseded。"""
        existing = self.store.scroll_document(document_id)
        if not existing:
            raise DocumentNotFoundError(document_id)
        current_max = max(c["document_version"] for c in existing)
        new_spec = spec.model_copy(update={"document_id": document_id, "version": current_max + 1})
        result = self._ingest(new_spec, file_path)
        self.store.set_status(document_id, ChunkStatus.SUPERSEDED, version=current_max)
        return {**result, "previous_version": current_max}

    def withdraw(self, document_id: str) -> None:
        """撤回:全部 active 块改 withdrawn,不再参与默认检索。"""
        self.store.set_status(document_id, ChunkStatus.WITHDRAWN, only_active=True)

    def delete(self, document_id: str) -> None:
        """删除:全部块(不论现状)改 deleted;块与引用均不再出现。"""
        self.store.set_status(document_id, ChunkStatus.DELETED)

    def versions(self, document_id: str) -> list[dict]:
        """聚合版本状态(调试与演示端点用)。"""
        by_version: dict[int, list[dict]] = {}
        for chunk in self.store.scroll_document(document_id):
            by_version.setdefault(chunk["document_version"], []).append(chunk)
        result = []
        for version, chunks in sorted(by_version.items()):
            statuses = {c["status"] for c in chunks}
            result.append({
                "version": version,
                "status": statuses.pop() if len(statuses) == 1 else "mixed",
                "chunk_count": len(chunks),
            })
        return result

    def _ingest(self, spec: DocumentSpec, file_path: Path) -> dict:
        items = parse_document(file_path)
        chunks = chunk_items(items, spec, max_chars=self.max_chars)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert_chunks(chunks, vectors)
        return {"document_id": spec.document_id, "version": spec.version, "chunk_count": len(chunks)}
