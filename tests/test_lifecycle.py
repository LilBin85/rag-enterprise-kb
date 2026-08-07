"""文档版本生命周期集成测试(蓝图 6.3 / 9 节阶段 2 优先测试)。

连接真实 Qdrant(独立 collection),使用确定性假 embedder;
覆盖:新增可查询、更新后旧版本 superseded、撤回/删除后块不再有效、重复新增拒绝。
"""

from pathlib import Path

import pytest

from rag_kb.lifecycle import (
    DocumentConflictError,
    DocumentLifecycle,
    DocumentNotFoundError,
)
from rag_kb.models import ChunkStatus, DocumentSpec, Role
from rag_kb.store import ChunkStore

DOC_V1 = """# 测试休假制度
## 年假
员工每年可享受带薪年假 10 天。
## 病假
员工每年可享受带薪病假 15 天。
"""

DOC_V2 = """# 测试休假制度
## 年假
员工每年可享受带薪年假 15 天。
## 病假
员工每年可享受带薪病假 15 天。
"""


def _spec(version: int = 1, filename: str = "测试休假制度-v1.md") -> DocumentSpec:
    return DocumentSpec(
        document_id="test-leave-policy",
        file=filename,
        title="测试休假制度",
        department="hr",
        allowed_roles=[Role.HR, Role.ADMIN],
        version=version,
    )


@pytest.fixture()
def lifecycle(store: ChunkStore, embedder, tmp_path: Path) -> DocumentLifecycle:
    return DocumentLifecycle(store=store, embedder=embedder, upload_dir=tmp_path / "uploads", max_chars=800)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_add_document_creates_active_chunks(lifecycle, store, tmp_path) -> None:
    result = lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1.md", DOC_V1))
    assert result["version"] == 1
    assert result["chunk_count"] == 2  # 年假、病假两章

    chunks = store.scroll_document("test-leave-policy")
    assert len(chunks) == 2
    assert all(c["status"] == ChunkStatus.ACTIVE for c in chunks)
    assert all(c["document_version"] == 1 for c in chunks)
    assert all(c["allowed_roles"] == ["hr", "admin"] for c in chunks)


def test_add_existing_active_version_rejected(lifecycle, tmp_path) -> None:
    lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1.md", DOC_V1))
    with pytest.raises(DocumentConflictError):
        lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1-again.md", DOC_V1))


def test_update_supersedes_previous_version(lifecycle, store, tmp_path) -> None:
    lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1.md", DOC_V1))
    result = lifecycle.update_document(
        "test-leave-policy", _spec(version=0, filename="测试休假制度-v2.md"),
        _write(tmp_path, "测试休假制度-v2.md", DOC_V2),
    )
    assert result["version"] == 2
    assert result["previous_version"] == 1

    chunks = store.scroll_document("test-leave-policy")
    by_version = {}
    for c in chunks:
        by_version.setdefault(c["document_version"], set()).add(c["status"])
    assert by_version[1] == {ChunkStatus.SUPERSEDED}
    assert by_version[2] == {ChunkStatus.ACTIVE}


def test_update_nonexistent_document_rejected(lifecycle, tmp_path) -> None:
    with pytest.raises(DocumentNotFoundError):
        lifecycle.update_document(
            "ghost", _spec(), _write(tmp_path, "ghost.md", DOC_V1)
        )


def test_withdraw_marks_all_chunks_withdrawn(lifecycle, store, tmp_path) -> None:
    lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1.md", DOC_V1))
    lifecycle.withdraw("test-leave-policy")
    chunks = store.scroll_document("test-leave-policy")
    assert chunks and all(c["status"] == ChunkStatus.WITHDRAWN for c in chunks)


def test_delete_marks_all_chunks_deleted(lifecycle, store, tmp_path) -> None:
    lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1.md", DOC_V1))
    lifecycle.update_document(
        "test-leave-policy", _spec(version=0, filename="测试休假制度-v2.md"),
        _write(tmp_path, "测试休假制度-v2.md", DOC_V2),
    )
    lifecycle.delete("test-leave-policy")
    chunks = store.scroll_document("test-leave-policy")
    assert chunks and all(c["status"] == ChunkStatus.DELETED for c in chunks)


def test_versions_aggregation(lifecycle, tmp_path) -> None:
    lifecycle.add_document(_spec(), _write(tmp_path, "测试休假制度-v1.md", DOC_V1))
    lifecycle.update_document(
        "test-leave-policy", _spec(version=0, filename="测试休假制度-v2.md"),
        _write(tmp_path, "测试休假制度-v2.md", DOC_V2),
    )
    versions = lifecycle.versions("test-leave-policy")
    assert [(v["version"], v["status"]) for v in versions] == [(1, "superseded"), (2, "active")]
    assert all(v["chunk_count"] == 2 for v in versions)


def test_versions_of_unknown_document_is_empty(lifecycle) -> None:
    assert lifecycle.versions("ghost") == []
