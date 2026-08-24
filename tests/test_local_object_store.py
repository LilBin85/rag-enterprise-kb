"""本地对象存储的路径边界与独占写回归。"""

from pathlib import Path

import pytest

from rag_kb.infrastructure.object_store.local import LocalObjectStore, ObjectConflictError


def test_put_exclusive_writes_once_and_rejects_duplicate(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    store.put_exclusive("tenant/doc/version/policy.md", b"first")
    assert store.read("tenant/doc/version/policy.md") == b"first"
    with pytest.raises(ObjectConflictError):
        store.put_exclusive("tenant/doc/version/policy.md", b"second")
    assert store.read("tenant/doc/version/policy.md") == b"first"


@pytest.mark.parametrize("key", ["../escape", "tenant/../../escape", "/absolute", ""])
def test_object_store_rejects_path_escape_keys(tmp_path: Path, key: str) -> None:
    store = LocalObjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.put_exclusive(key, b"must not write")
    assert not (tmp_path.parent / "escape").exists()
