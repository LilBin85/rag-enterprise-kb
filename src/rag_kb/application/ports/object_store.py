"""应用层所需的最小对象存储契约。"""

from typing import Protocol


class ObjectConflictError(RuntimeError):
    pass


class ObjectStorePort(Protocol):
    def put_exclusive(self, storage_key: str, content: bytes) -> None: ...

    def read(self, storage_key: str) -> bytes: ...

    def delete_exact(self, storage_key: str) -> None: ...
