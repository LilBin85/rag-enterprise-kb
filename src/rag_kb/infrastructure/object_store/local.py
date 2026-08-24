"""只允许服务端生成键的本地对象存储。"""

import os
from pathlib import Path, PurePosixPath
from uuid import uuid4


class ObjectConflictError(RuntimeError):
    pass


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, storage_key: str) -> Path:
        key = PurePosixPath(storage_key)
        if key.is_absolute() or not key.parts or any(part in {"", ".", ".."} for part in key.parts):
            raise ValueError("invalid_object_storage_key")
        target = self.root.joinpath(*key.parts).resolve()
        try:
            root_name = self._canonical_path(self.root)
            target_name = self._canonical_path(target)
            if os.path.commonpath((root_name, target_name)) != root_name:
                raise ValueError("outside_object_root")
        except ValueError as exc:
            raise ValueError("object_storage_path_escape") from exc
        return self._windows_long_path(target)

    @staticmethod
    def _canonical_path(path: Path) -> str:
        """统一 Windows 长路径前缀后再比较，避免安全检查误判。"""
        value = os.path.normcase(os.path.normpath(str(path)))
        if os.name == "nt" and value.startswith("\\\\?\\UNC\\"):
            return "\\\\" + value[8:]
        if os.name == "nt" and value.startswith("\\\\?\\"):
            return value[4:]
        return value

    @staticmethod
    def _windows_long_path(path: Path) -> Path:
        if os.name != "nt":
            return path
        value = str(path)
        if value.startswith("\\\\?\\") or len(value) < 240:
            return path
        if value.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + value[2:])
        return Path("\\\\?\\" + value)

    def put_exclusive(self, storage_key: str, content: bytes) -> None:
        target = self._resolve(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._windows_long_path(
            target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        )
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # 同目录硬链接只在完整临时文件落盘后发布，且绝不覆盖既有对象。
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ObjectConflictError("object_already_exists") from exc
            if os.name != "nt":
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, storage_key: str) -> bytes:
        return self._resolve(storage_key).read_bytes()

    def delete_exact(self, storage_key: str) -> None:
        self._resolve(storage_key).unlink(missing_ok=True)
