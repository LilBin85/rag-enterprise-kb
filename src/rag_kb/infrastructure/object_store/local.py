"""只允许服务端生成键的本地对象存储。"""

import os
from datetime import datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4


_RETENTION_PREFIXES = frozenset({"tool-results", "task-results"})


def _safe_object_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


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
        try:
            self._put_exclusive_once(target, content)
        except FileNotFoundError:
            # 目录可能被机会式空目录清理并发移除；重建目录后重试一次。
            target.parent.mkdir(parents=True, exist_ok=True)
            self._put_exclusive_once(target, content)

    def _put_exclusive_once(self, target: Path, content: bytes) -> None:
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
        target = self._resolve(storage_key)
        target.unlink(missing_ok=True)
        try:
            # ponytail: 机会式清父目录；多级空目录靠逐轮删除收敛。
            os.rmdir(target.parent)
        except OSError:
            pass

    def iter_refs(
        self,
        prefix: str,
        *,
        older_than: datetime,
        limit: int,
    ) -> tuple[str, ...]:
        """安全枚举服务端结果命名空间中早于指定时间的对象引用。"""
        entries, _exhausted, _resume = self.iter_ref_entries(
            prefix,
            older_than=older_than,
            limit=limit,
        )
        return tuple(ref for _mtime, ref in entries)

    def iter_ref_entries(
        self,
        prefix: str,
        *,
        older_than: datetime,
        limit: int,
        after: str | None = None,
        exclude: frozenset[str] = frozenset(),
        max_scanned: int = 10_000,
    ) -> tuple[tuple[tuple[float, str], ...], bool, str | None]:
        """按规范序位置游标枚举 retention 对象，不接受外部路径。

        返回 (entries, exhausted, resume)：resume 是本轮最后访问的文件 ref，
        截断时非空，作为下一轮 `after` 位置游标；自然扫尽时为 None。
        预算只约束 stat 成本；被排除/未到 cutoff 的条目照样消耗预算但不阻碍推进。
        """
        namespace = prefix.rstrip("/")
        if namespace not in _RETENTION_PREFIXES:
            raise ValueError("invalid_object_scan_prefix")
        if limit < 1:
            raise ValueError("invalid_object_scan_limit")
        if max_scanned < 1:
            raise ValueError("invalid_object_scan_budget")
        if older_than.tzinfo is None:
            raise ValueError("object_scan_time_must_be_aware")
        if after is not None and (
            not isinstance(after, str)
            or not after.startswith(f"{namespace}/")
        ):
            raise ValueError("invalid_object_scan_cursor")

        namespace_path = self.root / namespace
        try:
            # ponytail: 目录名单每轮重排序 O(D log D)；D 到 10⁵+ 且不收缩时升级持久化游标表。
            # 键带尾部 "/"，使目录间排序与完整 ref 字符串序严格一致，位置游标不回跳。
            task_names = sorted(
                (
                    entry.name
                    for entry in os.scandir(namespace_path)
                    if entry.is_dir(follow_symlinks=False)
                    and _safe_object_component(entry.name)
                ),
                key=lambda name: f"{name}/",
            )
        except (FileNotFoundError, NotADirectoryError):
            return (), True, None

        cutoff = older_than.timestamp()
        entries: list[tuple[float, str]] = []
        scanned = 0
        exhausted = True
        resume: str | None = None
        for task_name in task_names:
            task_path = namespace_path / task_name
            try:
                object_names = sorted(
                    entry.name
                    for entry in os.scandir(task_path)
                    if _safe_object_component(entry.name)
                    and entry.is_file(follow_symlinks=False)
                )
            except (FileNotFoundError, NotADirectoryError):
                continue
            # 扫描期机会式清理过期空目录（mtime 早于 cutoff 且无有效条目）。
            if not object_names:
                try:
                    if task_path.stat().st_mtime < cutoff:
                        self._windows_long_path(task_path).rmdir()
                except OSError:
                    pass
                continue
            for object_name in object_names:
                ref = PurePosixPath(namespace, task_name, object_name).as_posix()
                if after is not None and ref <= after:
                    continue
                scanned += 1
                if scanned > max_scanned:
                    # 预算溢出点尚未检验，不能作游标（否则下轮被 after 永久
                    # 跳过、夹缝对象不可达）；回退到最后一个已检验位置。
                    exhausted = False
                    break
                try:
                    mtime = self._windows_long_path(
                        task_path / object_name
                    ).stat().st_mtime
                except FileNotFoundError:
                    continue
                resume = ref
                if mtime < cutoff and ref not in exclude:
                    if len(entries) == limit:
                        # 规范序收满 limit 即截断；游标停到最后一个已返回的
                        # 候选上——溢出候选下轮重检，绝不落到游标之下形成
                        # 永久跳过夹缝。
                        exhausted = False
                        resume = entries[-1][1]
                        break
                    entries.append((mtime, ref))
            if not exhausted:
                break
        if exhausted:
            return tuple(sorted(entries)), True, None
        return tuple(sorted(entries)), False, resume
