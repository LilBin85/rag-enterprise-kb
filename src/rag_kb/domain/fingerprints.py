"""与任务运行方式无关的稳定摘要工具。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_fingerprint(kind: str, value: Any) -> str:
    """为 JSON 值生成带用途域隔离的稳定 SHA-256 摘要。"""
    if not kind:
        raise ValueError("fingerprint_kind_required")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(kind.encode('utf-8') + b'\0' + encoded).hexdigest()}"
