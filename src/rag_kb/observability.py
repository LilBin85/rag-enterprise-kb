"""比较工作流的最小结构化观测,只使用标准库并采用字段白名单。"""

import hashlib
import json
import logging
import time
import uuid

logger = logging.getLogger("rag_kb.policy_comparison")


def hash_identifier(value: str) -> str:
    """只输出不可逆标识摘要,避免日志携带原始文档 ID。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class RunObserver:
    """记录运行级 stage/tool 事件;不接受 prompt、正文或完整模型输出。"""

    def __init__(self, role: str, document_id: str) -> None:
        self.run_id = str(uuid.uuid4())
        self.role = role
        self.document_hash = hash_identifier(document_id)

    def emit(
        self,
        operation: str,
        stage: str,
        *,
        status: str,
        step: int = 0,
        tool: str | None = None,
        counts: dict[str, int] | None = None,
        started_at: float | None = None,
        error_code: str | None = None,
    ) -> None:
        event = {
            "run_id": self.run_id,
            "operation": operation,
            "stage": stage,
            "role": self.role,
            "document_hash": self.document_hash,
            "step": step,
            "tool": tool,
            "counts": counts or {},
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3)
            if started_at is not None
            else 0.0,
            "status": status,
            "error_code": error_code,
        }
        logger.info(json.dumps(event, ensure_ascii=False, sort_keys=True))
