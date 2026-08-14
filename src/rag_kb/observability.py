"""安全观测：长期日志使用白名单，一次性诊断需显式写入本地受限文件。"""

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger("rag_kb.policy_comparison")

NORMAL_FINISH_REASONS = frozenset({"stop", "tool_calls", "end_turn"})


def _response_metadata(raw_response: Any) -> dict[str, Any]:
    metadata = (
        raw_response.get("response_metadata")
        if isinstance(raw_response, dict)
        else getattr(raw_response, "response_metadata", None)
    )
    return metadata if isinstance(metadata, dict) else {}


def _raw_content(raw_response: Any) -> Any:
    if isinstance(raw_response, dict):
        return raw_response.get("content")
    return getattr(raw_response, "content", None)


def finish_reason(raw_response: Any) -> str | None:
    """从常见 Provider 元数据形状读取 finish reason，不记录其它元数据。"""
    metadata = _response_metadata(raw_response)
    value = metadata.get("finish_reason", metadata.get("stop_reason"))
    return value if isinstance(value, str) and value else None


def classify_model_failure(error: BaseException | None, *, raw_response: Any = None) -> str:
    """把模型/适配器失败归为稳定错误码，避免把敏感异常正文写入长期日志。"""
    error_name = type(error).__name__.lower() if error is not None else ""
    message = str(error).lower() if error is not None else ""
    reason = finish_reason(raw_response)

    if "jsondecodeerror" in error_name or (
        "json" in message
        and any(token in message for token in ("unterminated", "truncat", "unexpected end", "eof"))
    ):
        return "json_truncated"
    if "validationerror" in error_name or any(
        token in message for token in ("schema validation", "validation error", "schema mismatch")
    ):
        return "schema_validation_failed"
    if "timeout" in error_name or "timed out" in message or "timeout" in message:
        return "provider_timeout"
    if "connection" in error_name or any(
        token in message for token in ("connection error", "connection failed", "connection refused")
    ):
        return "provider_request_failed"
    if reason is not None and reason not in NORMAL_FINISH_REASONS:
        return "abnormal_finish_reason"
    if any(token in message for token in ("empty response", "empty content", "no generations")):
        return "provider_empty_response"
    if raw_response is not None and _raw_content(raw_response) in (None, "", [], {}):
        return "provider_empty_response"
    if "adapter" in error_name or "protocol" in error_name or any(
        token in message for token in ("adapter protocol", "unexpected response shape")
    ):
        return "adapter_protocol_error"
    if error is not None:
        return "model_output_parse_failed"
    return "provider_request_failed"


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    """限制诊断事件大小；未知对象只保留类型，不调用可能泄密的 repr。"""
    if depth >= 8:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 32_000 else value[:32_000] + "<truncated>"
    if isinstance(value, dict):
        return {
            str(key)[:256]: _bounded_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_json_value(item, depth=depth + 1) for item in value[:100]]
    return {"object_type": type(value).__name__}


class LocalDiagnosticTrace:
    """显式、短期、可删除的本地 JSONL 诊断文件。

    与 RunObserver 的长期安全日志隔离。调用方必须提供文件名；文件只能位于
    root 内、不得覆盖已有文件，并以 0600 创建。内容可能敏感，header 会记录
    到期时间；删除需由调用方显式调用 delete()。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        root: str | Path,
        ttl_seconds: int = 3600,
    ) -> None:
        if not 60 <= ttl_seconds <= 86_400:
            raise ValueError("diagnostic_trace_ttl_out_of_range")
        root_path = Path(root).resolve()
        requested = Path(path)
        target = (requested if requested.is_absolute() else root_path / requested).resolve()
        try:
            target.relative_to(root_path)
        except ValueError as exc:
            raise ValueError("diagnostic_trace_must_be_inside_root") from exc
        if target.suffix.lower() != ".jsonl":
            raise ValueError("diagnostic_trace_must_use_jsonl")
        if not target.parent.is_dir():
            raise ValueError("diagnostic_trace_parent_missing")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(target, flags, 0o600)
        try:
            os.chmod(target, 0o600)
            self._stream: TextIO = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        except Exception:
            os.close(descriptor)
            raise
        self.path = target
        self._closed = False
        created_at = datetime.now(timezone.utc)
        self._write(
            {
                "event_type": "diagnostic_trace_header",
                "created_at": created_at.isoformat(),
                "expires_at": (created_at + timedelta(seconds=ttl_seconds)).isoformat(),
                "contains_sensitive_diagnostic_data": True,
                "retention": "delete_explicitly_after_diagnosis",
            }
        )

    def write_event(self, event: dict[str, Any]) -> None:
        if self._closed:
            raise ValueError("diagnostic_trace_closed")
        self._write({**_bounded_json_value(event), "event_type": "planner_case"})

    def _write(self, event: dict[str, Any]) -> None:
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 256_000:
            raise ValueError("diagnostic_trace_event_too_large")
        self._stream.write(encoded + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def delete(self) -> None:
        """只删除本实例创建的确切文件；调用本方法即为显式删除动作。"""
        self.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "LocalDiagnosticTrace":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


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
