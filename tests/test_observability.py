import json
import os
from types import SimpleNamespace

import pytest

from rag_kb.observability import LocalDiagnosticTrace, classify_model_failure


class AdapterProtocolError(Exception):
    pass


class SchemaValidationError(Exception):
    pass


class SecretRepr:
    def __repr__(self) -> str:
        return "SECRET_FROM_REPR"


@pytest.mark.parametrize(
    ("error", "raw", "expected"),
    [
        (TimeoutError("request timed out"), None, "provider_timeout"),
        (type("APIConnectionError", (Exception,), {})("Connection error."), None, "provider_request_failed"),
        (json.JSONDecodeError("unterminated JSON", "{", 1), None, "json_truncated"),
        (SchemaValidationError("bad fields"), None, "schema_validation_failed"),
        (AdapterProtocolError("bad protocol"), None, "adapter_protocol_error"),
        (
            ValueError("incomplete"),
            SimpleNamespace(content="{", response_metadata={"finish_reason": "length"}),
            "abnormal_finish_reason",
        ),
        (
            ValueError("missing"),
            SimpleNamespace(content="", response_metadata={"finish_reason": "stop"}),
            "provider_empty_response",
        ),
    ],
)
def test_model_failure_classification_is_stable(error, raw, expected) -> None:
    assert classify_model_failure(error, raw_response=raw) == expected


def test_local_diagnostic_trace_is_explicit_bounded_and_deletable(tmp_path) -> None:
    trace = LocalDiagnosticTrace("planner.jsonl", root=tmp_path, ttl_seconds=60)
    trace.write_event(
        {
            "case": "one",
            "prompt": "SECRET_PROMPT",
            "long_field": "x" * 33_000,
            "unknown": SecretRepr(),
            "event_type": "cannot_override",
        }
    )
    path = trace.path
    trace.close()

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event_type"] == "diagnostic_trace_header"
    assert lines[0]["contains_sensitive_diagnostic_data"] is True
    assert lines[1]["prompt"] == "SECRET_PROMPT"
    assert lines[1]["event_type"] == "planner_case"
    assert lines[1]["long_field"].endswith("<truncated>")
    assert lines[1]["unknown"] == {"object_type": "SecretRepr"}
    assert "SECRET_FROM_REPR" not in path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0

    trace.delete()
    assert not path.exists()


def test_local_diagnostic_trace_rejects_oversized_event(tmp_path) -> None:
    trace = LocalDiagnosticTrace("large.jsonl", root=tmp_path, ttl_seconds=60)
    with pytest.raises(ValueError, match="event_too_large"):
        trace.write_event({"many_large_fields": ["x" * 32_000 for _ in range(9)]})
    trace.close()
    assert len((tmp_path / "large.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_local_diagnostic_trace_refuses_escape_and_overwrite(tmp_path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    with pytest.raises(ValueError, match="inside_root"):
        LocalDiagnosticTrace(outside, root=tmp_path)

    existing = tmp_path / "existing.jsonl"
    existing.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        LocalDiagnosticTrace(existing, root=tmp_path)
    assert existing.read_text(encoding="utf-8") == "user data"
