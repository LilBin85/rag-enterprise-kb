from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rag_kb import observability as observability_module

from rag_kb.observability import (
    JsonFormatter,
    MetadataOnlyLlmCallback,
    Observability,
    _SafeSpanExporter,
    bind_request_id,
    clear_request_id,
    current_request_id,
    extract_trace_context,
    normalize_request_id,
)


def _telemetry() -> tuple[Observability, InMemorySpanExporter]:
    observability = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    observability.start()
    exporter = InMemorySpanExporter()
    assert observability.provider is not None
    observability.provider.add_span_processor(SimpleSpanProcessor(exporter))
    return observability, exporter


def test_request_id_normalization_accepts_only_the_database_safe_contract() -> None:
    assert normalize_request_id("a" * 64) == "a" * 64
    assert normalize_request_id("a" * 65) != "a" * 65
    assert len(normalize_request_id("invalid request id")) == 32


def test_w3c_trace_context_accepts_valid_traceparent_without_baggage() -> None:
    context = extract_trace_context(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "vendor=value",
    )

    assert context is not None
    span_context = trace.get_current_span(context).get_span_context()
    assert f"{span_context.trace_id:032x}" == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert extract_trace_context("invalid", None) is None


def test_json_logging_uses_context_ids_and_excludes_unapproved_fields() -> None:
    formatter = JsonFormatter("rag-kb-test")
    observability = Observability(
        enabled=True,
        metrics_enabled=False,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    try:
        with observability.http_span(None):
            token = bind_request_id("request-1")
            try:
                record = logging.LogRecord(
                    "rag_kb.test",
                    logging.INFO,
                    __file__,
                    1,
                    "Bearer should-not-appear",
                    (),
                    None,
                )
                record.event = "observability.test"
                record.component = "test"
                record.secret = "should-not-appear"
                payload = json.loads(formatter.format(record))
            finally:
                clear_request_id(token)
    finally:
        observability.shutdown()

    assert payload["request_id"] == "request-1"
    assert payload["component"] == "test"
    assert len(payload["trace_id"]) == 32
    assert len(payload["span_id"]) == 16
    assert "Bearer" not in json.dumps(payload)
    assert "secret" not in payload
    assert current_request_id() is None


def test_safe_exporter_downgrades_export_failure() -> None:
    warnings: list[str] = []

    class FailingExporter(SpanExporter):
        def export(self, spans) -> SpanExportResult:
            raise RuntimeError("endpoint unavailable")

        def shutdown(self) -> None:
            return None

    exporter = _SafeSpanExporter(FailingExporter(), lambda: warnings.append("unavailable"))

    assert exporter.export(()) is SpanExportResult.FAILURE
    assert warnings == ["unavailable"]


def test_observability_can_start_a_local_trace_without_an_exporter() -> None:
    observability = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )

    with observability.http_span(None) as span:
        assert span.get_span_context().is_valid

    observability.shutdown()


def test_metric_label_contract_is_fixed_and_excludes_sensitive_or_high_cardinality_fields() -> None:
    observability = Observability(
        enabled=False,
        metrics_enabled=True,
        service_name="rag-kb-test",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    allowed = {
        "route",
        "method",
        "status_class",
        "outcome",
        "reason",
        "stage",
        "operation",
        "provider",
        "direction",
        "mode",
        "result",
        "error_category",
        "kind",
        "tool",
        "decision",
        "phase",
        "status",
        "worker_kind",
        "capability",
        "protocol",
    }
    label_names = {
        label
        for collector in observability.registry._collector_to_names
        for label in getattr(collector, "_labelnames", ())
    }

    assert label_names <= allowed
    assert not {
        "document_id",
        "task_id",
        "run_id",
        "user_id",
        "url",
        "question",
        "content",
        "hash",
    } & label_names
    payload = observability.metrics_payload().decode()
    assert 'rag_kb_security_boundary_violations_total{kind="write_unknown"} 0.0' in payload
    assert (
        'rag_kb_tool_calls_total{error_category="unknown",mode="write",outcome="unknown",tool="create_report_artifact"} 0.0'
        in payload
    )


def test_exporter_initialization_failure_does_not_block_local_traces(monkeypatch) -> None:
    def unavailable_exporter(**_kwargs):
        raise OSError("endpoint unavailable")

    monkeypatch.setattr(observability_module, "OTLPSpanExporter", unavailable_exporter)
    observability = Observability(
        enabled=True,
        metrics_enabled=False,
        service_name="rag-kb-test",
        otlp_endpoint="localhost:4317",
        trace_sample_ratio=1.0,
    )
    try:
        with observability.http_span(None) as span:
            assert span.get_span_context().is_valid
    finally:
        observability.shutdown()


def test_llm_callback_records_real_usage_and_never_exports_prompt_or_answer(caplog) -> None:
    observability, exporter = _telemetry()
    callback = MetadataOnlyLlmCallback(
        provider="deepseek", model_name="server-model", operation="knowledge.answer"
    )
    secret = "Bearer fake-token question=private answer=confidential"
    try:
        caplog.set_level(logging.INFO, logger="rag_kb.observability")
        with observability.activate():
            callback.on_chat_model_start({}, [[secret]], run_id="usage-present")
            callback.on_llm_end(
                SimpleNamespace(
                    generations=[
                        [SimpleNamespace(message=SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}))]
                    ]
                ),
                run_id="usage-present",
            )
            callback.on_chat_model_start({}, [[secret]], run_id="usage-missing")
            callback.on_llm_end(
                SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace())]]),
                run_id="usage-missing",
            )
        payload = observability.metrics_payload().decode()
        rendered_spans = repr(exporter.get_finished_spans())
    finally:
        observability.shutdown()

    spans = [span for span in exporter.get_finished_spans() if span.name == "llm.generate"]
    assert len(spans) == 2
    assert spans[0].attributes["input_tokens"] == 7
    assert spans[0].attributes["output_tokens"] == 3
    assert spans[0].attributes["total_tokens"] == 10
    assert 'direction="input"' in payload
    assert 'direction="output"' in payload
    assert 'rag_kb_llm_usage_missing_total{operation="knowledge.answer",provider="deepseek"} 1.0' in payload
    assert secret not in payload
    assert secret not in rendered_spans
    assert secret not in caplog.text
