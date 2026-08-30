"""可复现的可观测性 A/B 性能门；不访问外部模型或业务数据。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from rag_kb.application.document_service import DocumentService
from rag_kb.application.query_service import KnowledgeQueryService
from rag_kb.config import Settings
from rag_kb.domain.identity import AuthContext, Principal
from rag_kb.domain.retrieval import RetrievalCandidate, RetrievalResult
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.observability import (
    Observability,
    current_observability,
    finish_observed_span,
    observe_span,
)
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import build_document_service, build_identity_provider


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("invalid_percentile_input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def overhead_percent(off_seconds: float, on_seconds: float) -> float:
    if off_seconds <= 0:
        raise ValueError("baseline_must_be_positive")
    return (on_seconds / off_seconds - 1) * 100


def bootstrap_p95_overhead(
    rounds: list[list[tuple[float, float]]], *, repetitions: int = 1000
) -> tuple[float, float]:
    if not rounds or any(not pairs for pairs in rounds) or repetitions <= 0:
        raise ValueError("invalid_bootstrap_input")
    generator = random.Random(20260830)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample: list[tuple[float, float]] = []
        for _ in rounds:
            pairs = rounds[generator.randrange(len(rounds))]
            for stratum in (pairs[::2], pairs[1::2]):
                sample.extend(
                    stratum[generator.randrange(len(stratum))] for _ in stratum
                )
        estimates.append(
            overhead_percent(
                percentile([pair[0] for pair in sample], 0.95),
                percentile([pair[1] for pair in sample], 0.95),
            )
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def _measure(call: Callable[[], None]) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


def paired_samples(
    off: Callable[[], None],
    on: Callable[[], None],
    *,
    samples: int,
    warmup: int,
) -> list[tuple[float, float]]:
    if samples <= 0 or warmup < 0:
        raise ValueError("invalid_sample_configuration")
    for index in range(warmup):
        if index % 2 == 0:
            off()
            on()
        else:
            on()
            off()
    pairs: list[tuple[float, float]] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(samples):
            if index % 2 == 0:
                off_duration = _measure(off)
                on_duration = _measure(on)
            else:
                on_duration = _measure(on)
                off_duration = _measure(off)
            pairs.append((off_duration, on_duration))
    finally:
        if gc_was_enabled:
            gc.enable()
    return pairs


def _latencies(pairs: list[tuple[float, float]]) -> dict[str, object]:
    off = [pair[0] * 1000 for pair in pairs]
    on = [pair[1] * 1000 for pair in pairs]
    return {
        "samples_per_arm": len(pairs),
        "off_ms": {
            "p50": percentile(off, 0.50),
            "p95": percentile(off, 0.95),
            "p99": percentile(off, 0.99),
        },
        "on_ms": {
            "p50": percentile(on, 0.50),
            "p95": percentile(on, 0.95),
            "p99": percentile(on, 0.99),
        },
        "p95_overhead_percent": overhead_percent(
            percentile(off, 0.95), percentile(on, 0.95)
        ),
    }


def _summary(
    rounds: list[list[tuple[float, float]]], limit_percent: float
) -> dict[str, object]:
    pooled = [pair for pairs in rounds for pair in pairs]
    latencies = _latencies(pooled)
    p95_overhead = float(latencies["p95_overhead_percent"])
    interval = bootstrap_p95_overhead(rounds)
    return {
        "rounds": len(rounds),
        "samples_per_arm_per_round": len(rounds[0]),
        "samples_per_arm": len(pooled),
        "concurrency": 1,
        "per_round": [
            {"round": index, **_latencies(pairs)}
            for index, pairs in enumerate(rounds, start=1)
        ],
        "off_ms": latencies["off_ms"],
        "on_ms": latencies["on_ms"],
        "p95_overhead_percent": p95_overhead,
        "p95_overhead_95ci_percent": [interval[0], interval[1]],
        "p95_overhead_95ci_half_width_percent": (interval[1] - interval[0]) / 2,
        "limit_percent": limit_percent,
        "point_estimate_passed": p95_overhead < limit_percent,
        "decision_basis": "hierarchical_p95_overhead_95ci_upper",
        "passed": interval[1] < limit_percent,
    }


def _document_service(
    root: Path,
) -> tuple[DocumentService, object, DemoTokenIdentityProvider]:
    database = build_database(f"sqlite+pysqlite:///{(root / 'benchmark.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="Synthetic"))
        session.flush()
        session.add(
            PrincipalRow(tenant_id="tenant-a", user_id="user-a", roles=[], attributes={})
        )
        session.flush()
        session.add(
            DocumentRow(
                tenant_id="tenant-a",
                document_id="document-a",
                title="Synthetic",
                department="test",
                status="active",
                created_by="user-a",
                projection_generation=1,
                applied_projection_generation=1,
                applied_projection_version_id="version-1",
            )
        )
        session.flush()
        session.add(
            DocumentVersionRow(
                version_id="version-1",
                tenant_id="tenant-a",
                document_id="document-a",
                version=1,
                lifecycle_status="active",
                source_object_key="synthetic",
                source_file_name="synthetic.md",
                content_hash="sha256:source",
                index_version="synthetic",
                desired_projection_generation=1,
                applied_projection_generation=1,
            )
        )
        session.add(
            ACLGrantRow(
                grant_id="grant-1",
                tenant_id="tenant-a",
                resource_type="document",
                resource_id="document-a",
                subject_type="user",
                subject_id="user-a",
                permissions=["document:read"],
            )
        )
    provider = DemoTokenIdentityProvider(
        {
            "benchmark-token": Principal(
                tenant_id="tenant-a",
                user_id="user-a",
                attributes={"permissions": ["document:read"]},
            )
        }
    )
    return DocumentService(database, LocalObjectStore(root / "objects")), database, provider


def _document_api_round(
    root: Path, samples: int, warmup: int
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[float]
]:
    service, database, provider = _document_service(root)
    off_app = create_app(
        Settings(
            _env_file=None,
            observability_enabled=False,
            metrics_enabled=False,
            otel_exporter_otlp_endpoint="",
        )
    )
    on_app = create_app(
        Settings(
            _env_file=None,
            observability_enabled=True,
            metrics_enabled=True,
            otel_exporter_otlp_endpoint="",
        )
    )
    for app in (off_app, on_app):
        app.dependency_overrides[build_document_service] = lambda: service
        app.dependency_overrides[build_identity_provider] = lambda: provider
    headers = {
        "X-Demo-Role-Token": "benchmark-token",
        "X-Request-ID": "benchmark-request",
    }
    try:
        with TestClient(off_app) as off_client, TestClient(on_app) as on_client:
            def document(client: TestClient) -> dict[str, object]:
                response = client.get("/v1/documents/document-a", headers=headers)
                assert response.status_code == 200
                return response.json()

            # paired_samples 每次调用只应测一个 arm；响应一致性在计时外单独验证。
            assert document(off_client) == document(on_client)
            document_pairs = paired_samples(
                lambda: document(off_client),
                lambda: document(on_client),
                samples=samples,
                warmup=warmup,
            )
            health_pairs = paired_samples(
                lambda: off_client.get("/health/live").raise_for_status(),
                lambda: on_client.get("/health/live").raise_for_status(),
                samples=samples,
                warmup=warmup,
            )
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                metric_durations = [
                    _measure(lambda: on_client.get("/metrics").raise_for_status()) * 1000
                    for _ in range(samples)
                ]
            finally:
                if gc_was_enabled:
                    gc.enable()
            metrics = on_client.get("/metrics")
            metrics.raise_for_status()
            labels = (
                'route="/v1/documents/{document_id}"',
                'method="GET"',
                'status_class="2xx"',
            )
            assert any(
                line.startswith("rag_kb_http_requests_total{")
                and all(label in line for label in labels)
                for line in metrics.text.splitlines()
            )
    finally:
        off_app.dependency_overrides.clear()
        on_app.dependency_overrides.clear()
        database.engine.dispose()
    return document_pairs, health_pairs, metric_durations


def _health_summary(rounds: list[list[tuple[float, float]]]) -> dict[str, object]:
    pooled = [pair for pairs in rounds for pair in pairs]
    latencies = _latencies(pooled)
    off_ms = latencies["off_ms"]
    on_ms = latencies["on_ms"]
    assert isinstance(off_ms, dict) and isinstance(on_ms, dict)
    return {
        "rounds": len(rounds),
        "samples_per_arm_per_round": len(rounds[0]),
        "off_ms": off_ms,
        "on_ms": on_ms,
        "p95_absolute_delta_ms": float(on_ms["p95"]) - float(off_ms["p95"]),
        "limit_ms": 5.0,
        "decision_basis": "enabled_p95_absolute",
        "passed": float(on_ms["p95"]) < 5.0,
    }


def _api_benchmark(
    rounds: int, samples: int, warmup: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    document_rounds: list[list[tuple[float, float]]] = []
    health_rounds: list[list[tuple[float, float]]] = []
    metric_durations: list[float] = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory(prefix="rag-kb-m6-api-benchmark-") as directory:
            document_pairs, health_pairs, durations = _document_api_round(
                Path(directory), samples, warmup
            )
        document_rounds.append(document_pairs)
        health_rounds.append(health_pairs)
        metric_durations.extend(durations)
    metrics_p95 = percentile(metric_durations, 0.95)
    return (
        _summary(document_rounds, 5.0),
        _health_summary(health_rounds),
        {
            "rounds": rounds,
            "samples_per_round": samples,
            "samples": len(metric_durations),
            "concurrency": 1,
            "p95_ms": metrics_p95,
            "limit_ms": 100.0,
            "passed": metrics_p95 < 100.0,
        },
    )


@dataclass
class _Verdict:
    verdict: str = "answer"
    answer: str = "synthetic answer"
    refusal_kind: str | None = None
    used_chunk_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.used_chunk_ids is None:
            self.used_chunk_ids = ["chunk-1"]


class _Retriever:
    def __init__(self, candidate: RetrievalCandidate) -> None:
        self._result = RetrievalResult(
            query_fingerprint="synthetic-query",
            candidates=(candidate,),
            context_token_estimate=8,
            sufficient=True,
        )

    def search(self, _question: str, _auth: AuthContext) -> RetrievalResult:
        return self._result


class _Model:
    def answer(
        self, _question: str, _chunks: list[RetrievalCandidate]
    ) -> _Verdict:
        started = time.perf_counter()
        with observe_span(
            "llm.generate",
            attributes={"operation": "knowledge.answer", "provider": "deterministic"},
        ) as span:
            time.sleep(0.02)
            duration = time.perf_counter() - started
            finish_observed_span(span, outcome="succeeded")
            telemetry = current_observability()
            if telemetry is not None:
                telemetry.record_llm(
                    operation="knowledge.answer",
                    provider="deterministic",
                    outcome="succeeded",
                    duration_seconds=duration,
                    input_tokens=8,
                    output_tokens=4,
                )
        return _Verdict()


def _query_service(root: Path, name: str) -> tuple[KnowledgeQueryService, object]:
    database = build_database(f"sqlite+pysqlite:///{(root / name).as_posix()}")
    Base.metadata.create_all(database.engine)
    text = "synthetic authorized content"
    content_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="Synthetic"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="user-a", roles=[], attributes={}))
        session.flush()
        session.add(
            DocumentRow(
                tenant_id="tenant-a",
                document_id="document-a",
                title="Synthetic",
                department="test",
                status="active",
                created_by="user-a",
                projection_generation=1,
                applied_projection_generation=1,
                applied_projection_version_id="version-1",
            )
        )
        session.flush()
        session.add(
            DocumentVersionRow(
                version_id="version-1",
                tenant_id="tenant-a",
                document_id="document-a",
                version=1,
                lifecycle_status="active",
                source_object_key="synthetic",
                source_file_name="synthetic.md",
                content_hash="sha256:source",
                index_version="synthetic",
                desired_projection_generation=1,
                applied_projection_generation=1,
            )
        )
        session.flush()
        session.add(
            ChunkMetadataRow(
                chunk_id="chunk-1",
                version_id="version-1",
                tenant_id="tenant-a",
                document_id="document-a",
                document_version=1,
                content_hash=content_hash,
                source_locator="section/1",
                source_file_name="synthetic.md",
                index_version="synthetic",
            )
        )
        session.add(
            IngestionJobRow(
                job_id="job-1",
                tenant_id="tenant-a",
                version_id="version-1",
                status="succeeded",
                indexed_chunk_count=1,
                index_manifest_digest="sha256:manifest",
                index_completed_at=datetime.now(UTC),
                projection_generation=1,
                projection_lease_generation=1,
            )
        )
        session.add(
            ACLGrantRow(
                grant_id="grant-1",
                tenant_id="tenant-a",
                resource_type="document",
                resource_id="document-a",
                subject_type="user",
                subject_id="user-a",
                permissions=["knowledge:read", "document:read"],
            )
        )
    candidate = RetrievalCandidate(
        tenant_id="tenant-a",
        document_id="document-a",
        version_id="version-1",
        document_version=1,
        projection_generation=1,
        chunk_id="chunk-1",
        content_hash=content_hash,
        source_locator="section/1",
        source_file_name="synthetic.md",
        fused_score=1.0,
        text=text,
    )
    return KnowledgeQueryService(database, _Retriever(candidate), _Model()), database


def _rag_round(root: Path, samples: int, warmup: int) -> list[tuple[float, float]]:
    auth = AuthContext(
        tenant_id="tenant-a",
        user_id="user-a",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="synthetic-request",
    )
    telemetry = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-benchmark",
        otlp_endpoint="",
        trace_sample_ratio=1.0,
    )
    service, database = _query_service(root, "benchmark.sqlite3")

    def off() -> None:
        assert service.query("synthetic question", auth).verdict == "answer"

    def on() -> None:
        with telemetry.activate():
            assert service.query("synthetic question", auth).verdict == "answer"

    try:
        return paired_samples(off, on, samples=samples, warmup=warmup)
    finally:
        database.engine.dispose()
        telemetry.shutdown()


def _rag_benchmark(rounds: int, samples: int, warmup: int) -> dict[str, object]:
    results: list[list[tuple[float, float]]] = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory(prefix="rag-kb-m6-rag-benchmark-") as directory:
            results.append(_rag_round(Path(directory), samples, warmup))
    return _summary(results, 2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--api-samples", type=int, default=600)
    parser.add_argument("--rag-samples", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.disable(logging.CRITICAL)
    api, health, metrics = _api_benchmark(args.rounds, args.api_samples, args.warmup)
    rag = _rag_benchmark(args.rounds, args.rag_samples, args.warmup)
    report = {
        "method": {
            "order": "paired alternating AB/BA",
            "percentile": "linear interpolation at (n-1)*q",
            "overhead": "(enabled_p95 / disabled_p95 - 1) * 100",
            "uncertainty": (
                "hierarchical bootstrap by round then AB/BA strata, percentile 95% interval, "
                "deterministic seed, 1000 resamples"
            ),
            "rounds": args.rounds,
            "warmup_per_arm": args.warmup,
        },
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "api": api,
        "rag_deterministic": rag,
        "health_live": health,
        "metrics": metrics,
    }
    report["passed"] = bool(
        api["passed"] and rag["passed"] and health["passed"] and metrics["passed"]
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
