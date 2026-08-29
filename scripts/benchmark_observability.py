"""可复现的可观测性 A/B 性能门；不访问外部模型或业务数据。"""

from __future__ import annotations

import argparse
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

from rag_kb.application.query_service import KnowledgeQueryService
from rag_kb.config import Settings
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate, RetrievalResult
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
from rag_kb.web.app import create_app


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
    pairs: list[tuple[float, float]], *, repetitions: int = 1000
) -> tuple[float, float]:
    if not pairs or repetitions <= 0:
        raise ValueError("invalid_bootstrap_input")
    generator = random.Random(20260830)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
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
    for index in range(samples):
        if index % 2 == 0:
            off_duration = _measure(off)
            on_duration = _measure(on)
        else:
            on_duration = _measure(on)
            off_duration = _measure(off)
        pairs.append((off_duration, on_duration))
    return pairs


def _summary(pairs: list[tuple[float, float]], limit_percent: float) -> dict[str, object]:
    off = [pair[0] * 1000 for pair in pairs]
    on = [pair[1] * 1000 for pair in pairs]
    p95_overhead = overhead_percent(percentile(off, 0.95), percentile(on, 0.95))
    interval = bootstrap_p95_overhead(pairs)
    return {
        "samples_per_arm": len(pairs),
        "concurrency": 1,
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
        "p95_overhead_percent": p95_overhead,
        "p95_overhead_95ci_percent": [interval[0], interval[1]],
        "error_margin_percent": (interval[1] - interval[0]) / 2,
        "limit_percent": limit_percent,
        "passed": p95_overhead < limit_percent,
    }


def _api_benchmark(samples: int, warmup: int) -> tuple[dict[str, object], float]:
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
    with TestClient(off_app) as off_client, TestClient(on_app) as on_client:
        def off() -> None:
            assert off_client.get("/health/live").status_code == 200

        def on() -> None:
            assert on_client.get("/health/live").status_code == 200

        pairs = paired_samples(off, on, samples=samples, warmup=warmup)
        metric_durations = [
            _measure(lambda: on_client.get("/metrics").raise_for_status()) * 1000
            for _ in range(samples)
        ]
    return _summary(pairs, 5.0), percentile(metric_durations, 0.95)


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


def _rag_benchmark(samples: int, warmup: int) -> dict[str, object]:
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
    with tempfile.TemporaryDirectory(prefix="rag-kb-m6-benchmark-") as directory:
        root = Path(directory)
        off_service, off_database = _query_service(root, "off.sqlite3")
        on_service, on_database = _query_service(root, "on.sqlite3")

        def off() -> None:
            assert off_service.query("synthetic question", auth).verdict == "answer"

        def on() -> None:
            with telemetry.activate():
                assert on_service.query("synthetic question", auth).verdict == "answer"

        try:
            pairs = paired_samples(off, on, samples=samples, warmup=warmup)
        finally:
            off_database.engine.dispose()
            on_database.engine.dispose()
            telemetry.shutdown()
    return _summary(pairs, 2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-samples", type=int, default=600)
    parser.add_argument("--rag-samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.disable(logging.CRITICAL)
    api, metrics_p95 = _api_benchmark(args.api_samples, args.warmup)
    rag = _rag_benchmark(args.rag_samples, args.warmup)
    report = {
        "method": {
            "order": "paired alternating AB/BA",
            "percentile": "linear interpolation at (n-1)*q",
            "overhead": "(enabled_p95 / disabled_p95 - 1) * 100",
            "uncertainty": "paired bootstrap 95% interval, deterministic seed, 1000 resamples",
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
        "metrics": {
            "samples": args.api_samples,
            "concurrency": 1,
            "p95_ms": metrics_p95,
            "limit_ms": 100.0,
            "passed": metrics_p95 < 100.0,
        },
    }
    report["passed"] = bool(api["passed"] and rag["passed"] and report["metrics"]["passed"])
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
