import gc
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from rag_kb.application.document_service import DocumentResult, DocumentVersionResult
from rag_kb.domain.identity import Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from scripts import benchmark_observability as benchmark
from scripts.benchmark_observability import (
    bootstrap_p95_overhead,
    overhead_percent,
    paired_samples,
    percentile,
)


def test_observability_benchmark_math_is_deterministic() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert overhead_percent(2.0, 2.1) == pytest.approx(5.0)
    assert bootstrap_p95_overhead([[(1.0, 1.01)] * 10]) == pytest.approx((1.0, 1.0))


def test_paired_samples_alternates_order_and_keeps_each_pair() -> None:
    calls: list[str] = []

    pairs = paired_samples(
        lambda: calls.append("off"),
        lambda: calls.append("on"),
        samples=2,
        warmup=0,
    )

    assert calls == ["off", "on", "on", "off"]
    assert len(pairs) == 2
    assert all(off > 0 and on > 0 for off, on in pairs)


def test_paired_samples_keeps_warmup_order_and_disables_gc_only_for_samples() -> None:
    calls: list[tuple[str, bool]] = []
    gc_was_enabled = gc.isenabled()
    gc.enable()
    try:
        paired_samples(
            lambda: calls.append(("off", gc.isenabled())),
            lambda: calls.append(("on", gc.isenabled())),
            samples=2,
            warmup=2,
        )
        assert gc.isenabled()
    finally:
        if not gc_was_enabled:
            gc.disable()

    assert calls == [
        ("off", True),
        ("on", True),
        ("on", True),
        ("off", True),
        ("off", False),
        ("on", False),
        ("on", False),
        ("off", False),
    ]


def test_paired_samples_restores_gc_after_callback_error() -> None:
    gc_was_enabled = gc.isenabled()
    gc.enable()

    def fail() -> None:
        assert not gc.isenabled()
        raise RuntimeError("measurement failed")

    try:
        with pytest.raises(RuntimeError, match="measurement failed"):
            paired_samples(fail, lambda: None, samples=1, warmup=0)
        assert gc.isenabled()
    finally:
        if not gc_was_enabled:
            gc.disable()


def test_paired_samples_preserves_initially_disabled_gc() -> None:
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        paired_samples(
            lambda: assert_gc_disabled(),
            lambda: assert_gc_disabled(),
            samples=1,
            warmup=0,
        )
        assert not gc.isenabled()
    finally:
        if gc_was_enabled:
            gc.enable()


def assert_gc_disabled() -> None:
    assert not gc.isenabled()


def test_bootstrap_preserves_round_and_ab_ba_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stops: list[int] = []

    class RecordingRandom:
        def __init__(self, seed: int) -> None:
            assert seed == 20260830

        def randrange(self, stop: int) -> int:
            stops.append(stop)
            return 0

    monkeypatch.setattr(benchmark.random, "Random", RecordingRandom)

    rounds = [
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        [(4.0, 4.0), (5.0, 5.0), (6.0, 6.0)],
    ]
    bootstrap_p95_overhead(rounds, repetitions=2)

    assert stops == [2, 2, 2, 1, 2, 2, 2, 1] * 2


def test_bootstrap_is_deterministic_for_single_round_pair_and_odd_counts() -> None:
    assert bootstrap_p95_overhead([[(2.0, 2.2)]]) == pytest.approx((10.0, 10.0))
    rounds = [[(1.0, 1.01), (2.0, 2.02), (3.0, 3.03)]]
    assert bootstrap_p95_overhead(rounds) == bootstrap_p95_overhead(rounds)


def test_summary_uses_ci_upper_bound_for_formal_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "bootstrap_p95_overhead", lambda _rounds: (-1.0, 6.0))

    summary = benchmark._summary([[(1.0, 1.0)] * 2, [(1.0, 1.0)] * 2], 5.0)

    assert summary["point_estimate_passed"] is True
    assert summary["passed"] is False
    assert summary["decision_basis"] == "hierarchical_p95_overhead_95ci_upper"
    assert summary["p95_overhead_95ci_half_width_percent"] == 3.5
    assert summary["rounds"] == 2
    assert summary["samples_per_arm_per_round"] == 2
    assert len(summary["per_round"]) == 2
    assert "error_margin_percent" not in summary


def test_rag_round_reuses_one_service_and_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    query_states: list[bool] = []
    created_names: list[str] = []
    disposed: list[bool] = []
    shutdown: list[bool] = []

    class Service:
        def query(self, _question: str, _auth: object) -> object:
            query_states.append(active)
            return type("Result", (), {"verdict": "answer"})()

    class Database:
        class Engine:
            def dispose(self) -> None:
                disposed.append(True)

        engine = Engine()

    class Telemetry:
        @contextmanager
        def activate(self):
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def shutdown(self) -> None:
            shutdown.append(True)

    service = Service()
    database = Database()

    def query_service(_root: object, name: str) -> tuple[Service, Database]:
        created_names.append(name)
        return service, database

    def sample_once(off, on, *, samples: int, warmup: int):
        assert (samples, warmup) == (1, 0)
        off()
        on()
        return [(1.0, 1.0)]

    monkeypatch.setattr(benchmark, "Observability", lambda **_kwargs: Telemetry())
    monkeypatch.setattr(benchmark, "_query_service", query_service)
    monkeypatch.setattr(benchmark, "paired_samples", sample_once)
    assert benchmark._rag_round(Path("unused"), 1, 0) == [(1.0, 1.0)]
    assert created_names == ["benchmark.sqlite3"]
    assert query_states == [False, True]
    assert disposed == [True]
    assert shutdown == [True]


def test_document_api_round_shares_dependencies_and_checks_response_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    disposed: list[bool] = []

    class Service:
        def get_document(self, document_id: str, auth: object) -> DocumentResult:
            calls.append((document_id, auth.request_id))
            return DocumentResult(
                document_id=document_id,
                title="Synthetic",
                department="test",
                status="active",
                versions=(
                    DocumentVersionResult(
                        version_id="version-1", version=1, lifecycle_status="active"
                    ),
                ),
            )

    class Database:
        class Engine:
            def dispose(self) -> None:
                disposed.append(True)

        engine = Engine()

    service = Service()
    provider = DemoTokenIdentityProvider(
        {
            "benchmark-token": Principal(
                tenant_id="tenant-a",
                user_id="user-a",
                attributes={"permissions": ["document:read"]},
            )
        }
    )
    monkeypatch.setattr(
        benchmark, "_document_service", lambda _root: (service, Database(), provider)
    )

    document_pairs, health_pairs, metrics = benchmark._document_api_round(tmp_path, 1, 0)

    assert len(document_pairs) == len(health_pairs) == len(metrics) == 1
    assert calls == [("document-a", "benchmark-request")] * 4
    assert disposed == [True]


def test_health_uses_only_enabled_absolute_p95_for_decision() -> None:
    passed = benchmark._health_summary([[(0.010, 0.004)]])
    failed = benchmark._health_summary([[(0.001, 0.005)]])

    assert passed["p95_absolute_delta_ms"] == pytest.approx(-6.0)
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert passed["decision_basis"] == "enabled_p95_absolute"
    assert "p95_overhead_percent" not in passed


def test_benchmark_defaults_to_three_rounds_and_600_samples_per_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_observability.py"])

    args = benchmark._parse_args()

    assert (args.rounds, args.api_samples, args.rag_samples, args.warmup) == (3, 600, 600, 40)
