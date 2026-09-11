import gc
import json
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


def test_bootstrap_uses_multiple_rounds_and_both_strata_with_non_degenerate_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections = iter([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

    class ScriptedRandom:
        def __init__(self, seed: int) -> None:
            assert seed == benchmark.BOOTSTRAP_SEED

        def randrange(self, stop: int) -> int:
            selected = next(selections)
            assert selected < stop
            return selected

    rounds = [
        [(1.0, 1.01), (2.0, 2.04), (3.0, 3.09), (4.0, 4.16)],
        [(5.0, 5.25), (6.0, 6.36), (7.0, 7.49), (8.0, 8.64)],
    ]
    monkeypatch.setattr(benchmark.random, "Random", ScriptedRandom)

    interval = bootstrap_p95_overhead(rounds, repetitions=1)
    pooled = [pair for pairs in rounds for pair in pairs]
    expected = overhead_percent(
        percentile([pair[0] for pair in pooled], 0.95),
        percentile([pair[1] for pair in pooled], 0.95),
    )

    assert interval == pytest.approx((expected, expected))
    with pytest.raises(StopIteration):
        next(selections)


def test_bootstrap_is_deterministic_for_single_round_pair_and_odd_counts() -> None:
    assert bootstrap_p95_overhead([[(2.0, 2.2)]]) == pytest.approx((10.0, 10.0))
    rounds = [[(1.0, 1.01), (2.0, 2.02), (3.0, 3.03)]]
    assert bootstrap_p95_overhead(rounds) == bootstrap_p95_overhead(rounds)


def test_summary_uses_ci_upper_bound_for_formal_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol: dict[str, int] = {}

    def fake_bootstrap(
        _rounds: object, *, repetitions: int, seed: int
    ) -> tuple[float, float]:
        protocol.update(repetitions=repetitions, seed=seed)
        return -1.0, 6.0

    monkeypatch.setattr(benchmark, "bootstrap_p95_overhead", fake_bootstrap)

    summary = benchmark._summary([[(1.0, 1.0)] * 2, [(1.0, 1.0)] * 2], 5.0)

    assert summary["point_estimate_passed"] is True
    assert summary["passed"] is False
    assert summary["decision_basis"] == "hierarchical_p95_overhead_95ci_upper"
    assert summary["p95_overhead_95ci_half_width_percent"] == 3.5
    assert summary["rounds"] == 2
    assert summary["samples_per_arm_per_round"] == 2
    assert len(summary["per_round"]) == 2
    assert summary["bootstrap_repetitions"] == 10_000
    assert summary["bootstrap_seed"] == 20260830
    assert protocol == {"repetitions": 10_000, "seed": 20260830}
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


def test_benchmark_defaults_to_ten_rounds_and_180_samples_per_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_observability.py"])

    args = benchmark._parse_args()

    assert (args.rounds, args.api_samples, args.rag_samples, args.warmup) == (
        10,
        180,
        180,
        40,
    )
    assert args.output is None


def test_raw_pairs_record_only_auditable_timing_fields() -> None:
    pairs = benchmark._raw_pairs(
        [[(0.1, 0.2), (0.3, 0.4)], [(0.5, 0.6), (0.7, 0.8)]]
    )

    assert pairs == [
        {"round": 1, "pair": 1, "order": "AB", "off_seconds": 0.1, "on_seconds": 0.2},
        {"round": 1, "pair": 2, "order": "BA", "off_seconds": 0.3, "on_seconds": 0.4},
        {"round": 2, "pair": 1, "order": "AB", "off_seconds": 0.5, "on_seconds": 0.6},
        {"round": 2, "pair": 2, "order": "BA", "off_seconds": 0.7, "on_seconds": 0.8},
    ]


def test_main_writes_auditable_output_and_creates_missing_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "missing" / "nested" / "report.json"
    raw_pair = {
        "round": 1,
        "pair": 1,
        "order": "AB",
        "off_seconds": 0.1,
        "on_seconds": 0.2,
    }

    def fake_api(rounds: int, samples: int, warmup: int):
        assert (rounds, samples, warmup) == (10, 180, 40)
        return (
            {
                "passed": True,
                "rounds": rounds,
                "samples_per_arm": rounds * samples,
                "per_round": [{"round": 1}],
                "raw_pairs": [raw_pair],
            },
            {"passed": True, "raw_pairs": [raw_pair]},
            {"passed": True, "samples": rounds * samples},
        )

    def fake_rag(rounds: int, samples: int, warmup: int):
        assert (rounds, samples, warmup) == (10, 180, 40)
        return {
            "passed": False,
            "rounds": rounds,
            "samples_per_arm": rounds * samples,
            "per_round": [{"round": 1}],
            "raw_pairs": [raw_pair],
        }

    monkeypatch.setattr(sys, "argv", ["benchmark_observability.py", "--output", str(output)])
    monkeypatch.setattr(benchmark, "_api_benchmark", fake_api)
    monkeypatch.setattr(benchmark, "_rag_benchmark", fake_rag)

    assert benchmark.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))

    assert capsys.readouterr().out.strip() == f"FAIL: {output}"
    assert report["passed"] is False
    assert report["method"] == {
        "comparison": "full_observability_off_vs_on",
        "order": "paired alternating AB/BA",
        "percentile": "linear interpolation at (n-1)*q",
        "overhead": "(enabled_p95 / disabled_p95 - 1) * 100",
        "uncertainty": (
            "hierarchical bootstrap by round then AB/BA strata, percentile 95% interval, "
            "deterministic seed, 10000 resamples"
        ),
        "rounds": 10,
        "samples_per_arm_per_round": 180,
        "samples_per_arm_total": 1800,
        "warmup_per_arm": 40,
        "bootstrap_seed": 20260830,
        "bootstrap_repetitions": 10_000,
        "outlier_policy": "retained",
    }
    assert report["api"]["rounds"] == 10
    assert report["api"]["samples_per_arm"] == 1800
    assert report["api"]["raw_pairs"] == [raw_pair]
    assert report["rag_deterministic"]["raw_pairs"] == [raw_pair]
    assert report["health_live"]["raw_pairs"] == [raw_pair]
