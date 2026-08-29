import pytest

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
    assert bootstrap_p95_overhead([(1.0, 1.01)] * 10) == pytest.approx((1.0, 1.0))


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
