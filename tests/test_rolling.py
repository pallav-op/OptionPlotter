import math

import pytest

from plot_compute import RollingWindowStore


def test_mean_window():
    store = RollingWindowStore()
    for i in range(10):
        store.push("value", i * 60_000, float(i))  # one per minute
    # 5-minute window: last 5 values (5,6,7,8,9)
    result = store.mean("value", "5m")
    assert abs(result - 7.0) < 1e-9


def test_sum_window():
    store = RollingWindowStore()
    for i in range(6):
        store.push("value", i * 60_000, 1.0)
    result = store.sum("value", "3m")
    assert result == 3.0


def test_last():
    store = RollingWindowStore()
    store.push("value", 0, 10.0)
    store.push("value", 1000, 20.0)
    assert store.last("value") == 20.0


def test_min_max():
    store = RollingWindowStore()
    for v in [3.0, 1.0, 4.0, 1.0, 5.0]:
        store.push("value", 0, v)
    assert store.min("value", "1h") == 1.0
    assert store.max("value", "1h") == 5.0


def test_empty_series_nan():
    store = RollingWindowStore()
    assert math.isnan(store.mean("missing", "5m"))
    assert math.isnan(store.last("missing"))


def test_invalid_window_raises():
    store = RollingWindowStore()
    store.push("value", 0, 1.0)
    with pytest.raises(ValueError, match="Invalid window spec"):
        store.mean("value", "5x")


def test_clear():
    store = RollingWindowStore()
    store.push("value", 0, 1.0)
    store.clear()
    assert math.isnan(store.last("value"))


# ---------------------------------------------------------------------------
# Advanced aggregations
# ---------------------------------------------------------------------------

def test_std_basic():
    store = RollingWindowStore()
    for i, v in enumerate([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]):
        store.push("value", i * 1000, v)
    # sample std (ddof=1) of these 8 values is ~2.138
    assert abs(store.std("value", "1h") - 2.138089935) < 1e-6


def test_std_insufficient_points_is_nan():
    store = RollingWindowStore()
    store.push("value", 0, 5.0)
    assert math.isnan(store.std("value", "1h"))


def test_zscore():
    store = RollingWindowStore()
    for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        store.push("value", i * 1000, v)
    # latest=5, mean=3, sample std=1.5811 → z=(5-3)/1.5811≈1.2649
    assert abs(store.zscore("value", "1h") - 1.264911064) < 1e-6


def test_zscore_zero_variance_is_nan():
    store = RollingWindowStore()
    for i in range(5):
        store.push("value", i * 1000, 3.0)  # flat → std 0
    assert math.isnan(store.zscore("value", "1h"))


def test_ema_single_point_equals_value():
    store = RollingWindowStore()
    store.push("value", 0, 7.0)
    assert store.ema("value", "1h") == 7.0


def test_ema_recency_weighted():
    store = RollingWindowStore()
    for i, v in enumerate([1.0, 2.0, 3.0, 4.0]):
        store.push("value", i * 1000, v)
    ema = store.ema("value", "1h")
    # EMA weights recent points more → above the simple mean (2.5), below the last (4)
    assert 2.5 < ema < 4.0


def test_correlation_perfect_positive():
    store = RollingWindowStore()
    for i in range(5):
        store.push("a", i * 1000, float(i))
        store.push("b", i * 1000, 2.0 * i + 1.0)  # perfectly linear in a
    assert abs(store.correlation("a", "b", "1h") - 1.0) < 1e-9


def test_correlation_perfect_negative():
    store = RollingWindowStore()
    for i in range(5):
        store.push("a", i * 1000, float(i))
        store.push("b", i * 1000, -float(i))
    assert abs(store.correlation("a", "b", "1h") + 1.0) < 1e-9


def test_correlation_insufficient_is_nan():
    store = RollingWindowStore()
    store.push("a", 0, 1.0)
    store.push("b", 0, 2.0)
    assert math.isnan(store.correlation("a", "b", "1h"))


# ---------------------------------------------------------------------------
# Memory-leak fix: max-age retention ceiling
# ---------------------------------------------------------------------------

def test_eviction_respects_max_age():
    store = RollingWindowStore(max_age_ms=5000)  # keep only the last 5 seconds
    for i in range(100):
        store.push("value", i * 1000, float(i))
    dq = store._series["value"]
    # 5s ceiling at 1s cadence keeps ~6 points, not all 100
    assert len(dq) <= 6
    assert dq[-1][1] == 99.0  # latest always retained


def test_default_ceiling_keeps_intraday_history():
    # Default 24h ceiling → a full intraday push-then-query is never truncated,
    # regardless of query order (the bug the self-tuning version had).
    store = RollingWindowStore()
    for i in range(100):
        store.push("value", i * 1000, float(i))
    assert len(store._series["value"]) == 100
    # querying a large window after loading still sees everything
    assert store.mean("value", "1h") == sum(range(100)) / 100


def test_ceiling_never_drops_last_point():
    store = RollingWindowStore(max_age_ms=1)  # aggressive
    for i in range(10):
        store.push("value", i * 1000, float(i))
    assert store.last("value") == 9.0  # last point survives even when stale
