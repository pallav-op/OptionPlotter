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
