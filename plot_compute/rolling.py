from __future__ import annotations

import re
from collections import defaultdict, deque

import numpy as np

_WINDOW_RE = re.compile(r"^(\d+)(s|m|h)$")
_MULTIPLIERS = {"s": 1_000, "m": 60_000, "h": 3_600_000}


def _parse_window_ms(window: str) -> int:
    m = _WINDOW_RE.match(window.strip())
    if not m:
        raise ValueError(f"Invalid window spec {window!r}. Expected format: '5m', '30s', '1h'.")
    return int(m.group(1)) * _MULTIPLIERS[m.group(2)]


class RollingWindowStore:
    """Per-series deque-based rolling window calculator.

    Each series stores (timestamp_ms, value) tuples. Entries older than the
    requested window are dropped on each query.
    """

    def __init__(self) -> None:
        self._series: dict[str, deque[tuple[int, float]]] = defaultdict(deque)

    def push(self, series_name: str, timestamp_ms: int, value: float) -> None:
        self._series[series_name].append((timestamp_ms, value))

    def _window_values(self, series_name: str, window: str) -> list[float]:
        dq = self._series.get(series_name)
        if not dq:
            return []
        window_ms = _parse_window_ms(window)
        cutoff = dq[-1][0] - window_ms
        return [v for ts, v in dq if ts > cutoff and not np.isnan(v)]

    def mean(self, series_name: str, window: str) -> float:
        vals = self._window_values(series_name, window)
        return float(np.mean(vals)) if vals else float("nan")

    def sum(self, series_name: str, window: str) -> float:
        vals = self._window_values(series_name, window)
        return float(np.sum(vals)) if vals else 0.0

    def min(self, series_name: str, window: str) -> float:
        vals = self._window_values(series_name, window)
        return float(np.min(vals)) if vals else float("nan")

    def max(self, series_name: str, window: str) -> float:
        vals = self._window_values(series_name, window)
        return float(np.max(vals)) if vals else float("nan")

    def last(self, series_name: str) -> float:
        dq = self._series.get(series_name)
        if not dq:
            return float("nan")
        return float(dq[-1][1])

    def clear(self) -> None:
        self._series.clear()
