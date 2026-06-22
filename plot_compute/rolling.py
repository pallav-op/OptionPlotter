from __future__ import annotations

import re
from collections import defaultdict, deque

import numpy as np

_WINDOW_RE = re.compile(r"^(\d+)(s|m|h)$")
_MULTIPLIERS = {"s": 1_000, "m": 60_000, "h": 3_600_000}


# Default retention ceiling. Generous enough that no realistic intraday window is
# ever truncated (a trading day is ~6.5h), while bounding memory for a long-running
# live process that never resets — fixing the previously unbounded growth.
DEFAULT_MAX_AGE_MS = 24 * 60 * 60 * 1000  # 24 hours


def _parse_window_ms(window: str) -> int:
    m = _WINDOW_RE.match(window.strip())
    if not m:
        raise ValueError(f"Invalid window spec {window!r}. Expected format: '5m', '30s', '1h'.")
    return int(m.group(1)) * _MULTIPLIERS[m.group(2)]


class RollingWindowStore:
    """Per-series deque-based rolling-window calculator over a plot's own output.

    Each series stores ``(timestamp_ms, value)`` tuples in time order. Windowed
    queries (``"5m"``, ``"30s"``, ``"1h"``) drop entries older than the window.

    Memory bound
    ------------
    ``push`` evicts entries older than ``latest_ts - max_age_ms`` (default 24h). This
    fixes the previously unbounded growth without affecting correctness: any window
    up to ``max_age_ms`` returns exact results regardless of query order. Pass a
    smaller ``max_age_ms`` to cap memory harder (windows larger than it are then
    truncated). The most recent point is always retained so ``last`` keeps working.
    """

    def __init__(self, max_age_ms: int | None = DEFAULT_MAX_AGE_MS) -> None:
        self._series: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._max_age_ms = max_age_ms

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, series_name: str, timestamp_ms: int, value: float) -> None:
        dq = self._series[series_name]
        dq.append((timestamp_ms, value))
        if self._max_age_ms is not None:
            cutoff = timestamp_ms - self._max_age_ms
            # Always retain at least the most recent point (needed by last()).
            while len(dq) > 1 and dq[0][0] < cutoff:
                dq.popleft()

    def clear(self) -> None:
        self._series.clear()

    # ------------------------------------------------------------------
    # Window queries
    # ------------------------------------------------------------------

    def _pairs_in_window(self, series_name: str, window: str) -> list[tuple[int, float]]:
        """All (ts, value) within the window, newest data last."""
        window_ms = _parse_window_ms(window)
        dq = self._series.get(series_name)
        if not dq:
            return []
        cutoff = dq[-1][0] - window_ms
        return [(ts, v) for ts, v in dq if ts > cutoff]

    def _values_in_window(self, series_name: str, window: str) -> list[float]:
        return [v for _, v in self._pairs_in_window(series_name, window) if not np.isnan(v)]

    # ------------------------------------------------------------------
    # Aggregations (v1)
    # ------------------------------------------------------------------

    def mean(self, series_name: str, window: str) -> float:
        vals = self._values_in_window(series_name, window)
        return float(np.mean(vals)) if vals else float("nan")

    def sum(self, series_name: str, window: str) -> float:
        vals = self._values_in_window(series_name, window)
        return float(np.sum(vals)) if vals else 0.0

    def min(self, series_name: str, window: str) -> float:
        vals = self._values_in_window(series_name, window)
        return float(np.min(vals)) if vals else float("nan")

    def max(self, series_name: str, window: str) -> float:
        vals = self._values_in_window(series_name, window)
        return float(np.max(vals)) if vals else float("nan")

    def last(self, series_name: str) -> float:
        dq = self._series.get(series_name)
        if not dq:
            return float("nan")
        return float(dq[-1][1])

    # ------------------------------------------------------------------
    # Advanced aggregations
    # ------------------------------------------------------------------

    def std(self, series_name: str, window: str, ddof: int = 1) -> float:
        """Sample standard deviation (ddof=1) over the window. NaN if < 2 points."""
        vals = self._values_in_window(series_name, window)
        if len(vals) < 2:
            return float("nan")
        return float(np.std(vals, ddof=ddof))

    def zscore(self, series_name: str, window: str) -> float:
        """(latest - mean) / std over the window. NaN if std is 0 or < 2 points."""
        vals = self._values_in_window(series_name, window)
        if len(vals) < 2:
            return float("nan")
        sd = float(np.std(vals, ddof=1))
        if sd == 0.0 or np.isnan(sd):
            return float("nan")
        return float((vals[-1] - np.mean(vals)) / sd)

    def ema(self, series_name: str, window: str, span: int | None = None) -> float:
        """Exponentially-weighted mean over the windowed values.

        ``alpha = 2 / (span + 1)``; ``span`` defaults to the number of points in the
        window (i.e. pandas ``ewm(span=N).mean()`` over the window slice). Recursively
        weights the most recent point highest. NaN if the window is empty.
        """
        vals = self._values_in_window(series_name, window)
        if not vals:
            return float("nan")
        n = span if span is not None else len(vals)
        alpha = 2.0 / (n + 1.0)
        ewma = vals[0]
        for v in vals[1:]:
            ewma = alpha * v + (1.0 - alpha) * ewma
        return float(ewma)

    def correlation(self, series_a: str, series_b: str, window: str) -> float:
        """Pearson correlation between two own-output series over the window.

        Pairs are aligned by timestamp (intersection). NaN if fewer than 2 aligned
        finite pairs, or if either series has zero variance over the window.
        """
        pairs_a = {ts: v for ts, v in self._pairs_in_window(series_a, window) if not np.isnan(v)}
        pairs_b = {ts: v for ts, v in self._pairs_in_window(series_b, window) if not np.isnan(v)}
        common = sorted(set(pairs_a) & set(pairs_b))
        if len(common) < 2:
            return float("nan")
        x = np.array([pairs_a[ts] for ts in common], dtype=float)
        y = np.array([pairs_b[ts] for ts in common], dtype=float)
        xc, yc = x - x.mean(), y - y.mean()
        denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
        if denom == 0.0:
            return float("nan")
        return float(np.sum(xc * yc) / denom)
