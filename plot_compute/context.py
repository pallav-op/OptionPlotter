from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .chain_view import ChainView
from .models import ChainSnapshot, PlotRuntime
from .rolling import RollingWindowStore

if TYPE_CHECKING:
    pass


class WindowAPI:
    """Thin wrapper around RollingWindowStore exposed to user compute functions."""

    def __init__(self, store: RollingWindowStore) -> None:
        self._store = store

    def mean(self, series: str, window: str) -> float:
        return self._store.mean(series, window)

    def sum(self, series: str, window: str) -> float:
        return self._store.sum(series, window)

    def min(self, series: str, window: str) -> float:
        return self._store.min(series, window)

    def max(self, series: str, window: str) -> float:
        return self._store.max(series, window)

    def last(self, series: str) -> float:
        return self._store.last(series)

    def std(self, series: str, window: str) -> float:
        return self._store.std(series, window)

    def zscore(self, series: str, window: str) -> float:
        return self._store.zscore(series, window)

    def ema(self, series: str, window: str, span: int | None = None) -> float:
        return self._store.ema(series, window, span=span)

    def correlation(self, series_a: str, series_b: str, window: str) -> float:
        return self._store.correlation(series_a, series_b, window)


@dataclass
class IndicatorContext:
    ts: int
    seq_no: int
    key: str
    chain: ChainView
    state: dict
    window: WindowAPI


def build_context(plot: PlotRuntime, snapshot: ChainSnapshot, rolling_store: RollingWindowStore) -> IndicatorContext:
    return IndicatorContext(
        ts=snapshot.timestamp_ms,
        seq_no=snapshot.seq_no,
        key=snapshot.key,
        chain=ChainView(snapshot),
        state=plot.state,
        window=WindowAPI(rolling_store),
    )
