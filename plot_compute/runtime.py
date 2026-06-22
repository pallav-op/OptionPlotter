from __future__ import annotations

from typing import Callable, Iterable

from .context import build_context
from .errors import RuntimePlotError
from .models import ChainSnapshot, PlotResult, PlotRuntime
from .result_format import validate_result
from .rolling import RollingWindowStore

# Per-plot rolling stores — keyed by plot_id
_rolling_stores: dict[str, RollingWindowStore] = {}


def _get_rolling_store(plot_id: str) -> RollingWindowStore:
    if plot_id not in _rolling_stores:
        _rolling_stores[plot_id] = RollingWindowStore()
    return _rolling_stores[plot_id]


def _clear_rolling_store(plot_id: str) -> None:
    if plot_id in _rolling_stores:
        _rolling_stores[plot_id].clear()


def compute_one_tick(plot: PlotRuntime, snapshot: ChainSnapshot) -> PlotResult:
    """Core computation step used by both historical replay and live ticking."""
    rolling = _get_rolling_store(plot.plot_id)
    ctx = build_context(plot, snapshot, rolling)

    try:
        raw = plot.compute_fn(ctx)
    except Exception as exc:
        raise RuntimePlotError(plot.plot_id, plot.version, str(exc)) from exc

    normalised = validate_result(raw)
    result = PlotResult(series=normalised)
    _update_plot_series(plot, rolling, result, snapshot.timestamp_ms)
    return result


def _update_plot_series(
    plot: PlotRuntime,
    rolling: RollingWindowStore,
    result: PlotResult,
    timestamp_ms: int,
) -> None:
    for series_name, value in result.series.items():
        if series_name not in plot.output_series:
            plot.output_series[series_name] = []
        plot.output_series[series_name].append((timestamp_ms, value))
        rolling.push(series_name, timestamp_ms, value)
    plot.last_computed_ts = timestamp_ms


def replay_plot(plot: PlotRuntime, snapshots: Iterable[ChainSnapshot]) -> None:
    """Replay compute from day start, rebuilding state and output series."""
    plot.status = "backfilling"
    plot.state.clear()
    plot.output_series.clear()
    _clear_rolling_store(plot.plot_id)

    for snapshot in snapshots:
        if plot.status == "deleted":
            return
        try:
            compute_one_tick(plot, snapshot)
        except RuntimePlotError:
            # During replay, log and continue rather than aborting the whole replay
            raise

    plot.status = "live"
