import pytest

from plot_compute import (
    PlotRegistry,
    RuntimePlotError,
    ValidationError,
    build_sample_snapshot,
    compute_one_tick,
    replay_plot,
    validate_plot_code,
)
from plot_compute.models import PlotRuntime


def _make_plot(code: str, plot_id: str = "p1") -> PlotRuntime:
    fn, _ = validate_plot_code(code)
    return PlotRuntime(
        plot_id=plot_id,
        version=1,
        code=code,
        compute_fn=fn,
        status="live",
    )


# ---------------------------------------------------------------------------
# compute_one_tick
# ---------------------------------------------------------------------------

CODE_SIMPLE = """
def compute(ctx):
    call_qty = ctx.chain.sum(option_type="CALL", field="bid_quantity_l1")
    put_qty = ctx.chain.sum(option_type="PUT", field="bid_quantity_l1")
    return {"value": safe_div(call_qty - put_qty, call_qty + put_qty)}
"""


def test_compute_one_tick_returns_result():
    plot = _make_plot(CODE_SIMPLE)
    snap = build_sample_snapshot()
    result = compute_one_tick(plot, snap)
    assert "value" in result.series
    assert isinstance(result.series["value"], float)


def test_compute_one_tick_updates_output_series():
    plot = _make_plot(CODE_SIMPLE)
    snap = build_sample_snapshot()
    compute_one_tick(plot, snap)
    assert "value" in plot.output_series
    assert len(plot.output_series["value"]) == 1


def test_compute_one_tick_state_persists():
    code = """
def compute(ctx):
    ctx.state["count"] = ctx.state.get("count", 0) + 1
    return {"value": float(ctx.state["count"])}
"""
    plot = _make_plot(code)
    snap = build_sample_snapshot()
    compute_one_tick(plot, snap)
    compute_one_tick(plot, snap)
    compute_one_tick(plot, snap)
    assert plot.state["count"] == 3


def test_compute_one_tick_raises_on_user_error():
    # A compute that raises at runtime must surface as RuntimePlotError.
    # Construct the plot directly to bypass validation (validation would reject it).
    plot = PlotRuntime(
        plot_id="err_plot",
        version=1,
        code="def compute(ctx): return 1 / 0",
        compute_fn=lambda ctx: 1 / 0,
        status="live",
    )
    snap = build_sample_snapshot()
    with pytest.raises(RuntimePlotError):
        compute_one_tick(plot, snap)


# ---------------------------------------------------------------------------
# replay_plot
# ---------------------------------------------------------------------------

def test_replay_builds_full_series():
    plot = _make_plot(CODE_SIMPLE, plot_id="replay_p1")
    snap = build_sample_snapshot()
    snapshots = [snap] * 5
    replay_plot(plot, snapshots)
    assert plot.status == "live"
    assert len(plot.output_series["value"]) == 5


def test_replay_resets_state():
    code = """
def compute(ctx):
    ctx.state["count"] = ctx.state.get("count", 0) + 1
    return {"value": float(ctx.state["count"])}
"""
    plot = _make_plot(code, plot_id="replay_p2")
    snap = build_sample_snapshot()
    replay_plot(plot, [snap] * 3)
    assert plot.state["count"] == 3

    # Second replay should restart from 0
    replay_plot(plot, [snap] * 2)
    assert plot.state["count"] == 2


# ---------------------------------------------------------------------------
# PlotRegistry
# ---------------------------------------------------------------------------

def test_registry_create_and_get():
    r = PlotRegistry()
    plot = r.create_plot("plot1", CODE_SIMPLE)
    assert r.get_plot("plot1") is plot


def test_registry_duplicate_raises():
    r = PlotRegistry()
    r.create_plot("dup", CODE_SIMPLE)
    with pytest.raises(ValueError):
        r.create_plot("dup", CODE_SIMPLE)


def test_registry_update_increments_version():
    r = PlotRegistry()
    r.create_plot("upd", CODE_SIMPLE)
    plot = r.update_plot("upd", CODE_SIMPLE)
    assert plot.version == 2


def test_registry_update_clears_series():
    r = PlotRegistry()
    plot = r.create_plot("clr", CODE_SIMPLE)
    snap = build_sample_snapshot()
    compute_one_tick(plot, snap)
    assert plot.output_series

    r.update_plot("clr", CODE_SIMPLE)
    assert not plot.output_series


def test_registry_invalid_update_keeps_old_version():
    r = PlotRegistry()
    r.create_plot("safe", CODE_SIMPLE)
    bad_code = "def compute(ctx): import os"
    with pytest.raises(ValidationError):
        r.update_plot("safe", bad_code)
    assert r.get_plot("safe").version == 1


def test_registry_delete():
    r = PlotRegistry()
    r.create_plot("del", CODE_SIMPLE)
    r.delete_plot("del")
    with pytest.raises(KeyError):
        r.get_plot("del")


def test_registry_pause_resume():
    r = PlotRegistry()
    plot = r.create_plot("pr", CODE_SIMPLE)
    plot.status = "live"
    r.pause_plot("pr")
    assert plot.status == "paused"
    r.resume_plot("pr")
    assert plot.status == "live"


def test_active_plots_filters_paused():
    r = PlotRegistry()
    r.create_plot("a1", CODE_SIMPLE)
    r.create_plot("a2", CODE_SIMPLE)
    r._plots["a1"].status = "live"
    r._plots["a2"].status = "paused"
    active = r.active_plots()
    assert len(active) == 1
    assert active[0].plot_id == "a1"
