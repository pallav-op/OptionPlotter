import asyncio

import pytest

from plot_compute import (
    IterableSnapshotSource,
    LatestSnapshotSource,
    LiveScheduler,
    ManualClock,
    PlotRegistry,
    build_sample_snapshot,
    make_append_points,
    make_batch,
)
from plot_compute.models import PlotRuntime
from plot_compute.validation import validate_compute_signature

ROBUST_CODE = """
def compute(ctx):
    c = ctx.chain.sum(option_type="CALL", field="bid_quantity_l1")
    p = ctx.chain.sum(option_type="PUT", field="bid_quantity_l1")
    return {"value": safe_div(c - p, c + p)}
"""


def _live_plot(plot_id, compute_fn):
    return PlotRuntime(
        plot_id=plot_id, version=1, code="", compute_fn=compute_fn, status="live",
    )


def _registry_with(*plots):
    r = PlotRegistry()
    for p in plots:
        r._plots[p.plot_id] = p
    return r


# ---------------------------------------------------------------------------
# tick()
# ---------------------------------------------------------------------------

def test_tick_processes_active_plots():
    fn = validate_compute_signature(ROBUST_CODE)
    r = _registry_with(_live_plot("a", fn), _live_plot("b", fn))
    sched = LiveScheduler(r, IterableSnapshotSource([]))
    snap = build_sample_snapshot()

    batch = asyncio.run(sched.tick(snap))
    assert len(batch) == 2
    assert {u["plot_id"] for u in batch} == {"a", "b"}
    assert all(u["type"] == "append_points" for u in batch)
    assert all("value" in u["points"] for u in batch)


def test_tick_skips_paused_plots():
    fn = validate_compute_signature(ROBUST_CODE)
    live = _live_plot("live", fn)
    paused = _live_plot("paused", fn)
    paused.status = "paused"
    r = _registry_with(live, paused)
    sched = LiveScheduler(r, IterableSnapshotSource([]))

    batch = asyncio.run(sched.tick(build_sample_snapshot()))
    assert [u["plot_id"] for u in batch] == ["live"]


def test_error_isolation_one_plot_does_not_stop_others():
    good = _live_plot("good", validate_compute_signature(ROBUST_CODE))
    bad = _live_plot("bad", lambda ctx: 1 / 0)  # raises every tick
    r = _registry_with(good, bad)
    sched = LiveScheduler(r, IterableSnapshotSource([]))

    batch = asyncio.run(sched.tick(build_sample_snapshot()))
    by_id = {u["plot_id"]: u for u in batch}
    assert by_id["good"]["type"] == "append_points"
    assert by_id["bad"]["type"] == "plot_error"
    # the bad plot is quarantined (no longer active)
    assert bad.status == "error"
    assert r.active_plots() == [good]


# ---------------------------------------------------------------------------
# run() with deterministic clock + iterable source
# ---------------------------------------------------------------------------

def test_run_consumes_source_and_collects_batches():
    fn = validate_compute_signature(ROBUST_CODE)
    r = _registry_with(_live_plot("a", fn))
    snaps = [build_sample_snapshot() for _ in range(5)]

    batches = []
    sched = LiveScheduler(
        r, IterableSnapshotSource(snaps),
        publish=batches.append, clock=ManualClock(),
    )
    ticks = asyncio.run(sched.run())
    assert ticks == 5
    assert len(batches) == 5
    assert all(len(b) == 1 for b in batches)


def test_run_respects_max_ticks():
    fn = validate_compute_signature(ROBUST_CODE)
    r = _registry_with(_live_plot("a", fn))
    snaps = [build_sample_snapshot() for _ in range(100)]
    sched = LiveScheduler(r, IterableSnapshotSource(snaps), clock=ManualClock())
    ticks = asyncio.run(sched.run(max_ticks=3))
    assert ticks == 3


def test_run_stops_when_source_exhausted():
    fn = validate_compute_signature(ROBUST_CODE)
    r = _registry_with(_live_plot("a", fn))
    sched = LiveScheduler(r, IterableSnapshotSource([build_sample_snapshot()]),
                          clock=ManualClock())
    assert asyncio.run(sched.run()) == 1


def test_latest_snapshot_source():
    src = LatestSnapshotSource()
    assert asyncio.run(src.get_next()) is None
    snap = build_sample_snapshot()
    src.set_latest(snap)
    assert asyncio.run(src.get_next()) is snap


# ---------------------------------------------------------------------------
# per-tick timeout
# ---------------------------------------------------------------------------

def test_per_tick_timeout_quarantines_hung_plot():
    import time as _time

    def hang(ctx):
        _time.sleep(0.5)
        return {"value": 1.0}

    good = _live_plot("good", validate_compute_signature(ROBUST_CODE))
    slow = _live_plot("slow", hang)
    r = _registry_with(good, slow)
    sched = LiveScheduler(r, IterableSnapshotSource([]), per_tick_timeout=0.05)

    batch = asyncio.run(sched.tick(build_sample_snapshot()))
    by_id = {u["plot_id"]: u for u in batch}
    assert by_id["good"]["type"] == "append_points"
    assert by_id["slow"]["type"] == "plot_error"
    assert "timeout" in slow.error.lower()
    assert slow.status == "error"


# ---------------------------------------------------------------------------
# message builders
# ---------------------------------------------------------------------------

def test_make_batch_shape():
    fn = validate_compute_signature(ROBUST_CODE)
    plot = _live_plot("p", fn)
    r = _registry_with(plot)
    sched = LiveScheduler(r, IterableSnapshotSource([]))
    batch = asyncio.run(sched.tick(build_sample_snapshot()))
    wrapped = make_batch(batch)
    assert wrapped["type"] == "batch_append_points"
    assert wrapped["updates"] == batch
