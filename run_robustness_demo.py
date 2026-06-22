"""
run_robustness_demo.py

End-to-end demonstration of the plot lifecycle, with the focus on the
pre-registration robustness validation that protects production from buggy
user code.

It shows:
  1. The full path a plot takes: write code → register → validate → replay → live.
  2. A ROBUST indicator passing the whole adversarial battery and being registered.
  3. A BUGGY indicator being rejected, with a full pass/fail report.
  4. The same buggy indicator admitted in non-strict mode (report attached, no block).
  5. A short historical replay of the registered robust plot over generated data.
"""
from __future__ import annotations

from pathlib import Path

from plot_compute import (
    PlotRegistry,
    ValidationError,
    build_chain_snapshot,
    compute_one_tick,
    run_robustness_suite,
    save_edge_cases,
)
from plot_compute.validation import validate_compute_signature  # noqa: F401


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# The full lifecycle path (documentation)
# ─────────────────────────────────────────────────────────────────────────────

LIFECYCLE = r"""
  ┌─────────────────────┐
  │  user writes code   │   def compute(ctx): return {"value": ...}
  └──────────┬──────────┘
             ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PlotRegistry.create_plot(plot_id, code)                             │
  │  └─ validate_plot_code()  — 6 stages, runs BEFORE the plot exists:   │
  │        1. syntax        ast.parse(code)                              │
  │        2. safety        block import / eval / exec / open / getattr  │
  │        3. contract      compute(ctx) exists, callable, one arg       │
  │        4. ROBUSTNESS    run code vs 33 adversarial snapshots  ◀───── │
  │                         (NaN / zero / flat / extreme / empty / …)    │
  │        5. result_format each tick returns finite {value|series}      │
  │        6. performance   < 50 ms average over 100 runs                │
  │     any stage fails → ValidationError → plot is NOT registered       │
  └──────────┬──────────────────────────────────────────────────────────┘
             ▼  (validation passed; robustness_report attached to the plot)
  ┌─────────────────────┐
  │  PlotRuntime created │   status = "created"
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │  replay_plot(...)    │   recompute from day start
  │                      │   status: backfilling → live
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │  compute_one_tick()  │   once per 1s snapshot while live
  │   → output_series    │   → WebSocket: initial_series / append_points
  └─────────────────────┘
"""


ROBUST_CODE = """
def compute(ctx):
    # Defensive: safe_div handles empty/zero selections, returns finite numbers.
    call_qty = ctx.chain.sum(option_type="CALL", field="bid_quantity_l1")
    put_qty  = ctx.chain.sum(option_type="PUT",  field="bid_quantity_l1")
    return {"value": safe_div(call_qty - put_qty, call_qty + put_qty)}
"""

BUGGY_CODE = """
def compute(ctx):
    # Naive: mean(iv) is NaN on a data outage / no-quote chain,
    # and dividing by a spread that can be 0 raises ZeroDivisionError.
    iv = ctx.chain.mean(option_type="CALL", field="iv", has_quote=True)
    spread = ctx.chain.mean(field="ask_price_l1") - ctx.chain.mean(field="bid_price_l1")
    return {"value": iv / spread}
"""

MUTATING_CODE = """
def compute(ctx):
    # DANGEROUS: writes into the shared input snapshot in place. The harness
    # detects this and rejects the plot before it can corrupt other plots.
    ctx.chain._snap.bid_quantity[:] = 0
    return {"value": 1.0}
"""


def main() -> None:
    banner("1. THE FULL LIFECYCLE OF A PLOT")
    print(LIFECYCLE)

    # Regenerate the saved battery (so validation_data/ is always current)
    save_edge_cases("validation_data")
    print("  Adversarial battery saved → validation_data/  (pkl + manifest.json + README.md)")

    registry = PlotRegistry()

    # ── 2. Robust indicator: registers cleanly ──────────────────────────────
    banner("2. REGISTERING A ROBUST INDICATOR")
    plot = registry.create_plot("imbalance", ROBUST_CODE)
    print(f"  ✓ Registered '{plot.plot_id}'  (status={plot.status}, version={plot.version})")
    print()
    print(plot.robustness_report.format(show_passes=False))

    # ── 3. Buggy indicator: rejected in strict mode ─────────────────────────
    banner("3. REGISTERING A BUGGY INDICATOR (strict — default)")
    try:
        registry.create_plot("buggy", BUGGY_CODE)
    except ValidationError as e:
        print(f"  ✗ Rejected at stage '{e.stage}'. Plot was NOT registered.\n")
        print(e.message)

    # ── 4. Same buggy indicator: admitted in non-strict mode ────────────────
    banner("4. SAME BUGGY INDICATOR (non-strict — report attached, not blocking)")
    plot_buggy = registry.create_plot("buggy", BUGGY_CODE, strict_robustness=False)
    rep = plot_buggy.robustness_report
    print(f"  ⚠ Admitted '{plot_buggy.plot_id}' with {rep.n_failed} unresolved robustness "
          f"failure(s) out of {rep.total}.")
    print(f"    failure kinds: {rep.kind_counts()}")
    print(f"    weakest categories: "
          + ", ".join(f"{c}={p}/{t}" for c, (p, t) in sorted(rep.category_scores().items()) if p < t))

    # ── 4b. Mutating indicator: rejected (corrupts shared snapshot) ─────────
    banner("4b. REGISTERING A MUTATING INDICATOR (writes to ctx.chain data)")
    try:
        registry.create_plot("mutator", MUTATING_CODE)
    except ValidationError as e:
        report = run_robustness_suite(validate_compute_signature(MUTATING_CODE))
        muts = [f for f in report.failures if f.failure_kind == "mutated_input"]
        print(f"  ✗ Rejected at stage '{e.stage}': {len(muts)} cases caught the in-place write.")
        print(f"    e.g. {muts[0].name}: {muts[0].detail}")
        print("    (In the live loop the same snapshot is shared by every plot — "
              "this would corrupt all of them.)")

    # ── 5. Replay the robust plot over generated data ───────────────────────
    banner("5. HISTORICAL REPLAY OF THE ROBUST PLOT")
    pkl = Path("gen_data/day_data.pkl")
    if not pkl.exists():
        print("  (skipped — run `python -c \"from generate_day_data import save_day; "
              "save_day('gen_data/day_data.pkl')\"` first)")
    else:
        from generate_day_data import load_day
        raw = load_day(str(pkl))[:600]  # first 10 minutes is plenty for a demo
        snapshots = [build_chain_snapshot(s) for s in raw]
        plot.status = "backfilling"
        for snap in snapshots:
            compute_one_tick(plot, snap)
        plot.status = "live"
        pts = plot.output_series["value"]
        vals = [v for _, v in pts]
        print(f"  Replayed {len(pts)} ticks for '{plot.plot_id}'.")
        print(f"    value range : [{min(vals):+.4f}, {max(vals):+.4f}]")
        print(f"    first / last: {vals[0]:+.4f} / {vals[-1]:+.4f}")

    banner("DONE")
    print("  A plot only reaches replay/live if it survived all 33 adversarial cases.")


if __name__ == "__main__":
    main()
