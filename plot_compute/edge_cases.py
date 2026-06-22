"""Adversarial edge-case market data for robustness validation.

Each :class:`EdgeCase` pairs a deliberately hostile :class:`ChainSnapshot` with a
human-readable description of *why* it is dangerous. User ``compute(ctx)`` code is
run against the whole battery before a plot is registered, so that bugs that would
otherwise only surface in production (a NaN data outage, an empty chain at the open,
a divide-by-zero on a quiet strike) are caught up front.

The battery is intentionally exhaustive. Categories:

    structural  — odd chain shapes (empty, calls-only, no book levels, huge)
    nan         — NaN injected into prices / greeks / quantities / underlying
    zero        — zeros that invite divide-by-zero (quantities, prices, volume)
    flat        — zero cross-sectional variance (breaks std / zscore / range-normalisation)
    extreme     — overflow / underflow / negative / infinite values
    quote       — has_quote all-False or mixed (filters collapse to empty selections)
    normal      — healthy baselines that *should* always pass
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import ChainSnapshot

_BASE_TS_MS = 1_700_000_000_000

# Field groups for the flexible builder below.
_TWO_D_FIELDS = frozenset({
    "bid_price", "bid_quantity", "bid_order_count",
    "ask_price", "ask_quantity", "ask_order_count",
})

# Invariants checked on EVERY tick of EVERY case. A case passes only if all hold.
# (Failure-kind constants live in robustness.py: EXCEPTION / MUTATED_INPUT /
#  BAD_FORMAT / NON_FINITE — kept in sync with this list.)
INVARIANTS = [
    {"kind": "exception",
     "rule": "compute(ctx) must not raise on any market-data shape."},
    {"kind": "mutated_input",
     "rule": "compute(ctx) must not modify the input snapshot in place — the same "
             "ChainSnapshot is shared across all plots in a live tick, so mutating it "
             "corrupts every other plot and breaks replay determinism."},
    {"kind": "bad_format",
     "rule": "the return must be {'value': number} or {'series': {name: number, ...}}."},
    {"kind": "non_finite",
     "rule": "every returned number must be finite (no NaN / inf)."},
]


@dataclass(frozen=True)
class EdgeCase:
    name: str
    category: str
    description: str
    snapshot: ChainSnapshot

    @property
    def n_options(self) -> int:
        return int(len(self.snapshot.option_type))

    @property
    def n_levels(self) -> int:
        return int(self.snapshot.bid_price.shape[1]) if self.snapshot.bid_price.ndim == 2 else 0


# ---------------------------------------------------------------------------
# Flexible snapshot builder — realistic defaults, override anything per case
# ---------------------------------------------------------------------------

def _snapshot(option_types, *, levels: int = 3, **overrides) -> ChainSnapshot:
    """Build a ChainSnapshot from a list of option-type strings with sane defaults.

    Any field can be overridden. A scalar override broadcasts across all rows
    (and across levels for the 2-D book fields); an array override is used verbatim.
    """
    types = list(option_types)
    n = len(types)

    def col(v):
        return np.full(n, v, dtype=float)

    if n:
        base_px = np.linspace(1.0, 10.0, n)
    else:
        base_px = np.empty(0, dtype=float)

    # 1-D realistic defaults
    fields = {
        "option_type": np.array(types, dtype=object),
        "strike_px": (np.linspace(90.0, 110.0, n) if n else np.empty(0)),
        "expiry_timestamp": col(1_700_000_000),
        "mid_px": base_px.copy(),
        "last_traded_price": base_px.copy(),
        "total_traded_quantity": col(100.0),
        "total_traded_value": col(10_000.0),
        "volume_since_day_start": col(500.0),
        "moneyness": ((np.arange(n) - n // 2).astype(float) if n else np.empty(0)),
        "has_quote": np.ones(n, dtype=bool),
        "low_since_day_start": (base_px * 0.5),
        "high_since_day_start": (base_px * 1.5),
        "book_levels_per_side": col(levels),
        "underlying_price": col(100.0),
        "tte": col(0.1),
        "delta": (np.linspace(-0.9, 0.9, n) if n else np.empty(0)),
        "gamma": col(0.02),
        "vega": col(0.15),
        "theta": col(-0.05),
        "iv": col(0.25),
        "rate_of_interest": col(0.05),
    }

    # 2-D book defaults (shape [n, levels])
    if n and levels:
        b = base_px.reshape(n, 1)
        off = np.arange(levels).reshape(1, levels) * 0.1
        fields["bid_price"] = np.maximum(b - 0.5 - off, 0.01)
        fields["ask_price"] = b + 0.5 + off
        fields["bid_quantity"] = np.full((n, levels), 50.0)
        fields["ask_quantity"] = np.full((n, levels), 50.0)
        fields["bid_order_count"] = np.full((n, levels), 5.0)
        fields["ask_order_count"] = np.full((n, levels), 5.0)
    else:
        empty2d = np.empty((n, levels), dtype=float)
        for k in _TWO_D_FIELDS:
            fields[k] = empty2d.copy()

    # Apply overrides
    for key, val in overrides.items():
        if key in _TWO_D_FIELDS:
            fields[key] = np.full((n, levels), val, dtype=float) if np.isscalar(val) else np.asarray(val, dtype=float)
        elif key == "option_type":
            fields[key] = np.array(val, dtype=object)
        elif key == "has_quote":
            fields[key] = np.full(n, val, dtype=bool) if np.isscalar(val) else np.asarray(val, dtype=bool)
        else:
            fields[key] = np.full(n, val, dtype=float) if np.isscalar(val) else np.asarray(val, dtype=float)

    return ChainSnapshot(
        seq_no=1,
        key="EDGE",
        timestamp_ns=_BASE_TS_MS * 1_000_000,
        timestamp_ms=_BASE_TS_MS,
        **fields,
    )


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

def build_edge_cases() -> list[EdgeCase]:
    """Return the full adversarial edge-case battery."""
    C = lambda k: ["CALL"] * k      # noqa: E731
    P = lambda k: ["PUT"] * k       # noqa: E731
    mix = C(4) + P(4)               # default n=8 mixed chain
    n = len(mix)

    cases: list[EdgeCase] = []

    def add(name, category, description, snap):
        cases.append(EdgeCase(name=name, category=category, description=description, snapshot=snap))

    # ---- structural --------------------------------------------------------
    add("empty_chain", "structural",
        "Zero options in the chain. count()==0 → any division by a count/sum blows up.",
        _snapshot([]))
    add("single_call_only", "structural",
        "Exactly one CALL, no PUTs. Put-side aggregations are empty.",
        _snapshot(C(1)))
    add("single_put_only", "structural",
        "Exactly one PUT, no CALLs. Call-side aggregations are empty.",
        _snapshot(P(1)))
    add("calls_only", "structural",
        "Several CALLs, zero PUTs. put_qty==0 → call/put ratios divide by zero.",
        _snapshot(C(6)))
    add("puts_only", "structural",
        "Several PUTs, zero CALLs. call_qty==0 → put/call ratios divide by zero.",
        _snapshot(P(6)))
    add("single_book_level", "structural",
        "Only L1 exists. Accessing bid_price_l2 / _l3 hits missing levels.",
        _snapshot(mix, levels=1))
    add("no_book_levels", "structural",
        "Book has zero levels per side. Every _l1 access is out of range.",
        _snapshot(mix, levels=0))
    add("large_dense_chain", "structural",
        "100 options (50C/50P), 3 levels. Stresses per-tick cost and accumulation.",
        _snapshot(C(50) + P(50), levels=3))

    # ---- nan ---------------------------------------------------------------
    add("all_nan_book_prices", "nan",
        "Every bid/ask price is NaN. mean(bid_price_l1) → NaN unless guarded.",
        _snapshot(mix, bid_price=np.nan, ask_price=np.nan))
    add("all_nan_greeks", "nan",
        "delta/gamma/vega/theta/iv all NaN (greek-engine outage).",
        _snapshot(mix, delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, iv=np.nan))
    add("all_nan_iv", "nan",
        "Every IV is NaN. mean(iv) → NaN; IV-based indicators must use nz()/safe defaults.",
        _snapshot(mix, iv=np.nan))
    add("some_nan_iv", "nan",
        "Half the IVs are NaN (partial outage). Mixed NaN/finite aggregation.",
        _snapshot(mix, iv=np.where(np.arange(n) % 2 == 0, 0.25, np.nan)))
    add("nan_underlying", "nan",
        "underlying_price is NaN. Any moneyness/ratio vs underlying → NaN.",
        _snapshot(mix, underlying_price=np.nan))
    add("all_nan_quantities", "nan",
        "Every bid/ask quantity is NaN. nansum→0 but ratios of NaN qty can surprise.",
        _snapshot(mix, bid_quantity=np.nan, ask_quantity=np.nan))

    # ---- zero --------------------------------------------------------------
    add("all_zero_quantities", "zero",
        "All bid/ask quantities are 0. sum==0 → imbalance ratios divide by zero.",
        _snapshot(mix, bid_quantity=0.0, ask_quantity=0.0,
                  volume_since_day_start=0.0, total_traded_quantity=0.0))
    add("all_zero_prices", "zero",
        "All prices are 0. Division by price and log(price) blow up.",
        _snapshot(mix, bid_price=0.0, ask_price=0.0, mid_px=0.0, last_traded_price=0.0))
    add("zero_volume", "zero",
        "volume_since_day_start and traded value are 0. VWAP-style divide by volume.",
        _snapshot(mix, volume_since_day_start=0.0, total_traded_value=0.0, total_traded_quantity=0.0))
    add("zero_underlying", "zero",
        "underlying_price is 0. K/S and log(K/S) blow up.",
        _snapshot(mix, underlying_price=0.0))
    add("all_zero_everything", "zero",
        "Maximum divide-by-zero exposure: prices, quantities, greeks, volume all 0.",
        _snapshot(mix, bid_price=0.0, ask_price=0.0, mid_px=0.0, last_traded_price=0.0,
                  bid_quantity=0.0, ask_quantity=0.0, volume_since_day_start=0.0,
                  total_traded_quantity=0.0, total_traded_value=0.0, iv=0.0,
                  delta=0.0, gamma=0.0, vega=0.0, theta=0.0))

    # ---- flat (zero variance) ---------------------------------------------
    add("flat_strikes", "flat",
        "All strikes identical. Strike-spread / skew-vs-strike calcs degenerate.",
        _snapshot(mix, strike_px=100.0))
    add("flat_prices", "flat",
        "All mid/bid/ask prices identical. spread==0, range==0 → range-normalisation /0.",
        _snapshot(mix, mid_px=5.0, bid_price=4.5, ask_price=5.5))
    add("flat_delta", "flat",
        "All deltas identical. std(delta)==0 → zscore divides by zero std.",
        _snapshot(mix, delta=0.5))
    add("identical_options", "flat",
        "Every row identical (zero cross-sectional variance everywhere).",
        _snapshot(mix, strike_px=100.0, mid_px=5.0, iv=0.25, delta=0.5,
                  bid_price=4.5, ask_price=5.5, moneyness=0.0))

    # ---- extreme -----------------------------------------------------------
    add("extreme_large_values", "extreme",
        "Huge prices/quantities (1e15+). Products can overflow to inf.",
        _snapshot(mix, mid_px=1e15, bid_price=1e15, ask_price=1e15,
                  bid_quantity=1e12, ask_quantity=1e12, total_traded_value=1e18))
    add("extreme_small_values", "extreme",
        "Tiny values (1e-15). Divide-by-near-zero produces huge/inf results.",
        _snapshot(mix, mid_px=1e-15, bid_price=1e-12, ask_price=1e-12, iv=1e-9))
    add("negative_prices", "extreme",
        "Negative bid/ask/mid prices (bad feed). log()/sqrt() → domain errors / NaN.",
        _snapshot(mix, bid_price=-5.0, ask_price=-5.0, mid_px=-5.0))
    add("negative_greeks_extreme", "extreme",
        "Out-of-range greeks (delta=-5, gamma=-3). Sign/range assumptions break.",
        _snapshot(mix, delta=-5.0, gamma=-3.0))
    add("inf_prices", "extreme",
        "+inf bid/ask prices. inf propagates through any arithmetic.",
        _snapshot(mix, bid_price=np.inf, ask_price=np.inf))
    add("neg_inf_prices", "extreme",
        "-inf bid prices. -inf propagates; min()/comparisons surprise.",
        _snapshot(mix, bid_price=-np.inf))

    # ---- quote -------------------------------------------------------------
    add("all_no_quote", "quote",
        "has_quote is False everywhere. Filtering has_quote=True yields an empty set.",
        _snapshot(mix, has_quote=False))
    add("mixed_quote", "quote",
        "has_quote alternates. has_quote=True selects a sparse subset.",
        _snapshot(mix, has_quote=(np.arange(n) % 2 == 0)))

    # ---- normal baselines --------------------------------------------------
    add("normal_realistic", "normal",
        "Healthy 4C/4P chain with 3 book levels. Sanity baseline — should always pass.",
        _snapshot(mix))
    add("normal_two_expiries", "normal",
        "Healthy chain spanning two expiries. Baseline for expiry-filtered indicators.",
        _snapshot(mix, expiry_timestamp=np.where(np.arange(n) < n // 2, 1_700_000_000, 1_702_000_000)))

    return cases


# ---------------------------------------------------------------------------
# Persistence — save the battery so it can be inspected / reused as a fixture
# ---------------------------------------------------------------------------

def save_edge_cases(directory: str = "validation_data") -> Path:
    """Materialise the battery to ``directory``: edge_cases.pkl + manifest.json + README.md."""
    from .robustness import DEFAULT_REPEAT  # local import avoids an import cycle

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_edge_cases()

    (out / "edge_cases.pkl").write_bytes(pickle.dumps(cases, protocol=5))

    manifest = {
        "n_cases": len(cases),
        "ticks_per_case": DEFAULT_REPEAT,
        "invariants": INVARIANTS,
        "cases": [
            {
                "name": c.name,
                "category": c.category,
                "description": c.description,
                "n_options": c.n_options,
                "n_levels": c.n_levels,
            }
            for c in cases
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(_render_readme(cases))
    return out


def load_edge_cases(directory: str = "validation_data") -> list[EdgeCase]:
    """Load a previously saved battery (falls back to building fresh if absent)."""
    path = Path(directory) / "edge_cases.pkl"
    if not path.exists():
        return build_edge_cases()
    return pickle.loads(path.read_bytes())


def _render_readme(cases: list[EdgeCase]) -> str:
    by_cat: dict[str, list[EdgeCase]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)

    lines = [
        "# Robustness Validation Data",
        "",
        f"Auto-generated by `plot_compute.edge_cases.save_edge_cases()`. "
        f"{len(cases)} adversarial market-data snapshots used to stress-test user "
        "`compute(ctx)` code **before** a plot is registered.",
        "",
        "Each snapshot is run for several consecutive ticks (so per-tick state and the "
        "rolling window are exercised, including the cold-start first tick), on an "
        "isolated deep copy of the data. A case **passes** only if every tick upholds all "
        "four invariants below. Otherwise it is flagged so the bug is caught here instead "
        "of in production.",
        "",
        "## Invariants checked on every tick",
        "",
        "| failure kind | the rule compute(ctx) must obey |",
        "|--------------|----------------------------------|",
        *[f"| `{inv['kind']}` | {inv['rule']} |" for inv in INVARIANTS],
        "",
        "> **`mutated_input`** is the subtle one: in the live loop the *same* "
        "`ChainSnapshot` object is handed to every active plot, so if one plot writes to "
        "`ctx.chain` data in place, every other plot — and every later tick reusing the "
        "buffer — sees corrupted data. The harness runs each tick on a deep copy and "
        "compares array contents before/after to catch this at validation time.",
        "",
        "## Full lifecycle of a plot",
        "",
        "```",
        "  user writes compute(ctx) code string",
        "        │",
        "        ▼",
        "  PlotRegistry.create_plot(plot_id, code)",
        "        │",
        "        ▼",
        "  validate_plot_code()  ── 6 stages ──────────────────────────────",
        "     1. syntax        ast.parse",
        "     2. safety        block imports / eval / open / getattr …",
        "     3. contract      compute exists, callable, one arg",
        "     4. robustness    run against THIS battery; 4 invariants/tick   ◀── you are here",
        "     5. result_format {value|series} of finite numbers",
        "     6. performance   <50 ms avg over 100 runs",
        "        │  (any stage fails → ValidationError, plot NOT registered)",
        "        ▼",
        "  PlotRuntime created (status='created')",
        "        │",
        "        ▼",
        "  replay_plot()  — recompute from day start (status='backfilling' → 'live')",
        "        │",
        "        ▼",
        "  compute_one_tick() per 1s snapshot (live)  → output_series → WebSocket",
        "```",
        "",
        "## Test cases",
        "",
    ]
    for cat in ["structural", "nan", "zero", "flat", "extreme", "quote", "normal"]:
        group = by_cat.get(cat, [])
        if not group:
            continue
        lines.append(f"### `{cat}` ({len(group)})")
        lines.append("")
        lines.append("| case | options × levels | what it stresses |")
        lines.append("|------|------------------|------------------|")
        for c in group:
            lines.append(f"| `{c.name}` | {c.n_options} × {c.n_levels} | {c.description} |")
        lines.append("")
    return "\n".join(lines)
