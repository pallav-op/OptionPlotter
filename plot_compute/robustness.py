"""Robustness harness — run user ``compute(ctx)`` against the adversarial battery.

This is the engine behind validation Stage 4. It executes the user function against
every :class:`EdgeCase` for several consecutive ticks and records, per case, whether
the code upheld every safety invariant. The resulting :class:`RobustnessReport` is
both machine-readable (per-case structured results) and printable (a formatted
pass/fail summary with the reason for every failure).

A case PASSES only if, on every tick, the user code upholds all four invariants:

    1. no exception        — compute() must not raise                  (EXCEPTION)
    2. no input mutation   — compute() must not modify the snapshot    (MUTATED_INPUT)
    3. valid return shape  — {"value": ...} or {"series": {...}}       (BAD_FORMAT)
    4. finite values       — every returned number is finite           (NON_FINITE)

Invariant 2 matters because in the live loop the *same* ChainSnapshot object is
handed to every active plot in a tick (see runtime.compute_one_tick). If one plot
mutates the snapshot's arrays in place, every other plot — and any later tick that
reuses the buffer — sees corrupted data. The harness isolates each tick on a deep
copy so a mutating function cannot poison the shared battery, and compares the
arrays before/after to flag the bug at validation time, before the plot is ever
registered.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, replace
from typing import Callable

import numpy as np

from .edge_cases import EdgeCase, build_edge_cases
from .errors import ValidationError
from .models import ChainSnapshot, PlotRuntime
from .result_format import validate_result
from .rolling import RollingWindowStore

# Each case is run for this many consecutive ticks (cold-start + warm state/window).
DEFAULT_REPEAT = 3

# Failure taxonomy
EXCEPTION = "exception"          # compute() raised
MUTATED_INPUT = "mutated_input"  # compute() modified the input snapshot in place
NON_FINITE = "non_finite"        # returned NaN / inf
BAD_FORMAT = "bad_format"        # wrong return shape / type / None

# All recognised failure kinds (used by reporting / counting).
FAILURE_KINDS = (EXCEPTION, MUTATED_INPUT, BAD_FORMAT, NON_FINITE)


@dataclass(frozen=True)
class CaseResult:
    name: str
    category: str
    description: str
    passed: bool
    failure_kind: str | None = None   # one of FAILURE_KINDS
    detail: str | None = None         # tick index + reason
    output: dict | None = None        # last valid normalised output, if any


@dataclass
class RobustnessReport:
    results: list[CaseResult] = field(default_factory=list)
    repeat: int = DEFAULT_REPEAT

    # -- aggregate accessors -------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def ok(self) -> bool:
        """True if every case passed."""
        return self.n_failed == 0

    def kind_counts(self) -> dict[str, int]:
        """Failures grouped by kind. Every known kind is present (0 if none)."""
        counts: dict[str, int] = {k: 0 for k in FAILURE_KINDS}
        for r in self.failures:
            counts[r.failure_kind] = counts.get(r.failure_kind, 0) + 1
        return counts

    def category_scores(self) -> dict[str, tuple[int, int]]:
        """Map category -> (passed, total)."""
        scores: dict[str, list[int]] = {}
        for r in self.results:
            s = scores.setdefault(r.category, [0, 0])
            s[1] += 1
            if r.passed:
                s[0] += 1
        return {k: (v[0], v[1]) for k, v in scores.items()}

    # -- presentation --------------------------------------------------------
    def format(self, *, show_passes: bool = True) -> str:
        if not self.results:
            return "ROBUSTNESS REPORT — no cases run."

        name_w = max(len(r.name) for r in self.results)
        cat_w = max(len(r.category) for r in self.results)
        header_state = "PASSED" if self.ok() else "FAILED"

        lines: list[str] = []
        bar = "─" * 78
        lines.append(bar)
        lines.append(
            f"ROBUSTNESS REPORT  [{header_state}]   "
            f"{self.n_passed}/{self.total} cases passed   "
            f"({self.repeat} ticks/case)"
        )
        lines.append("invariants/tick: no exception · no input mutation · valid format · finite")
        lines.append(bar)

        for r in self.results:
            if r.passed and not show_passes:
                continue
            tag = "PASS" if r.passed else "FAIL"
            lines.append(
                f"{tag}  [{r.category:<{cat_w}}]  {r.name:<{name_w}}  {r.description}"
            )
            if not r.passed:
                lines.append(f"        └─ {r.failure_kind}: {r.detail}")

        # Footer summaries
        lines.append(bar)
        cat_summary = "  ".join(
            f"{cat}={p}/{t}" for cat, (p, t) in sorted(self.category_scores().items())
        )
        lines.append(f"by category : {cat_summary}")
        if self.failures:
            # show every known kind so the breakdown is stable and complete
            counts = self.kind_counts()
            kind_summary = "  ".join(f"{k}={counts[k]}" for k in FAILURE_KINDS)
            lines.append(f"by failure  : {kind_summary}")
            lines.append(
                f"\n{self.n_failed} case(s) failed. Make compute() defensive "
                "(safe_div / nz / clip / guard empty selections) and never mutate "
                "ctx.chain data; then re-validate."
            )
        else:
            lines.append("All adversarial cases survived. compute() is production-robust. ✓")
        lines.append(bar)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.format()


# ---------------------------------------------------------------------------
# Snapshot isolation + mutation detection
# ---------------------------------------------------------------------------

def _ndarray_field_names(snap: ChainSnapshot) -> list[str]:
    return [
        f.name for f in dataclass_fields(snap)
        if isinstance(getattr(snap, f.name), np.ndarray)
    ]

def _clone_snapshot(snap: ChainSnapshot, ts_ms: int) -> ChainSnapshot:
    """Deep copy of a snapshot at a new timestamp.

    Every ndarray field is copied so that user code mutating the working snapshot
    cannot corrupt the shared battery (or the next tick, which would otherwise reuse
    the same array buffers via ``dataclasses.replace``).
    """
    arrays = {name: getattr(snap, name).copy() for name in _ndarray_field_names(snap)}
    return replace(snap, timestamp_ms=ts_ms, timestamp_ns=ts_ms * 1_000_000, **arrays)


def _fingerprint(snap: ChainSnapshot, names: list[str]) -> dict[str, np.ndarray]:
    return {name: getattr(snap, name).copy() for name in names}


def _detect_mutation(snap: ChainSnapshot, before: dict[str, np.ndarray]) -> str | None:
    """Return the name of the first field whose contents changed, or None."""
    for name, arr0 in before.items():
        cur = getattr(snap, name)
        if cur.shape != arr0.shape:
            return name
        if arr0.dtype.kind == "f":
            if not np.array_equal(cur, arr0, equal_nan=True):
                return name
        elif not np.array_equal(cur, arr0):
            return name
    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _short_exc(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()
    head = msg[0] if msg else ""
    return f"{type(exc).__name__}: {head}" if head else type(exc).__name__


def _run_case(compute_fn: Callable, case: EdgeCase, repeat: int) -> CaseResult:
    """Run a single edge case for ``repeat`` ticks with fresh state + rolling window."""
    state: dict = {}
    rolling = RollingWindowStore()
    plot = PlotRuntime(
        plot_id="__robust__",
        version=0,
        code="",
        compute_fn=compute_fn,
        status="validating",
        state=state,
    )
    array_names = _ndarray_field_names(case.snapshot)

    def fail(kind: str, tick: int, detail: str, last_output) -> CaseResult:
        return CaseResult(
            name=case.name, category=case.category, description=case.description,
            passed=False, failure_kind=kind, detail=f"tick {tick}: {detail}",
            output=last_output,
        )

    last_output: dict | None = None
    for tick in range(repeat):
        ts_ms = case.snapshot.timestamp_ms + tick * 1000
        # Work on an isolated deep copy so user mutation cannot poison the battery.
        snap = _clone_snapshot(case.snapshot, ts_ms)
        before = _fingerprint(snap, array_names)

        from .context import build_context  # local import keeps the harness self-contained
        ctx = build_context(plot, snap, rolling)

        try:
            raw = compute_fn(ctx)
        except Exception as exc:  # noqa: BLE001 - catch everything user code throws
            return fail(EXCEPTION, tick, _short_exc(exc), last_output)

        # Invariant 2: the input snapshot must be untouched.
        mutated_field = _detect_mutation(snap, before)
        if mutated_field is not None:
            return fail(
                MUTATED_INPUT, tick,
                f"modified input in place (snapshot.{mutated_field}). "
                "The same snapshot is shared across all plots — never write to ctx.chain data.",
                last_output,
            )

        # Invariants 3 & 4: valid, finite return.
        try:
            normalised = validate_result(raw)
        except ValidationError as ve:
            kind = NON_FINITE if "finite" in ve.message.lower() else BAD_FORMAT
            return fail(kind, tick, ve.message, last_output)

        # Feed the rolling store so window queries on later ticks have data.
        for series_name, value in normalised.items():
            rolling.push(series_name, ts_ms, value)
        last_output = normalised

    return CaseResult(
        name=case.name, category=case.category, description=case.description,
        passed=True, output=last_output,
    )


def run_robustness_suite(
    compute_fn: Callable,
    *,
    cases: list[EdgeCase] | None = None,
    repeat: int = DEFAULT_REPEAT,
) -> RobustnessReport:
    """Run ``compute_fn`` against every edge case. Never raises — collects results.

    Parameters
    ----------
    compute_fn : the user's compiled ``compute(ctx)`` callable.
    cases      : override the battery (defaults to :func:`build_edge_cases`).
    repeat     : consecutive ticks per case (cold-start + warm state/window).
    """
    if cases is None:
        cases = build_edge_cases()
    results = [_run_case(compute_fn, c, repeat) for c in cases]
    return RobustnessReport(results=results, repeat=repeat)
