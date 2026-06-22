from __future__ import annotations

import ast
import inspect
import time
import warnings
from typing import Callable

from .context import IndicatorContext, build_context
from .errors import ValidationError
from .executor import build_user_namespace, compile_user_code
from .models import ChainSnapshot, PlotRuntime
from .result_format import validate_result  # re-exported for backwards compatibility
from .robustness import RobustnessReport, run_robustness_suite
from .rolling import RollingWindowStore

# ---------------------------------------------------------------------------
# Unsafe AST visitor
# ---------------------------------------------------------------------------

_UNSAFE_CALL_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
})


class _UnsafeCodeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(alias.name for alias in node.names)
        self.violations.append(f"import not allowed: {names} (line {node.lineno})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.violations.append(
            f"from-import not allowed: from {node.module} import ... (line {node.lineno})"
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in _UNSAFE_CALL_NAMES:
            self.violations.append(f"unsafe call to {name!r} (line {node.lineno})")
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Stage 1 — Syntax
# ---------------------------------------------------------------------------

def validate_syntax(code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValidationError("syntax", f"SyntaxError: {exc}") from exc


# ---------------------------------------------------------------------------
# Stage 2 — Safety
# ---------------------------------------------------------------------------

def validate_ast_safety(code: str) -> None:
    tree = ast.parse(code)
    visitor = _UnsafeCodeVisitor()
    visitor.visit(tree)
    if visitor.violations:
        msg = "; ".join(visitor.violations)
        raise ValidationError("safety", msg)


# ---------------------------------------------------------------------------
# Stage 3 — Function contract
# ---------------------------------------------------------------------------

def validate_compute_signature(code: str) -> Callable:
    code_obj = compile_user_code(code)
    ns = build_user_namespace(code_obj)

    compute = ns.get("compute")
    if compute is None or not callable(compute):
        raise ValidationError("contract", "A callable named 'compute' must be defined.")

    try:
        sig = inspect.signature(compute)
    except (ValueError, TypeError) as exc:
        raise ValidationError("contract", f"Cannot inspect compute signature: {exc}") from exc

    params = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(params) != 1:
        raise ValidationError(
            "contract",
            f"compute must accept exactly one positional argument (ctx). "
            f"Got {len(params)} required positional parameter(s).",
        )

    return compute


# ---------------------------------------------------------------------------
# Sample context for smoke / performance tests
# ---------------------------------------------------------------------------

def build_sample_snapshot() -> ChainSnapshot:
    """Synthetic snapshot: 2 calls + 2 puts, 2 book levels, various edge cases."""
    from .proto_adapter import build_chain_snapshot

    class _Lvl:
        def __init__(self, price, qty, oc):
            self.price = price
            self.quantity = qty
            self.order_count = oc

    class _Opt:
        def __init__(self, strike, otype_ignored, delta, iv, moneyness, has_q, bids, asks):
            self.strike_px = strike
            self.option_type = otype_ignored
            self.expiry_timestamp = 1_700_000_000
            self.mid_px = strike * 0.01
            self.last_traded_price = strike * 0.009
            self.total_traded_quantity = 100
            self.total_traded_value = 10000.0
            self.volume_since_day_start = 500
            self.moneyness = moneyness
            self.has_quote = has_q
            self.low_since_day_start = strike * 0.005
            self.high_since_day_start = strike * 0.015
            self.book_levels_per_side = 2
            self.underlying_price = 100.0
            self.tte = 0.25
            self.delta = delta
            self.gamma = 0.02
            self.vega = 0.15
            self.theta = -0.05
            self.iv = iv
            self.rate_of_interest = 0.05
            self.bids = bids
            self.asks = asks

    class _Msg:
        def __init__(self):
            self.seq_no = 1
            self.key = "SAMPLE"
            self.timestamp_ns = 1_700_000_000_000_000_000
            self.call_options = [
                _Opt(100, "CALL", 0.55, 0.25, 0, True,
                     [_Lvl(99.5, 10, 2), _Lvl(99.0, 5, 1)],
                     [_Lvl(100.5, 8, 2), _Lvl(101.0, 4, 1)]),
                _Opt(105, "CALL", 0.35, 0.28, -1, True,
                     [_Lvl(104.5, 7, 1), _Lvl(104.0, 3, 1)],
                     [_Lvl(105.5, 6, 1), _Lvl(106.0, 2, 1)]),
            ]
            self.put_options = [
                _Opt(100, "PUT", -0.45, 0.25, 0, True,
                     [_Lvl(99.5, 12, 3), _Lvl(99.0, 6, 2)],
                     [_Lvl(100.5, 9, 2), _Lvl(101.0, 5, 1)]),
                _Opt(95, "PUT", -0.25, 0.30, 1, False,  # no quote
                     [_Lvl(94.5, 0, 0)],
                     [_Lvl(95.5, 0, 0)]),
            ]

    return build_chain_snapshot(_Msg())


def build_sample_context() -> IndicatorContext:
    snap = build_sample_snapshot()
    rolling = RollingWindowStore()
    dummy_plot = PlotRuntime(
        plot_id="__sample__",
        version=0,
        code="",
        compute_fn=lambda ctx: None,
        status="validating",
    )
    return build_context(dummy_plot, snap, rolling)


# ---------------------------------------------------------------------------
# Stage 4 — Robustness (adversarial edge-case battery)
# ---------------------------------------------------------------------------

def run_robustness_validation(
    compute_fn: Callable,
    *,
    strict: bool = True,
    repeat: int = 3,
) -> RobustnessReport:
    """Run the adversarial edge-case battery (NaN / zero / flat / extreme / …).

    Returns the :class:`RobustnessReport`. If ``strict`` and any case fails, raises
    ``ValidationError("robustness", <formatted report>)`` so the plot is rejected
    with a full pass/fail breakdown attached.
    """
    report = run_robustness_suite(compute_fn, repeat=repeat)
    if strict and not report.ok():
        raise ValidationError("robustness", report.format(show_passes=False))
    return report


# ---------------------------------------------------------------------------
# Stage 5 — Return format validation lives in result_format.validate_result
#           (imported above and re-exported here for backwards compatibility)
# ---------------------------------------------------------------------------
# Stage 6 — Performance smoke test
# ---------------------------------------------------------------------------

_WARN_MS = 5.0
_REJECT_MS = 50.0
_PERF_ITERS = 100


def run_performance_test(compute_fn: Callable) -> None:
    ctx = build_sample_context()
    times = []
    for _ in range(_PERF_ITERS):
        t0 = time.perf_counter()
        try:
            compute_fn(ctx)
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    if avg_ms > _REJECT_MS:
        raise ValidationError(
            "performance",
            f"compute averaged {avg_ms:.1f} ms over {_PERF_ITERS} runs "
            f"(limit: {_REJECT_MS} ms).",
        )
    if avg_ms > _WARN_MS:
        warnings.warn(
            f"[plot_compute] compute is slow: {avg_ms:.1f} ms avg "
            f"(warn threshold: {_WARN_MS} ms)",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def validate_plot_code(
    code: str,
    *,
    strict_robustness: bool = True,
    robustness_repeat: int = 3,
) -> tuple[Callable, RobustnessReport]:
    """Run all 6 validation stages before a plot is accepted.

    Stages: syntax → safety → contract → robustness → (result_format, applied
    per-tick inside robustness) → performance.

    Returns ``(compute_fn, robustness_report)`` on success. Raises ``ValidationError``
    (with ``.stage`` set) if any stage fails. When ``strict_robustness`` is True
    (the default) a plot is rejected if any adversarial case fails; set it False to
    downgrade robustness failures to an inspectable report without blocking.
    """
    validate_syntax(code)
    validate_ast_safety(code)
    compute_fn = validate_compute_signature(code)
    report = run_robustness_validation(compute_fn, strict=strict_robustness, repeat=robustness_repeat)
    run_performance_test(compute_fn)
    return compute_fn, report
