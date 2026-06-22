"""plot_compute — option-chain indicator computation library."""

from .chain_view import ChainView
from .context import IndicatorContext, WindowAPI, build_context
from .edge_cases import EdgeCase, build_edge_cases, load_edge_cases, save_edge_cases
from .errors import RuntimePlotError, ValidationError
from .executor import SAFE_BUILTINS, compile_user_code, execute_compute
from .helpers import clip, is_nan, nz, safe_div
from .models import ChainSnapshot, PlotResult, PlotRuntime
from .proto_adapter import build_chain_snapshot
from .registry import PlotRegistry
from .result_format import validate_result
from .robustness import (
    FAILURE_KINDS,
    MUTATED_INPUT,
    CaseResult,
    RobustnessReport,
    run_robustness_suite,
)
from .rolling import RollingWindowStore
from .runtime import compute_one_tick, replay_plot
from .scheduler import (
    LiveScheduler,
    make_append_points,
    make_batch,
    make_plot_error,
)
from .sources import (
    Clock,
    IterableSnapshotSource,
    LatestSnapshotSource,
    ManualClock,
    RealClock,
    SnapshotSource,
)
from .validation import (
    build_sample_context,
    build_sample_snapshot,
    run_robustness_validation,
    validate_plot_code,
)

__all__ = [
    # Models
    "ChainSnapshot",
    "PlotRuntime",
    "PlotResult",
    # Core API
    "PlotRegistry",
    "build_chain_snapshot",
    "compute_one_tick",
    "replay_plot",
    # Context / view
    "ChainView",
    "IndicatorContext",
    "WindowAPI",
    "build_context",
    "RollingWindowStore",
    # Validation
    "validate_plot_code",
    "validate_result",
    "run_robustness_validation",
    "build_sample_snapshot",
    "build_sample_context",
    # Robustness
    "run_robustness_suite",
    "RobustnessReport",
    "CaseResult",
    "FAILURE_KINDS",
    "MUTATED_INPUT",
    "EdgeCase",
    "build_edge_cases",
    "save_edge_cases",
    "load_edge_cases",
    # Live scheduler
    "LiveScheduler",
    "make_append_points",
    "make_plot_error",
    "make_batch",
    # Sources / clocks (feed seam)
    "SnapshotSource",
    "IterableSnapshotSource",
    "LatestSnapshotSource",
    "Clock",
    "RealClock",
    "ManualClock",
    # Executor
    "compile_user_code",
    "execute_compute",
    "SAFE_BUILTINS",
    # Helpers (also injected into user code namespace)
    "safe_div",
    "clip",
    "is_nan",
    "nz",
    # Errors
    "ValidationError",
    "RuntimePlotError",
]
