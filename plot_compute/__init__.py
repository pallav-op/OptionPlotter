"""plot_compute — option-chain indicator computation library."""

from .chain_view import ChainView
from .context import IndicatorContext, WindowAPI, build_context
from .errors import RuntimePlotError, ValidationError
from .executor import SAFE_BUILTINS, compile_user_code, execute_compute
from .helpers import clip, is_nan, nz, safe_div
from .models import ChainSnapshot, PlotResult, PlotRuntime
from .proto_adapter import build_chain_snapshot
from .registry import PlotRegistry
from .rolling import RollingWindowStore
from .runtime import compute_one_tick, replay_plot
from .validation import (
    build_sample_context,
    build_sample_snapshot,
    validate_plot_code,
    validate_result,
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
    "build_sample_snapshot",
    "build_sample_context",
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
