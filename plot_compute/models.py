from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ChainSnapshot:
    seq_no: int
    key: str
    timestamp_ns: int
    timestamp_ms: int

    # 1-D per-option arrays
    option_type: np.ndarray          # dtype=object, values "CALL"/"PUT"
    strike_px: np.ndarray            # float64
    expiry_timestamp: np.ndarray     # uint64
    mid_px: np.ndarray               # float64
    last_traded_price: np.ndarray    # float64
    total_traded_quantity: np.ndarray  # int64
    total_traded_value: np.ndarray   # float64
    volume_since_day_start: np.ndarray  # int64
    moneyness: np.ndarray            # int32
    has_quote: np.ndarray            # bool
    low_since_day_start: np.ndarray  # float64
    high_since_day_start: np.ndarray  # float64
    book_levels_per_side: np.ndarray  # uint32
    underlying_price: np.ndarray     # float64
    tte: np.ndarray                  # float64
    delta: np.ndarray                # float64
    gamma: np.ndarray                # float64
    vega: np.ndarray                 # float64
    theta: np.ndarray                # float64
    iv: np.ndarray                   # float64
    rate_of_interest: np.ndarray     # float64

    # 2-D book arrays — shape [num_options, max_book_levels]
    bid_price: np.ndarray
    bid_quantity: np.ndarray
    bid_order_count: np.ndarray
    ask_price: np.ndarray
    ask_quantity: np.ndarray
    ask_order_count: np.ndarray


@dataclass
class PlotRuntime:
    plot_id: str
    version: int
    code: str
    compute_fn: Callable
    status: str                                   # created/validating/backfilling/live/paused/recomputing/error/deleted
    state: dict[str, Any] = field(default_factory=dict)
    output_series: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    last_computed_ts: int | None = None
    error: str | None = None


@dataclass
class PlotResult:
    series: dict[str, float]   # always multi-series form; single-value plots use key "value"
