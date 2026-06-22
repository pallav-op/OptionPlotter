from __future__ import annotations

import re

import numpy as np

from .models import ChainSnapshot

# Matches e.g. "bid_price_l1", "ask_quantity_l3"
_BOOK_FIELD_RE = re.compile(r"^(bid|ask)_(price|quantity|order_count)_l(\d+)$")

_BOOK_ARRAY_MAP = {
    "bid_price": "bid_price",
    "bid_quantity": "bid_quantity",
    "bid_order_count": "bid_order_count",
    "ask_price": "ask_price",
    "ask_quantity": "ask_quantity",
    "ask_order_count": "ask_order_count",
}


class ChainView:
    """Columnar view over a ChainSnapshot exposing safe aggregation methods."""

    def __init__(self, snapshot: ChainSnapshot) -> None:
        self._snap = snapshot

    # ------------------------------------------------------------------
    # Public aggregation methods
    # ------------------------------------------------------------------

    def sum(self, field: str, **filters) -> float:
        arr = self._resolve_field(field)
        mask = self._build_mask(**filters)
        return float(np.nansum(arr[mask]))

    def mean(self, field: str, **filters) -> float:
        sliced = self._resolve_field(field)[self._build_mask(**filters)]
        if sliced.size == 0 or np.all(np.isnan(sliced)):
            return float("nan")
        return float(np.nanmean(sliced))

    def min(self, field: str, **filters) -> float:
        sliced = self._resolve_field(field)[self._build_mask(**filters)]
        if sliced.size == 0 or np.all(np.isnan(sliced)):
            return float("nan")
        return float(np.nanmin(sliced))

    def max(self, field: str, **filters) -> float:
        sliced = self._resolve_field(field)[self._build_mask(**filters)]
        if sliced.size == 0 or np.all(np.isnan(sliced)):
            return float("nan")
        return float(np.nanmax(sliced))

    def count(self, **filters) -> int:
        mask = self._build_mask(**filters)
        return int(np.sum(mask))

    def weighted_mean(self, field: str, weight_field: str, **filters) -> float:
        arr = self._resolve_field(field)
        weights = self._resolve_field(weight_field)
        mask = self._build_mask(**filters)
        a = arr[mask].astype(float)
        w = weights[mask].astype(float)
        valid = ~np.isnan(a) & ~np.isnan(w)
        if not np.any(valid):
            return float("nan")
        total_w = np.sum(w[valid])
        if total_w == 0:
            return float("nan")
        return float(np.sum(a[valid] * w[valid]) / total_w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_field(self, field: str) -> np.ndarray:
        m = _BOOK_FIELD_RE.match(field)
        if m:
            side = m.group(1)       # bid / ask
            kind = m.group(2)       # price / quantity / order_count
            level = int(m.group(3)) - 1  # 1-indexed → 0-indexed
            arr_name = f"{side}_{kind}"
            book_arr = getattr(self._snap, arr_name)
            if level >= book_arr.shape[1]:
                return np.full(book_arr.shape[0], np.nan if kind == "price" else 0, dtype=float)
            return book_arr[:, level].astype(float)

        arr = getattr(self._snap, field, None)
        if arr is None:
            raise ValueError(f"Unknown field: {field!r}")
        return np.asarray(arr, dtype=float)

    def _build_mask(self, **filters) -> np.ndarray:
        n = len(self._snap.option_type)
        mask = np.ones(n, dtype=bool)

        for key, val in filters.items():
            if val is None:
                continue

            if key == "option_type":
                mask &= self._snap.option_type == val

            elif key == "expiry_timestamp":
                mask &= self._snap.expiry_timestamp == int(val)

            elif key == "min_strike":
                mask &= self._snap.strike_px >= float(val)

            elif key == "max_strike":
                mask &= self._snap.strike_px <= float(val)

            elif key == "min_delta":
                mask &= self._snap.delta >= float(val)

            elif key == "max_delta":
                mask &= self._snap.delta <= float(val)

            elif key == "min_moneyness":
                mask &= self._snap.moneyness >= int(val)

            elif key == "max_moneyness":
                mask &= self._snap.moneyness <= int(val)

            elif key == "has_quote":
                mask &= self._snap.has_quote == bool(val)

            else:
                raise ValueError(f"Unknown filter: {key!r}")

        return mask
