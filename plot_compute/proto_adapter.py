from __future__ import annotations

import numpy as np

from .models import ChainSnapshot


def _get(obj, attr, default=None):
    """Duck-typed attribute access: works on proto objects and plain dicts."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _get_list(obj, attr):
    val = _get(obj, attr, [])
    return val if val is not None else []


def build_chain_snapshot(proto_msg) -> ChainSnapshot:
    """Convert a BookPublisherOptionChainDataMessage (or duck-typed equivalent) to ChainSnapshot.

    Accepts real protobuf objects or plain dicts/SimpleNamespace for testing.
    """
    seq_no = int(_get(proto_msg, "seq_no", 0))
    key = str(_get(proto_msg, "key", ""))
    timestamp_ns = int(_get(proto_msg, "timestamp_ns", 0))
    timestamp_ms = timestamp_ns // 1_000_000

    call_options = list(_get_list(proto_msg, "call_options"))
    put_options = list(_get_list(proto_msg, "put_options"))

    # Interleave: all calls first, then all puts (preserves deterministic ordering)
    options = [(opt, "CALL") for opt in call_options] + [(opt, "PUT") for opt in put_options]
    n = len(options)

    if n == 0:
        max_levels = 0
    else:
        max_levels = max(
            int(_get(opt, "book_levels_per_side", 0)) for opt, _ in options
        )
        max_levels = max(max_levels, 1)

    # Scalar arrays
    option_type = np.empty(n, dtype=object)
    strike_px = np.zeros(n, dtype=np.float64)
    expiry_timestamp = np.zeros(n, dtype=np.uint64)
    mid_px = np.zeros(n, dtype=np.float64)
    last_traded_price = np.zeros(n, dtype=np.float64)
    total_traded_quantity = np.zeros(n, dtype=np.int64)
    total_traded_value = np.zeros(n, dtype=np.float64)
    volume_since_day_start = np.zeros(n, dtype=np.int64)
    moneyness = np.zeros(n, dtype=np.int32)
    has_quote = np.zeros(n, dtype=bool)
    low_since_day_start = np.zeros(n, dtype=np.float64)
    high_since_day_start = np.zeros(n, dtype=np.float64)
    book_levels_per_side = np.zeros(n, dtype=np.uint32)
    underlying_price = np.zeros(n, dtype=np.float64)
    tte = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    gamma = np.zeros(n, dtype=np.float64)
    vega = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    iv = np.zeros(n, dtype=np.float64)
    rate_of_interest = np.zeros(n, dtype=np.float64)

    # 2-D book arrays
    bid_price = np.full((n, max_levels), np.nan, dtype=np.float64)
    bid_quantity = np.zeros((n, max_levels), dtype=np.int64)
    bid_order_count = np.zeros((n, max_levels), dtype=np.int64)
    ask_price = np.full((n, max_levels), np.nan, dtype=np.float64)
    ask_quantity = np.zeros((n, max_levels), dtype=np.int64)
    ask_order_count = np.zeros((n, max_levels), dtype=np.int64)

    for i, (opt, otype) in enumerate(options):
        option_type[i] = otype
        strike_px[i] = float(_get(opt, "strike_px", 0.0))
        expiry_timestamp[i] = int(_get(opt, "expiry_timestamp", 0))
        mid_px[i] = float(_get(opt, "mid_px", 0.0))
        last_traded_price[i] = float(_get(opt, "last_traded_price", 0.0))
        total_traded_quantity[i] = int(_get(opt, "total_traded_quantity", 0))
        total_traded_value[i] = float(_get(opt, "total_traded_value", 0.0))
        volume_since_day_start[i] = int(_get(opt, "volume_since_day_start", 0))
        moneyness[i] = int(_get(opt, "moneyness", 0))
        has_quote[i] = bool(_get(opt, "has_quote", False))
        low_since_day_start[i] = float(_get(opt, "low_since_day_start", 0.0))
        high_since_day_start[i] = float(_get(opt, "high_since_day_start", 0.0))
        book_levels_per_side[i] = int(_get(opt, "book_levels_per_side", 0))
        underlying_price[i] = float(_get(opt, "underlying_price", 0.0))
        tte[i] = float(_get(opt, "tte", 0.0))
        delta[i] = float(_get(opt, "delta", 0.0))
        gamma[i] = float(_get(opt, "gamma", 0.0))
        vega[i] = float(_get(opt, "vega", 0.0))
        theta[i] = float(_get(opt, "theta", 0.0))
        iv[i] = float(_get(opt, "iv", 0.0))
        rate_of_interest[i] = float(_get(opt, "rate_of_interest", 0.0))

        for j, bid in enumerate(_get_list(opt, "bids")):
            if j >= max_levels:
                break
            bid_price[i, j] = float(_get(bid, "price", np.nan))
            bid_quantity[i, j] = int(_get(bid, "quantity", 0))
            bid_order_count[i, j] = int(_get(bid, "order_count", 0))

        for j, ask in enumerate(_get_list(opt, "asks")):
            if j >= max_levels:
                break
            ask_price[i, j] = float(_get(ask, "price", np.nan))
            ask_quantity[i, j] = int(_get(ask, "quantity", 0))
            ask_order_count[i, j] = int(_get(ask, "order_count", 0))

    return ChainSnapshot(
        seq_no=seq_no,
        key=key,
        timestamp_ns=timestamp_ns,
        timestamp_ms=timestamp_ms,
        option_type=option_type,
        strike_px=strike_px,
        expiry_timestamp=expiry_timestamp,
        mid_px=mid_px,
        last_traded_price=last_traded_price,
        total_traded_quantity=total_traded_quantity,
        total_traded_value=total_traded_value,
        volume_since_day_start=volume_since_day_start,
        moneyness=moneyness,
        has_quote=has_quote,
        low_since_day_start=low_since_day_start,
        high_since_day_start=high_since_day_start,
        book_levels_per_side=book_levels_per_side,
        underlying_price=underlying_price,
        tte=tte,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        iv=iv,
        rate_of_interest=rate_of_interest,
        bid_price=bid_price,
        bid_quantity=bid_quantity,
        bid_order_count=bid_order_count,
        ask_price=ask_price,
        ask_quantity=ask_quantity,
        ask_order_count=ask_order_count,
    )
