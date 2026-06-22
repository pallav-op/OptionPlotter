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

    # Split the (option, type) pairs once; build each column in a single pass.
    opts = [opt for opt, _ in options]
    otypes = [otype for _, otype in options]

    def _col(field, default, dtype):
        # np.fromiter builds the column directly from the generator — faster and
        # cleaner than n individual indexed assignments into a pre-allocated array.
        return np.fromiter(
            (_get(o, field, default) for o in opts), dtype=dtype, count=n
        )

    option_type = np.array(otypes, dtype=object)
    strike_px = _col("strike_px", 0.0, np.float64)
    expiry_timestamp = _col("expiry_timestamp", 0, np.uint64)
    mid_px = _col("mid_px", 0.0, np.float64)
    last_traded_price = _col("last_traded_price", 0.0, np.float64)
    total_traded_quantity = _col("total_traded_quantity", 0, np.int64)
    total_traded_value = _col("total_traded_value", 0.0, np.float64)
    volume_since_day_start = _col("volume_since_day_start", 0, np.int64)
    moneyness = _col("moneyness", 0, np.int32)
    has_quote = _col("has_quote", False, bool)
    low_since_day_start = _col("low_since_day_start", 0.0, np.float64)
    high_since_day_start = _col("high_since_day_start", 0.0, np.float64)
    book_levels_per_side = _col("book_levels_per_side", 0, np.uint32)
    underlying_price = _col("underlying_price", 0.0, np.float64)
    tte = _col("tte", 0.0, np.float64)
    delta = _col("delta", 0.0, np.float64)
    gamma = _col("gamma", 0.0, np.float64)
    vega = _col("vega", 0.0, np.float64)
    theta = _col("theta", 0.0, np.float64)
    iv = _col("iv", 0.0, np.float64)
    rate_of_interest = _col("rate_of_interest", 0.0, np.float64)

    # 2-D book arrays — ragged per-option depth, filled with an indexed loop.
    bid_price = np.full((n, max_levels), np.nan, dtype=np.float64)
    bid_quantity = np.zeros((n, max_levels), dtype=np.int64)
    bid_order_count = np.zeros((n, max_levels), dtype=np.int64)
    ask_price = np.full((n, max_levels), np.nan, dtype=np.float64)
    ask_quantity = np.zeros((n, max_levels), dtype=np.int64)
    ask_order_count = np.zeros((n, max_levels), dtype=np.int64)

    for i, opt in enumerate(opts):
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
