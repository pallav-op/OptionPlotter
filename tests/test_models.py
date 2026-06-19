import numpy as np
import pytest

from plot_compute import ChainSnapshot


def make_snapshot(n=4, levels=2):
    zeros = np.zeros(n)
    return ChainSnapshot(
        seq_no=1,
        key="TEST",
        timestamp_ns=1_000_000_000,
        timestamp_ms=1_000,
        option_type=np.array(["CALL", "CALL", "PUT", "PUT"], dtype=object)[:n],
        strike_px=zeros.copy(),
        expiry_timestamp=np.zeros(n, dtype=np.uint64),
        mid_px=zeros.copy(),
        last_traded_price=zeros.copy(),
        total_traded_quantity=np.zeros(n, dtype=np.int64),
        total_traded_value=zeros.copy(),
        volume_since_day_start=np.zeros(n, dtype=np.int64),
        moneyness=np.zeros(n, dtype=np.int32),
        has_quote=np.ones(n, dtype=bool),
        low_since_day_start=zeros.copy(),
        high_since_day_start=zeros.copy(),
        book_levels_per_side=np.full(n, levels, dtype=np.uint32),
        underlying_price=zeros.copy(),
        tte=zeros.copy(),
        delta=zeros.copy(),
        gamma=zeros.copy(),
        vega=zeros.copy(),
        theta=zeros.copy(),
        iv=zeros.copy(),
        rate_of_interest=zeros.copy(),
        bid_price=np.full((n, levels), np.nan),
        bid_quantity=np.zeros((n, levels), dtype=np.int64),
        bid_order_count=np.zeros((n, levels), dtype=np.int64),
        ask_price=np.full((n, levels), np.nan),
        ask_quantity=np.zeros((n, levels), dtype=np.int64),
        ask_order_count=np.zeros((n, levels), dtype=np.int64),
    )


def test_frozen():
    snap = make_snapshot()
    with pytest.raises((AttributeError, TypeError)):
        snap.seq_no = 999  # type: ignore[misc]


def test_shapes():
    snap = make_snapshot(n=4, levels=3)
    assert snap.bid_price.shape == (4, 3)
    assert snap.option_type.shape == (4,)


def test_timestamp_ms():
    snap = make_snapshot()
    assert snap.timestamp_ms == 1_000
