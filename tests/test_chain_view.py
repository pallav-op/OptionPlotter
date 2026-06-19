import numpy as np
import pytest

from plot_compute import ChainView, build_sample_snapshot


def test_sum_all():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    total = view.sum(field="volume_since_day_start")
    assert total > 0


def test_sum_option_type_filter():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    call_sum = view.sum(field="volume_since_day_start", option_type="CALL")
    put_sum = view.sum(field="volume_since_day_start", option_type="PUT")
    total = view.sum(field="volume_since_day_start")
    assert abs(call_sum + put_sum - total) < 1e-9


def test_mean_returns_float():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    result = view.mean(field="iv", option_type="CALL", has_quote=True)
    assert isinstance(result, float)


def test_count_filter():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    # sample has 2 calls + 2 puts; one put has no quote
    call_count = view.count(option_type="CALL")
    assert call_count == 2
    put_count = view.count(option_type="PUT")
    assert put_count == 2
    quoted = view.count(has_quote=True)
    assert quoted == 3


def test_book_level_field_l1():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    bid_qty_l1 = view.sum(field="bid_quantity_l1")
    assert bid_qty_l1 > 0


def test_book_level_field_l2():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    bid_qty_l2 = view.sum(field="bid_quantity_l2")
    assert bid_qty_l2 >= 0


def test_book_level_out_of_range():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    # Level 99 doesn't exist; should return 0 for quantity
    result = view.sum(field="bid_quantity_l99")
    assert result == 0.0


def test_unknown_filter_raises():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    with pytest.raises(ValueError, match="Unknown filter"):
        view.sum(field="iv", nonexistent_filter=True)


def test_unknown_field_raises():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    with pytest.raises((ValueError, AttributeError)):
        view.sum(field="nonexistent_field")


def test_weighted_mean():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    result = view.weighted_mean(field="iv", weight_field="volume_since_day_start")
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_min_max_strike_filter():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    # call strikes are 100 and 105
    count = view.count(option_type="CALL", min_strike=102.0, max_strike=110.0)
    assert count == 1


def test_delta_filter():
    snap = build_sample_snapshot()
    view = ChainView(snap)
    count = view.count(option_type="CALL", min_delta=0.5, max_delta=1.0)
    assert count == 1
