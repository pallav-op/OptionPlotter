import pytest

from plot_compute import ValidationError, validate_plot_code, validate_result


# ---------------------------------------------------------------------------
# Stage 1 — Syntax
# ---------------------------------------------------------------------------

def test_syntax_error():
    bad_code = "def compute(ctx):\n    return {"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(bad_code)
    assert exc_info.value.stage == "syntax"


# ---------------------------------------------------------------------------
# Stage 2 — Safety
# ---------------------------------------------------------------------------

def test_import_blocked():
    code = "import os\ndef compute(ctx): return {'value': 1.0}"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(code)
    assert exc_info.value.stage == "safety"


def test_from_import_blocked():
    code = "from os import path\ndef compute(ctx): return {'value': 1.0}"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(code)
    assert exc_info.value.stage == "safety"


def test_eval_blocked():
    code = "def compute(ctx):\n    eval('1+1')\n    return {'value': 1.0}"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(code)
    assert exc_info.value.stage == "safety"


def test_open_blocked():
    code = "def compute(ctx):\n    open('file.txt')\n    return {'value': 1.0}"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(code)
    assert exc_info.value.stage == "safety"


# ---------------------------------------------------------------------------
# Stage 3 — Contract
# ---------------------------------------------------------------------------

def test_missing_compute_function():
    code = "def helper(): pass"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(code)
    assert exc_info.value.stage == "contract"


def test_wrong_arg_count():
    code = "def compute(ctx, extra): return {'value': 1.0}"
    with pytest.raises(ValidationError) as exc_info:
        validate_plot_code(code)
    assert exc_info.value.stage == "contract"


# ---------------------------------------------------------------------------
# Stage 5 — Result format
# ---------------------------------------------------------------------------

def test_validate_result_value_ok():
    out = validate_result({"value": 1.5})
    assert out == {"value": 1.5}


def test_validate_result_series_ok():
    out = validate_result({"series": {"a": 1.0, "b": 2.0}})
    assert out == {"a": 1.0, "b": 2.0}


def test_validate_result_none_rejected():
    with pytest.raises(ValidationError):
        validate_result(None)


def test_validate_result_nan_rejected():
    import math
    with pytest.raises(ValidationError):
        validate_result({"value": math.nan})


def test_validate_result_inf_rejected():
    import math
    with pytest.raises(ValidationError):
        validate_result({"value": math.inf})


def test_validate_result_string_rejected():
    with pytest.raises(ValidationError):
        validate_result({"value": "hello"})


def test_validate_result_missing_key_rejected():
    with pytest.raises(ValidationError):
        validate_result({"foo": 1.0})


# ---------------------------------------------------------------------------
# Passing code
# ---------------------------------------------------------------------------

def test_valid_single_value():
    code = """
def compute(ctx):
    call_qty = ctx.chain.sum(option_type="CALL", field="bid_quantity_l1")
    put_qty = ctx.chain.sum(option_type="PUT", field="bid_quantity_l1")
    return {"value": safe_div(call_qty - put_qty, call_qty + put_qty)}
"""
    fn = validate_plot_code(code)
    assert callable(fn)


def test_valid_multi_series():
    code = """
def compute(ctx):
    return {
        "series": {
            "call_qty": ctx.chain.sum(option_type="CALL", field="bid_quantity_l1"),
            "put_qty": ctx.chain.sum(option_type="PUT", field="bid_quantity_l1"),
        }
    }
"""
    fn = validate_plot_code(code)
    assert callable(fn)


def test_valid_stateful():
    code = """
def compute(ctx):
    current = ctx.chain.sum(field="volume_since_day_start")
    ctx.state["cum"] = ctx.state.get("cum", 0.0) + current
    return {"value": ctx.state["cum"]}
"""
    fn = validate_plot_code(code)
    assert callable(fn)
