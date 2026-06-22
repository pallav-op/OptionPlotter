"""Return-format validation (Stage 5).

Kept in its own module so both ``validation`` and ``robustness`` can import it
without creating an import cycle.
"""
from __future__ import annotations

import math

from .errors import ValidationError


def validate_result(result) -> dict[str, float]:
    """Validate and normalise a compute() return value to ``{series_name: float}``.

    Returns the normalised dict on success; raises ValidationError on failure.

    Accepted forms:
        {"value": <finite number>}
        {"series": {"name": <finite number>, ...}}
    """
    if result is None:
        raise ValidationError("result_format", "compute returned None.")
    if not isinstance(result, dict):
        raise ValidationError(
            "result_format", f"compute must return a dict, got {type(result).__name__}."
        )

    if "value" in result:
        val = result["value"]
        _assert_finite(val, "value")
        return {"value": float(val)}

    if "series" in result:
        series = result["series"]
        if not isinstance(series, dict):
            raise ValidationError("result_format", "'series' must be a dict.")
        if not series:
            raise ValidationError("result_format", "'series' dict must not be empty.")
        out: dict[str, float] = {}
        for name, val in series.items():
            _assert_finite(val, f"series[{name!r}]")
            out[name] = float(val)
        return out

    raise ValidationError(
        "result_format",
        "Return dict must contain either 'value' or 'series' key.",
    )


def _assert_finite(val, label: str) -> None:
    if val is None:
        raise ValidationError("result_format", f"{label} is None.")
    if isinstance(val, bool):
        # bool is a subclass of int; treat as a type error to catch accidental returns
        raise ValidationError("result_format", f"{label} is a bool, expected a number.")
    try:
        fval = float(val)
    except (TypeError, ValueError):
        raise ValidationError("result_format", f"{label} is not a number: {val!r}.")
    if math.isnan(fval):
        raise ValidationError("result_format", f"{label} must be a finite number, got NaN.")
    if math.isinf(fval):
        raise ValidationError("result_format", f"{label} must be a finite number, got {fval}.")
