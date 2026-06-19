from __future__ import annotations

import types
from typing import Callable

from .context import IndicatorContext
from .helpers import clip, is_nan, nz, safe_div

SAFE_BUILTINS: dict = {
    # helpers injected directly so user code needs no import
    "safe_div": safe_div,
    "clip": clip,
    "is_nan": is_nan,
    "nz": nz,
    # safe standard builtins
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "print": print,   # useful for debugging; does not open files
    "True": True,
    "False": False,
    "None": None,
    "__build_class__": __build_class__,
}


def compile_user_code(code: str) -> types.CodeType:
    return compile(code, "<user_plot>", "exec")


def build_user_namespace(code_obj: types.CodeType) -> dict:
    """Execute compiled user code in a restricted namespace and return it."""
    ns: dict = {"__builtins__": SAFE_BUILTINS}
    exec(code_obj, ns)  # noqa: S102
    return ns


def execute_compute(compute_fn: Callable, ctx: IndicatorContext) -> dict:
    return compute_fn(ctx)
