import math


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or math.isnan(b) or math.isinf(b):
        return default
    result = a / b
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def clip(x: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, x))


def is_nan(x: float) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return False


def nz(x: float, default: float = 0.0) -> float:
    try:
        if math.isnan(x):
            return default
    except TypeError:
        return default
    return x
