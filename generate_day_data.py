"""
generate_day_data.py

Synthetic option-chain day data generator.
Produces 23,400 proto-like snapshots (1-second, 09:30–16:00 EDT, 2024-03-11).

Chain: key="SPX", 9 strikes × 2 expiries × 2 types = 36 options per snapshot.
Mirrors the protobuf spec from the plot_compute implementation spec exactly.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Proto-mirroring dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BookLevelData:
    price: float
    quantity: int
    order_count: int


@dataclass
class BookPublisherOptionDataMessage:
    strike_px: float
    option_type: str            # "CALL" | "PUT"
    expiry_timestamp: int       # unix epoch seconds
    mid_px: float
    last_traded_price: float
    total_traded_quantity: int
    total_traded_value: float
    bids: List[BookLevelData]
    asks: List[BookLevelData]
    volume_since_day_start: int
    moneyness: int              # integer units of 2.5 pts from ATM (strike-based)
    has_quote: bool
    low_since_day_start: float
    high_since_day_start: float
    book_levels_per_side: int
    underlying_price: float
    tte: float                  # time-to-expiry in years
    delta: float
    gamma: float
    vega: float                 # per 1% vol move
    theta: float                # per calendar day
    iv: float
    rate_of_interest: float


@dataclass
class BookPublisherOptionChainDataMessage:
    seq_no: int
    key: str
    timestamp_ns: int
    call_options: List[BookPublisherOptionDataMessage]
    put_options: List[BookPublisherOptionDataMessage]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Market parameters
# ─────────────────────────────────────────────────────────────────────────────

# 2024-03-11 — DST started 2024-03-10, so EDT = UTC-4
_OPEN  = datetime(2024, 3, 11, 13, 30, 0, tzinfo=timezone.utc)  # 09:30 EDT
_CLOSE = datetime(2024, 3, 11, 20,  0, 0, tzinfo=timezone.utc)  # 16:00 EDT

MARKET_OPEN_S:   int = int(_OPEN.timestamp())
TRADING_SECONDS: int = int((_CLOSE - _OPEN).total_seconds())    # 23,400

CHAIN_KEY    = "SPX"
S0           = 100.0    # starting underlying
ANNUAL_VOL   = 0.20     # base ATM annual vol
RATE         = 0.05     # risk-free rate
BOOK_LEVELS  = 3        # depth levels per side

# 9 strikes centred at 100, spaced 2.5 apart
STRIKES: List[float] = [90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0]

# Two expiries (unix epoch seconds at 16:00 ET on each date)
EXPIRIES: List[int] = [
    int(datetime(2024, 4, 12, 20, 0, 0, tzinfo=timezone.utc).timestamp()),  # ~32 DTE
    int(datetime(2024, 5, 17, 20, 0, 0, tzinfo=timezone.utc).timestamp()),  # ~67 DTE
]

# Per-second dt in trading-year fraction (252 days × 23,400 s/day)
TRADING_YEAR_S: int = 252 * TRADING_SECONDS
DT_YEARS: float = 1.0 / TRADING_YEAR_S


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Black-Scholes price and greeks
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))  # x / sqrt(2)


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) * 0.3989422804014327     # 1 / sqrt(2π)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, opt_type: str
) -> tuple[float, float, float, float, float]:
    """
    Returns (mid_px, delta, gamma, vega_per_pct, theta_per_day).
    Handles expired / zero-vol edge cases gracefully.
    """
    if T < 1e-6 or sigma < 1e-4:
        intrinsic = max(S - K, 0.0) if opt_type == "CALL" else max(K - S, 0.0)
        edge_d = (1.0 if S >= K else 0.0) if opt_type == "CALL" else (-1.0 if S <= K else 0.0)
        return intrinsic, edge_d, 0.0, 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1, nd2 = _ncdf(d1), _ncdf(d2)
    npd1 = _npdf(d1)
    disc = math.exp(-r * T)

    if opt_type == "CALL":
        price        = max(S * nd1 - K * disc * nd2, 0.0)
        delta        = nd1
        theta_rterm  = r * K * disc * nd2
    else:
        price        = max(K * disc * _ncdf(-d2) - S * _ncdf(-d1), 0.0)
        delta        = nd1 - 1.0
        theta_rterm  = -r * K * disc * _ncdf(-d2)

    gamma = npd1 / (S * sigma * sqrt_T)
    vega  = S * npd1 * sqrt_T * 0.01           # per 1% vol move
    theta = (-(S * npd1 * sigma) / (2.0 * sqrt_T) - theta_rterm) / 365.0

    return price, delta, gamma, vega, theta


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Vol smile (equity-index: negative skew + convexity + term structure)
# ─────────────────────────────────────────────────────────────────────────────

def smile_iv(S: float, K: float, T: float, atm_iv: float) -> float:
    """
    Surface parameterisation:
      IV(K) = atm_iv * exp(skew * log(K/S) + convexity * log(K/S)²)
    plus a small term-structure bump for longer expiries.
    """
    log_m = math.log(K / S)
    iv    = atm_iv * math.exp(-0.30 * log_m + 2.50 * log_m ** 2)
    iv   += 0.010 * math.sqrt(T * 252)   # slight upward term-structure
    return max(iv, 0.02)                  # floor at 2%


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Book level builder
# ─────────────────────────────────────────────────────────────────────────────

def make_book(
    mid: float,
    spread_pct: float,
    rng: np.random.Generator,
    levels: int = BOOK_LEVELS,
) -> tuple[List[BookLevelData], List[BookLevelData]]:
    """
    Builds bid/ask ladders. Width and quantity taper with depth.
    """
    half = max(mid, 0.01) * spread_pct * 0.5
    bids, asks = [], []
    for lvl in range(levels):
        w    = half * (1.0 + 0.80 * lvl)           # wider at each level
        qty  = max(1, int(rng.integers(10, 150) * math.exp(-0.60 * lvl)))
        ords = max(1, int(rng.integers(1, max(2, qty // 15))))
        bids.append(BookLevelData(
            price=round(max(mid - w, 0.01), 4), quantity=qty, order_count=ords))
        asks.append(BookLevelData(
            price=round(mid + w, 4), quantity=qty, order_count=ords))
    return bids, asks


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Path generators (pre-computed for the full day)
# ─────────────────────────────────────────────────────────────────────────────

def _underlying_path(seed: int) -> np.ndarray:
    """
    Geometric Brownian Motion, length TRADING_SECONDS+1.
    Annualised vol = ANNUAL_VOL, zero drift.
    """
    rng = np.random.default_rng(seed)
    dt  = DT_YEARS
    log_returns = (
        -0.5 * ANNUAL_VOL ** 2 * dt
        + ANNUAL_VOL * math.sqrt(dt) * rng.standard_normal(TRADING_SECONDS)
    )
    log_path = np.empty(TRADING_SECONDS + 1)
    log_path[0] = math.log(S0)
    log_path[1:] = log_path[0] + np.cumsum(log_returns)
    return np.exp(log_path)


def _iv_path(seed: int) -> np.ndarray:
    """
    Log-normal random walk for ATM vol, floored at 5%.
    Daily vol-of-vol ≈ 15% → IV drifts ~±3% of its value over the day.
    """
    rng = np.random.default_rng(seed + 99)
    per_tick = 0.15 / math.sqrt(TRADING_SECONDS)   # 15% annual vol-of-vol normalised
    log_iv = math.log(ANNUAL_VOL) + np.concatenate([
        [0.0], np.cumsum(per_tick * rng.standard_normal(TRADING_SECONDS))
    ])
    return np.maximum(np.exp(log_iv), 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Intraday volume weight  (U-shaped: heavy open/close, quiet midday)
# ─────────────────────────────────────────────────────────────────────────────

def _vol_weight(tick: int) -> float:
    x = tick / TRADING_SECONDS              # 0 → 1 through the day
    return 2.0 + 3.0 * (2.0 * abs(x - 0.5)) ** 4


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Per-option daily accumulators
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> dict:
    """Flat dict keyed (expiry_ts, strike, option_type)."""
    return {
        (exp, K, ot): {
            "vol": 0, "total_qty": 0, "total_val": 0.0,
            "last_px": None, "low": None, "high": None,
        }
        for exp in EXPIRIES
        for K   in STRIKES
        for ot  in ("CALL", "PUT")
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_day(seed: int = 42) -> Iterator[BookPublisherOptionChainDataMessage]:
    """
    Yields one BookPublisherOptionChainDataMessage per second.
    Total: 23,400 snapshots × 36 options (18 calls + 18 puts) each.

    Usage
    -----
        for snap in generate_day():
            process(snap)

        # or collect everything
        snapshots = list(generate_day())
    """
    rng      = np.random.default_rng(seed + 7)
    S_arr    = _underlying_path(seed)
    iv_arr   = _iv_path(seed)
    state    = _init_state()

    for tick in range(TRADING_SECONDS):
        ts_s  = MARKET_OPEN_S + tick
        ts_ns = ts_s * 1_000_000_000
        S     = float(S_arr[tick])
        atm_v = float(iv_arr[tick])
        vw    = _vol_weight(tick)

        calls: List[BookPublisherOptionDataMessage] = []
        puts:  List[BookPublisherOptionDataMessage] = []

        for exp_ts in EXPIRIES:
            tte = max((exp_ts - ts_s) / (365.0 * 86400.0), 1e-6)

            for K in STRIKES:
                for ot in ("CALL", "PUT"):
                    iv = smile_iv(S, K, tte, atm_v)
                    mid, delta, gamma, vega, theta = bs_greeks(S, K, tte, RATE, iv, ot)

                    # Spread widens for OTM and very short-dated options
                    log_m   = abs(math.log(K / S))
                    dtd     = tte * 252
                    sp_pct  = 0.004 + 0.008 * log_m * 5.0 + 0.003 * math.exp(-dtd / 10.0)
                    bids, asks = make_book(mid, sp_pct, rng)

                    # Stochastic trade simulation
                    st  = state[(exp_ts, K, ot)]
                    liq = math.exp(-log_m * 4.0)   # liquidity decays for OTM options
                    n_t = int(rng.poisson(max(0.05, 0.40 * liq * vw)))
                    for _ in range(n_t):
                        qty = int(rng.integers(1, 30))
                        px  = mid * (1.0 + float(rng.uniform(-sp_pct / 2.0, sp_pct / 2.0)))
                        st["vol"]       += 1
                        st["total_qty"] += qty
                        st["total_val"] += qty * px
                        st["last_px"]    = round(px, 4)

                    if st["low"]  is None or mid < st["low"]:  st["low"]  = mid
                    if st["high"] is None or mid > st["high"]: st["high"] = mid

                    # moneyness: integer number of 2.5-pt strike increments from ATM
                    moneyness = int(round((K - S) / 2.5))

                    opt = BookPublisherOptionDataMessage(
                        strike_px             = K,
                        option_type           = ot,
                        expiry_timestamp      = exp_ts,
                        mid_px                = round(mid, 4),
                        last_traded_price     = round(st["last_px"] or mid, 4),
                        total_traded_quantity = st["total_qty"],
                        total_traded_value    = round(st["total_val"], 2),
                        bids                  = bids,
                        asks                  = asks,
                        volume_since_day_start = st["vol"],
                        moneyness             = moneyness,
                        has_quote             = mid > 0.02,
                        low_since_day_start   = round(st["low"]  or mid, 4),
                        high_since_day_start  = round(st["high"] or mid, 4),
                        book_levels_per_side  = BOOK_LEVELS,
                        underlying_price      = round(S, 4),
                        tte                   = round(tte, 8),
                        delta                 = round(delta, 6),
                        gamma                 = round(gamma, 6),
                        vega                  = round(vega,  6),
                        theta                 = round(theta, 6),
                        iv                    = round(iv,    6),
                        rate_of_interest      = RATE,
                    )
                    (calls if ot == "CALL" else puts).append(opt)

        yield BookPublisherOptionChainDataMessage(
            seq_no       = tick + 1,
            key          = CHAIN_KEY,
            timestamp_ns = ts_ns,
            call_options = calls,
            put_options  = puts,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_day(path: str = "day_data.pkl", seed: int = 42) -> None:
    """Materialise all 23,400 snapshots and persist to pickle (protocol 5)."""
    snapshots = list(generate_day(seed=seed))
    Path(path).write_bytes(pickle.dumps(snapshots, protocol=5))
    print(f"Saved {len(snapshots):,} snapshots → {path}")


def load_day(path: str = "day_data.pkl") -> List[BookPublisherOptionChainDataMessage]:
    """Load a previously saved day."""
    return pickle.loads(Path(path).read_bytes())


# ─────────────────────────────────────────────────────────────────────────────
# 11. Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%H:%M:%S %Z")


def _print_snapshot(label: str, snap: BookPublisherOptionChainDataMessage) -> None:
    print(f"\n── {label}  seq={snap.seq_no:,}  t={_fmt_time(snap.timestamp_ns)} ──")

    # ATM call = strike 100 (index 4 in STRIKES), first expiry
    atm_c = snap.call_options[4]
    atm_p = snap.put_options[4]

    print(f"  Underlying : ${atm_c.underlying_price:.4f}")
    print(f"  ATM CALL   K={atm_c.strike_px:<6}  mid={atm_c.mid_px:<7.4f}  "
          f"iv={atm_c.iv:.3f}  δ={atm_c.delta:+.4f}  γ={atm_c.gamma:.5f}")
    print(f"  ATM PUT    K={atm_p.strike_px:<6}  mid={atm_p.mid_px:<7.4f}  "
          f"iv={atm_p.iv:.3f}  δ={atm_p.delta:+.4f}")
    print(f"  Bid/ask L1 : ${atm_c.bids[0].price:.4f} × {atm_c.bids[0].quantity}"
          f"  /  ${atm_c.asks[0].price:.4f} × {atm_c.asks[0].quantity}")
    print(f"  vol_since_start  call={atm_c.volume_since_day_start}  "
          f"put={atm_p.volume_since_day_start}")


def _print_smile(snap: BookPublisherOptionChainDataMessage) -> None:
    """Print the vol smile across all strikes for the near expiry."""
    print("\n  Vol smile (near expiry, near ATM):")
    print(f"  {'Strike':>8}  {'moneyness':>9}  {'CALL IV':>8}  {'CALL δ':>8}  "
          f"{'PUT IV':>8}  {'PUT δ':>8}")
    for i, K in enumerate(STRIKES):
        c = snap.call_options[i]    # first 9 = near expiry calls
        p = snap.put_options[i]
        print(f"  {K:>8.1f}  {c.moneyness:>9d}  {c.iv:>8.4f}  {c.delta:>+8.4f}  "
              f"{p.iv:>8.4f}  {p.delta:>+8.4f}")


if __name__ == "__main__":
    print("=" * 62)
    print("  Synthetic SPX option chain — 2024-03-11")
    print("  09:30–16:00 EDT  |  1-second resolution  |  seed=42")
    print("=" * 62)
    print(f"\n  Strikes  : {STRIKES}")
    print(f"  Expiries : {len(EXPIRIES)} (32 DTE, 67 DTE)")
    print(f"  Options  : {len(STRIKES) * len(EXPIRIES) * 2} per snapshot  "
          f"({len(STRIKES) * len(EXPIRIES)} calls + {len(STRIKES) * len(EXPIRIES)} puts)")
    print(f"  Ticks    : {TRADING_SECONDS:,}")
    print(f"  Total    : {TRADING_SECONDS * len(STRIKES) * len(EXPIRIES) * 2:,} option ticks")

    gen = generate_day(seed=42)

    first = next(gen)
    _print_snapshot("FIRST SNAPSHOT", first)
    _print_smile(first)

    # Fast-forward to last snapshot
    last = None
    for last in gen:
        pass

    _print_snapshot("LAST SNAPSHOT", last)

    # End-of-day ATM call stats
    atm_eod = last.call_options[4]
    print(f"\n  ATM call day range  : low={atm_eod.low_since_day_start:.4f}"
          f"  high={atm_eod.high_since_day_start:.4f}")
    print(f"  ATM call total vol  : {atm_eod.volume_since_day_start:,} trades  "
          f"/ {atm_eod.total_traded_quantity:,} contracts")

    print("\n  Done.")
