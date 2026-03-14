"""
Rule-based signal generators for the TradingEngine.

Two signal types:

1. generate_rules_signal() — EMA momentum continuation
   Entry: price already on the correct side of EMA, slope confirmed.
   Use case: trending market, ride momentum.

2. generate_pullback_signal() — Trend pullback resumption (PRIMARY production signal)
   Entry: higher-timeframe trend confirmed (EMA_slow slope), price pulled
   back against trend to EMA_fast, then CLOSED BACK THROUGH EMA_fast.
   This is the "dip and resume" pattern.

   Long setup:
     - EMA_slow slope > 0 (M5 trend bullish, approximated via EMA100 on M1)
     - Price pulled back: previous bar closed below EMA_fast (EMA25 on M1)
     - Resumption: current bar closes above EMA_fast
     - Slope quality gate: |EMA_slow slope| / ATR_pts >= min_slope_frac

   Short setup: mirror conditions.

   This requires 3 bars minimum: slow trend + pullback bar + resumption bar.

ML is OPTIONAL. XGBoost may veto signals but cannot generate them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── EMA momentum continuation ─────────────────────────────────────────────────

def generate_rules_signal(
    close: pd.Series,
    atr_pts: pd.Series,
    ema_period: int = 20,
    slope_lookback: int = 5,
    min_slope_atr_frac: float = 0.12,
) -> pd.Series:
    """
    Vectorised EMA momentum signal.

    LONG:  close > EMA AND slope > 0 AND |slope|/ATR >= min_frac
    SHORT: close < EMA AND slope < 0 AND |slope|/ATR >= min_frac
    """
    ema = close.ewm(span=ema_period, adjust=False).mean()
    slope = ema - ema.shift(slope_lookback)
    atr_safe = atr_pts.replace(0, np.nan)
    slope_quality = slope.abs() / atr_safe

    long_cond  = (close > ema) & (slope > 0) & (slope_quality >= min_slope_atr_frac)
    short_cond = (close < ema) & (slope < 0) & (slope_quality >= min_slope_atr_frac)

    sig = pd.Series(0, index=close.index, dtype=np.int8)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def generate_bar_signal(
    close_val: float,
    ema_val: float,
    slope_val: float,
    atr_pts_val: float,
    min_slope_atr_frac: float = 0.12,
) -> int:
    """Single-bar EMA momentum signal. Returns +1, -1, or 0."""
    if atr_pts_val <= 0 or np.isnan(slope_val) or np.isnan(ema_val):
        return 0
    quality = abs(slope_val) / atr_pts_val
    if quality < min_slope_atr_frac:
        return 0
    if close_val > ema_val and slope_val > 0:
        return 1
    if close_val < ema_val and slope_val < 0:
        return -1
    return 0


# ── Trend pullback resumption (PRIMARY production signal) ─────────────────────

def generate_pullback_signal(
    close: pd.Series,
    atr_pts: pd.Series,
    ema_fast_period: int = 25,    # M1 EMA25 ≈ M5 EMA5 — local swing reference
    ema_slow_period: int = 100,   # M1 EMA100 ≈ M5 EMA20 — directional bias
    slope_lookback: int = 5,
    min_slope_atr_frac: float = 0.15,
) -> pd.Series:
    """
    Vectorised trend pullback resumption signal.

    Detects the pattern: trend confirmed → pullback to EMA_fast → resumption.

    Long signal (+1) when ALL:
      1. EMA_slow slope > 0 (higher-timeframe bullish)
      2. EMA_slow |slope| / ATR >= min_slope_atr_frac (real trend, not drift)
      3. Previous bar close < EMA_fast (pullback occurred)
      4. Current bar close > EMA_fast (resumption confirmed)

    Short signal (-1): mirror conditions.

    This is a 2-bar pattern (requires t-1 and t). The signal fires at bar t
    (close of the resumption bar). Fill at bar t+1 open per execution model.

    Parameters
    ----------
    close               : Bar close prices.
    atr_pts             : ATR in price points (same index).
    ema_fast_period     : Fast EMA for pullback reference (M1 EMA25 ≈ M5 EMA5).
    ema_slow_period     : Slow EMA for trend direction (M1 EMA100 ≈ M5 EMA20).
    slope_lookback      : Bars to measure slow EMA slope.
    min_slope_atr_frac  : Minimum |slow_slope| / ATR to qualify as real trend.

    Returns
    -------
    pd.Series of int8: +1, -1, 0.
    """
    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()
    slow_slope = ema_slow - ema_slow.shift(slope_lookback)

    atr_safe      = atr_pts.replace(0, np.nan)
    slope_quality = slow_slope.abs() / atr_safe

    # Trend quality: slope must be real, not noise
    trend_strong = slope_quality >= min_slope_atr_frac

    # Previous bar close relative to fast EMA (shift by 1)
    prev_close    = close.shift(1)
    prev_ema_fast = ema_fast.shift(1)

    long_cond = (
        (slow_slope > 0)           # higher-TF trend bullish
        & trend_strong             # slope quality gate
        & (prev_close < prev_ema_fast)  # pullback: prior bar dipped below fast EMA
        & (close > ema_fast)       # resumption: current bar reclaimed fast EMA
    )
    short_cond = (
        (slow_slope < 0)           # higher-TF trend bearish
        & trend_strong
        & (prev_close > prev_ema_fast)  # pullback: prior bar bounced above fast EMA
        & (close < ema_fast)       # resumption: current bar broke back below fast EMA
    )

    sig = pd.Series(0, index=close.index, dtype=np.int8)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def generate_pullback_bar_signal(
    close_val: float,
    prev_close_val: float,
    ema_fast_val: float,
    prev_ema_fast_val: float,
    slow_slope_val: float,
    atr_pts_val: float,
    min_slope_atr_frac: float = 0.15,
) -> int:
    """
    Single-bar pullback resumption signal for the live paper engine bar loop.

    Parameters
    ----------
    close_val          : Current bar close.
    prev_close_val     : Previous bar close.
    ema_fast_val       : Current bar fast EMA value.
    prev_ema_fast_val  : Previous bar fast EMA value.
    slow_slope_val     : Current bar slow EMA slope (pre-computed).
    atr_pts_val        : ATR in price points.
    min_slope_atr_frac : Minimum |slow_slope| / ATR.

    Returns
    -------
    int: +1, -1, or 0.
    """
    if (atr_pts_val <= 0
            or np.isnan(slow_slope_val)
            or np.isnan(ema_fast_val)
            or np.isnan(prev_ema_fast_val)
            or np.isnan(prev_close_val)):
        return 0

    quality = abs(slow_slope_val) / atr_pts_val
    if quality < min_slope_atr_frac:
        return 0

    # Long: slow trend up, prev bar dipped below fast EMA, current resumed above
    if (slow_slope_val > 0
            and prev_close_val < prev_ema_fast_val
            and close_val > ema_fast_val):
        return 1

    # Short: slow trend down, prev bar bounced above fast EMA, current dropped below
    if (slow_slope_val < 0
            and prev_close_val > prev_ema_fast_val
            and close_val < ema_fast_val):
        return -1

    return 0
