"""
EMA momentum rule-based signal generator.

This is the PRIMARY signal source in the production rules-first architecture.
XGBoost is demoted to a confidence VETO only — it cannot generate entries,
only block them.

Signal logic
------------
  LONG  (return +1):  close > EMA(N)  AND  slope > 0   AND  |slope|/ATR_pts >= min_frac
  SHORT (return -1):  close < EMA(N)  AND  slope < 0   AND  |slope|/ATR_pts >= min_frac
  FLAT  (return  0):  all other conditions

``slope`` is defined as  EMA(N)[t] - EMA(N)[t - slope_lookback]  — not the
close vs EMA gap — so it captures the trend *momentum* rather than just
whether price is above the line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_rules_signal(
    close: pd.Series,
    atr_pts: pd.Series,
    ema_period: int = 20,
    slope_lookback: int = 5,
    min_slope_atr_frac: float = 0.12,
) -> pd.Series:
    """
    Vectorised EMA momentum signal for an entire price series.

    Parameters
    ----------
    close              : Bar close prices (DatetimeIndex).
    atr_pts            : ATR in price points (same index as close).
    ema_period         : EMA lookback (bars).
    slope_lookback     : How many bars back to measure EMA slope.
    min_slope_atr_frac : Minimum |slope| / ATR_pts to qualify.

    Returns
    -------
    pd.Series of int8: +1 (LONG), -1 (SHORT), 0 (FLAT).
    """
    ema = close.ewm(span=ema_period, adjust=False).mean()
    slope = ema - ema.shift(slope_lookback)

    atr_safe = atr_pts.replace(0, np.nan)
    slope_quality = slope.abs() / atr_safe

    long_cond = (
        (close > ema)
        & (slope > 0)
        & (slope_quality >= min_slope_atr_frac)
    )
    short_cond = (
        (close < ema)
        & (slope < 0)
        & (slope_quality >= min_slope_atr_frac)
    )

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
    """
    Single-bar version used in the live paper engine bar loop.

    Parameters
    ----------
    close_val          : Current bar close price.
    ema_val            : Current bar EMA value.
    slope_val          : EMA[t] - EMA[t - slope_lookback] (pre-computed).
    atr_pts_val        : ATR in price points (must be > 0).
    min_slope_atr_frac : Minimum |slope| / ATR_pts.

    Returns
    -------
    int: +1, -1, or 0.
    """
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
