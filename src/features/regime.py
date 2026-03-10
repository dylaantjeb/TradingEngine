"""
Regime detection for TradingEngine.

Classifies each bar into one of three market regimes:
  TRENDING  – directional price movement (ADX > threshold)
  VOLATILE  – high ATR relative to recent baseline (news, event risk)
  RANGING   – low ADX + normal ATR (mean-reversion environment)

All computations use only information available at bar t (no lookahead).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    TRENDING  = "TRENDING"
    VOLATILE  = "VOLATILE"
    RANGING   = "RANGING"


# ── Low-level indicator helpers ────────────────────────────────────────────────


def _true_range(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    hl   = high - low
    hpc  = np.abs(high - prev_close)
    lpc  = np.abs(low  - prev_close)
    return np.maximum(hl, np.maximum(hpc, lpc))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr  = _true_range(high, low, prev_close)
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


def _directional_movement(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (+DI, -DI, ADX) arrays."""
    prev_high  = np.roll(high, 1)
    prev_low   = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0]  = high[0]
    prev_low[0]   = low[0]
    prev_close[0] = close[0]

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = _atr(high, low, close, period)
    atr_vals = np.where(atr_vals == 0, 1e-10, atr_vals)

    plus_di  = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr_vals
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr_vals

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

    return plus_di, minus_di, adx


# ── Main public function ───────────────────────────────────────────────────────


def detect_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    atr_period: int = 14,
    atr_volatile_multiplier: float = 2.0,
    atr_baseline_period: int = 60,
) -> pd.Series:
    """
    Classify each bar into a Regime.

    Parameters
    ----------
    df                      : OHLCV DataFrame (must have high, low, close columns).
    adx_period              : ADX smoothing period.
    adx_trend_threshold     : ADX > this → TRENDING.
    atr_period              : ATR smoothing period.
    atr_volatile_multiplier : ATR > baseline × this multiplier → VOLATILE.
    atr_baseline_period     : Rolling window for ATR baseline (median).

    Returns
    -------
    pd.Series of Regime values, same index as df.
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_regime: missing columns {missing}")

    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    _, _, adx = _directional_movement(high, low, close, adx_period)
    atr_vals  = _atr(high, low, close, atr_period)

    atr_baseline = (
        pd.Series(atr_vals, index=df.index)
        .rolling(atr_baseline_period, min_periods=1)
        .median()
        .values
    )

    n = len(df)
    regimes = np.empty(n, dtype=object)

    for i in range(n):
        if atr_vals[i] > atr_baseline[i] * atr_volatile_multiplier:
            regimes[i] = Regime.VOLATILE
        elif adx[i] > adx_trend_threshold:
            regimes[i] = Regime.TRENDING
        else:
            regimes[i] = Regime.RANGING

    return pd.Series(regimes, index=df.index, name="regime", dtype=object)


def add_regime_column(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience wrapper: return df with a 'regime' column appended.
    Does NOT modify the input DataFrame in place.
    """
    out = df.copy()
    out["regime"] = detect_regime(df, **kwargs)
    return out
