"""
Feature engineering for the TradingEngine.

All features are computed from OHLCV minute bars.
No external TA library required – everything is implemented with pandas / numpy.

Feature list
────────────
Price-based
  ret_1, ret_5, ret_10, ret_20          log returns over N bars
  hl_spread                             (high - low) / close
  overnight_gap                         open vs prior close

Momentum
  rsi_14                                RSI(14)
  rsi_slope                             5-bar RSI slope (normalised)

Volatility
  atr_14                                ATR(14) normalised by close
  vol_regime_20_60                      20-bar realised vol / 60-bar realised vol
  vol_z                                 z-score of realised vol

Volume
  vol_ratio                             volume / 20-bar rolling mean volume
  vwap_dev                              (close - session VWAP) / session VWAP

Time
  hour_sin, hour_cos                    cyclical encoding of bar hour
  dow_sin,  dow_cos                     cyclical encoding of day-of-week
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum rows needed to compute all features
MIN_ROWS = 80


def build_features(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS,
) -> pd.DataFrame:
    """
    Given a OHLCV DataFrame (index=timestamp, columns={open,high,low,close,volume})
    return a DataFrame of features with the same index (NaN rows dropped).

    Parameters
    ----------
    df       : Raw OHLCV bars, DatetimeIndex, UTC preferred.
    min_rows : Raise ValueError if result has fewer rows than this.
    """
    _validate_input(df)
    feat = pd.DataFrame(index=df.index)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"].replace(0, np.nan)

    # ── Log returns ──────────────────────────────────────────────────────────
    log_c = np.log(c)
    for n in (1, 5, 10, 20):
        feat[f"ret_{n}"] = log_c.diff(n)

    # ── HL spread ─────────────────────────────────────────────────────────────
    feat["hl_spread"] = (h - l) / c

    # ── Overnight gap ─────────────────────────────────────────────────────────
    feat["overnight_gap"] = np.log(o) - log_c.shift(1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = _rsi(c, period=14)
    feat["rsi_14"] = rsi / 100.0  # scale to [0,1]
    feat["rsi_slope"] = rsi.diff(5) / 100.0

    # ── ATR normalised ────────────────────────────────────────────────────────
    atr = _atr(h, l, c, period=14)
    feat["atr_14"] = atr / c

    # ── Volatility regime ─────────────────────────────────────────────────────
    ret_1 = feat["ret_1"]
    vol_20 = ret_1.rolling(20).std()
    vol_60 = ret_1.rolling(60).std()
    feat["vol_regime_20_60"] = vol_20 / vol_60.replace(0, np.nan)

    # z-score of 20-bar vol relative to its own 60-bar rolling distribution
    feat["vol_z"] = (vol_20 - vol_20.rolling(60).mean()) / (
        vol_20.rolling(60).std().replace(0, np.nan)
    )

    # ── Volume features ───────────────────────────────────────────────────────
    vol_mean = v.rolling(20, min_periods=5).mean()
    feat["vol_ratio"] = v / vol_mean.replace(0, np.nan)

    # VWAP – session (calendar day) reset
    feat["vwap_dev"] = _vwap_deviation(df)

    # ── Time features ─────────────────────────────────────────────────────────
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = idx.dayofweek.astype(float)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # ── Drop NaN rows from lead/lag calculations ───────────────────────────────
    feat.dropna(inplace=True)
    log.debug("Feature matrix: %s rows × %s columns (after dropna)", *feat.shape)

    if len(feat) < min_rows:
        raise ValueError(
            f"Feature matrix has only {len(feat)} rows after NaN removal; "
            f"need at least {min_rows}. Fetch more data."
        )

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Low-level indicator helpers
# ──────────────────────────────────────────────────────────────────────────────


def _validate_input(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (same formula as most charting platforms)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP that resets each calendar day.
    Returns (close - vwap) / vwap.
    """
    c = df["close"]
    v = df["volume"].replace(0, np.nan)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0

    # Group by date for daily reset
    date_key = df.index.date
    cum_tv = (typical * v).groupby(date_key).transform("cumsum")
    cum_v = v.groupby(date_key).transform("cumsum")
    vwap = cum_tv / cum_v
    return (c - vwap) / vwap.replace(0, np.nan)
