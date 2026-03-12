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

Microstructure
  vwap_distance                         (close - VWAP) / ATR14
  momentum_3                            close - close[3 bars ago]
  momentum_6                            close - close[6 bars ago]
  range_ratio                           (high - low) / ATR14
  ema20_slope                           EMA(20) - EMA(20)[5 bars ago]
  volume_delta                          volume - volume[1 bar ago]

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

    # Work with numpy arrays directly for speed
    c_arr = df["close"].values.astype(np.float64)
    h_arr = df["high"].values.astype(np.float64)
    l_arr = df["low"].values.astype(np.float64)
    o_arr = df["open"].values.astype(np.float64)
    v_arr = df["volume"].values.astype(np.float64)
    v_arr = np.where(v_arr == 0, np.nan, v_arr)
    n_rows = len(c_arr)
    idx = df.index

    # Pre-allocate all feature columns as a dict of arrays (one-shot DataFrame)
    cols: dict[str, np.ndarray] = {}

    # ── Log returns ──────────────────────────────────────────────────────────
    log_c = np.log(c_arr)
    for lag in (1, 5, 10, 20):
        ret = np.empty(n_rows)
        ret[:lag] = np.nan
        ret[lag:] = log_c[lag:] - log_c[:-lag]
        cols[f"ret_{lag}"] = ret

    # ── HL spread ─────────────────────────────────────────────────────────────
    cols["hl_spread"] = (h_arr - l_arr) / c_arr

    # ── Overnight gap ─────────────────────────────────────────────────────────
    gap = np.empty(n_rows)
    gap[0] = np.nan
    gap[1:] = np.log(o_arr[1:]) - log_c[:-1]
    cols["overnight_gap"] = gap

    # ── RSI (Wilder, using pandas ewm for correctness) ───────────────────────
    c_series = df["close"]
    rsi = _rsi(c_series, period=14)
    rsi_vals = rsi.values / 100.0
    rsi_slope = np.empty(n_rows)
    rsi_slope[:5] = np.nan
    rsi_slope[5:] = rsi_vals[5:] - rsi_vals[:-5]
    cols["rsi_14"] = rsi_vals
    cols["rsi_slope"] = rsi_slope

    # ── ATR normalised ────────────────────────────────────────────────────────
    atr = _atr(df["high"], df["low"], c_series, period=14)
    cols["atr_14"] = atr.values / c_arr

    # ── Volatility regime ─────────────────────────────────────────────────────
    ret_1 = pd.Series(cols["ret_1"])
    vol_20 = ret_1.rolling(20).std()
    vol_60 = ret_1.rolling(60).std()
    vol_60_safe = vol_60.values.copy()
    vol_60_safe[vol_60_safe == 0] = np.nan
    cols["vol_regime_20_60"] = vol_20.values / vol_60_safe

    vol_20_mean_60 = vol_20.rolling(60).mean().values
    vol_20_std_60 = vol_20.rolling(60).std().values.copy()
    vol_20_std_60[vol_20_std_60 == 0] = np.nan
    cols["vol_z"] = (vol_20.values - vol_20_mean_60) / vol_20_std_60

    # ── Volume features ───────────────────────────────────────────────────────
    vol_mean = pd.Series(v_arr).rolling(20, min_periods=5).mean().values.copy()
    vol_mean[vol_mean == 0] = np.nan
    cols["vol_ratio"] = v_arr / vol_mean

    # ── VWAP deviation (optimised: avoids groupby for single-date buffers) ────
    _vwap_arr = _vwap_raw(h_arr, l_arr, c_arr, v_arr, idx)
    _vwap_safe = _vwap_arr.copy()
    _vwap_safe[_vwap_safe == 0] = np.nan
    cols["vwap_dev"] = (c_arr - _vwap_safe) / _vwap_safe

    # ── Microstructure features ───────────────────────────────────────────────
    _atr_raw = atr.values
    _atr_safe = _atr_raw.copy()
    _atr_safe[_atr_safe == 0] = np.nan

    cols["vwap_distance"] = (c_arr - _vwap_arr) / _atr_safe

    _mom3 = np.empty(n_rows)
    _mom3[:3] = np.nan
    _mom3[3:] = c_arr[3:] - c_arr[:-3]
    cols["momentum_3"] = _mom3

    _mom6 = np.empty(n_rows)
    _mom6[:6] = np.nan
    _mom6[6:] = c_arr[6:] - c_arr[:-6]
    cols["momentum_6"] = _mom6

    cols["range_ratio"] = (h_arr - l_arr) / _atr_safe

    _ema20 = c_series.ewm(span=20, adjust=False).mean().values
    _ema20_slope = np.empty(n_rows)
    _ema20_slope[:5] = np.nan
    _ema20_slope[5:] = _ema20[5:] - _ema20[:-5]
    cols["ema20_slope"] = _ema20_slope

    _vol_delta = np.empty(n_rows)
    _vol_delta[0] = np.nan
    _vol_delta[1:] = v_arr[1:] - v_arr[:-1]
    cols["volume_delta"] = _vol_delta

    # ── Time features (pure numpy, no pandas) ─────────────────────────────────
    hour = idx.hour + idx.minute / 60.0
    cols["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    cols["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = idx.dayofweek.astype(np.float64)
    cols["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    cols["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # ── Build DataFrame in one shot (faster than column-by-column) ────────────
    feat = pd.DataFrame(cols, index=idx)
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


def _vwap_raw(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    idx: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Intraday VWAP that resets each calendar day.
    Returns the raw VWAP price array (NaN where volume is zero).

    Optimised: uses numpy cumsum with day-boundary resets instead of
    pandas groupby().transform(), which is the main bottleneck.
    """
    n = len(close)
    typical = (high + low + close) / 3.0
    tv = typical * volume

    dates = idx.date
    cum_tv = np.empty(n, dtype=np.float64)
    cum_v = np.empty(n, dtype=np.float64)
    running_tv = 0.0
    running_v = 0.0

    for i in range(n):
        if i == 0 or dates[i] != dates[i - 1]:
            running_tv = 0.0
            running_v = 0.0
        running_tv += tv[i] if np.isfinite(tv[i]) else 0.0
        running_v += volume[i] if np.isfinite(volume[i]) else 0.0
        cum_tv[i] = running_tv
        cum_v[i] = running_v

    cum_v[cum_v == 0] = np.nan
    vwap = cum_tv / cum_v
    vwap[vwap == 0] = np.nan
    return vwap


# ─────────────────────────────────────────────────────────────────────────────
# Regime classifier – public utility (not a model feature)
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_ATR_MIN             = 0.80   # ATR / rolling_mean(ATR, lookback) lower bound
_REGIME_VOL_MIN             = 0.70   # vol_regime_20_60 lower bound
_REGIME_TREND_SLOPE_MIN     = 0.08   # |EMA20_slope| / ATR_pts lower bound
_REGIME_ATR_LOOKBACK        = 100    # bars for ATR rolling mean


def compute_regime(
    atr_ticks: pd.Series,
    ema20_slope_abs: pd.Series,
    vol_ratio: pd.Series,
    lookback: int = _REGIME_ATR_LOOKBACK,
    trend_slope_threshold: float = _REGIME_TREND_SLOPE_MIN,
    atr_regime_min: float = _REGIME_ATR_MIN,
    vol_regime_min: float = _REGIME_VOL_MIN,
) -> pd.Series:
    """
    Classify each bar into a market regime.

    Parameters
    ----------
    atr_ticks            : ATR expressed in ticks (or any consistent price unit).
    ema20_slope_abs      : |EMA(20) - EMA(20).shift(5)| in the same unit as atr_ticks.
    vol_ratio            : Short/long realised-vol ratio (e.g. vol_regime_20_60).
    lookback             : Rolling window for ATR percentile baseline.
    trend_slope_threshold: |slope| / ATR must exceed this to classify as trend.
    atr_regime_min       : ATR / rolling_mean(ATR) threshold; below = low_vol.
    vol_regime_min       : vol_ratio threshold; below = low_vol.

    Returns
    -------
    pd.Series of int8:
       1  = trend   (tradeable — adequate vol, strong directional slope)
       0  = chop    (blocked  — adequate vol but directionless)
      -1  = low_vol (blocked  — insufficient volatility)
    """
    atr_mean   = atr_ticks.rolling(lookback, min_periods=20).mean()
    atr_regime = (atr_ticks / atr_mean.replace(0, np.nan)).fillna(1.0)

    atr_safe       = atr_ticks.replace(0, np.nan)
    trend_strength = (ema20_slope_abs / atr_safe).fillna(0.0)

    vol = vol_ratio.fillna(1.0)

    low_vol_mask = (atr_regime < atr_regime_min) | (vol < vol_regime_min)
    trend_mask   = (~low_vol_mask) & (trend_strength >= trend_slope_threshold)

    regime = pd.Series(0, index=atr_ticks.index, dtype=np.int8)
    regime[low_vol_mask] = -1
    regime[trend_mask]   =  1
    return regime
