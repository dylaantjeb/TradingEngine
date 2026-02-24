"""
Triple-Barrier labeling (López de Prado, AFML ch. 3).

For each bar we set:
  upper barrier : entry_price * (1 + pt * daily_vol)
  lower barrier : entry_price * (1 - sl * daily_vol)
  time barrier  : entry_price + max_hold bars

Label:
  +1  → upper barrier hit first
  -1  → lower barrier hit first
   0  → time barrier hit first (neither profit nor stop reached)

daily_vol is a rolling estimate of daily returns volatility,
scaled from the minute bar returns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def label_triple_barrier(
    df: pd.DataFrame,
    *,
    pt: float = 1.5,
    sl: float = 1.0,
    max_hold: int = 60,
    vol_lookback: int = 20,
    min_vol: float = 1e-6,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df          : OHLCV DataFrame with DatetimeIndex.
    pt          : Profit-take multiplier on volatility.
    sl          : Stop-loss multiplier on volatility.
    max_hold    : Maximum holding period in bars.
    vol_lookback: Rolling window for volatility estimate.
    min_vol     : Floor on volatility to avoid degenerate barriers.

    Returns
    -------
    DataFrame with columns: label (-1/0/+1), hold_bars, ret_at_exit.
    Index aligned with df.
    """
    _validate_input(df)
    close = df["close"].values
    n = len(close)

    # Rolling volatility of log-returns (annualised then scaled to per-bar)
    log_ret = np.log(close[1:] / close[:-1])
    vol_series = pd.Series(log_ret).rolling(vol_lookback, min_periods=5).std()
    vol_series = vol_series.reindex(range(n - 1)).shift(1)  # lag by 1 to avoid look-ahead
    vol_arr = np.full(n, np.nan)
    vol_arr[1:] = vol_series.values
    vol_arr = np.nan_to_num(vol_arr, nan=np.nanmedian(vol_arr[~np.isnan(vol_arr)]) if np.any(~np.isnan(vol_arr)) else 0.001)
    vol_arr = np.clip(vol_arr, min_vol, None)

    labels = np.zeros(n, dtype=np.int8)
    hold_bars = np.full(n, max_hold, dtype=np.int32)
    ret_at_exit = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        entry = close[i]
        daily_vol = vol_arr[i]
        upper = entry * (1 + pt * daily_vol)
        lower = entry * (1 - sl * daily_vol)
        end_idx = min(i + max_hold, n - 1)

        for j in range(i + 1, end_idx + 1):
            price = close[j]
            if price >= upper:
                labels[i] = 1
                hold_bars[i] = j - i
                ret_at_exit[i] = np.log(price / entry)
                break
            if price <= lower:
                labels[i] = -1
                hold_bars[i] = j - i
                ret_at_exit[i] = np.log(price / entry)
                break
        else:
            # Time barrier hit
            labels[i] = 0
            hold_bars[i] = end_idx - i
            ret_at_exit[i] = np.log(close[end_idx] / entry)

    # Last bar has no forward path – mark as unknown (0) and hold=0
    labels[-1] = 0
    hold_bars[-1] = 0

    result = pd.DataFrame(
        {
            "label": labels,
            "hold_bars": hold_bars,
            "ret_at_exit": ret_at_exit,
            "daily_vol": vol_arr,
        },
        index=df.index,
    )

    dist = result["label"].value_counts().sort_index()
    log.info(
        "Triple-barrier labels  -1:%d  0:%d  +1:%d  (pt=%.2f sl=%.2f max_hold=%d)",
        dist.get(-1, 0),
        dist.get(0, 0),
        dist.get(1, 0),
        pt,
        sl,
        max_hold,
    )
    return result


def _validate_input(df: pd.DataFrame) -> None:
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must have a 'close' column")
    if len(df) < 10:
        raise ValueError(f"Too few rows ({len(df)}) to compute labels; need at least 10.")
