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

OPTIMISED: inner scan uses numpy array slicing (no Python inner loop).
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
    DataFrame with columns: label (-1/0/+1), hold_bars, ret_at_exit, daily_vol.
    Index aligned with df.
    """
    _validate_input(df)
    close = df["close"].values.astype(np.float64)
    n = len(close)

    # ── Rolling volatility (lagged by 1 to avoid look-ahead) ─────────────────
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = 0.0
    log_ret[1:] = np.log(close[1:] / close[:-1])

    vol_raw = pd.Series(log_ret).rolling(vol_lookback, min_periods=5).std().values
    vol_arr = np.empty(n, dtype=np.float64)
    vol_arr[0] = np.nan
    vol_arr[1:] = vol_raw[:-1]  # lag by 1

    finite = vol_arr[np.isfinite(vol_arr)]
    fill_val = np.median(finite) if len(finite) > 0 else 0.001
    vol_arr = np.where(np.isfinite(vol_arr), vol_arr, fill_val)
    np.maximum(vol_arr, min_vol, out=vol_arr)

    # ── Pre-compute barrier levels ────────────────────────────────────────────
    upper_px = close * (1.0 + pt * vol_arr)
    lower_px = close * (1.0 - sl * vol_arr)

    labels = np.zeros(n, dtype=np.int8)
    hold_bars = np.zeros(n, dtype=np.int32)
    ret_at_exit = np.zeros(n, dtype=np.float64)

    # Vectorised inner scan: numpy slicing replaces Python inner loop
    for i in range(n - 1):
        end = min(i + max_hold, n - 1)
        fwd = close[i + 1 : end + 1]              # forward price window
        hit_up = fwd >= upper_px[i]
        hit_dn = fwd <= lower_px[i]

        first_up = hit_up.argmax() if hit_up.any() else len(fwd)
        first_dn = hit_dn.argmax() if hit_dn.any() else len(fwd)

        if first_up < first_dn and hit_up.any():
            j = first_up + 1
            labels[i] = 1
        elif first_dn < first_up and hit_dn.any():
            j = first_dn + 1
            labels[i] = -1
        elif first_up == first_dn and hit_up.any():
            j = first_up + 1
            labels[i] = 1 if close[i + j] >= close[i] else -1
        else:
            j = end - i
            labels[i] = 0

        hold_bars[i] = j
        exit_idx = min(i + j, n - 1)
        ret_at_exit[i] = np.log(close[exit_idx] / close[i])

    # Last bar has no forward path
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
