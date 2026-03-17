"""
Generate synthetic OHLCV minute bars for development/testing.

Usage:
    python scripts/generate_synthetic_data.py --symbol ES --days 500

Output: data/raw/<SYMBOL>_M1.csv

Key design choices:
  - NYSE session hours only (13:30-16:00 UTC, 150 bars/day) so every bar passes
    the session filter without wasting bars on overnight/weekend gaps.
  - Regime-switching volatility: trending periods have HIGH sigma so the
    vol_regime_20_60 feature (short/long vol ratio) is > 1 in trends and < 1
    in consolidations.  This is critical for the regime filter to classify bars
    correctly as TRENDING vs RANGING.
  - Long regime durations (50-120 bars = 50-120 minutes) give EMA indicators
    enough time to warm up and register a clear slope.

This is ONLY for development -- use real IBKR data for production.
"""

import argparse
import sys
from datetime import timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate(symbol: str, n_days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    bars_per_session = 150  # 13:30-16:00 UTC = 150 minutes

    # Business days only (Mon-Fri)
    all_days = pd.bdate_range("2022-01-03", periods=n_days, freq="B")

    start_price = 5200.0
    close_all = []
    timestamps_all = []

    price = start_price
    regime = 0          # 0=ranging, 1=trend-up, -1=trend-down
    regime_bars_left = 0

    for day in all_days:
        day_dt = day.to_pydatetime().replace(tzinfo=timezone.utc)
        session_start = day_dt.replace(hour=13, minute=30, second=0, microsecond=0)

        for m in range(bars_per_session):
            ts = session_start + timedelta(minutes=m)

            if regime_bars_left <= 0:
                r = rng.random()
                if r < 0.35:
                    regime = 1           # trending up
                    regime_bars_left = int(rng.integers(50, 120))
                elif r < 0.70:
                    regime = -1          # trending down
                    regime_bars_left = int(rng.integers(50, 120))
                else:
                    regime = 0           # ranging / consolidation
                    regime_bars_left = int(rng.integers(30, 80))

            # CRITICAL: trending = HIGH vol, ranging = LOW vol.
            # This makes vol_regime_20_60 > 0.7 during trends (passes regime filter)
            # and < 0.7 during consolidations (correctly blocked).
            if regime == 1:
                drift = 1.8e-4
                sigma = 0.0010   # high vol = clear trend
            elif regime == -1:
                drift = -1.8e-4
                sigma = 0.0010   # high vol = clear trend
            else:
                drift = 0.0
                sigma = 0.0004   # low vol = consolidation / chop

            ret = rng.normal(drift, sigma)
            price = price * np.exp(ret)
            close_all.append(price)
            timestamps_all.append(ts)
            regime_bars_left -= 1

    close = np.array(close_all)
    n_bars = len(close)

    # Realistic intraday spread / high-low range
    atr_pts = close * 0.0015
    open_   = np.roll(close, 1)
    open_[0] = start_price
    high = close + rng.uniform(0.05, 0.9, n_bars) * atr_pts
    low  = close - rng.uniform(0.05, 0.9, n_bars) * atr_pts
    high = np.maximum(high, np.maximum(open_, close))
    low  = np.minimum(low,  np.minimum(open_, close))

    # Volume: higher at session open
    volume = rng.integers(500, 3000, n_bars).astype(float)
    for i in range(0, n_bars, bars_per_session):
        volume[i : i + 15] *= 3.0

    df = pd.DataFrame({
        "timestamp": timestamps_all,
        "open":      np.round(open_, 2),
        "high":      np.round(high, 2),
        "low":       np.round(low, 2),
        "close":     np.round(close, 2),
        "volume":    volume,
    })
    df.set_index("timestamp", inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV session bars")
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--days",   type=int, default=500,
                        help="Number of trading days to generate (default 500 ≈ 2 years)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    out_dir  = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}_M1.csv"

    df = generate(args.symbol, args.days, args.seed)
    df.to_csv(out_path)
    print(f"Generated {len(df)} bars for {args.symbol} → {out_path}")
    print(f"  Price range : {df['close'].min():.2f} – {df['close'].max():.2f}")
    print(f"  Date range  : {df.index[0]} → {df.index[-1]}")
    print(f"  Bars/day    : 150 (NYSE session 13:30–16:00 UTC only)")


if __name__ == "__main__":
    main()
