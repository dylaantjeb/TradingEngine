"""
Generate synthetic OHLCV minute bars for development/testing.

Usage:
    python scripts/generate_synthetic_data.py --symbol ES --bars 5000

Output: data/raw/<SYMBOL>_M1.csv

This is ONLY for development – use real IBKR data for production.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate(symbol: str, n_bars: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Start from a realistic ES price level with small drift
    start_price = 5200.0
    drift = 2e-5      # very small positive drift per bar
    sigma = 0.0008    # ~0.08% per bar, realistic for ES 1-min

    log_ret = rng.normal(drift, sigma, n_bars)
    close = start_price * np.exp(np.cumsum(log_ret))

    # Realistic intraday spread / high-low range
    atr_pts = close * 0.0012
    high = close + rng.uniform(0.1, 1.0, n_bars) * atr_pts
    low = close - rng.uniform(0.1, 1.0, n_bars) * atr_pts
    open_ = np.roll(close, 1)
    open_[0] = start_price

    # Volume: higher during open/close sessions
    base_vol = rng.integers(500, 3000, n_bars).astype(float)
    # Timestamps: continuous trading hours (Sunday 18:00 – Friday 17:00 CME)
    dates = pd.date_range("2024-01-02 14:30", periods=n_bars, freq="1min", tz="UTC")

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": base_vol,
        }
    )
    # Ensure OHLC consistency
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV bars")
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}_M1.csv"

    df = generate(args.symbol, args.bars, args.seed)
    df.set_index("timestamp", inplace=True)
    df.to_csv(out_path)
    print(f"Generated {args.bars} bars for {args.symbol} → {out_path}")
    print(f"  Price range: {df['close'].min():.2f} – {df['close'].max():.2f}")
    print(f"  Date range:  {df.index[0]} → {df.index[-1]}")


if __name__ == "__main__":
    main()
