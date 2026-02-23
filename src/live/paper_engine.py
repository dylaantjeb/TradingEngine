"""
Paper trading engine – CSV stream simulation.

Reads bars one-by-one from a CSV file to mimic live streaming.
Applies same signal generation + risk + paper fills as the backtest,
but row-by-row in "real time" (optionally with a configurable sleep).

Usage:
    python -m src.cli live-paper --symbol ES --input data/raw/ES_M1.csv

The engine:
  1. Loads the trained model, scaler, and feature schema.
  2. Maintains a rolling window of raw bars.
  3. For each new bar it re-computes features and queries the model.
  4. Applies a signal filter (confidence threshold + ATR-based spread).
  5. Simulates paper fills (no real orders are placed).
  6. Prints a live P&L summary every N bars.

PAPER TRADING ONLY – no live order placement.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque

import numpy as np
import pandas as pd

from src.features.builder import build_features, MIN_ROWS
from src.backtest.engine import (
    COMMISSION_PER_SIDE,
    DEFAULT_CONTRACT_MULTIPLIER,
    DEFAULT_TICK_SIZE,
)

log = logging.getLogger(__name__)

# ── Paper fill configuration ──────────────────────────────────────────────────
SLIPPAGE_TICKS = 1
SUMMARY_INTERVAL = 50       # print summary every N bars
CONFIDENCE_THRESHOLD = 0.50  # min probability of predicted class to take a trade
WARMUP_BARS = MIN_ROWS + 10  # bars needed before first signal


def run_paper(symbol: str, csv_path: Path, bar_delay: float = 0.0) -> None:
    """
    Stream `csv_path` bar-by-bar and simulate paper trading.

    Parameters
    ----------
    symbol     : Symbol name (used to load model artifacts).
    csv_path   : Path to raw OHLCV CSV produced by `fetch`.
    bar_delay  : Optional sleep in seconds between bars (0 = max speed).
    """
    # ── Load artifacts ─────────────────────────────────────────────────────────
    model, scaler, feature_names, inv_label_map = _load_artifacts(symbol)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading CSV %s …", csv_path)
    raw_df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    raw_df.sort_index(inplace=True)
    log.info("Loaded %d bars from %s to %s", len(raw_df), raw_df.index[0], raw_df.index[-1])

    # ── State ──────────────────────────────────────────────────────────────────
    bar_buffer: Deque[pd.Series] = deque(maxlen=WARMUP_BARS + 200)
    position = 0         # -1, 0, +1
    entry_price = 0.0
    equity = 0.0
    trades: list[dict] = []
    bar_count = 0

    slippage_pts = SLIPPAGE_TICKS * DEFAULT_TICK_SIZE

    print(f"\n{'='*60}")
    print(f"  Paper Engine – {symbol}  (SIMULATION ONLY)")
    print(f"{'='*60}")
    print(f"  Streaming {len(raw_df)} bars …  Ctrl-C to stop\n")

    try:
        for ts, row in raw_df.iterrows():
            bar_count += 1
            bar_buffer.append(row)

            if bar_count < WARMUP_BARS:
                continue  # not enough history for features

            # Build feature vector from buffer
            buf_df = pd.DataFrame(list(bar_buffer))
            buf_df.index = raw_df.index[bar_count - len(bar_buffer): bar_count]

            try:
                feat = build_features(buf_df, min_rows=1)
            except Exception:
                continue  # not enough clean rows yet

            if len(feat) == 0:
                continue

            last_features = feat.iloc[[-1]][feature_names]
            X_scaled = scaler.transform(last_features)

            proba = model.predict_proba(X_scaled)[0]  # [short_prob, flat_prob, long_prob]
            pred_enc = int(np.argmax(proba))
            confidence = float(proba[pred_enc])
            signal = inv_label_map[str(pred_enc)]

            # Apply confidence filter
            if confidence < CONFIDENCE_THRESHOLD:
                signal = 0

            px = float(row["close"])

            # ── Fill logic ─────────────────────────────────────────────────────
            if position == 0 and signal != 0:
                fill_px = px + signal * slippage_pts
                position = signal
                entry_price = fill_px
                log.debug(
                    "[%s] ENTER %s @ %.2f  (conf=%.2f)",
                    ts, "LONG" if signal > 0 else "SHORT", fill_px, confidence,
                )
            elif position != 0 and (signal != position or signal == 0):
                fill_px = px - position * slippage_pts
                raw_pnl = position * (fill_px - entry_price) * DEFAULT_CONTRACT_MULTIPLIER
                net_pnl = raw_pnl - 2 * COMMISSION_PER_SIDE
                equity += net_pnl
                trades.append(
                    {
                        "time": str(ts),
                        "dir": "L" if position > 0 else "S",
                        "entry": round(entry_price, 2),
                        "exit": round(fill_px, 2),
                        "pnl": round(net_pnl, 2),
                        "equity": round(equity, 2),
                    }
                )
                log.debug(
                    "[%s] EXIT %s @ %.2f  pnl=%.2f  equity=%.2f",
                    ts, "LONG" if position > 0 else "SHORT", fill_px, net_pnl, equity,
                )
                position = 0
                entry_price = 0.0

                if signal != 0:
                    fill_px2 = px + signal * slippage_pts
                    position = signal
                    entry_price = fill_px2

            # ── Periodic summary ───────────────────────────────────────────────
            if bar_count % SUMMARY_INTERVAL == 0:
                n_trades = len(trades)
                wins = sum(1 for t in trades if t["pnl"] > 0)
                wr = wins / n_trades if n_trades else 0
                print(
                    f"  bar {bar_count:6d} | pos={position:+d} | "
                    f"equity={equity:+.2f} | trades={n_trades} | "
                    f"winrate={wr:.1%} | last_signal={signal:+d} conf={confidence:.2f}"
                )

            if bar_delay > 0:
                time.sleep(bar_delay)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    # ── Final summary ──────────────────────────────────────────────────────────
    _print_summary(symbol, trades, equity, bar_count)


def _print_summary(symbol: str, trades: list[dict], equity: float, bars: int) -> None:
    n = len(trades)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n if n else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    print(f"\n{'='*60}")
    print(f"  PAPER SESSION COMPLETE – {symbol}")
    print(f"{'='*60}")
    print(f"  Total bars      : {bars}")
    print(f"  Total trades    : {n}")
    print(f"  Net P&L         : ${equity:+.2f}")
    print(f"  Win rate        : {wr:.1%}")
    print(f"  Avg win         : ${avg_win:+.2f}")
    print(f"  Avg loss        : ${avg_loss:+.2f}")
    if losses:
        print(f"  Profit factor   : {abs(sum(wins)) / (abs(sum(losses)) + 1e-9):.2f}")
    print(f"{'='*60}\n")


def _load_artifacts(symbol: str):
    try:
        import joblib
    except ImportError:
        log.error("joblib not installed:  pip install joblib")
        sys.exit(1)

    model_path = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    scaler_path = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    schema_path = Path(f"artifacts/schema/{symbol}_features.json")

    for p in (model_path, scaler_path, schema_path):
        if not p.exists():
            log.error(
                "Artifact not found: %s\n"
                "  Run:  python -m src.cli train --symbol %s",
                p, symbol,
            )
            sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_names: list[str] = schema["feature_names"]
    inv_label_map: dict[str, int] = {k: int(v) for k, v in schema["inv_label_map"].items()}

    log.info(
        "Loaded model (%s features) and scaler for %s", len(feature_names), symbol
    )
    return model, scaler, feature_names, inv_label_map
