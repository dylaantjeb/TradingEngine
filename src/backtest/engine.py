"""
Vectorised backtest engine.

Uses pre-trained model to generate entry signals on historical bars,
then simulates fills with realistic costs:
  - Commission : $2.05 / contract (ES NQ typical retail IB rate)
  - Slippage   : 1 tick per fill (tick size from universe config)

Produces:
  - Equity curve (pandas Series)
  - Trades list
  - Performance metrics: Sharpe, Sortino, Max Drawdown, Win Rate, Expectancy

Report saved to:  artifacts/reports/<SYM>_backtest.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Cost model ────────────────────────────────────────────────────────────────
COMMISSION_PER_SIDE = 2.05   # USD per contract
DEFAULT_TICK_SIZE = 0.25     # ES tick = 0.25 points
DEFAULT_TICK_VALUE = 12.50   # ES: 1 tick = $12.50 (50 * 0.25)
DEFAULT_CONTRACT_MULTIPLIER = 50  # ES: $50 per point


def run_backtest(
    symbol: str,
    commission: float = COMMISSION_PER_SIDE,
    slippage_ticks: int = 1,
) -> dict[str, Any]:
    """
    Load model + features for `symbol`, run backtest, persist report.
    Returns metrics dict.
    """
    try:
        import joblib
    except ImportError:
        log.error("joblib not installed:  pip install joblib")
        sys.exit(1)

    # ── Load artifacts ─────────────────────────────────────────────────────────
    model_path = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    scaler_path = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    schema_path = Path(f"artifacts/schema/{symbol}_features.json")
    feat_path = Path(f"data/processed/{symbol}_features.parquet")
    raw_path = Path(f"data/raw/{symbol}_M1.csv")

    for p in (model_path, scaler_path, schema_path, feat_path):
        if not p.exists():
            log.error(
                "Missing artifact: %s\n"
                "  Run train first:  python -m src.cli train --symbol %s",
                p, symbol,
            )
            sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_names: list[str] = schema["feature_names"]
    inv_label_map: dict[str, int] = {k: int(v) for k, v in schema["inv_label_map"].items()}

    # ── Load features + raw prices ─────────────────────────────────────────────
    try:
        features = pd.read_parquet(feat_path, engine="pyarrow")
    except ImportError:
        log.error("pyarrow required:  pip install pyarrow")
        sys.exit(1)

    # Validate feature schema
    missing_cols = set(feature_names) - set(features.columns)
    if missing_cols:
        log.error("Feature mismatch: missing %s – rebuild dataset", missing_cols)
        sys.exit(1)

    X = features[feature_names].copy()
    X_scaled = scaler.transform(X)

    # ── Generate predictions ───────────────────────────────────────────────────
    proba = model.predict_proba(X_scaled)  # shape (n, 3): [short, flat, long]
    pred_encoded = np.argmax(proba, axis=1)
    # Map encoded labels back to {-1, 0, 1} using fast array lookup
    lookup = np.array([inv_label_map[str(k)] for k in range(3)], dtype=np.int8)
    signal = lookup[pred_encoded]

    # ── Align with price series ────────────────────────────────────────────────
    if raw_path.exists():
        prices = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        # Fall back to close prices from features (not ideal but workable)
        prices = pd.DataFrame(index=features.index)
        prices["close"] = np.nan
        log.warning("Raw CSV not found; returns will be unavailable. Run fetch first.")

    # Align indices
    common_idx = features.index.intersection(prices.index)
    if len(common_idx) == 0:
        # Try to use features index with forward prices from features directly
        log.warning("No common index between features and price file; using features index")
        common_idx = features.index

    sig_series = pd.Series(signal, index=features.index).reindex(common_idx)
    close_prices = prices["close"].reindex(common_idx) if "close" in prices.columns else pd.Series(dtype=float)

    # ── Simulate trades ────────────────────────────────────────────────────────
    trades, equity_curve = _simulate_trades(
        signals=sig_series,
        prices=close_prices,
        commission=commission,
        slippage_ticks=slippage_ticks,
        multiplier=DEFAULT_CONTRACT_MULTIPLIER,
        tick_size=DEFAULT_TICK_SIZE,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = _compute_metrics(equity_curve, trades)
    metrics["symbol"] = symbol
    metrics["n_bars"] = len(features)
    metrics["n_trades"] = len(trades)

    # ── Save report ────────────────────────────────────────────────────────────
    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{symbol}_backtest.json"

    report = {
        "symbol": symbol,
        "run_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "trades": trades[:500],  # limit to 500 for JSON size
        "equity_curve": equity_curve.round(2).tolist(),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Report saved to %s", report_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _simulate_trades(
    signals: pd.Series,
    prices: pd.Series,
    commission: float,
    slippage_ticks: int,
    multiplier: float,
    tick_size: float,
) -> tuple[list[dict], pd.Series]:
    """
    Bar-by-bar simulation using numpy arrays for speed.
    Position sizing: 1 contract per signal.
    Signal +1 → long, -1 → short, 0 → flat.
    """
    slippage_pts = slippage_ticks * tick_size
    cost_rt = 2 * commission  # round-trip commission

    sig_arr = signals.values.astype(np.int8)
    px_arr = prices.reindex(signals.index).values.astype(np.float64)
    ts_arr = signals.index
    n = len(sig_arr)

    # Pre-allocate
    equity_arr = np.zeros(n, dtype=np.float64)
    trades: list[dict] = []

    position = 0
    entry_price = 0.0
    entry_idx = 0
    equity = 0.0

    for i in range(n):
        px = px_arr[i]
        if np.isnan(px):
            equity_arr[i] = equity
            continue

        sig = int(sig_arr[i])

        if position == 0 and sig != 0:
            entry_price = px + sig * slippage_pts
            position = sig
            entry_idx = i
        elif position != 0 and (sig != position or sig == 0):
            fill_px = px - position * slippage_pts
            raw_pnl = position * (fill_px - entry_price) * multiplier
            net_pnl = raw_pnl - cost_rt
            equity += net_pnl
            trades.append(
                {
                    "entry_time": str(ts_arr[entry_idx]),
                    "exit_time": str(ts_arr[i]),
                    "direction": position,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(fill_px, 4),
                    "raw_pnl": round(raw_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "equity": round(equity, 2),
                }
            )
            if sig != 0:
                entry_price = px + sig * slippage_pts
                position = sig
                entry_idx = i
            else:
                position = 0
                entry_price = 0.0

        equity_arr[i] = equity

    equity_series = pd.Series(equity_arr, index=signals.index, name="equity")
    return trades, equity_series


def _compute_metrics(equity: pd.Series, trades: list[dict]) -> dict[str, Any]:
    if len(trades) == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0,
            "win_rate": 0.0, "expectancy_usd": 0.0, "total_pnl_usd": 0.0,
            "profit_factor": 0.0,
        }

    pnls = [t["net_pnl"] for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)
    expectancy = np.mean(pnls)
    profit_factor = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")

    # Sharpe / Sortino on equity curve changes
    returns = equity.diff().dropna()
    mu = returns.mean()
    sigma = returns.std()
    sharpe = (mu / sigma * np.sqrt(252 * 390)) if sigma > 0 else 0.0  # annualised (390 bars/day)

    downside = returns[returns < 0].std()
    sortino = (mu / downside * np.sqrt(252 * 390)) if downside > 0 else 0.0

    # Max drawdown
    cummax = equity.cummax()
    dd = (equity - cummax)
    max_dd = dd.min()
    max_dd_pct = (max_dd / (cummax.max() + 1e-9)) * 100 if cummax.max() > 0 else 0.0

    return {
        "total_pnl_usd": round(total_pnl, 2),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_drawdown_usd": round(float(max_dd), 2),
        "max_drawdown_pct": round(float(max_dd_pct), 2),
        "win_rate": round(win_rate, 4),
        "expectancy_usd": round(float(expectancy), 2),
        "profit_factor": round(float(profit_factor), 4),
    }
