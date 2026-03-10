"""
Walk-forward validation for TradingEngine.

Splits the dataset into rolling train/test windows, trains a fresh model
on each train window, then runs the backtest on the following test window.
Aggregates per-window metrics and prints a summary table.

Usage
-----
python -m src.cli walk-forward --symbol ES --train-bars 10000 --test-bars 2000
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_trades: int
    net_pnl: float
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float


@dataclass
class WalkForwardResult:
    symbol: str
    n_windows: int
    windows: list[WindowResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


def run_walk_forward(
    symbol: str,
    train_bars: int = 10_000,
    test_bars: int = 2_000,
    step_bars: Optional[int] = None,
    n_trials: int = 10,
    save_report: bool = True,
) -> WalkForwardResult:
    """
    Run walk-forward validation.

    Parameters
    ----------
    symbol      : Symbol to use (must have data/processed/<symbol>_*.parquet).
    train_bars  : Number of bars in each training window.
    test_bars   : Number of bars in each test window.
    step_bars   : Step between windows (default = test_bars → non-overlapping).
    n_trials    : Optuna trials per window training.
    save_report : Save JSON report to artifacts/reports/.
    """
    from src.training.train import train as _train
    from src.backtest.engine import run_backtest

    step_bars = step_bars or test_bars

    feat_path = Path(f"data/processed/{symbol}_features.parquet")
    lbl_path  = Path(f"data/processed/{symbol}_labels.parquet")

    if not feat_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(
            f"Processed data not found for {symbol}.  "
            "Run:  python -m src.cli build-dataset --symbol ES --input <csv>"
        )

    features = pd.read_parquet(feat_path)
    labels   = pd.read_parquet(lbl_path)

    aligned = features.join(labels[["label"]], how="inner")
    aligned = aligned.sort_index()
    n_total = len(aligned)

    log.info(
        "Walk-forward: %d total bars | train=%d test=%d step=%d",
        n_total, train_bars, test_bars, step_bars,
    )

    windows: list[WindowResult] = []
    start_idx = 0
    window_idx = 0

    while start_idx + train_bars + test_bars <= n_total:
        train_slice = aligned.iloc[start_idx : start_idx + train_bars]
        test_slice  = aligned.iloc[start_idx + train_bars : start_idx + train_bars + test_bars]

        log.info(
            "Window %d: train [%s → %s] | test [%s → %s]",
            window_idx,
            train_slice.index[0], train_slice.index[-1],
            test_slice.index[0],  test_slice.index[-1],
        )

        # ── Retrain on this window ─────────────────────────────────────────────
        try:
            _train_on_slice(symbol, train_slice, n_trials)
        except Exception as exc:
            log.warning("Window %d train failed: %s – skipping", window_idx, exc)
            start_idx += step_bars
            window_idx += 1
            continue

        # ── Backtest on test window ────────────────────────────────────────────
        try:
            metrics = _backtest_on_slice(symbol, test_slice)
        except Exception as exc:
            log.warning("Window %d backtest failed: %s – skipping", window_idx, exc)
            start_idx += step_bars
            window_idx += 1
            continue

        wr = WindowResult(
            window_idx=window_idx,
            train_start=str(train_slice.index[0]),
            train_end=str(train_slice.index[-1]),
            test_start=str(test_slice.index[0]),
            test_end=str(test_slice.index[-1]),
            n_trades=metrics.get("n_trades", 0),
            net_pnl=metrics.get("net_pnl", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            sharpe=metrics.get("sharpe", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
        )
        windows.append(wr)

        start_idx += step_bars
        window_idx += 1

    result = WalkForwardResult(
        symbol=symbol,
        n_windows=len(windows),
        windows=windows,
        aggregate=_aggregate(windows),
    )

    _print_results(result)

    if save_report and windows:
        _save_report(symbol, result)

    return result


# ── Internal helpers ───────────────────────────────────────────────────────────


def _train_on_slice(symbol: str, train_slice: pd.DataFrame, n_trials: int) -> None:
    """Train model on a DataFrame slice (saves artifacts to disk as usual)."""
    import joblib
    from src.training.train import _build_model   # reuse internal helper

    feat_cols = [c for c in train_slice.columns if c != "label"]
    X = train_slice[feat_cols].values
    y = train_slice["label"].values

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Minimal training – use default params for speed in walk-forward
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y_enc)

    # Save artefacts so backtest can load them
    arts = Path("artifacts")
    (arts / "models").mkdir(parents=True, exist_ok=True)
    (arts / "scalers").mkdir(parents=True, exist_ok=True)
    (arts / "schema").mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  arts / "models"  / f"{symbol}_xgb_best.joblib")
    joblib.dump(scaler, arts / "scalers" / f"{symbol}_scaler.joblib")

    inv_label_map = {str(i): int(cls) for i, cls in enumerate(le.classes_)}
    with open(arts / "schema" / f"{symbol}_features.json", "w") as f:
        json.dump({"feature_names": feat_cols, "inv_label_map": inv_label_map}, f)


def _backtest_on_slice(symbol: str, test_slice: pd.DataFrame) -> dict:
    """Run a mini backtest on a DataFrame slice, return metrics dict."""
    from src.backtest.engine import _simulate_trades, _compute_metrics, _load_cfg

    cfg = _load_cfg(symbol)

    feat_cols = [c for c in test_slice.columns if c != "label"]
    X = test_slice[feat_cols].values

    import joblib
    arts = Path("artifacts")
    model  = joblib.load(arts / "models"  / f"{symbol}_xgb_best.joblib")
    scaler = joblib.load(arts / "scalers" / f"{symbol}_scaler.joblib")
    with open(arts / "schema" / f"{symbol}_features.json") as f:
        schema = json.load(f)

    inv_label_map = {k: int(v) for k, v in schema["inv_label_map"].items()}
    X_scaled = scaler.transform(X)
    proba    = model.predict_proba(X_scaled)
    pred_enc = np.argmax(proba, axis=1)
    conf     = np.max(proba, axis=1)
    signals  = np.array([inv_label_map[str(e)] for e in pred_enc])
    signals[conf < 0.50] = 0

    trades, equity_curve = _simulate_trades(test_slice, signals, cfg, symbol)
    metrics = _compute_metrics(trades, equity_curve)
    return metrics


def _aggregate(windows: list[WindowResult]) -> dict:
    if not windows:
        return {}
    net_pnls     = [w.net_pnl for w in windows]
    win_rates    = [w.win_rate for w in windows]
    sharpes      = [w.sharpe for w in windows]
    pfs          = [w.profit_factor for w in windows]
    mdd          = [w.max_drawdown for w in windows]
    positive     = sum(1 for p in net_pnls if p > 0)

    return {
        "total_net_pnl":         round(sum(net_pnls), 2),
        "avg_net_pnl_per_window": round(float(np.mean(net_pnls)), 2),
        "pct_profitable_windows": round(positive / len(windows), 4),
        "avg_win_rate":          round(float(np.mean(win_rates)), 4),
        "avg_sharpe":            round(float(np.mean(sharpes)), 4),
        "avg_profit_factor":     round(float(np.mean(pfs)), 4),
        "avg_max_drawdown":      round(float(np.mean(mdd)), 4),
        "n_windows":             len(windows),
    }


def _print_results(result: WalkForwardResult) -> None:
    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD VALIDATION – {result.symbol}  ({result.n_windows} windows)")
    print(f"{'='*72}")
    header = f"{'Win':>4}  {'Test start':>20}  {'Trades':>6}  {'NetPnL':>10}  {'WR':>6}  {'PF':>6}  {'Sharpe':>7}"
    print(header)
    print("-" * len(header))
    for w in result.windows:
        print(
            f"{w.window_idx:>4}  {w.test_start:>20}  {w.n_trades:>6}  "
            f"{w.net_pnl:>+10.2f}  {w.win_rate:>6.1%}  {w.profit_factor:>6.2f}  "
            f"{w.sharpe:>7.3f}"
        )
    print("-" * len(header))
    agg = result.aggregate
    if agg:
        print(
            f"{'TOTAL':>4}  {'':>20}  {'':>6}  "
            f"{agg['total_net_pnl']:>+10.2f}  "
            f"{agg['avg_win_rate']:>6.1%}  "
            f"{agg['avg_profit_factor']:>6.2f}  "
            f"{agg['avg_sharpe']:>7.3f}"
        )
        print(f"\n  Profitable windows: {agg['pct_profitable_windows']:.0%}  "
              f"| Avg max-DD: {agg['avg_max_drawdown']:.1%}")
    print(f"{'='*72}\n")


def _save_report(symbol: str, result: WalkForwardResult) -> None:
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_walk_forward.json"
    data = {
        "symbol":    result.symbol,
        "n_windows": result.n_windows,
        "aggregate": result.aggregate,
        "windows":   [w.__dict__ for w in result.windows],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Walk-forward report saved → %s", out_path)
