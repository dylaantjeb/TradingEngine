"""
Walk-forward (time-series cross-validation) for TradingEngine.

Three modes
-----------
  rolling  : fixed-size train window slides forward with each fold  (default)
  expanding: anchored train window grows with each fold (anchored at t=0)
  split    : single train/test cut by time fraction (simplest OOS test)

Leakage guarantees
------------------
  • Model and RobustScaler are fit ONLY on the train slice for each fold.
  • Production artifacts (artifacts/) are NEVER written during walk-forward;
    per-fold training runs entirely in memory.
  • Test slice timestamps are strictly greater than train slice timestamps
    (enforced by integer-index slicing on a time-sorted DataFrame).
  • EMA(200) for the trend filter is computed on the full close series so
    every test window has proper warm-up history before its first bar.
  • Raw prices for execution fills come from data/raw/<symbol>_M1.csv;
    if the file is absent a close-only proxy is used with a logged warning.

Stability metrics reported
--------------------------
  pct_profitable_folds : fraction of folds with net_pnl > 0
  avg_profit_factor    : mean profit factor across folds
  pnl_cv               : coeff. of variation of fold PnLs (lower = more stable)
  std_sharpe           : std of fold Sharpe ratios        (lower = more stable)
  t_stat_pnl           : t-statistic of fold PnLs         (>1.65 → 95% sig.)
  funded_ready         : FUNDED-READY | PROMISING | MARGINAL | NOT READY

Stability thresholds (stricter than industry-standard for funded-account safety)
---------------------------------------------------------------------------------
  FUNDED-READY  : pct_profitable ≥ 0.70  AND  avg_pf ≥ 1.50  AND
                  pnl_cv ≤ 1.0           AND  t_stat ≥ 1.65
  PROMISING     : pct_profitable ≥ 0.60  AND  avg_pf ≥ 1.30  AND
                  pnl_cv ≤ 1.5           AND  total_pnl > 0
  MARGINAL      : pct_profitable ≥ 0.40  AND  avg_pf ≥ 1.10
  NOT READY     : anything else

Usage
-----
  # rolling (default) – fixed train window slides forward
  python -m src.cli walk-forward --symbol ES

  # expanding – train window grows from origin
  python -m src.cli walk-forward --symbol ES --mode expanding \\
      --train-bars 8000 --test-bars 2000

  # single train/test split (80/20)
  python -m src.cli walk-forward --symbol ES --mode split --split-pct 0.8

  # skip Optuna → fast fixed hyperparameters (recommended for many folds)
  python -m src.cli walk-forward --symbol ES --no-optuna
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_ARTS = Path("artifacts")
_UNIVERSE_CFG = Path("config/universe.yaml")

# Funded-account readiness thresholds
_FUNDED_MIN_PCT_PROF   = 0.70   # ≥ 70% of folds profitable
_FUNDED_MIN_PF         = 1.50   # avg profit factor ≥ 1.5
_FUNDED_MAX_CV         = 1.00   # PnL CV ≤ 1.0  (tight consistency)
_FUNDED_MIN_TSTAT      = 1.65   # t-statistic ≥ 1.65 (95% one-sided)

_PROMISING_MIN_PCT_PROF = 0.60
_PROMISING_MIN_PF       = 1.30
_PROMISING_MAX_CV       = 1.50

_MARGINAL_MIN_PCT_PROF  = 0.40
_MARGINAL_MIN_PF        = 1.10


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    """Per-fold backtest results."""
    fold_idx:             int
    mode:                 str
    train_start:          str
    train_end:            str
    train_bars:           int
    test_start:           str
    test_end:             str
    test_bars:            int
    n_trades:             int   = 0
    net_pnl:              float = 0.0
    gross_pnl:            float = 0.0
    win_rate:             float = 0.0
    profit_factor:        float = 0.0
    sharpe:               float = 0.0
    sortino:              float = 0.0
    max_drawdown_usd:     float = 0.0
    max_drawdown_pct:     float = 0.0
    expectancy_usd:       float = 0.0
    avg_hold_bars:        float = 0.0
    trades_per_day:       float = 0.0
    consecutive_losses_max: int = 0
    profitable:           bool  = False


@dataclass
class WalkForwardSummary:
    symbol:     str
    mode:       str
    n_folds:    int
    train_bars: int
    test_bars:  int
    folds:      list[FoldResult] = field(default_factory=list)
    aggregate:  dict             = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_walk_forward(
    symbol: str,
    mode: str = "rolling",
    train_bars: int = 10_000,
    test_bars: int = 2_000,
    step_bars: Optional[int] = None,
    min_train_bars: Optional[int] = None,
    split_pct: float = 0.8,
    n_trials: int = 0,
    save_report: bool = True,
) -> WalkForwardSummary:
    """
    Run walk-forward / OOS validation.

    Parameters
    ----------
    symbol        : Symbol to validate (must have data/processed/ parquets).
    mode          : 'rolling' | 'expanding' | 'split'.
    train_bars    : Training window size in bars (rolling / expanding).
    test_bars     : Test window size in bars.
    step_bars     : Step between folds (default = test_bars → non-overlapping).
    min_train_bars: Expanding mode: minimum initial training size
                    (default = train_bars).
    split_pct     : Split mode: fraction of dataset used for training
                    (default 0.8).
    n_trials      : Optuna trials per fold (0 = fast fixed hyperparameters;
                    recommended for walk-forward with many folds).
    save_report   : Write JSON report to artifacts/reports/.

    Returns
    -------
    WalkForwardSummary with per-fold FoldResult objects and aggregate metrics.
    """
    if mode not in ("rolling", "expanding", "split"):
        raise ValueError(
            f"mode must be 'rolling', 'expanding', or 'split', got {mode!r}"
        )

    step    = step_bars    or test_bars
    min_tr  = min_train_bars or train_bars

    # ── Load features + labels ─────────────────────────────────────────────
    feat_path = Path(f"data/processed/{symbol}_features.parquet")
    lbl_path  = Path(f"data/processed/{symbol}_labels.parquet")
    for p in (feat_path, lbl_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Processed data not found: {p}\n"
                f"Run:  python -m src.cli build-dataset --symbol {symbol} --input <csv>"
            )

    features = pd.read_parquet(feat_path)
    labels   = pd.read_parquet(lbl_path)
    aligned  = features.join(labels[["label"]], how="inner").sort_index()
    n_total  = len(aligned)

    feature_names = [c for c in aligned.columns if c != "label"]

    log.info(
        "Walk-forward [%s] %s → %s  |  %d bars  |  %d features  |  mode=%s",
        symbol,
        aligned.index[0].date(), aligned.index[-1].date(),
        n_total, len(feature_names), mode,
    )

    # ── Raw prices + EMA (computed once on full series) ────────────────────
    raw_prices, full_ema = _load_prices(symbol, aligned.index)

    # ── Engine config ──────────────────────────────────────────────────────
    from src.backtest.engine import _load_cfg
    cfg   = _load_cfg(symbol)
    specs = _load_specs(symbol)
    starting_equity = float(cfg.get("starting_equity", 100_000.0))

    # ── Build fold index tuples ────────────────────────────────────────────
    if mode == "rolling":
        fold_ranges = _build_folds_rolling(n_total, train_bars, test_bars, step)
    elif mode == "expanding":
        fold_ranges = _build_folds_expanding(n_total, min_tr, test_bars, step)
    else:  # split
        fold_ranges = _build_folds_split(n_total, split_pct)

    if not fold_ranges:
        raise ValueError(
            f"No folds could be built: need ≥ {train_bars + test_bars} bars, "
            f"have {n_total}."
        )

    log.info(
        "%d fold(s) created  (mode=%s  train=%d  test=%d  step=%d)",
        len(fold_ranges), mode, train_bars, test_bars, step,
    )

    # ── Main fold loop ─────────────────────────────────────────────────────
    folds: list[FoldResult] = []

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(fold_ranges):
        train_df = aligned.iloc[tr_s:tr_e]
        test_df  = aligned.iloc[te_s:te_e]

        log.info(
            "Fold %d/%d  train [%s → %s, %d bars]  test [%s → %s, %d bars]",
            fold_idx + 1, len(fold_ranges),
            train_df.index[0].date(), train_df.index[-1].date(), len(train_df),
            test_df.index[0].date(),  test_df.index[-1].date(),  len(test_df),
        )

        # ── Train (in-memory; no disk writes) ─────────────────────────────
        try:
            model, scaler, inv_label_map = _train_on_slice(
                train_df, feature_names, n_trials
            )
        except Exception as exc:
            log.warning("Fold %d  training failed: %s — skipping", fold_idx, exc)
            continue

        # ── Backtest on test window ────────────────────────────────────────
        try:
            metrics = _backtest_on_slice(
                model=model,
                scaler=scaler,
                feature_names=feature_names,
                inv_label_map=inv_label_map,
                test_df=test_df,
                raw_prices=raw_prices,
                full_ema=full_ema,
                cfg=cfg,
                specs=specs,
                starting_equity=starting_equity,
            )
        except Exception as exc:
            log.warning("Fold %d  backtest failed: %s — skipping", fold_idx, exc)
            continue

        fr = FoldResult(
            fold_idx=fold_idx,
            mode=mode,
            train_start=str(train_df.index[0]),
            train_end=str(train_df.index[-1]),
            train_bars=len(train_df),
            test_start=str(test_df.index[0]),
            test_end=str(test_df.index[-1]),
            test_bars=len(test_df),
            n_trades=metrics.get("n_trades", 0),
            net_pnl=metrics.get("net_pnl", 0.0),
            gross_pnl=metrics.get("gross_pnl", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            sharpe=metrics.get("sharpe", 0.0),
            sortino=metrics.get("sortino", 0.0),
            max_drawdown_usd=metrics.get("max_drawdown_usd", 0.0),
            max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
            expectancy_usd=metrics.get("expectancy_usd", 0.0),
            avg_hold_bars=metrics.get("avg_trade_duration_bars", 0.0),
            trades_per_day=metrics.get("trades_per_day", 0.0),
            consecutive_losses_max=metrics.get("consecutive_losses_max", 0),
            profitable=metrics.get("net_pnl", 0.0) > 0,
        )
        folds.append(fr)

    agg = _aggregate(folds)
    summary = WalkForwardSummary(
        symbol=symbol,
        mode=mode,
        n_folds=len(folds),
        train_bars=train_bars,
        test_bars=test_bars,
        folds=folds,
        aggregate=agg,
    )

    _print_summary(summary)

    if save_report and folds:
        _save_report(symbol, summary)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Fold-building helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_folds_rolling(
    n: int, train_bars: int, test_bars: int, step_bars: int,
) -> list[tuple[int, int, int, int]]:
    """
    Fixed-size rolling window folds.

    Each fold:
      train = [start, start + train_bars)
      test  = [start + train_bars, start + train_bars + test_bars)

    Window advances by step_bars; stops when test window would exceed n.

    Returns list of (train_start, train_end, test_start, test_end) index tuples.
    """
    folds: list[tuple[int, int, int, int]] = []
    start = 0
    while start + train_bars + test_bars <= n:
        folds.append((
            start,
            start + train_bars,
            start + train_bars,
            start + train_bars + test_bars,
        ))
        start += step_bars
    return folds


def _build_folds_expanding(
    n: int, min_train_bars: int, test_bars: int, step_bars: int,
) -> list[tuple[int, int, int, int]]:
    """
    Expanding (anchored) window folds.

    Train always starts at 0 and grows by step_bars each fold.
    First fold: train = [0, min_train_bars), test = [min_train_bars, +test_bars).

    Returns list of (train_start, train_end, test_start, test_end) index tuples.
    """
    folds: list[tuple[int, int, int, int]] = []
    train_end = min_train_bars
    while train_end + test_bars <= n:
        folds.append((0, train_end, train_end, train_end + test_bars))
        train_end += step_bars
    return folds


def _build_folds_split(
    n: int, split_pct: float,
) -> list[tuple[int, int, int, int]]:
    """
    Single time-based train/test split.

    train = [0, int(n * split_pct))
    test  = [int(n * split_pct), n)

    Returns a list containing exactly one tuple.
    """
    split = int(n * split_pct)
    if split <= 0 or split >= n:
        raise ValueError(
            f"split_pct={split_pct:.2f} gives a degenerate split (split={split}, n={n})."
        )
    return [(0, split, split, n)]


# ─────────────────────────────────────────────────────────────────────────────
# Training helper (in-memory — never writes to artifacts/)
# ─────────────────────────────────────────────────────────────────────────────


def _train_on_slice(
    train_df: pd.DataFrame,
    feature_names: list[str],
    n_trials: int = 0,
) -> tuple:
    """
    Train XGBoost + RobustScaler on train_df in memory.

    Hyperparameters are tuned for OOS robustness (shallower trees,
    stronger regularisation) rather than in-sample accuracy.

    Parameters
    ----------
    train_df      : DataFrame with feature columns + "label" column.
    feature_names : Ordered list of feature column names to use.
    n_trials      : Optuna trials (0 = fast fixed hyperparameters).

    Returns
    -------
    (model, scaler, inv_label_map)
      model         — fitted XGBClassifier
      scaler        — fitted RobustScaler (fitted on train_df only)
      inv_label_map — {str(encoded_class): original_label_int}

    Raises
    ------
    ValueError if train_df has fewer than 2 distinct label classes.
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder, RobustScaler

    feat_cols = [c for c in feature_names if c in train_df.columns]
    if not feat_cols:
        raise ValueError("No matching feature columns found in train_df.")

    X_df  = train_df[feat_cols]          # keep as DataFrame (preserves feature names)
    y_raw = train_df["label"].values

    unique_classes = np.unique(y_raw)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Train slice has only {len(unique_classes)} unique label class(es): "
            f"{unique_classes}. XGBoost requires at least 2."
        )

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_df)      # scaler fitted on train window only

    inv_label_map = {str(i): int(cls) for i, cls in enumerate(le.classes_)}

    if n_trials > 0:
        model = _optuna_train(X_scaled, y_enc, n_trials)
    else:
        # Conservative fixed hyperparameters designed for OOS robustness:
        # - shallow trees (max_depth=3) → less overfitting
        # - strong L2 regularisation (reg_lambda=3)
        # - high min_child_weight → each leaf needs many samples
        # - slower learning rate → generalises across regimes
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=3.0,
            gamma=1.0,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_scaled, y_enc)

    return model, scaler, inv_label_map


def _optuna_train(
    X_scaled: np.ndarray,
    y_enc: np.ndarray,
    n_trials: int,
):
    """
    Optuna hyperparameter search within a training slice.

    Inner validation: last 20% of the training slice (time-ordered, no shuffle).
    Search space is biased towards conservative / generalisable models.
    """
    import optuna
    import xgboost as xgb
    from sklearn.metrics import f1_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split   = int(len(X_scaled) * 0.8)
    X_tr    = X_scaled[:split];  X_val = X_scaled[split:]
    y_tr    = y_enc[:split];     y_val = y_enc[split:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Conservative ranges — prefer depth ≤ 5 and higher regularisation
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 2, 5),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
            "gamma":            trial.suggest_float("gamma", 0.5, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "eval_metric":      "mlogloss",
            "random_state":     42,
            "verbosity":        0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        return float(f1_score(y_val, pred, average="macro", zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    model = xgb.XGBClassifier(
        **study.best_params,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y_enc)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Backtest helper (purely in-memory)
# ─────────────────────────────────────────────────────────────────────────────


def _backtest_on_slice(
    model,
    scaler,
    feature_names: list[str],
    inv_label_map: dict[str, int],
    test_df: pd.DataFrame,
    raw_prices: Optional[pd.DataFrame],
    full_ema: Optional[pd.Series],
    cfg: dict,
    specs: dict,
    starting_equity: float = 100_000.0,
) -> dict:
    """
    Run backtest on a test-window DataFrame.

    All inputs are in-memory; no disk reads.
    The scaler was fit on the train slice only — transform here is OOS.
    """
    from src.backtest.engine import _simulate_trades, _compute_metrics

    feat_cols = [c for c in feature_names if c in test_df.columns]
    X_df      = test_df[feat_cols]
    X_scaled  = scaler.transform(X_df)       # OOS transform — scaler fit on train only
    proba     = model.predict_proba(X_scaled)
    pred_enc  = np.argmax(proba, axis=1)
    conf      = np.max(proba, axis=1)

    sig_arr     = np.array([inv_label_map[str(e)] for e in pred_enc])
    sig_series  = pd.Series(sig_arr, index=test_df.index)
    conf_series = pd.Series(conf,    index=test_df.index)

    # Cost parameters (don't mutate the caller's dict)
    cfg          = dict(cfg)
    tick_size    = float(specs.get("tick_size",  cfg.get("tick_size", 0.25)))
    multiplier   = float(specs.get("multiplier", 50.0))
    cfg["tick_size"] = tick_size
    commission_rt = 2.0 * float(cfg.get("commission_per_side_usd", 1.50))
    slippage_pts  = float(cfg.get("slippage_ticks_per_side", 0.5)) * tick_size
    half_spread   = float(cfg.get("spread_ticks", 1.0)) * 0.5 * tick_size
    friction_pts  = slippage_pts + half_spread

    # Open / close prices for execution fills
    if raw_prices is not None:
        rp           = raw_prices.reindex(test_df.index)
        has_prices   = "close" in rp.columns and not rp["close"].isna().all()
    else:
        has_prices = False

    if has_prices:
        close_prices = rp["close"].ffill().bfill().fillna(5000.0)
        open_prices  = (
            rp["open"].ffill().bfill().fillna(close_prices)
            if "open" in rp.columns
            else close_prices
        )
    else:
        if raw_prices is not None:
            log.warning(
                "Raw prices reindexed to test window are all NaN — using 5000.0 proxy"
            )
        open_prices  = pd.Series(5000.0, index=test_df.index)
        close_prices = pd.Series(5000.0, index=test_df.index)

    # ATR in ticks
    atr_ticks = pd.Series(20.0, index=test_df.index)
    if "atr_14" in test_df.columns:
        atr_ticks = (test_df["atr_14"] * close_prices / tick_size).fillna(20.0)

    # EMA series: pass the full-dataset series so the test window has
    # proper warm-up history; _simulate_trades reindexes to test index.
    ema_series: Optional[pd.Series] = None
    if cfg.get("trend_filter_enabled", False) and full_ema is not None:
        ema_series = full_ema

    trades, equity_curve, cost_summary = _simulate_trades(
        signals      = sig_series,
        open_prices  = open_prices,
        atr_ticks    = atr_ticks,
        cfg          = cfg,
        friction_pts = friction_pts,
        commission_rt= commission_rt,
        multiplier   = multiplier,
        conf_series  = conf_series,
        close_prices = close_prices,
        ema_series   = ema_series,
    )

    metrics = _compute_metrics(
        equity_curve, trades, cost_summary,
        starting_equity=starting_equity,
    )
    metrics["n_trades"] = len(trades)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate(folds: list[FoldResult]) -> dict:
    """
    Compute aggregate + stability metrics across all completed folds.

    Funded-account readiness thresholds (stricter than old STABLE/PROMISING)
    -------------------------------------------------------------------------
    FUNDED-READY : pct_profitable ≥ 70%  AND  avg_pf ≥ 1.5   AND
                   pnl_cv ≤ 1.0          AND  t_stat ≥ 1.65
    PROMISING    : pct_profitable ≥ 60%  AND  avg_pf ≥ 1.3   AND
                   pnl_cv ≤ 1.5          AND  total_pnl > 0
    MARGINAL     : pct_profitable ≥ 40%  AND  avg_pf ≥ 1.1
    NOT READY    : anything else (avg_pf < 1.1, majority losing, or high variance)

    Stability metrics
    -----------------
    pnl_cv       : std(pnl) / |mean(pnl)|  — coefficient of variation.
                   Lower is more stable. Reported as None when mean_pnl ≤ 0.
    std_sharpe   : std of fold Sharpe ratios across folds.
    t_stat_pnl   : mean_pnl / (std_pnl / √n_folds).
                   Tests H₀: mean = 0. > 1.65 → one-sided 95% significance.
                   Reported as None when n_folds < 2 or std_pnl = 0.
    """
    if not folds:
        return {}

    n           = len(folds)
    net_pnls    = [f.net_pnl           for f in folds]
    win_rates   = [f.win_rate          for f in folds]
    sharpes     = [f.sharpe            for f in folds]
    sortinos    = [f.sortino           for f in folds]
    pfs         = [f.profit_factor     for f in folds]
    mdd_usd     = [f.max_drawdown_usd  for f in folds]
    mdd_pct     = [f.max_drawdown_pct  for f in folds]
    expectancy  = [f.expectancy_usd    for f in folds]
    tpd         = [f.trades_per_day    for f in folds]
    n_trades    = [f.n_trades          for f in folds]
    profitable  = sum(1 for f in folds if f.profitable)

    total_pnl   = float(sum(net_pnls))
    mean_pnl    = float(np.mean(net_pnls))
    std_pnl     = float(np.std(net_pnls, ddof=1)) if n > 1 else 0.0
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe  = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0
    avg_pf      = float(np.mean(pfs))
    pct_prof    = profitable / n

    # Coefficient of variation (meaningful only when mean_pnl > 0)
    pnl_cv: Optional[float] = (
        round(std_pnl / mean_pnl, 4) if mean_pnl > 0 else None
    )
    pnl_cv_val = pnl_cv if pnl_cv is not None else float("inf")

    # t-statistic of fold PnLs
    t_stat: Optional[float] = (
        round(mean_pnl / (std_pnl / math.sqrt(n)), 4)
        if (n > 1 and std_pnl > 0)
        else None
    )
    t_stat_val = t_stat if t_stat is not None else 0.0

    # ── Funded-account readiness verdict ─────────────────────────────────
    # Rules listed in decreasing order of strictness.
    # A strategy must pass ALL conditions for each tier.
    if (
        pct_prof >= _FUNDED_MIN_PCT_PROF
        and avg_pf >= _FUNDED_MIN_PF
        and pnl_cv_val <= _FUNDED_MAX_CV
        and t_stat_val >= _FUNDED_MIN_TSTAT
    ):
        verdict = "FUNDED-READY"
    elif (
        pct_prof >= _PROMISING_MIN_PCT_PROF
        and avg_pf >= _PROMISING_MIN_PF
        and pnl_cv_val <= _PROMISING_MAX_CV
        and total_pnl > 0
    ):
        verdict = "PROMISING"
    elif (
        pct_prof >= _MARGINAL_MIN_PCT_PROF
        and avg_pf >= _MARGINAL_MIN_PF
    ):
        verdict = "MARGINAL"
    else:
        verdict = "NOT READY"

    return {
        # Returns
        "total_net_pnl":          round(total_pnl, 2),
        "avg_net_pnl_per_fold":   round(mean_pnl, 2),
        "std_net_pnl":            round(std_pnl, 2),
        "total_n_trades":         int(sum(n_trades)),
        # Profitability
        "n_profitable_folds":     profitable,
        "pct_profitable_folds":   round(pct_prof, 4),
        # Win rate
        "avg_win_rate":           round(float(np.mean(win_rates)), 4),
        "median_win_rate":        round(float(np.median(win_rates)), 4),
        # Profit factor
        "avg_profit_factor":      round(avg_pf, 4),
        "median_profit_factor":   round(float(np.median(pfs)), 4),
        # Risk-adjusted returns
        "avg_sharpe":             round(mean_sharpe, 4),
        "std_sharpe":             round(std_sharpe, 4),
        "avg_sortino":            round(float(np.mean(sortinos)), 4),
        # Drawdown (% is always relative to starting_equity — no exploding values)
        "avg_max_drawdown_usd":   round(float(np.mean(mdd_usd)), 2),
        "avg_max_drawdown_pct":   round(float(np.mean(mdd_pct)), 4),
        "worst_drawdown_pct":     round(float(min(mdd_pct)), 4),
        # Activity
        "avg_expectancy_usd":     round(float(np.mean(expectancy)), 2),
        "avg_trades_per_day":     round(float(np.mean(tpd)), 2),
        # Stability / robustness
        "pnl_cv":                 pnl_cv,
        "t_stat_pnl":             t_stat,
        "funded_ready":           verdict,
        # Legacy alias so old JSON reports / tests still work
        "stability":              verdict,
        "n_folds":                n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

# Emoji-free verdict labels and their one-line assessment
_VERDICT_LINES = {
    "FUNDED-READY": "FUNDED-READY  — Passes all funded-account robustness criteria.",
    "PROMISING":    "PROMISING     — Positive OOS expectancy; tighten further before live.",
    "MARGINAL":     "MARGINAL      — Marginally profitable; not ready for funded trading.",
    "NOT READY":    "NOT READY     — Fails minimum robustness criteria. Do not trade live.",
}


def _print_summary(result: WalkForwardSummary) -> None:
    W = 92
    bar = "=" * W

    print(f"\n{bar}")
    print(
        f"  WALK-FORWARD VALIDATION  |  {result.symbol}"
        f"  |  mode={result.mode}  |  n_folds={result.n_folds}"
    )
    print(bar)

    # Per-fold table header
    col_w = 25
    hdr = (
        f"  {'Fold':>4}  {'Test period':<{col_w}}  {'Bars':>5}  "
        f"{'Trd':>4}  {'Net PnL ($)':>11}  {'WR%':>5}  "
        f"{'PF':>5}  {'Sharpe':>6}  {'MaxDD%':>7}  {'MaxDD($)':>10}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for f in result.folds:
        period = f"{f.test_start[:10]}>{f.test_end[:10]}"
        ok     = "[+]" if f.profitable else "[-]"
        print(
            f"  {f.fold_idx:>4}  {period:<{col_w}}  {f.test_bars:>5}  "
            f"{f.n_trades:>4}  {f.net_pnl:>+11.2f}  {f.win_rate:>5.1%}  "
            f"{f.profit_factor:>5.2f}  {f.sharpe:>6.3f}  "
            f"{f.max_drawdown_pct:>6.2f}%  {f.max_drawdown_usd:>+10.2f}  "
            f"{ok}"
        )

    agg = result.aggregate
    if not agg:
        print(f"{bar}\n")
        return

    print(sep)

    # Totals / averages row
    print(
        f"  {'SUM/AVG':>4}  {'':^{col_w}}  {'':>5}  "
        f"{agg['total_n_trades']:>4}  "
        f"{agg['total_net_pnl']:>+11.2f}  "
        f"{agg['avg_win_rate']:>5.1%}  "
        f"{agg['avg_profit_factor']:>5.2f}  "
        f"{agg['avg_sharpe']:>6.3f}  "
        f"{agg['avg_max_drawdown_pct']:>6.2f}%  "
        f"{agg['avg_max_drawdown_usd']:>+10.2f}"
    )

    # Aggregate block
    lbl = 36
    print(f"\n  {'AGGREGATE PERFORMANCE':-<{W - 4}}")
    print(f"  {'Profitable folds':{lbl}}: "
          f"{agg['n_profitable_folds']} / {agg['n_folds']}  "
          f"({agg['pct_profitable_folds']:.0%})")
    print(f"  {'Total net P&L':{lbl}}: ${agg['total_net_pnl']:>+,.2f}")
    print(f"  {'Avg net P&L / fold':{lbl}}: ${agg['avg_net_pnl_per_fold']:>+,.2f}"
          f"  +/- ${agg['std_net_pnl']:,.2f}")
    print(f"  {'Avg profit factor':{lbl}}: {agg['avg_profit_factor']:.3f}"
          f"  (median {agg['median_profit_factor']:.3f})"
          f"  [funded-account target >= 1.50]")
    print(f"  {'Avg win rate':{lbl}}: {agg['avg_win_rate']:.1%}"
          f"  (median {agg['median_win_rate']:.1%})")
    print(f"  {'Avg Sharpe (annualised)':{lbl}}: {agg['avg_sharpe']:.3f}"
          f"  (std {agg['std_sharpe']:.3f})")
    print(f"  {'Avg Sortino':{lbl}}: {agg['avg_sortino']:.3f}")
    print(f"  {'Avg max drawdown':{lbl}}: {agg['avg_max_drawdown_pct']:.2f}%"
          f"  / ${agg['avg_max_drawdown_usd']:+,.2f}"
          f"  (worst {agg['worst_drawdown_pct']:.2f}%)")
    print(f"  {'Avg trades / day':{lbl}}: {agg['avg_trades_per_day']:.2f}")
    print(f"  {'Avg expectancy / trade':{lbl}}: ${agg['avg_expectancy_usd']:+.2f}")

    # Stability block
    pnl_cv_str = (f"{agg['pnl_cv']:.3f}"
                  if agg.get("pnl_cv") is not None
                  else "n/a (mean PnL <= 0)")
    t_stat_str = (f"{agg['t_stat_pnl']:.2f}"
                  if agg.get("t_stat_pnl") is not None
                  else "n/a (< 2 folds)")
    verdict    = agg.get("funded_ready", agg.get("stability", "UNKNOWN"))

    print(f"\n  {'ROBUSTNESS / STABILITY':-<{W - 4}}")
    print(f"  {'PnL coeff. of variation (CV)':{lbl}}: {pnl_cv_str}"
          f"  [funded-account target <= 1.0]")
    print(f"  {'Sharpe std across folds':{lbl}}: {agg['std_sharpe']:.3f}"
          f"  [target < 0.40]")
    print(f"  {'PnL t-statistic':{lbl}}: {t_stat_str}"
          f"  [>= 1.65 -> 95% significance]")

    # Funded-readiness verdict box
    verdict_line = _VERDICT_LINES.get(verdict, f"UNKNOWN ({verdict})")
    print(f"\n  {'FUNDED-ACCOUNT READINESS':-<{W - 4}}")
    print(f"\n    >> {verdict_line}\n")
    print(f"  {'Criteria (all must pass for FUNDED-READY)':{lbl}}")
    pct_ok  = agg['pct_profitable_folds'] >= _FUNDED_MIN_PCT_PROF
    pf_ok   = agg['avg_profit_factor'] >= _FUNDED_MIN_PF
    cv_ok   = (agg.get('pnl_cv') is not None and agg['pnl_cv'] <= _FUNDED_MAX_CV)
    tst_ok  = (agg.get('t_stat_pnl') is not None
               and agg['t_stat_pnl'] >= _FUNDED_MIN_TSTAT)
    _pass_fail(f"  Profitable folds >= {_FUNDED_MIN_PCT_PROF:.0%}",
               pct_ok, f"{agg['pct_profitable_folds']:.0%}", lbl)
    _pass_fail(f"  Avg profit factor >= {_FUNDED_MIN_PF:.2f}",
               pf_ok, f"{agg['avg_profit_factor']:.3f}", lbl)
    _pass_fail(f"  PnL CV <= {_FUNDED_MAX_CV:.1f}",
               cv_ok, pnl_cv_str, lbl)
    _pass_fail(f"  PnL t-stat >= {_FUNDED_MIN_TSTAT:.2f}",
               tst_ok, t_stat_str, lbl)
    print(f"{bar}\n")


def _pass_fail(label: str, passed: bool, value: str, lbl_width: int) -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"    [{mark}]  {label:{lbl_width - 4}}: {value}")


def _save_report(symbol: str, result: WalkForwardSummary) -> None:
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_walk_forward.json"
    data = {
        "symbol":     result.symbol,
        "mode":       result.mode,
        "n_folds":    result.n_folds,
        "train_bars": result.train_bars,
        "test_bars":  result.test_bars,
        "aggregate":  result.aggregate,
        "folds":      [asdict(f) for f in result.folds],
    }
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=2, default=str)
    log.info("Walk-forward report -> %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Config / price loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_prices(
    symbol: str,
    idx: pd.DatetimeIndex,
) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load raw OHLCV prices and compute full-series EMA for trend filter.

    Returns
    -------
    (raw_prices, full_ema)
      raw_prices : DataFrame reindexed to `idx`, or None if CSV not found.
      full_ema   : EMA(period) of close over the full series, or None.
    """
    raw_path = Path(f"data/raw/{symbol}_M1.csv")
    if not raw_path.exists():
        log.warning(
            "Raw CSV not found: %s — open-price fills will use close proxy. "
            "Fetch data with:  python -m src.cli fetch --symbol %s",
            raw_path, symbol,
        )
        return None, None

    raw = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
    raw.sort_index(inplace=True)
    raw = raw.reindex(idx)     # align to the features index

    # EMA — computed on the FULL aligned series so any test window inherits
    # proper warm-up history (no cold-start artefact in trend filter)
    full_ema: Optional[pd.Series] = None
    if _UNIVERSE_CFG.exists() and "close" in raw.columns:
        with open(_UNIVERSE_CFG) as f:
            uc = yaml.safe_load(f) or {}
        tf = uc.get("trend_filter", {})
        if tf.get("enabled", False):
            ema_period = int(tf.get("ema_period", 200))
            full_ema   = raw["close"].ewm(span=ema_period, adjust=False).mean()

    return raw, full_ema


def _load_specs(symbol: str) -> dict:
    """Return contract_specs dict for symbol from universe.yaml."""
    if not _UNIVERSE_CFG.exists():
        return {}
    with open(_UNIVERSE_CFG) as f:
        uc = yaml.safe_load(f) or {}
    return uc.get("contract_specs", {}).get(symbol, {})
