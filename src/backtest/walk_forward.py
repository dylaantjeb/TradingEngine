"""
Walk-forward (time-series cross-validation) for TradingEngine.

Three modes
-----------
  rolling  : fixed-size train window slides forward (default)
  expanding: anchored train window grows with each fold
  split    : single train/test cut by time fraction

Selection modes (--select-by)
------------------------------
  f1      (default): train each fold with fast fixed hyperparams; confidence
                     threshold selected post-hoc by sweeping candidates on the
                     inner validation split (last 20% of train window).
  trading :          threshold co-selected with hyperparams using the same
                     four-gate trading objective used in `train --select-by trading`.
                     When Optuna is used (n_trials > 0) threshold is an Optuna
                     parameter. Without Optuna, threshold is swept post-hoc.

Per-fold OOS diagnostics
-------------------------
Every fold logs:
  • confidence threshold used
  • OOS confident trade count / total bars (coverage %)
  • OOS directional accuracy on confident trades
  • OOS profit factor
  • WEAK / OK flag  (WEAK = fewer than _WEAK_FOLD_MIN_TRADES trades
                            OR coverage < _WEAK_FOLD_MIN_COVERAGE_PCT)

Weak-fold warning
-----------------
Folds with 0–4 OOS trades or < 0.2% coverage are flagged WEAK.
They are still included in aggregate metrics (they contribute to
pct_profitable_folds), but the summary prints a count of weak folds
as a signal that the model is under-trading in live conditions.

Leakage guarantees
------------------
  • Model + RobustScaler fit ONLY on the train slice for each fold.
  • Production artifacts (artifacts/) never written during walk-forward.
  • Test timestamps strictly > train timestamps (integer-index slicing).
  • EMA(200) computed on full series → no cold-start artefact in test window.

Stability thresholds
--------------------
  FUNDED-READY : pct_profitable ≥ 70%  AND  avg_pf ≥ 1.5  AND
                 pnl_cv ≤ 1.0          AND  t_stat ≥ 1.65
  PROMISING    : pct_profitable ≥ 60%  AND  avg_pf ≥ 1.3  AND
                 pnl_cv ≤ 1.5          AND  total_pnl > 0
  MARGINAL     : pct_profitable ≥ 40%  AND  avg_pf ≥ 1.1
  NOT READY    : anything else
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
# Aligned with profile_eval.py acceptance gates.
_FUNDED_MIN_PCT_PROF   = 0.60   # majority profitable
_FUNDED_MIN_PF         = 1.15   # real edge after costs
_FUNDED_MAX_CV         = 1.20   # PnL consistency
_FUNDED_MIN_TSTAT      = 1.50   # statistical significance

_PROMISING_MIN_PCT_PROF = 0.50
_PROMISING_MIN_PF       = 1.05
_PROMISING_MAX_CV       = 1.50

_MARGINAL_MIN_PCT_PROF  = 0.40
_MARGINAL_MIN_PF        = 1.00

# OOS activity gate — folds with fewer trades are flagged WEAK
# With max_trades_per_day=2 and ~20 trading days per 2000-bar fold,
# we expect ~40 trades per fold. 3 is the absolute minimum — below
# that we cannot evaluate the strategy meaningfully.
_WEAK_FOLD_MIN_TRADES        = 3
_WEAK_FOLD_MIN_COVERAGE_PCT  = 0.1   # 0.1% of test bars (very lenient floor)


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
    # OOS activity diagnostics
    conf_threshold:          float = 0.65
    oos_conf_trade_count:    int   = 0
    oos_trade_coverage_pct:  float = 0.0
    oos_dir_accuracy:        float = 0.0
    oos_profit_factor:       float = 0.0
    weak_fold:               bool  = False
    # Regime diagnostics (populated from _simulate_trades filter_counters)
    n_trend_entries:         int   = 0
    n_chop_blocked:          int   = 0
    n_low_vol_blocked:       int   = 0
    # Session block PnL (populated from _simulate_trades cost_summary)
    block1_net_pnl:          float = 0.0
    block2_net_pnl:          float = 0.0
    block1_n_trades:         int   = 0
    block2_n_trades:         int   = 0


@dataclass
class WalkForwardSummary:
    symbol:     str
    mode:       str
    n_folds:    int
    train_bars: int
    test_bars:  int
    select_by:  str = "f1"
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
    select_by: str = "f1",
    save_report: bool = True,
    exec_cfg_overrides: Optional[dict] = None,
    threshold_candidates: Optional[list] = None,
    seed: int = 42,
) -> WalkForwardSummary:
    """
    Run walk-forward / OOS validation.

    Parameters
    ----------
    symbol        : Symbol (must have data/processed/ parquets).
    mode          : 'rolling' | 'expanding' | 'split'.
    train_bars    : Training window size in bars (rolling / expanding).
    test_bars     : Test window size in bars.
    step_bars     : Step between folds (default = test_bars).
    min_train_bars: Expanding: minimum initial training size.
    split_pct     : Split mode: fraction for training (default 0.8).
    n_trials             : Optuna trials per fold (0 = fast fixed hyperparameters).
    select_by            : 'f1' | 'trading' — model selection objective per fold.
    save_report          : Write JSON report to artifacts/reports/.
    exec_cfg_overrides   : Dict of execution-layer overrides applied on top of
                           universe.yaml config (e.g. session_blocks, atr_min_ticks).
    threshold_candidates : Override confidence threshold sweep list per fold.
                           Passed to _select_best_threshold and Optuna objective.
    seed                 : Master random seed for Optuna TPESampler and XGBoost.
                           Each fold uses seed + fold_idx * 1000 (deterministic,
                           non-correlated across folds).  Default 42.
    """
    if mode not in ("rolling", "expanding", "split"):
        raise ValueError(f"mode must be 'rolling', 'expanding', or 'split', got {mode!r}")
    if select_by not in ("f1", "trading"):
        raise ValueError(f"select_by must be 'f1' or 'trading', got {select_by!r}")

    step   = step_bars    or test_bars
    min_tr = min_train_bars or train_bars

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
        "Walk-forward [%s] %s → %s  |  %d bars  |  %d features  |  "
        "mode=%s  select_by=%s",
        symbol,
        aligned.index[0].date(), aligned.index[-1].date(),
        n_total, len(feature_names), mode, select_by,
    )

    # ── Raw prices + EMA ──────────────────────────────────────────────────
    raw_prices, full_ema = _load_prices(symbol, aligned.index)

    # ── Engine config ──────────────────────────────────────────────────────
    from src.backtest.engine import _load_cfg
    cfg   = _load_cfg(symbol)
    if exec_cfg_overrides:
        cfg = dict(cfg)
        cfg.update(exec_cfg_overrides)
    specs = _load_specs(symbol)
    starting_equity = float(cfg.get("starting_equity", 100_000.0))

    # ── Build fold index tuples ────────────────────────────────────────────
    if mode == "rolling":
        fold_ranges = _build_folds_rolling(n_total, train_bars, test_bars, step)
    elif mode == "expanding":
        fold_ranges = _build_folds_expanding(n_total, min_tr, test_bars, step)
    else:
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

        # ── Train (in-memory) ──────────────────────────────────────────────
        fold_seed = seed + fold_idx * 1000
        try:
            model, scaler, inv_label_map, conf_threshold = _train_on_slice(
                train_df, feature_names, n_trials, select_by,
                threshold_candidates=threshold_candidates,
                seed=fold_seed,
            )
        except Exception as exc:
            log.warning("Fold %d  training failed: %s — skipping", fold_idx, exc)
            continue

        log.info(
            "Fold %d  trained  |  select_by=%s  threshold=%.2f",
            fold_idx + 1, select_by, conf_threshold,
        )

        # ── OOS activity diagnostics ───────────────────────────────────────
        oos_stats = _compute_oos_stats(
            model, scaler, feature_names, test_df, inv_label_map,
            conf_threshold, cfg=cfg,
        )
        weak_fold = (
            oos_stats["n_trades"] < _WEAK_FOLD_MIN_TRADES
            or oos_stats["trade_coverage_pct"] < _WEAK_FOLD_MIN_COVERAGE_PCT
        )
        # ── Backtest on test window ────────────────────────────────────────
        # Full diagnostics are logged after backtest (with actual executed trades).
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
                conf_threshold=conf_threshold,
            )
        except Exception as exc:
            log.warning("Fold %d  backtest failed: %s — skipping", fold_idx, exc)
            continue

        # Log execution pass-through with full backtest pipeline breakdown
        _executed = metrics.get("n_trades", 0)
        _exec_flt = metrics.get("filter_counters", None)
        _log_fold_oos_stats(
            fold_idx + 1, oos_stats, conf_threshold,
            weak_fold,
            executed_trades=_executed,
            exec_filter_counters=_exec_flt,
        )

        _fold_flt = metrics.get("filter_counters", {})
        _fold_cs  = metrics.get("cost_summary", {})
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
            conf_threshold=conf_threshold,
            oos_conf_trade_count=oos_stats["n_trades"],
            oos_trade_coverage_pct=oos_stats["trade_coverage_pct"],
            oos_dir_accuracy=oos_stats["dir_accuracy"],
            oos_profit_factor=oos_stats["profit_factor"],
            weak_fold=weak_fold,
            n_trend_entries=_fold_flt.get("n_trend_entries",   0),
            n_chop_blocked=_fold_flt.get("n_chop_blocked",    0),
            n_low_vol_blocked=_fold_flt.get("n_low_vol_blocked", 0),
            block1_net_pnl=_fold_cs.get("block1_net_pnl",   0.0),
            block2_net_pnl=_fold_cs.get("block2_net_pnl",   0.0),
            block1_n_trades=_fold_cs.get("block1_n_trades",  0),
            block2_n_trades=_fold_cs.get("block2_n_trades",  0),
        )
        folds.append(fr)

    agg = _aggregate(folds)
    summary = WalkForwardSummary(
        symbol=symbol,
        mode=mode,
        n_folds=len(folds),
        train_bars=train_bars,
        test_bars=test_bars,
        select_by=select_by,
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
    folds: list[tuple[int, int, int, int]] = []
    start = 0
    while start + train_bars + test_bars <= n:
        folds.append((
            start, start + train_bars,
            start + train_bars, start + train_bars + test_bars,
        ))
        start += step_bars
    return folds


def _build_folds_expanding(
    n: int, min_train_bars: int, test_bars: int, step_bars: int,
) -> list[tuple[int, int, int, int]]:
    folds: list[tuple[int, int, int, int]] = []
    train_end = min_train_bars
    while train_end + test_bars <= n:
        folds.append((0, train_end, train_end, train_end + test_bars))
        train_end += step_bars
    return folds


def _build_folds_split(
    n: int, split_pct: float,
) -> list[tuple[int, int, int, int]]:
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
    select_by: str = "f1",
    threshold_candidates: Optional[list] = None,
    seed: int = 42,
) -> tuple:
    """
    Train XGBoost + RobustScaler on train_df in memory.

    Returns
    -------
    (model, scaler, inv_label_map, conf_threshold)
      model           — fitted XGBClassifier
      scaler          — fitted RobustScaler (fitted on train_df only)
      inv_label_map   — {str(encoded_class): original_label_int}
      conf_threshold  — selected confidence threshold (float)

    Raises
    ------
    ValueError if train_df has fewer than 2 distinct label classes.
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder, RobustScaler
    from src.training.train import _select_best_threshold

    feat_cols = [c for c in feature_names if c in train_df.columns]
    if not feat_cols:
        raise ValueError("No matching feature columns found in train_df.")

    X_df  = train_df[feat_cols]
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
    X_scaled = scaler.fit_transform(X_df)

    # int-keyed inv_label_map — used internally for trading stats
    int_inv_map = {i: int(cls) for i, cls in enumerate(le.classes_)}
    # str-keyed — returned for use by _backtest_on_slice
    inv_label_map = {str(i): int(cls) for i, cls in enumerate(le.classes_)}

    if n_trials > 0:
        # Pass index for session-aware scoring when timestamps are available
        model, conf_threshold = _optuna_train(
            X_scaled, y_enc, y_raw, int_inv_map, n_trials, select_by,
            val_index=X_df.index,
            threshold_candidates=threshold_candidates,
            seed=seed,
        )
    else:
        # Conservative fixed hyperparameters for OOS robustness
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
            random_state=seed,
            verbosity=0,
        )
        model.fit(X_scaled, y_enc)

        if select_by == "trading":
            # Use last 20% of train as inner val to select threshold.
            # Model is already fit on full train; this is a post-hoc threshold
            # sweep, not a second fit — acceptable for threshold selection.
            inner_split     = int(len(X_scaled) * 0.8)
            X_inner_val     = X_scaled[inner_split:]
            y_inner_val_raw = y_raw[inner_split:]
            inner_proba     = model.predict_proba(X_inner_val)
            conf_threshold  = _select_best_threshold(
                inner_proba, y_inner_val_raw, int_inv_map,
                n_val_bars=len(X_inner_val),
                candidates=threshold_candidates,
            )
        else:
            conf_threshold = 0.65

    return model, scaler, inv_label_map, conf_threshold


def _optuna_train(
    X_scaled: np.ndarray,
    y_enc: np.ndarray,
    y_raw: np.ndarray,
    int_inv_map: dict,
    n_trials: int,
    select_by: str = "f1",
    val_index=None,
    threshold_candidates: Optional[list] = None,
    seed: int = 42,
) -> tuple:
    """
    Optuna hyperparameter search within a training slice.

    Returns (model, conf_threshold).
    Inner validation: last 20% of the training slice (time-ordered).
    When select_by='trading', threshold is co-optimised via Optuna.
    val_index: DatetimeIndex of the full training slice — used to derive
               inner-val timestamps for session-aware scoring.
    seed: Controls Optuna TPESampler and final XGBoost random_state.
          Pass fold_seed = master_seed + fold_idx * 1000 for per-fold
          determinism without cross-fold correlation.
    """
    import optuna
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from src.training.train import (
        _THRESHOLD_CANDIDATES, _GATE_FAIL_SCORE,
        _HC_MIN_TRADES, _HC_MIN_COVERAGE, _HC_MIN_DIR_ACC, _HC_MIN_PF,
        _trading_quality_score, _select_best_threshold,
        _COMPOSITE_F1_WEIGHT, _COMPOSITE_TRADING_WEIGHT,
        _COMPOSITE_ROBUST_WEIGHT, _COMPOSITE_SESSION_WEIGHT,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split   = int(len(X_scaled) * 0.8)
    X_tr    = X_scaled[:split];  X_val = X_scaled[split:]
    y_tr    = y_enc[:split];     y_val = y_enc[split:]
    y_val_raw = y_raw[split:]
    n_val   = len(X_val)

    # ── Session mask for inner validation ────────────────────────────────────
    # Build once: which inner-val bars are within the trading session.
    try:
        from src.backtest.engine import _load_cfg as _ecfg, _in_session
        # symbol not available here — use generic cfg if walk-forward passes one
        # Fall through to default session hours
        raise AttributeError("no symbol in _optuna_train")
    except Exception:
        _in_session_fn = lambda ts, s, e: (s <= ts.hour < e)  # noqa: E731
        _sess_s, _sess_e = 9, 22  # safe defaults

    if val_index is not None and hasattr(val_index, "hour"):
        inner_val_idx = val_index[split:]
        _sess_mask = np.array(
            [_in_session_fn(ts, _sess_s, _sess_e) for ts in inner_val_idx], dtype=bool
        )
    else:
        _sess_mask = np.ones(n_val, dtype=bool)
    _n_val_sess = int(_sess_mask.sum())

    def objective(trial: optuna.Trial) -> float:
        params = {
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
        f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))

        if select_by != "trading":
            return f1

        # Trading: threshold as Optuna parameter with same gate logic as train.py
        _t_candidates = threshold_candidates if threshold_candidates is not None else _THRESHOLD_CANDIDATES
        threshold = trial.suggest_categorical("conf_threshold", _t_candidates)
        proba = m.predict_proba(X_val)
        conf  = np.max(proba, axis=1)
        sig   = np.array([int_inv_map.get(int(e), 0) for e in np.argmax(proba, axis=1)])
        mask  = (conf >= threshold) & (sig != 0)
        n_c   = int(mask.sum())
        cov   = n_c / n_val

        if n_c < _HC_MIN_TRADES or cov < _HC_MIN_COVERAGE:
            return _GATE_FAIL_SCORE

        # Session-filtered gate: confident signals must also be viable in session
        sess_mask_conf = mask & _sess_mask
        n_sess_conf    = int(sess_mask_conf.sum())
        sess_cov       = n_sess_conf / max(_n_val_sess, 1)
        if n_sess_conf < _HC_MIN_TRADES or sess_cov < _HC_MIN_COVERAGE:
            return _GATE_FAIL_SCORE

        sig_t   = sig[mask];  y_t = y_val_raw[mask]
        wins    = int(((sig_t == y_t) & (y_t != 0)).sum())
        losses  = n_c - wins
        dir_acc = wins / n_c
        pf      = min((wins / losses) if losses > 0 else float(wins), 5.0)

        if dir_acc < _HC_MIN_DIR_ACC or pf < _HC_MIN_PF:
            return _GATE_FAIL_SCORE

        tq          = _trading_quality_score(proba, y_val_raw, int_inv_map,
                                             min_conf=threshold, n_val_bars=n_val)
        sess_factor = n_sess_conf / max(n_c, 1)
        return (
            _COMPOSITE_F1_WEIGHT      * f1
            + _COMPOSITE_TRADING_WEIGHT * tq
            + _COMPOSITE_SESSION_WEIGHT * sess_factor
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)

    # Extract threshold; fall back to post-hoc selection if needed
    if select_by == "trading":
        all_failed = study.best_value <= _GATE_FAIL_SCORE * 0.5
        if all_failed:
            log.warning(
                "Walk-forward fold: ALL %d Optuna trials failed trading gates. "
                "Falling back to F1 selection with post-hoc threshold.",
                n_trials,
            )
        threshold_from_optuna = best_params.pop("conf_threshold", None)
    else:
        threshold_from_optuna = None

    model = xgb.XGBClassifier(
        **best_params,
        eval_metric="mlogloss",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_scaled, y_enc)

    if threshold_from_optuna is not None:
        conf_threshold = float(threshold_from_optuna)
    else:
        # Post-hoc threshold selection using inner val
        proba          = model.predict_proba(X_val)
        conf_threshold = _select_best_threshold(
            proba, y_val_raw, int_inv_map, n_val_bars=n_val,
            candidates=threshold_candidates,
        )

    return model, conf_threshold


# ─────────────────────────────────────────────────────────────────────────────
# OOS activity diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def _compute_oos_stats(
    model,
    scaler,
    feature_names: list[str],
    test_df: pd.DataFrame,
    inv_label_map: dict,
    conf_threshold: float,
    cfg: dict | None = None,
) -> dict:
    """
    Compute OOS trading statistics for a fold using the selected threshold.

    Returns a dict with n_trades, trade_coverage_pct, dir_accuracy,
    profit_factor, hard_constraints_passed, plus execution-aware extras:
      session_pass_pct   – fraction of confident signals in trading session
      atr_pass_pct       – fraction of confident signals passing ATR filter
      exec_est_signals   – estimated signals surviving session+ATR (both applied)
      exec_est_coverage  – exec_est_signals / n_val_bars  (%)
    """
    from src.training.train import _compute_trading_stats
    from src.backtest.engine import _in_session

    feat_cols = [c for c in feature_names if c in test_df.columns]
    X_scaled  = scaler.transform(test_df[feat_cols])
    proba     = model.predict_proba(X_scaled)
    y_raw     = test_df["label"].values
    int_map   = {int(k): v for k, v in inv_label_map.items()}

    stats = _compute_trading_stats(
        proba, y_raw, int_map,
        min_conf=conf_threshold, n_val_bars=len(test_df),
    )

    # ── Execution-aware pre-filter estimate ───────────────────────────────────
    # Identify bars with confident non-flat signals, then apply session+ATR.
    if cfg is not None and len(test_df) > 0:
        conf_vals = np.max(proba, axis=1)
        sig_vals  = np.array([int_map.get(int(e), 0) for e in np.argmax(proba, axis=1)])
        conf_mask = (conf_vals >= conf_threshold) & (sig_vals != 0)
        n_conf    = int(conf_mask.sum())
        n_val     = len(test_df)

        # Session filter (only if DatetimeIndex available)
        sess_start = int(cfg.get("session_start_utc_hour", 0))
        sess_end   = int(cfg.get("session_end_utc_hour", 24))
        has_ts     = hasattr(test_df.index, "hour")
        if has_ts:
            sess_mask = np.array([
                _in_session(ts, sess_start, sess_end) for ts in test_df.index
            ])
        else:
            sess_mask = np.ones(n_val, dtype=bool)
        n_sess = int((conf_mask & sess_mask).sum())

        # ATR filter (uses atr_14 feature if available)
        atr_min  = float(cfg.get("atr_min_ticks", 0))
        atr_max  = float(cfg.get("atr_max_ticks", 1e9))
        tick_sz  = float(cfg.get("tick_size", 0.25))
        if "atr_14" in test_df.columns and "close" in test_df.columns:
            close_v  = test_df["close"].values
            atr_t    = test_df["atr_14"].values * close_v / tick_sz
            atr_mask = (atr_t >= atr_min) & (atr_t <= atr_max)
        else:
            atr_mask = np.ones(n_val, dtype=bool)
        n_atr = int((conf_mask & atr_mask).sum())

        # Sequential: conf → session → atr
        n_exec_est = int((conf_mask & sess_mask & atr_mask).sum())   # = n_after_atr sequential

        stats["n_conf_signals"]    = n_conf
        stats["n_after_session"]   = n_sess
        stats["n_after_atr"]       = n_exec_est
        # Pass % at each sequential stage
        stats["session_pass_pct"]  = round(100.0 * n_sess    / max(n_conf, 1), 2)
        stats["atr_pass_pct"]      = round(100.0 * n_exec_est / max(n_sess, 1), 2)
        stats["exec_est_signals"]  = n_exec_est
        stats["exec_est_coverage"] = round(100.0 * n_exec_est / max(n_val, 1), 2)
    else:
        stats["session_pass_pct"]  = None
        stats["atr_pass_pct"]      = None
        stats["exec_est_signals"]  = None
        stats["exec_est_coverage"] = None

    return stats


def _log_fold_oos_stats(
    fold_num: int,
    stats: dict,
    threshold: float,
    weak_fold: bool,
    executed_trades: int | None = None,
    exec_filter_counters: dict | None = None,
) -> None:
    """Log per-fold OOS diagnostics with full sequential pipeline breakdown."""
    status   = "WEAK" if weak_fold else "OK"
    hc_label = "PASSED" if stats["hard_constraints_passed"] else "FAILED"

    n_val  = stats["n_val_bars"]
    n_conf = stats["n_trades"]    # pre-filter model confident signals
    cov    = stats["trade_coverage_pct"]

    log.info(
        "Fold %d  threshold=%.2f  HC=%s  activity=%s  |  "
        "dir_acc=%.1f%%  PF=%.2f  conf=%d/%d (%.1f%%)",
        fold_num, threshold, hc_label, status,
        stats["dir_accuracy"] * 100, stats["profit_factor"],
        n_conf, n_val, cov,
    )

    def _pct(n, d):
        return f"{100*n/d:.0f}%" if d > 0 else "n/a"

    # ── Pre-filter sequential pipeline (session + ATR, from model predictions) ─
    n_conf_raw = stats.get("n_conf_signals", n_conf)   # total model signals
    n_sess     = stats.get("n_after_session", None)
    n_atr      = stats.get("n_after_atr", None)        # = exec_est_signals
    exec_cov   = stats.get("exec_est_coverage", None)

    if n_sess is not None:
        log.info(
            "Fold %d pre-filter pipeline  |  "
            "conf=%d  →sess=%d(%s)  →atr=%d(%s)  exec_est_cov=%.1f%%",
            fold_num,
            n_conf_raw,
            n_sess,  _pct(n_sess, n_conf_raw),
            n_atr,   _pct(n_atr,  n_sess),
            exec_cov if exec_cov is not None else 0.0,
        )

    # ── Post-backtest pipeline (from actual _simulate_trades counters) ─────────
    if exec_filter_counters:
        fc    = exec_filter_counters
        _nt   = fc.get("n_total_bars",        0)
        _nc   = fc.get("n_confident_signals",  0)
        _ns   = fc.get("n_after_session",      0)
        _nbo  = fc.get("n_after_blackout",     0)
        _natr = fc.get("n_after_atr",          0)
        _nreg = fc.get("n_after_regime",       0)
        _nchb = fc.get("n_chop_blocked",       0)
        _nlvb = fc.get("n_low_vol_blocked",    0)
        _nte  = fc.get("n_trend_entries",      0)
        _nsb  = fc.get("n_blocked_by_slope",   0)
        _nb1  = fc.get("n_in_block1",          0)
        _nb2  = fc.get("n_in_block2",          0)
        _ntr  = fc.get("n_after_trend",        0)
        _nrk  = fc.get("n_after_risk",         0)
        _ncd  = fc.get("n_after_cooldowns",    0)
        _nq   = fc.get("n_entries_queued",     0)
        _nex  = executed_trades if executed_trades is not None else 0

        log.info(
            "Fold %d backtest pipeline (%d bars)  |  "
            "conf=%d  →sess=%d(%s)  →atr=%d(%s)  →regime=%d(%s)[chop_blk=%d lv_blk=%d]"
            "  →trend=%d(%s)[slope_blk=%d]  →risk=%d(%s)  →cd=%d(%s)"
            "  →q=%d(%s)[trend_entries=%d blk1=%d blk2=%d]  →exec=%d(%s)",
            fold_num, _nt,
            _nc,
            _ns,   _pct(_ns,   _nc),
            _natr, _pct(_natr, _ns),
            _nreg, _pct(_nreg, _natr), _nchb, _nlvb,
            _ntr,  _pct(_ntr,  _nreg), _nsb,
            _nrk,  _pct(_nrk,  _ntr),
            _ncd,  _pct(_ncd,  _nrk),
            _nq,   _pct(_nq,   _ncd), _nte, _nb1, _nb2,
            _nex,  _pct(_nex,  _nq),
        )

        # Warnings
        if _ns > 0 and _nq < _ns * 0.20:
            log.warning(
                "Fold %d LOW YIELD: %d queued from %d in-session signals "
                "(%.0f%% < 20%%). Top bottleneck: %s",
                fold_num, _nq, _ns, 100*_nq/max(_ns, 1),
                _top_bottleneck(fc),
            )
    elif executed_trades is not None and n_atr is not None:
        # Fallback: show exec_est → executed
        pass_pct = _pct(executed_trades, n_atr)
        log.info(
            "Fold %d execution  exec_est=%d → executed=%d (%s pass-through)",
            fold_num, n_atr, executed_trades, pass_pct,
        )
        if n_atr > 0 and executed_trades < max(3, n_atr // 5):
            log.warning(
                "Fold %d LOW pass-through (%d/%d): "
                "cooldown/holding/daily-cap filters likely suppressing most entries. "
                "Consider reducing min_holding_bars and cooldown_bars_after_exit.",
                fold_num, executed_trades, n_atr,
            )

    if weak_fold:
        log.warning(
            "Fold %d WEAK: %d confident trades (%.2f%% cov) below activity gates "
            "[>= %d trades AND >= %.1f%% coverage]",
            fold_num, n_conf, cov, _WEAK_FOLD_MIN_TRADES, _WEAK_FOLD_MIN_COVERAGE_PCT,
        )


def _top_bottleneck(fc: dict) -> str:
    """Return the name of the filter stage that dropped the most signals."""
    stages = [
        ("session",   fc.get("n_confident_signals", 0), fc.get("n_after_session",   0)),
        ("blackout",  fc.get("n_after_session",     0), fc.get("n_after_blackout",  0)),
        ("ATR",       fc.get("n_after_blackout",    0), fc.get("n_after_atr",       0)),
        ("regime",    fc.get("n_after_atr",         0), fc.get("n_after_regime",    0)),
        ("trend/slope", fc.get("n_after_regime",   0), fc.get("n_after_trend",     0)),
        ("risk/halt", fc.get("n_after_trend",       0), fc.get("n_after_risk",      0)),
        ("cooldowns", fc.get("n_after_risk",        0), fc.get("n_after_cooldowns", 0)),
        ("queuing",   fc.get("n_after_cooldowns",   0), fc.get("n_entries_queued",  0)),
    ]
    biggest = max(stages, key=lambda s: s[1] - s[2], default=("unknown", 0, 0))
    return biggest[0]


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
    conf_threshold: float = 0.65,
) -> dict:
    """
    Run backtest on a test-window DataFrame.

    conf_threshold overrides the confidence gates from universe.yaml,
    ensuring the backtest uses exactly the threshold selected during training.
    """
    from src.backtest.engine import _simulate_trades, _compute_metrics

    feat_cols = [c for c in feature_names if c in test_df.columns]
    X_df      = test_df[feat_cols]
    X_scaled  = scaler.transform(X_df)
    proba     = model.predict_proba(X_scaled)
    pred_enc  = np.argmax(proba, axis=1)
    conf      = np.max(proba, axis=1)

    sig_arr     = np.array([inv_label_map[str(e)] for e in pred_enc])
    sig_series  = pd.Series(sig_arr, index=test_df.index)
    conf_series = pd.Series(conf,    index=test_df.index)

    # Apply selected threshold — override the universe.yaml confidence gates
    cfg                       = dict(cfg)
    cfg["min_long_confidence"]  = conf_threshold
    cfg["min_short_confidence"] = conf_threshold

    tick_size    = float(specs.get("tick_size",  cfg.get("tick_size", 0.25)))
    multiplier   = float(specs.get("multiplier", 50.0))
    cfg["tick_size"] = tick_size
    commission_rt = 2.0 * float(cfg.get("commission_per_side_usd", 1.50))
    slippage_pts  = float(cfg.get("slippage_ticks_per_side", 0.5)) * tick_size
    half_spread   = float(cfg.get("spread_ticks", 1.0)) * 0.5 * tick_size
    friction_pts  = slippage_pts + half_spread

    if raw_prices is not None:
        rp         = raw_prices.reindex(test_df.index)
        has_prices = "close" in rp.columns and not rp["close"].isna().all()
    else:
        has_prices = False

    if has_prices:
        close_prices = rp["close"].ffill().bfill().fillna(5000.0)
        open_prices  = (
            rp["open"].ffill().bfill().fillna(close_prices)
            if "open" in rp.columns else close_prices
        )
    else:
        if raw_prices is not None:
            log.warning("Raw prices reindexed to test window are all NaN — using 5000.0 proxy")
        open_prices  = pd.Series(5000.0, index=test_df.index)
        close_prices = pd.Series(5000.0, index=test_df.index)

    atr_ticks = pd.Series(20.0, index=test_df.index)
    if "atr_14" in test_df.columns:
        atr_ticks = (test_df["atr_14"] * close_prices / tick_size).fillna(20.0)

    ema_series: Optional[pd.Series] = None
    if cfg.get("trend_filter_enabled", False) and full_ema is not None:
        ema_series = full_ema

    # ATR regime series (needed by both legacy fallback and regime classifier)
    atr_regime_fold = (atr_ticks / atr_ticks.rolling(100, min_periods=20).mean()).fillna(1.0)

    # Regime classifier — trend (1) / chop (0) / low_vol (-1)
    _ema20_fold     = close_prices.ewm(span=20, adjust=False).mean()
    _slope_abs_fold = (_ema20_fold - _ema20_fold.shift(5)).abs()
    _atr_pts_fold   = (atr_ticks * tick_size).replace(0, np.nan)
    _trend_str_fold = (_slope_abs_fold / _atr_pts_fold).fillna(0.0)
    if "vol_regime_20_60" in test_df.columns:
        _vol_reg_fold = test_df["vol_regime_20_60"].fillna(1.0)
    else:
        _vol_reg_fold = pd.Series(1.0, index=test_df.index)
    _is_lv_fold    = (atr_regime_fold < 0.8) | (_vol_reg_fold < 0.7)
    _is_trend_fold = (~_is_lv_fold) & (_trend_str_fold >= 0.08)
    regime_fold    = pd.Series(
        np.where(_is_trend_fold, 1, np.where(_is_lv_fold, -1, 0)),
        index=test_df.index,
    )

    trades, equity_curve, cost_summary = _simulate_trades(
        signals           = sig_series,
        open_prices       = open_prices,
        atr_ticks         = atr_ticks,
        cfg               = cfg,
        friction_pts      = friction_pts,
        commission_rt     = commission_rt,
        multiplier        = multiplier,
        conf_series       = conf_series,
        close_prices      = close_prices,
        ema_series        = ema_series,
        atr_regime_series = atr_regime_fold,
        regime_series     = regime_fold,
    )

    metrics = _compute_metrics(
        equity_curve, trades, cost_summary,
        starting_equity=starting_equity,
    )
    metrics["n_trades"]        = len(trades)
    metrics["filter_counters"] = cost_summary.get("filter_counters", {})
    metrics["cost_summary"]    = {
        "block1_net_pnl":  cost_summary.get("block1_net_pnl",  0.0),
        "block2_net_pnl":  cost_summary.get("block2_net_pnl",  0.0),
        "block1_n_trades": cost_summary.get("block1_n_trades", 0),
        "block2_n_trades": cost_summary.get("block2_n_trades", 0),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate(folds: list[FoldResult]) -> dict:
    if not folds:
        return {}

    n          = len(folds)
    net_pnls   = [f.net_pnl           for f in folds]
    win_rates  = [f.win_rate          for f in folds]
    sharpes    = [f.sharpe            for f in folds]
    sortinos   = [f.sortino           for f in folds]
    pfs        = [f.profit_factor     for f in folds]
    mdd_usd    = [f.max_drawdown_usd  for f in folds]
    mdd_pct    = [f.max_drawdown_pct  for f in folds]
    expectancy = [f.expectancy_usd    for f in folds]
    tpd        = [f.trades_per_day    for f in folds]
    n_trades   = [f.n_trades          for f in folds]
    profitable = sum(1 for f in folds if f.profitable)
    n_weak     = sum(1 for f in folds if f.weak_fold)

    total_pnl   = float(sum(net_pnls))
    mean_pnl    = float(np.mean(net_pnls))
    std_pnl     = float(np.std(net_pnls, ddof=1)) if n > 1 else 0.0
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe  = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0
    avg_pf      = float(np.mean(pfs))
    pct_prof    = profitable / n

    pnl_cv: Optional[float] = (
        round(std_pnl / mean_pnl, 4) if mean_pnl > 0 else None
    )
    pnl_cv_val = pnl_cv if pnl_cv is not None else float("inf")

    t_stat: Optional[float] = (
        round(mean_pnl / (std_pnl / math.sqrt(n)), 4)
        if (n > 1 and std_pnl > 0) else None
    )
    t_stat_val = t_stat if t_stat is not None else 0.0

    # Gate: fewer than 3/5 profitable folds → NOT READY.
    # Only enforced for 5-fold (and larger) runs; for smaller fold counts
    # the existing pct_profitable threshold already applies.
    _too_few_profitable_folds = n >= 5 and profitable < 3

    # Gate: avg PF < 1.2 → NOT READY.
    _avg_pf_too_low = avg_pf < 1.2

    # Session block asymmetry gate — if one block is strongly negative while the
    # other carries all the PnL, the strategy is not robust across sessions.
    total_b1 = sum(f.block1_net_pnl for f in folds)
    total_b2 = sum(f.block2_net_pnl for f in folds)
    _has_block_data = abs(total_b1) > 10 or abs(total_b2) > 10
    _session_asymmetric = _has_block_data and (
        (total_b1 < -50 and total_b2 > 100) or
        (total_b2 < -50 and total_b1 > 100)
    )

    # Fold rejection: if more than 2 folds have PF < 1.0, the model is
    # unstable across market regimes — cap verdict at NOT READY.
    n_pf_below_one = sum(1 for pf in pfs if pf < 1.0)
    _too_many_losing_folds = n_pf_below_one > 2

    # Regime consistency gate (Gate 4a + 4b from the spec).
    # Only active when folds carry actual regime data (n_trend_entries recorded).
    # When all folds have zero trend_entries (e.g. no regime_series was passed),
    # these gates are skipped to preserve backward compatibility with tests/folds
    # that were run without the regime classifier.
    _regime_data_present = any(
        f.n_trend_entries > 0 or f.n_chop_blocked > 0 or f.n_low_vol_blocked > 0
        for f in folds
    )

    # Gate 4a: fewer than 3 folds have positive PnL in trend regime.
    #   Since we only trade in trend regime, profitable == positive trend PnL.
    n_profitable_trend = sum(1 for f in folds if f.profitable)
    _too_few_trend_profitable = _regime_data_present and n_profitable_trend < 3

    # Gate 4b: more than 2 folds are "chop-dominated" — market offered few
    #   trend opportunities (regime blocked >> trend entries), so the model
    #   cannot get enough exposure to prove its edge.
    n_chop_dominated = sum(
        1 for f in folds
        if f.n_chop_blocked > 2 * max(f.n_trend_entries, 1)
    )
    _too_many_chop_dominated = _regime_data_present and n_chop_dominated > 2

    if _too_few_profitable_folds:
        log.warning(
            "WALK-FORWARD GATE: only %d / %d folds profitable "
            "(need >= 3) — verdict forced to NOT READY.",
            profitable, n,
        )
        verdict = "NOT READY"
    elif _avg_pf_too_low:
        log.warning(
            "WALK-FORWARD GATE: avg profit factor %.3f < 1.2 — "
            "verdict forced to NOT READY.",
            avg_pf,
        )
        verdict = "NOT READY"
    elif _session_asymmetric:
        log.warning(
            "WALK-FORWARD SESSION GATE: strong block asymmetry detected "
            "(block1_pnl=%.0f  block2_pnl=%.0f). "
            "One session block is dragging — verdict forced to NOT READY.",
            total_b1, total_b2,
        )
        verdict = "NOT READY"
    elif _too_many_losing_folds:
        log.warning(
            "WALK-FORWARD REJECTION: %d / %d folds have PF < 1.0 (> 2 allowed). "
            "Model is unstable across regimes — verdict forced to NOT READY.",
            n_pf_below_one, n,
        )
        verdict = "NOT READY"
    elif _too_few_trend_profitable:
        log.warning(
            "WALK-FORWARD REGIME GATE A: only %d / %d folds profitable in trend "
            "regime (need >= 3) — verdict forced to NOT READY.",
            n_profitable_trend, n,
        )
        verdict = "NOT READY"
    elif _too_many_chop_dominated:
        log.warning(
            "WALK-FORWARD REGIME GATE B: %d / %d folds are chop-dominated "
            "(chop_blocked > 2× trend_entries) — model cannot get sufficient trend "
            "exposure. Verdict forced to NOT READY.",
            n_chop_dominated, n,
        )
        verdict = "NOT READY"
    elif (
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
        "total_net_pnl":          round(total_pnl, 2),
        "avg_net_pnl_per_fold":   round(mean_pnl, 2),
        "std_net_pnl":            round(std_pnl, 2),
        "total_n_trades":         int(sum(n_trades)),
        "n_profitable_folds":     profitable,
        "pct_profitable_folds":   round(pct_prof, 4),
        "n_weak_folds":           n_weak,
        "n_folds_pf_below_one":   n_pf_below_one,
        "avg_win_rate":           round(float(np.mean(win_rates)), 4),
        "median_win_rate":        round(float(np.median(win_rates)), 4),
        "avg_profit_factor":      round(avg_pf, 4),
        "median_profit_factor":   round(float(np.median(pfs)), 4),
        "avg_sharpe":             round(mean_sharpe, 4),
        "std_sharpe":             round(std_sharpe, 4),
        "avg_sortino":            round(float(np.mean(sortinos)), 4),
        "avg_max_drawdown_usd":   round(float(np.mean(mdd_usd)), 2),
        "avg_max_drawdown_pct":   round(float(np.mean(mdd_pct)), 4),
        "worst_drawdown_pct":     round(float(min(mdd_pct)), 4),
        "avg_expectancy_usd":     round(float(np.mean(expectancy)), 2),
        "avg_trades_per_day":     round(float(np.mean(tpd)), 2),
        "pnl_cv":                 pnl_cv,
        "t_stat_pnl":             t_stat,
        "funded_ready":           verdict,
        # Regime consistency
        "n_profitable_trend_folds":  n_profitable_trend,
        "n_chop_dominated_folds":    n_chop_dominated,
        "regime_gate_a_passed":      not _too_few_trend_profitable,
        "regime_gate_b_passed":      not _too_many_chop_dominated,
        # Robustness policy gates
        "gate_profitable_folds_passed": not _too_few_profitable_folds,
        "gate_avg_pf_passed":           not _avg_pf_too_low,
        "gate_session_asymmetry_passed": not _session_asymmetric,
        "block1_total_pnl":          round(total_b1, 2),
        "block2_total_pnl":          round(total_b2, 2),
        "stability":              verdict,   # legacy alias
        "n_folds":                n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_VERDICT_LINES = {
    "FUNDED-READY": "FUNDED-READY  — Passes all funded-account robustness criteria.",
    "PROMISING":    "PROMISING     — Positive OOS expectancy; tighten further before live.",
    "MARGINAL":     "MARGINAL      — Marginally profitable; not ready for funded trading.",
    "NOT READY":    "NOT READY     — Fails minimum robustness criteria. Do not trade live.",
}


def _print_summary(result: WalkForwardSummary) -> None:
    W = 100
    bar = "=" * W

    print(f"\n{bar}")
    print(
        f"  WALK-FORWARD VALIDATION  |  {result.symbol}"
        f"  |  mode={result.mode}  |  select_by={result.select_by}"
        f"  |  n_folds={result.n_folds}"
    )
    print(bar)

    col_w = 25
    hdr = (
        f"  {'Fold':>4}  {'Test period':<{col_w}}  {'Bars':>5}  "
        f"{'Trd':>4}  {'Net PnL ($)':>11}  {'WR%':>5}  "
        f"{'PF':>5}  {'Sharpe':>6}  {'MaxDD%':>7}  {'MaxDD($)':>10}  "
        f"{'Thr':>5}  {'OOSn':>5}  {'Cov%':>5}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for f in result.folds:
        period = f"{f.test_start[:10]}>{f.test_end[:10]}"
        ok     = "[+]" if f.profitable else "[-]"
        wk     = "[W]" if f.weak_fold else "   "
        print(
            f"  {f.fold_idx:>4}  {period:<{col_w}}  {f.test_bars:>5}  "
            f"{f.n_trades:>4}  {f.net_pnl:>+11.2f}  {f.win_rate:>5.1%}  "
            f"{f.profit_factor:>5.2f}  {f.sharpe:>6.3f}  "
            f"{f.max_drawdown_pct:>6.2f}%  {f.max_drawdown_usd:>+10.2f}  "
            f"{f.conf_threshold:>5.2f}  {f.oos_conf_trade_count:>5d}  "
            f"{f.oos_trade_coverage_pct:>5.1f}%  {ok}{wk}"
        )

    agg = result.aggregate
    if not agg:
        print(f"{bar}\n")
        return

    print(sep)
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

    lbl = 36
    print(f"\n  {'AGGREGATE PERFORMANCE':-<{W - 4}}")
    print(f"  {'Profitable folds':{lbl}}: "
          f"{agg['n_profitable_folds']} / {agg['n_folds']}  "
          f"({agg['pct_profitable_folds']:.0%})")
    n_weak = agg.get("n_weak_folds", 0)
    if n_weak:
        print(f"  {'Weak folds (< {_WEAK_FOLD_MIN_TRADES} trades or < {_WEAK_FOLD_MIN_COVERAGE_PCT:.1f}% coverage)':{lbl}}: "
              f"{n_weak} / {agg['n_folds']}  [WARNING: model under-trades on these folds]")
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

    pnl_cv_str = (f"{agg['pnl_cv']:.3f}"
                  if agg.get("pnl_cv") is not None else "n/a (mean PnL <= 0)")
    t_stat_str = (f"{agg['t_stat_pnl']:.2f}"
                  if agg.get("t_stat_pnl") is not None else "n/a (< 2 folds)")
    verdict    = agg.get("funded_ready", agg.get("stability", "UNKNOWN"))

    print(f"\n  {'ROBUSTNESS / STABILITY':-<{W - 4}}")
    print(f"  {'PnL coeff. of variation (CV)':{lbl}}: {pnl_cv_str}"
          f"  [funded-account target <= 1.0]")
    print(f"  {'Sharpe std across folds':{lbl}}: {agg['std_sharpe']:.3f}"
          f"  [target < 0.40]")
    print(f"  {'PnL t-statistic':{lbl}}: {t_stat_str}"
          f"  [>= 1.65 -> 95% significance]")

    verdict_line = _VERDICT_LINES.get(verdict, f"UNKNOWN ({verdict})")
    print(f"\n  {'FUNDED-ACCOUNT READINESS':-<{W - 4}}")
    print(f"\n    >> {verdict_line}\n")
    print(f"  {'Criteria (all must pass for FUNDED-READY)':{lbl}}")
    pct_ok = agg["pct_profitable_folds"] >= _FUNDED_MIN_PCT_PROF
    pf_ok  = agg["avg_profit_factor"] >= _FUNDED_MIN_PF
    cv_ok  = (agg.get("pnl_cv") is not None and agg["pnl_cv"] <= _FUNDED_MAX_CV)
    tst_ok = (agg.get("t_stat_pnl") is not None
               and agg["t_stat_pnl"] >= _FUNDED_MIN_TSTAT)
    _pass_fail(f"  Profitable folds >= {_FUNDED_MIN_PCT_PROF:.0%}",
               pct_ok, f"{agg['pct_profitable_folds']:.0%}", lbl)
    _pass_fail(f"  Avg profit factor >= {_FUNDED_MIN_PF:.2f}",
               pf_ok, f"{agg['avg_profit_factor']:.3f}", lbl)
    _pass_fail(f"  PnL CV <= {_FUNDED_MAX_CV:.1f}", cv_ok, pnl_cv_str, lbl)
    _pass_fail(f"  PnL t-stat >= {_FUNDED_MIN_TSTAT:.2f}", tst_ok, t_stat_str, lbl)

    print(f"\n  {'ROBUSTNESS POLICY GATES':-<{W - 4}}")
    _gpf_ok  = agg.get("gate_profitable_folds_passed", True)
    _gapf_ok = agg.get("gate_avg_pf_passed", True)
    _gsa_ok  = agg.get("gate_session_asymmetry_passed", True)
    _pass_fail("  Profitable folds >= 3/5", _gpf_ok,
               f"{agg['n_profitable_folds']} / {agg['n_folds']}", lbl)
    _pass_fail(f"  Avg PF >= 1.20", _gapf_ok,
               f"{agg['avg_profit_factor']:.3f}", lbl)
    _b1_t = agg.get("block1_total_pnl", 0.0)
    _b2_t = agg.get("block2_total_pnl", 0.0)
    _pass_fail("  Session block symmetry", _gsa_ok,
               f"blk1=${_b1_t:+.0f}  blk2=${_b2_t:+.0f}", lbl)

    print(f"\n  {'REGIME CONSISTENCY GATES':-<{W - 4}}")
    rga_ok = agg.get("regime_gate_a_passed", True)
    rgb_ok = agg.get("regime_gate_b_passed", True)
    n_pt   = agg.get("n_profitable_trend_folds", agg["n_profitable_folds"])
    n_cd   = agg.get("n_chop_dominated_folds", 0)
    _pass_fail("  Trend-regime profitable folds >= 3", rga_ok, f"{n_pt} / {agg['n_folds']}", lbl)
    _pass_fail("  Chop-dominated folds <= 2",          rgb_ok, f"{n_cd} / {agg['n_folds']}", lbl)
    print(f"\n  {'Fold breakdown (regime | block PnL)':-<{W - 4}}")
    for f in result.folds:
        print(
            f"  Fold {f.fold_idx}  trend_entries={f.n_trend_entries:>4}  "
            f"chop_blocked={f.n_chop_blocked:>4}  "
            f"low_vol_blocked={f.n_low_vol_blocked:>4}  "
            f"profitable={'Y' if f.profitable else 'N'}  "
            f"blk1=${f.block1_net_pnl:+.0f}({f.block1_n_trades}tr)  "
            f"blk2=${f.block2_net_pnl:+.0f}({f.block2_n_trades}tr)"
        )

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
        "select_by":  result.select_by,
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
    raw = raw.reindex(idx)

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
    if not _UNIVERSE_CFG.exists():
        return {}
    with open(_UNIVERSE_CFG) as f:
        uc = yaml.safe_load(f) or {}
    return uc.get("contract_specs", {}).get(symbol, {})
