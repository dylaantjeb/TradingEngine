"""
Model training pipeline.

Steps
──────
1. Load processed features + labels (parquet).
2. Align and clean (drop rows without overlap).
3. Time-based train / validation split (no shuffling!).
4. Scale features with RobustScaler (persisted for inference).
5. Optuna hyperparameter search — two selection modes:
     f1       (default): maximise macro-F1 on val split.
     trading  (new):     two-level gate then balanced composite score.
6. Retrain best model on full train set.
7. Persist model, scaler, and feature schema.

How --select-by trading works
──────────────────────────────
Each Optuna trial is evaluated in two stages:

  STAGE 1 — Inline hard gates (enforced directly in the objective).
    These are computed in the objective function itself, NOT delegated
    to any helper. If either gate fails the trial receives _GATE_FAIL_SCORE
    (-100.0) which is lower than any passing model can achieve, so Optuna
    will never select it regardless of F1.

      Gate 1: confident_trade_count >= 50
      Gate 2: trade_coverage >= 3%  (confident_trades / total_val_bars)

    "Confident trade" = model confidence >= _TRADING_SCORE_MIN_CONF (0.65)
                        AND predicted direction != flat.

  STAGE 2 — Balanced composite score (only reached when both gates pass).
    composite = 0.40 * macro_F1  +  0.60 * trading_quality

    trading_quality is a five-component score bounded to [-10, 1]:
      0.30 * directional_accuracy_confident
      0.25 * normalized_profit_factor
      0.20 * trade_coverage_score
      0.15 * activity_score
      0.10 * pnl_per_trade_score

    Secondary hard constraints (checked inside trading_quality scorer):
      • profit_factor >= 1.10   (else tq = _HC_PENALTY_MEDIUM = -3.0)
      • directional_accuracy >= 52%   (else tq = _HC_PENALTY_MEDIUM = -3.0)

After study.optimize(), if best_value <= _GATE_FAIL_SCORE + 1 it means
ALL trials failed the hard gates — a warning is logged and the selected
model will still be saved, but the caller is told to use --select-by f1
instead or to lower the confidence threshold / acquire more data.

Artifacts
──────────
  artifacts/models/<SYM>_xgb_best.joblib
  artifacts/scalers/<SYM>_scaler.joblib
  artifacts/schema/<SYM>_features.json
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 200
VAL_FRACTION   = 0.20

# Minimum confidence for a prediction to count as a "trade".
# Matches the confidence gate in universe.yaml.
_TRADING_SCORE_MIN_CONF = 0.65

# Composite objective weights
_COMPOSITE_F1_WEIGHT      = 0.40
_COMPOSITE_TRADING_WEIGHT = 0.60

# ── Hard-gate thresholds ──────────────────────────────────────────────────────
# These are enforced at TWO levels:
#   1. Inline in the Optuna objective (STAGE 1 gates) → _GATE_FAIL_SCORE
#   2. Inside _compute_trading_stats (secondary gates)  → penalty values
_HC_MIN_TRADES   = 50      # minimum confident non-flat predictions on val set
_HC_MIN_COVERAGE = 0.03    # minimum fraction of val bars that are "trades" (3%)
_HC_MIN_PF       = 1.10    # minimum profit factor (secondary gate)
_HC_MIN_DIR_ACC  = 0.52    # minimum directional accuracy (secondary gate)

# Sentinel score returned by the Optuna objective when STAGE 1 gates fail.
# Must be sufficiently negative that no passing model can match it.
# The best possible composite for a passing model is ~0.40*1 + 0.60*1 = 1.0
# so -100 guarantees Optuna will always prefer any passing model.
_GATE_FAIL_SCORE = -100.0

# Penalty values returned by _compute_trading_stats for secondary gate failures.
# Negative so composite objective can distinguish them from passing models.
_HC_PENALTY_SEVERE = -10.0   # trade_count < 50 or coverage < 3% (should not
                               # reach here since STAGE 1 gates fire first)
_HC_PENALTY_MEDIUM = -3.0    # pf < 1.10 or dir_accuracy < 52%

# Sub-component weights (sum to 1.0)
_W_DIR_ACC  = 0.30
_W_NORM_PF  = 0.25
_W_COVERAGE = 0.20
_W_ACTIVITY = 0.15
_W_PPT      = 0.10


def train(
    symbol: str,
    n_trials: int = 20,
    select_by: str = "f1",
) -> None:
    """
    Full training pipeline for `symbol`.

    Parameters
    ----------
    symbol     : Symbol identifier (e.g. "ES").
    n_trials   : Number of Optuna trials.
    select_by  : 'f1' (maximise macro-F1) or 'trading' (balanced composite
                 with hard gates for trade count and coverage).
    """
    if select_by not in ("f1", "trading"):
        log.error("select_by must be 'f1' or 'trading', got %r", select_by)
        sys.exit(1)

    try:
        import joblib
        import optuna
        import xgboost as xgb
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import RobustScaler
    except ImportError as e:
        log.error("Missing dependency: %s  –  run:  pip install -r requirements.txt", e)
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Load data ─────────────────────────────────────────────────────────────
    feat_path = Path(f"data/processed/{symbol}_features.parquet")
    lbl_path  = Path(f"data/processed/{symbol}_labels.parquet")

    for p in (feat_path, lbl_path):
        if not p.exists():
            log.error(
                "File not found: %s\n"
                "Run:  python -m src.cli build-dataset --symbol %s --input data/raw/%s_M1.csv",
                p, symbol, symbol,
            )
            sys.exit(1)

    try:
        features  = pd.read_parquet(feat_path, engine="pyarrow")
        labels_df = pd.read_parquet(lbl_path,  engine="pyarrow")
    except ImportError:
        log.error("pyarrow required:  pip install pyarrow")
        sys.exit(1)

    # Align on index
    idx = features.index.intersection(labels_df.index)
    X = features.loc[idx]
    y = labels_df.loc[idx, "label"]

    # Map labels {-1, 0, 1} → {0, 1, 2} for XGBoost multi-class
    label_map     = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y_enc         = y.map(label_map).astype(int)

    n = len(X)
    log.info("Dataset: %d rows × %d features", n, X.shape[1])
    log.info("Label distribution: %s", y.value_counts().sort_index().to_dict())

    if n < MIN_TRAIN_ROWS:
        log.error(
            "Need at least %d rows for training, got %d. "
            "Fetch more data and rebuild the dataset.",
            MIN_TRAIN_ROWS, n,
        )
        sys.exit(1)

    # ── Train / Val split (time-ordered — never shuffle) ─────────────────────
    split          = int(n * (1 - VAL_FRACTION))
    X_train, X_val = X.iloc[:split],    X.iloc[split:]
    y_train, y_val = y_enc.iloc[:split], y_enc.iloc[split:]

    # Pull rare classes from val into train if absent (prevents XGBoost crash)
    all_classes        = {0, 1, 2}
    missing_from_train = all_classes - set(y_train.unique())
    if missing_from_train:
        log.info(
            "Classes %s absent from train split – pulling rare samples from val",
            missing_from_train,
        )
        for cls in missing_from_train:
            cls_val_idx = y_val[y_val == cls].index
            if len(cls_val_idx) > 0:
                X_train = pd.concat([X_train, X_val.loc[cls_val_idx]])
                y_train = pd.concat([y_train, y_val.loc[cls_val_idx]])
                X_val   = X_val.drop(cls_val_idx)
                y_val   = y_val.drop(cls_val_idx)

    log.info(
        "Train: %d rows, Val: %d rows (select_by=%s)",
        len(X_train), len(X_val), select_by,
    )

    # ── Scale (fit on train only) ─────────────────────────────────────────────
    scaler    = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    feature_names = list(X.columns)
    n_val_bars    = len(X_val_s)      # total bars in validation window

    # Raw val labels in {-1, 0, 1} — used by trading quality scorer
    y_val_raw = y_val.map(inv_label_map).values

    log.info(
        "Validation set: %d bars  "
        "Hard gates: count >= %d AND coverage >= %.0f%%",
        n_val_bars, _HC_MIN_TRADES, _HC_MIN_COVERAGE * 100,
    )

    # ── Optuna objective ───────────────────────────────────────────────────────
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
            "gamma":            trial.suggest_float("gamma", 0.5, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "objective":        "multi:softprob",
            "num_class":        3,
            "eval_metric":      "mlogloss",
            "verbosity":        0,
            "tree_method":      "hist",
            "random_state":     42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_s, y_train, verbose=False)

        preds = model.predict(X_val_s)
        f1    = float(f1_score(y_val, preds, average="macro", zero_division=0))

        if select_by == "f1":
            return f1

        # ── STAGE 1: Inline hard gates ────────────────────────────────────────
        # Computed HERE in the objective, NOT delegated to any helper function.
        # When these gates fail, return _GATE_FAIL_SCORE (-100.0) immediately.
        # This sentinel is so negative that Optuna can never select a gated-out
        # trial over any trial that passes, regardless of F1 score.
        proba   = model.predict_proba(X_val_s)
        _conf   = np.max(proba, axis=1)
        _sig    = np.array([inv_label_map.get(int(e), 0)
                            for e in np.argmax(proba, axis=1)])
        _mask   = (_conf >= _TRADING_SCORE_MIN_CONF) & (_sig != 0)
        n_conf  = int(_mask.sum())
        cov     = n_conf / n_val_bars   # fraction of val bars that are trades

        if n_conf < _HC_MIN_TRADES:
            return _GATE_FAIL_SCORE     # too few trades — disqualified

        if cov < _HC_MIN_COVERAGE:
            return _GATE_FAIL_SCORE     # coverage below 3% — disqualified

        # ── STAGE 2: Full balanced trading quality score ──────────────────────
        tq = _trading_quality_score(
            proba, y_val_raw, inv_label_map, n_val_bars=n_val_bars,
        )
        return _COMPOSITE_F1_WEIGHT * f1 + _COMPOSITE_TRADING_WEIGHT * tq

    log.info("Running Optuna (%d trials, objective=%s) …", n_trials, select_by)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_score = study.best_value

    # ── Warn if ALL trials failed the hard gates ───────────────────────────────
    all_gates_failed = select_by == "trading" and best_score <= _GATE_FAIL_SCORE + 1.0
    if all_gates_failed:
        log.warning(
            "ALL %d Optuna trials failed the hard-gate checks "
            "(count >= %d  AND  coverage >= %.0f%%). "
            "The model will still be saved but is likely too sparse for live trading. "
            "Consider: --select-by f1, lower confidence threshold in universe.yaml, "
            "or acquiring more data.",
            n_trials, _HC_MIN_TRADES, _HC_MIN_COVERAGE * 100,
        )

    best_params = study.best_params
    log.info(
        "Best val score (obj=%s): %.4f  params: %s",
        select_by, best_score, best_params,
    )

    # ── Post-selection evaluation on val set ──────────────────────────────────
    best_params.update({
        "objective":    "multi:softprob",
        "num_class":    3,
        "eval_metric":  "mlogloss",
        "verbosity":    0,
        "tree_method":  "hist",
        "random_state": 42,
    })
    eval_model = xgb.XGBClassifier(**best_params)
    eval_model.fit(X_train_s, y_train, verbose=False)

    val_preds = eval_model.predict(X_val_s)
    val_proba = eval_model.predict_proba(X_val_s)
    val_f1    = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_stats = _compute_trading_stats(
        val_proba, y_val_raw, inv_label_map, n_val_bars=n_val_bars,
    )
    _log_trading_stats(val_f1, val_stats)

    # ── Retrain on full dataset ────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(scaler.transform(X), y_enc, verbose=False)

    # ── Persist artifacts ─────────────────────────────────────────────────────
    art_model  = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    art_scaler = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    art_schema = Path(f"artifacts/schema/{symbol}_features.json")

    for p in (art_model, art_scaler, art_schema):
        p.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, art_model)
    joblib.dump(scaler,      art_scaler)

    hc_passed = val_stats["hard_constraints_passed"]
    schema = {
        "symbol":                    symbol,
        "feature_names":             feature_names,
        "n_features":                len(feature_names),
        "label_map":                 label_map,
        "inv_label_map":             {str(k): v for k, v in inv_label_map.items()},
        "best_val_f1":               round(val_f1, 6),
        "best_val_trading_quality":  round(val_stats["trading_quality"], 6),
        "val_trade_coverage_pct":    round(val_stats["trade_coverage_pct"], 4),
        "val_confident_trades":      val_stats["n_trades"],
        "val_dir_accuracy":          round(val_stats["dir_accuracy"], 6),
        "val_profit_factor":         round(val_stats["profit_factor"], 6),
        "hard_constraints_passed":   hc_passed,
        "all_gates_failed":          all_gates_failed,
        "select_by":                 select_by,
        "n_trials":                  n_trials,
        "train_rows":                len(X_train),
    }
    art_schema.write_text(json.dumps(schema, indent=2))

    log.info("Model   -> %s", art_model)
    log.info("Scaler  -> %s", art_scaler)
    log.info("Schema  -> %s", art_schema)

    hc_label        = "PASSED" if hc_passed else "FAILED"
    all_gates_label = "  [WARNING: all Optuna trials failed hard gates]" if all_gates_failed else ""
    print(
        f"\nTraining complete.{all_gates_label}"
        f"\n  Selection mode         : {select_by}"
        f"\n  Val macro-F1           : {val_f1:.4f}"
        f"\n  Val trading quality    : {val_stats['trading_quality']:.4f}"
        f"\n  Confident trades       : {val_stats['n_trades']:,d} / {n_val_bars:,d} bars"
        f"\n  Trade coverage         : {val_stats['trade_coverage_pct']:.2f}%"
        f"  [gate: >= {_HC_MIN_COVERAGE * 100:.0f}%]"
        f"\n  Directional accuracy   : {val_stats['dir_accuracy']:.1%}"
        f"  [gate: >= {_HC_MIN_DIR_ACC:.0%}]"
        f"\n  Val profit factor      : {val_stats['profit_factor']:.3f}"
        f"  [gate: >= {_HC_MIN_PF:.2f}]"
        f"\n  Hard constraints       : {hc_label}"
    )
    if not hc_passed:
        print(
            f"  Constraint failures    : "
            f"{'; '.join(val_stats['hard_constraint_failures'])}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trading-quality scorer helpers — all public for direct unit testing
# ─────────────────────────────────────────────────────────────────────────────


def _trading_quality_score(
    proba: np.ndarray,
    y_raw: np.ndarray,
    inv_label_map: dict,
    min_conf: float = _TRADING_SCORE_MIN_CONF,
    n_val_bars: int | None = None,
) -> float:
    """
    Return the trading quality score component.

    When STAGE 1 hard gates pass, this returns a value in [-10, 1].
    A healthy model returns [0, 1].  Secondary gate failures return
    _HC_PENALTY_MEDIUM or _HC_PENALTY_SEVERE (both negative).

    This function is called AFTER the inline gates in the objective already
    confirmed count >= 50 and coverage >= 3%, so the primary penalties
    (_HC_PENALTY_SEVERE) should not be reached in normal Optuna runs.
    They exist as a safety net for direct calls (e.g. from tests).
    """
    stats = _compute_trading_stats(
        proba, y_raw, inv_label_map, min_conf=min_conf, n_val_bars=n_val_bars,
    )
    return stats["trading_quality"]


def _compute_trading_stats(
    proba: np.ndarray,
    y_raw: np.ndarray,
    inv_label_map: dict,
    min_conf: float = _TRADING_SCORE_MIN_CONF,
    n_val_bars: int | None = None,
) -> dict:
    """
    Compute the full set of trading-quality statistics.

    Returns
    -------
    dict with keys:
      n_trades                 — confident non-flat predictions
      n_val_bars               — total validation bars
      trade_coverage_pct       — 100 * n_trades / n_val_bars
      n_wins                   — directional wins on confident predictions
      dir_accuracy             — n_wins / n_trades  (0.0 if n_trades == 0)
      profit_factor            — simplified PF, capped at 5
      hard_constraints_passed  — bool: all HCs pass
      hard_constraint_failures — list[str] of failed constraint descriptions
      score_components         — dict of each sub-score (only when HCs pass)
      trading_quality          — final score; negative if any HC fails
    """
    if n_val_bars is None:
        n_val_bars = len(proba)

    conf     = np.max(proba, axis=1)
    pred_enc = np.argmax(proba, axis=1)
    signals  = np.array([inv_label_map.get(int(e), 0) for e in pred_enc])

    trade_mask = (conf >= min_conf) & (signals != 0)
    n_trades   = int(trade_mask.sum())
    coverage   = n_trades / n_val_bars if n_val_bars > 0 else 0.0

    if n_trades > 0:
        sig_t    = signals[trade_mask]
        y_t      = y_raw[trade_mask]
        wins_arr = (sig_t == y_t) & (y_t != 0)
        n_wins   = int(wins_arr.sum())
        n_losses = n_trades - n_wins
        dir_acc  = n_wins / n_trades
        pf       = (n_wins / n_losses) if n_losses > 0 else float(n_wins)
        pf       = min(pf, 5.0)
    else:
        n_wins = n_losses = 0
        dir_acc = pf = 0.0

    # ── Hard constraint checks ────────────────────────────────────────────────
    penalty  = 0.0
    failures: list[str] = []

    if n_trades < _HC_MIN_TRADES:
        failures.append(f"trade_count={n_trades} < {_HC_MIN_TRADES}")
        penalty = max(penalty, abs(_HC_PENALTY_SEVERE))

    if coverage < _HC_MIN_COVERAGE:
        failures.append(f"coverage={coverage:.2%} < {_HC_MIN_COVERAGE:.0%}")
        penalty = max(penalty, abs(_HC_PENALTY_SEVERE))

    if pf < _HC_MIN_PF:
        failures.append(f"profit_factor={pf:.3f} < {_HC_MIN_PF:.2f}")
        penalty = max(penalty, abs(_HC_PENALTY_MEDIUM))

    if dir_acc < _HC_MIN_DIR_ACC:
        failures.append(f"dir_accuracy={dir_acc:.1%} < {_HC_MIN_DIR_ACC:.0%}")
        penalty = max(penalty, abs(_HC_PENALTY_MEDIUM))

    hc_passed = len(failures) == 0
    base = {
        "n_trades":                n_trades,
        "n_val_bars":              n_val_bars,
        "trade_coverage_pct":      round(coverage * 100, 4),
        "n_wins":                  n_wins,
        "dir_accuracy":            round(dir_acc, 6),
        "profit_factor":           round(pf, 6),
        "hard_constraints_passed": hc_passed,
        "hard_constraint_failures": failures,
    }

    if not hc_passed:
        return {**base, "score_components": {}, "trading_quality": -penalty}

    # ── Five sub-components (only when all HCs pass) ──────────────────────────
    comp_dir_acc  = dir_acc
    comp_norm_pf  = _normalized_pf(pf)
    comp_coverage = _coverage_score(coverage)
    comp_activity = _activity_score(n_trades)
    comp_ppt      = _ppt_score(n_wins, n_trades)

    tq = (
        _W_DIR_ACC  * comp_dir_acc
        + _W_NORM_PF  * comp_norm_pf
        + _W_COVERAGE * comp_coverage
        + _W_ACTIVITY * comp_activity
        + _W_PPT      * comp_ppt
    )
    tq = float(max(0.0, min(tq, 1.0)))   # clamp to [0, 1]

    return {
        **base,
        "score_components": {
            "dir_accuracy":   round(comp_dir_acc,  6),
            "normalized_pf":  round(comp_norm_pf,  6),
            "trade_coverage": round(comp_coverage, 6),
            "activity":       round(comp_activity, 6),
            "pnl_per_trade":  round(comp_ppt,      6),
        },
        "trading_quality": round(tq, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pure sub-component functions — each maps to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────


def _coverage_score(coverage: float) -> float:
    """
    Map trade-coverage fraction to a score in [0, 1].

    Target range for futures minute-bar trading: 3–20% of bars as entries.

      0.0%  →  0.00
      1.0%  →  0.10  (far too sparse)
      3.0%  →  1.00  (lower bound of target — full credit starts here)
     20.0%  →  1.00  (upper bound of target)
     35.0%  →  0.50  (heavy overtrading)
     50.0%+ →  0.25
    """
    if coverage <= 0.0:
        return 0.0
    if coverage < 0.01:
        return coverage / 0.01 * 0.10
    if coverage < 0.03:
        return 0.10 + (coverage - 0.01) / 0.02 * 0.90
    if coverage <= 0.20:
        return 1.0
    if coverage <= 0.35:
        return 1.0 - (coverage - 0.20) / 0.15 * 0.50
    return max(0.25, 0.50 - (coverage - 0.35) / 0.15 * 0.25)


def _activity_score(n_trades: int) -> float:
    """
    Map trade count to a score in [0, 1].

      0–9   trades  →  0.00
     10–49  trades  →  ramp 0.00 → 0.80
     50–199 trades  →  ramp 0.80 → 1.00  (target range)
     200+   trades  →  1.00
    """
    if n_trades < 10:
        return 0.0
    if n_trades < 50:
        return (n_trades - 10) / 40 * 0.80
    if n_trades < 200:
        return 0.80 + (n_trades - 50) / 150 * 0.20
    return 1.0


def _normalized_pf(pf: float) -> float:
    """
    Map profit factor to [0, 1].
    pf=1.0 → 0.0, pf=2.0 → 0.5, pf>=3.0 → 1.0
    """
    return float(min(max((pf - 1.0) / 2.0, 0.0), 1.0))


def _ppt_score(n_wins: int, n_trades: int) -> float:
    """
    Per-trade quality score in [0, 1].
    avg = (wins - losses) / n_trades ∈ [-1, +1]; normalise to [0, 1].
    50% win rate → 0.50, 100% → 1.00, 0% → 0.00
    """
    if n_trades <= 0:
        return 0.0
    avg = (n_wins - (n_trades - n_wins)) / n_trades
    return float((avg + 1.0) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────


def _log_trading_stats(val_f1: float, stats: dict) -> None:
    hc  = "PASSED" if stats["hard_constraints_passed"] else "FAILED"
    log.info(
        "Val trading stats  |  macro-F1=%.4f  |  HardConstraints=%s",
        val_f1, hc,
    )
    log.info(
        "  confident_trade_count   : %d / %d bars  (coverage=%.2f%%)  "
        "[gate: >= %d  AND  >= %.0f%%]",
        stats["n_trades"], stats["n_val_bars"], stats["trade_coverage_pct"],
        _HC_MIN_TRADES, _HC_MIN_COVERAGE * 100,
    )
    log.info(
        "  directional_accuracy    : %.1f%%  [gate: >= %.0f%%]",
        stats["dir_accuracy"] * 100, _HC_MIN_DIR_ACC * 100,
    )
    log.info(
        "  validation_profit_factor: %.3f  [gate: >= %.2f]",
        stats["profit_factor"], _HC_MIN_PF,
    )
    log.info("  trading_quality_score   : %.4f", stats["trading_quality"])

    if not stats["hard_constraints_passed"]:
        log.warning(
            "  HARD CONSTRAINT FAILURES: %s",
            " | ".join(stats["hard_constraint_failures"]),
        )
    if stats.get("score_components"):
        sc = stats["score_components"]
        log.info(
            "  sub-components: dir_acc=%.3f  norm_pf=%.3f  coverage=%.3f  "
            "activity=%.3f  ppt=%.3f",
            sc["dir_accuracy"], sc["normalized_pf"], sc["trade_coverage"],
            sc["activity"], sc["pnl_per_trade"],
        )
