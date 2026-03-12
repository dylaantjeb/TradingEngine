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
     trading  (new):     four inline hard gates then balanced composite score.
6. Retrain best model on full train set.
7. Persist model, scaler, and feature schema.

How --select-by trading works
──────────────────────────────
Each Optuna trial passes through FOUR inline hard gates.  If ANY gate fails,
the trial receives _GATE_FAIL_SCORE (-1e9) and is excluded from consideration.
No passing model can ever achieve a score that low, so Optuna will always prefer
a gate-passing model over a gate-failing one.

  Gate 1: confident_trade_count  >= 50
  Gate 2: trade_coverage         >= 3%   (confident_trades / total_val_bars)
  Gate 3: directional_accuracy   >= 52%
  Gate 4: profit_factor          >= 1.10

"Confident trade" = model confidence >= _TRADING_SCORE_MIN_CONF (0.65)
                    AND predicted direction != flat.

If ALL n_trials fail the gates:
  • The bad model is NOT saved.
  • A fallback F1 study is run with the same budget.
  • The fallback model IS saved, labelled selected_via=fallback_f1.
  • A loud WARNING is printed.

After the study, selected model stats are always logged:
  confident_trade_count, trade_coverage_pct, directional_accuracy,
  validation_profit_factor, trading_quality, hard_constraints_passed,
  selected_via.

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
_TRADING_SCORE_MIN_CONF = 0.65

# Composite objective weights
_COMPOSITE_F1_WEIGHT      = 0.40
_COMPOSITE_TRADING_WEIGHT = 0.60

# ── Hard-gate thresholds (ALL four enforced inline in the Optuna objective) ────
_HC_MIN_TRADES   = 50      # minimum confident non-flat predictions on val set
_HC_MIN_COVERAGE = 0.03    # minimum fraction of val bars that are "trades" (3%)
_HC_MIN_PF       = 1.10    # minimum profit factor
_HC_MIN_DIR_ACC  = 0.52    # minimum directional accuracy

# Sentinel score returned when any STAGE 1 gate fails.
# Must be so negative that Optuna can never prefer a failed trial over a passing
# one.  The best possible composite is ~1.0, so -1e9 is unambiguous.
_GATE_FAIL_SCORE = -1e9

# Penalty values used by _compute_trading_stats (secondary check, safety net).
_HC_PENALTY_SEVERE = -10.0   # trade_count < 50 or coverage < 3%
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
                 with four inline hard gates).
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
    n_val_bars    = len(X_val_s)

    # Raw val labels in {-1, 0, 1} — used by trading quality scorer
    y_val_raw = y_val.map(inv_label_map).values

    log.info(
        "Validation set: %d bars  |  Hard gates (ALL must pass): "
        "count >= %d, coverage >= %.0f%%, dir_acc >= %.0f%%, PF >= %.2f",
        n_val_bars,
        _HC_MIN_TRADES, _HC_MIN_COVERAGE * 100,
        _HC_MIN_DIR_ACC * 100, _HC_MIN_PF,
    )

    # ── Shared param sampler (used by both trading and fallback studies) ───────
    def _sample_params(trial: "optuna.Trial") -> dict:
        return {
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

    # ── Optuna trading objective ───────────────────────────────────────────────
    def trading_objective(trial: "optuna.Trial") -> float:
        model = xgb.XGBClassifier(**_sample_params(trial))
        model.fit(X_train_s, y_train, verbose=False)

        preds = model.predict(X_val_s)
        f1    = float(f1_score(y_val, preds, average="macro", zero_division=0))

        # ── STAGE 1: All four inline hard gates ───────────────────────────────
        # Each gate returns _GATE_FAIL_SCORE (-1e9) immediately on failure.
        # This sentinel is so negative that Optuna will NEVER select a failed
        # trial over any passing trial, even if ALL trials fail and scores
        # are all -1e9 (fallback logic catches that case after study.optimize).
        proba   = model.predict_proba(X_val_s)
        conf    = np.max(proba, axis=1)
        sig     = np.array([inv_label_map.get(int(e), 0)
                             for e in np.argmax(proba, axis=1)])
        mask    = (conf >= _TRADING_SCORE_MIN_CONF) & (sig != 0)
        n_conf  = int(mask.sum())
        cov     = n_conf / n_val_bars

        # Gate 1 — minimum trade count
        if n_conf < _HC_MIN_TRADES:
            return _GATE_FAIL_SCORE

        # Gate 2 — minimum coverage
        if cov < _HC_MIN_COVERAGE:
            return _GATE_FAIL_SCORE

        # Gate 3 — minimum directional accuracy
        sig_t   = sig[mask]
        y_t     = y_val_raw[mask]
        wins    = int(((sig_t == y_t) & (y_t != 0)).sum())
        losses  = n_conf - wins
        dir_acc = wins / n_conf
        if dir_acc < _HC_MIN_DIR_ACC:
            return _GATE_FAIL_SCORE

        # Gate 4 — minimum profit factor
        pf = min((wins / losses) if losses > 0 else float(wins), 5.0)
        if pf < _HC_MIN_PF:
            return _GATE_FAIL_SCORE

        # ── STAGE 2: Full balanced trading quality score ──────────────────────
        tq = _trading_quality_score(
            proba, y_val_raw, inv_label_map, n_val_bars=n_val_bars,
        )
        return _COMPOSITE_F1_WEIGHT * f1 + _COMPOSITE_TRADING_WEIGHT * tq

    # ── Optuna F1 objective (used as fallback and for select_by=f1) ───────────
    def f1_objective(trial: "optuna.Trial") -> float:
        model = xgb.XGBClassifier(**_sample_params(trial))
        model.fit(X_train_s, y_train, verbose=False)
        preds = model.predict(X_val_s)
        return float(f1_score(y_val, preds, average="macro", zero_division=0))

    # ── Run primary study ─────────────────────────────────────────────────────
    log.info("Running Optuna (%d trials, objective=%s) …", n_trials, select_by)

    primary_objective = trading_objective if select_by == "trading" else f1_objective
    study = optuna.create_study(direction="maximize")
    study.optimize(primary_objective, n_trials=n_trials, show_progress_bar=False)

    best_score = study.best_value

    # ── Detect all-gates-failed and fall back to F1 ───────────────────────────
    # A score of exactly _GATE_FAIL_SCORE means every single Optuna trial was
    # disqualified.  We must NOT save that model.  Instead we fall back to a
    # fresh F1 study so the user always gets a usable model.
    all_gates_failed = select_by == "trading" and best_score <= _GATE_FAIL_SCORE * 0.5
    selected_via: str

    if all_gates_failed:
        log.warning(
            "=" * 70,
        )
        log.warning(
            "ALL %d Optuna trials failed the trading hard-gate checks.",
            n_trials,
        )
        log.warning(
            "  Required: count >= %d  AND  coverage >= %.0f%%  "
            "AND  dir_acc >= %.0f%%  AND  PF >= %.2f",
            _HC_MIN_TRADES, _HC_MIN_COVERAGE * 100,
            _HC_MIN_DIR_ACC * 100, _HC_MIN_PF,
        )
        log.warning(
            "  The sparse/invalid model will NOT be saved."
        )
        log.warning(
            "  Running fallback F1 study (%d trials) — "
            "consider --select-by f1 or acquire more data.",
            n_trials,
        )
        log.warning("=" * 70)

        fallback_study = optuna.create_study(direction="maximize")
        fallback_study.optimize(f1_objective, n_trials=n_trials, show_progress_bar=False)
        best_params = fallback_study.best_params
        best_score  = fallback_study.best_value
        selected_via = "fallback_f1"
    else:
        best_params  = study.best_params
        selected_via = "trading" if select_by == "trading" else "f1"

    log.info(
        "Selected model  |  objective=%s  |  score=%.4f  |  via=%s",
        select_by, best_score, selected_via,
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
    _log_selected_model(val_f1, val_stats, selected_via)

    # ── Retrain on full dataset ────────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(scaler.transform(X), y_enc, verbose=False)

    # ── Persist artifacts ─────────────────────────────────────────────────────
    art_model  = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    art_scaler = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    art_schema = Path(f"artifacts/schema/{symbol}_features.json")

    for p in (art_model, art_scaler, art_schema):
        p.parent.mkdir(parents=True, exist_ok=True)

    import joblib as _joblib  # already imported above; alias avoids redeclaration
    _joblib.dump(final_model, art_model)
    _joblib.dump(scaler,      art_scaler)

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
        "selected_via":              selected_via,
        "select_by":                 select_by,
        "n_trials":                  n_trials,
        "train_rows":                len(X_train),
    }
    art_schema.write_text(json.dumps(schema, indent=2))

    log.info("Model   -> %s", art_model)
    log.info("Scaler  -> %s", art_scaler)
    log.info("Schema  -> %s", art_schema)

    hc_label = "PASSED" if hc_passed else "FAILED"
    fallback_note = (
        "  [FALLBACK: all trading trials failed gates — saved best F1 model]\n"
        if all_gates_failed else ""
    )
    print(
        f"\nTraining complete."
        f"\n{fallback_note}"
        f"  Selected via           : {selected_via}"
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
    """Return the trading quality score component in [-10, 1]."""
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
      hard_constraints_passed  — bool: all four HCs pass
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

    # ── Hard constraint checks (all four) ─────────────────────────────────────
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
        "n_trades":                 n_trades,
        "n_val_bars":               n_val_bars,
        "trade_coverage_pct":       round(coverage * 100, 4),
        "n_wins":                   n_wins,
        "dir_accuracy":             round(dir_acc, 6),
        "profit_factor":            round(pf, 6),
        "hard_constraints_passed":  hc_passed,
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


def _log_selected_model(val_f1: float, stats: dict, selected_via: str) -> None:
    hc = "PASSED" if stats["hard_constraints_passed"] else "FAILED"
    log.info(
        "Selected model stats  |  macro-F1=%.4f  |  selected_via=%s  |  HC=%s",
        val_f1, selected_via, hc,
    )
    log.info(
        "  confident_trade_count   : %d / %d bars",
        stats["n_trades"], stats["n_val_bars"],
    )
    log.info(
        "  trade_coverage_pct      : %.2f%%  [gate: >= %.0f%%]",
        stats["trade_coverage_pct"], _HC_MIN_COVERAGE * 100,
    )
    log.info(
        "  directional_accuracy    : %.1f%%  [gate: >= %.0f%%]",
        stats["dir_accuracy"] * 100, _HC_MIN_DIR_ACC * 100,
    )
    log.info(
        "  validation_profit_factor: %.3f  [gate: >= %.2f]",
        stats["profit_factor"], _HC_MIN_PF,
    )
    log.info(
        "  trading_quality         : %.4f  (bounded [0,1] when all HCs pass)",
        stats["trading_quality"],
    )
    log.info(
        "  hard_constraints_passed : %s",
        hc,
    )
    if not stats["hard_constraints_passed"]:
        log.warning(
            "  HC FAILURES: %s",
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
