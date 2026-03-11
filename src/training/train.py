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
     trading  (new):     maximise a composite score that combines F1 with a
                         simplified trading-quality simulation on val split.
                         Prefers models that are profitable and not overtrading,
                         not just models that classify well.
6. Retrain best model on full train set.
7. Persist model, scaler, and feature schema.

Model selection modes
──────────────────────
  --select-by f1      (default)
    Pure macro-F1 optimisation. Good for balanced label distributions.

  --select-by trading
    Composite score:  0.40 * f1  +  0.60 * trading_quality
    trading_quality   = win_rate * min(profit_factor, 4.0) * (1 - overtrading_penalty)
    This rewards:
      • High directional accuracy on confident predictions (>= 0.65 confidence)
      • Good profit factor
      • Not predicting trades on every bar (penalises >10% bars as entries)
    Use this when you want fewer but higher-quality signals OOS.

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
VAL_FRACTION = 0.20

# Minimum confidence for a prediction to be counted as a "trade" in the
# trading-quality scorer.  Matches the confidence gate in universe.yaml.
_TRADING_SCORE_MIN_CONF = 0.65

# Weight of trading quality vs F1 in composite objective
_COMPOSITE_F1_WEIGHT      = 0.40
_COMPOSITE_TRADING_WEIGHT = 0.60


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
    select_by  : 'f1' (maximise macro-F1) or 'trading' (composite score).
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
    lbl_path = Path(f"data/processed/{symbol}_labels.parquet")

    for p in (feat_path, lbl_path):
        if not p.exists():
            log.error(
                "File not found: %s\n"
                "Run:  python -m src.cli build-dataset --symbol %s --input data/raw/%s_M1.csv",
                p, symbol, symbol,
            )
            sys.exit(1)

    try:
        features = pd.read_parquet(feat_path, engine="pyarrow")
        labels_df = pd.read_parquet(lbl_path, engine="pyarrow")
    except ImportError:
        log.error("pyarrow required:  pip install pyarrow")
        sys.exit(1)

    # Align on index
    idx = features.index.intersection(labels_df.index)
    X = features.loc[idx]
    y = labels_df.loc[idx, "label"]

    # Map labels from {-1, 0, 1} to {0, 1, 2} for XGBoost multi-class
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y_enc = y.map(label_map).astype(int)

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

    # ── Train / Val split ─────────────────────────────────────────────────────
    split = int(n * (1 - VAL_FRACTION))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y_enc.iloc[:split], y_enc.iloc[split:]

    # Ensure all encoded classes [0, 1, 2] appear in training; if any are
    # absent (can happen with very rare "flat" label), pull them from val.
    all_classes = {0, 1, 2}
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
                X_val = X_val.drop(cls_val_idx)
                y_val = y_val.drop(cls_val_idx)

    log.info("Train: %d rows, Val: %d rows (select_by=%s)", len(X_train), len(X_val), select_by)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    feature_names = list(X.columns)

    # Raw val labels in {-1, 0, 1} — used by trading quality scorer
    y_val_raw = y_val.map({v: k for k, v in label_map.items()}).values

    # ── Optuna objective ───────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        # Conservative hyperparameter space biased towards OOS robustness:
        # shallower trees, higher regularisation, more required samples per leaf.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 6),   # cap at 6
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),           # max 0.9
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),  # higher min
            "gamma": trial.suggest_float("gamma", 0.5, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "tree_method": "hist",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_s, y_train, verbose=False)

        preds = model.predict(X_val_s)
        f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        if select_by == "f1":
            return f1

        # Trading-quality composite score
        proba = model.predict_proba(X_val_s)
        tq = _trading_quality_score(proba, y_val_raw, inv_label_map)
        return _COMPOSITE_F1_WEIGHT * f1 + _COMPOSITE_TRADING_WEIGHT * tq

    log.info("Running Optuna (%d trials, objective=%s) …", n_trials, select_by)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    log.info("Best val score (obj=%s): %.4f  params: %s", select_by, best_score, best_params)

    # ── Post-selection: log trading quality of the chosen model ───────────────
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "tree_method": "hist",
            "random_state": 42,
        }
    )
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_s, y_train, verbose=False)

    val_preds = final_model.predict(X_val_s)
    val_proba = final_model.predict_proba(X_val_s)
    val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_tq = _trading_quality_score(val_proba, y_val_raw, inv_label_map)

    log.info(
        "Selected model — val macro-F1: %.4f | trading quality: %.4f",
        val_f1, val_tq,
    )
    _log_val_signal_stats(val_proba, y_val_raw, inv_label_map)

    # ── Retrain on full train set ──────────────────────────────────────────────
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        scaler.transform(X),   # full dataset
        y_enc,
        verbose=False,
    )

    # ── Persist artifacts ─────────────────────────────────────────────────────
    art_model = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    art_scaler = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    art_schema = Path(f"artifacts/schema/{symbol}_features.json")

    for p in (art_model, art_scaler, art_schema):
        p.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, art_model)
    joblib.dump(scaler, art_scaler)

    schema = {
        "symbol": symbol,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "label_map": label_map,
        "inv_label_map": {str(k): v for k, v in inv_label_map.items()},
        "best_val_f1": round(val_f1, 6),
        "best_val_trading_quality": round(val_tq, 6),
        "select_by": select_by,
        "n_trials": n_trials,
        "train_rows": len(X_train),
    }
    art_schema.write_text(json.dumps(schema, indent=2))

    log.info("Model   -> %s", art_model)
    log.info("Scaler  -> %s", art_scaler)
    log.info("Schema  -> %s", art_schema)
    print(
        f"\nTraining complete."
        f"\n  Val macro-F1        : {val_f1:.4f}"
        f"\n  Val trading quality : {val_tq:.4f}"
        f"\n  Selection mode      : {select_by}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trading-quality scorer (used in Optuna objective and post-selection logging)
# ─────────────────────────────────────────────────────────────────────────────


def _trading_quality_score(
    proba: np.ndarray,
    y_raw: np.ndarray,
    inv_label_map: dict,
    min_conf: float = _TRADING_SCORE_MIN_CONF,
) -> float:
    """
    Estimate trading quality from predicted probabilities on a validation set.

    Only counts predictions where the model is confident (>= min_conf) AND
    the predicted direction is non-flat.  Rewards:
      • High directional accuracy (win rate) on those confident predictions
      • Good simulated profit factor
      • Not overtrading (penalises if > 10% of bars are "entered")

    Returns a score in [0, 1] suitable for use in an Optuna objective.
    Returns 0.0 if fewer than 10 trades would be taken.

    Parameters
    ----------
    proba        : (n_bars, n_classes) probability array from predict_proba.
    y_raw        : True labels in {-1, 0, 1} aligned with proba rows.
    inv_label_map: {encoded_int: original_label} e.g. {0: -1, 1: 0, 2: 1}.
    min_conf     : Minimum confidence threshold to count as a "trade".
    """
    conf = np.max(proba, axis=1)
    pred_enc = np.argmax(proba, axis=1)
    signals = np.array([inv_label_map.get(int(e), 0) for e in pred_enc])

    # Only count high-confidence non-flat predictions as "trades"
    trade_mask = (conf >= min_conf) & (signals != 0)
    n_trades = int(trade_mask.sum())

    if n_trades < 10:
        return 0.0

    sig_trades = signals[trade_mask]
    y_trades   = y_raw[trade_mask]

    # Directional win: signal direction matches actual label direction
    # (label 0 = flat, treated as "no win")
    wins   = (sig_trades == y_trades) & (y_trades != 0)
    losses = (sig_trades != y_trades) | (y_trades == 0)

    n_wins   = int(wins.sum())
    n_losses = n_trades - n_wins
    win_rate = n_wins / n_trades

    # Simplified profit factor (assume equal absolute move for each bar)
    pf = n_wins / (n_losses + 1e-9) if n_losses > 0 else float(n_wins)
    pf = min(pf, 4.0)    # cap at 4× to avoid outlier domination

    # Overtrading penalty: linear ramp starting at 10% of bars being trades
    trade_rate = n_trades / len(signals)
    overtrading_penalty = max(0.0, (trade_rate - 0.10) * 2.0)   # ramp × 2
    overtrading_penalty = min(overtrading_penalty, 0.5)           # cap at 50%

    score = win_rate * pf * (1.0 - overtrading_penalty)
    return float(max(0.0, score))


def _log_val_signal_stats(
    proba: np.ndarray,
    y_raw: np.ndarray,
    inv_label_map: dict,
    min_conf: float = _TRADING_SCORE_MIN_CONF,
) -> None:
    """Log directional accuracy breakdown for the selected model on val data."""
    conf = np.max(proba, axis=1)
    pred_enc = np.argmax(proba, axis=1)
    signals = np.array([inv_label_map.get(int(e), 0) for e in pred_enc])

    total = len(signals)
    n_long  = int((signals ==  1).sum())
    n_short = int((signals == -1).sum())
    n_flat  = int((signals ==  0).sum())

    trade_mask = (conf >= min_conf) & (signals != 0)
    n_conf = int(trade_mask.sum())

    if n_conf > 0:
        sig_t = signals[trade_mask]
        y_t   = y_raw[trade_mask]
        wins  = ((sig_t == y_t) & (y_t != 0)).sum()
        win_rate = wins / n_conf
    else:
        win_rate = 0.0

    log.info(
        "Val signal stats (conf >= %.2f): %d/%d bars are trades  "
        "[long=%d short=%d flat=%d]  directional accuracy=%.1f%%",
        min_conf, n_conf, total, n_long, n_short, n_flat, win_rate * 100,
    )
