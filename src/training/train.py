"""
Model training pipeline.

Steps
──────
1. Load processed features + labels (parquet).
2. Align and clean (drop rows without overlap).
3. Time-based train / validation split (no shuffling!).
4. Scale features with RobustScaler (persisted for inference).
5. Optuna hyperparameter search over XGBoost.
6. Retrain best model on full train set.
7. Persist model, scaler, and feature schema.

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
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 200
VAL_FRACTION = 0.20


def train(symbol: str, n_trials: int = 20) -> None:
    """Full training pipeline for `symbol`."""
    try:
        import joblib
        import optuna
        import xgboost as xgb
        from sklearn.metrics import f1_score, log_loss
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

    log.info("Train: %d rows, Val: %d rows", len(X_train), len(X_val))

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    feature_names = list(X.columns)

    # ── Optuna objective ───────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "tree_method": "hist",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        # Do NOT pass eval_set: class imbalance may cause missing classes in val split
        model.fit(X_train_s, y_train, verbose=False)
        preds = model.predict(X_val_s)
        # Optimise macro-F1 (we care about signal quality, not class distribution)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    log.info("Running Optuna (%d trials) …", n_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    log.info("Best val macro-F1: %.4f  params: %s", best_score, best_params)

    # ── Retrain on full train set ──────────────────────────────────────────────
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
        "best_val_f1": best_score,
        "n_trials": n_trials,
        "train_rows": len(X_train),
    }
    art_schema.write_text(json.dumps(schema, indent=2))

    log.info("Model   → %s", art_model)
    log.info("Scaler  → %s", art_scaler)
    log.info("Schema  → %s", art_schema)
    print(f"\nTraining complete. Best macro-F1 (val): {best_score:.4f}")
