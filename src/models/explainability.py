"""
Signal explainability for TradingEngine.

Uses XGBoost's built-in `pred_contribs=True` to compute per-feature
SHAP-like contributions without requiring the `shap` library.

The contributions are the marginal gain each feature provides to the
model's log-odds prediction for the winning class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the current signal."""
    name: str
    value: float          # raw feature value (scaled)
    contribution: float   # SHAP-like marginal contribution to log-odds


@dataclass
class SignalExplanation:
    """Full explanation for one bar's signal."""
    signal: int           # -1, 0, +1
    confidence: float
    top_features: list[FeatureContribution] = field(default_factory=list)
    rationale: str = ""


def explain_signal(
    model: Any,
    scaler: Any,
    feature_names: list[str],
    inv_label_map: dict[str, int],
    feature_row: np.ndarray,
    top_n: int = 5,
) -> SignalExplanation:
    """
    Explain why the model produced its current signal for a single bar.

    Parameters
    ----------
    model         : Trained XGBoost Booster or sklearn-API XGBClassifier.
    scaler        : Fitted scaler (RobustScaler or similar).
    feature_names : Ordered list of feature names matching feature_row.
    inv_label_map : Mapping from encoded class index (str) to signal (-1/0/1).
    feature_row   : 1-D raw (unscaled) feature array for the bar.
    top_n         : Number of top contributors to include.

    Returns
    -------
    SignalExplanation
    """
    import xgboost as xgb

    X = feature_row.reshape(1, -1)
    X_scaled = scaler.transform(X)

    # ── Prediction ─────────────────────────────────────────────────────────────
    proba = model.predict_proba(X_scaled)[0]
    pred_enc = int(np.argmax(proba))
    confidence = float(proba[pred_enc])
    signal = inv_label_map[str(pred_enc)]

    # ── SHAP contributions via pred_contribs ───────────────────────────────────
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dmat = xgb.DMatrix(X_scaled, feature_names=[f"f{i}" for i in range(X_scaled.shape[1])])
        # contribs shape: (n_samples, n_features + 1, n_classes) or (n_samples, n_classes * (n_features+1))
        contribs = booster.predict(dmat, pred_contribs=True)

        # Reshape to (n_features+1, n_classes) – XGBoost flattens multiclass
        n_feat   = X_scaled.shape[1]
        n_class  = len(proba)

        if contribs.ndim == 1:
            # Multiclass: flattened (n_samples * n_classes * (n_features+1),)
            contribs = contribs.reshape(n_class, n_feat + 1)
        elif contribs.ndim == 2:
            # contribs shape (1, n_classes * (n_features+1))
            contribs = contribs[0].reshape(n_class, n_feat + 1)

        # Contribution for the predicted class, excluding bias term (last column)
        class_contribs = contribs[pred_enc, :n_feat]

    except Exception as exc:
        log.debug("pred_contribs unavailable: %s", exc)
        class_contribs = np.zeros(len(feature_names))

    # ── Build top-N feature list ───────────────────────────────────────────────
    abs_contribs = np.abs(class_contribs)
    top_idx = np.argsort(abs_contribs)[::-1][:top_n].tolist()

    top_features = [
        FeatureContribution(
            name=feature_names[i],
            value=float(X_scaled[0, i]),
            contribution=float(class_contribs[i]),
        )
        for i in top_idx
    ]

    # ── Human-readable rationale ───────────────────────────────────────────────
    direction = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signal, "FLAT")
    parts = []
    for fc in top_features[:3]:
        direction_word = "drove" if fc.contribution > 0 else "countered"
        parts.append(f"{fc.name} ({fc.value:+.3f}) {direction_word} signal")

    rationale = (
        f"Signal: {direction} (confidence={confidence:.1%}). "
        + "; ".join(parts)
        + ("." if parts else "")
    )

    return SignalExplanation(
        signal=signal,
        confidence=confidence,
        top_features=top_features,
        rationale=rationale,
    )


def explain_dataframe(
    model: Any,
    scaler: Any,
    feature_names: list[str],
    inv_label_map: dict[str, int],
    features_df: pd.DataFrame,
    top_n: int = 5,
) -> list[SignalExplanation]:
    """
    Explain signals for every row in features_df.
    Returns a list of SignalExplanation (one per row).
    """
    results = []
    X = features_df[feature_names].values
    for row in X:
        exp = explain_signal(model, scaler, feature_names, inv_label_map, row, top_n=top_n)
        results.append(exp)
    return results
