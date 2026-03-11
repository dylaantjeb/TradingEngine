"""
Signal engine for TradingEngine.

Loads trained artifacts, runs feature pipeline on a bar window,
produces a rich SignalOutput that includes:
  - signal (-1 / 0 / +1)
  - confidence
  - top contributing features
  - which filters passed / failed
  - recommended stop-loss, take-profit, and position size
  - current market regime
  - human-readable rationale
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import get_config
from src.features.regime import Regime, detect_regime
from src.utils.time_utils import in_blackout, in_session

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.50


@dataclass
class SignalOutput:
    """Structured output from the signal engine for one bar."""

    # Core signal
    signal: int              # -1 = short, 0 = flat, +1 = long
    confidence: float        # max class probability

    # Filters
    filters_passed: list[str] = field(default_factory=list)
    filters_failed: list[str] = field(default_factory=list)

    # Explainability
    top_features: list[dict] = field(default_factory=list)  # [{name, value, contribution}]
    rationale: str = ""

    # Risk guidance
    recommended_sl_pts: float = 0.0    # stop-loss distance in price points
    recommended_tp_pts: float = 0.0    # take-profit distance in price points
    recommended_size: int = 1          # contracts (from PositionSizer)

    # Context
    regime: str = Regime.RANGING
    atr_pts: float = 0.0
    raw_signal: int = 0    # pre-filter signal


class SignalEngine:
    """
    Stateless signal generator.  Load once, call generate() per bar.
    """

    def __init__(
        self,
        symbol: str,
        artifacts_dir: Path | None = None,
    ):
        self.symbol = symbol
        arts = artifacts_dir or Path("artifacts")
        self._model, self._scaler, self._feature_names, self._inv_label_map = (
            _load_artifacts(symbol, arts)
        )
        self._cfg = get_config()

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        window: pd.DataFrame,
        account_equity: float = 100_000.0,
        top_n_features: int = 5,
    ) -> SignalOutput:
        """
        Generate a signal from the most recent window of OHLCV bars.

        Parameters
        ----------
        window          : DataFrame with OHLCV columns, newest bar last.
                          Must have at least MIN_ROWS rows.
        account_equity  : Current equity for position sizing.
        top_n_features  : How many top feature contributions to include.

        Returns
        -------
        SignalOutput
        """
        from src.features.builder import build_features, MIN_ROWS

        if len(window) < MIN_ROWS:
            return SignalOutput(
                signal=0, confidence=0.0,
                rationale=f"Insufficient bars ({len(window)} < {MIN_ROWS})",
            )

        # ── Feature computation ────────────────────────────────────────────────
        try:
            features = build_features(window, min_rows=1)
        except Exception as exc:
            log.warning("Feature build failed: %s", exc)
            return SignalOutput(signal=0, confidence=0.0, rationale=str(exc))

        if features.empty:
            return SignalOutput(signal=0, confidence=0.0, rationale="Empty feature frame")

        last_feat_row = features[self._feature_names].iloc[-1].values.astype(float)
        last_ts = features.index[-1]

        X_scaled = self._scaler.transform(last_feat_row.reshape(1, -1))

        # ── Model prediction ───────────────────────────────────────────────────
        proba     = self._model.predict_proba(X_scaled)[0]
        pred_enc  = int(np.argmax(proba))
        confidence = float(proba[pred_enc])
        raw_signal = self._inv_label_map[str(pred_enc)]

        if confidence < CONFIDENCE_THRESHOLD:
            raw_signal = 0

        # ── ATR and regime ─────────────────────────────────────────────────────
        contract  = self._cfg.get_contract(self.symbol)
        atr_rel   = float(features["atr_14"].iloc[-1])
        close_val = float(window["close"].iloc[-1])
        atr_pts   = atr_rel * close_val   # ATR in price points
        atr_ticks = atr_pts / contract.tick_size

        try:
            regime_series = detect_regime(window)
            regime_val = regime_series.iloc[-1]
            regime = regime_val.value if hasattr(regime_val, "value") else str(regime_val)
        except Exception:
            regime = Regime.RANGING.value

        # ── Filters ────────────────────────────────────────────────────────────
        cfg = self._cfg
        filters_passed: list[str] = []
        filters_failed: list[str] = []
        sig = raw_signal

        # Confidence
        if confidence >= CONFIDENCE_THRESHOLD:
            filters_passed.append("confidence")
        else:
            filters_failed.append(f"confidence({confidence:.2f}<{CONFIDENCE_THRESHOLD})")
            sig = 0

        # Session
        if in_session(last_ts, cfg.filters.session_start_utc_hour, cfg.filters.session_end_utc_hour):
            filters_passed.append("session")
        else:
            filters_failed.append("session")
            sig = 0

        # Blackout
        if not in_blackout(last_ts, cfg.filters.news_blackout_windows):
            filters_passed.append("no_blackout")
        else:
            filters_failed.append("news_blackout")
            sig = 0

        # ATR
        if cfg.filters.atr_min_ticks <= atr_ticks <= cfg.filters.atr_max_ticks:
            filters_passed.append("atr")
        else:
            filters_failed.append(
                f"atr({atr_ticks:.1f} not in [{cfg.filters.atr_min_ticks},{cfg.filters.atr_max_ticks}])"
            )
            sig = 0

        # ── Explainability ─────────────────────────────────────────────────────
        top_features = _get_contributions(
            self._model, X_scaled, pred_enc,
            self._feature_names, last_feat_row, top_n_features
        )

        # ── Risk guidance ──────────────────────────────────────────────────────
        sl_atr_mult = 1.5
        tp_atr_mult = 2.5
        recommended_sl = atr_pts * sl_atr_mult
        recommended_tp = atr_pts * tp_atr_mult

        risk_frac = 0.01   # 1% of equity per trade
        stop_val  = recommended_sl * contract.multiplier
        recommended_size = max(1, int(account_equity * risk_frac / max(stop_val, 1)))

        # ── Rationale ──────────────────────────────────────────────────────────
        direction = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(sig, "FLAT")
        driver_parts = [
            f"{f['name']}={f['value']:+.3f}" for f in top_features[:3]
        ]
        rationale = (
            f"Signal: {direction} | conf={confidence:.1%} | "
            f"regime={regime} | ATR={atr_pts:.2f}pts | "
            f"drivers: {', '.join(driver_parts)}"
        )
        if filters_failed:
            rationale += f" | BLOCKED by: {', '.join(filters_failed)}"

        return SignalOutput(
            signal=sig,
            confidence=confidence,
            filters_passed=filters_passed,
            filters_failed=filters_failed,
            top_features=top_features,
            rationale=rationale,
            recommended_sl_pts=recommended_sl,
            recommended_tp_pts=recommended_tp,
            recommended_size=recommended_size,
            regime=regime,
            atr_pts=atr_pts,
            raw_signal=raw_signal,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_artifacts(symbol: str, arts: Path):
    try:
        import joblib
    except ImportError:
        log.error("joblib required:  pip install joblib")
        sys.exit(1)

    model_p  = arts / "models"  / f"{symbol}_xgb_best.joblib"
    scaler_p = arts / "scalers" / f"{symbol}_scaler.joblib"
    schema_p = arts / "schema"  / f"{symbol}_features.json"

    for p in (model_p, scaler_p, schema_p):
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}\n"
                f"  Run:  python -m src.cli train --symbol {symbol}"
            )

    model  = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    with open(schema_p) as f:
        schema = json.load(f)

    feature_names: list[str]       = schema["feature_names"]
    inv_label_map: dict[str, int]  = {k: int(v) for k, v in schema["inv_label_map"].items()}
    return model, scaler, feature_names, inv_label_map


def _get_contributions(
    model: Any,
    X_scaled: np.ndarray,
    pred_enc: int,
    feature_names: list[str],
    raw_row: np.ndarray,
    top_n: int,
) -> list[dict]:
    try:
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dmat    = xgb.DMatrix(
            X_scaled,
            feature_names=[f"f{i}" for i in range(X_scaled.shape[1])]
        )
        contribs = booster.predict(dmat, pred_contribs=True)
        n_feat   = X_scaled.shape[1]
        n_class  = len(model.classes_) if hasattr(model, "classes_") else 3

        if contribs.ndim == 1:
            contribs = contribs.reshape(n_class, n_feat + 1)
        elif contribs.ndim == 2:
            contribs = contribs[0].reshape(n_class, n_feat + 1)

        class_contribs = contribs[pred_enc, :n_feat]
    except Exception:
        class_contribs = np.zeros(len(feature_names))

    # Ensure 1-D before argsort so indices are plain scalars, not sub-arrays
    class_contribs_1d = np.asarray(class_contribs).flatten()[:len(feature_names)]
    top_idx = [int(i) for i in np.argsort(np.abs(class_contribs_1d))[::-1][:top_n]]
    return [
        {
            "name": feature_names[i],
            "value": float(X_scaled[0, i]),
            "contribution": float(class_contribs_1d[i]),
        }
        for i in top_idx
    ]
