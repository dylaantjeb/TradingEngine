"""
Unit tests for src/strategy/signal_engine.py

Most tests use SignalEngine.generate() with a mock model and scaler
to avoid requiring trained artifacts on disk.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    close = 5000.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    spread = 0.25
    high = close + rng.uniform(0, spread, n)
    low  = close - rng.uniform(0, spread, n)
    open_ = close * np.exp(rng.normal(0, 0.0005, n))
    volume = rng.integers(100, 2000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _fake_artifacts(tmp: Path, symbol: str = "TEST"):
    """Write minimal fake joblib artifacts for testing."""
    import joblib
    from sklearn.preprocessing import RobustScaler
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import LabelEncoder

    # Fake feature names (must match builder output)
    from src.features.builder import build_features
    df = _make_ohlcv(300)
    feats = build_features(df, min_rows=1)
    feature_names = list(feats.columns)
    n_feat = len(feature_names)

    # Tiny XGBoost model
    X = np.random.randn(90, n_feat).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = xgb.XGBClassifier(n_estimators=3, verbosity=0, use_label_encoder=False,
                               eval_metric="mlogloss")
    model.fit(X, y_enc)

    scaler = RobustScaler()
    scaler.fit(X)

    # inv_label_map: encoded class → signal (-1/0/1)
    inv_label_map = {"0": -1, "1": 0, "2": 1}

    (tmp / "models").mkdir(parents=True, exist_ok=True)
    (tmp / "scalers").mkdir(parents=True, exist_ok=True)
    (tmp / "schema").mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  tmp / "models"  / f"{symbol}_xgb_best.joblib")
    joblib.dump(scaler, tmp / "scalers" / f"{symbol}_scaler.joblib")
    with open(tmp / "schema" / f"{symbol}_features.json", "w") as f:
        json.dump({"feature_names": feature_names, "inv_label_map": inv_label_map}, f)


class TestSignalEngineOutput:
    @pytest.fixture(scope="class")
    def engine_and_df(self):
        """Create a SignalEngine with fake artifacts and a test DataFrame."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            _fake_artifacts(tmp, "TEST")
            from src.strategy.signal_engine import SignalEngine
            engine = SignalEngine("TEST", artifacts_dir=tmp)
            df = _make_ohlcv(300)
            yield engine, df

    def test_generate_returns_signal_output(self, engine_and_df):
        from src.strategy.signal_engine import SignalOutput
        engine, df = engine_and_df
        result = engine.generate(df)
        assert isinstance(result, SignalOutput)

    def test_signal_in_valid_range(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert result.signal in (-1, 0, 1)

    def test_confidence_in_range(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert 0.0 <= result.confidence <= 1.0

    def test_top_features_populated(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert isinstance(result.top_features, list)
        # Each feature dict has required keys
        for fc in result.top_features:
            assert "name" in fc
            assert "value" in fc
            assert "contribution" in fc

    def test_rationale_is_string(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0

    def test_atr_positive(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert result.atr_pts >= 0

    def test_recommended_sl_positive(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert result.recommended_sl_pts >= 0

    def test_regime_is_valid(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        assert result.regime in ("TRENDING", "RANGING", "VOLATILE")

    def test_insufficient_bars_returns_flat(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            _fake_artifacts(tmp, "TEST2")
            from src.strategy.signal_engine import SignalEngine
            engine = SignalEngine("TEST2", artifacts_dir=tmp)
            tiny_df = _make_ohlcv(5)
            result = engine.generate(tiny_df)
            assert result.signal == 0

    def test_filters_populated(self, engine_and_df):
        engine, df = engine_and_df
        result = engine.generate(df)
        # At least one filter should be evaluated
        total = len(result.filters_passed) + len(result.filters_failed)
        assert total > 0
