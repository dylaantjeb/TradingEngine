"""
Unit tests for src/features/regime.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features.regime import Regime, detect_regime, add_regime_column


def _make_df(n: int = 200, trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    close = 5000.0 * np.exp(np.cumsum(rng.normal(trend, 0.001, n)))
    spread = 0.25
    high = close + rng.uniform(0, 0.5, n)
    low  = close - rng.uniform(0, 0.5, n)
    open_ = close * np.exp(rng.normal(0, 0.0005, n))
    volume = rng.integers(100, 2000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestDetectRegime:
    def test_returns_series(self):
        df = _make_df(200)
        regimes = detect_regime(df)
        assert isinstance(regimes, pd.Series)

    def test_same_length_as_input(self):
        df = _make_df(200)
        regimes = detect_regime(df)
        assert len(regimes) == len(df)

    def test_same_index_as_input(self):
        df = _make_df(200)
        regimes = detect_regime(df)
        assert regimes.index.equals(df.index)

    def test_valid_regime_values(self):
        df = _make_df(200)
        regimes = detect_regime(df)
        valid = {Regime.TRENDING, Regime.RANGING, Regime.VOLATILE}
        assert set(regimes.unique()).issubset(valid)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"open": [1.0] * 50, "high": [1.1] * 50},
                          index=pd.date_range("2024-01-01", periods=50, freq="1min"))
        with pytest.raises(ValueError, match="missing"):
            detect_regime(df)

    def test_strong_trend_gives_trending(self):
        """Very strong uptrend should produce at least some TRENDING bars."""
        df = _make_df(500, trend=0.005, seed=1)
        regimes = detect_regime(df, adx_trend_threshold=20)
        assert (regimes == Regime.TRENDING).any()

    def test_add_regime_column(self):
        df = _make_df(200)
        out = add_regime_column(df)
        assert "regime" in out.columns
        assert len(out) == len(df)
        # Original df is not modified
        assert "regime" not in df.columns

    def test_add_regime_column_values(self):
        df = _make_df(200)
        out = add_regime_column(df)
        valid = {Regime.TRENDING, Regime.RANGING, Regime.VOLATILE}
        assert set(out["regime"].unique()).issubset(valid)
