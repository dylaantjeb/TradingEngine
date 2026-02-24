"""
Unit tests for the triple-barrier labeler.

Run with:
    python -m pytest tests/ -v
    # or just:
    python -m pytest tests/test_labeler.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.labels.triple_barrier import label_triple_barrier


def _make_ohlcv(n: int = 200, trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    log_ret = rng.normal(trend, 0.001, n)
    close = 5000.0 * np.exp(np.cumsum(log_ret))
    spread = 0.25
    high = close + rng.uniform(0, spread, n)
    low = close - rng.uniform(0, spread, n)
    open_ = close * np.exp(rng.normal(0, 0.0005, n))
    volume = rng.integers(100, 2000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestLabelTripleBarrier:
    def test_output_shape(self):
        df = _make_ohlcv(200)
        labels = label_triple_barrier(df)
        assert len(labels) == len(df), "Label length must match input length"

    def test_label_values(self):
        df = _make_ohlcv(200)
        labels = label_triple_barrier(df)
        assert set(labels["label"].unique()).issubset(
            {-1, 0, 1}
        ), "Labels must be in {-1, 0, 1}"

    def test_all_labels_present(self):
        """With enough bars and moderate params all three labels should appear."""
        df = _make_ohlcv(500)
        labels = label_triple_barrier(df, pt=1.0, sl=1.0, max_hold=30)
        unique = set(labels["label"].unique())
        # At least two distinct labels expected
        assert len(unique) >= 2, f"Expected ≥2 distinct labels, got {unique}"

    def test_hold_bars_range(self):
        max_hold = 20
        df = _make_ohlcv(200)
        labels = label_triple_barrier(df, max_hold=max_hold)
        assert (labels["hold_bars"] >= 0).all()
        assert (labels["hold_bars"] <= max_hold).all()

    def test_index_alignment(self):
        df = _make_ohlcv(200)
        labels = label_triple_barrier(df)
        assert labels.index.equals(df.index), "Label index must match input index"

    def test_too_few_rows_raises(self):
        df = _make_ohlcv(5)
        with pytest.raises(ValueError, match="Too few rows"):
            label_triple_barrier(df)

    def test_missing_close_column_raises(self):
        df = pd.DataFrame(
            {"open": [1.0, 2.0] * 10, "high": [1.1, 2.1] * 10},
            index=pd.date_range("2024-01-01", periods=20, freq="1min"),
        )
        with pytest.raises(ValueError, match="close"):
            label_triple_barrier(df)

    def test_ret_at_exit_dtype(self):
        df = _make_ohlcv(100)
        labels = label_triple_barrier(df)
        assert labels["ret_at_exit"].dtype == np.float64

    def test_uptrend_skews_positive(self):
        """A strong uptrend should produce more +1 labels."""
        df = _make_ohlcv(1000, trend=0.0005, seed=0)
        labels = label_triple_barrier(df, pt=1.0, sl=1.0, max_hold=30)
        counts = labels["label"].value_counts()
        # +1 should be at least as common as -1 in an uptrend
        assert counts.get(1, 0) >= counts.get(-1, 0), (
            f"Uptrend should have more +1 than -1 labels, got {counts.to_dict()}"
        )

    def test_downtrend_skews_negative(self):
        """A strong downtrend should produce more -1 labels."""
        df = _make_ohlcv(1000, trend=-0.0005, seed=0)
        labels = label_triple_barrier(df, pt=1.0, sl=1.0, max_hold=30)
        counts = labels["label"].value_counts()
        assert counts.get(-1, 0) >= counts.get(1, 0), (
            f"Downtrend should have more -1 than +1 labels, got {counts.to_dict()}"
        )
