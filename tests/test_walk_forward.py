"""
Tests for the walk-forward / OOS validation framework.

Covers:
  - _build_folds_rolling   : correct fold count, boundaries, no leakage
  - _build_folds_expanding : train always anchored at 0 and grows
  - _build_folds_split     : single fold, correct split point
  - _aggregate             : all keys present, computed values correct
  - _train_on_slice        : returns (model, scaler, inv_label_map) in memory
  - FoldResult             : dataclass construction and profitable flag
  - No disk writes         : _train_on_slice must not touch artifacts/

Run with:  PYTHONPATH=/home/user/TradingEngine python3 -m pytest tests/test_walk_forward.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import (
    FoldResult,
    WalkForwardSummary,
    _aggregate,
    _build_folds_expanding,
    _build_folds_rolling,
    _build_folds_split,
    _train_on_slice,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_fold(
    idx: int,
    net_pnl: float,
    win_rate: float = 0.55,
    profit_factor: float = 1.3,
    sharpe: float = 0.4,
    sortino: float = 0.5,
    max_dd_pct: float = -2.5,
    n_trades: int = 30,
) -> FoldResult:
    return FoldResult(
        fold_idx=idx,
        mode="rolling",
        train_start="2020-01-01 00:00:00",
        train_end="2020-06-01 00:00:00",
        train_bars=5000,
        test_start="2020-06-01 00:00:00",
        test_end="2020-09-01 00:00:00",
        test_bars=2000,
        n_trades=n_trades,
        net_pnl=net_pnl,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd_pct,
        profitable=(net_pnl > 0),
    )


def _make_train_df(
    n: int = 600,
    n_features: int = 4,
    seed: int = 42,
    n_classes: int = 3,
) -> pd.DataFrame:
    """Synthetic DataFrame with feature columns + 'label'."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="1min", tz="UTC")
    cols = [f"f{i}" for i in range(n_features)]
    X    = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=cols, index=idx)
    y    = pd.Series(
        rng.choice(list(range(-(n_classes // 2), n_classes // 2 + 1))[:n_classes],
                   size=n),
        index=idx,
        name="label",
    )
    return X.join(y)


# ─────────────────────────────────────────────────────────────────────────────
# 1. _build_folds_rolling
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFoldsRolling:
    def test_fold_count_basic(self):
        # n=100, train=50, test=20, step=20
        # start=0: 0+50+20=70≤100 ✓  |  start=20: 20+50+20=90≤100 ✓
        # start=40: 40+50+20=110>100 ✗  →  2 folds
        folds = _build_folds_rolling(100, 50, 20, 20)
        assert len(folds) == 2

    def test_fold_count_exact_fit(self):
        # n=70, train=50, test=20, step=20  →  exactly 1 fold
        folds = _build_folds_rolling(70, 50, 20, 20)
        assert len(folds) == 1

    def test_empty_when_data_too_short(self):
        folds = _build_folds_rolling(60, 50, 20, 20)
        assert len(folds) == 0

    def test_no_leakage(self):
        """Train end must be ≤ test start for every fold."""
        folds = _build_folds_rolling(300, 100, 40, 40)
        for tr_s, tr_e, te_s, te_e in folds:
            assert tr_e <= te_s, (
                f"Leakage: train_end={tr_e} > test_start={te_s}"
            )

    def test_train_end_equals_test_start(self):
        """Rolling: train and test windows are contiguous (no gap)."""
        folds = _build_folds_rolling(200, 80, 30, 30)
        for tr_s, tr_e, te_s, te_e in folds:
            assert tr_e == te_s

    def test_non_overlapping_test_windows(self):
        """Default (step = test_bars): consecutive test windows don't overlap."""
        folds = _build_folds_rolling(400, 100, 50, 50)
        test_ends   = [te_e for _, _, _, te_e in folds]
        test_starts = [te_s for _, _, te_s, _ in folds]
        for i in range(len(folds) - 1):
            assert test_ends[i] <= test_starts[i + 1], (
                f"Fold {i} test [{test_starts[i]}:{test_ends[i]}] "
                f"overlaps fold {i+1} test [{test_starts[i+1]}:…]"
            )

    def test_overlapping_test_windows_allowed_when_step_small(self):
        """step < test_bars → overlapping test windows, but still no leakage."""
        folds = _build_folds_rolling(300, 100, 50, 10)
        assert len(folds) > 1
        for tr_s, tr_e, te_s, te_e in folds:
            assert tr_e <= te_s

    def test_train_window_fixed_size(self):
        """Rolling: every fold's train window has the same size."""
        folds = _build_folds_rolling(500, 200, 50, 50)
        sizes = [tr_e - tr_s for tr_s, tr_e, _, _ in folds]
        assert all(s == 200 for s in sizes)

    def test_boundaries_correct(self):
        """Spot-check exact fold boundaries."""
        folds = _build_folds_rolling(150, 80, 30, 30)
        # fold 0: (0, 80, 80, 110)
        # fold 1: (30, 110, 110, 140)
        assert folds[0] == (0, 80, 80, 110)
        assert folds[1] == (30, 110, 110, 140)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _build_folds_expanding
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFoldsExpanding:
    def test_fold_count_basic(self):
        # n=100, min_train=50, test=20, step=20
        # train_end=50: 50+20=70≤100 ✓  |  train_end=70: 70+20=90≤100 ✓
        # train_end=90: 90+20=110>100 ✗  →  2 folds
        folds = _build_folds_expanding(100, 50, 20, 20)
        assert len(folds) == 2

    def test_train_always_starts_at_zero(self):
        folds = _build_folds_expanding(300, 80, 40, 40)
        for tr_s, _, _, _ in folds:
            assert tr_s == 0, "Expanding mode: train must always start at index 0"

    def test_train_grows_each_fold(self):
        folds = _build_folds_expanding(400, 80, 40, 40)
        sizes = [tr_e - tr_s for tr_s, tr_e, _, _ in folds]
        for i in range(len(sizes) - 1):
            assert sizes[i + 1] > sizes[i], (
                f"Train size should grow: fold {i} size={sizes[i]}, "
                f"fold {i+1} size={sizes[i+1]}"
            )

    def test_no_leakage(self):
        folds = _build_folds_expanding(300, 80, 40, 40)
        for tr_s, tr_e, te_s, te_e in folds:
            assert tr_e <= te_s

    def test_empty_when_data_too_short(self):
        folds = _build_folds_expanding(60, 50, 20, 20)
        assert len(folds) == 0

    def test_boundaries_correct(self):
        """Spot-check exact expanding fold boundaries."""
        folds = _build_folds_expanding(200, 80, 40, 40)
        # fold 0: train [0:80], test [80:120]
        # fold 1: train [0:120], test [120:160]
        assert folds[0] == (0, 80, 80, 120)
        assert folds[1] == (0, 120, 120, 160)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _build_folds_split
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFoldsSplit:
    def test_always_one_fold(self):
        folds = _build_folds_split(100, 0.8)
        assert len(folds) == 1

    def test_split_point_correct(self):
        folds = _build_folds_split(100, 0.8)
        tr_s, tr_e, te_s, te_e = folds[0]
        assert tr_s == 0
        assert tr_e == 80      # int(100 * 0.8)
        assert te_s == 80
        assert te_e == 100

    def test_no_leakage(self):
        folds = _build_folds_split(100, 0.75)
        tr_s, tr_e, te_s, te_e = folds[0]
        assert tr_e <= te_s

    def test_full_dataset_covered(self):
        """Train + test together cover the entire dataset."""
        n     = 200
        folds = _build_folds_split(n, 0.7)
        tr_s, tr_e, te_s, te_e = folds[0]
        assert tr_s == 0
        assert te_e == n

    def test_degenerate_pct_zero_raises(self):
        with pytest.raises(ValueError):
            _build_folds_split(100, 0.0)

    def test_degenerate_pct_one_raises(self):
        with pytest.raises(ValueError):
            _build_folds_split(100, 1.0)

    def test_various_split_pcts(self):
        for pct in (0.5, 0.6, 0.7, 0.8, 0.9):
            folds = _build_folds_split(1000, pct)
            assert len(folds) == 1
            _, tr_e, te_s, _ = folds[0]
            assert tr_e == int(1000 * pct)
            assert te_s == int(1000 * pct)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _aggregate
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_empty_returns_empty_dict(self):
        assert _aggregate([]) == {}

    def test_all_required_keys_present(self):
        folds = [_make_fold(i, 100.0 * (i + 1)) for i in range(4)]
        agg   = _aggregate(folds)
        required = {
            "total_net_pnl", "avg_net_pnl_per_fold", "std_net_pnl",
            "n_profitable_folds", "pct_profitable_folds",
            "avg_win_rate", "median_win_rate",
            "avg_profit_factor", "median_profit_factor",
            "avg_sharpe", "std_sharpe", "avg_sortino",
            "avg_max_drawdown_pct", "worst_drawdown_pct",
            "avg_expectancy_usd", "avg_trades_per_day",
            "pnl_cv", "t_stat_pnl",
            "funded_ready",    # new primary verdict key
            "stability",       # legacy alias (same value as funded_ready)
            "n_folds",
        }
        missing = required - set(agg.keys())
        assert not missing, f"Missing aggregate keys: {missing}"

    def test_total_pnl_correct(self):
        folds = [_make_fold(i, 100.0) for i in range(5)]   # 5 × 100 = 500
        agg   = _aggregate(folds)
        assert abs(agg["total_net_pnl"] - 500.0) < 1e-6

    def test_pct_profitable_correct(self):
        folds = [
            _make_fold(0,  200.0),   # profitable
            _make_fold(1, -100.0),   # not
            _make_fold(2,  300.0),   # profitable
            _make_fold(3,  -50.0),   # not
        ]
        agg = _aggregate(folds)
        assert agg["n_profitable_folds"] == 2
        assert abs(agg["pct_profitable_folds"] - 0.5) < 1e-9

    def test_avg_win_rate_correct(self):
        folds = [_make_fold(i, 100.0, win_rate=0.60) for i in range(3)]
        agg   = _aggregate(folds)
        assert abs(agg["avg_win_rate"] - 0.60) < 1e-6

    def test_pnl_cv_none_when_mean_nonpositive(self):
        folds = [_make_fold(i, -100.0) for i in range(3)]
        agg   = _aggregate(folds)
        assert agg["pnl_cv"] is None

    def test_pnl_cv_positive_when_mean_positive(self):
        folds = [_make_fold(i, 100.0 + i * 10) for i in range(4)]
        agg   = _aggregate(folds)
        assert agg["pnl_cv"] is not None
        assert agg["pnl_cv"] >= 0.0

    def test_t_stat_none_for_single_fold(self):
        folds = [_make_fold(0, 500.0)]
        agg   = _aggregate(folds)
        assert agg["t_stat_pnl"] is None

    def test_t_stat_positive_for_consistent_profits(self):
        # Large positive PnLs, low variance → high t-stat
        folds = [_make_fold(i, 1000.0 + i * 5) for i in range(10)]
        agg   = _aggregate(folds)
        assert agg["t_stat_pnl"] is not None
        assert agg["t_stat_pnl"] > 1.65

    def test_stability_funded_ready(self):
        # 10 folds, all highly profitable with low variance → FUNDED-READY
        # profit_factor=2.0 (>= 1.5), pct=1.0 (>= 0.70), high t-stat, low CV
        folds = [
            _make_fold(i, 800.0 + i * 20, profit_factor=2.0)
            for i in range(10)
        ]
        agg = _aggregate(folds)
        assert agg["funded_ready"] == "FUNDED-READY"
        assert agg["stability"]    == "FUNDED-READY"   # legacy alias

    def test_stability_not_ready_all_losing(self):
        folds = [_make_fold(i, -200.0, profit_factor=0.5) for i in range(5)]
        agg   = _aggregate(folds)
        assert agg["funded_ready"] == "NOT READY"
        assert agg["stability"]    == "NOT READY"

    def test_stability_marginal(self):
        # Exactly 50% profitable, avg PF = 1.3 → MARGINAL
        # (pct_prof 0.5 >= 0.40, avg_pf 1.3 >= 1.10, but below PROMISING threshold)
        folds = [
            _make_fold(0,  100.0, profit_factor=1.3),
            _make_fold(1, -100.0, profit_factor=1.3),
            _make_fold(2,  100.0, profit_factor=1.3),
            _make_fold(3, -100.0, profit_factor=1.3),
        ]
        agg = _aggregate(folds)
        assert agg["funded_ready"] == "MARGINAL"

    def test_worst_drawdown_is_minimum(self):
        folds = [
            _make_fold(0, 100.0, max_dd_pct=-1.0),
            _make_fold(1, 100.0, max_dd_pct=-5.0),
            _make_fold(2, 100.0, max_dd_pct=-2.0),
        ]
        agg = _aggregate(folds)
        assert abs(agg["worst_drawdown_pct"] - (-5.0)) < 1e-6

    def test_n_folds_correct(self):
        folds = [_make_fold(i, 100.0) for i in range(7)]
        agg   = _aggregate(folds)
        assert agg["n_folds"] == 7

    def test_total_n_trades(self):
        folds = [_make_fold(i, 100.0, n_trades=20) for i in range(3)]
        agg   = _aggregate(folds)
        assert agg["total_n_trades"] == 60


# ─────────────────────────────────────────────────────────────────────────────
# 5. _train_on_slice (requires xgboost + sklearn)
# ─────────────────────────────────────────────────────────────────────────────


xgb = pytest.importorskip("xgboost", reason="xgboost not installed")


class TestTrainOnSlice:
    FEATURE_NAMES = ["f0", "f1", "f2", "f3"]

    def _df(self, n: int = 600, seed: int = 0) -> pd.DataFrame:
        return _make_train_df(n, n_features=4, seed=seed, n_classes=3)

    def test_returns_three_tuple(self):
        result = _train_on_slice(self._df(), self.FEATURE_NAMES, n_trials=0)
        assert len(result) == 3

    def test_model_has_predict_proba(self):
        model, _, _ = _train_on_slice(self._df(), self.FEATURE_NAMES, n_trials=0)
        assert callable(getattr(model, "predict_proba", None))

    def test_scaler_is_fitted(self):
        _, scaler, _ = _train_on_slice(self._df(), self.FEATURE_NAMES, n_trials=0)
        # RobustScaler stores center_ after fitting
        assert hasattr(scaler, "center_"), "Scaler must be fitted (center_ missing)"

    def test_inv_label_map_string_keys_int_values(self):
        _, _, inv_map = _train_on_slice(self._df(), self.FEATURE_NAMES, n_trials=0)
        for k, v in inv_map.items():
            assert isinstance(k, str), f"Key {k!r} must be str"
            assert isinstance(v, int), f"Value {v!r} for key {k!r} must be int"

    def test_inv_label_map_covers_all_classes(self):
        df = self._df()
        _, _, inv_map = _train_on_slice(df, self.FEATURE_NAMES, n_trials=0)
        # Every encoded class index must appear
        for i in range(len(inv_map)):
            assert str(i) in inv_map, f"Class index {i} missing from inv_label_map"

    def test_predict_proba_shape(self):
        df  = self._df(n=300)
        model, scaler, inv_map = _train_on_slice(df, self.FEATURE_NAMES, n_trials=0)
        X   = df[self.FEATURE_NAMES].values
        proba = model.predict_proba(scaler.transform(df[self.FEATURE_NAMES]))
        n_classes = len(inv_map)
        assert proba.shape == (len(df), n_classes)

    def test_no_disk_artifacts_written(self):
        """_train_on_slice must not write to artifacts/ directory."""
        arts = Path("artifacts")
        before: dict = {}
        if arts.exists():
            before = {p: p.stat().st_mtime for p in arts.rglob("*") if p.is_file()}

        _train_on_slice(self._df(), self.FEATURE_NAMES, n_trials=0)

        after: dict = {}
        if arts.exists():
            after = {p: p.stat().st_mtime for p in arts.rglob("*") if p.is_file()}

        changed = {str(p) for p in after if after[p] != before.get(p, -1.0)}
        new     = {str(p) for p in after if p not in before}
        written = changed | new
        assert not written, (
            f"_train_on_slice must not write to artifacts/. Modified: {written}"
        )

    def test_two_class_slice_works(self):
        """Slices with only 2 of 3 classes are valid (common in WF)."""
        df = _make_train_df(n=400, n_classes=2, seed=7)  # only -1 and 1
        result = _train_on_slice(df, self.FEATURE_NAMES, n_trials=0)
        assert len(result) == 3

    def test_single_class_raises(self):
        """Slices with a single label class must raise ValueError."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=200, freq="1min", tz="UTC")
        df  = pd.DataFrame(
            {"f0": rng.normal(size=200), "f1": rng.normal(size=200),
             "f2": rng.normal(size=200), "f3": rng.normal(size=200),
             "label": np.zeros(200, dtype=int)},
            index=idx,
        )
        with pytest.raises(ValueError, match="unique label class"):
            _train_on_slice(df, self.FEATURE_NAMES, n_trials=0)

    def test_missing_feature_columns_raise(self):
        df = self._df()
        with pytest.raises((ValueError, KeyError)):
            _train_on_slice(df, ["nonexistent_col"], n_trials=0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FoldResult / WalkForwardSummary dataclass sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestDataclasses:
    def test_fold_result_profitable_flag_true(self):
        fr = _make_fold(0, 100.0)
        assert fr.profitable is True

    def test_fold_result_profitable_flag_false(self):
        fr = _make_fold(0, -50.0)
        assert fr.profitable is False

    def test_fold_result_zero_pnl_not_profitable(self):
        fr = _make_fold(0, 0.0)
        # 0.0 > 0 is False
        assert fr.profitable is False

    def test_walk_forward_summary_defaults(self):
        s = WalkForwardSummary(
            symbol="ES", mode="rolling", n_folds=0,
            train_bars=10_000, test_bars=2_000,
        )
        assert s.folds == []
        assert s.aggregate == {}


# ─────────────────────────────────────────────────────────────────────────────
# 7. No-leakage property across all modes (parametrised)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode,folds_fn,kwargs", [
    ("rolling",   _build_folds_rolling,   {"n": 500, "train_bars": 150, "test_bars": 50, "step_bars": 50}),
    ("expanding", _build_folds_expanding, {"n": 500, "min_train_bars": 150, "test_bars": 50, "step_bars": 50}),
    ("split",     _build_folds_split,     {"n": 500, "split_pct": 0.8}),
])
def test_no_leakage_parametrised(mode, folds_fn, kwargs):
    folds = folds_fn(**kwargs)
    assert len(folds) > 0, f"Expected at least 1 fold for mode={mode}"
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_e <= te_s, (
            f"[{mode}] Leakage: train_end={tr_e} > test_start={te_s}"
        )
        assert tr_s < tr_e,   f"[{mode}] Empty train window"
        assert te_s < te_e,   f"[{mode}] Empty test window"
