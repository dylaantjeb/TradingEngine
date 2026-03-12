"""
Unit tests for src/training/train.py — trading-quality scorer.

Tests cover:
  - _coverage_score      : shape of the piecewise coverage curve
  - _activity_score      : shape of the trade-count curve
  - _normalized_pf       : profit factor normalization
  - _ppt_score           : per-trade quality score
  - _compute_trading_stats:
      • hard-constraint failures (sparse model, low coverage, low PF, low accuracy)
      • all hard constraints pass → positive score
      • determinism (same inputs → same output)
      • sparse model (10 / 4135 bars) scores well below a healthy model
      • sparse model cannot beat a model with slightly lower accuracy but
        healthy trade coverage (the key regression scenario)
  - _trading_quality_score: thin wrapper consistency

All tests are pure-function tests that do NOT train an actual XGBoost model.
Proba arrays are constructed synthetically.

Run with:
  PYTHONPATH=/home/user/TradingEngine python3 -m pytest tests/test_train.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.train import (
    _HC_MIN_COVERAGE,
    _HC_MIN_DIR_ACC,
    _HC_MIN_PF,
    _HC_MIN_TRADES,
    _HC_PENALTY_MEDIUM,
    _HC_PENALTY_SEVERE,
    _activity_score,
    _compute_trading_stats,
    _coverage_score,
    _normalized_pf,
    _ppt_score,
    _trading_quality_score,
)

# Standard inv_label_map used by all synthetic helpers:
# encoded 0 → raw -1 (short), 1 → raw 0 (flat), 2 → raw +1 (long)
_INV = {0: -1, 1: 0, 2: 1}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic proba builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_proba(
    n_bars: int,
    n_confident_trades: int,
    win_rate: float = 0.55,
    conf_level: float = 0.75,
    min_conf: float = 0.65,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic (proba, y_raw) pair where exactly `n_confident_trades`
    bars have confidence >= min_conf and a non-flat predicted signal.

    Returns
    -------
    proba   : (n_bars, 3) float array
    y_raw   : (n_bars,) int array in {-1, 0, 1}
    """
    rng = np.random.default_rng(seed)

    # Start everyone as "flat with low confidence"
    proba = np.zeros((n_bars, 3), dtype=float)
    proba[:, 1] = 1.0  # all predict encoded-class 1 (flat) with 100% conf

    y_raw = np.zeros(n_bars, dtype=int)

    if n_confident_trades == 0:
        return proba, y_raw

    # Select which bars become confident directional trades
    trade_idx = rng.choice(n_bars, size=n_confident_trades, replace=False)
    trade_idx.sort()

    n_wins = int(round(n_confident_trades * win_rate))

    for i, bar in enumerate(trade_idx):
        direction = 2 if rng.random() > 0.5 else 0  # encoded long (2) or short (0)
        raw_dir   = _INV[direction]

        # Set high confidence for this direction
        proba[bar] = [0.0, 0.0, 0.0]
        proba[bar, direction] = conf_level
        proba[bar, 1]         = 1.0 - conf_level  # spread remainder to flat

        # True label: win → same direction, loss → opposite
        if i < n_wins:
            y_raw[bar] = raw_dir       # win
        else:
            y_raw[bar] = -raw_dir      # loss (opposite direction)

    return proba, y_raw


# ─────────────────────────────────────────────────────────────────────────────
# 1. _coverage_score
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverageScore:
    def test_zero_coverage_is_zero(self):
        assert _coverage_score(0.0) == 0.0

    def test_below_1pct_is_very_low(self):
        # 0.5 % → should be at most 0.10
        assert _coverage_score(0.005) < 0.10

    def test_at_1pct_is_low(self):
        # 1% is still below the sweet spot
        score = _coverage_score(0.01)
        assert score <= 0.10 + 1e-9    # at boundary of the ramp

    def test_below_3pct_below_full(self):
        # 2 % — on the ramp, should be well below full credit
        score = _coverage_score(0.02)
        assert score < 1.0
        assert score > 0.10

    def test_at_3pct_is_full_credit(self):
        assert abs(_coverage_score(0.03) - 1.0) < 1e-9

    def test_midrange_is_full_credit(self):
        for cov in (0.05, 0.08, 0.10, 0.15, 0.20):
            assert abs(_coverage_score(cov) - 1.0) < 1e-9, f"coverage={cov:.0%}"

    def test_overtrading_decreases_score(self):
        # 25 % > 20 % upper bound → score should drop below 1
        assert _coverage_score(0.25) < 1.0

    def test_heavy_overtrading_still_positive(self):
        # Even extreme overtrading should not reach 0 (no zero-division)
        assert _coverage_score(0.50) > 0.0

    def test_monotone_in_sparse_regime(self):
        # More trades in sparse region → better coverage score
        assert _coverage_score(0.02) > _coverage_score(0.01)
        assert _coverage_score(0.01) > _coverage_score(0.005)

    def test_monotone_decline_in_overtrade_regime(self):
        assert _coverage_score(0.30) < _coverage_score(0.25)
        assert _coverage_score(0.25) < _coverage_score(0.20)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _activity_score
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityScore:
    def test_zero_trades_is_zero(self):
        assert _activity_score(0) == 0.0

    def test_below_10_is_zero(self):
        for n in (1, 5, 9):
            assert _activity_score(n) == 0.0, f"n={n}"

    def test_at_10_starts_ramping(self):
        assert _activity_score(10) == 0.0     # ramp starts at 10 (inclusive edge)

    def test_ramp_between_10_and_50(self):
        # Values strictly between 10 and 50 should be in (0, 0.80)
        assert 0.0 < _activity_score(30) < 0.80

    def test_at_50_is_near_0_8(self):
        # 50 trades = bottom of target range
        assert abs(_activity_score(50) - 0.80) < 1e-9

    def test_at_200_is_full(self):
        assert abs(_activity_score(200) - 1.0) < 1e-9

    def test_above_200_stays_full(self):
        for n in (250, 500, 1000):
            assert _activity_score(n) == 1.0, f"n={n}"

    def test_monotone_increasing(self):
        prev = -1.0
        for n in (0, 5, 10, 30, 50, 100, 200, 500):
            cur = _activity_score(n)
            assert cur >= prev, f"not monotone at n={n}: {cur} < {prev}"
            prev = cur


# ─────────────────────────────────────────────────────────────────────────────
# 3. _normalized_pf and _ppt_score
# ─────────────────────────────────────────────────────────────────────────────

class TestNormPfAndPpt:
    def test_pf_below_one_is_zero(self):
        for pf in (0.0, 0.5, 0.99, 1.0):
            assert _normalized_pf(pf) == 0.0, f"pf={pf}"

    def test_pf_2_is_half(self):
        assert abs(_normalized_pf(2.0) - 0.5) < 1e-9

    def test_pf_3_is_full(self):
        assert abs(_normalized_pf(3.0) - 1.0) < 1e-9

    def test_pf_above_3_capped_at_one(self):
        for pf in (4.0, 10.0, 100.0):
            assert _normalized_pf(pf) == 1.0, f"pf={pf}"

    def test_ppt_50pct_wr_is_0_5(self):
        assert abs(_ppt_score(50, 100) - 0.5) < 1e-9

    def test_ppt_all_wins_is_1(self):
        assert abs(_ppt_score(100, 100) - 1.0) < 1e-9

    def test_ppt_all_losses_is_0(self):
        assert abs(_ppt_score(0, 100) - 0.0) < 1e-9

    def test_ppt_zero_trades_is_zero(self):
        assert _ppt_score(0, 0) == 0.0

    def test_ppt_higher_wr_gives_higher_score(self):
        assert _ppt_score(60, 100) > _ppt_score(55, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _compute_trading_stats — hard constraint failures
# ─────────────────────────────────────────────────────────────────────────────

class TestHardConstraints:
    """Each test verifies that failing a single hard constraint
    returns a negative trading_quality and hc_passed=False."""

    N_VAL = 4135   # typical ES val set size

    def test_too_few_trades_fails(self):
        # 10 trades / 4135 bars = 0.24% coverage — fails BOTH trade_count AND coverage
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=10, win_rate=0.80)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["hard_constraints_passed"] is False
        assert stats["trading_quality"] < 0.0
        assert any("trade_count" in f or "coverage" in f
                   for f in stats["hard_constraint_failures"])

    def test_sparse_model_reported_correctly(self):
        # Confirm the exact scenario from the bug report: 10/4135 bars
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=10, win_rate=0.90)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["n_trades"] == 10
        assert stats["n_val_bars"] == self.N_VAL
        assert stats["trade_coverage_pct"] < 1.0      # well below 1%
        assert stats["hard_constraints_passed"] is False
        assert stats["trading_quality"] <= _HC_PENALTY_SEVERE

    def test_below_50_trades_fails_even_with_ok_coverage(self):
        # 40 trades / 1000 bars = 4.0% coverage (above 3% HC) but count < 50
        proba, y_raw = _make_proba(1000, n_confident_trades=40, win_rate=0.65)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=1000)
        assert stats["hard_constraints_passed"] is False
        assert "trade_count=40" in " ".join(stats["hard_constraint_failures"])

    def test_below_3pct_coverage_fails_even_with_50_trades(self):
        # 50 trades / 4135 bars = 1.2% coverage — below 3% HC
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=50, win_rate=0.65)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["trade_coverage_pct"] < 3.0
        assert stats["hard_constraints_passed"] is False
        assert any("coverage" in f for f in stats["hard_constraint_failures"])

    def test_low_profit_factor_fails(self):
        # 200 trades / 2000 bars (10% coverage), but only 45% win rate → PF < 1.0
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=0.45)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["profit_factor"] < _HC_MIN_PF
        assert stats["hard_constraints_passed"] is False
        assert any("profit_factor" in f for f in stats["hard_constraint_failures"])

    def test_low_dir_accuracy_fails(self):
        # 200 trades / 2000 bars, 50% win rate → accuracy < 52% HC
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=0.50)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["dir_accuracy"] < _HC_MIN_DIR_ACC
        assert stats["hard_constraints_passed"] is False
        assert any("dir_accuracy" in f for f in stats["hard_constraint_failures"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. _compute_trading_stats — passing models
# ─────────────────────────────────────────────────────────────────────────────

class TestPassingModels:
    """Verify that a model meeting all hard constraints gets a positive score
    and that all sub-components are present."""

    def _healthy_proba(self, n_bars=2000, n_trades=120, win_rate=0.60, seed=1):
        return _make_proba(n_bars, n_trades, win_rate=win_rate, seed=seed)

    def test_all_hc_pass(self):
        proba, y_raw = self._healthy_proba()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["hard_constraints_passed"] is True
        assert stats["hard_constraint_failures"] == []

    def test_score_is_positive(self):
        proba, y_raw = self._healthy_proba()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["trading_quality"] > 0.0

    def test_score_in_unit_range(self):
        proba, y_raw = self._healthy_proba()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert 0.0 <= stats["trading_quality"] <= 1.0

    def test_score_components_all_present(self):
        proba, y_raw = self._healthy_proba()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        required = {"dir_accuracy", "normalized_pf", "trade_coverage",
                    "activity", "pnl_per_trade"}
        assert required <= set(stats["score_components"].keys())

    def test_score_components_all_in_unit_range(self):
        proba, y_raw = self._healthy_proba()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        for k, v in stats["score_components"].items():
            assert 0.0 <= v <= 1.0, f"component {k!r} = {v} not in [0,1]"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Key ordering invariant: sparse model < healthy model
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreOrdering:
    """The central regression test: a model with only 10 trades / 4135 bars
    (even with perfect accuracy) must score lower than a model with 120/2000
    trades and 60% accuracy."""

    N_VAL = 4135

    def test_sparse_loses_to_healthy(self):
        # "Sparse" — the exact scenario from the bug report
        sparse_proba, sparse_y = _make_proba(
            self.N_VAL, n_confident_trades=10, win_rate=0.90, seed=42
        )
        sparse_stats = _compute_trading_stats(
            sparse_proba, sparse_y, _INV, n_val_bars=self.N_VAL
        )

        # "Healthy" — realistic trade frequency, modest accuracy
        healthy_proba, healthy_y = _make_proba(
            self.N_VAL, n_confident_trades=200, win_rate=0.58, seed=42
        )
        healthy_stats = _compute_trading_stats(
            healthy_proba, healthy_y, _INV, n_val_bars=self.N_VAL
        )

        assert sparse_stats["trading_quality"] < healthy_stats["trading_quality"], (
            f"Sparse model score {sparse_stats['trading_quality']:.4f} should be "
            f"< healthy model score {healthy_stats['trading_quality']:.4f}"
        )

    def test_sparse_has_negative_score(self):
        # Negative score is the key signal to Optuna to avoid sparse models
        sparse_proba, sparse_y = _make_proba(
            self.N_VAL, n_confident_trades=10, win_rate=0.90, seed=42
        )
        stats = _compute_trading_stats(
            sparse_proba, sparse_y, _INV, n_val_bars=self.N_VAL
        )
        assert stats["trading_quality"] < 0.0, (
            f"Sparse model must have negative score, got {stats['trading_quality']:.4f}"
        )

    def test_more_trades_beats_fewer_when_both_accurate(self):
        # 100 trades vs 60 trades, same win rate — more trades should win
        # (both above the 50-trade HC threshold)
        proba_100, y_100 = _make_proba(2000, n_confident_trades=100, win_rate=0.60, seed=5)
        proba_60,  y_60  = _make_proba(2000, n_confident_trades=60,  win_rate=0.60, seed=5)

        stats_100 = _compute_trading_stats(proba_100, y_100, _INV, n_val_bars=2000)
        stats_60  = _compute_trading_stats(proba_60,  y_60,  _INV, n_val_bars=2000)

        # Both should pass hard constraints
        assert stats_100["hard_constraints_passed"]
        assert stats_60["hard_constraints_passed"]
        # 100 trades ≥ 60 trades in score (activity + coverage both reward more)
        assert stats_100["trading_quality"] >= stats_60["trading_quality"]

    def test_higher_accuracy_beats_lower_when_both_active(self):
        # Same trade count, higher accuracy should win
        proba_hi, y_hi = _make_proba(2000, n_confident_trades=100, win_rate=0.70, seed=7)
        proba_lo, y_lo = _make_proba(2000, n_confident_trades=100, win_rate=0.55, seed=7)

        stats_hi = _compute_trading_stats(proba_hi, y_hi, _INV, n_val_bars=2000)
        stats_lo = _compute_trading_stats(proba_lo, y_lo, _INV, n_val_bars=2000)

        assert stats_hi["hard_constraints_passed"]
        assert stats_hi["trading_quality"] > stats_lo["trading_quality"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Same inputs must always produce the same output (no randomness in scorer)."""

    def test_score_is_deterministic(self):
        proba, y_raw = _make_proba(2000, n_confident_trades=100, win_rate=0.60, seed=99)
        scores = [
            _trading_quality_score(proba, y_raw, _INV, n_val_bars=2000)
            for _ in range(5)
        ]
        assert len(set(scores)) == 1, f"Non-deterministic scores: {scores}"

    def test_stats_deterministic(self):
        proba, y_raw = _make_proba(3000, n_confident_trades=150, win_rate=0.62, seed=11)
        stats_a = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=3000)
        stats_b = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=3000)
        assert stats_a["trading_quality"] == stats_b["trading_quality"]
        assert stats_a["hard_constraints_passed"] == stats_b["hard_constraints_passed"]


# ─────────────────────────────────────────────────────────────────────────────
# 8. _trading_quality_score wrapper consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingQualityScoreWrapper:
    def test_wrapper_matches_stats(self):
        """_trading_quality_score must return exactly stats['trading_quality']."""
        proba, y_raw = _make_proba(2000, n_confident_trades=100, win_rate=0.60, seed=3)
        score = _trading_quality_score(proba, y_raw, _INV, n_val_bars=2000)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert score == stats["trading_quality"]

    def test_wrapper_negative_for_sparse(self):
        proba, y_raw = _make_proba(4000, n_confident_trades=5, win_rate=1.0, seed=0)
        score = _trading_quality_score(proba, y_raw, _INV, n_val_bars=4000)
        assert score < 0.0

    def test_wrapper_defaults_n_val_bars_to_proba_len(self):
        proba, y_raw = _make_proba(500, n_confident_trades=30, win_rate=0.65, seed=0)
        # When n_val_bars is None, should use len(proba)=500
        score_none = _trading_quality_score(proba, y_raw, _INV, n_val_bars=None)
        score_500  = _trading_quality_score(proba, y_raw, _INV, n_val_bars=500)
        assert score_none == score_500


# ─────────────────────────────────────────────────────────────────────────────
# 9. Penalty magnitudes
# ─────────────────────────────────────────────────────────────────────────────

class TestPenaltyMagnitudes:
    """Verify the hard-constraint penalty values are as specified."""

    def test_severe_penalty_value(self):
        # trade_count < 50 → penalty = abs(_HC_PENALTY_SEVERE) = 1.0
        proba, y_raw = _make_proba(4135, n_confident_trades=10, win_rate=0.80)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=4135)
        expected = -abs(_HC_PENALTY_SEVERE)
        assert stats["trading_quality"] == pytest.approx(expected, abs=1e-6)

    def test_medium_penalty_for_low_accuracy_only(self):
        # 200 trades / 2000 bars (10% coverage > 3%), but 50% win rate → dir_acc HC fails
        # PF will also fail (PF < 1.10 at 50% win rate)
        # Since both medium HCs fail, penalty is still abs(_HC_PENALTY_MEDIUM)
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=0.50, seed=1)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        # At 50% win rate: pf = 1.0 < 1.10, dir_acc = 0.50 < 0.52 — both medium
        assert stats["hard_constraints_passed"] is False
        assert stats["trading_quality"] == pytest.approx(
            -abs(_HC_PENALTY_MEDIUM), abs=1e-6
        )
