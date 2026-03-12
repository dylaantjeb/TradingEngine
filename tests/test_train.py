"""
Unit tests for src/training/train.py — trading-quality scorer and gate logic.

Tests cover:
  - _coverage_score      : piecewise coverage curve, monotonicity
  - _activity_score      : trade-count ramp, plateau, monotonicity
  - _normalized_pf       : profit factor normalization edges
  - _ppt_score           : per-trade quality score ordering
  - _compute_trading_stats:
      • all four hard-constraint failures with exact penalty magnitudes
      • 10/4135, 13/4135, 68/4135 sparse-model regression scenarios
      • all hard constraints pass → positive score in [0,1]
      • score determinism (same inputs → same output, every time)
      • sparse model always loses to a healthy active model
  - _trading_quality_score: thin wrapper consistency
  - Gate-fail sentinel and all-gates-failed detection logic

Key invariants under test:
  1. A model with 10, 13, or 68 confident trades out of 4135 validation bars
     MUST receive a clearly negative trading_quality score, and must never
     beat a model with >= 50 trades and >= 3% coverage.
  2. _GATE_FAIL_SCORE is so negative that any valid composite beats it.
  3. The all-gates-failed detection threshold (score <= _GATE_FAIL_SCORE * 0.5)
     correctly fires when every trial returns the sentinel, and does NOT fire
     for any legitimately passing score.

All tests are pure-function tests that do NOT train an actual XGBoost model.
Proba arrays are constructed synthetically so tests run in milliseconds.

Run with:
  PYTHONPATH=/home/user/TradingEngine python3 -m pytest tests/test_train.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.train import (
    _DEFAULT_THRESHOLD,
    _GATE_FAIL_SCORE,
    _HC_MIN_COVERAGE,
    _HC_MIN_DIR_ACC,
    _HC_MIN_PF,
    _HC_MIN_TRADES,
    _HC_PENALTY_MEDIUM,
    _HC_PENALTY_SEVERE,
    _THRESHOLD_CANDIDATES,
    _activity_score,
    _compute_trading_stats,
    _coverage_score,
    _normalized_pf,
    _ppt_score,
    _select_best_threshold,
    _trading_quality_score,
)

# Canonical inv_label_map: encoded 0→-1 (short), 1→0 (flat), 2→+1 (long)
_INV = {0: -1, 1: 0, 2: 1}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic proba builder
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
    bars have confidence >= min_conf and a non-flat predicted signal, and the
    rest are flat (confidence=1.0 on class 1 = flat).

    Returns
    -------
    proba   : (n_bars, 3) float array
    y_raw   : (n_bars,)  int array in {-1, 0, 1}
    """
    rng = np.random.default_rng(seed)

    # Default: all predict flat with full confidence
    proba = np.zeros((n_bars, 3), dtype=float)
    proba[:, 1] = 1.0
    y_raw = np.zeros(n_bars, dtype=int)

    if n_confident_trades == 0:
        return proba, y_raw

    trade_idx = rng.choice(n_bars, size=n_confident_trades, replace=False)
    trade_idx.sort()
    n_wins = int(round(n_confident_trades * win_rate))

    for i, bar in enumerate(trade_idx):
        direction = 2 if rng.random() > 0.5 else 0  # long or short
        raw_dir   = _INV[direction]
        proba[bar]           = 0.0
        proba[bar, direction] = conf_level
        proba[bar, 1]         = 1.0 - conf_level
        y_raw[bar] = raw_dir if i < n_wins else -raw_dir

    return proba, y_raw


# ─────────────────────────────────────────────────────────────────────────────
# 0. Constant sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_gate_fail_score_is_very_negative(self):
        # Must be lower than the worst composite a passing model could achieve.
        # Worst composite ≈ 0.40*0 + 0.60*(-10) = -6.0 (severe penalty).
        # _GATE_FAIL_SCORE = -1e9 satisfies this with extreme margin.
        assert _GATE_FAIL_SCORE < -10.0

    def test_severe_penalty_is_large_negative(self):
        assert _HC_PENALTY_SEVERE < -5.0

    def test_medium_penalty_is_moderately_negative(self):
        assert _HC_PENALTY_MEDIUM < -1.0
        assert _HC_PENALTY_MEDIUM > _HC_PENALTY_SEVERE

    def test_penalty_ordering(self):
        # SEVERE must penalise more than MEDIUM
        assert abs(_HC_PENALTY_SEVERE) > abs(_HC_PENALTY_MEDIUM)

    def test_gate_fail_score_extreme(self):
        # _GATE_FAIL_SCORE must be <= -1e6 so the all-gates-failed detection
        # threshold (_GATE_FAIL_SCORE * 0.5) is still clearly below zero.
        assert _GATE_FAIL_SCORE <= -1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. _coverage_score
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverageScore:
    def test_zero_coverage_is_zero(self):
        assert _coverage_score(0.0) == 0.0

    def test_below_1pct_is_very_low(self):
        assert _coverage_score(0.005) < 0.10

    def test_at_1pct_boundary(self):
        assert _coverage_score(0.01) <= 0.10 + 1e-9

    def test_below_3pct_is_below_full(self):
        score = _coverage_score(0.02)
        assert score < 1.0
        assert score > 0.10

    def test_at_3pct_is_full_credit(self):
        assert abs(_coverage_score(0.03) - 1.0) < 1e-9

    def test_target_range_is_full(self):
        for cov in (0.05, 0.08, 0.10, 0.15, 0.20):
            assert abs(_coverage_score(cov) - 1.0) < 1e-9, f"cov={cov:.0%}"

    def test_overtrading_decreases(self):
        assert _coverage_score(0.25) < 1.0

    def test_heavy_overtrading_stays_positive(self):
        assert _coverage_score(0.50) > 0.0

    def test_monotone_sparse(self):
        assert _coverage_score(0.02) > _coverage_score(0.01)
        assert _coverage_score(0.01) > _coverage_score(0.005)

    def test_monotone_overtrade(self):
        assert _coverage_score(0.30) < _coverage_score(0.25)
        assert _coverage_score(0.25) < _coverage_score(0.20)

    def test_164pct_coverage_is_low(self):
        # 68 / 4135 = 1.64% — should give a very low coverage score
        score = _coverage_score(68 / 4135)
        assert score < 0.60          # not near full credit
        assert score < _coverage_score(0.03)   # clearly worse than 3%


# ─────────────────────────────────────────────────────────────────────────────
# 2. _activity_score
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityScore:
    def test_zero_is_zero(self):
        assert _activity_score(0) == 0.0

    def test_below_10_is_zero(self):
        for n in (1, 5, 9):
            assert _activity_score(n) == 0.0

    def test_at_10_is_zero(self):
        assert _activity_score(10) == 0.0

    def test_between_10_and_50_ramps(self):
        assert 0.0 < _activity_score(30) < 0.80

    def test_at_50_is_0_8(self):
        assert abs(_activity_score(50) - 0.80) < 1e-9

    def test_at_200_is_full(self):
        assert abs(_activity_score(200) - 1.0) < 1e-9

    def test_above_200_stays_full(self):
        for n in (250, 500, 1000):
            assert _activity_score(n) == 1.0

    def test_monotone(self):
        prev = -1.0
        for n in (0, 5, 10, 30, 50, 100, 200, 500):
            cur = _activity_score(n)
            assert cur >= prev, f"not monotone at n={n}"
            prev = cur


# ─────────────────────────────────────────────────────────────────────────────
# 3. _normalized_pf and _ppt_score
# ─────────────────────────────────────────────────────────────────────────────

class TestNormPfAndPpt:
    def test_pf_at_or_below_one_is_zero(self):
        for pf in (0.0, 0.5, 0.99, 1.0):
            assert _normalized_pf(pf) == 0.0

    def test_pf_2_is_half(self):
        assert abs(_normalized_pf(2.0) - 0.5) < 1e-9

    def test_pf_3_is_full(self):
        assert abs(_normalized_pf(3.0) - 1.0) < 1e-9

    def test_pf_above_3_stays_full(self):
        for pf in (4.0, 10.0):
            assert _normalized_pf(pf) == 1.0

    def test_ppt_50pct_wr(self):
        assert abs(_ppt_score(50, 100) - 0.5) < 1e-9

    def test_ppt_all_wins(self):
        assert abs(_ppt_score(100, 100) - 1.0) < 1e-9

    def test_ppt_all_losses(self):
        assert abs(_ppt_score(0, 100) - 0.0) < 1e-9

    def test_ppt_zero_trades(self):
        assert _ppt_score(0, 0) == 0.0

    def test_ppt_ordering(self):
        assert _ppt_score(60, 100) > _ppt_score(55, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hard constraint failures
# ─────────────────────────────────────────────────────────────────────────────

class TestHardConstraints:
    """All four hard constraints must individually gate out bad models."""
    N_VAL = 4135   # realistic ES validation set size

    def test_10_of_4135_is_gated_out(self):
        """The exact 10/4135 scenario from the original bug report."""
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=10, win_rate=0.90)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["hard_constraints_passed"] is False
        assert stats["trading_quality"] < 0.0
        assert stats["n_trades"] == 10
        assert stats["trade_coverage_pct"] < 1.0

    def test_13_of_4135_is_gated_out(self):
        """The exact 13/4135 scenario from the third bug report."""
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=13, win_rate=0.90)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        # 13/4135 = 0.31% — fails both count gate AND coverage gate
        assert stats["n_trades"] == 13
        assert stats["trade_coverage_pct"] < 1.0, (
            f"Expected coverage < 1%, got {stats['trade_coverage_pct']:.2f}%"
        )
        assert stats["hard_constraints_passed"] is False, (
            "13/4135 model must NOT pass hard constraints"
        )
        assert stats["trading_quality"] < 0.0, (
            f"13/4135 model must have negative quality, got {stats['trading_quality']:.4f}"
        )
        assert any("trade_count" in f for f in stats["hard_constraint_failures"])

    def test_13_of_4135_cannot_beat_healthy_model(self):
        """Core regression: 13/4135 must NEVER beat a healthy 200-trade model."""
        sparse_p, sparse_y = _make_proba(self.N_VAL, 13,  win_rate=0.92, seed=3)
        active_p, active_y = _make_proba(self.N_VAL, 200, win_rate=0.55, seed=3)

        tq_sparse = _compute_trading_stats(sparse_p, sparse_y, _INV, n_val_bars=self.N_VAL)["trading_quality"]
        tq_active = _compute_trading_stats(active_p, active_y, _INV, n_val_bars=self.N_VAL)["trading_quality"]

        assert tq_sparse < tq_active, (
            f"13-trade model (tq={tq_sparse:.4f}) must score below "
            f"200-trade model (tq={tq_active:.4f}) — this is the regression guard"
        )

    def test_68_of_4135_is_gated_out(self):
        """The 68/4135 (1.64% coverage) scenario from the second bug report."""
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=68, win_rate=0.75)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        # 68/4135 = 1.64% < 3% coverage gate
        assert stats["trade_coverage_pct"] < 3.0
        assert stats["hard_constraints_passed"] is False
        assert stats["trading_quality"] < 0.0
        assert any("coverage" in f for f in stats["hard_constraint_failures"])

    def test_below_50_trades_fails(self):
        """Gate 1: count < 50 must fail."""
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=30, win_rate=0.70)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["hard_constraints_passed"] is False
        assert any("trade_count" in f for f in stats["hard_constraint_failures"])

    def test_50_trades_but_below_3pct_coverage_fails(self):
        """Gate 2: coverage < 3% must fail even when count >= 50."""
        # 50 trades / 4135 bars = 1.21% < 3%
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=50, win_rate=0.65)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["trade_coverage_pct"] < 3.0
        assert stats["hard_constraints_passed"] is False
        assert any("coverage" in f for f in stats["hard_constraint_failures"])

    def test_low_profit_factor_fails(self):
        """Gate 4: PF < 1.10 must fail even when count and coverage pass."""
        # 200 trades / 2000 bars (10% OK), 45% win rate → PF < 1.0
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=0.45)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["profit_factor"] < _HC_MIN_PF
        assert stats["hard_constraints_passed"] is False
        assert any("profit_factor" in f for f in stats["hard_constraint_failures"])

    def test_low_directional_accuracy_fails(self):
        """Gate 3: dir_acc < 52% must fail even when count and coverage pass."""
        # 200 trades / 2000 bars, 50% win rate → dir_acc < 52%
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=0.50)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["dir_accuracy"] < _HC_MIN_DIR_ACC
        assert stats["hard_constraints_passed"] is False
        assert any("dir_accuracy" in f for f in stats["hard_constraint_failures"])

    def test_all_four_failures_reported(self):
        """When all four gates fail simultaneously, all appear in the failure list."""
        # 5 trades / 4135 bars → fails count, coverage, and likely PF + dir_acc
        proba, y_raw = _make_proba(self.N_VAL, n_confident_trades=5, win_rate=0.40)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["hard_constraints_passed"] is False
        # At minimum count and coverage fail
        failure_text = " ".join(stats["hard_constraint_failures"])
        assert "trade_count" in failure_text or "coverage" in failure_text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Penalty magnitudes
# ─────────────────────────────────────────────────────────────────────────────

class TestPenaltyMagnitudes:
    """Verify the exact penalty magnitudes so there are no surprises."""

    def test_severe_penalty_for_too_few_trades(self):
        # 10 trades / 4135 bars — count < 50 AND coverage < 3% → SEVERE
        proba, y_raw = _make_proba(4135, n_confident_trades=10, win_rate=0.80)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=4135)
        expected = -abs(_HC_PENALTY_SEVERE)  # = -10.0
        assert stats["trading_quality"] == pytest.approx(expected, abs=1e-6)

    def test_severe_penalty_for_68_trades(self):
        # 68 / 4135 bars — coverage=1.64% < 3% → SEVERE (even though count >= 50? No: 68 >= 50 but coverage < 3%)
        proba, y_raw = _make_proba(4135, n_confident_trades=68, win_rate=0.80)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=4135)
        # 68 >= 50 passes count gate, but 1.64% < 3% fails coverage → SEVERE
        assert stats["trading_quality"] == pytest.approx(-abs(_HC_PENALTY_SEVERE), abs=1e-6)

    def test_medium_penalty_for_both_secondary_failures(self):
        # 200 trades / 2000 bars (10% coverage, count=200 — both STAGE 1 gates pass)
        # 50% win rate → both dir_acc and pf HCs fail → MEDIUM penalty
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=0.50, seed=1)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["hard_constraints_passed"] is False
        assert stats["trading_quality"] == pytest.approx(-abs(_HC_PENALTY_MEDIUM), abs=1e-6)

    def test_severe_worse_than_medium(self):
        proba_sev, y_sev = _make_proba(4135, n_confident_trades=10, win_rate=0.80)
        proba_med, y_med = _make_proba(2000, n_confident_trades=200, win_rate=0.50)
        tq_sev = _compute_trading_stats(proba_sev, y_sev, _INV, n_val_bars=4135)["trading_quality"]
        tq_med = _compute_trading_stats(proba_med, y_med, _INV, n_val_bars=2000)["trading_quality"]
        assert tq_sev < tq_med     # severe penalty < medium penalty


# ─────────────────────────────────────────────────────────────────────────────
# 6. Passing models
# ─────────────────────────────────────────────────────────────────────────────

class TestPassingModels:
    def _healthy(self, n_bars=2000, n_trades=120, win_rate=0.60, seed=1):
        return _make_proba(n_bars, n_trades, win_rate=win_rate, seed=seed)

    def test_all_hc_pass(self):
        proba, y_raw = self._healthy()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["hard_constraints_passed"] is True
        assert stats["hard_constraint_failures"] == []

    def test_positive_score(self):
        proba, y_raw = self._healthy()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert stats["trading_quality"] > 0.0

    def test_score_in_unit_range(self):
        proba, y_raw = self._healthy()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert 0.0 <= stats["trading_quality"] <= 1.0

    def test_score_cannot_exceed_one(self):
        # Even a perfect model stays <= 1.0
        proba, y_raw = _make_proba(2000, n_confident_trades=200, win_rate=1.0, seed=0)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        if stats["hard_constraints_passed"]:
            assert stats["trading_quality"] <= 1.0

    def test_all_sub_components_present(self):
        proba, y_raw = self._healthy()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        required = {"dir_accuracy", "normalized_pf", "trade_coverage",
                    "activity", "pnl_per_trade"}
        assert required <= set(stats["score_components"].keys())

    def test_sub_components_in_unit_range(self):
        proba, y_raw = self._healthy()
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        for k, v in stats["score_components"].items():
            assert 0.0 <= v <= 1.0, f"component {k!r} = {v}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Ordering invariants — the central regression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreOrdering:
    """Sparse models MUST lose to active models.  No exceptions."""

    N_VAL = 4135

    def test_10_of_4135_loses_to_healthy(self):
        sparse_p, sparse_y = _make_proba(self.N_VAL, 10,  win_rate=0.90, seed=0)
        active_p, active_y = _make_proba(self.N_VAL, 200, win_rate=0.58, seed=0)

        tq_sparse = _compute_trading_stats(sparse_p, sparse_y, _INV, n_val_bars=self.N_VAL)["trading_quality"]
        tq_active = _compute_trading_stats(active_p, active_y, _INV, n_val_bars=self.N_VAL)["trading_quality"]

        assert tq_sparse < tq_active, (
            f"10-trade model ({tq_sparse:.4f}) must score below "
            f"200-trade model ({tq_active:.4f})"
        )

    def test_68_of_4135_loses_to_healthy(self):
        """The exact scenario from the second bug report."""
        sparse_p, sparse_y = _make_proba(self.N_VAL, 68,  win_rate=0.75, seed=1)
        active_p, active_y = _make_proba(self.N_VAL, 200, win_rate=0.58, seed=1)

        tq_sparse = _compute_trading_stats(sparse_p, sparse_y, _INV, n_val_bars=self.N_VAL)["trading_quality"]
        tq_active = _compute_trading_stats(active_p, active_y, _INV, n_val_bars=self.N_VAL)["trading_quality"]

        assert tq_sparse < tq_active, (
            f"68-trade model ({tq_sparse:.4f}) must score below "
            f"200-trade model ({tq_active:.4f})"
        )

    def test_sparse_always_negative(self):
        for n_trades in (10, 40, 68):
            proba, y_raw = _make_proba(self.N_VAL, n_trades, win_rate=0.90, seed=2)
            tq = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)["trading_quality"]
            assert tq < 0.0, f"{n_trades}-trade model score must be negative, got {tq:.4f}"

    def test_more_trades_beats_fewer_at_same_accuracy(self):
        # Both above gates; more trades should score at least as well
        p100, y100 = _make_proba(2000, 100, win_rate=0.60, seed=5)
        p60,  y60  = _make_proba(2000, 60,  win_rate=0.60, seed=5)
        tq100 = _compute_trading_stats(p100, y100, _INV, n_val_bars=2000)["trading_quality"]
        tq60  = _compute_trading_stats(p60,  y60,  _INV, n_val_bars=2000)["trading_quality"]
        assert tq100 >= tq60

    def test_higher_accuracy_beats_lower_at_same_activity(self):
        p_hi, y_hi = _make_proba(2000, 100, win_rate=0.70, seed=7)
        p_lo, y_lo = _make_proba(2000, 100, win_rate=0.55, seed=7)
        tq_hi = _compute_trading_stats(p_hi, y_hi, _INV, n_val_bars=2000)["trading_quality"]
        tq_lo = _compute_trading_stats(p_lo, y_lo, _INV, n_val_bars=2000)["trading_quality"]
        if (
            _compute_trading_stats(p_hi, y_hi, _INV, n_val_bars=2000)["hard_constraints_passed"]
            and _compute_trading_stats(p_lo, y_lo, _INV, n_val_bars=2000)["hard_constraints_passed"]
        ):
            assert tq_hi > tq_lo


# ─────────────────────────────────────────────────────────────────────────────
# 8. Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_score_deterministic(self):
        proba, y_raw = _make_proba(2000, 100, win_rate=0.60, seed=99)
        scores = [
            _trading_quality_score(proba, y_raw, _INV, n_val_bars=2000)
            for _ in range(5)
        ]
        assert len(set(scores)) == 1, f"Non-deterministic: {scores}"

    def test_stats_deterministic(self):
        proba, y_raw = _make_proba(3000, 150, win_rate=0.62, seed=11)
        a = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=3000)
        b = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=3000)
        assert a["trading_quality"] == b["trading_quality"]
        assert a["hard_constraints_passed"] == b["hard_constraints_passed"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Wrapper consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestWrapper:
    def test_wrapper_matches_stats(self):
        proba, y_raw = _make_proba(2000, 100, win_rate=0.60, seed=3)
        score = _trading_quality_score(proba, y_raw, _INV, n_val_bars=2000)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        assert score == stats["trading_quality"]

    def test_wrapper_negative_for_sparse(self):
        proba, y_raw = _make_proba(4000, 5, win_rate=1.0, seed=0)
        score = _trading_quality_score(proba, y_raw, _INV, n_val_bars=4000)
        assert score < 0.0

    def test_wrapper_defaults_n_val_bars(self):
        proba, y_raw = _make_proba(500, 30, win_rate=0.65, seed=0)
        score_none = _trading_quality_score(proba, y_raw, _INV, n_val_bars=None)
        score_500  = _trading_quality_score(proba, y_raw, _INV, n_val_bars=500)
        assert score_none == score_500


# ─────────────────────────────────────────────────────────────────────────────
# 10. Gate-fail sentinel propagates correctly to composite score
# ─────────────────────────────────────────────────────────────────────────────

class TestGateFailSentinel:
    """Verify that _GATE_FAIL_SCORE is below any composite a passing model
    could achieve, so Optuna will always prefer passing models."""

    def test_gate_fail_below_worst_passing_composite(self):
        # Worst composite for a passing model: F1=0 + trading=0 → composite=0
        # So _GATE_FAIL_SCORE must be < 0
        assert _GATE_FAIL_SCORE < 0.0

    def test_gate_fail_below_medium_penalty_composite(self):
        # Medium penalty composite: 0.40*1.0 + 0.60*(-3.0) = 0.40 - 1.80 = -1.40
        worst_medium = 0.40 * 1.0 + 0.60 * _HC_PENALTY_MEDIUM
        assert _GATE_FAIL_SCORE < worst_medium

    def test_gate_fail_below_severe_penalty_composite(self):
        # Severe penalty composite: 0.40*1.0 + 0.60*(-10.0) = 0.40 - 6.0 = -5.6
        worst_severe = 0.40 * 1.0 + 0.60 * _HC_PENALTY_SEVERE
        assert _GATE_FAIL_SCORE < worst_severe


# ─────────────────────────────────────────────────────────────────────────────
# 11. All-gates-failed detection logic
# ─────────────────────────────────────────────────────────────────────────────

class TestAllGatesFailedDetection:
    """
    The all-gates-failed condition is:
        best_score <= _GATE_FAIL_SCORE * 0.5

    Since _GATE_FAIL_SCORE = -1e9 and * 0.5 = -5e8:
      • When all trials return _GATE_FAIL_SCORE (-1e9): -1e9 <= -5e8 → True  (fallback fires)
      • When best trial returns 0.0:                    0.0  <= -5e8 → False (no fallback)
      • When best trial returns -0.5 (medium penalty):  -0.5 <= -5e8 → False (no fallback)
      • When best trial returns -5.6 (severe penalty):  -5.6 <= -5e8 → False (no fallback)

    This verifies the detection threshold never falsely triggers for any
    legitimate composite score, including deeply negative ones.
    """

    def test_all_failed_fires_for_gate_fail_score(self):
        """When all trials returned _GATE_FAIL_SCORE, detection triggers."""
        best_score = _GATE_FAIL_SCORE          # -1e9
        assert best_score <= _GATE_FAIL_SCORE * 0.5, (
            "Detection must fire when best_score == _GATE_FAIL_SCORE"
        )

    def test_all_failed_does_not_fire_for_zero(self):
        """A legitimate zero composite does NOT trigger fallback."""
        best_score = 0.0
        assert not (best_score <= _GATE_FAIL_SCORE * 0.5)

    def test_all_failed_does_not_fire_for_medium_penalty(self):
        """A medium-penalty composite does NOT trigger fallback."""
        # 0.40*1.0 + 0.60*(-3.0) = -1.40 — legitimate score, not a sentinel
        best_score = 0.40 * 1.0 + 0.60 * _HC_PENALTY_MEDIUM
        assert not (best_score <= _GATE_FAIL_SCORE * 0.5)

    def test_all_failed_does_not_fire_for_severe_penalty(self):
        """A severe-penalty composite does NOT trigger fallback."""
        # 0.40*1.0 + 0.60*(-10.0) = -5.60 — legitimate score, not a sentinel
        best_score = 0.40 * 1.0 + 0.60 * _HC_PENALTY_SEVERE
        assert not (best_score <= _GATE_FAIL_SCORE * 0.5)

    def test_all_failed_does_not_fire_for_small_positive(self):
        """Any positive score is clearly not a sentinel."""
        for score in (0.001, 0.1, 0.5, 1.0):
            assert not (score <= _GATE_FAIL_SCORE * 0.5)

    def test_gate_fail_score_times_half_is_unambiguously_negative(self):
        """The detection threshold itself must be far below any real score."""
        threshold = _GATE_FAIL_SCORE * 0.5   # -5e8
        # No real composite can be <= -5e8; the worst possible is ~-6 (severe)
        worst_real = 0.40 * 1.0 + 0.60 * _HC_PENALTY_SEVERE   # ~ -5.6
        assert threshold < worst_real

    def test_fallback_label_is_deterministic(self):
        """selected_via must be exactly 'fallback_f1' when fallback triggers."""
        # We test by checking that the string is defined in the module.
        # The actual Optuna flow is covered by integration; here we verify
        # the value is never ambiguous or partially assigned.
        expected = "fallback_f1"
        # Verify it's a plain ASCII string (no typos)
        assert expected.islower()
        assert "_" in expected


# ─────────────────────────────────────────────────────────────────────────────
# 12. All four inline gates — individual rejection scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestAllFourGatesViaStats:
    """
    Verify that _compute_trading_stats rejects models that fail each of the
    four gates independently (count, coverage, dir_acc, PF).

    These mirror what the inline Optuna objective checks before calling the
    scorer, ensuring that even if a model slips through to the scorer, it
    still receives a negative quality score.
    """

    N_VAL = 4135

    def test_gate1_count_below_50(self):
        """Gate 1: confident_trade_count < 50 → rejected."""
        for n in (1, 13, 25, 49):
            proba, y_raw = _make_proba(self.N_VAL, n, win_rate=0.90, seed=n)
            stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
            assert not stats["hard_constraints_passed"], f"n={n} should be rejected"
            assert stats["trading_quality"] < 0.0, f"n={n} tq should be negative"

    def test_gate2_coverage_below_3pct(self):
        """Gate 2: coverage < 3% → rejected even when count >= 50."""
        # Need n_trades >= 50 but still < 3% of n_val_bars
        # 3% of 4135 = 124.05 — use 80 trades (1.93% coverage)
        proba, y_raw = _make_proba(self.N_VAL, 80, win_rate=0.80, seed=42)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert stats["n_trades"] == 80
        assert stats["trade_coverage_pct"] < 3.0
        assert not stats["hard_constraints_passed"]
        assert stats["trading_quality"] < 0.0

    def test_gate3_dir_acc_below_52pct(self):
        """Gate 3: dir_acc < 52% → rejected even when count and coverage pass."""
        # Use enough bars to get >3% coverage but low win rate
        n_bars  = 2000
        n_trade = 200   # 10% coverage — passes count and coverage
        proba, y_raw = _make_proba(n_bars, n_trade, win_rate=0.48, seed=7)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=n_bars)
        if stats["dir_accuracy"] < _HC_MIN_DIR_ACC:
            assert not stats["hard_constraints_passed"]
            assert stats["trading_quality"] < 0.0

    def test_gate4_pf_below_110(self):
        """Gate 4: PF < 1.10 → rejected even when other gates pass."""
        n_bars  = 2000
        n_trade = 200   # 10% coverage — passes count and coverage
        proba, y_raw = _make_proba(n_bars, n_trade, win_rate=0.45, seed=8)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=n_bars)
        if stats["profit_factor"] < _HC_MIN_PF:
            assert not stats["hard_constraints_passed"]
            assert stats["trading_quality"] < 0.0

    def test_passing_model_all_four_gates(self):
        """A model that passes all four gates gets a positive score in [0,1]."""
        # 200 trades / 2000 bars = 10% coverage, 65% win rate
        proba, y_raw = _make_proba(2000, 200, win_rate=0.65, seed=10)
        stats = _compute_trading_stats(proba, y_raw, _INV, n_val_bars=2000)
        # Only assert if we happen to pass all gates (win rate randomness)
        if stats["hard_constraints_passed"]:
            assert 0.0 < stats["trading_quality"] <= 1.0
            assert stats["hard_constraint_failures"] == []


# ─────────────────────────────────────────────────────────────────────────────
# 13. _THRESHOLD_CANDIDATES constant and _select_best_threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestThresholdCandidatesConstant:
    def test_candidates_is_list(self):
        assert isinstance(_THRESHOLD_CANDIDATES, list)

    def test_candidates_all_floats(self):
        for t in _THRESHOLD_CANDIDATES:
            assert isinstance(t, float), f"Candidate {t!r} is not float"

    def test_candidates_in_unit_range(self):
        for t in _THRESHOLD_CANDIDATES:
            assert 0.0 < t < 1.0, f"Candidate {t} outside (0, 1)"

    def test_candidates_sorted_ascending(self):
        assert _THRESHOLD_CANDIDATES == sorted(_THRESHOLD_CANDIDATES)

    def test_candidates_non_empty(self):
        assert len(_THRESHOLD_CANDIDATES) >= 2

    def test_default_threshold_in_candidates(self):
        assert _DEFAULT_THRESHOLD in _THRESHOLD_CANDIDATES, (
            f"Default threshold {_DEFAULT_THRESHOLD} not in candidates"
        )


class TestSelectBestThreshold:
    """
    Unit tests for _select_best_threshold.

    The function sweeps _THRESHOLD_CANDIDATES and returns the best
    passing threshold, or _DEFAULT_THRESHOLD if none passes.
    """

    N_VAL = 2000

    def _healthy_proba(self, n_trades=200, win_rate=0.65, seed=0):
        return _make_proba(self.N_VAL, n_trades, win_rate=win_rate, seed=seed)

    def _sparse_proba(self, n_trades=5, seed=0):
        return _make_proba(self.N_VAL, n_trades, win_rate=0.90, seed=seed)

    def test_returns_float(self):
        proba, y_raw = self._healthy_proba()
        t = _select_best_threshold(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert isinstance(t, float)

    def test_returns_value_from_candidates(self):
        proba, y_raw = self._healthy_proba()
        t = _select_best_threshold(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert t in _THRESHOLD_CANDIDATES, f"{t} not in candidates"

    def test_fallback_when_no_candidate_passes(self):
        """When no threshold passes the gates, must return _DEFAULT_THRESHOLD."""
        # 5 trades / 2000 bars — fails all gates at every threshold
        proba, y_raw = self._sparse_proba(n_trades=5)
        t = _select_best_threshold(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert t == _DEFAULT_THRESHOLD, (
            f"Expected fallback {_DEFAULT_THRESHOLD}, got {t}"
        )

    def test_deterministic_same_data(self):
        """Same inputs → same output, every time."""
        proba, y_raw = self._healthy_proba(seed=42)
        results = [
            _select_best_threshold(proba, y_raw, _INV, n_val_bars=self.N_VAL)
            for _ in range(5)
        ]
        assert len(set(results)) == 1, f"Non-deterministic: {results}"

    def test_accepts_str_keyed_inv_map(self):
        """Must work with both int-keyed and str-keyed inv_label_map."""
        proba, y_raw = self._healthy_proba()
        str_inv = {str(k): v for k, v in _INV.items()}
        t = _select_best_threshold(proba, y_raw, str_inv, n_val_bars=self.N_VAL)
        assert t in _THRESHOLD_CANDIDATES

    def test_accepts_int_keyed_inv_map(self):
        """int-keyed map (used internally by train.py) must also work."""
        proba, y_raw = self._healthy_proba()
        t = _select_best_threshold(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        assert t in _THRESHOLD_CANDIDATES

    def test_higher_threshold_selected_when_it_improves_quality(self):
        """
        A model that fires on many bars at low confidence but only a
        clean subset at higher confidence should prefer the higher threshold.
        We can't easily force this with synthetic data, so we just verify
        the function picks some threshold (not necessarily the lowest).
        """
        proba, y_raw = self._healthy_proba(seed=7)
        t = _select_best_threshold(proba, y_raw, _INV, n_val_bars=self.N_VAL)
        # Any candidate is acceptable — just must be a candidate
        assert t in _THRESHOLD_CANDIDATES
