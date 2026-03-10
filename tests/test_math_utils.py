"""
Unit tests for src/utils/math_utils.py
"""

import numpy as np
import pytest

from src.utils.math_utils import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    profit_factor,
    kelly_fraction,
    half_kelly,
    atr_position_size,
    fixed_fractional_size,
)


class TestSharpeRatio:
    def test_zero_returns_gives_zero(self):
        returns = np.zeros(100)
        assert sharpe_ratio(returns) == 0.0

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.005, 252)
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_single_return_gives_zero(self):
        assert sharpe_ratio(np.array([0.01])) == 0.0

    def test_negative_returns_negative_sharpe(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(-0.001, 0.005, 252)
        sr = sharpe_ratio(returns)
        assert sr < 0


class TestMaxDrawdown:
    def test_always_rising_gives_zero_drawdown(self):
        equity = np.cumsum(np.ones(100))
        assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-6)

    def test_always_falling_gives_large_drawdown(self):
        equity = np.linspace(100, 0, 100)
        mdd = max_drawdown(equity)
        assert mdd < -0.5  # significant drawdown

    def test_drawdown_returns_non_positive(self):
        rng = np.random.default_rng(42)
        equity = np.cumsum(rng.normal(0, 1, 200))
        assert max_drawdown(equity) <= 0

    def test_short_series(self):
        assert max_drawdown(np.array([100.0])) == 0.0


class TestProfitFactor:
    def test_all_wins(self):
        pnls = np.array([100.0, 200.0, 50.0])
        assert profit_factor(pnls) == float("inf")

    def test_equal_wins_losses(self):
        pnls = np.array([100.0, -100.0])
        assert profit_factor(pnls) == pytest.approx(1.0)

    def test_ratio_correct(self):
        pnls = np.array([300.0, -100.0])
        assert profit_factor(pnls) == pytest.approx(3.0)


class TestKelly:
    def test_kelly_zero_when_no_edge(self):
        # 50% win rate, avg win = avg loss → kelly = 0
        f = kelly_fraction(win_rate=0.5, avg_win=1.0, avg_loss=1.0)
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_kelly_positive_with_edge(self):
        f = kelly_fraction(win_rate=0.6, avg_win=1.5, avg_loss=1.0)
        assert f > 0

    def test_kelly_clamped_to_one(self):
        f = kelly_fraction(win_rate=0.99, avg_win=10.0, avg_loss=0.01)
        assert f <= 1.0

    def test_half_kelly_half_of_kelly(self):
        k  = kelly_fraction(0.6, 1.5, 1.0)
        hk = half_kelly(0.6, 1.5, 1.0)
        assert hk == pytest.approx(k / 2)

    def test_kelly_invalid_inputs(self):
        assert kelly_fraction(0.5, 0.0, 1.0) == 0.0
        assert kelly_fraction(0.5, 1.0, 0.0) == 0.0


class TestPositionSizing:
    def test_atr_size_minimum_one(self):
        size = atr_position_size(
            account_equity=100_000,
            risk_per_trade_frac=0.0001,  # tiny risk
            atr_pts=100.0,               # huge ATR
            multiplier=50.0,
        )
        assert size >= 1

    def test_atr_size_increases_with_equity(self):
        base = atr_position_size(100_000, 0.01, 5.0, 50.0)
        big  = atr_position_size(1_000_000, 0.01, 5.0, 50.0)
        assert big >= base

    def test_fixed_frac_minimum_one(self):
        size = fixed_fractional_size(100_000, 0.0001, 100.0, 50.0)
        assert size >= 1

    def test_fixed_frac_zero_stop_gives_one(self):
        assert fixed_fractional_size(100_000, 0.01, 0.0, 50.0) == 1
