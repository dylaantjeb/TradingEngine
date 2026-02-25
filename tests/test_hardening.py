"""
Backtest hardening tests.

Run with:  python -m pytest tests/test_hardening.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _simulate_trades, _compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_prices(n: int = 20, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Return (open_series, close_series) with a stable DatetimeIndex."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    base = 5000.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_  = pd.Series(base + rng.uniform(-0.1, 0.1, n), index=idx)
    close_ = pd.Series(base + rng.uniform(-0.1, 0.1, n), index=idx)
    return open_, close_


def _flat_atr(n: int, value: float = 20.0) -> pd.Series:
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    return pd.Series(value, index=idx)


def _run_sim(signals, open_prices, atr=None, cfg_overrides=None):
    """Convenience wrapper around _simulate_trades with realistic defaults."""
    cfg = {
        "execution_delay_bars": 1,
        "session_start_utc_hour": 0,
        "session_end_utc_hour": 24,
        "atr_min_ticks": 0,
        "atr_max_ticks": 1e9,
        "news_blackout_windows": [],
        "max_trades_per_day": 9999,
        "min_holding_bars": 1,
        "cooldown_bars_after_exit": 0,
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)

    n = len(signals)
    if atr is None:
        atr = _flat_atr(n, value=20.0).reindex(signals.index)

    return _simulate_trades(
        signals       = signals,
        open_prices   = open_prices,
        atr_ticks     = atr,
        cfg           = cfg,
        friction_pts  = 0.0,    # no friction unless test overrides
        commission_rt = 0.0,
        multiplier    = 50.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: No same-bar fill
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSameBarFill:
    """Ensure the engine NEVER fills on the same bar as the signal."""

    def test_entry_fills_next_bar(self):
        """Entry at bar i must fill at bar i+1, not bar i."""
        n = 10
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        # Signal = +1 at bar 2, flat everywhere else
        sig_arr = np.zeros(n, dtype=np.int8)
        sig_arr[2] = 1
        signals     = pd.Series(sig_arr, index=idx)
        open_prices = pd.Series(np.arange(5000, 5000 + n, dtype=float), index=idx)

        trades, _, _ = _run_sim(signals, open_prices)

        assert len(trades) >= 1, "Expected at least one trade (the entry + auto-exit)"
        # The trade was entered: entry_time should be bar 3 (i+1 = 3), not bar 2
        entry_ts = pd.Timestamp(trades[0]["entry_time"])
        signal_ts = idx[2]
        assert entry_ts > signal_ts, (
            f"Fill at {entry_ts} should be AFTER signal at {signal_ts} "
            f"(1-bar delay violated)"
        )
        # Specifically, entry should be at bar 3 (index[3])
        assert entry_ts == idx[3], (
            f"Expected fill at bar 3 ({idx[3]}), got {entry_ts}"
        )

    def test_exit_fills_next_bar(self):
        """Exit signal at bar i must fill at bar i+1."""
        n = 15
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        # Enter at bar 1, exit at bar 5
        sig_arr = np.zeros(n, dtype=np.int8)
        sig_arr[1] = 1   # entry signal
        sig_arr[2] = 1
        sig_arr[3] = 1
        sig_arr[4] = 1
        sig_arr[5] = 0   # exit signal (drop to flat)
        signals     = pd.Series(sig_arr, index=idx)
        open_prices = pd.Series(np.arange(5000, 5000 + n, dtype=float), index=idx)

        trades, _, _ = _run_sim(
            signals, open_prices,
            cfg_overrides={"min_holding_bars": 1},
        )

        completed = [t for t in trades if "exit_time" in t]
        if completed:
            exit_ts   = pd.Timestamp(completed[0]["exit_time"])
            signal_ts = idx[5]   # signal emitted at bar 5
            assert exit_ts > signal_ts, (
                f"Exit fill at {exit_ts} should be AFTER signal at {signal_ts}"
            )

    def test_no_same_bar_fill_bulk(self):
        """Randomly generate signals; for every trade verify exit > entry (different bars)."""
        rng = np.random.default_rng(42)
        n   = 100
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        raw_sig = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.2, 0.5]).astype(np.int8)
        signals     = pd.Series(raw_sig, index=idx)
        open_prices = pd.Series(5000.0 + np.cumsum(rng.normal(0, 0.25, n)), index=idx)

        trades, _, _ = _run_sim(signals, open_prices)

        for t in trades:
            assert t["entry_time"] != t["exit_time"], (
                f"Entry and exit on the same bar: {t}"
            )
            assert pd.Timestamp(t["exit_time"]) > pd.Timestamp(t["entry_time"]), (
                f"Exit before entry: {t}"
            )

    def test_delay_zero_is_same_bar(self):
        """With delay=0 the fill IS same-bar (leaky mode) – verify it works but is leaky."""
        n = 10
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        sig_arr = np.zeros(n, dtype=np.int8)
        sig_arr[3] = 1
        signals     = pd.Series(sig_arr, index=idx)
        open_prices = pd.Series(np.arange(5000.0, 5000.0 + n), index=idx)

        trades, _, _ = _run_sim(
            signals, open_prices,
            cfg_overrides={"execution_delay_bars": 0},
        )
        # With delay=0, entry should be at the same bar as signal (bar 3)
        if trades:
            entry_ts = pd.Timestamp(trades[0]["entry_time"])
            assert entry_ts == idx[3], (
                f"delay=0: expected fill at signal bar {idx[3]}, got {entry_ts}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Costs reduce PnL
# ─────────────────────────────────────────────────────────────────────────────

class TestCostsReducePnL:
    """Costs (commission + friction) must always reduce net PnL vs zero-cost."""

    def _run_with_costs(self, friction=0.0, commission=0.0):
        n   = 30
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        # Strong uptrend: long signal for most bars
        sig_arr = np.ones(n, dtype=np.int8)
        sig_arr[-3:] = 0           # flatten at end
        signals     = pd.Series(sig_arr, index=idx)
        prices_arr  = np.linspace(5000.0, 5050.0, n)
        open_prices = pd.Series(prices_arr, index=idx)

        cfg = {
            "execution_delay_bars": 1,
            "session_start_utc_hour": 0,
            "session_end_utc_hour": 24,
            "atr_min_ticks": 0,
            "atr_max_ticks": 1e9,
            "news_blackout_windows": [],
            "max_trades_per_day": 9999,
            "min_holding_bars": 1,
            "cooldown_bars_after_exit": 0,
        }
        atr = _flat_atr(n, 20.0).reindex(idx)
        trades, equity, cost_summary = _simulate_trades(
            signals       = signals,
            open_prices   = open_prices,
            atr_ticks     = atr,
            cfg           = cfg,
            friction_pts  = friction,
            commission_rt = commission,
            multiplier    = 50.0,
        )
        return equity.iloc[-1], cost_summary

    def test_zero_cost_baseline(self):
        """Zero-cost run should produce positive PnL on an uptrend."""
        final_equity, summary = self._run_with_costs(friction=0.0, commission=0.0)
        assert final_equity >= 0, f"Expected ≥0 PnL on uptrend with no costs, got {final_equity}"

    def test_commission_reduces_pnl(self):
        """Adding commission must reduce net PnL."""
        eq_nocost, _ = self._run_with_costs(friction=0.0, commission=0.0)
        eq_withcost, _ = self._run_with_costs(friction=0.0, commission=3.0)
        assert eq_withcost <= eq_nocost, (
            f"Commission should reduce PnL: no_cost={eq_nocost:.2f} "
            f"with_cost={eq_withcost:.2f}"
        )

    def test_friction_reduces_pnl(self):
        """Adding slippage/spread friction must reduce net PnL."""
        eq_nocost, _ = self._run_with_costs(friction=0.0, commission=0.0)
        eq_fric, _   = self._run_with_costs(friction=0.25, commission=0.0)
        assert eq_fric <= eq_nocost, (
            f"Friction should reduce PnL: no_cost={eq_nocost:.2f} "
            f"with_friction={eq_fric:.2f}"
        )

    def test_cost_summary_accounting(self):
        """cost_summary gross - total_costs == net_pnl."""
        _, summary = self._run_with_costs(friction=0.125, commission=3.0)
        gross = summary["gross_pnl"]
        costs = summary["total_costs"]
        net   = summary["net_pnl"]
        assert abs((gross - costs) - net) < 1e-6, (
            f"Accounting mismatch: {gross} - {costs} != {net}"
        )

    def test_costs_logged_in_trades(self):
        """Each trade dict should have 'cost' and 'gross_pnl' fields."""
        n   = 20
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        sig = np.ones(n, dtype=np.int8)
        sig[-2:] = 0
        signals     = pd.Series(sig, index=idx)
        open_prices = pd.Series(np.linspace(5000, 5020, n), index=idx)

        cfg = {
            "execution_delay_bars": 1,
            "session_start_utc_hour": 0,
            "session_end_utc_hour": 24,
            "atr_min_ticks": 0,
            "atr_max_ticks": 1e9,
            "news_blackout_windows": [],
            "max_trades_per_day": 9999,
            "min_holding_bars": 1,
            "cooldown_bars_after_exit": 0,
        }
        atr = _flat_atr(n, 20.0).reindex(idx)
        trades, _, _ = _simulate_trades(
            signals=signals, open_prices=open_prices, atr_ticks=atr,
            cfg=cfg, friction_pts=0.125, commission_rt=3.0, multiplier=50.0,
        )
        for t in trades:
            assert "cost" in t,      f"Trade missing 'cost' field: {t}"
            assert "gross_pnl" in t, f"Trade missing 'gross_pnl' field: {t}"
            assert "net_pnl" in t,   f"Trade missing 'net_pnl' field: {t}"
            assert t["cost"] >= 0,   f"Cost must be non-negative: {t}"
            assert abs(t["gross_pnl"] - t["cost"] - t["net_pnl"]) < 1e-4, (
                f"Per-trade accounting: gross - cost != net: {t}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Throttles
# ─────────────────────────────────────────────────────────────────────────────

class TestThrottles:
    def test_max_trades_per_day(self):
        """max_trades_per_day=2 should produce at most 2 round trips per day."""
        n   = 60
        # All bars on the same calendar day
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        # Alternate +1 / -1 every 2 bars to generate many signals
        sig_arr = np.tile([1, 1, -1, -1], n // 4 + 1)[:n].astype(np.int8)
        signals     = pd.Series(sig_arr, index=idx)
        open_prices = pd.Series(5000.0 + np.arange(n, dtype=float) * 0.1, index=idx)

        trades, _, _ = _run_sim(
            signals, open_prices,
            cfg_overrides={
                "max_trades_per_day": 2,
                "min_holding_bars": 1,
                "cooldown_bars_after_exit": 0,
            },
        )
        assert len(trades) <= 2, (
            f"max_trades_per_day=2 violated: {len(trades)} trades found"
        )

    def test_cooldown_prevents_reentry(self):
        """After an exit, cooldown_bars_after_exit bars must pass before re-entry."""
        n   = 30
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        # Signal: enter at bar 1, exit at bar 5, then immediately try to re-enter
        sig_arr = np.ones(n, dtype=np.int8)
        sig_arr[5] = 0          # exit
        sig_arr[6] = 1          # immediate re-entry attempt (should be blocked)
        sig_arr[7] = 1          # still in cooldown (cooldown=2)
        signals     = pd.Series(sig_arr, index=idx)
        open_prices = pd.Series(5000.0 + np.arange(n, dtype=float) * 0.1, index=idx)

        trades, _, _ = _run_sim(
            signals, open_prices,
            cfg_overrides={
                "min_holding_bars": 1,
                "cooldown_bars_after_exit": 2,
            },
        )
        # Should have exactly 2 trades: one that exits at ~bar 6, one that re-enters after cooldown
        if len(trades) >= 2:
            exit_ts  = pd.Timestamp(trades[0]["exit_time"])
            entry2   = pd.Timestamp(trades[1]["entry_time"])
            gap_bars = (entry2 - exit_ts).total_seconds() / 60
            # Gap must be > cooldown_bars_after_exit (2 bars)
            assert gap_bars >= 2, (
                f"Cooldown violated: re-entry only {gap_bars:.0f} bars after exit"
            )

    def test_min_holding_bars(self):
        """Position must be held for at least min_holding_bars before exiting."""
        n   = 20
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        sig_arr = np.ones(n, dtype=np.int8)
        sig_arr[3] = 0   # premature exit signal (should be ignored)
        sig_arr[4] = 0   # still ignored
        signals     = pd.Series(sig_arr, index=idx)
        open_prices = pd.Series(5000.0 + np.arange(n, dtype=float) * 0.1, index=idx)

        trades, _, _ = _run_sim(
            signals, open_prices,
            cfg_overrides={"min_holding_bars": 5, "cooldown_bars_after_exit": 0},
        )
        for t in trades:
            assert t["hold_bars"] >= 5, (
                f"min_holding_bars=5 violated: trade held only {t['hold_bars']} bars: {t}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_keys_present(self):
        """_compute_metrics must return all required keys."""
        required = {
            "total_pnl_usd", "gross_pnl", "total_costs", "net_pnl",
            "sharpe", "sortino", "rolling_sharpe_20d",
            "max_drawdown_usd", "max_drawdown_pct",
            "win_rate", "expectancy_usd", "profit_factor",
            "avg_trade_duration_bars", "trades_per_day",
        }
        n   = 30
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        sig = np.ones(n, dtype=np.int8); sig[-3:] = 0
        signals     = pd.Series(sig, index=idx)
        open_prices = pd.Series(np.linspace(5000, 5020, n), index=idx)

        trades, equity, cost_summary = _run_sim(signals, open_prices)
        metrics = _compute_metrics(equity, trades, cost_summary)

        missing = required - set(metrics.keys())
        assert not missing, f"Metrics missing keys: {missing}"

    def test_empty_trades_returns_zeros(self):
        """Zero trades should return a valid metrics dict with zero values."""
        idx    = pd.date_range("2024-01-02 14:30", periods=10, freq="1min", tz="UTC")
        equity = pd.Series(0.0, index=idx)
        metrics = _compute_metrics(equity, [], {"gross_pnl": 0, "total_costs": 0, "net_pnl": 0})
        assert metrics["sharpe"] == 0.0
        assert metrics["win_rate"] == 0.0
