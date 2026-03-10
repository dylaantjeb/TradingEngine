"""
Math utilities: performance metrics, position sizing helpers.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Return statistics
# ──────────────────────────────────────────────────────────────────────────────


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    if len(returns) < 2:
        return 0.0
    mu  = np.mean(returns)
    sig = np.std(returns, ddof=1)
    if sig == 0:
        return 0.0
    return float(mu / sig * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    if len(returns) < 2:
        return 0.0
    mu   = np.mean(returns)
    neg  = returns[returns < 0]
    if len(neg) == 0:
        return float("inf")
    dd   = np.std(neg, ddof=1)
    if dd == 0:
        return 0.0
    return float(mu / dd * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown as a fraction (0.0 – 1.0).
    equity_curve should be cumulative P&L or portfolio value series.
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd   = (equity_curve - peak) / (np.abs(peak) + 1e-12)
    return float(np.min(dd))


def calmar_ratio(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    if len(equity_curve) < 2:
        return 0.0
    total_return = (equity_curve[-1] - equity_curve[0]) / (abs(equity_curve[0]) + 1e-12)
    n_years = len(equity_curve) / periods_per_year
    ann_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return float("inf")
    return float(ann_return / mdd)


def profit_factor(pnls: np.ndarray) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    gross_profit = pnls[pnls > 0].sum()
    gross_loss   = abs(pnls[pnls < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Position sizing
# ──────────────────────────────────────────────────────────────────────────────


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly fraction.

    f* = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win / avg_loss.
    Clamped to [0, 1].
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f = (win_rate * b - q) / b
    return float(np.clip(f, 0.0, 1.0))


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Half-Kelly (more conservative, typically preferred in practice)."""
    return kelly_fraction(win_rate, avg_win, avg_loss) * 0.5


def atr_position_size(
    account_equity: float,
    risk_per_trade_frac: float,
    atr_pts: float,
    multiplier: float,
    sl_atr_multiple: float = 1.5,
) -> int:
    """
    Number of contracts to trade using ATR-based stop.

    risk_dollars = account_equity * risk_per_trade_frac
    stop_pts     = atr_pts * sl_atr_multiple
    contracts    = risk_dollars / (stop_pts * multiplier)
    """
    if atr_pts <= 0 or multiplier <= 0:
        return 1
    risk_dollars = account_equity * risk_per_trade_frac
    stop_value   = atr_pts * sl_atr_multiple * multiplier
    contracts    = risk_dollars / stop_value
    return max(1, int(contracts))


def fixed_fractional_size(
    account_equity: float,
    risk_per_trade_frac: float,
    stop_pts: float,
    multiplier: float,
) -> int:
    """Fixed-fractional: risk X% of account on a fixed stop in points."""
    if stop_pts <= 0 or multiplier <= 0:
        return 1
    risk_dollars = account_equity * risk_per_trade_frac
    stop_value   = stop_pts * multiplier
    return max(1, int(risk_dollars / stop_value))
