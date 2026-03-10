"""
Position sizing strategies for TradingEngine.

All sizers return an integer number of contracts (≥ 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.utils.math_utils import (
    atr_position_size,
    fixed_fractional_size,
    half_kelly,
    kelly_fraction,
)

log = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    FIXED       = "fixed"
    FIXED_FRAC  = "fixed_frac"
    ATR         = "atr"
    KELLY       = "kelly"
    HALF_KELLY  = "half_kelly"


@dataclass
class SizingParams:
    method: SizingMethod = SizingMethod.FIXED
    fixed_contracts: int = 1
    risk_per_trade_frac: float = 0.01     # 1% of equity per trade
    sl_atr_multiple: float = 1.5          # stop = sl_atr_multiple × ATR
    kelly_lookback: int = 50              # trades to estimate win-rate / avg P&L


def compute_size(
    method: SizingMethod,
    fixed_contracts: int,
    risk_per_trade_frac: float,
    sl_atr_multiple: float,
    account_equity: float,
    atr_pts: float,
    multiplier: float,
    recent_pnls: Optional[np.ndarray] = None,
) -> int:
    """
    Return number of contracts to trade.

    Parameters
    ----------
    method              : Sizing algorithm to use.
    fixed_contracts     : Contracts for FIXED method.
    risk_per_trade_frac : Fraction of equity to risk per trade.
    sl_atr_multiple     : Stop-loss distance in ATR multiples.
    account_equity      : Current account equity in USD.
    atr_pts             : Current ATR in price points.
    multiplier          : Contract point value (e.g. $50 for ES).
    recent_pnls         : Array of recent trade P&L (for Kelly methods).
    """
    if method == SizingMethod.FIXED:
        return max(1, fixed_contracts)

    if method == SizingMethod.FIXED_FRAC:
        stop_pts = atr_pts * sl_atr_multiple
        return fixed_fractional_size(account_equity, risk_per_trade_frac, stop_pts, multiplier)

    if method == SizingMethod.ATR:
        return atr_position_size(
            account_equity, risk_per_trade_frac, atr_pts, multiplier, sl_atr_multiple
        )

    if method in (SizingMethod.KELLY, SizingMethod.HALF_KELLY):
        if recent_pnls is None or len(recent_pnls) < 10:
            log.debug("Kelly: insufficient trade history, defaulting to fixed_frac")
            return fixed_fractional_size(account_equity, risk_per_trade_frac, atr_pts * sl_atr_multiple, multiplier)

        wins   = recent_pnls[recent_pnls > 0]
        losses = recent_pnls[recent_pnls < 0]

        win_rate = len(wins) / len(recent_pnls)
        avg_win  = float(np.mean(wins))  if len(wins)   > 0 else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 1.0

        if method == SizingMethod.KELLY:
            frac = kelly_fraction(win_rate, avg_win, avg_loss)
        else:
            frac = half_kelly(win_rate, avg_win, avg_loss)

        # Convert fraction → contracts: risk frac*equity, divide by stop value
        risk_dollars = account_equity * frac
        stop_value   = atr_pts * sl_atr_multiple * multiplier
        if stop_value <= 0:
            return 1
        return max(1, int(risk_dollars / stop_value))

    return 1


class PositionSizer:
    """Stateful sizer that tracks trade history for Kelly sizing."""

    def __init__(self, params: SizingParams | None = None):
        self.params = params or SizingParams()
        self._trade_pnls: list[float] = []

    def record_trade(self, pnl: float) -> None:
        """Record a closed trade P&L for Kelly estimation."""
        self._trade_pnls.append(pnl)

    def size(
        self,
        account_equity: float,
        atr_pts: float,
        multiplier: float,
    ) -> int:
        recent = (
            np.array(self._trade_pnls[-self.params.kelly_lookback:])
            if self._trade_pnls
            else None
        )
        return compute_size(
            method=self.params.method,
            fixed_contracts=self.params.fixed_contracts,
            risk_per_trade_frac=self.params.risk_per_trade_frac,
            sl_atr_multiple=self.params.sl_atr_multiple,
            account_equity=account_equity,
            atr_pts=atr_pts,
            multiplier=multiplier,
            recent_pnls=recent,
        )
