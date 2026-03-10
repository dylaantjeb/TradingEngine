"""
Risk manager for TradingEngine.

Enforces hard risk limits:
  - Daily loss limit (drawdown from start-of-day equity)
  - Maximum drawdown guard (from session high-water mark)
  - Kill switch (manual or automatic disable)
  - Max concurrent positions (always 1 for single-symbol)

Usage
-----
rm = RiskManager(daily_loss_limit_usd=500, max_drawdown_frac=0.05)
ok, reason = rm.check_new_trade(signal=1)
if not ok:
    log.warning("Trade blocked: %s", reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RiskParams:
    """Risk limits configuration."""
    daily_loss_limit_usd: float   = 1_000.0    # max daily loss before kill
    max_drawdown_frac: float      = 0.05        # 5% of peak equity
    max_position_size: int        = 5           # max contracts per symbol
    max_concurrent_positions: int = 1           # single-symbol: always 1
    kill_switch: bool             = False       # manual emergency stop


class RiskManager:
    """
    Stateful risk manager.  Call update() after each bar, check_new_trade()
    before each entry/reversal.
    """

    def __init__(self, params: RiskParams | None = None):
        self.params = params or RiskParams()

        # Session-level state
        self._session_start_equity: float | None = None
        self._peak_equity: float          = 0.0
        self._current_equity: float       = 0.0
        self._position: int               = 0
        self._daily_loss: float           = 0.0
        self._last_date: Optional[date]   = None
        self._killed: bool                = False

    # ── State updates ──────────────────────────────────────────────────────────

    def set_initial_equity(self, equity: float) -> None:
        """Call once at session start with starting equity."""
        self._session_start_equity = equity
        self._current_equity = equity
        self._peak_equity = equity

    def update(self, equity: float, position: int, current_date: date | None = None) -> None:
        """
        Update internal state after each bar.

        Parameters
        ----------
        equity       : Current cumulative net P&L.
        position     : Current position (+1 long, -1 short, 0 flat).
        current_date : Trading date for daily reset (optional).
        """
        self._current_equity = equity
        self._position = position
        self._peak_equity = max(self._peak_equity, equity)

        # Daily loss reset
        if current_date is not None and current_date != self._last_date:
            self._daily_loss = 0.0
            self._last_date  = current_date

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a completed trade P&L for daily loss tracking."""
        if pnl < 0:
            self._daily_loss += abs(pnl)
        self._current_equity += pnl
        self._peak_equity = max(self._peak_equity, self._current_equity)

        # Auto-kill on limit breach
        if self._daily_loss >= self.params.daily_loss_limit_usd:
            if not self._killed:
                log.warning(
                    "RISK: Daily loss limit breached (%.2f >= %.2f). Kill switch activated.",
                    self._daily_loss, self.params.daily_loss_limit_usd,
                )
                self._killed = True

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if drawdown >= self.params.max_drawdown_frac:
                if not self._killed:
                    log.warning(
                        "RISK: Max drawdown breached (%.2f%% >= %.2f%%). Kill switch activated.",
                        drawdown * 100, self.params.max_drawdown_frac * 100,
                    )
                    self._killed = True

    # ── Trade gate ─────────────────────────────────────────────────────────────

    def check_new_trade(
        self,
        signal: int,
        size: int = 1,
    ) -> tuple[bool, str]:
        """
        Decide whether a new trade (entry or reversal) is allowed.

        Returns
        -------
        (allowed: bool, reason: str)
        """
        # Manual kill switch
        if self.params.kill_switch or self._killed:
            return False, "kill_switch_active"

        # Flat signal – always allowed to close
        if signal == 0:
            return True, "ok"

        # Daily loss limit
        if self._daily_loss >= self.params.daily_loss_limit_usd:
            return False, (
                f"daily_loss_limit_breached "
                f"({self._daily_loss:.2f} >= {self.params.daily_loss_limit_usd:.2f})"
            )

        # Max drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if drawdown >= self.params.max_drawdown_frac:
                return False, (
                    f"max_drawdown_breached "
                    f"({drawdown:.2%} >= {self.params.max_drawdown_frac:.2%})"
                )

        # Position size
        if size > self.params.max_position_size:
            return False, f"size_exceeds_limit ({size} > {self.params.max_position_size})"

        return True, "ok"

    # ── Manual controls ────────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "manual") -> None:
        """Manually activate kill switch."""
        log.warning("RISK: Kill switch activated manually. Reason: %s", reason)
        self._killed = True

    def reset_kill_switch(self) -> None:
        """Reset kill switch (use with caution)."""
        log.info("RISK: Kill switch reset.")
        self._killed = False

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def is_killed(self) -> bool:
        return self._killed or self.params.kill_switch

    def status(self) -> dict:
        return {
            "killed":             self.is_killed,
            "daily_loss_usd":     round(self._daily_loss, 2),
            "daily_loss_limit":   self.params.daily_loss_limit_usd,
            "current_equity":     round(self._current_equity, 2),
            "peak_equity":        round(self._peak_equity, 2),
            "drawdown_frac":      round(
                max(0.0, (self._peak_equity - self._current_equity) / max(self._peak_equity, 1)),
                4,
            ),
            "max_drawdown_limit": self.params.max_drawdown_frac,
            "position":           self._position,
        }
