"""
Funded-account governance state machine.

GovState machine
----------------
  SAFE → CAUTION → REDUCED_RISK → HALTED

Transitions tighten *during* the session on bad events (losses, drawdown).
They loosen on a new calendar day — except HALTED, which requires a manual
``manual_reset_halt()`` call.

Design principles
-----------------
* Only ``on_exit()`` can trigger state tightening — we never tighten on
  live unrealised P&L because that creates premature halts on normal
  intra-trade drawdown.
* HALTED is terminal within a session; the operator must explicitly reset.
* ``validate_entry()`` is the single gate consulted before any new trade.
  It returns (allowed: bool, reason: str).
* ``size_contracts()`` applies a per-state size factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


# ── State enumeration ─────────────────────────────────────────────────────────

class GovState(str, Enum):
    SAFE         = "SAFE"
    CAUTION      = "REDUCED"        # mild: consec_losses==2 or daily_pnl <= -soft
    REDUCED_RISK = "REDUCED_RISK"   # moderate: triggered on new day after prior caution
    HALTED       = "HALTED"         # terminal: daily hard limit or max drawdown hit


# ── FundedGuard ───────────────────────────────────────────────────────────────

@dataclass
class FundedGuard:
    """
    Runtime governance for a funded-account strategy session.

    Parameters
    ----------
    daily_loss_hard_usd     : Hard daily loss limit. HALTED immediately when hit.
    daily_loss_soft_usd     : Soft daily loss limit. Transitions to CAUTION.
    trailing_drawdown_usd   : Trailing high-water drawdown limit.
    max_drawdown_usd        : Absolute max drawdown limit. HALTED when hit.
    max_consecutive_losses  : HALTED after this many consecutive losses.
    starting_equity         : Account starting balance (USD).
    cooldown_bars_caution   : Extra cooldown bars in CAUTION / REDUCED_RISK state.
    """
    daily_loss_hard_usd:    float = 500.0
    daily_loss_soft_usd:    float = 300.0
    trailing_drawdown_usd:  float = 2000.0
    max_drawdown_usd:       float = 3000.0
    max_consecutive_losses: int   = 3
    starting_equity:        float = 50_000.0
    cooldown_bars_caution:  int   = 3

    # ── Runtime state (not constructor params) ────────────────────────────────
    state:              GovState = field(default=GovState.SAFE, init=False)
    equity:             float    = field(default=0.0, init=False)   # cumulative P&L delta
    peak_equity:        float    = field(default=0.0, init=False)
    daily_pnl:          float    = field(default=0.0, init=False)
    consecutive_losses: int      = field(default=0,   init=False)
    trades_today:       int      = field(default=0,   init=False)
    _last_date:         Optional[date] = field(default=None, init=False)
    _state_history:     list     = field(default_factory=list, init=False)
    _prior_day_caution: bool     = field(default=False, init=False)

    # ── State transitions ─────────────────────────────────────────────────────

    def _transition(self, new_state: GovState, reason: str) -> None:
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state
        self._state_history.append({
            "from":   old.value,
            "to":     new_state.value,
            "reason": reason,
            "equity": round(self.equity, 2),
            "daily":  round(self.daily_pnl, 2),
        })
        log.warning(
            "[FundedGuard] %s → %s  reason=%s  daily_pnl=%.2f  equity=%.2f",
            old.value, new_state.value, reason, self.daily_pnl, self.equity,
        )

    def _check_transitions(self) -> None:
        """Evaluate whether current metrics warrant a state upgrade (tighten)."""
        if self.state == GovState.HALTED:
            return  # terminal — needs manual reset

        # Hard halt conditions
        if -self.daily_pnl >= self.daily_loss_hard_usd:
            self._transition(GovState.HALTED, f"daily_loss_hard=${-self.daily_pnl:.0f}")
            return

        drawdown = self.peak_equity - self.equity
        if drawdown >= self.max_drawdown_usd:
            self._transition(GovState.HALTED, f"max_drawdown=${drawdown:.0f}")
            return

        trailing_dd = max(0.0, self.peak_equity - self.equity)
        if trailing_dd >= self.trailing_drawdown_usd:
            self._transition(GovState.HALTED, f"trailing_drawdown=${trailing_dd:.0f}")
            return

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._transition(GovState.HALTED, f"consec_losses={self.consecutive_losses}")
            return

        # Soft / caution conditions
        if self.state in (GovState.SAFE, GovState.REDUCED_RISK):
            if (-self.daily_pnl >= self.daily_loss_soft_usd
                    or self.consecutive_losses >= 2):
                self._transition(
                    GovState.CAUTION,
                    f"soft daily_pnl={self.daily_pnl:.0f} or consec={self.consecutive_losses}",
                )

    # ── Public hooks ─────────────────────────────────────────────────────────

    def on_bar(self, ts: datetime) -> None:
        """
        Called once per bar before entry decisions.
        Handles calendar-day boundary resets.
        """
        today = ts.date() if hasattr(ts, "date") else ts
        if today != self._last_date:
            self._prior_day_caution = self.state in (
                GovState.CAUTION, GovState.REDUCED_RISK,
            )
            self._last_date   = today
            self.daily_pnl    = 0.0
            self.trades_today = 0
            # Clear halt if it was daily-level (hard daily reset); preserve
            # drawdown-based halts — those persist until manual reset.
            if self.state == GovState.HALTED:
                # Don't auto-clear — operator must call manual_reset_halt()
                pass
            elif self.state in (GovState.CAUTION,):
                if self._prior_day_caution:
                    self._transition(GovState.REDUCED_RISK, "new_day_after_caution")
                else:
                    self._transition(GovState.SAFE, "new_day_clear")
            elif self.state == GovState.REDUCED_RISK:
                # Stay at REDUCED_RISK until operator manually resets or drawdown clears
                pass
            else:
                self._transition(GovState.SAFE, "new_day_clear")

    def on_entry(self) -> None:
        """Called when a trade is entered."""
        self.trades_today += 1

    def on_exit(self, net_pnl: float) -> None:
        """
        Called when a trade is closed.
        Updates equity, daily P&L, consecutive losses, then re-evaluates state.
        """
        self.equity      += net_pnl
        self.daily_pnl   += net_pnl
        self.peak_equity  = max(self.peak_equity, self.equity)

        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self._check_transitions()

    def validate_entry(
        self,
        bar_ts: datetime,
        atr_ticks: float = 0.0,
        atr_min_ticks: float = 5.0,
        max_trades_per_day: int = 4,
    ) -> tuple[bool, str]:
        """
        Gate check before queuing a new entry.

        Returns
        -------
        (allowed, reason)  — reason is "" when allowed=True.
        """
        if self.state == GovState.HALTED:
            return False, "HALTED"

        if atr_ticks > 0 and atr_ticks < atr_min_ticks:
            return False, f"atr_too_low({atr_ticks:.1f}<{atr_min_ticks})"

        if self.trades_today >= max_trades_per_day:
            return False, f"max_trades_per_day({self.trades_today}>={max_trades_per_day})"

        return True, ""

    def size_contracts(self, base: int = 1, max_contracts: int = 2) -> int:
        """
        Return position size in contracts, adjusted for governance state.

        SAFE          : base
        CAUTION       : base (no size increase; state is cautionary)
        REDUCED_RISK  : max(1, base // 2)
        HALTED        : 0 (should not be called — validate_entry blocks it)
        """
        if self.state == GovState.HALTED:
            return 0
        if self.state == GovState.REDUCED_RISK:
            return max(1, base // 2)
        return min(base, max_contracts)

    def manual_reset_halt(self, reason: str = "manual") -> None:
        """
        Operator-called reset of a HALTED state.
        Only valid when state == HALTED; logs a warning otherwise.
        """
        if self.state != GovState.HALTED:
            log.warning("[FundedGuard] manual_reset_halt called but state=%s", self.state.value)
            return
        self._transition(GovState.SAFE, f"manual_reset: {reason}")
        self.consecutive_losses = 0
        log.info("[FundedGuard] HALT cleared by operator (%s)", reason)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def session_summary(self) -> dict:
        """Return a snapshot dict suitable for JSON serialisation."""
        return {
            "state":              self.state.value,
            "equity_delta":       round(self.equity, 2),
            "peak_equity_delta":  round(self.peak_equity, 2),
            "daily_pnl":          round(self.daily_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "trades_today":       self.trades_today,
            "drawdown":           round(self.peak_equity - self.equity, 2),
            "state_history":      self._state_history,
        }
