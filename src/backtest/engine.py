"""
Vectorised backtest engine – hardened against lookahead and overtrading.

Execution model
───────────────
  Signal generated at bar t close → order submitted → filled at bar t+1 OPEN.
  `execution_delay_bars` (default 1) controls the lag; 0 = same-bar (leaky).

Cost model per round trip (one contract)
─────────────────────────────────────────
  friction_pts_per_side = slippage_ticks_per_side * tick_size
                        + spread_ticks * 0.5 * tick_size

  entry_fill = next_open + direction * friction_pts_per_side   (adverse)
  exit_fill  = next_open - direction * friction_pts_per_side   (adverse)
  commission = 2 * commission_per_side_usd

  net_pnl = direction * (exit_fill - entry_fill) * multiplier * n_contracts
          - commission * n_contracts

Prop-firm safety filters (applied at signal time)
──────────────────────────────────────────────────
  1.  Confidence gating   : model confidence must exceed threshold
  2.  Session filter      : UTC-hour window
  3.  News blackout       : configurable HH:MM windows
  4.  ATR filter          : volatility in-range check
  5.  Trend filter        : close vs EMA(200) — blocks counter-trend trades
  6.  Risk gate           : daily halt / kill switch
  7.  Loss cooldown       : bars to skip after a losing trade
  8.  Exit cooldown       : bars to skip after any exit
  9.  Min holding period  : prevent premature exits
  10. Daily trade cap     : hard ceiling on new entries per calendar day

Report saved to:  artifacts/reports/<SYM>_backtest.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_UNIVERSE_CFG_PATH = Path("config/universe.yaml")

# ── Default parameters (overridden by universe.yaml) ──────────────────────────
_DEFAULTS = {
    # Execution
    "execution_delay_bars": 1,
    # Cost model
    "commission_per_side_usd": 1.50,
    "slippage_ticks_per_side": 0.5,
    "spread_ticks": 1.0,
    # Filters
    "session_start_utc_hour": 9,
    "session_end_utc_hour": 22,
    "atr_min_ticks": 4,
    "atr_max_ticks": 200,
    "news_blackout_windows": [],
    # Throttles
    "max_trades_per_day": 6,
    "min_holding_bars": 5,
    "cooldown_bars_after_exit": 5,
    # Confidence gating (0 = disabled — safe default for backward compat)
    "min_long_confidence": 0.0,
    "min_short_confidence": 0.0,
    # Trend filter (disabled by default — safe for backward compat)
    "trend_filter_enabled": False,
    "trend_filter_ema_period": 200,
    # Risk limits (very large = effectively disabled by default)
    "starting_equity": 100_000.0,
    "max_daily_loss_usd": 1e15,
    "max_daily_loss_pct": 1.0,
    "max_total_drawdown_usd": 1e15,
    "max_total_drawdown_pct": 1.0,
    "max_consecutive_losses": 9999,
    "cooldown_bars_after_loss": 0,
    # Position sizing
    "position_sizing_method": "fixed",
    "fixed_contracts": 1,
    "risk_per_trade_usd": 500.0,
    "risk_per_trade_pct": 0.01,
    "atr_stop_multiplier": 1.5,
    "max_contracts": 3,
    "tick_size": 0.25,
}


def _load_universe_cfg() -> dict:
    if not _UNIVERSE_CFG_PATH.exists():
        return {}
    with open(_UNIVERSE_CFG_PATH) as f:
        return yaml.safe_load(f) or {}


def _build_run_cfg(universe_cfg: dict, overrides: dict) -> dict:
    """Merge universe.yaml sections + CLI overrides into a flat config dict."""
    cfg = dict(_DEFAULTS)
    # Core sections
    cfg.update(universe_cfg.get("execution", {}))
    cfg.update(universe_cfg.get("cost_model", {}))
    cfg.update(universe_cfg.get("filters", {}))
    cfg.update(universe_cfg.get("throttles", {}))
    # Confidence
    cfg.update(universe_cfg.get("confidence", {}))
    # Trend filter (nested keys need remapping)
    tf = universe_cfg.get("trend_filter", {})
    if "enabled" in tf:
        cfg["trend_filter_enabled"] = tf["enabled"]
    if "ema_period" in tf:
        cfg["trend_filter_ema_period"] = tf["ema_period"]
    # Risk limits
    cfg.update(universe_cfg.get("risk_limits", {}))
    # Position sizing (method key remapped to avoid collision)
    ps = universe_cfg.get("position_sizing", {})
    if "method" in ps:
        cfg["position_sizing_method"] = ps["method"]
    for k, v in ps.items():
        if k != "method":
            cfg[k] = v
    # CLI overrides win
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _load_cfg(symbol: str) -> dict:
    """Return flat run-config dict for symbol (used by walk_forward)."""
    universe_cfg = _load_universe_cfg()
    return _build_run_cfg(universe_cfg, {})


def run_backtest(
    symbol: str,
    universe_cfg_path: Path = _UNIVERSE_CFG_PATH,
    # CLI override kwargs (None = use config default)
    execution_delay_bars: int | None = None,
    max_trades_per_day: int | None = None,
    slippage_ticks_per_side: float | None = None,
    commission_per_side_usd: float | None = None,
) -> dict[str, Any]:
    """
    Load model + features for `symbol`, run hardened backtest, persist report.
    Returns metrics dict.
    """
    try:
        import joblib
    except ImportError:
        log.error("joblib not installed:  pip install joblib")
        sys.exit(1)

    universe_cfg = _load_universe_cfg()
    cfg = _build_run_cfg(
        universe_cfg,
        {
            "execution_delay_bars": execution_delay_bars,
            "max_trades_per_day": max_trades_per_day,
            "slippage_ticks_per_side": slippage_ticks_per_side,
            "commission_per_side_usd": commission_per_side_usd,
        },
    )

    delay = int(cfg["execution_delay_bars"])
    if delay == 0:
        log.warning(
            "execution_delay_bars=0 → same-bar fills. This is LEAKY and should only be "
            "used for debugging. Set to 1 for realistic simulation."
        )

    # ── Contract specs ──────────────────────────────────────────────────────────
    specs = universe_cfg.get("contract_specs", {}).get(symbol, {})
    tick_size   = float(specs.get("tick_size", cfg.get("tick_size", 0.25)))
    tick_value  = float(specs.get("tick_value", 12.50))
    multiplier  = float(specs.get("multiplier", 50))
    cfg["tick_size"] = tick_size

    # ── Derived cost params ─────────────────────────────────────────────────────
    commission_rt     = 2.0 * float(cfg["commission_per_side_usd"])
    slippage_pts      = float(cfg["slippage_ticks_per_side"]) * tick_size
    half_spread_pts   = float(cfg["spread_ticks"]) * 0.5 * tick_size
    friction_per_side = slippage_pts + half_spread_pts

    log.info(
        "Cost model [%s]: commission RT=$%.2f | slippage=%.4f pts/side | "
        "half-spread=%.4f pts/side | friction_per_side=%.4f pts | "
        "total RT (excl. price move) ≈ $%.2f",
        symbol, commission_rt, slippage_pts, half_spread_pts,
        friction_per_side,
        2 * friction_per_side * multiplier + commission_rt,
    )

    # ── Load artifacts ──────────────────────────────────────────────────────────
    model_path  = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    scaler_path = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    schema_path = Path(f"artifacts/schema/{symbol}_features.json")
    feat_path   = Path(f"data/processed/{symbol}_features.parquet")
    raw_path    = Path(f"data/raw/{symbol}_M1.csv")

    for p in (model_path, scaler_path, schema_path, feat_path):
        if not p.exists():
            log.error(
                "Missing artifact: %s\n  Run:  python -m src.cli train --symbol %s", p, symbol
            )
            sys.exit(1)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_names: list[str] = schema["feature_names"]
    inv_label_map: dict[str, int] = {k: int(v) for k, v in schema["inv_label_map"].items()}

    # ── Load features ───────────────────────────────────────────────────────────
    try:
        features = pd.read_parquet(feat_path, engine="pyarrow")
    except ImportError:
        log.error("pyarrow required:  pip install pyarrow")
        sys.exit(1)

    missing_cols = set(feature_names) - set(features.columns)
    if missing_cols:
        log.error("Feature mismatch: missing %s – rebuild dataset", missing_cols)
        sys.exit(1)

    # ── Label alignment guardrail ───────────────────────────────────────────────
    from src.utils.alignment import check_label_alignment
    check_label_alignment(features, symbol=symbol)

    # ── Generate predictions ─────────────────────────────────────────────────
    # Pass DataFrame (not .values) to preserve feature_names_in_ → no scaler warning
    X_df = features[feature_names]
    X_scaled = scaler.transform(X_df)
    proba = model.predict_proba(X_scaled)
    pred_encoded = np.argmax(proba, axis=1)
    lookup = np.array([inv_label_map[str(k)] for k in range(len(inv_label_map))], dtype=np.int8)
    signal = lookup[pred_encoded]
    confidence = np.max(proba, axis=1)

    # ── Load raw prices (need 'open' for next-bar fills) ──────────────────────
    if raw_path.exists():
        prices = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
        prices.sort_index(inplace=True)
    else:
        log.warning("Raw CSV not found at %s; using close-only proxy (suboptimal)", raw_path)
        prices = pd.DataFrame(
            {"open": features.get("close", pd.Series(dtype=float)),
             "close": features.get("close", pd.Series(dtype=float))},
            index=features.index,
        )

    # Align to common index
    common_idx = features.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = features.index
        log.warning("No common index between features and prices; using features index only")

    sig_series    = pd.Series(signal, index=features.index).reindex(common_idx)
    conf_series   = pd.Series(confidence, index=features.index).reindex(common_idx)
    open_prices   = prices["open"].reindex(common_idx) if "open" in prices.columns \
                    else prices["close"].reindex(common_idx)
    close_prices  = prices["close"].reindex(common_idx)

    # ATR in ticks: atr_14 feature is atr_pts/close
    atr_ticks_series = (
        features["atr_14"].reindex(common_idx) * close_prices / tick_size
    ).fillna(0)

    # ── EMA for trend filter ────────────────────────────────────────────────────
    ema_series = None
    if cfg.get("trend_filter_enabled", False):
        ema_period = int(cfg.get("trend_filter_ema_period", 200))
        ema_series = close_prices.ewm(span=ema_period, adjust=False).mean()
        log.info("Trend filter enabled: EMA(%d)", ema_period)

    # ── Simulate trades ─────────────────────────────────────────────────────────
    trades, equity_curve, cost_summary = _simulate_trades(
        signals       = sig_series,
        open_prices   = open_prices,
        atr_ticks     = atr_ticks_series,
        cfg           = cfg,
        friction_pts  = friction_per_side,
        commission_rt = commission_rt,
        multiplier    = multiplier,
        conf_series   = conf_series,
        close_prices  = close_prices,
        ema_series    = ema_series,
    )

    # ── Metrics ─────────────────────────────────────────────────────────────────
    metrics = _compute_metrics(
        equity_curve, trades, cost_summary,
        starting_equity=float(cfg.get("starting_equity", 100_000.0)),
    )
    metrics["symbol"]  = symbol
    metrics["n_bars"]  = len(features)
    metrics["n_trades"] = len(trades)

    _print_metrics_table(symbol, metrics)

    # ── Save report ─────────────────────────────────────────────────────────────
    report_dir  = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{symbol}_backtest.json"

    report = {
        "symbol":       symbol,
        "run_at":       datetime.utcnow().isoformat(),
        "cfg":          {k: str(v) for k, v in cfg.items()},
        "metrics":      metrics,
        "trades":       trades[:500],
        "equity_curve": equity_curve.round(2).tolist(),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Report saved to %s", report_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────


def _in_session(ts: pd.Timestamp, start_h: int, end_h: int) -> bool:
    """True if `ts` falls within [start_h, end_h) UTC hours."""
    if start_h == 0 and end_h >= 24:
        return True
    h = ts.hour + ts.minute / 60.0
    return start_h <= h < end_h


def _in_blackout(ts: pd.Timestamp, windows: list) -> bool:
    """True if `ts` falls inside any news-blackout window (UTC HH:MM)."""
    if not windows:
        return False
    t = dt_time(ts.hour, ts.minute)
    for w in windows:
        try:
            s_h, s_m = map(int, w[0].split(":"))
            e_h, e_m = map(int, w[1].split(":"))
            if dt_time(s_h, s_m) <= t <= dt_time(e_h, e_m):
                return True
        except Exception:
            pass
    return False


def _simulate_trades(
    signals: pd.Series,
    open_prices: pd.Series,
    atr_ticks: pd.Series,
    cfg: dict,
    friction_pts: float,
    commission_rt: float,
    multiplier: float,
    conf_series: Optional[pd.Series] = None,
    close_prices: Optional[pd.Series] = None,
    ema_series: Optional[pd.Series] = None,
) -> tuple[list[dict], pd.Series, dict]:
    """
    Bar-by-bar simulation with 1-bar execution delay, full cost model,
    10-filter prop-firm safety pipeline, and position sizing.

    Backward compatible: conf_series/close_prices/ema_series are optional.
    When omitted, those filters are skipped (safe for old test calls).

    Signal at bar i → pending order → filled at bar i+1 open.
    """
    # ── Static config ────────────────────────────────────────────────────────
    delay         = int(cfg.get("execution_delay_bars", 1))
    sess_start    = int(cfg.get("session_start_utc_hour", 0))
    sess_end      = int(cfg.get("session_end_utc_hour", 24))
    atr_min       = float(cfg.get("atr_min_ticks", 0))
    atr_max       = float(cfg.get("atr_max_ticks", 1e9))
    blackouts     = cfg.get("news_blackout_windows", [])
    max_tpd       = int(cfg.get("max_trades_per_day", 9999))
    min_hold      = int(cfg.get("min_holding_bars", 1))
    cooldown_bars = int(cfg.get("cooldown_bars_after_exit", 0))

    # Confidence gating (default 0.0 = no threshold → backward compat)
    min_long_conf  = float(cfg.get("min_long_confidence", 0.0))
    min_short_conf = float(cfg.get("min_short_confidence", 0.0))

    # Trend filter (default False → backward compat)
    trend_filter_on = bool(cfg.get("trend_filter_enabled", False))

    # Risk limits (very large defaults → backward compat)
    starting_equity        = float(cfg.get("starting_equity", 100_000.0))
    max_daily_loss_usd     = float(cfg.get("max_daily_loss_usd", 1e15))
    max_daily_loss_pct     = float(cfg.get("max_daily_loss_pct", 1.0))
    max_total_dd_usd       = float(cfg.get("max_total_drawdown_usd", 1e15))
    max_total_dd_pct       = float(cfg.get("max_total_drawdown_pct", 1.0))
    max_consec_losses      = int(cfg.get("max_consecutive_losses", 9999))
    loss_cooldown_cfg      = int(cfg.get("cooldown_bars_after_loss", 0))

    # Position sizing
    ps_method      = str(cfg.get("position_sizing_method", "fixed"))
    fixed_contracts = int(cfg.get("fixed_contracts", 1))
    risk_usd       = float(cfg.get("risk_per_trade_usd", 500.0))
    risk_pct       = float(cfg.get("risk_per_trade_pct", 0.01))
    atr_stop_mult  = float(cfg.get("atr_stop_multiplier", 1.5))
    max_contracts  = int(cfg.get("max_contracts", 3))
    tick_size      = float(cfg.get("tick_size", 0.25))

    # ── Pre-allocate arrays ──────────────────────────────────────────────────
    sig_arr   = signals.values.astype(np.int8)
    open_arr  = open_prices.reindex(signals.index).values.astype(np.float64)
    atr_arr   = atr_ticks.reindex(signals.index).values.astype(np.float64)
    ts_arr    = signals.index
    n         = len(sig_arr)

    conf_arr  = None
    if conf_series is not None:
        conf_arr = conf_series.reindex(signals.index).values.astype(np.float64)

    close_arr = None
    if close_prices is not None:
        close_arr = close_prices.reindex(signals.index).values.astype(np.float64)

    ema_arr = None
    if ema_series is not None:
        ema_arr = ema_series.reindex(signals.index).values.astype(np.float64)

    equity_arr = np.zeros(n, dtype=np.float64)
    trades: list[dict] = []

    # ── Mutable state ────────────────────────────────────────────────────────
    position    = 0        # current: -1, 0, +1
    entry_price = 0.0
    entry_idx   = 0
    entry_contracts = 1
    equity      = 0.0
    gross_pnl   = 0.0
    total_cost  = 0.0

    # Throttle state
    cooldown_left     = 0
    daily_trade_count = 0
    last_trade_date   = None
    pending_target    = None
    pending_contracts = 1

    # Risk / prop-firm state
    daily_pnl          = 0.0
    peak_equity        = 0.0       # max equity achieved (for drawdown)
    consecutive_losses = 0
    consec_losses_max  = 0
    daily_halt         = False
    kill_switch_active = False
    loss_cooldown_left = 0
    daily_halt_count   = 0

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _size(atr_t: float) -> int:
        """Compute position size (contracts) based on sizing method."""
        if ps_method == "fixed_dollar_risk":
            atr_pts  = atr_t * tick_size
            stop_pts = atr_pts * atr_stop_mult
            if stop_pts > 0 and multiplier > 0:
                n_c = int(risk_usd / (stop_pts * multiplier))
            else:
                n_c = 1
        elif ps_method == "fixed_pct_risk":
            total_eq = starting_equity + equity
            r_usd    = total_eq * risk_pct
            atr_pts  = atr_t * tick_size
            stop_pts = atr_pts * atr_stop_mult
            if stop_pts > 0 and multiplier > 0:
                n_c = int(r_usd / (stop_pts * multiplier))
            else:
                n_c = 1
        else:  # "fixed"
            n_c = fixed_contracts
        return max(1, min(n_c, max_contracts))

    def _update_risk(net_pnl: float, ts) -> None:
        nonlocal equity, daily_pnl, peak_equity, consecutive_losses, consec_losses_max
        nonlocal daily_halt, kill_switch_active, loss_cooldown_left, daily_halt_count

        equity    += net_pnl
        daily_pnl += net_pnl
        peak_equity = max(peak_equity, equity)

        if net_pnl < 0:
            consecutive_losses += 1
            consec_losses_max   = max(consec_losses_max, consecutive_losses)
            if loss_cooldown_cfg > 0:
                loss_cooldown_left = loss_cooldown_cfg
        else:
            consecutive_losses = 0

        # Daily halt
        if not daily_halt:
            total_eq_before = starting_equity + equity - net_pnl
            daily_loss = -daily_pnl
            if (daily_loss >= max_daily_loss_usd
                    or (total_eq_before > 0 and daily_loss / total_eq_before >= max_daily_loss_pct)
                    or consecutive_losses >= max_consec_losses):
                daily_halt = True
                daily_halt_count += 1
                log.warning(
                    "[%s] Daily halt triggered — daily_pnl=%.2f consec_losses=%d",
                    ts, daily_pnl, consecutive_losses,
                )

        # Kill switch (permanent — does not reset with new day)
        if not kill_switch_active:
            drawdown = equity - peak_equity   # negative when in DD
            dd_base  = starting_equity + peak_equity
            if (-drawdown >= max_total_dd_usd
                    or (dd_base > 0 and -drawdown / dd_base >= max_total_dd_pct)):
                kill_switch_active = True
                log.warning(
                    "[%s] Kill switch activated — drawdown=%.2f (peak=%.2f)",
                    ts, drawdown, peak_equity,
                )

    # ── Main loop ────────────────────────────────────────────────────────────

    for i in range(n):
        ts    = ts_arr[i]
        today = ts.date()

        # ── Reset daily counters ─────────────────────────────────────────────
        if today != last_trade_date:
            daily_trade_count = 0
            last_trade_date   = today
            if daily_halt and not kill_switch_active:
                daily_halt = False   # kill switch stays; daily halt resets
                log.info("[%s] Daily halt cleared (new trading day)", ts)
            daily_pnl = 0.0

        # ── STEP 1: Execute pending order from previous bar at open[i] ───────
        if pending_target is not None and delay > 0:
            o = open_arr[i]
            if not np.isnan(o):
                n_c = pending_contracts   # contracts decided at signal time

                if position != 0 and pending_target != position:
                    # Exit (and possibly reverse)
                    exit_fill = o - position * friction_pts
                    raw_pnl   = position * (exit_fill - entry_price) * multiplier * entry_contracts
                    cost      = commission_rt * entry_contracts
                    net_pnl   = raw_pnl - cost
                    gross_pnl  += raw_pnl
                    total_cost += cost
                    trades.append({
                        "entry_time":    str(ts_arr[entry_idx]),
                        "exit_time":     str(ts),
                        "direction":     position,
                        "entry_price":   round(entry_price, 4),
                        "exit_price":    round(exit_fill, 4),
                        "hold_bars":     i - entry_idx,
                        "n_contracts":   entry_contracts,
                        "gross_pnl":     round(raw_pnl, 2),
                        "cost":          round(cost, 2),
                        "net_pnl":       round(net_pnl, 2),
                        "equity":        round(equity + net_pnl, 2),
                    })
                    _update_risk(net_pnl, ts)
                    if cooldown_bars > 0:
                        cooldown_left = cooldown_bars
                    position = 0

                    if pending_target != 0:
                        # Reversal into new position
                        entry_price     = o + pending_target * friction_pts
                        position        = pending_target
                        entry_idx       = i
                        entry_contracts = n_c

                elif position == 0 and pending_target != 0:
                    entry_price     = o + pending_target * friction_pts
                    position        = pending_target
                    entry_idx       = i
                    entry_contracts = n_c

            pending_target = None

        # ── STEP 2: Generate signal for this bar ──────────────────────────────
        o   = open_arr[i]
        sig = int(sig_arr[i])
        if np.isnan(o):
            equity_arr[i] = equity
            continue

        # Filter 1: Confidence gating
        if conf_arr is not None:
            conf_t = conf_arr[i] if not np.isnan(conf_arr[i]) else 0.0
            if sig == 1 and conf_t < min_long_conf:
                sig = 0
            elif sig == -1 and conf_t < min_short_conf:
                sig = 0

        # Filter 2: Session filter
        if not _in_session(ts, sess_start, sess_end):
            sig = 0

        # Filter 3: News blackout
        if _in_blackout(ts, blackouts):
            sig = 0

        # Filter 4: ATR filter
        atr_t = atr_arr[i]
        if not (atr_min <= atr_t <= atr_max):
            sig = 0

        # Filter 5: Trend filter (close vs EMA)
        if trend_filter_on and ema_arr is not None and close_arr is not None:
            ema_t   = ema_arr[i]
            close_t = close_arr[i]
            if not (np.isnan(ema_t) or np.isnan(close_t)):
                if sig == 1 and close_t < ema_t:    # long blocked (below EMA)
                    sig = 0
                elif sig == -1 and close_t > ema_t: # short blocked (above EMA)
                    sig = 0

        # Filter 6: Risk gate — kill switch / daily halt
        if kill_switch_active or daily_halt:
            if position == 0:
                sig = 0   # no new entries
            elif sig * position < 0:
                sig = 0   # convert reversal to plain exit

        # Filter 7: Loss cooldown (after losing trade)
        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            if position == 0:
                sig = 0

        # Filter 8: Exit cooldown (after any exit)
        if cooldown_left > 0:
            cooldown_left -= 1
            if position == 0:
                sig = 0

        # Filter 9: Minimum holding period
        if position != 0 and (i - entry_idx) < min_hold:
            sig = position   # force hold

        # Filter 10: Daily trade cap
        if sig != 0 and sig != position and daily_trade_count >= max_tpd:
            sig = 0 if position == 0 else position

        # ── Determine target and queue ────────────────────────────────────────
        target = int(sig)

        if delay == 0:
            # Same-bar fill (leaky debug mode)
            if target != position:
                if position != 0:
                    exit_fill = o - position * friction_pts
                    raw_pnl   = position * (exit_fill - entry_price) * multiplier * entry_contracts
                    cost      = commission_rt * entry_contracts
                    net_pnl   = raw_pnl - cost
                    gross_pnl  += raw_pnl
                    total_cost += cost
                    trades.append({
                        "entry_time":  str(ts_arr[entry_idx]),
                        "exit_time":   str(ts),
                        "direction":   position,
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(exit_fill, 4),
                        "hold_bars":   i - entry_idx,
                        "n_contracts": entry_contracts,
                        "gross_pnl":   round(raw_pnl, 2),
                        "cost":        round(cost, 2),
                        "net_pnl":     round(net_pnl, 2),
                        "equity":      round(equity + net_pnl, 2),
                    })
                    _update_risk(net_pnl, ts)
                    position = 0
                if target != 0:
                    n_c         = _size(atr_t)
                    entry_price     = o + target * friction_pts
                    position        = target
                    entry_idx       = i
                    entry_contracts = n_c
        else:
            # Normal delayed execution: queue for next bar
            if target != position and i < n - 1:
                pending_target    = target
                pending_contracts = _size(atr_t) if target != 0 else entry_contracts
                if target != 0:
                    daily_trade_count += 1

        equity_arr[i] = equity

    equity_series = pd.Series(equity_arr, index=signals.index, name="equity")
    cost_summary  = {
        "gross_pnl":              gross_pnl,
        "total_costs":            total_cost,
        "net_pnl":                gross_pnl - total_cost,
        "consecutive_losses_max": consec_losses_max,
        "kill_switch_triggered":  kill_switch_active,
        "daily_halt_count":       daily_halt_count,
    }
    return trades, equity_series, cost_summary


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _compute_metrics(
    equity: pd.Series,
    trades: list[dict],
    cost_summary: dict,
    starting_equity: float = 100_000.0,
) -> dict[str, Any]:
    """
    Compute performance metrics from an equity curve and trade list.

    Parameters
    ----------
    equity          : Cumulative P&L series (starts at 0, not at starting_equity).
    trades          : List of closed-trade dicts from _simulate_trades.
    cost_summary    : Dict with gross_pnl, total_costs, etc.
    starting_equity : Account starting capital — used as the denominator for
                      max_drawdown_pct so the % is always meaningful (e.g. -5%
                      means 5% of starting capital was lost at the trough).
    """
    if len(trades) == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0,
            "win_rate": 0.0, "expectancy_usd": 0.0, "total_pnl_usd": 0.0,
            "profit_factor": 0.0, "gross_pnl": 0.0, "total_costs": 0.0,
            "net_pnl": 0.0, "trades_per_day": 0.0,
            "avg_trade_duration_bars": 0.0, "rolling_sharpe_20d": None,
            "max_drawdown_usd": 0.0,
            "consecutive_losses_max": cost_summary.get("consecutive_losses_max", 0),
            "kill_switch_triggered": cost_summary.get("kill_switch_triggered", False),
            "daily_halt_count": cost_summary.get("daily_halt_count", 0),
        }

    net_pnls   = [t["net_pnl"]   for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    hold_bars  = [t["hold_bars"] for t in trades]
    total_pnl  = sum(net_pnls)
    wins       = [p for p in net_pnls if p > 0]
    losses     = [p for p in net_pnls if p <= 0]
    win_rate      = len(wins) / len(net_pnls)
    expectancy    = float(np.mean(net_pnls))
    profit_factor = (sum(wins) / (abs(sum(losses)) + 1e-9)) if losses else float("inf")

    # Sharpe / Sortino on bar-level equity changes (annualised, 390 bars/day)
    returns = equity.diff().dropna()
    mu      = returns.mean()
    sigma   = returns.std()
    annual_factor = np.sqrt(252 * 390)
    sharpe  = float((mu / sigma) * annual_factor) if sigma > 0 else 0.0
    down    = returns[returns < 0].std()
    sortino = float((mu / down) * annual_factor) if down > 0 else 0.0

    # Max drawdown
    cummax = equity.cummax()
    dd     = equity - cummax
    max_dd = float(dd.min())
    # Percentage relative to starting_equity — stable denominator for all regimes.
    # Avoids the exploding-% bug that occurs when cummax is 0 or very small
    # (e.g. a strategy that never became profitable gives cummax ≈ 0, so dividing
    # by cummax produces values like -50000%).
    # Interpretation: -5% means 5% of starting capital was lost at the worst point.
    max_dd_pct = float(max_dd / starting_equity * 100) if starting_equity > 0 else 0.0

    # Trades per day
    if len(trades) >= 2:
        t0 = pd.Timestamp(trades[0]["entry_time"])
        t1 = pd.Timestamp(trades[-1]["exit_time"])
        days_span = max((t1 - t0).days, 1)
        trades_per_day = len(trades) / days_span
    else:
        trades_per_day = float(len(trades))

    # Rolling Sharpe (20 trading days)
    rolling_sharpe_20d = None
    try:
        daily_eq  = equity.resample("D").last().dropna()
        daily_ret = daily_eq.diff().dropna()
        if len(daily_ret) >= 20:
            roll = daily_ret.rolling(20)
            rs   = (roll.mean() / roll.std()) * np.sqrt(252)
            rolling_sharpe_20d = round(float(rs.iloc[-1]), 4)
    except Exception:
        pass

    return {
        "total_pnl_usd":             round(total_pnl, 2),
        "gross_pnl":                 round(cost_summary.get("gross_pnl", 0), 2),
        "total_costs":               round(cost_summary.get("total_costs", 0), 2),
        "net_pnl":                   round(cost_summary.get("net_pnl", total_pnl), 2),
        "sharpe":                    round(sharpe, 4),
        "sortino":                   round(sortino, 4),
        "rolling_sharpe_20d":        rolling_sharpe_20d,
        "max_drawdown_usd":          round(max_dd, 2),
        "max_drawdown_pct":          round(max_dd_pct, 2),
        "win_rate":                  round(win_rate, 4),
        "expectancy_usd":            round(expectancy, 2),
        "profit_factor":             round(float(profit_factor), 4),
        "avg_trade_duration_bars":   round(float(np.mean(hold_bars)), 1),
        "trades_per_day":            round(trades_per_day, 2),
        "consecutive_losses_max":    cost_summary.get("consecutive_losses_max", 0),
        "kill_switch_triggered":     cost_summary.get("kill_switch_triggered", False),
        "daily_halt_count":          cost_summary.get("daily_halt_count", 0),
    }


def _print_metrics_table(symbol: str, metrics: dict) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {symbol} backtest results  (hardened: 1-bar delay, prop-firm rules)")
    print(f"{'='*width}")
    ordered = [
        ("total_pnl_usd",           "Net P&L ($)"),
        ("gross_pnl",               "Gross P&L ($)"),
        ("total_costs",             "Total costs ($)"),
        ("sharpe",                  "Sharpe (annualised)"),
        ("rolling_sharpe_20d",      "Rolling Sharpe 20d"),
        ("sortino",                 "Sortino"),
        ("max_drawdown_usd",        "Max DD ($)"),
        ("max_drawdown_pct",        "Max DD (%)"),
        ("win_rate",                "Win rate"),
        ("expectancy_usd",          "Expectancy / trade ($)"),
        ("profit_factor",           "Profit factor"),
        ("avg_trade_duration_bars", "Avg hold (bars)"),
        ("trades_per_day",          "Trades / day"),
        ("n_trades",                "Total trades"),
        ("n_bars",                  "Bars evaluated"),
        ("consecutive_losses_max",  "Max consec. losses"),
        ("kill_switch_triggered",   "Kill switch triggered"),
        ("daily_halt_count",        "Daily halts"),
    ]
    for key, label in ordered:
        val = metrics.get(key, "—")
        print(f"  {label:35s}: {val}")
    print(f"{'='*width}\n")
