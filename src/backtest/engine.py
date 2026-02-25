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

  net_pnl = direction * (exit_fill - entry_fill) * multiplier - commission

Throttles applied at signal time
──────────────────────────────────
  • max_trades_per_day   : hard ceiling on new entries per calendar day
  • min_holding_bars     : cannot exit before holding N bars
  • cooldown_bars_after_exit : bars to wait after exit before re-entering
  • session filter (UTC hours)
  • ATR filter (ticks)

Report saved to:  artifacts/reports/<SYM>_backtest.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_UNIVERSE_CFG_PATH = Path("config/universe.yaml")

# ── Default hardening parameters (overridden by universe.yaml) ────────────────
_DEFAULTS = {
    "execution_delay_bars": 1,
    "commission_per_side_usd": 1.50,
    "slippage_ticks_per_side": 0.5,
    "spread_ticks": 1.0,
    "session_start_utc_hour": 9,
    "session_end_utc_hour": 22,
    "atr_min_ticks": 4,
    "atr_max_ticks": 200,
    "news_blackout_windows": [],
    "max_trades_per_day": 20,
    "min_holding_bars": 3,
    "cooldown_bars_after_exit": 2,
}


def _load_universe_cfg() -> dict:
    if not _UNIVERSE_CFG_PATH.exists():
        return {}
    with open(_UNIVERSE_CFG_PATH) as f:
        return yaml.safe_load(f) or {}


def _build_run_cfg(universe_cfg: dict, overrides: dict) -> dict:
    """Merge universe.yaml sections + CLI overrides into a flat config dict."""
    cfg = dict(_DEFAULTS)
    # Merge YAML sections
    cfg.update(universe_cfg.get("execution", {}))
    cfg.update(universe_cfg.get("cost_model", {}))
    cfg.update(universe_cfg.get("filters", {}))
    cfg.update(universe_cfg.get("throttles", {}))
    # CLI overrides win
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


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
    tick_size   = float(specs.get("tick_size", 0.25))
    tick_value  = float(specs.get("tick_value", 12.50))
    multiplier  = float(specs.get("multiplier", 50))

    # ── Derived cost params ─────────────────────────────────────────────────────
    commission_rt     = 2.0 * float(cfg["commission_per_side_usd"])
    slippage_pts      = float(cfg["slippage_ticks_per_side"]) * tick_size
    half_spread_pts   = float(cfg["spread_ticks"]) * 0.5 * tick_size
    friction_per_side = slippage_pts + half_spread_pts   # adverse pts on each fill

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

    # ── Generate predictions ────────────────────────────────────────────────────
    X = features[feature_names].copy()
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)
    pred_encoded = np.argmax(proba, axis=1)
    lookup = np.array([inv_label_map[str(k)] for k in range(len(inv_label_map))], dtype=np.int8)
    signal = lookup[pred_encoded]

    # ── Load raw prices (need 'open' for next-bar fills) ───────────────────────
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

    sig_series   = pd.Series(signal, index=features.index).reindex(common_idx)
    open_prices  = prices["open"].reindex(common_idx) if "open" in prices.columns \
                   else prices["close"].reindex(common_idx)
    close_prices = prices["close"].reindex(common_idx)

    # ATR in ticks (for filter): atr_14 feature = atr_pts / close
    atr_ticks_series = (
        features["atr_14"].reindex(common_idx) * close_prices / tick_size
    ).fillna(0)

    # ── Simulate trades ─────────────────────────────────────────────────────────
    trades, equity_curve, cost_summary = _simulate_trades(
        signals        = sig_series,
        open_prices    = open_prices,
        atr_ticks      = atr_ticks_series,
        cfg            = cfg,
        friction_pts   = friction_per_side,
        commission_rt  = commission_rt,
        multiplier     = multiplier,
    )

    # ── Metrics ─────────────────────────────────────────────────────────────────
    metrics = _compute_metrics(equity_curve, trades, cost_summary)
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
) -> tuple[list[dict], pd.Series, dict]:
    """
    Bar-by-bar simulation with 1-bar execution delay, full cost model,
    session/ATR filters, and trade-frequency throttles.

    Signal at bar i → pending order → filled at bar i+1 open.
    """
    delay         = int(cfg.get("execution_delay_bars", 1))
    sess_start    = int(cfg.get("session_start_utc_hour", 0))
    sess_end      = int(cfg.get("session_end_utc_hour", 24))
    atr_min       = float(cfg.get("atr_min_ticks", 0))
    atr_max       = float(cfg.get("atr_max_ticks", 1e9))
    blackouts     = cfg.get("news_blackout_windows", [])
    max_tpd       = int(cfg.get("max_trades_per_day", 9999))
    min_hold      = int(cfg.get("min_holding_bars", 1))
    cooldown_bars = int(cfg.get("cooldown_bars_after_exit", 0))

    sig_arr   = signals.values.astype(np.int8)
    open_arr  = open_prices.reindex(signals.index).values.astype(np.float64)
    atr_arr   = atr_ticks.reindex(signals.index).values.astype(np.float64)
    ts_arr    = signals.index
    n         = len(sig_arr)

    equity_arr = np.zeros(n, dtype=np.float64)
    trades: list[dict] = []

    position    = 0        # current position: -1, 0, +1
    entry_price = 0.0
    entry_idx   = 0
    equity      = 0.0
    gross_pnl   = 0.0
    total_cost  = 0.0

    # Throttle state
    cooldown_left     = 0
    daily_trade_count = 0
    last_trade_date   = None
    pending_target    = None   # None=no pending, 0=exit pending, ±1=entry pending

    for i in range(n):
        ts      = ts_arr[i]
        today   = ts.date()

        # ── Reset daily counter ─────────────────────────────────────────────────
        if today != last_trade_date:
            daily_trade_count = 0
            last_trade_date   = today

        # ── STEP 1: Execute pending order from previous bar at open[i] ──────────
        if pending_target is not None and delay > 0:
            o = open_arr[i]
            if not np.isnan(o):
                if position != 0 and pending_target != position:
                    # Exit (and possibly reverse)
                    exit_fill = o - position * friction_pts
                    raw_pnl   = position * (exit_fill - entry_price) * multiplier
                    cost      = commission_rt
                    net_pnl   = raw_pnl - cost
                    equity   += net_pnl
                    gross_pnl += raw_pnl
                    total_cost += cost
                    trades.append({
                        "entry_time":    str(ts_arr[entry_idx]),
                        "exit_time":     str(ts),
                        "direction":     position,
                        "entry_price":   round(entry_price, 4),
                        "exit_price":    round(exit_fill, 4),
                        "hold_bars":     i - entry_idx,
                        "gross_pnl":     round(raw_pnl, 2),
                        "cost":          round(cost, 2),
                        "net_pnl":       round(net_pnl, 2),
                        "equity":        round(equity, 2),
                    })
                    if cooldown_bars > 0:
                        cooldown_left = cooldown_bars
                    position = 0

                    if pending_target != 0:
                        # Immediate reversal into new position
                        entry_price = o + pending_target * friction_pts
                        position    = pending_target
                        entry_idx   = i

                elif position == 0 and pending_target != 0:
                    entry_price = o + pending_target * friction_pts
                    position    = pending_target
                    entry_idx   = i

            pending_target = None

        # For delay=0 (same-bar), we execute the signal immediately below
        # (pending_target is set and then checked in next iteration, but
        #  for delay=0 we skip the pending mechanism and fill directly)

        # ── STEP 2: Generate signal at bar i ────────────────────────────────────
        o   = open_arr[i]
        sig = int(sig_arr[i])
        if np.isnan(o):
            equity_arr[i] = equity
            continue

        # Session filter
        if not _in_session(ts, sess_start, sess_end):
            sig = 0

        # News blackout
        if _in_blackout(ts, blackouts):
            sig = 0

        # ATR filter
        atr_t = atr_arr[i]
        if not (atr_min <= atr_t <= atr_max):
            sig = 0

        # Cooldown after exit
        if cooldown_left > 0:
            cooldown_left -= 1
            if position == 0:
                sig = 0

        # Minimum holding period – force hold, don't change signal
        if position != 0 and (i - entry_idx) < min_hold:
            sig = position   # stay in current direction

        # Daily trade cap: limits new position initiations (entries + reversals).
        # A reversal (long→short or short→long) counts as one new entry.
        # A pure exit (→flat) does NOT count against the cap.
        if sig != 0 and sig != position and daily_trade_count >= max_tpd:
            if position == 0:
                sig = 0           # block new entry from flat
            else:
                sig = position    # block reversal: force hold

        # Determine target position
        target = int(sig)   # desired state: -1, 0, or +1

        if delay == 0:
            # Same-bar fill (leaky mode – only for debug)
            if target != position:
                if position != 0:
                    exit_fill = o - position * friction_pts
                    raw_pnl   = position * (exit_fill - entry_price) * multiplier
                    cost      = commission_rt
                    net_pnl   = raw_pnl - cost
                    equity   += net_pnl
                    gross_pnl += raw_pnl
                    total_cost += cost
                    trades.append({
                        "entry_time": str(ts_arr[entry_idx]),
                        "exit_time":  str(ts),
                        "direction":  position,
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(exit_fill, 4),
                        "hold_bars":   i - entry_idx,
                        "gross_pnl":  round(raw_pnl, 2),
                        "cost":       round(cost, 2),
                        "net_pnl":    round(net_pnl, 2),
                        "equity":     round(equity, 2),
                    })
                    position = 0
                if target != 0:
                    entry_price = o + target * friction_pts
                    position    = target
                    entry_idx   = i
        else:
            # Normal delayed execution: queue order for next bar
            if target != position and i < n - 1:
                pending_target = target
                # Count both entries (flat→dir) and reversals (dir→-dir)
                if target != 0:
                    daily_trade_count += 1

        equity_arr[i] = equity

    equity_series = pd.Series(equity_arr, index=signals.index, name="equity")
    cost_summary  = {"gross_pnl": gross_pnl, "total_costs": total_cost,
                     "net_pnl": gross_pnl - total_cost}
    return trades, equity_series, cost_summary


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _compute_metrics(
    equity: pd.Series, trades: list[dict], cost_summary: dict
) -> dict[str, Any]:
    if len(trades) == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0,
            "win_rate": 0.0, "expectancy_usd": 0.0, "total_pnl_usd": 0.0,
            "profit_factor": 0.0, "gross_pnl": 0.0, "total_costs": 0.0,
            "net_pnl": 0.0, "trades_per_day": 0.0,
            "avg_trade_duration_bars": 0.0, "rolling_sharpe_20d": None,
        }

    net_pnls  = [t["net_pnl"]  for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    hold_bars  = [t["hold_bars"] for t in trades]
    total_pnl  = sum(net_pnls)
    wins   = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
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
    max_dd_pct = float((max_dd / (cummax.max() + 1e-9)) * 100) if cummax.max() > 0 else 0.0

    # Trades per day
    if len(trades) >= 2:
        t0 = pd.Timestamp(trades[0]["entry_time"])
        t1 = pd.Timestamp(trades[-1]["exit_time"])
        days_span = max((t1 - t0).days, 1)
        trades_per_day = len(trades) / days_span
    else:
        trades_per_day = float(len(trades))

    # Rolling Sharpe (20 trading days on daily equity returns)
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
    }


def _print_metrics_table(symbol: str, metrics: dict) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {symbol} backtest results  (hardened: 1-bar delay)")
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
    ]
    for key, label in ordered:
        val = metrics.get(key, "—")
        print(f"  {label:35s}: {val}")
    print(f"{'='*width}\n")
