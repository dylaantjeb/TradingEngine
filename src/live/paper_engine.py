"""
Paper trading engine – CSV stream simulation (hardened).

Execution model (identical to backtest):
  Signal generated at bar t close → filled at bar t+1 OPEN.

Cost model, session filter, ATR filter, and throttles are loaded from
config/universe.yaml and applied identically to the backtest.

PAPER TRADING ONLY – no live order placement.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.builder import build_features, MIN_ROWS

log = logging.getLogger(__name__)

SUMMARY_INTERVAL     = 50
CONFIDENCE_THRESHOLD = 0.50
WARMUP_BARS          = MIN_ROWS + 10

_UNIVERSE_CFG_PATH = Path("config/universe.yaml")

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


def _load_cfg(symbol: str) -> tuple[dict, dict]:
    """Return (run_cfg, contract_specs)."""
    if not _UNIVERSE_CFG_PATH.exists():
        return dict(_DEFAULTS), {}
    with open(_UNIVERSE_CFG_PATH) as f:
        u = yaml.safe_load(f) or {}

    cfg = dict(_DEFAULTS)
    cfg.update(u.get("execution", {}))
    cfg.update(u.get("cost_model", {}))
    cfg.update(u.get("filters", {}))
    cfg.update(u.get("throttles", {}))
    specs = u.get("contract_specs", {}).get(symbol, {})
    return cfg, specs


def _in_session(ts: pd.Timestamp, start_h: int, end_h: int) -> bool:
    if start_h == 0 and end_h >= 24:
        return True
    h = ts.hour + ts.minute / 60.0
    return start_h <= h < end_h


def _in_blackout(ts: pd.Timestamp, windows: list) -> bool:
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


def run_paper(symbol: str, csv_path: Path, bar_delay: float = 0.0) -> None:
    """
    Stream `csv_path` bar-by-bar and simulate paper trading with 1-bar fill delay.

    Parameters
    ----------
    symbol    : Symbol name (used to load artifacts + contract specs).
    csv_path  : Raw OHLCV CSV from `fetch` (must have open column).
    bar_delay : Optional sleep between bars in seconds (0 = max speed).
    """
    model, scaler, feature_names, inv_label_map = _load_artifacts(symbol)

    cfg, specs = _load_cfg(symbol)

    tick_size   = float(specs.get("tick_size", 0.25))
    multiplier  = float(specs.get("multiplier", 50))
    commission_rt     = 2.0 * float(cfg["commission_per_side_usd"])
    slippage_pts      = float(cfg["slippage_ticks_per_side"]) * tick_size
    half_spread_pts   = float(cfg["spread_ticks"]) * 0.5 * tick_size
    friction_per_side = slippage_pts + half_spread_pts

    delay        = int(cfg.get("execution_delay_bars", 1))
    sess_start   = int(cfg.get("session_start_utc_hour", 0))
    sess_end     = int(cfg.get("session_end_utc_hour", 24))
    atr_min      = float(cfg.get("atr_min_ticks", 0))
    atr_max      = float(cfg.get("atr_max_ticks", 1e9))
    blackouts    = cfg.get("news_blackout_windows", [])
    max_tpd      = int(cfg.get("max_trades_per_day", 9999))
    min_hold     = int(cfg.get("min_holding_bars", 1))
    cooldown_cfg = int(cfg.get("cooldown_bars_after_exit", 0))

    if delay == 0:
        log.warning("execution_delay_bars=0 → same-bar fills (leaky mode).")

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading CSV %s …", csv_path)
    raw_df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    raw_df.sort_index(inplace=True)
    log.info(
        "Loaded %d bars from %s to %s",
        len(raw_df), raw_df.index[0], raw_df.index[-1],
    )

    # ── Pre-compute all features + predictions at startup ─────────────────────
    log.info("Pre-computing features on %d bars …", len(raw_df))
    try:
        all_features = build_features(raw_df, min_rows=1)
    except ValueError as exc:
        log.error("Feature computation failed: %s", exc)
        sys.exit(1)

    log.info("Features ready: %d rows", len(all_features))

    X_all        = all_features[feature_names].values
    X_scaled_all = scaler.transform(X_all)
    proba_all    = model.predict_proba(X_scaled_all)
    pred_enc_all = np.argmax(proba_all, axis=1)
    conf_all     = np.max(proba_all, axis=1)
    signal_all   = np.array([inv_label_map[str(e)] for e in pred_enc_all])

    # Confidence filter
    signal_all[conf_all < CONFIDENCE_THRESHOLD] = 0

    # ATR in ticks (for filter)
    atr_14_all = all_features["atr_14"].values          # atr_pts / close
    close_vals = all_features.get("close", pd.Series(dtype=float))
    # We need close from raw_df aligned to all_features index
    raw_close_aligned = raw_df["close"].reindex(all_features.index)
    atr_ticks_all = (atr_14_all * raw_close_aligned.values) / tick_size

    feat_idx    = all_features.index
    feat_ts_set = set(feat_idx)

    # Build a position-lookup from feat_idx into raw_df index
    raw_idx_pos = {ts: i for i, ts in enumerate(raw_df.index)}

    # ── State ──────────────────────────────────────────────────────────────────
    position         = 0
    entry_price      = 0.0
    entry_bar_num    = 0
    equity           = 0.0
    gross_pnl_total  = 0.0
    total_cost       = 0.0
    trades: list[dict] = []
    bar_count        = 0
    feat_cursor      = 0

    # Throttle state
    cooldown_left     = 0
    daily_trade_count = 0
    last_trade_date   = None
    pending_target    = None   # signal queued for next-bar fill

    last_signal     = 0
    last_confidence = 0.0

    print(f"\n{'='*60}")
    print(f"  Paper Engine – {symbol}  (SIMULATION ONLY, 1-bar fill delay)")
    print(f"{'='*60}")
    print(f"  Streaming {len(raw_df)} bars …  Ctrl-C to stop")
    log.info(
        "Cost: commission RT=$%.2f | friction/side=%.4f pts | "
        "total RT cost ≈ $%.2f",
        commission_rt, friction_per_side,
        2 * friction_per_side * multiplier + commission_rt,
    )
    print()

    try:
        raw_list = list(raw_df.iterrows())
        n_raw    = len(raw_list)

        for raw_i, (ts, row) in enumerate(raw_list):
            bar_count += 1
            today = ts.date()

            if today != last_trade_date:
                daily_trade_count = 0
                last_trade_date   = today

            has_open = "open" in row and not pd.isna(row["open"])
            open_px  = float(row["open"]) if has_open else float(row["close"])
            close_px = float(row["close"])

            # ── STEP 1: Execute pending order at this bar's open ───────────────
            if pending_target is not None and delay > 0:
                o = open_px
                if position != 0 and pending_target != position:
                    exit_fill = o - position * friction_per_side
                    raw_pnl   = position * (exit_fill - entry_price) * multiplier
                    cost      = commission_rt
                    net_pnl   = raw_pnl - cost
                    equity   += net_pnl
                    gross_pnl_total += raw_pnl
                    total_cost      += cost
                    trades.append({
                        "time":   str(ts),
                        "dir":    "L" if position > 0 else "S",
                        "entry":  round(entry_price, 2),
                        "exit":   round(exit_fill, 2),
                        "hold_bars": bar_count - entry_bar_num,
                        "gross":  round(raw_pnl, 2),
                        "cost":   round(cost, 2),
                        "pnl":    round(net_pnl, 2),
                        "equity": round(equity, 2),
                    })
                    log.debug(
                        "[%s] EXIT %s @ %.2f  gross=%.2f cost=%.2f net=%.2f  eq=%.2f",
                        ts, "L" if position > 0 else "S", exit_fill,
                        raw_pnl, cost, net_pnl, equity,
                    )
                    position    = 0
                    if cooldown_cfg > 0:
                        cooldown_left = cooldown_cfg

                    if pending_target != 0:
                        entry_price   = o + pending_target * friction_per_side
                        position      = pending_target
                        entry_bar_num = bar_count
                        log.debug(
                            "[%s] ENTER %s @ %.2f (reversal)",
                            ts, "L" if position > 0 else "S", entry_price,
                        )

                elif position == 0 and pending_target != 0:
                    entry_price   = o + pending_target * friction_per_side
                    position      = pending_target
                    entry_bar_num = bar_count
                    log.debug(
                        "[%s] ENTER %s @ %.2f",
                        ts, "L" if position > 0 else "S", entry_price,
                    )

                pending_target = None

            # ── STEP 2: Get signal for this bar (execute at next bar) ──────────
            if ts in feat_ts_set:
                raw_sig    = int(signal_all[feat_cursor])
                confidence = float(conf_all[feat_cursor])
                atr_ticks  = float(atr_ticks_all[feat_cursor])
                feat_cursor += 1
            else:
                raw_sig    = 0
                confidence = 0.0
                atr_ticks  = 0.0

            sig = raw_sig

            # Filters
            if not _in_session(ts, sess_start, sess_end):
                sig = 0
            if _in_blackout(ts, blackouts):
                sig = 0
            if not (atr_min <= atr_ticks <= atr_max):
                sig = 0
            if cooldown_left > 0:
                cooldown_left -= 1
                if position == 0:
                    sig = 0
            if position != 0 and (bar_count - entry_bar_num) < min_hold:
                sig = position   # force hold minimum
            if sig != 0 and sig != position and daily_trade_count >= max_tpd:
                sig = 0 if position == 0 else position  # block entry/reversal

            target = int(sig)
            last_signal     = raw_sig
            last_confidence = confidence

            if delay == 0:
                # Same-bar fill (leaky, debug only)
                if target != position:
                    o = open_px
                    if position != 0:
                        exit_fill = o - position * friction_per_side
                        raw_pnl   = position * (exit_fill - entry_price) * multiplier
                        cost      = commission_rt
                        net_pnl   = raw_pnl - cost
                        equity   += net_pnl
                        gross_pnl_total += raw_pnl
                        total_cost      += cost
                        trades.append({
                            "time": str(ts), "dir": "L" if position > 0 else "S",
                            "entry": round(entry_price, 2), "exit": round(exit_fill, 2),
                            "hold_bars": bar_count - entry_bar_num,
                            "gross": round(raw_pnl, 2), "cost": round(cost, 2),
                            "pnl": round(net_pnl, 2), "equity": round(equity, 2),
                        })
                        position = 0
                    if target != 0:
                        entry_price   = o + target * friction_per_side
                        position      = target
                        entry_bar_num = bar_count
            else:
                # Queue for next bar
                if target != position and raw_i < n_raw - 1:
                    pending_target = target
                    if target != 0:   # count entries AND reversals
                        daily_trade_count += 1

            # ── Periodic summary ───────────────────────────────────────────────
            if bar_count % SUMMARY_INTERVAL == 0:
                n_t  = len(trades)
                wins = sum(1 for t in trades if t["pnl"] > 0)
                wr   = wins / n_t if n_t else 0.0
                print(
                    f"  bar {bar_count:6d} | pos={position:+d} | "
                    f"equity={equity:+.2f} | trades={n_t} | "
                    f"winrate={wr:.1%} | last_signal={last_signal:+d} "
                    f"conf={last_confidence:.2f}"
                )

            if bar_delay > 0:
                time.sleep(bar_delay)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    _print_summary(symbol, trades, equity, gross_pnl_total, total_cost, bar_count)


def _print_summary(
    symbol: str, trades: list[dict], equity: float,
    gross_pnl: float, total_cost: float, bars: int,
) -> None:
    n      = len(trades)
    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    wr     = len(wins) / n if n else 0
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    holds    = [t.get("hold_bars", 0) for t in trades]
    avg_hold = float(np.mean(holds)) if holds else 0.0

    print(f"\n{'='*60}")
    print(f"  PAPER SESSION COMPLETE – {symbol}")
    print(f"{'='*60}")
    print(f"  Total bars           : {bars}")
    print(f"  Total trades         : {n}")
    print(f"  Gross P&L            : ${gross_pnl:+.2f}")
    print(f"  Total costs          : ${total_cost:+.2f}")
    print(f"  Net P&L              : ${equity:+.2f}")
    print(f"  Win rate             : {wr:.1%}")
    print(f"  Avg win              : ${avg_win:+.2f}")
    print(f"  Avg loss             : ${avg_loss:+.2f}")
    print(f"  Avg hold (bars)      : {avg_hold:.1f}")
    if losses:
        pf = abs(sum(wins)) / (abs(sum(losses)) + 1e-9)
        print(f"  Profit factor        : {pf:.2f}")
    print(f"{'='*60}\n")


def _load_artifacts(symbol: str):
    try:
        import joblib
    except ImportError:
        log.error("joblib not installed:  pip install joblib")
        sys.exit(1)

    model_path  = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    scaler_path = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    schema_path = Path(f"artifacts/schema/{symbol}_features.json")

    for p in (model_path, scaler_path, schema_path):
        if not p.exists():
            log.error(
                "Artifact not found: %s\n  Run:  python -m src.cli train --symbol %s",
                p, symbol,
            )
            sys.exit(1)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(schema_path) as f:
        schema = json.load(f)

    feature_names: list[str]       = schema["feature_names"]
    inv_label_map: dict[str, int]  = {k: int(v) for k, v in schema["inv_label_map"].items()}

    log.info("Loaded model (%s features) and scaler for %s", len(feature_names), symbol)
    return model, scaler, feature_names, inv_label_map
