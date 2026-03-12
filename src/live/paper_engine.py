"""
Paper trading engine – CSV stream simulation (hardened, prop-firm rules).

Execution model (identical to backtest):
  Signal generated at bar t close → filled at bar t+1 OPEN.

Cost model, session filter, ATR filter, trend filter, confidence gating,
risk limits, and throttles are loaded from config/universe.yaml and applied
identically to the backtest.

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

SUMMARY_INTERVAL = 50
WARMUP_BARS      = MIN_ROWS + 10

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
    "max_trades_per_day": 6,
    "min_holding_bars": 5,
    "cooldown_bars_after_exit": 5,
    # Confidence gating
    "min_long_confidence": 0.60,
    "min_short_confidence": 0.60,
    # Trend filter
    "trend_filter_enabled": True,
    "trend_filter_ema_period": 200,
    # Risk limits
    "starting_equity": 100_000.0,
    "max_daily_loss_usd": 1_000.0,
    "max_daily_loss_pct": 0.02,
    "max_total_drawdown_usd": 3_000.0,
    "max_total_drawdown_pct": 0.06,
    "max_consecutive_losses": 3,
    "cooldown_bars_after_loss": 10,
    # Position sizing
    "position_sizing_method": "fixed",
    "fixed_contracts": 1,
    "risk_per_trade_usd": 500.0,
    "risk_per_trade_pct": 0.01,
    "atr_stop_multiplier": 1.5,
    "max_contracts": 3,
    "tick_size": 0.25,
}


def _load_cfg(symbol: str) -> tuple[dict, dict]:
    """Return (run_cfg, contract_specs_for_symbol)."""
    if not _UNIVERSE_CFG_PATH.exists():
        return dict(_DEFAULTS), {}
    with open(_UNIVERSE_CFG_PATH) as f:
        u = yaml.safe_load(f) or {}

    cfg = dict(_DEFAULTS)
    cfg.update(u.get("execution", {}))
    cfg.update(u.get("cost_model", {}))
    cfg.update(u.get("filters", {}))
    cfg.update(u.get("throttles", {}))
    cfg.update(u.get("confidence", {}))
    tf = u.get("trend_filter", {})
    if "enabled" in tf:
        cfg["trend_filter_enabled"] = tf["enabled"]
    if "ema_period" in tf:
        cfg["trend_filter_ema_period"] = tf["ema_period"]
    cfg.update(u.get("risk_limits", {}))
    ps = u.get("position_sizing", {})
    if "method" in ps:
        cfg["position_sizing_method"] = ps["method"]
    for k, v in ps.items():
        if k != "method":
            cfg[k] = v
    specs = u.get("contract_specs", {}).get(symbol, {})
    return cfg, specs


def _in_session(ts: pd.Timestamp, start_h: float, end_h: float) -> bool:
    """True if `ts` falls within [start_h, end_h) UTC hours.

    Supports fractional values (e.g. 13.5 = 13:30 UTC).
    """
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
    model, scaler, feature_names, inv_label_map, schema_threshold = _load_artifacts(symbol)
    cfg, specs = _load_cfg(symbol)

    tick_size         = float(specs.get("tick_size", cfg.get("tick_size", 0.25)))
    multiplier        = float(specs.get("multiplier", cfg.get("multiplier", 50)))
    commission_rt     = 2.0 * float(cfg["commission_per_side_usd"])
    slippage_pts      = float(cfg["slippage_ticks_per_side"]) * tick_size
    half_spread_pts   = float(cfg["spread_ticks"]) * 0.5 * tick_size
    friction_per_side = slippage_pts + half_spread_pts

    delay          = int(cfg.get("execution_delay_bars", 1))
    sess_start     = float(cfg.get("session_start_utc_hour", 0))
    sess_end       = float(cfg.get("session_end_utc_hour", 24))
    atr_min        = float(cfg.get("atr_min_ticks", 0))
    atr_max        = float(cfg.get("atr_max_ticks", 1e9))
    blackouts      = cfg.get("news_blackout_windows", [])
    max_tpd        = int(cfg.get("max_trades_per_day", 9999))
    min_hold       = int(cfg.get("min_holding_bars", 1))
    cooldown_cfg   = int(cfg.get("cooldown_bars_after_exit", 0))

    # Confidence threshold: prefer schema value (set during training) over config.
    # The schema threshold is co-selected with the model, so it matches exactly
    # what was used during validation.  Fall back to config if schema has none.
    if schema_threshold > 0.0:
        min_long_conf  = schema_threshold
        min_short_conf = schema_threshold
        log.info(
            "Using confidence threshold from training schema: %.2f "
            "(overrides config min_long/short_confidence)",
            schema_threshold,
        )
    else:
        min_long_conf  = float(cfg.get("min_long_confidence", 0.0))
        min_short_conf = float(cfg.get("min_short_confidence", 0.0))
        log.info(
            "No schema threshold found; using config: "
            "min_long=%.2f  min_short=%.2f",
            min_long_conf, min_short_conf,
        )

    # Trend filter
    trend_filter_on  = bool(cfg.get("trend_filter_enabled", False))
    trend_ema_period = int(cfg.get("trend_filter_ema_period", 200))

    # Risk limits
    starting_equity    = float(cfg.get("starting_equity", 100_000.0))
    max_daily_loss_usd = float(cfg.get("max_daily_loss_usd", 1e15))
    max_daily_loss_pct = float(cfg.get("max_daily_loss_pct", 1.0))
    max_total_dd_usd   = float(cfg.get("max_total_drawdown_usd", 1e15))
    max_total_dd_pct   = float(cfg.get("max_total_drawdown_pct", 1.0))
    max_consec_losses  = int(cfg.get("max_consecutive_losses", 9999))
    loss_cooldown_cfg  = int(cfg.get("cooldown_bars_after_loss", 0))

    # Position sizing
    ps_method      = str(cfg.get("position_sizing_method", "fixed"))
    fixed_contracts = int(cfg.get("fixed_contracts", 1))
    risk_usd       = float(cfg.get("risk_per_trade_usd", 500.0))
    risk_pct       = float(cfg.get("risk_per_trade_pct", 0.01))
    atr_stop_mult  = float(cfg.get("atr_stop_multiplier", 1.5))
    max_contracts  = int(cfg.get("max_contracts", 3))

    if delay == 0:
        log.warning("execution_delay_bars=0 → same-bar fills (leaky mode).")

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading CSV %s …", csv_path)
    raw_df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    raw_df.sort_index(inplace=True)
    log.info("Loaded %d bars from %s to %s", len(raw_df), raw_df.index[0], raw_df.index[-1])

    # ── Pre-compute all features + predictions at startup ─────────────────────
    log.info("Pre-computing features on %d bars …", len(raw_df))
    try:
        all_features = build_features(raw_df, min_rows=1)
    except ValueError as exc:
        log.error("Feature computation failed: %s", exc)
        sys.exit(1)
    log.info("Features ready: %d rows", len(all_features))

    # Pass DataFrame (not .values) to preserve feature_names_in_ → no warning
    X_df         = all_features[feature_names]
    X_scaled_all = scaler.transform(X_df)
    proba_all    = model.predict_proba(X_scaled_all)
    _inv_lookup  = {inv_label_map[str(k)]: k for k in range(len(inv_label_map))}
    _short_col   = _inv_lookup.get(-1, 0)
    _long_col    = _inv_lookup.get(1, 2)
    _edge_all    = proba_all[:, _long_col] - proba_all[:, _short_col]
    # Confidence gap filter: abs(edge) > 0.20 required — matches engine.py
    signal_all   = np.where(_edge_all > 0.20, 1, np.where(_edge_all < -0.20, -1, 0)).astype(np.int8)
    conf_all     = np.max(proba_all, axis=1)

    # Pre-filter confidence coverage diagnostic
    _conf_threshold_diag = min_long_conf  # same for long/short
    if _conf_threshold_diag > 0.0:
        _conf_mask = conf_all >= _conf_threshold_diag
        _sig_mask  = signal_all != 0
        _n_conf    = int((_conf_mask & _sig_mask).sum())
        _n_total   = len(conf_all)
        _pct       = 100.0 * _n_conf / max(_n_total, 1)
        log.info(
            "[%s] Confidence coverage: %d / %d bars (%.1f%%) have "
            "confidence >= %.2f and non-flat signal (before session/ATR/trend filters)",
            symbol, _n_conf, _n_total, _pct, _conf_threshold_diag,
        )
        if _pct < 0.5:
            log.warning(
                "[%s] Very low confidence coverage (%.1f%%) — "
                "model may be under-confident on this dataset. "
                "Consider retraining or lowering the threshold.",
                symbol, _pct,
            )

    # ATR in ticks
    raw_close_aligned = raw_df["close"].reindex(all_features.index)
    atr_ticks_all = (all_features["atr_14"].values * raw_close_aligned.values) / tick_size

    # EMA slope for trend filter (matches engine.py: EMA - EMA.shift(5))
    ema_all = None
    if trend_filter_on:
        _raw_ema_s = raw_close_aligned.ewm(span=trend_ema_period, adjust=False).mean()
        ema_all = (_raw_ema_s - _raw_ema_s.shift(5)).values
        log.info("Trend filter enabled: EMA(%d) slope", trend_ema_period)

    # ATR volatility regime: ATR / rolling_mean(ATR, 100) — skip if < 0.8
    _atr_series = pd.Series(atr_ticks_all, index=all_features.index)
    _atr_regime_all = (_atr_series / _atr_series.rolling(100, min_periods=20).mean()).fillna(1.0).values

    # Regime classifier: trend (1) / chop (0) / low_vol (-1)
    # Uses EMA(20) slope magnitude / ATR_pts and vol_regime_20_60 feature.
    _ema20_close   = raw_close_aligned.ewm(span=20, adjust=False).mean()
    _ema20_slope_a = (_ema20_close - _ema20_close.shift(5)).abs()
    _atr_pts_s     = _atr_series * tick_size
    _trend_str_s   = (_ema20_slope_a / _atr_pts_s.replace(0, np.nan)).fillna(0.0)
    if "vol_regime_20_60" in all_features.columns:
        _vol_reg_s = all_features["vol_regime_20_60"].fillna(1.0)
    else:
        _vol_reg_s = pd.Series(1.0, index=all_features.index)
    _atr_reg_s    = pd.Series(_atr_regime_all, index=all_features.index)
    _is_lv_s      = (_atr_reg_s < 0.8) | (_vol_reg_s < 0.7)
    _is_trend_s   = (~_is_lv_s) & (_trend_str_s >= 0.08)
    _regime_all   = np.where(_is_trend_s, 1, np.where(_is_lv_s, -1, 0)).astype(np.int8)
    _rn_t  = int((_regime_all == 1).sum())
    _rn_c  = int((_regime_all == 0).sum())
    _rn_lv = int((_regime_all == -1).sum())
    log.info(
        "[%s] Regime distribution: trend=%d (%.1f%%) | chop=%d (%.1f%%) | low_vol=%d (%.1f%%)",
        symbol,
        _rn_t,  100 * _rn_t  / max(len(_regime_all), 1),
        _rn_c,  100 * _rn_c  / max(len(_regime_all), 1),
        _rn_lv, 100 * _rn_lv / max(len(_regime_all), 1),
    )

    feat_idx    = all_features.index
    feat_ts_set = set(feat_idx)

    # ── Position sizing helper ─────────────────────────────────────────────────
    def _size(atr_t: float, session_factor: float = 1.0) -> int:
        if ps_method == "fixed_dollar_risk":
            atr_pts  = atr_t * tick_size
            stop_pts = atr_pts * atr_stop_mult
            n_c = int(risk_usd / (stop_pts * multiplier)) if stop_pts > 0 else 1
        elif ps_method == "fixed_pct_risk":
            total_eq = starting_equity + equity
            r_usd    = total_eq * risk_pct
            atr_pts  = atr_t * tick_size
            stop_pts = atr_pts * atr_stop_mult
            n_c = int(r_usd / (stop_pts * multiplier)) if stop_pts > 0 else 1
        else:
            n_c = fixed_contracts
        return max(1, min(int(n_c * session_factor), max_contracts))

    # ── State ──────────────────────────────────────────────────────────────────
    position         = 0
    entry_price      = 0.0
    entry_bar_num    = 0
    entry_contracts  = 1
    equity           = 0.0
    gross_pnl_total  = 0.0
    total_cost_total = 0.0
    trades: list[dict] = []
    bar_count        = 0
    feat_cursor      = 0

    # Throttle state
    cooldown_left     = 0
    daily_trade_count = 0
    last_trade_date   = None
    pending_target    = None
    pending_contracts = 1

    # Risk state
    daily_pnl          = 0.0
    peak_equity        = 0.0
    consecutive_losses = 0
    daily_halt         = False
    kill_switch_active = False
    loss_cooldown_left = 0

    # Sequential pipeline counters (each stage counts signals surviving *in order*)
    flt = {
        "n_total_bars":        0,
        "n_confident_signals": 0,
        "n_after_session":     0,
        "n_after_blackout":    0,
        "n_after_atr":         0,
        "n_after_regime":      0,
        "n_chop_blocked":      0,
        "n_low_vol_blocked":   0,
        "n_after_trend":       0,
        "n_after_risk":        0,
        "n_after_cooldowns":   0,
        "n_entries_queued":    0,
        "n_trend_entries":     0,
    }

    last_signal     = 0
    last_confidence = 0.0

    print(f"\n{'='*60}")
    print(f"  Paper Engine – {symbol}  (SIMULATION ONLY, 1-bar fill delay)")
    print(f"{'='*60}")
    print(f"  Streaming {len(raw_df)} bars …  Ctrl-C to stop")
    log.info(
        "Cost: commission RT=$%.2f | friction/side=%.4f pts | total RT ≈ $%.2f",
        commission_rt, friction_per_side,
        2 * friction_per_side * multiplier + commission_rt,
    )
    if trend_filter_on:
        log.info("Trend filter: EMA(%d)", trend_ema_period)
    log.info(
        "Risk limits: daily_loss=$%.0f (%.1f%%) | total_dd=$%.0f (%.1f%%) | max_consec=%d",
        max_daily_loss_usd, max_daily_loss_pct * 100,
        max_total_dd_usd, max_total_dd_pct * 100,
        max_consec_losses,
    )
    print()

    def _update_risk(net_pnl: float, ts) -> None:
        nonlocal equity, daily_pnl, peak_equity, consecutive_losses
        nonlocal daily_halt, kill_switch_active, loss_cooldown_left

        equity    += net_pnl
        daily_pnl += net_pnl
        peak_equity = max(peak_equity, equity)

        if net_pnl < 0:
            consecutive_losses += 1
            if loss_cooldown_cfg > 0:
                loss_cooldown_left = loss_cooldown_cfg
        else:
            consecutive_losses = 0

        if not daily_halt:
            total_eq_before = starting_equity + equity - net_pnl
            daily_loss = -daily_pnl
            if (daily_loss >= max_daily_loss_usd
                    or (total_eq_before > 0 and daily_loss / total_eq_before >= max_daily_loss_pct)
                    or consecutive_losses >= max_consec_losses):
                daily_halt = True
                log.warning(
                    "[%s] Daily halt triggered — daily_pnl=%.2f consec_losses=%d",
                    ts, daily_pnl, consecutive_losses,
                )

        if not kill_switch_active:
            drawdown = equity - peak_equity
            dd_base  = starting_equity + peak_equity
            if (-drawdown >= max_total_dd_usd
                    or (dd_base > 0 and -drawdown / dd_base >= max_total_dd_pct)):
                kill_switch_active = True
                log.warning(
                    "[%s] Kill switch activated — drawdown=%.2f",
                    ts, drawdown,
                )

    try:
        raw_list = list(raw_df.iterrows())
        n_raw    = len(raw_list)

        for raw_i, (ts, row) in enumerate(raw_list):
            bar_count += 1
            today = ts.date()

            # Daily reset
            if today != last_trade_date:
                daily_trade_count = 0
                last_trade_date   = today
                if daily_halt and not kill_switch_active:
                    daily_halt = False
                    log.info("[%s] Daily halt cleared (new trading day)", ts)
                daily_pnl = 0.0

            has_open = "open" in row and not pd.isna(row["open"])
            open_px  = float(row["open"]) if has_open else float(row["close"])
            close_px = float(row["close"])

            # ── STEP 1: Execute pending order at this bar's open ───────────────
            if pending_target is not None and delay > 0:
                o   = open_px
                n_c = pending_contracts
                if position != 0 and pending_target != position:
                    exit_fill = o - position * friction_per_side
                    raw_pnl   = position * (exit_fill - entry_price) * multiplier * entry_contracts
                    cost      = commission_rt * entry_contracts
                    net_pnl   = raw_pnl - cost
                    gross_pnl_total  += raw_pnl
                    total_cost_total += cost
                    trades.append({
                        "time":      str(ts),
                        "dir":       "L" if position > 0 else "S",
                        "entry":     round(entry_price, 2),
                        "exit":      round(exit_fill, 2),
                        "hold_bars": bar_count - entry_bar_num,
                        "n_contr":   entry_contracts,
                        "gross":     round(raw_pnl, 2),
                        "cost":      round(cost, 2),
                        "pnl":       round(net_pnl, 2),
                        "equity":    round(equity + net_pnl, 2),
                    })
                    log.debug(
                        "[%s] EXIT %s @ %.2f  gross=%.2f cost=%.2f net=%.2f  eq=%.2f",
                        ts, "L" if position > 0 else "S", exit_fill,
                        raw_pnl, cost, net_pnl, equity + net_pnl,
                    )
                    _update_risk(net_pnl, ts)
                    if cooldown_cfg > 0:
                        cooldown_left = cooldown_cfg
                    position = 0

                    if pending_target != 0:
                        entry_price     = o + pending_target * friction_per_side
                        position        = pending_target
                        entry_bar_num   = bar_count
                        entry_contracts = n_c
                        log.debug(
                            "[%s] ENTER %s @ %.2f (reversal, %d contracts)",
                            ts, "L" if position > 0 else "S", entry_price, n_c,
                        )

                elif position == 0 and pending_target != 0:
                    entry_price     = o + pending_target * friction_per_side
                    position        = pending_target
                    entry_bar_num   = bar_count
                    entry_contracts = n_c
                    log.debug(
                        "[%s] ENTER %s @ %.2f (%d contracts)",
                        ts, "L" if position > 0 else "S", entry_price, n_c,
                    )

                pending_target = None

            # ── STEP 2: Get signal + apply 10-filter pipeline ─────────────────
            if ts in feat_ts_set:
                raw_sig    = int(signal_all[feat_cursor])
                confidence = float(conf_all[feat_cursor])
                atr_ticks  = float(atr_ticks_all[feat_cursor])
                ema_val    = float(ema_all[feat_cursor]) if ema_all is not None else float("nan")
                atr_regime = float(_atr_regime_all[feat_cursor])
                regime_val = int(_regime_all[feat_cursor])
                feat_cursor += 1
            else:
                raw_sig    = 0
                confidence = 0.0
                atr_ticks  = 0.0
                ema_val    = float("nan")
                atr_regime = 1.0
                regime_val = 1   # default pass when no feature data

            sig = raw_sig

            # Diagnostics dict for this bar
            filters_passed: list[str] = []
            filters_failed: list[str] = []

            def _check(name: str, passed: bool) -> None:
                (filters_passed if passed else filters_failed).append(name)

            _session_factor = 1.0  # updated in Filter 2; used for position sizing

            # Filter 1: Confidence gating
            conf_ok = True
            if sig == 1 and confidence < min_long_conf:
                conf_ok = False
            elif sig == -1 and confidence < min_short_conf:
                conf_ok = False
            _check(f"confidence({confidence:.2f})", conf_ok)
            if not conf_ok:
                sig = 0

            # ── Sequential pipeline diagnostics after confidence gate ─────────
            flt["n_total_bars"] += 1
            _conf_sig = int(sig)
            if _conf_sig != 0:
                flt["n_confident_signals"] += 1
                if _in_session(ts, sess_start, sess_end):
                    flt["n_after_session"] += 1
                    if not _in_blackout(ts, blackouts):
                        flt["n_after_blackout"] += 1
                        if atr_min <= atr_ticks <= atr_max and atr_regime >= 0.8:
                            flt["n_after_atr"] += 1
                            if regime_val == 1:  # trend regime only
                                flt["n_after_regime"] += 1
                                _tr_ok = True
                                if trend_filter_on and not np.isnan(ema_val):
                                    # ema_val is now slope (EMA - EMA.shift(5))
                                    if (_conf_sig == 1 and ema_val <= 0) or \
                                       (_conf_sig == -1 and ema_val >= 0):
                                        _tr_ok = False
                                if _tr_ok:
                                    flt["n_after_trend"] += 1
                                    _risk_ok = not (
                                        (kill_switch_active or daily_halt) and
                                        (position == 0 or _conf_sig * position < 0)
                                    )
                                    if _risk_ok:
                                        flt["n_after_risk"] += 1
                                        _cd_ok = (
                                            not (loss_cooldown_left > 0 and position == 0) and
                                            not (cooldown_left > 0 and position == 0) and
                                            not (daily_trade_count >= max_tpd and
                                                 _conf_sig != position)
                                        )
                                        if _cd_ok:
                                            flt["n_after_cooldowns"] += 1

            # Filter 2: Session — soft block: 50% size outside session
            sess_ok = _in_session(ts, sess_start, sess_end)
            _session_factor = 1.0 if sess_ok else 0.5
            _check("session", sess_ok)
            # Signal NOT zeroed out — size halved instead (see _size calls below)

            # Filter 3: Blackout
            bo_ok = not _in_blackout(ts, blackouts)
            _check("blackout", bo_ok)
            if not bo_ok:
                sig = 0

            # Filter 4: ATR range check
            atr_ok = (atr_min <= atr_ticks <= atr_max)
            _check(f"atr({atr_ticks:.1f})", atr_ok)
            if not atr_ok:
                sig = 0

            # Filter 4.5: Regime classifier — block chop and low_vol entries
            if sig != 0:
                if regime_val != 1:
                    if regime_val == 0:
                        flt["n_chop_blocked"] += 1
                        _check(f"regime_chop({atr_regime:.2f})", False)
                    else:
                        flt["n_low_vol_blocked"] += 1
                        _check(f"regime_lv({atr_regime:.2f})", False)
                    sig = 0
                else:
                    _check(f"regime_trend({atr_regime:.2f})", True)

            # Filter 5: Trend slope filter (ema_val is EMA slope, not raw EMA)
            # Long allowed when slope > 0; short allowed when slope < 0.
            if trend_filter_on and not np.isnan(ema_val):
                trend_ok = True
                if sig == 1 and ema_val <= 0:
                    trend_ok = False
                elif sig == -1 and ema_val >= 0:
                    trend_ok = False
                _check(f"trend_slope({ema_val:.4f})", trend_ok)
                if not trend_ok:
                    sig = 0

            # Filter 6: Risk gate
            risk_blocked = False
            if kill_switch_active or daily_halt:
                if position == 0:
                    risk_blocked = True
                    sig = 0
                elif sig * position < 0:  # reversal
                    risk_blocked = True
                    sig = 0
            _check("risk_gate", not risk_blocked)

            # Filter 7: Loss cooldown
            lc_ok = True
            if loss_cooldown_left > 0:
                loss_cooldown_left -= 1
                if position == 0:
                    lc_ok = False
                    sig = 0
            _check("loss_cooldown", lc_ok)

            # Filter 8: Exit cooldown
            ec_ok = True
            if cooldown_left > 0:
                cooldown_left -= 1
                if position == 0:
                    ec_ok = False
                    sig = 0
            _check("exit_cooldown", ec_ok)

            # Filter 9: Min hold
            if position != 0 and (bar_count - entry_bar_num) < min_hold:
                sig = position

            # Filter 10: Daily trade cap
            if sig != 0 and sig != position and daily_trade_count >= max_tpd:
                sig = 0 if position == 0 else position

            target = int(sig)
            last_signal     = raw_sig
            last_confidence = confidence

            if delay == 0:
                if target != position:
                    o = open_px
                    if position != 0:
                        exit_fill = o - position * friction_per_side
                        raw_pnl   = position * (exit_fill - entry_price) * multiplier * entry_contracts
                        cost      = commission_rt * entry_contracts
                        net_pnl   = raw_pnl - cost
                        gross_pnl_total  += raw_pnl
                        total_cost_total += cost
                        trades.append({
                            "time": str(ts), "dir": "L" if position > 0 else "S",
                            "entry": round(entry_price, 2), "exit": round(exit_fill, 2),
                            "hold_bars": bar_count - entry_bar_num,
                            "n_contr": entry_contracts,
                            "gross": round(raw_pnl, 2), "cost": round(cost, 2),
                            "pnl": round(net_pnl, 2), "equity": round(equity + net_pnl, 2),
                        })
                        _update_risk(net_pnl, ts)
                        position = 0
                    if target != 0:
                        n_c         = _size(atr_ticks, _session_factor)
                        entry_price     = o + target * friction_per_side
                        position        = target
                        entry_bar_num   = bar_count
                        entry_contracts = n_c
            else:
                if target != position and raw_i < n_raw - 1:
                    pending_target    = target
                    pending_contracts = _size(atr_ticks, _session_factor) if target != 0 else entry_contracts
                    if target != 0:
                        daily_trade_count += 1
                        flt["n_entries_queued"] += 1
                        if regime_val == 1:
                            flt["n_trend_entries"] += 1

            # ── Per-bar diagnostics (debug level) ─────────────────────────────
            if log.isEnabledFor(logging.DEBUG):
                status = f"pos={position:+d}"
                if kill_switch_active:
                    status += " KILL_SWITCH"
                elif daily_halt:
                    status += " DAILY_HALT"
                log.debug(
                    "[%s] raw=%+d conf=%.2f target=%+d | %s | pass=%s fail=%s",
                    ts, raw_sig, confidence, target, status,
                    ",".join(filters_passed) or "none",
                    ",".join(filters_failed) or "none",
                )

            # ── Periodic summary ───────────────────────────────────────────────
            if bar_count % SUMMARY_INTERVAL == 0:
                n_t  = len(trades)
                wins = sum(1 for t in trades if t["pnl"] > 0)
                wr   = wins / n_t if n_t else 0.0
                halt_str = " [KILL_SWITCH]" if kill_switch_active else (" [HALT]" if daily_halt else "")
                print(
                    f"  bar {bar_count:6d} | pos={position:+d} | "
                    f"equity={equity:+.2f} | trades={n_t} | "
                    f"winrate={wr:.1%} | sig={last_signal:+d} "
                    f"conf={last_confidence:.2f}{halt_str}"
                )

            if bar_delay > 0:
                time.sleep(bar_delay)

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    # ── Sequential execution pipeline breakdown ───────────────────────────────
    _nt    = flt.get("n_total_bars",        0)
    _nc    = flt.get("n_confident_signals",  0)
    _ns    = flt.get("n_after_session",      0)
    _nbo   = flt.get("n_after_blackout",     0)
    _natr  = flt.get("n_after_atr",          0)
    _nreg  = flt.get("n_after_regime",       0)
    _nchb  = flt.get("n_chop_blocked",       0)
    _nlvb  = flt.get("n_low_vol_blocked",    0)
    _nte   = flt.get("n_trend_entries",      0)
    _ntr   = flt.get("n_after_trend",        0)
    _nrisk = flt.get("n_after_risk",         0)
    _ncd   = flt.get("n_after_cooldowns",    0)
    _nq    = flt.get("n_entries_queued",     0)
    _nex   = len(trades)

    def _pp(n, d):
        return f"{100*n/d:.0f}%" if d > 0 else "n/a"

    log.info("[%s] Execution pipeline (%d total bars):", symbol, _nt)
    log.info("[%s]   confident signals : %5d        (%s of bars)", symbol, _nc, _pp(_nc, _nt))
    log.info("[%s]   → after session   : %5d → %5d  (%s pass)", symbol, _nc, _ns, _pp(_ns, _nc))
    log.info("[%s]   → after blackout  : %5d → %5d  (%s pass)", symbol, _ns, _nbo, _pp(_nbo, _ns))
    log.info("[%s]   → after ATR       : %5d → %5d  (%s pass)", symbol, _nbo, _natr, _pp(_natr, _nbo))
    log.info(
        "[%s]   → after regime    : %5d → %5d  (%s pass)  [chop_blk=%d  lv_blk=%d  trend_entries=%d]",
        symbol, _natr, _nreg, _pp(_nreg, _natr), _nchb, _nlvb, _nte,
    )
    log.info("[%s]   → after trend     : %5d → %5d  (%s pass)", symbol, _nreg, _ntr, _pp(_ntr, _nreg))
    log.info("[%s]   → after risk/halt : %5d → %5d  (%s pass)", symbol, _ntr, _nrisk, _pp(_nrisk, _ntr))
    log.info("[%s]   → after cooldowns : %5d → %5d  (%s pass)", symbol, _nrisk, _ncd, _pp(_ncd, _nrisk))
    log.info("[%s]   → entries queued  : %5d → %5d  (%s pass)", symbol, _ncd, _nq, _pp(_nq, _ncd))
    log.info("[%s]   → trades executed : %5d → %5d  (%s of queued)", symbol, _nq, _nex, _pp(_nex, _nq))
    log.info(
        "[%s]   OVERALL: conf→executed = %d → %d  (%s end-to-end pass-through)",
        symbol, _nc, _nex, _pp(_nex, _nc),
    )

    _overall_pct = 100.0 * _nex / max(_nc, 1) if _nc > 0 else 0.0

    # ── Pipeline health check (<5% pass-through) ──────────────────────────────
    if _nc > 0 and _nex < _nc * 0.05:
        log.warning(
            "[%s] WARNING PIPELINE HEALTH\n"
            "  %d confident signals\n"
            "  %d in-session\n"
            "  %d after ATR\n"
            "  %d after regime  (chop_blk=%d  lv_blk=%d)\n"
            "  %d after trend\n"
            "  %d queued  (trend_entries=%d)\n"
            "  %d executed\n"
            "  Overall pass-through = %.1f%%",
            symbol,
            _nc, _ns, _natr, _nreg, _nchb, _nlvb, _ntr, _nq, _nte, _nex, _overall_pct,
        )

    # ── CRITICAL pipeline blockage check (<10%) ───────────────────────────────
    if _nc > 0 and _nex < _nc * 0.10:
        _stages = [
            ("session",   _nc,    _ns),
            ("blackout",  _ns,    _nbo),
            ("ATR",       _nbo,   _natr),
            ("regime",    _natr,  _nreg),
            ("trend",     _nreg,  _ntr),
            ("risk/halt", _ntr,   _nrisk),
            ("cooldown",  _nrisk, _ncd),
            ("queue",     _ncd,   _nq),
        ]
        _bottleneck = max(_stages, key=lambda s: s[1] - s[2])
        _bn, _bi, _bo = _bottleneck
        log.critical(
            "[%s] CRITICAL PIPELINE BLOCKAGE\n"
            "  Top bottleneck: %s filter (blocked %d signals)\n"
            "  conf→executed = %d → %d (%.0f%% end-to-end)\n"
            "  %s filter: %d → %d (%.0f%% pass)",
            symbol,
            _bn, _bi - _bo,
            _nc, _nex, _overall_pct,
            _bn, _bi, _bo, 100.0 * _bo / max(_bi, 1),
        )

    # ── Pass-through warnings ─────────────────────────────────────────────────
    if _ns > 0 and _nq < _ns * 0.20:
        log.warning(
            "[%s] LOW PIPELINE YIELD: only %d queued entries from %d in-session "
            "confident signals (%.0f%% < 20%% threshold). "
            "Consider loosening: cooldown_bars_after_exit, cooldown_bars_after_loss, "
            "min_holding_bars, max_trades_per_day, or max_consecutive_losses.",
            symbol, _nq, _ns, 100*_nq/max(_ns, 1),
        )
    if _nq > 0 and _nex < _nq * 0.10:
        log.warning(
            "[%s] EXECUTION UNDERPERFORMANCE: only %d trades executed from %d queued "
            "entries (%.0f%% < 10%% threshold). "
            "The delay=1 bar fill or position logic may be losing orders.",
            symbol, _nex, _nq, 100*_nex/max(_nq, 1),
        )

    _print_summary(symbol, trades, equity, gross_pnl_total, total_cost_total, bar_count)


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

    feature_names: list[str]      = schema["feature_names"]
    inv_label_map: dict[str, int] = {k: int(v) for k, v in schema["inv_label_map"].items()}
    conf_threshold: float         = float(schema.get("selected_conf_threshold", 0.0))

    log.info(
        "Loaded model (%d features) and scaler for %s  |  "
        "select_by=%-8s  threshold=%.2f  "
        "val_trades=%s  val_coverage=%.1f%%  val_dir_acc=%.1f%%  val_PF=%.3f",
        len(feature_names), symbol,
        schema.get("select_by", "unknown"),
        conf_threshold,
        schema.get("val_confident_trades", "?"),
        float(schema.get("val_trade_coverage_pct", 0)),
        float(schema.get("val_dir_accuracy", 0)) * 100,
        float(schema.get("val_profit_factor", 0)),
    )
    return model, scaler, feature_names, inv_label_map, conf_threshold
