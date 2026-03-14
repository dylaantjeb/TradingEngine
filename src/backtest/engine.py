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
  5.  Trend filter        : close vs EMA(ema_period) — blocks counter-trend trades
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
    "trend_slope_min_atr_frac": 0.0,   # min |slope|/ATR_pts gate (0 = disabled)
    # Session blocks (empty = fall back to session_start/end single window)
    "session_blocks": [],
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
    if "min_slope_atr_frac" in tf:
        cfg["trend_slope_min_atr_frac"] = tf["min_slope_atr_frac"]
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
    cfg_overrides: dict | None = None,
    save_report: bool = True,
    use_rules_signal: bool = False,
) -> dict[str, Any]:
    """
    Load model + features for `symbol`, run hardened backtest, persist report.
    Returns metrics dict.

    Parameters
    ----------
    cfg_overrides     : Dict of execution-layer overrides applied after _build_run_cfg.
                        Use this to inject profile-specific params without modifying
                        universe.yaml.
    save_report       : If False, skip writing the JSON report to disk.
    use_rules_signal  : If True, use EMA momentum rules as the PRIMARY signal.
                        XGBoost is then used only as a confidence VETO — it can
                        block entries but cannot generate them.
                        If False (default), the existing ML-primary path is used.
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
    if cfg_overrides:
        cfg = dict(cfg)
        cfg.update(cfg_overrides)

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

    feature_names: list[str]      = schema["feature_names"]
    inv_label_map: dict[str, int] = {k: int(v) for k, v in schema["inv_label_map"].items()}

    # Load threshold selected during training and override the universe.yaml
    # confidence gates, so backtest uses exactly the same threshold as inference.
    saved_threshold = float(schema.get("selected_conf_threshold", 0.0))
    if saved_threshold > 0.0:
        cfg["min_long_confidence"]  = saved_threshold
        cfg["min_short_confidence"] = saved_threshold

    # Log all schema metadata so user can see what was saved during training
    log.info(
        "[%s] Schema  select_by=%-8s  threshold=%.2f  "
        "val_trades=%s  val_coverage=%.1f%%  val_dir_acc=%.1f%%  val_PF=%.3f",
        symbol,
        schema.get("select_by", "unknown"),
        saved_threshold,
        schema.get("val_confident_trades", "?"),
        float(schema.get("val_trade_coverage_pct", 0)),
        float(schema.get("val_dir_accuracy", 0)) * 100,
        float(schema.get("val_profit_factor", 0)),
    )

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
    _inv_lookup = {inv_label_map[str(k)]: k for k in range(len(inv_label_map))}
    _short_col = _inv_lookup.get(-1, 0)
    _long_col  = _inv_lookup.get(1, 2)
    _edge = proba[:, _long_col] - proba[:, _short_col]
    # Confidence gap filter: abs(prob_up - prob_down) > 0.20 required for a trade.
    # Raises the conviction bar beyond simple argmax — weak predictions stay flat.
    signal = np.where(_edge > 0.20, 1, np.where(_edge < -0.20, -1, 0)).astype(np.int8)
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

    ml_sig_series = pd.Series(signal, index=features.index).reindex(common_idx)
    conf_series   = pd.Series(confidence, index=features.index).reindex(common_idx)
    open_prices   = prices["open"].reindex(common_idx) if "open" in prices.columns \
                    else prices["close"].reindex(common_idx)
    close_prices  = prices["close"].reindex(common_idx)

    # ── Dual-confirmation signal path (rules-first, ML-veto) ─────────────────
    # When use_rules_signal=True:
    #   1. Compute EMA momentum rules signal (PRIMARY).
    #   2. Pass it only if ML confidence >= threshold AND ML direction agrees.
    #   3. ML cannot generate entries — only block them.
    if use_rules_signal:
        from src.strategy.rules_signal import generate_rules_signal
        _prod_cfg = cfg_overrides or {}
        _ema_p    = int(_prod_cfg.get("signal", {}).get("ema_period", 20)
                        if isinstance(_prod_cfg.get("signal"), dict)
                        else cfg.get("signal_ema_period", 20))
        _slb      = int(_prod_cfg.get("signal", {}).get("slope_lookback_bars", 5)
                        if isinstance(_prod_cfg.get("signal"), dict)
                        else cfg.get("signal_slope_lookback_bars", 5))
        _msfrac   = float(_prod_cfg.get("signal", {}).get("min_slope_atr_frac", 0.12)
                          if isinstance(_prod_cfg.get("signal"), dict)
                          else cfg.get("signal_min_slope_atr_frac", 0.12))
        _ml_min   = float(cfg.get("min_long_confidence", 0.65))
        _atr_pts  = (features["atr_14"].reindex(common_idx) * close_prices).fillna(0)
        rules_sig = generate_rules_signal(
            close              = close_prices,
            atr_pts            = _atr_pts,
            ema_period         = _ema_p,
            slope_lookback     = _slb,
            min_slope_atr_frac = _msfrac,
        )
        # ML veto: require conf >= threshold AND direction agreement
        ml_veto_pass = (
            (conf_series >= _ml_min)
            & ((rules_sig == 0) | (ml_sig_series == rules_sig))
        )
        sig_series = rules_sig.where(ml_veto_pass, other=pd.Series(0, index=common_idx))
        sig_series = sig_series.astype(np.int8)
        _n_rules   = int((rules_sig != 0).sum())
        _n_vetoed  = int((rules_sig != 0) & ~ml_veto_pass).sum()
        log.info(
            "[%s] Rules-first mode: %d rules signals → %d vetoed by ML → %d passed",
            symbol, _n_rules, _n_vetoed, int((sig_series != 0).sum()),
        )
    else:
        sig_series = ml_sig_series

    # Pre-filter confidence coverage diagnostic: how many bars would trade
    # before session/ATR/trend filters are applied.
    _threshold_diag = cfg.get("min_long_confidence", 0.0)
    if _threshold_diag > 0.0:
        _conf_mask  = conf_series.values >= _threshold_diag
        _sig_mask   = sig_series.values != 0
        _n_conf     = int((_conf_mask & _sig_mask).sum())
        _n_bars     = len(conf_series)
        _pct        = 100.0 * _n_conf / max(_n_bars, 1)
        log.info(
            "[%s] Confidence coverage: %d / %d bars (%.1f%%) have "
            "confidence >= %.2f and non-flat signal (before session/ATR/trend filters)",
            symbol, _n_conf, _n_bars, _pct, _threshold_diag,
        )
        if _pct < 0.5:
            log.warning(
                "[%s] Very low confidence coverage (%.1f%%) — "
                "model may be under-confident on this dataset. "
                "Consider retraining or lowering the threshold.",
                symbol, _pct,
            )

    # ATR in ticks: atr_14 feature is atr_pts/close
    atr_ticks_series = (
        features["atr_14"].reindex(common_idx) * close_prices / tick_size
    ).fillna(0)

    # ── EMA slope for trend filter ───────────────────────────────────────────────
    # Uses slope = EMA(N) - EMA(N).shift(5) instead of close-vs-EMA.
    # Longs allowed when slope > 0 (uptrend); shorts when slope < 0 (downtrend).
    # Softer than price-vs-EMA: catches the trend direction rather than position.
    ema_series = None
    if cfg.get("trend_filter_enabled", False):
        ema_period = int(cfg.get("trend_filter_ema_period", 200))
        _raw_ema = close_prices.ewm(span=ema_period, adjust=False).mean()
        ema_series = _raw_ema - _raw_ema.shift(5)   # slope = delta over 5 bars
        log.info("Trend filter enabled: EMA(%d) slope", ema_period)

    # ── Simulate trades ─────────────────────────────────────────────────────────
    # Regime classifier removed: ATR filter + trend filter are sufficient.
    # The 3-layer regime stack (ATR regime + vol_regime_20_60 + EMA slope)
    # caused zero-trade folds by requiring all three to agree simultaneously.
    trades, equity_curve, cost_summary = _simulate_trades(
        signals            = sig_series,
        open_prices        = open_prices,
        atr_ticks          = atr_ticks_series,
        cfg                = cfg,
        friction_pts       = friction_per_side,
        commission_rt      = commission_rt,
        multiplier         = multiplier,
        conf_series        = conf_series,
        close_prices       = close_prices,
        ema_series         = ema_series,
        atr_regime_series  = pd.Series(1.0, index=common_idx),  # disabled
        regime_series      = pd.Series(1,   index=common_idx),   # always pass
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

    # ── Sequential execution pipeline breakdown ───────────────────────────────
    flt        = cost_summary.get("filter_counters", {})
    _n_total   = flt.get("n_total_bars",        0)
    _n_conf    = flt.get("n_confident_signals",  0)
    _n_sess    = flt.get("n_after_session",      0)
    _n_bo      = flt.get("n_after_blackout",     0)
    _n_atr     = flt.get("n_after_atr",          0)
    _n_regime  = flt.get("n_after_regime",       0)
    _n_chop_b  = flt.get("n_chop_blocked",       0)
    _n_lv_b    = flt.get("n_low_vol_blocked",    0)
    _n_trend   = flt.get("n_after_trend",        0)
    _n_slope_b = flt.get("n_blocked_by_slope",   0)
    _n_risk    = flt.get("n_after_risk",         0)
    _n_cd      = flt.get("n_after_cooldowns",    0)
    _n_queued  = flt.get("n_entries_queued",     0)
    _n_trend_e = flt.get("n_trend_entries",      0)
    _n_blk1    = flt.get("n_in_block1",          0)
    _n_blk2    = flt.get("n_in_block2",          0)
    _n_exec    = len(trades)
    _b1_pnl    = cost_summary.get("block1_net_pnl", 0.0)
    _b2_pnl    = cost_summary.get("block2_net_pnl", 0.0)
    _b1_tr     = cost_summary.get("block1_n_trades", 0)
    _b2_tr     = cost_summary.get("block2_n_trades", 0)

    def _pct(num, den):
        return f"{100*num/den:.0f}%" if den > 0 else "n/a"

    log.info(
        "[%s] Execution pipeline (%d total bars):",
        symbol, _n_total,
    )
    log.info(
        "[%s]   confident signals : %5d        (%s of bars)",
        symbol, _n_conf, _pct(_n_conf, _n_total),
    )
    log.info(
        "[%s]   → after session   : %5d → %5d  (%s pass)",
        symbol, _n_conf, _n_sess, _pct(_n_sess, _n_conf),
    )
    log.info(
        "[%s]   → after blackout  : %5d → %5d  (%s pass)",
        symbol, _n_sess, _n_bo, _pct(_n_bo, _n_sess),
    )
    log.info(
        "[%s]   → after ATR       : %5d → %5d  (%s pass)",
        symbol, _n_bo, _n_atr, _pct(_n_atr, _n_bo),
    )
    log.info(
        "[%s]   → after regime    : %5d → %5d  (%s pass)  "
        "[chop_blocked=%d  low_vol_blocked=%d  trend_entries=%d]",
        symbol, _n_atr, _n_regime, _pct(_n_regime, _n_atr),
        _n_chop_b, _n_lv_b, _n_trend_e,
    )
    log.info(
        "[%s]   → after trend     : %5d → %5d  (%s pass)  [slope_blocked=%d]",
        symbol, _n_regime, _n_trend, _pct(_n_trend, _n_regime), _n_slope_b,
    )
    log.info(
        "[%s]   → after risk/halt : %5d → %5d  (%s pass)",
        symbol, _n_trend, _n_risk, _pct(_n_risk, _n_trend),
    )
    log.info(
        "[%s]   → after cooldowns : %5d → %5d  (%s pass)",
        symbol, _n_risk, _n_cd, _pct(_n_cd, _n_risk),
    )
    log.info(
        "[%s]   → entries queued  : %5d → %5d  (%s pass)  "
        "[block1=%d  block2=%d]",
        symbol, _n_cd, _n_queued, _pct(_n_queued, _n_cd), _n_blk1, _n_blk2,
    )
    log.info(
        "[%s]   → trades executed : %5d → %5d  (%s of queued)  "
        "[block1: %d trades $%.0f | block2: %d trades $%.0f]",
        symbol, _n_queued, _n_exec, _pct(_n_exec, _n_queued),
        _b1_tr, _b1_pnl, _b2_tr, _b2_pnl,
    )
    log.info(
        "[%s]   OVERALL: conf→executed = %d → %d  (%s end-to-end pass-through)",
        symbol, _n_conf, _n_exec, _pct(_n_exec, _n_conf),
    )

    # ── Pipeline health check (<5% pass-through) ──────────────────────────────
    _overall_pct = 100.0 * _n_exec / max(_n_conf, 1) if _n_conf > 0 else 0.0
    if _n_conf > 0 and _n_exec < _n_conf * 0.05:
        log.warning(
            "[%s] WARNING PIPELINE HEALTH\n"
            "  %d confident signals\n"
            "  %d in-session\n"
            "  %d after ATR\n"
            "  %d after regime  (chop_blk=%d  lv_blk=%d)\n"
            "  %d after trend  (slope_blk=%d)\n"
            "  %d queued  [block1=%d block2=%d  trend_entries=%d]\n"
            "  %d executed  [block1: %d trades $%.0f | block2: %d trades $%.0f]\n"
            "  Overall pass-through = %.1f%%",
            symbol,
            _n_conf, _n_sess, _n_atr,
            _n_regime, _n_chop_b, _n_lv_b,
            _n_trend, _n_slope_b,
            _n_queued, _n_blk1, _n_blk2, _n_trend_e,
            _n_exec, _b1_tr, _b1_pnl, _b2_tr, _b2_pnl, _overall_pct,
        )

    # ── CRITICAL pipeline blockage check ─────────────────────────────────────
    # Compute sequential drop at each stage; identify the dominant bottleneck.
    if _n_conf > 0 and _n_exec < _n_conf * 0.10:
        _stages = [
            ("session",   _n_conf,   _n_sess),
            ("blackout",  _n_sess,   _n_bo),
            ("ATR",       _n_bo,     _n_atr),
            ("regime",    _n_atr,    _n_regime),
            ("trend",     _n_regime, _n_trend),
            ("risk/halt", _n_trend,  _n_risk),
            ("cooldown",  _n_risk,   _n_cd),
            ("queue",     _n_cd,     _n_queued),
        ]
        # Note: slope_blocked contributes to trend stage drop
        _worst_stage = max(_stages, key=lambda s: s[1] - s[2])
        _w_name, _w_in, _w_out = _worst_stage
        log.critical(
            "[%s] CRITICAL PIPELINE BLOCKAGE\n"
            "  Top bottleneck: %s filter (blocked %d signals)\n"
            "  conf→executed = %d → %d (%.0f%% end-to-end)\n"
            "  %s filter: %d → %d (%.0f%% pass)",
            symbol,
            _w_name, _w_in - _w_out,
            _n_conf, _n_exec, _overall_pct,
            _w_name, _w_in, _w_out, 100.0 * _w_out / max(_w_in, 1),
        )

    # ── Pass-through warnings ─────────────────────────────────────────────────
    if _n_sess > 0 and _n_queued < _n_sess * 0.20:
        log.warning(
            "[%s] LOW PIPELINE YIELD: only %d queued entries from %d in-session "
            "confident signals (%.0f%% < 20%% threshold). "
            "Consider loosening: cooldown_bars_after_exit, cooldown_bars_after_loss, "
            "min_holding_bars, max_trades_per_day, or max_consecutive_losses.",
            symbol, _n_queued, _n_sess, 100*_n_queued/max(_n_sess, 1),
        )
    if _n_queued > 0 and _n_exec < _n_queued * 0.10:
        log.warning(
            "[%s] EXECUTION UNDERPERFORMANCE: only %d trades executed from %d queued "
            "entries (%.0f%% < 10%% threshold). "
            "The delay=1 bar fill or position-already-open logic may be losing orders.",
            symbol, _n_exec, _n_queued, 100*_n_exec/max(_n_queued, 1),
        )

    # Post-run sanity safeguard: warn if trades are far below what validation suggested
    _val_trades = int(schema.get("val_confident_trades", 0))
    _val_coverage = float(schema.get("val_trade_coverage_pct", 0))
    if _val_trades > 0 and _val_coverage > 0:
        _expected = max(1, int(_val_coverage * len(features)))
        _actual   = len(trades)
        if _actual < max(5, _expected // 10):
            log.warning(
                "[%s] Only %d trade(s) executed (validation suggested ~%d). "
                "Selected threshold may be too restrictive for deployment — "
                "check confidence coverage log above or retrain.",
                symbol, _actual, _expected,
            )

    # ── Save report ─────────────────────────────────────────────────────────────
    if save_report:
        report_dir  = Path("artifacts/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{symbol}_backtest.json"

        report = {
            "symbol":          symbol,
            "run_at":          datetime.utcnow().isoformat(),
            "cfg":             {k: str(v) for k, v in cfg.items()},
            "metrics":         metrics,
            "filter_counters": cost_summary.get("filter_counters", {}),
            "trades":          trades[:500],
            "equity_curve":    equity_curve.round(2).tolist(),
        }
        report_path.write_text(json.dumps(report, indent=2, default=str))
        log.info("Report saved to %s", report_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────


def _in_session(ts: pd.Timestamp, start_h: float, end_h: float) -> bool:
    """True if `ts` falls within [start_h, end_h) UTC hours.

    start_h and end_h support fractional values (e.g. 13.5 = 13:30 UTC).
    """
    if start_h == 0 and end_h >= 24:
        return True
    h = ts.hour + ts.minute / 60.0
    return start_h <= h < end_h


def _in_session_blocks(
    ts: pd.Timestamp,
    blocks: list,
) -> tuple[bool, int]:
    """Return (in_any_block, block_idx) where block_idx is 1-based (0 = not in session).

    blocks: list of [start_h, end_h] pairs (fractional UTC hours).
    """
    h = ts.hour + ts.minute / 60.0
    for idx, b in enumerate(blocks):
        if b[0] <= h < b[1]:
            return True, idx + 1
    return False, 0


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
    atr_regime_series: Optional[pd.Series] = None,
    regime_series: Optional[pd.Series] = None,
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
    sess_start    = float(cfg.get("session_start_utc_hour", 0))
    sess_end      = float(cfg.get("session_end_utc_hour", 24))
    # Session blocks: list of [start_h, end_h] pairs. When non-empty, replace
    # the single session window with dual hard-blocked trading windows.
    _raw_blocks   = cfg.get("session_blocks", [])
    sess_blocks   = [(float(b[0]), float(b[1])) for b in _raw_blocks] if _raw_blocks else []
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
    trend_filter_on  = bool(cfg.get("trend_filter_enabled", False))
    # Trend-quality gate: minimum |EMA_slope| / ATR_pts fraction (0 = disabled)
    trend_slope_min  = float(cfg.get("trend_slope_min_atr_frac", 0.0))
    tick_size        = float(cfg.get("tick_size", 0.25))

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
    # tick_size already loaded above for trend_slope_min_atr_frac

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

    atr_regime_arr = None
    if atr_regime_series is not None:
        atr_regime_arr = atr_regime_series.reindex(signals.index).values.astype(np.float64)

    regime_arr = None
    if regime_series is not None:
        regime_arr = regime_series.reindex(signals.index).values.astype(np.int8)

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
    pending_block     = 0   # session block of pending queued entry (1 or 2; 0 = unknown)
    entry_block       = 0   # session block of current open position's entry bar

    # Session-block PnL tracking
    block1_pnl    = 0.0
    block2_pnl    = 0.0
    block1_trades = 0
    block2_trades = 0

    # Risk / prop-firm state
    daily_pnl          = 0.0
    peak_equity        = 0.0       # max equity achieved (for drawdown)
    consecutive_losses = 0
    consec_losses_max  = 0
    daily_halt         = False
    kill_switch_active = False
    loss_cooldown_left = 0
    daily_halt_count   = 0

    # ── Per-filter block counters ─────────────────────────────────────────────
    # Sequential pipeline: each stage counts how many confident signals survive
    # after applying that filter *and all previous filters* in order.
    # n_after_cooldowns counts bars where cooldown/holding/daily-cap would pass
    # at that instant (state-dependent, checked on the same per-bar state).
    flt = {
        "n_total_bars":        0,   # total bars processed
        "n_confident_signals": 0,   # conf >= threshold AND raw_sig != 0
        "n_after_session":     0,   # survived session block filter
        "n_after_blackout":    0,   # survived news blackout
        "n_after_atr":         0,   # survived ATR range filter
        "n_after_regime":      0,   # survived regime classifier (trend only)
        "n_chop_blocked":      0,   # blocked: chop regime (adequate vol, no slope)
        "n_low_vol_blocked":   0,   # blocked: low-vol regime (ATR or vol too low)
        "n_after_trend":       0,   # survived EMA slope direction + quality filter
        "n_blocked_by_slope":  0,   # blocked: slope magnitude < min_slope_atr_frac
        "n_after_risk":        0,   # survived risk/halt gate (kill-switch, daily halt)
        "n_after_cooldowns":   0,   # survived cooldowns + holding + daily cap
        "n_entries_queued":    0,   # target != position — new entry queued
        "n_trend_entries":     0,   # queued entries that occurred in trend regime
        "n_in_block1":         0,   # queued entries in session block 1
        "n_in_block2":         0,   # queued entries in session block 2
    }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _size(atr_t: float, session_factor: float = 1.0) -> int:
        """Compute position size (contracts) based on sizing method.

        session_factor: 1.0 inside session, 0.5 outside (soft session weight).
        """
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
        return max(1, min(int(n_c * session_factor), max_contracts))

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
                    # Attribute PnL to the block of the entry bar
                    if entry_block == 1:
                        block1_pnl    += net_pnl
                        block1_trades += 1
                    elif entry_block == 2:
                        block2_pnl    += net_pnl
                        block2_trades += 1
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
                        "entry_block":   entry_block,
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
                        entry_block     = pending_block

                elif position == 0 and pending_target != 0:
                    entry_price     = o + pending_target * friction_pts
                    position        = pending_target
                    entry_idx       = i
                    entry_contracts = n_c
                    entry_block     = pending_block

            pending_target = None

        # ── STEP 2: Generate signal for this bar ──────────────────────────────
        o   = open_arr[i]
        sig = int(sig_arr[i])
        if np.isnan(o):
            equity_arr[i] = equity
            continue

        # Filter 1: Confidence gating
        conf_t = 0.0
        _session_factor = 1.0   # updated in Filter 2; used for sizing
        if conf_arr is not None:
            conf_t = conf_arr[i] if not np.isnan(conf_arr[i]) else 0.0
            if sig == 1 and conf_t < min_long_conf:
                sig = 0
            elif sig == -1 and conf_t < min_short_conf:
                sig = 0

        # ── Session block membership for this bar ────────────────────────────
        # Pre-compute once; used by diagnostic counter AND actual Filter 2.
        atr_t = atr_arr[i]
        if sess_blocks:
            _in_block, _block_idx = _in_session_blocks(ts, sess_blocks)
        else:
            _in_block = _in_session(ts, sess_start, sess_end)
            _block_idx = 1 if _in_block else 0

        # ── Sequential pipeline diagnostics ──────────────────────────────────
        # Track how many confident signals survive each filter in order.
        # Each stage is only counted if all prior stages passed.
        flt["n_total_bars"] += 1
        _conf_sig = int(sig)   # save post-confidence value
        if _conf_sig != 0:
            flt["n_confident_signals"] += 1
            if _in_block:
                flt["n_after_session"] += 1
                if not _in_blackout(ts, blackouts):
                    flt["n_after_blackout"] += 1
                    _regime_t = atr_regime_arr[i] if atr_regime_arr is not None else 1.0
                    if atr_min <= atr_t <= atr_max and _regime_t >= 0.8:
                        flt["n_after_atr"] += 1
                        # Regime classifier gate
                        _regime_val = int(regime_arr[i]) if regime_arr is not None else 1
                        if _regime_val == 1:
                            flt["n_after_regime"] += 1
                            _trend_ok = True
                            if trend_filter_on and ema_arr is not None:
                                _et = ema_arr[i]
                                if not np.isnan(_et):
                                    if (_conf_sig == 1 and _et <= 0) or \
                                       (_conf_sig == -1 and _et >= 0):
                                        _trend_ok = False
                                    elif trend_slope_min > 0.0:
                                        _atp = atr_t * tick_size
                                        if _atp > 0 and abs(_et) / _atp < trend_slope_min:
                                            _trend_ok = False
                            if _trend_ok:
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
                                        not (daily_trade_count >= max_tpd and _conf_sig != position)
                                    )
                                    if _cd_ok:
                                        flt["n_after_cooldowns"] += 1

        # Filter 2: Session blocks — hard block outside all defined windows.
        # When session_blocks is configured: signals outside both blocks are zeroed.
        # Legacy (session_blocks empty): soft block — signal kept, size halved.
        _session_factor = 1.0
        if sess_blocks:
            if not _in_block:
                sig = 0
        else:
            _session_factor = 1.0 if _in_block else 0.5

        # Filter 3: News blackout
        if _in_blackout(ts, blackouts):
            sig = 0

        # Filter 4: ATR range check
        if not (atr_min <= atr_t <= atr_max):
            sig = 0

        # Filter 4.5: Regime classifier — allow new entries only in trend regime.
        # chop (0) and low_vol (-1) are blocked; trend (1) passes.
        # When regime_arr not provided, fall back to legacy ATR regime check.
        elif regime_arr is not None:
            _rval = int(regime_arr[i])
            if _rval != 1:
                if sig != 0:
                    if _rval == 0:
                        flt["n_chop_blocked"] += 1
                    else:
                        flt["n_low_vol_blocked"] += 1
                sig = 0
        else:
            # Legacy fallback: ATR regime only (no full regime classifier)
            _regime_t = atr_regime_arr[i] if atr_regime_arr is not None else 1.0
            if _regime_t < 0.8:
                sig = 0   # skip trading in low-volatility chop

        # Filter 5: Trend slope filter — direction (5a) + quality gate (5b).
        # ema_arr holds EMA(N) - EMA(N).shift(5) — the 5-bar slope.
        # 5a: Long allowed when slope > 0; short allowed when slope < 0.
        # 5b: |slope| / ATR_pts must exceed trend_slope_min_atr_frac (if set).
        if trend_filter_on and ema_arr is not None:
            ema_t = ema_arr[i]
            if not np.isnan(ema_t):
                if sig == 1 and ema_t <= 0:        # 5a: long blocked — downsloping EMA
                    sig = 0
                elif sig == -1 and ema_t >= 0:     # 5a: short blocked — upsloping EMA
                    sig = 0
                elif sig != 0 and trend_slope_min > 0.0:  # 5b: slope magnitude gate
                    _atr_pts = atr_t * tick_size
                    if _atr_pts > 0 and abs(ema_t) / _atr_pts < trend_slope_min:
                        flt["n_blocked_by_slope"] += 1
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
                    n_c         = _size(atr_t, _session_factor)
                    entry_price     = o + target * friction_pts
                    position        = target
                    entry_idx       = i
                    entry_contracts = n_c
        else:
            # Normal delayed execution: queue for next bar
            if target != position and i < n - 1:
                pending_target    = target
                pending_contracts = _size(atr_t, _session_factor) if target != 0 else entry_contracts
                pending_block     = _block_idx
                if target != 0:
                    daily_trade_count += 1
                    flt["n_entries_queued"] += 1
                    if regime_arr is not None and int(regime_arr[i]) == 1:
                        flt["n_trend_entries"] += 1
                    if _block_idx == 1:
                        flt["n_in_block1"] += 1
                    elif _block_idx == 2:
                        flt["n_in_block2"] += 1

        equity_arr[i] = equity

    equity_series = pd.Series(equity_arr, index=signals.index, name="equity")
    cost_summary  = {
        "gross_pnl":              gross_pnl,
        "total_costs":            total_cost,
        "net_pnl":                gross_pnl - total_cost,
        "consecutive_losses_max": consec_losses_max,
        "kill_switch_triggered":  kill_switch_active,
        "daily_halt_count":       daily_halt_count,
        "filter_counters":        flt,
        "block1_net_pnl":         block1_pnl,
        "block2_net_pnl":         block2_pnl,
        "block1_n_trades":        block1_trades,
        "block2_n_trades":        block2_trades,
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
