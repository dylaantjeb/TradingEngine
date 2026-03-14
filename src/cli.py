"""
TradingEngine CLI – single entry point.
Usage:  python -m src.cli <subcommand> [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.utils.logger import configure_logging

configure_logging()
log = logging.getLogger("cli")


def _seed_everything(seed: int) -> None:
    """Seed Python random, NumPy, and set PYTHONHASHSEED for reproducibility."""
    import os
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Global seeds set: seed=%d  (random, numpy, PYTHONHASHSEED)", seed)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ──────────────────────────────────────────────────────────────────────────────


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch minute bars from IBKR and save to data/raw/<SYMBOL>_M1.csv."""
    from src.data_engine.ibkr_fetch import IBKRFetcher

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.symbol}_M1.csv"

    use_rth: bool | None = None
    if args.use_rth is not None:
        use_rth = args.use_rth.lower() in ("1", "true", "yes")

    fetcher = IBKRFetcher()
    fetcher.fetch(
        symbol=args.symbol,
        days=args.days,
        out_path=out_path,
        bar_size=args.bar_size,
        use_rth=use_rth,
    )
    log.info("Saved %s", out_path)


def cmd_yahoo_fetch(args: argparse.Namespace) -> None:
    """Fetch bars from Yahoo Finance (free, no IBKR required)."""
    from src.data.brokers.yahoo_adapter import fetch_yahoo

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.symbol}_M1.csv"

    fetch_yahoo(
        symbol=args.symbol,
        days=args.days,
        bar_size=args.bar_size,
        out_path=out_path,
    )


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Load raw CSV → engineer features → label → save parquets."""
    import pandas as pd
    from src.features.builder import build_features
    from src.labels.triple_barrier import label_triple_barrier

    log.info("Loading %s", args.input)
    try:
        df = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    except FileNotFoundError:
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    log.info("Loaded %d bars", len(df))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / f"{args.symbol}_features.parquet"
    lbl_path  = out_dir / f"{args.symbol}_labels.parquet"

    try:
        features = build_features(df)
        log.info("Features shape: %s", features.shape)
        features.to_parquet(feat_path, engine="pyarrow")

        labels = label_triple_barrier(df, pt=args.pt, sl=args.sl, max_hold=args.max_hold)
        log.info("Labels distribution:\n%s", labels["label"].value_counts())
        labels.to_parquet(lbl_path, engine="pyarrow")
    except ImportError:
        log.error("pyarrow is required:  pip install pyarrow")
        sys.exit(1)

    log.info("Features → %s", feat_path)
    log.info("Labels   → %s", lbl_path)


def cmd_train(args: argparse.Namespace) -> None:
    """Train XGBoost model with Optuna hyperparameter search."""
    from src.training.train import train

    train(
        symbol=args.symbol,
        n_trials=args.trials,
        select_by=getattr(args, "select_by", "f1"),
    )


def _load_profile_overrides(symbol: str, profile_name: str) -> tuple[dict, list | None]:
    """
    Load exec_cfg_overrides and threshold_candidates for a named profile.
    Returns (overrides_dict, threshold_candidates_or_None).
    Exits with code 1 if the profile file or profile name is not found.
    """
    from src.backtest.profile_eval import load_profiles, _profile_to_cfg_overrides

    try:
        profiles = load_profiles(symbol)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if profile_name not in profiles:
        log.error(
            "Profile '%s' not found in config/profiles/%s_profiles.yaml.\n"
            "  Available profiles: %s",
            profile_name, symbol, ", ".join(profiles.keys()),
        )
        sys.exit(1)

    profile    = profiles[profile_name]
    overrides  = _profile_to_cfg_overrides(profile)
    thresholds = profile.get("threshold_candidates", None)
    log.info(
        "Profile [%s] loaded — overrides: %s  threshold_candidates: %s",
        profile_name, list(overrides.keys()), thresholds,
    )
    return overrides, thresholds


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run vectorised backtest and print metrics."""
    import yaml
    from src.backtest.engine import run_backtest

    with open(args.universe) as f:
        universe_cfg = yaml.safe_load(f)

    profile_name = getattr(args, "profile", None)

    for symbol in universe_cfg.get("symbols", []):
        cfg_overrides: dict | None = None
        if profile_name == "production":
            from src.deployment import get_exec_cfg_overrides
            cfg_overrides = get_exec_cfg_overrides(symbol)
            log.info("[%s] Backtest using PRODUCTION deployment artifact", symbol)
        elif profile_name:
            cfg_overrides, _ = _load_profile_overrides(symbol, profile_name)
            log.info("[%s] Backtest using profile [%s]", symbol, profile_name)

        log.info("Backtesting %s …", symbol)
        run_backtest(
            symbol=symbol,
            execution_delay_bars=args.execution_delay,
            max_trades_per_day=args.max_trades_per_day,
            slippage_ticks_per_side=args.slippage_ticks,
            commission_per_side_usd=args.commission,
            cfg_overrides=cfg_overrides,
        )


def cmd_live_paper(args: argparse.Namespace) -> None:
    """Simulate live paper trading by streaming rows from a CSV file."""
    from src.live.paper_engine import run_paper

    profile_name = getattr(args, "profile", None)
    cfg_overrides: dict | None = None
    threshold_candidates: list | None = None

    if profile_name == "production":
        # --profile production: load from the locked deployment artifact
        from src.deployment import get_exec_cfg_overrides, get_threshold_candidates
        cfg_overrides        = get_exec_cfg_overrides(args.symbol)
        threshold_candidates = get_threshold_candidates(args.symbol)
        log.info(
            "[%s] live-paper using PRODUCTION deployment artifact "
            "(exec overrides: %s)",
            args.symbol, list(cfg_overrides.keys()),
        )
    elif profile_name:
        cfg_overrides, threshold_candidates = _load_profile_overrides(
            args.symbol, profile_name
        )

    run_paper(
        symbol=args.symbol,
        csv_path=Path(args.input),
        cfg_overrides=cfg_overrides,
    )


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run walk-forward / OOS validation."""
    from src.backtest.walk_forward import run_walk_forward

    n_trials = 0 if getattr(args, "no_optuna", False) else args.trials

    profile_name = getattr(args, "profile", None)
    exec_overrides: dict | None = None
    threshold_candidates: list | None = None

    if profile_name == "production":
        from src.deployment import get_exec_cfg_overrides, get_threshold_candidates
        exec_overrides       = get_exec_cfg_overrides(args.symbol)
        threshold_candidates = get_threshold_candidates(args.symbol)
        log.info(
            "[%s] walk-forward using PRODUCTION deployment artifact "
            "(exec overrides: %s  thresholds: %s)",
            args.symbol, list(exec_overrides.keys()), threshold_candidates,
        )
    elif profile_name:
        exec_overrides, threshold_candidates = _load_profile_overrides(
            args.symbol, profile_name
        )

    run_walk_forward(
        symbol=args.symbol,
        mode=args.mode,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        split_pct=args.split_pct,
        n_trials=n_trials,
        select_by=getattr(args, "select_by", "f1"),
        save_report=True,
        exec_cfg_overrides=exec_overrides,
        threshold_candidates=threshold_candidates,
    )


def cmd_forward_test(args: argparse.Namespace) -> None:
    """
    Forward-test the ACCEPTED production winner.

    Loads the production deployment artifact, verifies the deployment state is
    ACCEPTED or higher, runs walk-forward with the exact config that was accepted,
    and prints a pass/fail verdict.

    This is the ONLY correct way to validate the strategy before going live:
      python -m src.cli forward-test --symbol ES
    """
    from src.backtest.walk_forward import run_walk_forward
    from src.deployment import (
        load_or_fail, get_exec_cfg_overrides, get_threshold_candidates,
        DeploymentState, update_deployment_state,
    )

    artifact = load_or_fail(args.symbol)
    state_str = artifact.get("deployment_state", "RESEARCH")
    state = DeploymentState(state_str)
    if not state.is_deployable():
        log.error(
            "[%s] Deployment state is '%s' — not deployable.\n"
            "  Run evaluate-profiles first, or manually advance the state.",
            args.symbol, state_str,
        )
        sys.exit(1)

    exec_overrides       = get_exec_cfg_overrides(args.symbol)
    threshold_candidates = get_threshold_candidates(args.symbol)
    profile_name         = artifact.get("profile_name", "unknown")

    seed = getattr(args, "seed", 42)
    _seed_everything(seed)

    log.info(
        "[%s] forward-test — profile=[%s]  state=%s  seed=%d  overrides=%s",
        args.symbol, profile_name, state_str, seed, list(exec_overrides.keys()),
    )

    n_trials = 0 if getattr(args, "no_optuna", False) else args.trials

    wf = run_walk_forward(
        symbol=args.symbol,
        mode=getattr(args, "mode", "rolling"),
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=getattr(args, "step_bars", None),
        split_pct=getattr(args, "split_pct", 0.8),
        n_trials=n_trials,
        select_by=getattr(args, "select_by", "f1"),
        save_report=True,
        exec_cfg_overrides=exec_overrides,
        threshold_candidates=threshold_candidates,
        seed=seed,
    )

    # Verdict
    agg = wf.aggregate if hasattr(wf, "aggregate") else {}
    folds = wf.folds if hasattr(wf, "folds") else []
    n_profitable = sum(1 for f in folds if f.profitable)
    total_pnl    = sum(f.net_pnl for f in folds)
    avg_pf_vals  = [f.profit_factor for f in folds]
    avg_pf       = sum(avg_pf_vals) / len(avg_pf_vals) if avg_pf_vals else 0.0

    print(f"\n{'='*70}")
    print(f"  FORWARD-TEST RESULT — {args.symbol}  [profile: {profile_name}]")
    print(f"{'='*70}")
    print(f"  Deployment state   : {state_str}")
    print(f"  Profitable folds   : {n_profitable}/{len(folds)}")
    print(f"  Total PnL          : ${total_pnl:+.0f}")
    print(f"  Avg profit factor  : {avg_pf:.2f}")

    if n_profitable >= 3 and avg_pf >= 1.2 and total_pnl > 0:
        print(f"\n  VERDICT: PASS — strategy holding up in forward test.")
        if state == DeploymentState.ACCEPTED and getattr(args, "promote", False):
            update_deployment_state(args.symbol, DeploymentState.FORWARD_DEPLOYED)
            print(f"  State promoted: ACCEPTED → FORWARD_DEPLOYED")
    else:
        print(f"\n  VERDICT: FAIL — re-run evaluate-profiles before going live.")

    print(f"{'='*70}\n")


def cmd_evaluate_profiles(args: argparse.Namespace) -> None:
    """Evaluate all execution profiles and determine if strategy is READY."""
    from src.backtest.profile_eval import (
        evaluate_all_profiles,
        print_scoreboard,
        save_artifacts,
        pick_winner,
    )

    seed             = getattr(args, "seed", 42)
    check_runs       = getattr(args, "acceptance_check_runs", 1)

    _seed_everything(seed)

    results = evaluate_all_profiles(
        symbol=args.symbol,
        n_trials=args.trials,
        select_by=getattr(args, "select_by", "f1"),
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        run_backtest=not getattr(args, "no_backtest", True),
        seed=seed,
        acceptance_check_runs=check_runs,
    )

    print_scoreboard(results, symbol=args.symbol)
    save_artifacts(
        args.symbol, results,
        seed=seed,
        acceptance_check_runs=check_runs,
    )

    winner = pick_winner(results)
    if winner is None:
        import sys
        sys.exit(2)   # non-zero exit → CI/CD can detect NO WINNER


def cmd_explain_signal(args: argparse.Namespace) -> None:
    """Explain the signal for the most recent bar in a CSV file."""
    import pandas as pd
    import json
    from src.strategy.signal_engine import SignalEngine
    from src.features.builder import MIN_ROWS

    log.info("Loading %s …", args.input)
    try:
        df = pd.read_csv(args.input, parse_dates=["timestamp"], index_col="timestamp")
    except FileNotFoundError:
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    df.sort_index(inplace=True)

    engine = SignalEngine(symbol=args.symbol)

    # Use the last (MIN_ROWS + buffer) bars as the window
    window_size = MIN_ROWS + 50
    window = df.iloc[-window_size:] if len(df) >= window_size else df

    output = engine.generate(window, account_equity=args.equity)

    print(f"\n{'='*60}")
    print(f"  Signal Explanation – {args.symbol}")
    print(f"{'='*60}")
    print(f"  Signal        : {output.signal:+d}  ({['SHORT','FLAT','LONG'][output.signal+1]})")
    print(f"  Confidence    : {output.confidence:.1%}")
    print(f"  Regime        : {output.regime}")
    print(f"  ATR           : {output.atr_pts:.4f} pts")
    print(f"  Rec. SL       : {output.recommended_sl_pts:.4f} pts")
    print(f"  Rec. TP       : {output.recommended_tp_pts:.4f} pts")
    print(f"  Rec. Size     : {output.recommended_size} contract(s)")
    print()
    print(f"  Filters passed: {', '.join(output.filters_passed) or 'none'}")
    print(f"  Filters failed: {', '.join(output.filters_failed) or 'none'}")
    print()
    print("  Top feature contributions:")
    for fc in output.top_features[:5]:
        bar = "▲" if fc["contribution"] > 0 else "▼"
        print(f"    {bar} {fc['name']:30s}  val={fc['value']:+.3f}  contrib={fc['contribution']:+.4f}")
    print()
    print(f"  Rationale: {output.rationale}")
    print(f"{'='*60}\n")


def cmd_run_production(args: argparse.Namespace) -> None:
    """
    Run the production engine with the rules-first architecture and FundedGuard.

    This command is the ONLY correct way to run the funded-account strategy in
    production or final paper-test mode.  It refuses to start if:

      - config/production_<SYMBOL>.yaml does not exist
      - The deployment artifact does not exist or is not in a deployable state
      - Any health check fails

    Architecture:
      EMA momentum rules (PRIMARY) → ML veto (≥0.65 conf + direction match) →
      FundedGuard ($500 daily halt / $300 soft / $3000 max DD)

    Usage:
      python -m src.cli run-production --symbol ES --input data/raw/ES_M1.csv
    """
    import yaml

    symbol = args.symbol
    prod_cfg_path = Path(f"config/production_{symbol}.yaml")

    # ── 1. Require production config ─────────────────────────────────────────
    if not prod_cfg_path.exists():
        log.error(
            "[%s] Production config not found: %s\n"
            "  Create config/production_%s.yaml before running run-production.\n"
            "  See config/production_ES.yaml for the canonical template.",
            symbol, prod_cfg_path, symbol,
        )
        sys.exit(1)

    with open(prod_cfg_path) as f:
        prod_cfg = yaml.safe_load(f) or {}

    # Flatten the nested production config into the overrides dict that
    # run_paper / health checks expect.
    _sig   = prod_cfg.get("signal",    {})
    _vol   = prod_cfg.get("volatility",{})
    _ml    = prod_cfg.get("ml_veto",   {})
    _trade = prod_cfg.get("trade",     {})
    _thr   = prod_cfg.get("throttles", {})
    _costs = prod_cfg.get("costs",     {})
    _risk  = prod_cfg.get("risk",      {})
    _sess  = prod_cfg.get("session",   {})
    _cont  = prod_cfg.get("contract",  {})

    # Build session blocks from start/end strings
    def _parse_hour(t: str) -> float:
        h, m = map(int, t.split(":"))
        return h + m / 60.0

    _s_start = _parse_hour(_sess.get("start_utc", "13:30"))
    _s_end   = _parse_hour(_sess.get("end_utc",   "16:00"))

    cfg_overrides: dict = {
        # Session
        "session_blocks":             [[_s_start, _s_end]],
        # Signal config (stored nested so paper_engine can read them)
        "signal":                     _sig,
        "signal_ema_period":          int(_sig.get("ema_period", 20)),
        "signal_slope_lookback_bars": int(_sig.get("slope_lookback_bars", 5)),
        "signal_min_slope_atr_frac":  float(_sig.get("min_slope_atr_frac", 0.12)),
        # Volatility
        "atr_min_ticks":              float(_vol.get("atr_min_ticks", 5)),
        "atr_max_ticks":              float(_vol.get("atr_max_ticks", 150)),
        # ML veto
        "min_long_confidence":        float(_ml.get("min_confidence", 0.65)),
        "min_short_confidence":       float(_ml.get("min_confidence", 0.65)),
        # Trend filter kept on (rules signal provides direction; EMA filter aligns)
        "trend_filter_enabled":       True,
        "trend_filter_ema_period":    int(_sig.get("ema_period", 20)),
        "trend_slope_min_atr_frac":   float(_sig.get("min_slope_atr_frac", 0.12)),
        # Trade params
        "execution_delay_bars":       int(_trade.get("execution_delay_bars", 1)),
        "atr_stop_multiplier":        float(_trade.get("atr_stop_multiplier", 2.0)),
        "atr_target_multiplier":      float(_trade.get("atr_target_multiplier", 3.0)),
        # Throttles
        "max_trades_per_day":         int(_thr.get("max_trades_per_day", 4)),
        "cooldown_bars_after_exit":   int(_thr.get("cooldown_bars_after_exit", 1)),
        "cooldown_bars_after_loss":   int(_thr.get("cooldown_bars_after_loss", 3)),
        "min_holding_bars":           int(_thr.get("min_holding_bars", 1)),
        # Costs
        "commission_per_side_usd":    float(_costs.get("commission_per_side_usd", 1.50)),
        "slippage_ticks_per_side":    float(_costs.get("slippage_ticks_per_side", 1.0)),
        "spread_ticks":               float(_costs.get("spread_ticks", 1.0)),
        # Risk (nested — FundedGuard reads this dict directly)
        "risk":                       _risk,
        "starting_equity":            float(_risk.get("starting_equity_usd", 50000.0)),
        "max_daily_loss_usd":         float(_risk.get("daily_loss_hard_usd", 500.0)),
        "max_total_drawdown_usd":     float(_risk.get("max_drawdown_usd", 3000.0)),
        "max_consecutive_losses":     int  (_risk.get("max_consecutive_losses", 3)),
        # Contract
        "tick_size":                  float(_cont.get("tick_size", 0.25)),
        "multiplier":                 float(_cont.get("multiplier", 50)),
        "fixed_contracts":            int  (_risk.get("contracts", 1)),
        "max_contracts":              int  (_risk.get("max_contracts", 2)),
    }

    # ── 2. Require valid deployment artifact + health checks ─────────────────
    from src.deployment import assert_health_or_abort
    assert_health_or_abort(symbol, cfg_overrides, require_artifact=True, mode="run-production")

    # ── 3. Research-mode warning ──────────────────────────────────────────────
    log.info(
        "[%s] run-production: rules-first engine active\n"
        "  Signal:    EMA(%d) momentum + slope quality gate (min_frac=%.2f)\n"
        "  ML veto:   conf >= %.2f + direction match required\n"
        "  Session:   %s – %s UTC (%.1f – %.1f)\n"
        "  FundedGuard: hard=$%.0f  soft=$%.0f  trail_dd=$%.0f  max_dd=$%.0f",
        symbol,
        int(_sig.get("ema_period", 20)),
        float(_sig.get("min_slope_atr_frac", 0.12)),
        float(_ml.get("min_confidence", 0.65)),
        _sess.get("start_utc", "13:30"), _sess.get("end_utc", "16:00"),
        _s_start, _s_end,
        float(_risk.get("daily_loss_hard_usd", 500)),
        float(_risk.get("daily_loss_soft_usd", 300)),
        float(_risk.get("trailing_drawdown_usd", 2000)),
        float(_risk.get("max_drawdown_usd", 3000)),
    )

    # ── 4. Run ────────────────────────────────────────────────────────────────
    from src.live.paper_engine import run_paper
    run_paper(
        symbol           = symbol,
        csv_path         = Path(args.input),
        bar_delay        = getattr(args, "bar_delay", 0.0),
        cfg_overrides    = cfg_overrides,
        use_rules_signal = True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="TradingEngine – fetch, build, train, backtest, paper-trade",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # ── fetch ──────────────────────────────────────────────────────────────────
    p_fetch = sub.add_parser("fetch", help="Fetch IBKR minute bars")
    p_fetch.add_argument("--symbol", required=True, help="Root symbol, e.g. ES")
    p_fetch.add_argument(
        "--days", type=int, default=10,
        help="Calendar days of history to fetch (default 10)",
    )
    p_fetch.add_argument(
        "--out", default=None,
        help="Output CSV path (default: data/raw/<SYMBOL>_M1.csv)",
    )
    p_fetch.add_argument(
        "--bar-size", dest="bar_size", default="1 min",
        help='IBKR bar size string (default "1 min")',
    )
    p_fetch.add_argument(
        "--use-rth", dest="use_rth", default=None,
        metavar="true|false",
        help="Regular trading hours only? (default: from config)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    # ── yahoo-fetch ────────────────────────────────────────────────────────────
    p_yf = sub.add_parser(
        "yahoo-fetch",
        help="Fetch bars from Yahoo Finance (free, no IBKR required)",
    )
    p_yf.add_argument("--symbol", required=True,
                      help="Symbol (ES/NQ → SPY/QQQ proxy) or any Yahoo ticker")
    p_yf.add_argument(
        "--days", type=int, default=7,
        help="Days of history (1-min bars limited to last 7 days; default 7)",
    )
    p_yf.add_argument(
        "--bar-size", dest="bar_size", default="1 min",
        help='Bar size: "1 min", "5 min", "15 min", "1 hour", "1 day" (default "1 min")',
    )
    p_yf.add_argument(
        "--out", default=None,
        help="Output CSV path (default: data/raw/<SYMBOL>_M1.csv)",
    )
    p_yf.set_defaults(func=cmd_yahoo_fetch)

    # ── build-dataset ──────────────────────────────────────────────────────────
    p_build = sub.add_parser("build-dataset", help="Engineer features + labels from raw CSV")
    p_build.add_argument("--symbol", required=True, help="Symbol name (used for output filenames)")
    p_build.add_argument("--input", required=True, help="Path to raw CSV file")
    p_build.add_argument("--pt", type=float, default=1.5, help="Profit-take multiplier (default 1.5)")
    p_build.add_argument("--sl", type=float, default=1.0, help="Stop-loss multiplier (default 1.0)")
    p_build.add_argument(
        "--max-hold", type=int, default=60, dest="max_hold",
        help="Max holding period in bars (default 60)"
    )
    p_build.set_defaults(func=cmd_build_dataset)

    # ── train ──────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train XGBoost model (Optuna tuning)")
    p_train.add_argument("--symbol", required=True)
    p_train.add_argument(
        "--trials", type=int, default=20, help="Number of Optuna trials (default 20)"
    )
    p_train.add_argument(
        "--select-by", dest="select_by", default="f1",
        choices=["f1", "trading"],
        help=(
            "Model selection objective: "
            "'f1' (maximise macro-F1, default) or "
            "'trading' (composite of F1 + trading quality — prefers fewer, "
            "higher-confidence signals with good directional accuracy)"
        ),
    )
    p_train.set_defaults(func=cmd_train)

    # ── backtest ───────────────────────────────────────────────────────────────
    p_bt = sub.add_parser("backtest", help="Vectorised backtest")
    p_bt.add_argument(
        "--universe",
        default="config/universe.yaml",
        help="Path to universe YAML (default config/universe.yaml)",
    )
    p_bt.add_argument(
        "--execution-delay", dest="execution_delay", type=int, default=None,
        help="Fill delay in bars (default from config, usually 1)",
    )
    p_bt.add_argument(
        "--max-trades-per-day", dest="max_trades_per_day", type=int, default=None,
        help="Daily trade cap (default from config, usually 20)",
    )
    p_bt.add_argument(
        "--slippage-ticks", dest="slippage_ticks", type=float, default=None,
        help="Slippage ticks per side (default from config, usually 0.5)",
    )
    p_bt.add_argument(
        "--commission", dest="commission", type=float, default=None,
        help="Commission per contract per side USD (default from config, usually 1.50)",
    )
    p_bt.add_argument(
        "--profile", dest="profile", default=None, metavar="PROFILE_NAME",
        help=(
            "Apply named execution profile from config/profiles/<symbol>_profiles.yaml. "
            "Use 'production' to load the locked deployment artifact. "
            "Overrides session_blocks, ATR limits, trend filter, throttles, etc."
        ),
    )
    p_bt.set_defaults(func=cmd_backtest)

    # ── live-paper ─────────────────────────────────────────────────────────────
    p_live = sub.add_parser("live-paper", help="Paper-trade simulation from CSV stream")
    p_live.add_argument("--symbol", required=True)
    p_live.add_argument("--input", required=True, help="Path to CSV to stream")
    p_live.add_argument(
        "--profile", dest="profile", default=None, metavar="PROFILE_NAME",
        help=(
            "Apply named execution profile overrides. "
            "Use 'production' to load the locked deployment artifact "
            "(recommended for all production runs). "
            "Without --profile, falls back to config/universe.yaml defaults."
        ),
    )
    p_live.set_defaults(func=cmd_live_paper)

    # ── walk-forward ───────────────────────────────────────────────────────────
    p_wf = sub.add_parser(
        "walk-forward",
        help="Walk-forward / OOS validation (rolling | expanding | split)",
    )
    p_wf.add_argument("--symbol", required=True)
    p_wf.add_argument(
        "--mode", default="rolling", choices=["rolling", "expanding", "split"],
        help=(
            "Validation mode: 'rolling' (fixed train window slides forward), "
            "'expanding' (train window grows from origin), "
            "'split' (single time-based train/test cut). Default: rolling"
        ),
    )
    p_wf.add_argument(
        "--train-bars", dest="train_bars", type=int, default=10_000,
        help="Bars in each training window – rolling/expanding (default 10000)",
    )
    p_wf.add_argument(
        "--test-bars", dest="test_bars", type=int, default=2_000,
        help="Bars in each test window (default 2000)",
    )
    p_wf.add_argument(
        "--step-bars", dest="step_bars", type=int, default=None,
        help="Step between folds (default = test_bars → non-overlapping test windows)",
    )
    p_wf.add_argument(
        "--split-pct", dest="split_pct", type=float, default=0.8,
        help="Fraction of data used for training in split mode (default 0.8)",
    )
    p_wf.add_argument(
        "--trials", type=int, default=10,
        help="Optuna trials per fold – ignored when --no-optuna is set (default 10)",
    )
    p_wf.add_argument(
        "--no-optuna", dest="no_optuna", action="store_true",
        help="Skip Optuna; use fast fixed hyperparameters (recommended for many folds)",
    )
    p_wf.add_argument(
        "--select-by", dest="select_by", default="f1",
        choices=["f1", "trading"],
        help=(
            "Per-fold model selection objective: "
            "'f1' (macro-F1, default) or 'trading' (composite with four hard gates; "
            "confidence threshold co-selected per fold). "
            "Mirrors the --select-by flag on the train command."
        ),
    )
    p_wf.add_argument(
        "--profile", dest="profile", default=None, metavar="PROFILE_NAME",
        help=(
            "Apply named execution profile from config/profiles/<symbol>_profiles.yaml. "
            "Use 'production' to load the locked deployment artifact. "
            "Without --profile the universe.yaml defaults are used (generic walk-forward)."
        ),
    )
    p_wf.set_defaults(func=cmd_walk_forward)

    # ── forward-test ───────────────────────────────────────────────────────────
    p_ft = sub.add_parser(
        "forward-test",
        help=(
            "Validate the accepted production winner on new data. "
            "Requires a deployment artifact from evaluate-profiles."
        ),
    )
    p_ft.add_argument("--symbol", required=True)
    p_ft.add_argument(
        "--train-bars", dest="train_bars", type=int, default=10_000,
        help="Bars in each training window (default 10000)",
    )
    p_ft.add_argument(
        "--test-bars", dest="test_bars", type=int, default=2_000,
        help="Bars in each test window (default 2000)",
    )
    p_ft.add_argument(
        "--trials", type=int, default=10,
        help="Optuna trials per fold (default 10)",
    )
    p_ft.add_argument(
        "--no-optuna", dest="no_optuna", action="store_true",
        help="Skip Optuna; use fast fixed hyperparameters",
    )
    p_ft.add_argument(
        "--select-by", dest="select_by", default="f1",
        choices=["f1", "trading"],
        help="Per-fold model selection objective (default f1)",
    )
    p_ft.add_argument(
        "--promote", action="store_true",
        help="If verdict PASS, advance deployment state ACCEPTED → FORWARD_DEPLOYED",
    )
    p_ft.add_argument(
        "--seed", type=int, default=42,
        help=(
            "Master random seed matching the one used during evaluate-profiles "
            "(recorded in the deployment artifact). Default 42."
        ),
    )
    p_ft.set_defaults(func=cmd_forward_test)

    # ── evaluate-profiles ──────────────────────────────────────────────────────
    p_ep = sub.add_parser(
        "evaluate-profiles",
        help="Batch-evaluate all execution profiles and rank them",
    )
    p_ep.add_argument("--symbol", required=True, help="Symbol (must have processed data)")
    p_ep.add_argument(
        "--trials", type=int, default=20,
        help="Optuna trials per fold per profile (0 = fast fixed params, default 20)",
    )
    p_ep.add_argument(
        "--train-bars", dest="train_bars", type=int, default=10_000,
        help="Walk-forward training window in bars (default 10000)",
    )
    p_ep.add_argument(
        "--test-bars", dest="test_bars", type=int, default=2_000,
        help="Walk-forward test window in bars (default 2000)",
    )
    p_ep.add_argument(
        "--select-by", dest="select_by", default="f1",
        choices=["f1", "trading"],
        help="Per-fold model selection objective (default f1)",
    )
    p_ep.add_argument(
        "--no-backtest", dest="no_backtest", action="store_true",
        help="Skip standalone backtest per profile (default: skip)",
    )
    p_ep.add_argument(
        "--seed", type=int, default=42,
        help=(
            "Master random seed for Optuna TPESampler and XGBoost across all folds "
            "and profiles. Each profile gets seed + profile_idx*100000; each fold "
            "gets profile_seed + fold_idx*1000. Default 42."
        ),
    )
    p_ep.add_argument(
        "--acceptance-check-runs", dest="acceptance_check_runs", type=int, default=1,
        metavar="N",
        help=(
            "Reproducibility check: run the winner profile a second time with "
            "seed+1 and require it to also pass acceptance. "
            "1 = single run (default), 2 = reproducibility check required. "
            "Prevents stochastic optimizer drift from producing false positives."
        ),
    )
    p_ep.set_defaults(func=cmd_evaluate_profiles)

    # ── explain-signal ─────────────────────────────────────────────────────────
    p_exp = sub.add_parser(
        "explain-signal",
        help="Explain why the model fired its signal on the most recent bar",
    )
    p_exp.add_argument("--symbol", required=True)
    p_exp.add_argument("--input", required=True, help="Path to OHLCV CSV")
    p_exp.add_argument(
        "--equity", type=float, default=100_000.0,
        help="Account equity for position-size calculation (default $100,000)",
    )
    p_exp.set_defaults(func=cmd_explain_signal)

    # ── run-production ─────────────────────────────────────────────────────────
    p_rp = sub.add_parser(
        "run-production",
        help=(
            "Run the rules-first production engine with FundedGuard governance. "
            "Requires config/production_<SYMBOL>.yaml and a valid deployment artifact."
        ),
    )
    p_rp.add_argument("--symbol", required=True, help="Symbol, e.g. ES")
    p_rp.add_argument(
        "--input", required=True,
        help="Path to OHLCV CSV to stream (e.g. data/raw/ES_M1.csv)",
    )
    p_rp.add_argument(
        "--bar-delay", dest="bar_delay", type=float, default=0.0,
        help="Optional sleep between bars in seconds (0 = max speed, default 0)",
    )
    p_rp.set_defaults(func=cmd_run_production)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
