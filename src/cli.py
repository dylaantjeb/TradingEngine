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


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run vectorised backtest and print metrics."""
    import yaml
    from src.backtest.engine import run_backtest

    with open(args.universe) as f:
        universe_cfg = yaml.safe_load(f)

    for symbol in universe_cfg.get("symbols", []):
        log.info("Backtesting %s …", symbol)
        run_backtest(
            symbol=symbol,
            execution_delay_bars=args.execution_delay,
            max_trades_per_day=args.max_trades_per_day,
            slippage_ticks_per_side=args.slippage_ticks,
            commission_per_side_usd=args.commission,
        )


def cmd_live_paper(args: argparse.Namespace) -> None:
    """Simulate live paper trading by streaming rows from a CSV file."""
    from src.live.paper_engine import run_paper

    run_paper(symbol=args.symbol, csv_path=Path(args.input))


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run walk-forward / OOS validation."""
    from src.backtest.walk_forward import run_walk_forward

    n_trials = 0 if getattr(args, "no_optuna", False) else args.trials

    run_walk_forward(
        symbol=args.symbol,
        mode=args.mode,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        split_pct=args.split_pct,
        n_trials=n_trials,
        save_report=True,
    )


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
    p_bt.set_defaults(func=cmd_backtest)

    # ── live-paper ─────────────────────────────────────────────────────────────
    p_live = sub.add_parser("live-paper", help="Paper-trade simulation from CSV stream")
    p_live.add_argument("--symbol", required=True)
    p_live.add_argument("--input", required=True, help="Path to CSV to stream")
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
    p_wf.set_defaults(func=cmd_walk_forward)

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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
