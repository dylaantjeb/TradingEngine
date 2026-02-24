"""
TradingEngine CLI – single entry point.
Usage:  python -m src.cli <subcommand> [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
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

    # --use-rth: accept "true"/"false" strings or leave None for config default
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
    lbl_path = out_dir / f"{args.symbol}_labels.parquet"

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

    train(symbol=args.symbol, n_trials=args.trials)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run vectorised backtest and print metrics."""
    import yaml
    from src.backtest.engine import run_backtest

    with open(args.universe) as f:
        universe_cfg = yaml.safe_load(f)

    for symbol in universe_cfg.get("symbols", []):
        log.info("Backtesting %s …", symbol)
        metrics = run_backtest(symbol=symbol)
        print(f"\n{'='*60}")
        print(f"  {symbol} backtest results")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v}")
        print()


def cmd_live_paper(args: argparse.Namespace) -> None:
    """Simulate live paper trading by streaming rows from a CSV file."""
    from src.live.paper_engine import run_paper

    run_paper(symbol=args.symbol, csv_path=Path(args.input))


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
        help="Regular trading hours only? (default: from config, usually false for futures)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

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
    p_train.set_defaults(func=cmd_train)

    # ── backtest ───────────────────────────────────────────────────────────────
    p_bt = sub.add_parser("backtest", help="Vectorised backtest")
    p_bt.add_argument(
        "--universe",
        default="config/universe.yaml",
        help="Path to universe YAML (default config/universe.yaml)",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # ── live-paper ─────────────────────────────────────────────────────────────
    p_live = sub.add_parser("live-paper", help="Paper-trade simulation from CSV stream")
    p_live.add_argument("--symbol", required=True)
    p_live.add_argument("--input", required=True, help="Path to CSV to stream")
    p_live.set_defaults(func=cmd_live_paper)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
