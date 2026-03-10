"""
Centralised path constants for TradingEngine.

All paths are absolute and based on the repository root so they work
regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

# Repository root (two levels up from this file: src/utils/paths.py)
ROOT = Path(__file__).parent.parent.parent.resolve()

# Config
CONFIG_DIR          = ROOT / "config"
UNIVERSE_CFG        = CONFIG_DIR / "universe.yaml"
IBKR_CFG            = CONFIG_DIR / "ibkr.yaml"

# Raw + processed data
DATA_DIR            = ROOT / "data"
RAW_DIR             = DATA_DIR / "raw"
PROCESSED_DIR       = DATA_DIR / "processed"

# Artifacts
ARTIFACTS_DIR       = ROOT / "artifacts"
MODELS_DIR          = ARTIFACTS_DIR / "models"
SCALERS_DIR         = ARTIFACTS_DIR / "scalers"
SCHEMA_DIR          = ARTIFACTS_DIR / "schema"
REPORTS_DIR         = ARTIFACTS_DIR / "reports"

# Logs
LOGS_DIR            = ROOT / "logs"


# ──────────────────────────────────────────────────────────────────────────────
# Per-symbol helpers
# ──────────────────────────────────────────────────────────────────────────────


def raw_csv(symbol: str, bar: str = "M1") -> Path:
    """data/raw/<SYMBOL>_<BAR>.csv"""
    return RAW_DIR / f"{symbol}_{bar}.csv"


def features_parquet(symbol: str) -> Path:
    return PROCESSED_DIR / f"{symbol}_features.parquet"


def labels_parquet(symbol: str) -> Path:
    return PROCESSED_DIR / f"{symbol}_labels.parquet"


def model_path(symbol: str) -> Path:
    return MODELS_DIR / f"{symbol}_xgb_best.joblib"


def scaler_path(symbol: str) -> Path:
    return SCALERS_DIR / f"{symbol}_scaler.joblib"


def schema_path(symbol: str) -> Path:
    return SCHEMA_DIR / f"{symbol}_features.json"


def backtest_report(symbol: str) -> Path:
    return REPORTS_DIR / f"{symbol}_backtest.json"


# ──────────────────────────────────────────────────────────────────────────────
# Ensure directories exist on import
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all standard directories if they don't already exist."""
    for d in (
        RAW_DIR, PROCESSED_DIR,
        MODELS_DIR, SCALERS_DIR, SCHEMA_DIR, REPORTS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
