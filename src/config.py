"""
Unified configuration loader for TradingEngine.

Merges config/universe.yaml + config/ibkr.yaml, applies environment-variable
overrides (prefixed TE_), and exposes typed dataclass views.

Usage
-----
from src.config import get_config
cfg = get_config()
print(cfg.cost.commission_per_side_usd)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).parent.parent   # repo root


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs (typed dataclasses)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecutionConfig:
    execution_delay_bars: int = 1


@dataclass
class CostConfig:
    commission_per_side_usd: float = 1.50
    slippage_ticks_per_side: float = 0.5
    spread_ticks: float = 1.0


@dataclass
class FilterConfig:
    session_start_utc_hour: int = 9
    session_end_utc_hour: int = 22
    atr_min_ticks: float = 4.0
    atr_max_ticks: float = 200.0
    news_blackout_windows: List = field(default_factory=list)


@dataclass
class ThrottleConfig:
    max_trades_per_day: int = 20
    min_holding_bars: int = 3
    cooldown_bars_after_exit: int = 2
    max_concurrent_positions: int = 1


@dataclass
class ContractSpec:
    multiplier: float = 50.0
    tick_size: float = 0.25
    tick_value: float = 12.50


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    chunk_days: int = 5
    max_retries: int = 3
    base_backoff_sec: float = 5.0
    max_backoff_sec: float = 60.0
    request_timeout_sec: float = 120.0
    pacing_pause_sec: float = 2.0
    resume: bool = True
    use_rth: bool = False


@dataclass
class AppConfig:
    symbols: List[str] = field(default_factory=lambda: ["ES"])
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    throttles: ThrottleConfig = field(default_factory=ThrottleConfig)
    contract_specs: Dict[str, ContractSpec] = field(default_factory=dict)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)

    def get_contract(self, symbol: str) -> ContractSpec:
        """Return ContractSpec for symbol, falling back to ES defaults."""
        return self.contract_specs.get(symbol, ContractSpec())


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────


def _env(key: str, default: Any) -> Any:
    """Read environment variable TE_<KEY> (uppercase), cast to type of default."""
    raw = os.environ.get(f"TE_{key.upper()}")
    if raw is None:
        return default
    try:
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes")
        return type(default)(raw)
    except (ValueError, TypeError):
        return default


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_execution(u: dict) -> ExecutionConfig:
    sec = u.get("execution", {})
    return ExecutionConfig(
        execution_delay_bars=_env(
            "execution_delay_bars",
            int(sec.get("execution_delay_bars", 1)),
        ),
    )


def _build_cost(u: dict) -> CostConfig:
    sec = u.get("cost_model", {})
    return CostConfig(
        commission_per_side_usd=_env(
            "commission_per_side_usd",
            float(sec.get("commission_per_side_usd", 1.50)),
        ),
        slippage_ticks_per_side=_env(
            "slippage_ticks_per_side",
            float(sec.get("slippage_ticks_per_side", 0.5)),
        ),
        spread_ticks=_env(
            "spread_ticks",
            float(sec.get("spread_ticks", 1.0)),
        ),
    )


def _build_filters(u: dict) -> FilterConfig:
    sec = u.get("filters", {})
    return FilterConfig(
        session_start_utc_hour=_env(
            "session_start_utc_hour",
            int(sec.get("session_start_utc_hour", 9)),
        ),
        session_end_utc_hour=_env(
            "session_end_utc_hour",
            int(sec.get("session_end_utc_hour", 22)),
        ),
        atr_min_ticks=_env(
            "atr_min_ticks",
            float(sec.get("atr_min_ticks", 4.0)),
        ),
        atr_max_ticks=_env(
            "atr_max_ticks",
            float(sec.get("atr_max_ticks", 200.0)),
        ),
        news_blackout_windows=sec.get("news_blackout_windows", []),
    )


def _build_throttles(u: dict) -> ThrottleConfig:
    sec = u.get("throttles", {})
    return ThrottleConfig(
        max_trades_per_day=_env(
            "max_trades_per_day",
            int(sec.get("max_trades_per_day", 20)),
        ),
        min_holding_bars=_env(
            "min_holding_bars",
            int(sec.get("min_holding_bars", 3)),
        ),
        cooldown_bars_after_exit=_env(
            "cooldown_bars_after_exit",
            int(sec.get("cooldown_bars_after_exit", 2)),
        ),
        max_concurrent_positions=_env(
            "max_concurrent_positions",
            int(sec.get("max_concurrent_positions", 1)),
        ),
    )


def _build_contracts(u: dict) -> Dict[str, ContractSpec]:
    specs: Dict[str, ContractSpec] = {}
    for sym, raw in u.get("contract_specs", {}).items():
        specs[sym] = ContractSpec(
            multiplier=float(raw.get("multiplier", 50.0)),
            tick_size=float(raw.get("tick_size", 0.25)),
            tick_value=float(raw.get("tick_value", 12.50)),
        )
    return specs


def _build_ibkr(ib: dict) -> IBKRConfig:
    conn = ib.get("connection", {})
    fetch = ib.get("fetch", {})
    return IBKRConfig(
        host=_env("ibkr_host", str(conn.get("host", "127.0.0.1"))),
        port=_env("ibkr_port", int(conn.get("port", 7497))),
        client_id=_env("ibkr_client_id", int(conn.get("client_id", 1))),
        chunk_days=int(fetch.get("chunk_days", 5)),
        max_retries=int(fetch.get("max_retries", 3)),
        base_backoff_sec=float(fetch.get("base_backoff_sec", 5.0)),
        max_backoff_sec=float(fetch.get("max_backoff_sec", 60.0)),
        request_timeout_sec=float(fetch.get("request_timeout_sec", 120.0)),
        pacing_pause_sec=float(fetch.get("pacing_pause_sec", 2.0)),
        resume=bool(fetch.get("resume", True)),
        use_rth=bool(fetch.get("use_rth", False)),
    )


@lru_cache(maxsize=1)
def get_config(
    universe_path: str | None = None,
    ibkr_path: str | None = None,
) -> AppConfig:
    """
    Load and cache AppConfig.

    Parameters
    ----------
    universe_path : override path to universe.yaml
    ibkr_path     : override path to ibkr.yaml
    """
    u_path = Path(universe_path) if universe_path else _ROOT / "config" / "universe.yaml"
    i_path = Path(ibkr_path) if ibkr_path else _ROOT / "config" / "ibkr.yaml"

    u = _load_yaml(u_path)
    ib = _load_yaml(i_path)

    return AppConfig(
        symbols=u.get("symbols", ["ES"]),
        execution=_build_execution(u),
        cost=_build_cost(u),
        filters=_build_filters(u),
        throttles=_build_throttles(u),
        contract_specs=_build_contracts(u),
        ibkr=_build_ibkr(ib),
    )


def reload_config(
    universe_path: str | None = None,
    ibkr_path: str | None = None,
) -> AppConfig:
    """Force reload (clears lru_cache)."""
    get_config.cache_clear()
    return get_config(universe_path, ibkr_path)
