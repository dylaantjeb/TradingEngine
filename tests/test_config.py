"""
Unit tests for src/config.py
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import AppConfig, get_config, reload_config


_SAMPLE_UNIVERSE = {
    "symbols": ["ES", "NQ"],
    "execution": {"execution_delay_bars": 2},
    "cost_model": {
        "commission_per_side_usd": 2.50,
        "slippage_ticks_per_side": 1.0,
        "spread_ticks": 2.0,
    },
    "filters": {
        "session_start_utc_hour": 13,
        "session_end_utc_hour": 21,
        "atr_min_ticks": 5,
        "atr_max_ticks": 100,
        "news_blackout_windows": [["13:25", "13:35"]],
    },
    "throttles": {
        "max_trades_per_day": 10,
        "min_holding_bars": 5,
        "cooldown_bars_after_exit": 3,
    },
    "contract_specs": {
        "ES": {"multiplier": 50, "tick_size": 0.25, "tick_value": 12.50},
        "NQ": {"multiplier": 20, "tick_size": 0.25, "tick_value": 5.00},
    },
}


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestConfigLoad:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        reload_config.cache_clear() if hasattr(reload_config, "cache_clear") else None
        get_config.cache_clear()
        yield
        get_config.cache_clear()

    def test_loads_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            assert cfg.symbols == ["ES", "NQ"]

    def test_cost_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            assert cfg.cost.commission_per_side_usd == pytest.approx(2.50)
            assert cfg.cost.slippage_ticks_per_side == pytest.approx(1.0)

    def test_execution_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            assert cfg.execution.execution_delay_bars == 2

    def test_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            assert cfg.filters.session_start_utc_hour == 13
            assert cfg.filters.atr_min_ticks == pytest.approx(5.0)
            assert len(cfg.filters.news_blackout_windows) == 1

    def test_contract_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            es = cfg.get_contract("ES")
            assert es.multiplier == pytest.approx(50.0)
            assert es.tick_size  == pytest.approx(0.25)

    def test_missing_file_uses_defaults(self):
        cfg = get_config(universe_path="/nonexistent/universe.yaml")
        # Should not raise; uses defaults
        assert isinstance(cfg, AppConfig)
        assert cfg.execution.execution_delay_bars == 1

    def test_get_contract_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            # Unknown symbol → ES defaults
            spec = cfg.get_contract("UNKNOWN")
            assert spec.multiplier == pytest.approx(50.0)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TE_COMMISSION_PER_SIDE_USD", "5.00")
        get_config.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            u_path = Path(tmp) / "universe.yaml"
            _write_yaml(u_path, _SAMPLE_UNIVERSE)
            cfg = get_config(universe_path=str(u_path))
            assert cfg.cost.commission_per_side_usd == pytest.approx(5.00)
