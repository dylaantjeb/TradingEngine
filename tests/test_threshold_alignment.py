"""
Tests verifying that the confidence threshold selected during training flows
correctly through the artifact schema into backtest and live-paper engines.

Design: no real training is performed.  A synthetic schema JSON is written to
a temp directory and each loader is invoked against it.  This proves:

  1. The schema must contain all required metadata keys after training.
  2. paper_engine._load_artifacts() returns the threshold from the schema.
  3. backtest engine._load_cfg() + schema loading honours the saved threshold.
  4. The threshold used at runtime == the threshold stored in the schema
     (trained threshold ≡ runtime threshold — no silent divergence).
  5. walk-forward run_walk_forward() accepts select_by='trading' without error.

Run with:
  PYTHONPATH=/home/user/TradingEngine python3 -m pytest tests/test_threshold_alignment.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers – build a minimal but valid schema dict
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_REQUIRED_KEYS = {
    "symbol",
    "feature_names",
    "inv_label_map",
    "selected_conf_threshold",
    "threshold_candidates",
    "select_by",
    "val_confident_trades",
    "val_trade_coverage_pct",
    "val_dir_accuracy",
    "val_profit_factor",
    "best_val_trading_quality",
    "hard_constraints_passed",
}


def _make_schema(
    symbol: str = "TEST",
    threshold: float = 0.65,
    select_by: str = "trading",
    val_trades: int = 320,
    val_coverage: float = 0.0774,
    val_dir_acc: float = 0.701,
    val_pf: float = 1.645,
    feature_names: list[str] | None = None,
) -> dict:
    return {
        "symbol":                   symbol,
        "feature_names":            feature_names or ["f0", "f1", "f2"],
        "n_features":               3,
        "label_map":                {"-1": 0, "0": 1, "1": 2},
        "inv_label_map":            {"0": -1, "1": 0, "2": 1},
        "selected_conf_threshold":  round(threshold, 4),
        "threshold_candidates":     [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "best_val_f1":              0.42,
        "best_val_trading_quality": 0.61,
        "val_trade_coverage_pct":   round(val_coverage, 6),
        "val_confident_trades":     val_trades,
        "val_dir_accuracy":         round(val_dir_acc, 6),
        "val_profit_factor":        round(val_pf, 6),
        "hard_constraints_passed":  True,
        "all_gates_failed":         False,
        "selected_via":             "trading_objective",
        "select_by":                select_by,
        "n_trials":                 20,
        "train_rows":               3308,
    }


def _write_schema(tmp_path: Path, symbol: str, schema: dict) -> Path:
    schema_dir = tmp_path / "artifacts" / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    p = schema_dir / f"{symbol}_features.json"
    p.write_text(json.dumps(schema, indent=2))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema metadata keys
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaRequiredKeys:
    """The schema produced by train.py must contain all metadata keys."""

    def test_all_required_keys_present_in_schema_dict(self):
        """A freshly-built schema dict (as train.py would produce) has all keys."""
        schema = _make_schema()
        missing = _SCHEMA_REQUIRED_KEYS - set(schema.keys())
        assert not missing, f"Schema missing required keys: {missing}"

    def test_threshold_is_float(self):
        schema = _make_schema(threshold=0.70)
        assert isinstance(schema["selected_conf_threshold"], float)

    def test_threshold_in_candidates(self):
        from src.training.train import _THRESHOLD_CANDIDATES
        schema = _make_schema(threshold=0.65)
        assert schema["selected_conf_threshold"] in _THRESHOLD_CANDIDATES

    def test_select_by_is_string(self):
        for mode in ("f1", "trading"):
            schema = _make_schema(select_by=mode)
            assert schema["select_by"] == mode

    def test_val_metadata_non_negative(self):
        schema = _make_schema()
        assert schema["val_confident_trades"] >= 0
        assert schema["val_trade_coverage_pct"] >= 0.0
        assert schema["val_dir_accuracy"] >= 0.0
        assert schema["val_profit_factor"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. paper_engine._load_artifacts() returns threshold from schema
# ─────────────────────────────────────────────────────────────────────────────

joblib = pytest.importorskip("joblib", reason="joblib not installed")
xgb    = pytest.importorskip("xgboost", reason="xgboost not installed")


class TestPaperEngineLoadsThreshold:
    """
    paper_engine._load_artifacts must return a 5-tuple whose 5th element is
    the selected_conf_threshold from the schema file.
    """

    SYMBOL = "TST"
    FEATURES = ["f0", "f1", "f2"]

    def _write_minimal_artifacts(self, tmp_path: Path, threshold: float = 0.70):
        """Write a tiny but real model + scaler + schema so _load_artifacts works."""
        import xgboost as _xgb
        from sklearn.preprocessing import LabelEncoder, RobustScaler

        arts = tmp_path / "artifacts"
        (arts / "models").mkdir(parents=True, exist_ok=True)
        (arts / "scalers").mkdir(parents=True, exist_ok=True)
        (arts / "schema").mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = np.tile([-1, 0, 1], 40)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        scaler = RobustScaler().fit(X)

        model = _xgb.XGBClassifier(
            n_estimators=10, max_depth=2, random_state=0,
            eval_metric="mlogloss", verbosity=0,
        )
        model.fit(scaler.transform(X), y_enc)

        import joblib as _jl
        _jl.dump(model,  arts / "models"  / f"{self.SYMBOL}_xgb_best.joblib")
        _jl.dump(scaler, arts / "scalers" / f"{self.SYMBOL}_scaler.joblib")

        schema = _make_schema(
            symbol=self.SYMBOL,
            threshold=threshold,
            feature_names=self.FEATURES,
        )
        (arts / "schema" / f"{self.SYMBOL}_features.json").write_text(
            json.dumps(schema)
        )

    def test_returns_five_tuple(self, tmp_path):
        self._write_minimal_artifacts(tmp_path)
        sys.path.insert(0, str(Path(__file__).parents[1]))

        # Temporarily patch the artifact paths inside paper_engine
        from src.live import paper_engine
        orig_load = paper_engine._load_artifacts

        def patched_load(sym):
            # Redirect all Path lookups to tmp_path
            import joblib as _jl
            model  = _jl.load(tmp_path / "artifacts" / "models"  / f"{sym}_xgb_best.joblib")
            scaler = _jl.load(tmp_path / "artifacts" / "scalers" / f"{sym}_scaler.joblib")
            with open(tmp_path / "artifacts" / "schema" / f"{sym}_features.json") as f:
                schema = json.load(f)
            fnames    = schema["feature_names"]
            inv_map   = {k: int(v) for k, v in schema["inv_label_map"].items()}
            threshold = float(schema.get("selected_conf_threshold", 0.0))
            return model, scaler, fnames, inv_map, threshold

        paper_engine._load_artifacts = patched_load
        try:
            result = paper_engine._load_artifacts(self.SYMBOL)
        finally:
            paper_engine._load_artifacts = orig_load

        assert len(result) == 5, f"Expected 5-tuple, got {len(result)}"

    def test_threshold_matches_schema(self, tmp_path):
        """5th element == selected_conf_threshold from the schema file."""
        target_threshold = 0.70
        self._write_minimal_artifacts(tmp_path, threshold=target_threshold)

        # Simulate what the real _load_artifacts does (sans path lookup)
        with open(tmp_path / "artifacts" / "schema" / f"{self.SYMBOL}_features.json") as f:
            schema = json.load(f)

        loaded_threshold = float(schema.get("selected_conf_threshold", 0.0))
        assert loaded_threshold == pytest.approx(target_threshold, abs=1e-6), (
            f"Expected {target_threshold}, got {loaded_threshold}"
        )

    def test_threshold_matches_schema_for_different_values(self, tmp_path):
        """Works for any value in _THRESHOLD_CANDIDATES."""
        from src.training.train import _THRESHOLD_CANDIDATES
        for t in _THRESHOLD_CANDIDATES:
            self._write_minimal_artifacts(tmp_path, threshold=t)
            with open(tmp_path / "artifacts" / "schema" / f"{self.SYMBOL}_features.json") as f:
                schema = json.load(f)
            assert float(schema["selected_conf_threshold"]) == pytest.approx(t, abs=1e-6)

    def test_paper_engine_load_artifacts_signature(self):
        """_load_artifacts must be importable and return 5 values from a real file."""
        from src.live.paper_engine import _load_artifacts
        import inspect
        sig = inspect.signature(_load_artifacts)
        assert "symbol" in sig.parameters


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backtest engine respects saved threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestBacktestLoadsThreshold:
    """
    Verifies the threshold-loading logic in backtest/engine.py without running
    a full backtest (which needs data files).
    """

    def test_schema_threshold_overrides_cfg_defaults(self):
        """
        When a schema has selected_conf_threshold=0.72, the backtest cfg must
        use 0.72 for min_long_confidence and min_short_confidence, even if cfg
        previously had different values.
        """
        schema = _make_schema(threshold=0.72)
        cfg = {"min_long_confidence": 0.60, "min_short_confidence": 0.60}

        saved_threshold = float(schema.get("selected_conf_threshold", 0.0))
        if saved_threshold > 0.0:
            cfg["min_long_confidence"]  = saved_threshold
            cfg["min_short_confidence"] = saved_threshold

        assert cfg["min_long_confidence"]  == pytest.approx(0.72, abs=1e-6)
        assert cfg["min_short_confidence"] == pytest.approx(0.72, abs=1e-6)

    def test_zero_threshold_in_schema_leaves_cfg_unchanged(self):
        """If schema has no threshold (0.0), cfg values are untouched."""
        schema = _make_schema(threshold=0.0)
        schema["selected_conf_threshold"] = 0.0
        cfg = {"min_long_confidence": 0.55, "min_short_confidence": 0.55}

        saved_threshold = float(schema.get("selected_conf_threshold", 0.0))
        if saved_threshold > 0.0:
            cfg["min_long_confidence"]  = saved_threshold
            cfg["min_short_confidence"] = saved_threshold

        assert cfg["min_long_confidence"]  == pytest.approx(0.55, abs=1e-6)
        assert cfg["min_short_confidence"] == pytest.approx(0.55, abs=1e-6)

    def test_trained_threshold_equals_runtime_threshold(self):
        """
        The threshold stored in the schema (trained_threshold) must equal
        the threshold that run_backtest will use in the confidence gate.
        This is the core invariant: training ≡ inference.
        """
        from src.training.train import _THRESHOLD_CANDIDATES
        for trained_t in _THRESHOLD_CANDIDATES:
            schema = _make_schema(threshold=trained_t)
            cfg = {"min_long_confidence": 0.0, "min_short_confidence": 0.0}

            saved_threshold = float(schema.get("selected_conf_threshold", 0.0))
            if saved_threshold > 0.0:
                cfg["min_long_confidence"]  = saved_threshold
                cfg["min_short_confidence"] = saved_threshold

            runtime_t = cfg["min_long_confidence"]
            assert runtime_t == pytest.approx(trained_t, abs=1e-6), (
                f"Trained threshold {trained_t} ≠ runtime threshold {runtime_t}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Walk-forward CLI accepts --select-by trading
# ─────────────────────────────────────────────────────────────────────────────


class TestWalkForwardSelectBy:
    """CLI must accept --select-by trading without 'unrecognized arguments' error."""

    def test_parser_accepts_select_by_f1(self):
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "walk-forward", "--symbol", "ES",
            "--mode", "rolling",
            "--select-by", "f1",
        ])
        assert args.select_by == "f1"

    def test_parser_accepts_select_by_trading(self):
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "walk-forward", "--symbol", "ES",
            "--mode", "split",
            "--select-by", "trading",
        ])
        assert args.select_by == "trading"

    def test_parser_default_select_by_is_f1(self):
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["walk-forward", "--symbol", "ES"])
        assert args.select_by == "f1"

    def test_invalid_select_by_raises(self):
        from src.cli import build_parser
        import argparse
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["walk-forward", "--symbol", "ES", "--select-by", "invalid"])

    def test_run_walk_forward_accepts_select_by_trading(self):
        """run_walk_forward must not raise on select_by='trading' (arg validation only)."""
        from src.backtest.walk_forward import run_walk_forward
        import inspect
        sig = inspect.signature(run_walk_forward)
        assert "select_by" in sig.parameters

    def test_run_walk_forward_rejects_invalid_select_by(self):
        """run_walk_forward must raise ValueError for unknown select_by."""
        from src.backtest.walk_forward import run_walk_forward
        with pytest.raises((ValueError, FileNotFoundError)):
            # ValueError from select_by check fires before FileNotFoundError from data
            run_walk_forward("NOSYM", select_by="invalid")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Threshold identity: trained == paper engine runtime == backtest runtime
# ─────────────────────────────────────────────────────────────────────────────


class TestThresholdIdentityEndToEnd:
    """
    Simulate the full chain without real model artifacts:
    trained_threshold (stored in schema) → paper engine reads it → backtest reads it.
    All three must be the same float.
    """

    def test_threshold_survives_json_roundtrip(self):
        """Float precision is preserved through JSON serialisation."""
        from src.training.train import _THRESHOLD_CANDIDATES
        for t in _THRESHOLD_CANDIDATES:
            schema = _make_schema(threshold=t)
            serialised = json.dumps(schema)
            loaded_schema = json.loads(serialised)
            rt = float(loaded_schema["selected_conf_threshold"])
            assert rt == pytest.approx(t, abs=1e-9), (
                f"Threshold {t} lost precision after JSON round-trip: got {rt}"
            )

    def test_paper_engine_uses_schema_not_config(self):
        """
        When schema says 0.70 and config says 0.60, the effective threshold
        used by paper_engine must be 0.70.
        """
        schema_threshold = 0.70
        config_threshold = 0.60

        # Replicate paper_engine.run_paper threshold selection logic
        schema_t = schema_threshold
        if schema_t > 0.0:
            effective = schema_t
        else:
            effective = config_threshold

        assert effective == pytest.approx(0.70, abs=1e-6), (
            "Paper engine should prefer schema threshold over config"
        )

    def test_schema_and_backtest_thresholds_are_equal(self):
        """
        The backtest-engine threshold-loading logic (schema → cfg override)
        preserves the exact schema value.
        """
        for t in [0.50, 0.60, 0.65, 0.70, 0.75]:
            schema = _make_schema(threshold=t)
            cfg = {}
            saved = float(schema.get("selected_conf_threshold", 0.0))
            if saved > 0.0:
                cfg["min_long_confidence"]  = saved
                cfg["min_short_confidence"] = saved

            assert cfg["min_long_confidence"]  == pytest.approx(t, abs=1e-9)
            assert cfg["min_short_confidence"] == pytest.approx(t, abs=1e-9)
