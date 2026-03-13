"""
Production deployment module for TradingEngine.

Manages the lifecycle of a production-ready strategy deployment:

  RESEARCH → CANDIDATE → ACCEPTED → FORWARD_DEPLOYED → LIVE_SMALL → LIVE_SCALED → HALTED

The deployment artifact (artifacts/production/<symbol>_deployment.json) is the
single source of truth for which profile + config is in production. It is written
by ``evaluate-profiles`` when a profile passes all acceptance gates, and is read
by ``forward-test``, ``walk-forward --profile``, ``backtest --profile``, and
``live-paper --profile``.

NON-NEGOTIABLE PRODUCTION RULES
--------------------------------
1. ``forward-test`` and ``live-paper`` REFUSE to start without a valid artifact.
2. The deployment state must be ACCEPTED or higher before production commands run.
3. Config hash in the artifact must match the overrides dict on every load.
4. ``evaluate-profiles`` is the ONLY command that can create a new artifact.
5. State transitions (→ FORWARD_DEPLOYED, → HALTED, etc.) are the only manual edits.
6. The ``early_session_only`` profile is the accepted winner; other profiles remain
   available for research but are not promoted unless evaluate-profiles re-runs.

Usage
-----
  from src.deployment import load_or_fail, run_health_checks, DeploymentState
  artifact = load_or_fail("ES")
  overrides = artifact["exec_cfg_overrides"]
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_PRODUCTION_DIR = Path("artifacts/production")


# ── Deployment state machine ──────────────────────────────────────────────────

class DeploymentState(str, Enum):
    RESEARCH         = "RESEARCH"
    CANDIDATE        = "CANDIDATE"
    ACCEPTED         = "ACCEPTED"
    FORWARD_DEPLOYED = "FORWARD_DEPLOYED"
    LIVE_SMALL       = "LIVE_SMALL"
    LIVE_SCALED      = "LIVE_SCALED"
    HALTED           = "HALTED"

    def is_deployable(self) -> bool:
        """True when the state allows live/forward-test execution."""
        return self in (
            DeploymentState.ACCEPTED,
            DeploymentState.FORWARD_DEPLOYED,
            DeploymentState.LIVE_SMALL,
            DeploymentState.LIVE_SCALED,
        )

    @classmethod
    def _missing_(cls, value: object) -> Optional["DeploymentState"]:
        return None


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def _deployment_path(symbol: str) -> Path:
    return _PRODUCTION_DIR / f"{symbol}_deployment.json"


def _hash_cfg(cfg: dict) -> str:
    """Stable 16-char SHA-256 hex of sorted JSON config."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_deployment_artifact(
    symbol: str,
    profile_name: str,
    profile_config: dict,
    exec_cfg_overrides: dict,
    metrics: dict,
    state: DeploymentState = DeploymentState.ACCEPTED,
) -> Path:
    """
    Write (or overwrite) the production deployment artifact for ``symbol``.

    Called by ``profile_eval.save_artifacts()`` when a winner is found.
    Human operators may later call ``update_deployment_state()`` to advance
    the state through the lifecycle.
    """
    _PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    path = _deployment_path(symbol)

    artifact = {
        "symbol":              symbol,
        "profile_name":        profile_name,
        "deployment_state":    state.value,
        "accepted_at":         datetime.now(timezone.utc).isoformat(),
        "profile_config":      profile_config,
        "exec_cfg_overrides":  exec_cfg_overrides,
        "metrics":             metrics,
        "config_hash":         _hash_cfg(exec_cfg_overrides),
    }
    path.write_text(json.dumps(artifact, indent=2, default=str))
    log.info("Deployment artifact written → %s", path)
    return path


def load_deployment_artifact(symbol: str) -> Optional[dict]:
    """Return the deployment artifact dict, or None if it doesn't exist."""
    path = _deployment_path(symbol)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_or_fail(symbol: str) -> dict:
    """
    Load the deployment artifact or exit(1) with a clear remediation message.

    Use this in any command that must NOT run without a validated winner.
    """
    artifact = load_deployment_artifact(symbol)
    if artifact is None:
        log.error(
            "No production deployment artifact found for %s.\n"
            "  Expected: %s\n"
            "  Run:  python -m src.cli evaluate-profiles --symbol %s\n"
            "  The artifact is written when a profile passes all acceptance gates.",
            symbol, _deployment_path(symbol), symbol,
        )
        sys.exit(1)
    return artifact


def get_exec_cfg_overrides(symbol: str) -> dict:
    """Return exec_cfg_overrides from the production artifact (exits on missing)."""
    return load_or_fail(symbol).get("exec_cfg_overrides", {})


def get_threshold_candidates(symbol: str) -> Optional[list]:
    """Return threshold_candidates from the production artifact (None if absent)."""
    artifact = load_or_fail(symbol)
    return artifact.get("profile_config", {}).get("threshold_candidates", None)


def update_deployment_state(symbol: str, new_state: DeploymentState) -> None:
    """
    Advance the deployment state in the artifact.

    This is the only mechanism to promote a strategy from ACCEPTED to
    FORWARD_DEPLOYED / LIVE_SMALL / LIVE_SCALED, or to HALT it.
    """
    artifact = load_or_fail(symbol)
    old_state = artifact.get("deployment_state", "RESEARCH")
    artifact["deployment_state"] = new_state.value
    artifact["state_updated_at"] = datetime.now(timezone.utc).isoformat()
    _deployment_path(symbol).write_text(json.dumps(artifact, indent=2, default=str))
    log.info(
        "[%s] Deployment state: %s → %s",
        symbol, old_state, new_state.value,
    )


# ── Health gates ──────────────────────────────────────────────────────────────

def run_health_checks(
    symbol: str,
    cfg: dict,
    *,
    require_artifact: bool = True,
) -> list[str]:
    """
    Run pre-trade health checks. Returns a list of FAILED check descriptions.
    An empty return list means ALL checks passed.

    Parameters
    ----------
    symbol           : Trading symbol (ES, NQ, …).
    cfg              : Merged run-config dict (universe.yaml + profile overrides).
    require_artifact : If True (default), fail if no production artifact exists.
    """
    failures: list[str] = []

    # ── Check 1: Model artifact ───────────────────────────────────────────────
    model_path = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    if not model_path.exists():
        failures.append(f"MODEL_MISSING: {model_path}")

    # ── Check 2: Scaler artifact ──────────────────────────────────────────────
    scaler_path = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    if not scaler_path.exists():
        failures.append(f"SCALER_MISSING: {scaler_path}")

    # ── Check 3: Schema artifact + feature names + confidence threshold ───────
    schema_path = Path(f"artifacts/schema/{symbol}_features.json")
    if not schema_path.exists():
        failures.append(f"SCHEMA_MISSING: {schema_path}")
    else:
        try:
            with open(schema_path) as f:
                schema = json.load(f)
            if not schema.get("feature_names"):
                failures.append("SCHEMA_EMPTY: feature_names list is empty")
            conf_threshold = float(schema.get("selected_conf_threshold", 0.0))
            if conf_threshold <= 0.0:
                failures.append(
                    "CONF_THRESHOLD_UNSET: schema has no selected_conf_threshold > 0 "
                    "(retrain with --select-by trading)"
                )
        except Exception as exc:
            failures.append(f"SCHEMA_PARSE_ERROR: {exc}")

    # ── Check 4: Production deployment artifact + state + hash ───────────────
    if require_artifact:
        artifact = load_deployment_artifact(symbol)
        if artifact is None:
            failures.append(
                f"NO_DEPLOYMENT_ARTIFACT: run 'evaluate-profiles --symbol {symbol}'"
            )
        else:
            state_str = artifact.get("deployment_state", "RESEARCH")
            state = DeploymentState(state_str) if DeploymentState(state_str) else None
            if state is None:
                failures.append(f"DEPLOYMENT_STATE_UNKNOWN: '{state_str}'")
            elif not state.is_deployable():
                failures.append(
                    f"DEPLOYMENT_STATE_NOT_DEPLOYABLE: state={state.value} "
                    "(must be ACCEPTED / FORWARD_DEPLOYED / LIVE_SMALL / LIVE_SCALED)"
                )
            stored_hash = artifact.get("config_hash", "")
            overrides   = artifact.get("exec_cfg_overrides", {})
            if stored_hash != _hash_cfg(overrides):
                failures.append(
                    f"CONFIG_HASH_MISMATCH: artifact may have been manually edited "
                    f"(stored={stored_hash})"
                )

    # ── Check 5: Session blocks defined ──────────────────────────────────────
    sess_blocks = cfg.get("session_blocks", [])
    if not sess_blocks:
        failures.append("SESSION_BLOCKS_EMPTY: no session_blocks defined in config")

    # ── Check 6: ATR limits sensible ─────────────────────────────────────────
    atr_min = float(cfg.get("atr_min_ticks", 0))
    atr_max = float(cfg.get("atr_max_ticks", 1e9))
    if atr_min >= atr_max:
        failures.append(f"ATR_RANGE_INVALID: atr_min={atr_min} >= atr_max={atr_max}")
    if atr_max > 500:
        failures.append(f"ATR_MAX_EXCESSIVE: atr_max={atr_max} > 500 (check config)")

    # ── Check 7: Risk limits present ─────────────────────────────────────────
    if float(cfg.get("max_total_drawdown_usd", 0)) <= 0:
        failures.append("RISK_LIMITS_MISSING: max_total_drawdown_usd not set or zero")

    # ── Check 8: Confidence threshold not dangerously low ────────────────────
    min_long_conf = float(cfg.get("min_long_confidence", 0.0))
    if min_long_conf < 0.50 and float(cfg.get("selected_conf_threshold", min_long_conf)) < 0.50:
        failures.append(
            f"CONF_THRESHOLD_LOW: min_long_confidence={min_long_conf:.2f} < 0.50 "
            "(funded-account standard is ≥ 0.60)"
        )

    # ── Check 9: Max trades per day reasonable ───────────────────────────────
    max_tpd = int(cfg.get("max_trades_per_day", 9999))
    if max_tpd > 20:
        failures.append(
            f"MAX_TRADES_EXCESSIVE: max_trades_per_day={max_tpd} > 20 "
            "(overtrading risk in funded account)"
        )

    # ── Check 10: Trend filter enabled ───────────────────────────────────────
    if not bool(cfg.get("trend_filter_enabled", False)):
        failures.append(
            "TREND_FILTER_DISABLED: trend_filter_enabled=false — "
            "regime filter is required for production"
        )

    # ── Check 11: Execution delay is non-zero ────────────────────────────────
    if int(cfg.get("execution_delay_bars", 0)) == 0:
        failures.append(
            "EXECUTION_DELAY_ZERO: execution_delay_bars=0 → same-bar fills "
            "(leaky / debug mode only)"
        )

    # ── Check 12: Max consecutive losses set ─────────────────────────────────
    max_consec = int(cfg.get("max_consecutive_losses", 9999))
    if max_consec > 10:
        failures.append(
            f"MAX_CONSEC_LOSSES_HIGH: max_consecutive_losses={max_consec} > 10 "
            "(should be ≤ 5 for funded-account safety)"
        )

    return failures


def assert_health_or_abort(
    symbol: str,
    cfg: dict,
    *,
    require_artifact: bool = True,
    mode: str = "production",
) -> None:
    """
    Run health checks and abort (sys.exit(1)) if any fail.

    Parameters
    ----------
    mode : Label used in the log message ('production', 'forward-test', etc.).
    """
    failures = run_health_checks(symbol, cfg, require_artifact=require_artifact)
    if failures:
        log.error(
            "[%s] %s health check FAILED — %d issue(s):\n  %s",
            symbol, mode, len(failures), "\n  ".join(failures),
        )
        sys.exit(1)
    log.info("[%s] %s health check PASSED (%d checks)", symbol, mode, 12)
