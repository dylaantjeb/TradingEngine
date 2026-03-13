"""
Profile-based strategy evaluation for TradingEngine.

Replaces manual trial-and-error tuning with a deterministic framework:

  1. Load all profiles from config/profiles/<SYMBOL>_profiles.yaml.
  2. For each profile, run walk-forward with profile-specific exec overrides.
  3. Score every profile on a rich set of metrics.
  4. Apply hard acceptance gates to determine ACCEPTED vs REJECTED profiles.
  5. Rank accepted profiles by composite score; rank rejected last.
  6. Declare winner (rank-1 ACCEPTED) or "REJECT STRATEGY – no deployable
     profile found" if none pass.
  7. Save artifacts/reports/<SYMBOL>_profiles_summary.json + .csv.

CLI entry-point
---------------
  python -m src.cli evaluate-profiles --symbol ES --trials 50

Acceptance gates (ALL must pass)
---------------------------------
  • profitable_folds    ≥ 3   (in a 5-fold walk-forward)
  • avg_pf              ≥ 1.50
  • cv                  ≤ 1.0
  • t_stat              ≥ 1.65
  • avg_expectancy      > 0
  • block2_not_dragging : block2_total_pnl ≥ −100  OR  block2_n_trades == 0
  • trend_profitable_folds ≥ 3
  • chop_dominated_folds   ≤ 2

Composite acceptance score (higher is better, used for ranking)
----------------------------------------------------------------
  0.40 × norm(avg_pf)   +  0.30 × norm(t_stat)
+ 0.20 × norm(avg_sharpe) + 0.10 × (1 − norm(cv))
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

# ── Acceptance thresholds ──────────────────────────────────────────────────────
_ACCEPT_MIN_PROFITABLE_FOLDS  = 3
_ACCEPT_MIN_AVG_PF            = 1.50
_ACCEPT_MAX_CV                = 1.0
_ACCEPT_MIN_TSTAT             = 1.65
_ACCEPT_MIN_AVG_EXPECTANCY    = 0.0
_ACCEPT_MIN_TREND_FOLDS       = 3
_ACCEPT_MAX_CHOP_DOMINATED    = 2

# Composite score weights
_W_PF     = 0.40
_W_TSTAT  = 0.30
_W_SHARPE = 0.20
_W_CV     = 0.10


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ProfileScore:
    profile_name:          str
    description:           str

    # Walk-forward aggregate metrics
    n_folds:               int   = 0
    profitable_folds:      int   = 0
    pct_profitable:        float = 0.0
    avg_pf:                float = 0.0
    median_pf:             float = 0.0
    pnl_cv:                float = 0.0
    t_stat:                float = 0.0
    avg_sharpe:            float = 0.0
    max_drawdown_usd:      float = 0.0
    avg_expectancy:        float = 0.0
    trades_per_day:        float = 0.0
    total_pnl:             float = 0.0

    # Session block attribution
    block1_total_pnl:      float = 0.0
    block2_total_pnl:      float = 0.0
    block1_n_trades:       int   = 0
    block2_n_trades:       int   = 0

    # Regime / fold quality
    trend_profitable_folds: int  = 0
    chop_dominated_folds:  int   = 0
    weak_folds:            int   = 0

    # Acceptance
    accepted:              bool  = False
    rejection_reasons:     list  = field(default_factory=list)
    composite_score:       float = 0.0
    rank:                  int   = 0

    # Raw walk-forward summary (not serialised to CSV)
    _wf_summary:           Any   = field(default=None, repr=False)


# ── Profile loading ────────────────────────────────────────────────────────────

def load_profiles(symbol: str) -> dict[str, dict]:
    """Load profiles from config/profiles/<symbol>_profiles.yaml."""
    profiles_path = Path(f"config/profiles/{symbol}_profiles.yaml")
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"No profiles file found: {profiles_path}\n"
            f"Create it with the profile definitions for {symbol}."
        )
    with open(profiles_path) as f:
        raw = yaml.safe_load(f)
    profiles = raw.get("profiles", {})
    if not profiles:
        raise ValueError(f"No profiles defined in {profiles_path}")
    log.info("Loaded %d profile(s) for %s from %s", len(profiles), symbol, profiles_path)
    return profiles


# ── cfg overrides builder ─────────────────────────────────────────────────────

def _profile_to_cfg_overrides(profile: dict) -> dict:
    """Convert a profile dict to engine cfg_overrides keys."""
    overrides: dict = {}
    _map = {
        "atr_min_ticks":            "atr_min_ticks",
        "atr_max_ticks":            "atr_max_ticks",
        "trend_filter_enabled":     "trend_filter_enabled",
        "trend_filter_ema_period":  "trend_filter_ema_period",
        "trend_slope_min_atr_frac": "trend_slope_min_atr_frac",
        "cooldown_bars_after_exit": "cooldown_bars_after_exit",
        "cooldown_bars_after_loss": "cooldown_bars_after_loss",
        "max_trades_per_day":       "max_trades_per_day",
        "min_holding_bars":         "min_holding_bars",
        "session_blocks":           "session_blocks",
    }
    for profile_key, cfg_key in _map.items():
        if profile_key in profile:
            overrides[cfg_key] = profile[profile_key]
    return overrides


# ── Per-profile evaluation ────────────────────────────────────────────────────

def _evaluate_profile(
    symbol: str,
    profile_name: str,
    profile: dict,
    n_trials: int,
    select_by: str,
    train_bars: int,
    test_bars: int,
    run_backtest: bool,
) -> ProfileScore:
    """Run walk-forward for one profile and return a ProfileScore."""
    from src.backtest.walk_forward import run_walk_forward

    description = profile.get("description", "")
    if isinstance(description, str):
        description = description.strip().replace("\n", " ")

    threshold_candidates = profile.get("threshold_candidates", None)
    exec_overrides       = _profile_to_cfg_overrides(profile)

    log.info(
        "Evaluating profile [%s] – threshold_candidates=%s  overrides=%s",
        profile_name, threshold_candidates, list(exec_overrides.keys()),
    )

    try:
        wf = run_walk_forward(
            symbol=symbol,
            mode="rolling",
            train_bars=train_bars,
            test_bars=test_bars,
            n_trials=n_trials,
            select_by=select_by,
            save_report=False,
            exec_cfg_overrides=exec_overrides,
            threshold_candidates=threshold_candidates,
        )
    except Exception as exc:
        log.error("Profile [%s] walk-forward failed: %s", profile_name, exc)
        ps = ProfileScore(
            profile_name=profile_name,
            description=description,
            rejection_reasons=[f"Walk-forward error: {exc}"],
        )
        return ps

    agg = wf.aggregate if hasattr(wf, "aggregate") else {}
    folds = wf.folds if hasattr(wf, "folds") else []

    # ── Collect fold metrics ──────────────────────────────────────────────────
    n_folds         = len(folds)
    profitable_folds = sum(1 for f in folds if f.profitable)
    pf_values       = [f.profit_factor for f in folds]
    pnl_values      = [f.net_pnl for f in folds]
    sharpe_values   = [f.sharpe for f in folds if f.sharpe and not math.isnan(f.sharpe)]
    exp_values      = [f.expectancy_usd for f in folds]
    tpd_values      = [f.trades_per_day for f in folds]
    dd_values       = [f.max_drawdown_usd for f in folds]

    avg_pf      = float(sum(pf_values) / len(pf_values)) if pf_values else 0.0
    median_pf   = float(sorted(pf_values)[len(pf_values) // 2]) if pf_values else 0.0
    avg_sharpe  = float(sum(sharpe_values) / len(sharpe_values)) if sharpe_values else 0.0
    avg_exp     = float(sum(exp_values) / len(exp_values)) if exp_values else 0.0
    avg_tpd     = float(sum(tpd_values) / len(tpd_values)) if tpd_values else 0.0
    max_dd      = max(dd_values, default=0.0)
    total_pnl   = sum(pnl_values)

    # CV and t-stat: prefer aggregate pre-computed value, fall back to local calc.
    # Must use _coalesce_metric — both keys can exist in the dict with value None:
    #   "pnl_cv"     → None when mean_pnl <= 0  (walk_forward._aggregate line ~1057)
    #   "t_stat_pnl" → None when n < 2 or std == 0  (walk_forward._aggregate line ~1062)
    # dict.get(key, default) only uses the default when the key is absent; a
    # present-but-None value passes straight through and crashes float(None).
    # Note: the aggregate key is "t_stat_pnl", not "t_stat".
    pnl_cv = _coalesce_metric(agg.get("pnl_cv"),      _safe_cv(pnl_values))
    t_stat = _coalesce_metric(agg.get("t_stat_pnl"),  _safe_tstat(pnl_values))
    pct_prof = profitable_folds / n_folds if n_folds else 0.0

    # Session block totals
    b1_pnl    = sum(f.block1_net_pnl  for f in folds)
    b2_pnl    = sum(f.block2_net_pnl  for f in folds)
    b1_trades = sum(f.block1_n_trades for f in folds)
    b2_trades = sum(f.block2_n_trades for f in folds)

    # Regime / fold quality
    # Uses the same 2× multiplier as walk_forward._aggregate (line ~1111):
    #   n_chop_dominated counts a fold only when chop blocks exceed TWICE the
    #   trend entries, matching the "regime blocked >> trend entries" intent.
    # The previous formula used 1× and was stricter than the actual WF gate,
    # causing false rejections when the WF gate itself would have passed.
    trend_profitable = sum(
        1 for f in folds if f.n_trend_entries > 0 and f.profitable
    )
    chop_dominated = sum(
        1 for f in folds if f.n_chop_blocked > 2 * max(f.n_trend_entries, 1)
    )
    weak_folds = sum(1 for f in folds if f.weak_fold)

    ps = ProfileScore(
        profile_name=profile_name,
        description=description,
        n_folds=n_folds,
        profitable_folds=profitable_folds,
        pct_profitable=pct_prof,
        avg_pf=avg_pf,
        median_pf=median_pf,
        pnl_cv=pnl_cv,
        t_stat=t_stat,
        avg_sharpe=avg_sharpe,
        max_drawdown_usd=max_dd,
        avg_expectancy=avg_exp,
        trades_per_day=avg_tpd,
        total_pnl=total_pnl,
        block1_total_pnl=b1_pnl,
        block2_total_pnl=b2_pnl,
        block1_n_trades=b1_trades,
        block2_n_trades=b2_trades,
        trend_profitable_folds=trend_profitable,
        chop_dominated_folds=chop_dominated,
        weak_folds=weak_folds,
        _wf_summary=wf,
    )

    _apply_acceptance(ps)
    return ps


# ── Acceptance gate ───────────────────────────────────────────────────────────

def _apply_acceptance(ps: ProfileScore) -> None:
    """Populate ps.accepted and ps.rejection_reasons in-place."""
    reasons: list[str] = []

    if ps.profitable_folds < _ACCEPT_MIN_PROFITABLE_FOLDS:
        reasons.append(
            f"profitable_folds={ps.profitable_folds} < {_ACCEPT_MIN_PROFITABLE_FOLDS}"
        )
    if ps.avg_pf < _ACCEPT_MIN_AVG_PF:
        reasons.append(f"avg_pf={ps.avg_pf:.2f} < {_ACCEPT_MIN_AVG_PF}")
    if ps.pnl_cv > _ACCEPT_MAX_CV:
        reasons.append(f"cv={ps.pnl_cv:.2f} > {_ACCEPT_MAX_CV}")
    if ps.t_stat < _ACCEPT_MIN_TSTAT:
        reasons.append(f"t_stat={ps.t_stat:.2f} < {_ACCEPT_MIN_TSTAT}")
    if ps.avg_expectancy <= _ACCEPT_MIN_AVG_EXPECTANCY:
        reasons.append(f"avg_expectancy=${ps.avg_expectancy:.0f} ≤ 0")

    # Session block dragging: block2 strongly negative while block1 carries PnL
    if ps.block2_n_trades > 0 and ps.block1_total_pnl > 100 and ps.block2_total_pnl < -100:
        reasons.append(
            f"block2 dragging: block2_pnl=${ps.block2_total_pnl:.0f} "
            f"while block1_pnl=${ps.block1_total_pnl:.0f}"
        )

    if ps.trend_profitable_folds < _ACCEPT_MIN_TREND_FOLDS:
        reasons.append(
            f"trend_profitable_folds={ps.trend_profitable_folds} < {_ACCEPT_MIN_TREND_FOLDS}"
        )
    if ps.chop_dominated_folds > _ACCEPT_MAX_CHOP_DOMINATED:
        reasons.append(
            f"chop_dominated_folds={ps.chop_dominated_folds} > {_ACCEPT_MAX_CHOP_DOMINATED}"
        )

    ps.rejection_reasons = reasons
    ps.accepted = len(reasons) == 0

    if ps.accepted:
        ps.composite_score = _composite_score(ps)


def _composite_score(ps: ProfileScore) -> float:
    """Compute a composite score in [0, 1] range for ranking accepted profiles."""
    # Normalise each metric against practical bounds
    pf_score  = min(max((ps.avg_pf   - 1.0) / 2.0,  0.0), 1.0)  # 1.0→0, 3.0→1
    ts_score  = min(max((ps.t_stat   - 1.65) / 3.35, 0.0), 1.0)  # 1.65→0, 5.0→1
    sh_score  = min(max((ps.avg_sharpe) / 3.0,        0.0), 1.0)  # 0→0, 3.0→1
    cv_score  = min(max(1.0 - ps.pnl_cv,               0.0), 1.0)  # 0→1, 1.0→0

    return (
        _W_PF     * pf_score
        + _W_TSTAT  * ts_score
        + _W_SHARPE * sh_score
        + _W_CV     * cv_score
    )


# ── Ranking ───────────────────────────────────────────────────────────────────

def _rank_profiles(results: list[ProfileScore]) -> list[ProfileScore]:
    """Sort: accepted first (desc composite_score), then rejected (desc avg_pf)."""
    accepted = sorted(
        [r for r in results if r.accepted],
        key=lambda r: r.composite_score,
        reverse=True,
    )
    rejected = sorted(
        [r for r in results if not r.accepted],
        key=lambda r: r.avg_pf,
        reverse=True,
    )
    ranked = accepted + rejected
    for i, r in enumerate(ranked):
        r.rank = i + 1
    return ranked


def pick_winner(results: list[ProfileScore]) -> Optional[ProfileScore]:
    """Return the rank-1 accepted profile, or None if none pass all gates."""
    accepted = [r for r in results if r.accepted]
    if not accepted:
        return None
    return min(accepted, key=lambda r: r.rank)


# ── Output ────────────────────────────────────────────────────────────────────

def print_scoreboard(results: list[ProfileScore], symbol: str) -> None:
    """Print formatted scoreboard to stdout."""
    ranked = sorted(results, key=lambda r: r.rank)

    print(f"\n{'='*90}")
    print(f"  PROFILE EVALUATION SCOREBOARD — {symbol}")
    print(f"{'='*90}")
    print(
        f"  {'Rk':>2}  {'Profile':<26}  {'OK':>2}  "
        f"{'PrftFolds':>9}  {'AvgPF':>6}  {'CV':>5}  {'t-stat':>6}  "
        f"{'Shrp':>5}  {'Exp$':>7}  {'TotPnL':>8}"
    )
    print(f"  {'-'*86}")

    for r in ranked:
        ok    = "✓" if r.accepted else "✗"
        print(
            f"  {r.rank:>2}  {r.profile_name:<26}  {ok:>2}  "
            f"  {r.profitable_folds}/{r.n_folds} ({r.pct_profitable:>4.0%})  "
            f"{r.avg_pf:>6.2f}  {r.pnl_cv:>5.2f}  {r.t_stat:>6.2f}  "
            f"{r.avg_sharpe:>5.2f}  {r.avg_expectancy:>7.0f}  {r.total_pnl:>8.0f}"
        )
        if not r.accepted:
            for reason in r.rejection_reasons:
                print(f"      ✗ {reason}")

    print(f"{'='*90}")

    winner = pick_winner(ranked)
    if winner:
        print(f"\n  WINNER: [{winner.profile_name}]  composite_score={winner.composite_score:.3f}")
        print(f"  Strategy is READY FOR DEPLOYMENT.\n")
    else:
        print(f"\n  REJECT STRATEGY — no deployable profile found.\n")
        print(f"  All {len(results)} profiles failed at least one acceptance gate.\n")


def save_artifacts(symbol: str, results: list[ProfileScore]) -> None:
    """Write JSON + CSV summaries to artifacts/reports/ and (if winner) production artifact."""
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(results, key=lambda r: r.rank)

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = out_dir / f"{symbol}_profiles_summary.json"
    json_data = []
    for r in ranked:
        d = asdict(r)
        d.pop("_wf_summary", None)   # not serialisable
        json_data.append(d)
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    log.info("Profile summary (JSON) → %s", json_path)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = out_dir / f"{symbol}_profiles_summary.csv"
    _CSV_FIELDS = [
        "rank", "profile_name", "accepted", "composite_score",
        "n_folds", "profitable_folds", "pct_profitable",
        "avg_pf", "median_pf", "pnl_cv", "t_stat", "avg_sharpe",
        "max_drawdown_usd", "avg_expectancy", "trades_per_day", "total_pnl",
        "block1_total_pnl", "block2_total_pnl", "block1_n_trades", "block2_n_trades",
        "trend_profitable_folds", "chop_dominated_folds", "weak_folds",
        "rejection_reasons",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in ranked:
            d = asdict(r)
            d.pop("_wf_summary", None)
            d["rejection_reasons"] = "; ".join(d.get("rejection_reasons", []))
            writer.writerow({k: d.get(k, "") for k in _CSV_FIELDS})
    log.info("Profile summary (CSV) → %s", csv_path)

    # ── Production deployment artifact (winner only) ──────────────────────────
    winner = pick_winner(ranked)
    if winner is not None:
        from src.deployment import write_deployment_artifact
        from src.backtest.profile_eval import _profile_to_cfg_overrides
        try:
            profiles = load_profiles(symbol)
            winner_profile = profiles.get(winner.profile_name, {})
        except Exception:
            winner_profile = {}
        exec_overrides = _profile_to_cfg_overrides(winner_profile)
        metrics = {
            "profitable_folds":      winner.profitable_folds,
            "n_folds":               winner.n_folds,
            "avg_pf":                winner.avg_pf,
            "t_stat":                winner.t_stat,
            "pnl_cv":                winner.pnl_cv,
            "avg_sharpe":            winner.avg_sharpe,
            "avg_expectancy":        winner.avg_expectancy,
            "total_pnl":             winner.total_pnl,
            "composite_score":       winner.composite_score,
            "block1_total_pnl":      winner.block1_total_pnl,
            "block2_total_pnl":      winner.block2_total_pnl,
            "trend_profitable_folds": winner.trend_profitable_folds,
            "chop_dominated_folds":  winner.chop_dominated_folds,
        }
        dep_path = write_deployment_artifact(
            symbol=symbol,
            profile_name=winner.profile_name,
            profile_config=winner_profile,
            exec_cfg_overrides=exec_overrides,
            metrics=metrics,
        )
        log.info(
            "Production artifact → %s  [profile=%s  composite=%.3f]",
            dep_path, winner.profile_name, winner.composite_score,
        )
    else:
        log.warning(
            "[%s] No accepted winner — production deployment artifact NOT written.", symbol
        )


# ── Main evaluation entry-point ───────────────────────────────────────────────

def evaluate_all_profiles(
    symbol: str,
    n_trials: int = 20,
    select_by: str = "f1",
    train_bars: int = 10_000,
    test_bars: int = 2_000,
    run_backtest: bool = False,
) -> list[ProfileScore]:
    """
    Evaluate all profiles for `symbol` and return ranked ProfileScore list.

    Parameters
    ----------
    symbol       : Symbol (must have data/processed/ parquets + model artifacts).
    n_trials     : Optuna trials per fold per profile (0 = fast fixed params).
    select_by    : 'f1' | 'trading' — per-fold selection objective.
    train_bars   : Walk-forward training window size.
    test_bars    : Walk-forward test window size.
    run_backtest : If True, also run standalone backtest per profile (slower).
    """
    profiles = load_profiles(symbol)

    results: list[ProfileScore] = []
    for name, profile in profiles.items():
        log.info("─── Profile [%s] ───", name)
        ps = _evaluate_profile(
            symbol=symbol,
            profile_name=name,
            profile=profile,
            n_trials=n_trials,
            select_by=select_by,
            train_bars=train_bars,
            test_bars=test_bars,
            run_backtest=run_backtest,
        )
        results.append(ps)
        _status = "ACCEPTED" if ps.accepted else f"REJECTED ({len(ps.rejection_reasons)} gate(s))"
        log.info(
            "Profile [%s] → %s  avg_pf=%.2f  t=%.2f  cv=%.2f  profitable=%d/%d",
            name, _status, ps.avg_pf, ps.t_stat, ps.pnl_cv,
            ps.profitable_folds, ps.n_folds,
        )

    results = _rank_profiles(results)
    return results


# ── Helper maths ─────────────────────────────────────────────────────────────

def _coalesce_metric(value: Any, fallback: float) -> float:
    """Return `fallback` when `value` is None, otherwise cast to float.

    Needed because dict.get(key, default) only uses the default when the key is
    *absent*; when the key exists with an explicit None value (as _aggregate does
    for pnl_cv when mean_pnl ≤ 0 and for t_stat_pnl when n < 2), .get() returns
    None and float(None) raises TypeError.
    """
    if value is None:
        return fallback
    return float(value)


def _safe_cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) < 1e-9:
        return float("inf")
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return std / abs(mean)


def _safe_tstat(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var < 1e-9:
        return 0.0
    return mean / math.sqrt(var / n)
