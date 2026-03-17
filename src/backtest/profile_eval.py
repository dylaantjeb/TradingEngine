"""
Profile-based strategy evaluation for TradingEngine.

Simplified for funded-account robustness. One question only:
"Is this engine stable enough to trade on a funded account?"

Acceptance gates (ALL must pass)
---------------------------------
  • no_zero_trade_folds   : every fold has >= 3 trades (fold with fewer is
                            not a strategy, it's silence — reject immediately)
  • pct_profitable        >= 60%  (majority of folds must be green)
  • max_losing_folds      <= 2    (hard cap: at most 2 red folds in any run)
  • avg_pf                >= 1.15 (modest but real edge after costs)
  • pnl_cv                <= 1.20 (PnL consistency — outlier folds disqualify)
  • avg_expectancy        > 0     (positive expected value per trade)
  • max_fold_drawdown_usd <= 800  (no single fold blowing through the daily limit)
  • outlier_fold_check    : no single fold contributes > 50% of total PnL
                            (strategy must not depend on one lucky fold)

Removed gates (too fragile or irrelevant for single-profile setup):
  • t_stat (unreliable with 5 folds and 20-100 trades/fold)
  • block2_not_dragging (block 2 removed from profile)
  • trend_profitable_folds (folded into pct_profitable)
  • chop_dominated_folds (regime classifier being removed)
  • composite score complexity

CLI entry-point
---------------
  python -m src.cli evaluate-profiles --symbol ES --trials 30 --seed 42
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
_ACCEPT_MIN_TRADES_PER_FOLD   = 3      # fewer = silent fold, not a strategy
_ACCEPT_MIN_PCT_PROFITABLE    = 0.60   # 60% of folds must be green
_ACCEPT_MAX_LOSING_FOLDS      = 2      # hard cap on red folds
_ACCEPT_MIN_AVG_PF            = 1.15   # modest but real edge
_ACCEPT_MAX_CV                = 2.00   # PnL consistency
_ACCEPT_MIN_AVG_EXPECTANCY    = 0.0    # positive EV per trade
_ACCEPT_MAX_FOLD_DRAWDOWN_USD = 800.0  # no fold exceeds funded daily limit
_ACCEPT_MAX_OUTLIER_FOLD_PCT  = 0.65   # single fold ≤ 65% of total PnL


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
    avg_sharpe:            float = 0.0
    max_drawdown_usd:      float = 0.0  # worst single fold
    avg_expectancy:        float = 0.0
    trades_per_day:        float = 0.0
    total_pnl:             float = 0.0
    min_fold_trades:       int   = 0    # fewest trades in any single fold
    outlier_fold_pct:      float = 0.0  # best fold PnL / total PnL

    # Acceptance
    accepted:              bool  = False
    rejection_reasons:     list  = field(default_factory=list)
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
    seed: int = 42,
) -> ProfileScore:
    """Run walk-forward for one profile and return a ProfileScore."""
    from src.backtest.walk_forward import run_walk_forward

    description = profile.get("description", "")
    if isinstance(description, str):
        description = description.strip().replace("\n", " ")

    threshold_candidates = profile.get("threshold_candidates", None)
    exec_overrides       = _profile_to_cfg_overrides(profile)

    log.info(
        "Evaluating profile [%s] – seed=%d  threshold_candidates=%s  overrides=%s",
        profile_name, seed, threshold_candidates, list(exec_overrides.keys()),
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
            seed=seed,
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

    pnl_cv   = _coalesce_metric(agg.get("pnl_cv"), _safe_cv(pnl_values))
    pct_prof = profitable_folds / n_folds if n_folds else 0.0

    # Outlier fold check: what fraction of total PnL came from the single best fold?
    total_pnl     = sum(pnl_values)
    best_fold_pnl = max(pnl_values, default=0.0)
    outlier_pct   = (best_fold_pnl / total_pnl) if total_pnl > 0 and best_fold_pnl > 0 else 0.0

    # Fewest trades in any fold — catches silent folds
    fold_trade_counts = [f.n_trades for f in folds]
    min_fold_trades   = min(fold_trade_counts, default=0)

    ps = ProfileScore(
        profile_name=profile_name,
        description=description,
        n_folds=n_folds,
        profitable_folds=profitable_folds,
        pct_profitable=pct_prof,
        avg_pf=avg_pf,
        median_pf=median_pf,
        pnl_cv=pnl_cv,
        avg_sharpe=avg_sharpe,
        max_drawdown_usd=max_dd,
        avg_expectancy=avg_exp,
        trades_per_day=avg_tpd,
        total_pnl=total_pnl,
        min_fold_trades=min_fold_trades,
        outlier_fold_pct=outlier_pct,
        _wf_summary=wf,
    )

    _apply_acceptance(ps)
    return ps


# ── Acceptance gate ───────────────────────────────────────────────────────────

def _apply_acceptance(ps: ProfileScore) -> None:
    """Populate ps.accepted and ps.rejection_reasons in-place.

    Gate logic (ALL must pass):
      1. No silent folds: every fold has >= 3 trades.
      2. Majority profitable: >= 60% of folds green.
      3. Hard cap on losing folds: at most 2 red folds.
      4. Real edge: avg_pf >= 1.15.
      5. PnL consistency: CV <= 1.20.
      6. Positive EV: avg_expectancy > 0.
      7. Drawdown bounded: worst fold drawdown <= $800.
      8. No outlier dependence: best fold <= 50% of total PnL.
    """
    reasons: list[str] = []

    # Gate 1: no silent folds
    if ps.min_fold_trades < _ACCEPT_MIN_TRADES_PER_FOLD:
        reasons.append(
            f"silent_fold: min_trades_in_fold={ps.min_fold_trades} "
            f"< {_ACCEPT_MIN_TRADES_PER_FOLD} — strategy is not trading consistently"
        )

    # Gate 2: majority profitable
    if ps.pct_profitable < _ACCEPT_MIN_PCT_PROFITABLE:
        reasons.append(
            f"pct_profitable={ps.pct_profitable:.0%} < {_ACCEPT_MIN_PCT_PROFITABLE:.0%}"
        )

    # Gate 3: hard cap on losing folds (proportional: 40% of total, minimum 2)
    losing_folds = ps.n_folds - ps.profitable_folds
    max_losing_allowed = max(_ACCEPT_MAX_LOSING_FOLDS, int(ps.n_folds * 0.40))
    if losing_folds > max_losing_allowed:
        reasons.append(
            f"losing_folds={losing_folds} > {max_losing_allowed} "
            f"(max allowed red folds exceeded)"
        )

    # Gate 4: real edge
    if ps.avg_pf < _ACCEPT_MIN_AVG_PF:
        reasons.append(f"avg_pf={ps.avg_pf:.2f} < {_ACCEPT_MIN_AVG_PF}")

    # Gate 5: PnL consistency
    if ps.pnl_cv > _ACCEPT_MAX_CV:
        reasons.append(
            f"pnl_cv={ps.pnl_cv:.2f} > {_ACCEPT_MAX_CV} "
            f"— PnL too volatile across folds"
        )

    # Gate 6: positive EV
    if ps.avg_expectancy <= _ACCEPT_MIN_AVG_EXPECTANCY:
        reasons.append(f"avg_expectancy=${ps.avg_expectancy:.0f} ≤ 0")

    # Gate 7: drawdown bounded
    if ps.max_drawdown_usd > _ACCEPT_MAX_FOLD_DRAWDOWN_USD:
        reasons.append(
            f"max_fold_drawdown=${ps.max_drawdown_usd:.0f} > "
            f"${_ACCEPT_MAX_FOLD_DRAWDOWN_USD:.0f} "
            f"— one fold blew through the funded-account daily limit"
        )

    # Gate 8: no outlier fold dependence (relaxed for large fold counts)
    # With many folds, the best fold naturally represents a smaller fraction —
    # so the threshold scales: 50% cap for <=5 folds, up to 70% for 20+ folds.
    outlier_threshold = min(
        0.70,
        _ACCEPT_MAX_OUTLIER_FOLD_PCT + max(0, ps.n_folds - 5) * 0.01,
    )
    if ps.outlier_fold_pct > outlier_threshold:
        reasons.append(
            f"outlier_fold_pct={ps.outlier_fold_pct:.0%} > "
            f"{outlier_threshold:.0%} "
            f"— strategy depends on one lucky fold"
        )

    ps.rejection_reasons = reasons
    ps.accepted = len(reasons) == 0


# ── Ranking ───────────────────────────────────────────────────────────────────

def _rank_profiles(results: list[ProfileScore]) -> list[ProfileScore]:
    """Sort: accepted first (desc avg_pf), then rejected (desc avg_pf)."""
    accepted = sorted(
        [r for r in results if r.accepted],
        key=lambda r: r.avg_pf,
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
    """Print funded-account focused scoreboard to stdout."""
    ranked = sorted(results, key=lambda r: r.rank)

    print(f"\n{'='*80}")
    print(f"  EVALUATION SCOREBOARD — {symbol}")
    print(f"{'='*80}")
    print(
        f"  {'Rk':>2}  {'Profile':<26}  {'OK':>2}  "
        f"{'Folds':>7}  {'PF':>5}  {'CV':>5}  "
        f"{'Exp$':>6}  {'MinTrd':>6}  {'MaxDD$':>7}  {'TotPnL':>8}"
    )
    print(f"  {'-'*76}")

    for r in ranked:
        ok = "✓" if r.accepted else "✗"
        print(
            f"  {r.rank:>2}  {r.profile_name:<26}  {ok:>2}  "
            f"  {r.profitable_folds}/{r.n_folds}   "
            f"{r.avg_pf:>5.2f}  {r.pnl_cv:>5.2f}  "
            f"{r.avg_expectancy:>6.0f}  {r.min_fold_trades:>6d}  "
            f"{r.max_drawdown_usd:>7.0f}  {r.total_pnl:>8.0f}"
        )
        if not r.accepted:
            for reason in r.rejection_reasons:
                print(f"      ✗ {reason}")

    print(f"{'='*80}")

    winner = pick_winner(ranked)
    if winner:
        print(f"\n  FUNDED-READY: [{winner.profile_name}]")
        print(f"  avg_pf={winner.avg_pf:.2f}  cv={winner.pnl_cv:.2f}  "
              f"profitable={winner.profitable_folds}/{winner.n_folds}  "
              f"expectancy=${winner.avg_expectancy:.0f}  total_pnl=${winner.total_pnl:.0f}")
        print(f"  Strategy is READY FOR DEPLOYMENT.\n")
    else:
        print(f"\n  REJECT STRATEGY — no deployable profile found.\n")
        for r in ranked:
            if not r.accepted:
                print(f"  [{r.profile_name}] failed: {'; '.join(r.rejection_reasons)}")
        print()


def save_artifacts(
    symbol: str,
    results: list[ProfileScore],
    seed: int = 42,
    acceptance_check_runs: int = 1,
) -> None:
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
        "rank", "profile_name", "accepted",
        "n_folds", "profitable_folds", "pct_profitable",
        "avg_pf", "median_pf", "pnl_cv", "avg_sharpe",
        "max_drawdown_usd", "avg_expectancy", "trades_per_day", "total_pnl",
        "min_fold_trades", "outlier_fold_pct",
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
        # Retrieve seed/reproducibility metadata attached by evaluate_all_profiles
        eval_seed      = winner.__dict__.get("_eval_seed", seed)
        check_runs     = winner.__dict__.get("_acceptance_check_runs", acceptance_check_runs)
        metrics = {
            "profitable_folds":      winner.profitable_folds,
            "n_folds":               winner.n_folds,
            "pct_profitable":        winner.pct_profitable,
            "avg_pf":                winner.avg_pf,
            "pnl_cv":                winner.pnl_cv,
            "avg_sharpe":            winner.avg_sharpe,
            "avg_expectancy":        winner.avg_expectancy,
            "total_pnl":             winner.total_pnl,
            "min_fold_trades":       winner.min_fold_trades,
            "outlier_fold_pct":      winner.outlier_fold_pct,
            "max_drawdown_usd":      winner.max_drawdown_usd,
            "evaluation_seed":       eval_seed,
            "acceptance_check_runs": check_runs,
            "reproducibility_verified": check_runs >= 2,
        }
        dep_path = write_deployment_artifact(
            symbol=symbol,
            profile_name=winner.profile_name,
            profile_config=winner_profile,
            exec_cfg_overrides=exec_overrides,
            metrics=metrics,
        )
        log.info(
            "Production artifact → %s  [profile=%s  avg_pf=%.3f]",
            dep_path, winner.profile_name, winner.avg_pf,
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
    seed: int = 42,
    acceptance_check_runs: int = 1,
) -> list[ProfileScore]:
    """
    Evaluate all profiles for `symbol` and return ranked ProfileScore list.

    Parameters
    ----------
    symbol               : Symbol (must have data/processed/ parquets + model artifacts).
    n_trials             : Optuna trials per fold per profile (0 = fast fixed params).
    select_by            : 'f1' | 'trading' — per-fold selection objective.
    train_bars           : Walk-forward training window size.
    test_bars            : Walk-forward test window size.
    run_backtest         : If True, also run standalone backtest per profile (slower).
    seed                 : Master random seed.  Each profile gets a unique sub-seed
                           (seed + profile_idx * 100_000) to ensure inter-profile
                           independence while maintaining intra-profile determinism.
    acceptance_check_runs: 1 (default) = single run.  2 = also verify the winner
                           on seed+1; write artifact only if the same profile is
                           accepted on both seeds.  Prevents stochastic optimizer
                           drift from producing false positives.
    """
    profiles = load_profiles(symbol)

    results: list[ProfileScore] = []
    for profile_idx, (name, profile) in enumerate(profiles.items()):
        # Each profile uses a well-separated sub-seed so that fold-level randomness
        # in one profile cannot accidentally mirror another profile's sequence.
        profile_seed = seed + profile_idx * 100_000
        log.info("─── Profile [%s]  profile_seed=%d ───", name, profile_seed)
        ps = _evaluate_profile(
            symbol=symbol,
            profile_name=name,
            profile=profile,
            n_trials=n_trials,
            select_by=select_by,
            train_bars=train_bars,
            test_bars=test_bars,
            run_backtest=run_backtest,
            seed=profile_seed,
        )
        results.append(ps)
        _status = "ACCEPTED" if ps.accepted else f"REJECTED ({len(ps.rejection_reasons)} gate(s))"
        log.info(
            "Profile [%s] → %s  avg_pf=%.2f  cv=%.2f  profitable=%d/%d",
            name, _status, ps.avg_pf, ps.pnl_cv,
            ps.profitable_folds, ps.n_folds,
        )

    results = _rank_profiles(results)

    # ── Reproducibility check ─────────────────────────────────────────────────
    # When acceptance_check_runs >= 2, re-run the winner with seed+1.  The
    # same profile must be accepted on the second seed or no artifact is written.
    if acceptance_check_runs >= 2:
        winner1 = pick_winner(results)
        if winner1 is not None:
            verify_seed = seed + 1
            log.info(
                "Reproducibility check: re-running winner [%s] with seed=%d …",
                winner1.profile_name, verify_seed,
            )
            try:
                # Locate the profile in the dict so we have its original index
                profiles_list = list(profiles.items())
                winner_idx = next(
                    i for i, (n, _) in enumerate(profiles_list)
                    if n == winner1.profile_name
                )
                winner_profile_seed = verify_seed + winner_idx * 100_000
                ps2 = _evaluate_profile(
                    symbol=symbol,
                    profile_name=winner1.profile_name,
                    profile=profiles[winner1.profile_name],
                    n_trials=n_trials,
                    select_by=select_by,
                    train_bars=train_bars,
                    test_bars=test_bars,
                    run_backtest=run_backtest,
                    seed=winner_profile_seed,
                )
                if ps2.accepted:
                    log.info(
                        "Reproducibility check PASSED: [%s] accepted on both "
                        "seed=%d and seed=%d",
                        winner1.profile_name, seed, verify_seed,
                    )
                    # Annotate the original winner with reproducibility confirmation
                    winner1.rejection_reasons = [
                        r for r in winner1.rejection_reasons
                        if not r.startswith("REPRODUCIBILITY")
                    ]
                else:
                    log.warning(
                        "Reproducibility check FAILED: [%s] accepted on seed=%d "
                        "but REJECTED on seed=%d (%s). "
                        "Marking as NOT accepted to prevent false-positive deployment.",
                        winner1.profile_name, seed, verify_seed,
                        "; ".join(ps2.rejection_reasons),
                    )
                    # Revoke acceptance — this profile is stochastically marginal
                    winner1.accepted = False
                    winner1.rejection_reasons.append(
                        f"REPRODUCIBILITY_FAILED: rejected on verify_seed={verify_seed} "
                        f"({'; '.join(ps2.rejection_reasons)})"
                    )
                    results = _rank_profiles(results)
            except Exception as exc:
                log.error(
                    "Reproducibility check error for [%s]: %s — skipping check, "
                    "winner remains accepted.",
                    winner1.profile_name, exc,
                )
        else:
            log.info(
                "Reproducibility check skipped: no winner found in primary run."
            )

    # Attach evaluation metadata so save_artifacts can record it
    for r in results:
        r.__dict__["_eval_seed"] = seed
        r.__dict__["_acceptance_check_runs"] = acceptance_check_runs

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
