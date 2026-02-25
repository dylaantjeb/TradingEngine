"""
Label / feature alignment guardrails.

Call `check_label_alignment` before any backtest or training run to catch
common leakage and alignment mistakes early.  All checks emit log warnings
(never hard-errors) so the pipeline keeps running.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def check_label_alignment(
    features: pd.DataFrame,
    symbol: str = "",
    labels: pd.DataFrame | None = None,
    execution_delay_bars: int = 1,
) -> None:
    """
    Run a battery of alignment and leakage checks and log warnings for any
    violations.  Does NOT raise – the caller decides whether to abort.

    Checks
    ──────
    1. execution_delay_bars == 0  → same-bar fill warning
    2. Features index is a DatetimeIndex
    3. Features are sorted and monotone (no accidental shuffles)
    4. Labels, if provided:
       a. Index alignment with features
       b. Label distribution – warn if flat (0) class is < 1 %
       c. Labels must not contain NaN
    """
    prefix = f"[{symbol}] " if symbol else ""

    # ── Check 1: execution delay ────────────────────────────────────────────
    if execution_delay_bars == 0:
        log.warning(
            "%sexecution_delay_bars=0 → fills on the SAME bar as the signal. "
            "This is a lookahead leak and will inflate backtest performance. "
            "Set execution_delay_bars=1 in config/universe.yaml.",
            prefix,
        )
    elif execution_delay_bars < 0:
        log.warning(
            "%sexecution_delay_bars=%d is negative — this is invalid.",
            prefix, execution_delay_bars,
        )

    # ── Check 2: DatetimeIndex ─────────────────────────────────────────────
    if not isinstance(features.index, pd.DatetimeIndex):
        log.warning(
            "%sFeature DataFrame index is not a DatetimeIndex (got %s). "
            "Timestamp-based alignment checks are skipped.",
            prefix, type(features.index).__name__,
        )
        return

    # ── Check 3: Sorted and monotone ──────────────────────────────────────
    if not features.index.is_monotonic_increasing:
        log.warning(
            "%sFeature index is NOT sorted in ascending order. "
            "This can cause incorrect bar-level lookups.",
            prefix,
        )

    if features.index.duplicated().any():
        n_dup = features.index.duplicated().sum()
        log.warning(
            "%s%d duplicate timestamps in feature index. "
            "De-duplicate before backtesting.",
            prefix, n_dup,
        )

    # ── Check 4: Labels ────────────────────────────────────────────────────
    if labels is None:
        # Try loading from default path
        lbl_path = Path(f"data/processed/{symbol}_labels.parquet")
        if lbl_path.exists():
            try:
                labels = pd.read_parquet(lbl_path, engine="pyarrow")
            except Exception:
                pass

    if labels is not None:
        # 4a. Index alignment
        common = features.index.intersection(labels.index)
        n_feat = len(features)
        n_lbl  = len(labels)
        n_common = len(common)

        if n_common == 0:
            log.warning(
                "%sFeature and label indices share NO common timestamps. "
                "Training will be empty. Check build-dataset alignment.",
                prefix,
            )
        elif n_common < min(n_feat, n_lbl) * 0.9:
            log.warning(
                "%sOnly %d / %d feature rows align with labels (%.0f%%). "
                "Possible index mismatch – rebuild dataset.",
                prefix, n_common, n_feat, 100 * n_common / n_feat,
            )

        # 4b. Label distribution
        if "label" in labels.columns:
            counts = labels["label"].value_counts(normalize=True)
            flat_pct = counts.get(0, 0.0)
            if flat_pct < 0.01:
                log.warning(
                    "%sFlat (0) label is only %.2f%% of samples. "
                    "Consider increasing --pt or --sl so the triple-barrier "
                    "produces more neutral outcomes.",
                    prefix, flat_pct * 100,
                )
            for cls, pct in counts.items():
                if pct > 0.95:
                    log.warning(
                        "%sLabel class %s dominates (%.1f%%) — model will likely "
                        "predict a single class and have poor real-world performance.",
                        prefix, cls, pct * 100,
                    )

        # 4c. NaN check
        if "label" in labels.columns and labels["label"].isna().any():
            n_nan = labels["label"].isna().sum()
            log.warning(
                "%s%d NaN values in label column. These rows will be skipped during "
                "training but may cause index mismatches downstream.",
                prefix, n_nan,
            )

    log.debug("%sAlignment checks complete.", prefix)


def assert_no_same_bar_fill(
    signal_times: list,
    fill_times: list,
) -> None:
    """
    Assert that no fill occurs at the same timestamp as its generating signal.
    Raises AssertionError if any same-bar fill is found.
    Used in tests.
    """
    for s, f in zip(signal_times, fill_times):
        assert s != f, (
            f"Same-bar fill detected: signal_time={s} == fill_time={f}. "
            "Set execution_delay_bars=1."
        )
