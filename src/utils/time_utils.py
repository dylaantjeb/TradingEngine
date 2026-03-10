"""
Time / date utilities for TradingEngine.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import pandas as pd


UTC = timezone.utc


def utcnow() -> datetime:
    """Current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


def to_utc(dt: datetime) -> datetime:
    """Convert a naive or tz-aware datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def bar_date(ts: pd.Timestamp) -> date:
    """Extract UTC date from a pandas Timestamp."""
    return ts.date()


def in_session(ts: pd.Timestamp, start_hour: int, end_hour: int) -> bool:
    """
    Return True if ts falls within [start_hour, end_hour) UTC.
    Set start_hour=0, end_hour=24 to always return True.
    """
    if start_hour == 0 and end_hour >= 24:
        return True
    h = ts.hour + ts.minute / 60.0
    return start_hour <= h < end_hour


def in_blackout(ts: pd.Timestamp, windows: list[list[str]]) -> bool:
    """
    Return True if ts falls within any blackout window.

    windows: list of [\"HH:MM\", \"HH:MM\"] pairs (UTC).
    """
    if not windows:
        return False
    t = time(ts.hour, ts.minute)
    for w in windows:
        try:
            s_h, s_m = map(int, w[0].split(":"))
            e_h, e_m = map(int, w[1].split(":"))
            if time(s_h, s_m) <= t <= time(e_h, e_m):
                return True
        except Exception:
            pass
    return False


def date_range_chunks(
    start: date,
    end: date,
    chunk_days: int,
) -> Iterable[tuple[date, date]]:
    """
    Yield (chunk_start, chunk_end) pairs covering [start, end] with at
    most chunk_days calendar days per chunk (end-inclusive).
    """
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def format_duration(seconds: float) -> str:
    """Human-readable duration string from seconds."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"
