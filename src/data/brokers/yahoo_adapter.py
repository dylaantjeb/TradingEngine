"""
Yahoo Finance data adapter (free, no IBKR required).

Limitations
-----------
- 1-minute bars: last 7 calendar days only (yfinance constraint).
- Intraday intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m.
- Daily bars: any date range.

Output
------
A CSV matching the TradingEngine OHLCV format:
  timestamp,open,high,low,close,volume
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# yfinance ticker suffixes for futures-like ETFs (nearest approximation
# for users who cannot access IBKR futures data).
_SYMBOL_MAP: dict[str, str] = {
    "ES": "SPY",    # S&P 500 ETF – closest free proxy
    "NQ": "QQQ",    # Nasdaq-100 ETF
    "MES": "SPY",
    "MNQ": "QQQ",
    "YM": "DIA",    # Dow Jones ETF
    "RTY": "IWM",   # Russell 2000
    "CL": "USO",    # Crude oil ETF
    "GC": "GLD",    # Gold ETF
    "SI": "SLV",    # Silver ETF
    "ZN": "TLT",    # 10-year Treasury ETF
}

_INTERVAL_MAP: dict[str, str] = {
    "1 min":  "1m",
    "2 min":  "2m",
    "5 min":  "5m",
    "15 min": "15m",
    "30 min": "30m",
    "1 hour": "60m",
    "1 day":  "1d",
}


def _yf_ticker(symbol: str) -> str:
    """Map a futures symbol to the best-available Yahoo ticker."""
    return _SYMBOL_MAP.get(symbol.upper(), symbol.upper())


def fetch_yahoo(
    symbol: str,
    days: int = 7,
    bar_size: str = "1 min",
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV bars from Yahoo Finance and return a DataFrame.

    Parameters
    ----------
    symbol   : Futures symbol (ES, NQ, …) or any Yahoo ticker.
    days     : Number of calendar days of history.  For 1-min bars Yahoo
               only returns the last 7 days; larger values are silently
               capped by Yahoo.
    bar_size : IBKR-style bar-size string ("1 min", "5 min", "1 day", …).
    out_path : If provided, save the result as CSV.

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume] and a
    DatetimeIndex named "timestamp" (UTC).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for Yahoo data.  Install with:  pip install yfinance"
        ) from exc

    interval = _INTERVAL_MAP.get(bar_size, bar_size)
    ticker   = _yf_ticker(symbol)

    if ticker != symbol:
        log.info(
            "Symbol %s mapped to Yahoo ticker %s (ETF proxy – not exact futures data)",
            symbol, ticker,
        )

    period = f"{min(days, 60)}d"   # Yahoo caps intraday history at ~60 days
    log.info("Downloading %s  interval=%s  period=%s from Yahoo Finance …", ticker, interval, period)

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {ticker} "
            f"(interval={interval}, period={period}).\n"
            "  • 1-min bars are only available for the last 7 calendar days.\n"
            "  • Try --bar-size '5 min' or '--bar-size 1 day' for longer history."
        )

    # ── Normalise columns ──────────────────────────────────────────────────────
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # ── Index → UTC DatetimeIndex named 'timestamp' ────────────────────────────
    df.index.name = "timestamp"
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)

    log.info(
        "Downloaded %d bars for %s from %s to %s",
        len(df), ticker, df.index[0], df.index[-1],
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        log.info("Saved → %s", out_path)

    return df
