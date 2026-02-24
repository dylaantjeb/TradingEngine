"""
IBKR historical data fetcher using ib_insync.

Connects to TWS / IB Gateway on the configured host/port, resolves the
front-month futures contract, and downloads 1-minute bars in safe chunks
(≤ 30-day windows) to avoid pacing violations.

Output CSV columns: timestamp,open,high,low,close,volume
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

log = logging.getLogger(__name__)

# Default config path (relative to repo root)
_CONFIG_PATH = Path("config/ibkr.yaml")

# IBKR pacing: max 60 historical requests / 10-min window.
# For 1-min bars a single request can span at most 30 calendar days.
_MAX_DAYS_PER_REQUEST = 25  # conservative
_PACING_SLEEP_S = 12  # seconds between chunks


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"IBKR config not found at {_CONFIG_PATH}. "
            "Create it or pass the path explicitly."
        )
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


class IBKRFetcher:
    """Fetch historical minute bars from IBKR TWS / IB Gateway."""

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg = _load_config() if config_path is None else yaml.safe_load(
            (config_path or _CONFIG_PATH).read_text()
        )
        self._ib = None  # lazy connect

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fetch(self, symbol: str, days: int, out_path: Path) -> pd.DataFrame:
        """
        Fetch `days` calendar days of 1-minute bars for `symbol` and write
        them to `out_path` as CSV.

        Raises SystemExit with a clear message if TWS / Gateway is not running.
        """
        try:
            from ib_insync import IB, Contract, util
        except ImportError:
            log.error("ib_insync not installed: pip install ib_insync")
            sys.exit(1)

        ib = self._connect(IB)
        try:
            contract = self._resolve_contract(ib, symbol)
            bars_df = self._download_history(ib, contract, days)
        finally:
            ib.disconnect()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        bars_df.to_csv(out_path)
        log.info(
            "Written %d bars (%s → %s) to %s",
            len(bars_df),
            bars_df.index.min(),
            bars_df.index.max(),
            out_path,
        )
        return bars_df

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _connect(self, IB_class):  # noqa: N803 – keep ib_insync name
        host = self._cfg.get("host", "127.0.0.1")
        port = self._cfg.get("port", 7497)
        client_id = self._cfg.get("client_id", 1)

        ib = IB_class()
        log.info("Connecting to IBKR at %s:%s (clientId=%s) …", host, port, client_id)
        try:
            ib.connect(host, port, clientId=client_id, timeout=10)
        except ConnectionRefusedError:
            log.error(
                "Connection refused on %s:%s – is TWS / IB Gateway running?\n"
                "  Paper trading: port 7497 (TWS) or 4002 (Gateway)\n"
                "  Live trading:  port 7496 (TWS) or 4001 (Gateway)\n"
                "  Also check API access is enabled in TWS → Edit → Global Config → API",
                host,
                port,
            )
            sys.exit(1)
        except Exception as exc:
            log.error("IBKR connection failed: %s", exc)
            sys.exit(1)

        log.info("Connected. Server version: %s", ib.client.serverVersion())
        return ib

    def _resolve_contract(self, ib, symbol: str):
        from ib_insync import Future

        contracts_map: dict = self._cfg.get("contracts", {})
        sym_cfg: dict = contracts_map.get(symbol, {})

        exchange = sym_cfg.get("exchange", "CME")
        currency = sym_cfg.get("currency", "USD")
        last_trade_date = sym_cfg.get("lastTradeDateOrContractMonth", "")

        if last_trade_date:
            contract = Future(
                symbol=symbol,
                lastTradeDateOrContractMonth=last_trade_date,
                exchange=exchange,
                currency=currency,
            )
            log.info(
                "Using explicit contract: %s %s %s", symbol, last_trade_date, exchange
            )
            return contract

        # Resolve front-month via reqContractDetails
        generic = Future(symbol=symbol, exchange=exchange, currency=currency)
        log.info("Resolving front-month contract for %s on %s …", symbol, exchange)
        details = ib.reqContractDetails(generic)
        if not details:
            log.error(
                "No contract details returned for %s. "
                "Check symbol spelling and exchange in config/ibkr.yaml.",
                symbol,
            )
            sys.exit(1)

        # Sort by expiry and pick the nearest front month
        details_sorted = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
        front = details_sorted[0].contract
        log.info(
            "Front-month contract: %s %s conId=%s",
            front.localSymbol,
            front.lastTradeDateOrContractMonth,
            front.conId,
        )
        return front

    def _download_history(self, ib, contract, days: int) -> pd.DataFrame:
        from ib_insync import util

        end_dt = datetime.now(timezone.utc)
        chunks: list[pd.DataFrame] = []

        remaining = days
        while remaining > 0:
            chunk_days = min(remaining, _MAX_DAYS_PER_REQUEST)
            duration_str = f"{chunk_days} D"

            log.info(
                "Requesting %s of 1-min bars ending %s …",
                duration_str,
                end_dt.strftime("%Y%m%d %H:%M:%S"),
            )
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration_str,
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,  # UTC epoch
                    keepUpToDate=False,
                )
            except Exception as exc:
                log.error("reqHistoricalData failed: %s", exc)
                sys.exit(1)

            if not bars:
                log.warning("No bars returned for chunk ending %s", end_dt)
            else:
                df = util.df(bars)
                df.rename(
                    columns={
                        "date": "timestamp",
                        "barCount": "bar_count",
                        "average": "avg",
                    },
                    inplace=True,
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df = df[df["volume"] > 0]  # drop zero-volume bars
                chunks.append(df)
                log.info("  Got %d bars", len(df))

            end_dt = end_dt - timedelta(days=chunk_days)
            remaining -= chunk_days

            if remaining > 0:
                log.info("  Pacing pause %ss …", _PACING_SLEEP_S)
                time.sleep(_PACING_SLEEP_S)

        if not chunks:
            log.error("No data returned. Check contract, permissions, and market data subscription.")
            sys.exit(1)

        result = pd.concat(chunks).sort_index()
        result = result[~result.index.duplicated(keep="first")]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Quick connectivity test (python -m src.data_engine.ibkr_fetch)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Testing IBKR connectivity …")
    try:
        from ib_insync import IB

        cfg = _load_config()
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 7497)
        ib = IB()
        ib.connect(host, port, clientId=99, timeout=5)
        print(f"SUCCESS – connected to {host}:{port}, version {ib.client.serverVersion()}")
        ib.disconnect()
    except ConnectionRefusedError:
        print(f"FAILED – Connection refused on {host}:{port}. Is TWS / Gateway running?")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED – {e}")
        sys.exit(1)
