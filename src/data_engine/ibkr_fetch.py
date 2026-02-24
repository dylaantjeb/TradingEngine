"""
IBKR historical data fetcher using ib_insync.

Connects to TWS / IB Gateway on the configured host/port, resolves the
front-month futures contract, and downloads 1-minute bars in pacing-safe
chunks with retries, exponential backoff, chunk-size halving, and resume.

Output CSV columns: timestamp,open,high,low,close,volume
"""

from __future__ import annotations

import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

log = logging.getLogger(__name__)

_CONFIG_PATH = Path("config/ibkr.yaml")

# IBKR error codes we treat as retriable pacing / availability errors
_PACING_ERRORS = frozenset({
    162,   # Historical Market Data Service error (pacing violation / query cancelled)
    165,   # Historical data farm connection is OK (informational – ignore)
    366,   # No historical data query found for ticker
})
_PERMISSION_ERRORS = frozenset({
    354,   # Requested market data is not subscribed
    200,   # No security definition has been found
    321,   # Error validating request
})


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"IBKR config not found at {_CONFIG_PATH}. "
            "Create it or pass the path explicitly."
        )
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _make_error_handler(error_list: list):
    """Factory so each call gets its own closure over a fresh list."""
    def on_error(*args):
        req_id   = args[0] if len(args) > 0 else -1
        code     = args[1] if len(args) > 1 else 0
        msg      = args[2] if len(args) > 2 else ""
        error_list.append(code)
        if code not in (165, 2104, 2106, 2158):  # skip routine info codes
            log.warning("IBKR error %d (reqId=%s): %s", code, req_id, msg)
    return on_error


class IBKRFetcher:
    """Fetch historical minute bars from IBKR TWS / IB Gateway."""

    def __init__(self, config_path: Optional[Path] = None):
        self._cfg = _load_config() if config_path is None else yaml.safe_load(
            (config_path or _CONFIG_PATH).read_text()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        symbol: str,
        days: int,
        out_path: Path,
        bar_size: str = "1 min",
        use_rth: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Fetch `days` of `bar_size` bars for `symbol`.

        • Chunks the request to honour IBKR pacing rules.
        • Retries on Error 162 / timeout with exponential backoff.
        • Halves chunk size when all retries are exhausted.
        • Resumes from existing CSV so reruns are cheap.
        • Writes partial results to `out_path` after every successful chunk.

        Raises SystemExit with a clear message if TWS / Gateway is not running.
        """
        try:
            from ib_insync import IB
        except ImportError:
            log.error("ib_insync not installed: pip install ib_insync")
            sys.exit(1)

        fetch_cfg = self._cfg.get("fetch", {})

        # Config defaults, overridable by caller
        if use_rth is None:
            use_rth = fetch_cfg.get("use_rth", False)

        # Resume: load existing data if present
        out_path = Path(out_path)
        existing_df = self._load_existing(out_path, fetch_cfg)

        ib = self._connect(IB)
        try:
            contract = self._resolve_contract(ib, symbol)
            bars_df = self._download_history(
                ib, contract, days, existing_df,
                bar_size, use_rth, fetch_cfg, out_path,
            )
        finally:
            ib.disconnect()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        bars_df.to_csv(out_path)
        log.info(
            "Fetch complete: %d bars  %s → %s  saved to %s",
            len(bars_df),
            bars_df.index.min(),
            bars_df.index.max(),
            out_path,
        )
        return bars_df

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_existing(self, out_path: Path, fetch_cfg: dict) -> Optional[pd.DataFrame]:
        if not fetch_cfg.get("resume", True):
            return None
        if not out_path.exists():
            return None
        try:
            df = pd.read_csv(out_path, parse_dates=["timestamp"], index_col="timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            log.info(
                "Resume: loaded %d existing bars (%s → %s)",
                len(df), df.index.min(), df.index.max(),
            )
            return df
        except Exception as exc:
            log.warning("Could not load existing CSV for resume (%s) – starting fresh.", exc)
            return None

    def _connect(self, IB_class):
        host      = self._cfg.get("host", "127.0.0.1")
        port      = self._cfg.get("port", 7497)
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
                "  Check: TWS → Edit → Global Config → API → Enable ActiveX and Socket Clients",
                host, port,
            )
            sys.exit(1)
        except Exception as exc:
            log.error("IBKR connection failed: %s", exc)
            sys.exit(1)

        log.info("Connected. Server version: %s", ib.client.serverVersion())
        return ib

    def _resolve_contract(self, ib, symbol: str):
        from ib_insync import Future

        sym_cfg   = self._cfg.get("contracts", {}).get(symbol, {})
        exchange  = sym_cfg.get("exchange", "CME")
        currency  = sym_cfg.get("currency", "USD")
        expiry    = sym_cfg.get("lastTradeDateOrContractMonth", "")

        if expiry:
            contract = Future(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                exchange=exchange,
                currency=currency,
            )
            log.info("Using explicit contract: %s %s %s", symbol, expiry, exchange)
            return contract

        generic = Future(symbol=symbol, exchange=exchange, currency=currency)
        log.info("Resolving front-month contract for %s on %s …", symbol, exchange)
        details = ib.reqContractDetails(generic)
        if not details:
            log.error(
                "No contract details for %s. Check symbol and exchange in config/ibkr.yaml. "
                "Also verify market data subscription for this symbol.",
                symbol,
            )
            sys.exit(1)

        details_sorted = sorted(
            details, key=lambda d: d.contract.lastTradeDateOrContractMonth
        )
        front = details_sorted[0].contract
        log.info(
            "Front-month: %s %s  conId=%s",
            front.localSymbol, front.lastTradeDateOrContractMonth, front.conId,
        )
        return front

    # ── Download orchestrator ─────────────────────────────────────────────────

    def _download_history(
        self, ib, contract, days: int,
        existing_df: Optional[pd.DataFrame],
        bar_size: str, use_rth: bool,
        fetch_cfg: dict, out_path: Path,
    ) -> pd.DataFrame:
        from ib_insync import util

        chunk_days    = int(fetch_cfg.get("chunk_days", 5))
        max_retries   = int(fetch_cfg.get("max_retries", 3))
        base_backoff  = float(fetch_cfg.get("base_backoff_sec", 5))
        max_backoff   = float(fetch_cfg.get("max_backoff_sec", 60))
        timeout       = float(fetch_cfg.get("request_timeout_sec", 120))
        pacing_pause  = float(fetch_cfg.get("pacing_pause_sec", 2))

        now = datetime.now(timezone.utc)

        # ── Determine start point ─────────────────────────────────────────────
        accumulated: list[pd.DataFrame] = []
        if existing_df is not None and not existing_df.empty:
            accumulated.append(existing_df)
            earliest     = existing_df.index.min()
            already_days = max(0, int((now - earliest).total_seconds() / 86400))
            if already_days >= days:
                log.info(
                    "Resume: existing data spans %d days (>= requested %d). Nothing to fetch.",
                    already_days, days,
                )
                return existing_df
            remaining = days - already_days
            end_dt    = earliest   # fetch backwards from the earliest bar we have
            log.info(
                "Resume: have %d days already, fetching %d more days before %s",
                already_days, remaining, earliest.strftime("%Y-%m-%d %H:%M UTC"),
            )
        else:
            remaining = days
            end_dt    = now

        # ── Chunk loop ────────────────────────────────────────────────────────
        cur_chunk_days = chunk_days   # may shrink on persistent failures

        while remaining > 0:
            req_days      = min(remaining, cur_chunk_days)
            duration_str  = f"{req_days} D"

            log.info(
                "Chunk: %s ending %s  (remaining: %d days)",
                duration_str,
                end_dt.strftime("%Y%m%d %H:%M:%S UTC"),
                remaining,
            )

            raw_bars, had_pacing_error = self._fetch_chunk_with_retry(
                ib, contract, end_dt, duration_str, bar_size, use_rth,
                timeout, max_retries, base_backoff, max_backoff,
            )

            if raw_bars:
                df = util.df(raw_bars)
                df.rename(
                    columns={"date": "timestamp", "barCount": "bar_count", "average": "avg"},
                    inplace=True,
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df = df[df["volume"] > 0]
                log.info("  ✓ %d bars received (chunk %s ending %s)", len(df), duration_str, end_dt)

                accumulated.append(df)
                cur_chunk_days = chunk_days   # reset after success

                # Incremental save — survives crashes mid-run
                self._save_partial(accumulated, out_path)

                end_dt    -= timedelta(days=req_days)
                remaining -= req_days

            else:
                # All retries exhausted → halve chunk size and retry same window
                new_chunk = max(1, req_days // 2)
                if new_chunk < req_days:
                    log.warning(
                        "All retries failed for %d D chunk — halving to %d D and retrying",
                        req_days, new_chunk,
                    )
                    cur_chunk_days = new_chunk
                    # do NOT advance end_dt or decrement remaining — retry
                    continue
                else:
                    # Already at 1 D and still failing: skip this window
                    log.error(
                        "Chunk still failing at 1 D (end=%s) — skipping",
                        end_dt.strftime("%Y%m%d %H:%M:%S"),
                    )
                    end_dt    -= timedelta(days=req_days)
                    remaining -= req_days

            if remaining > 0:
                log.info("  Pacing pause %.1f s …", pacing_pause)
                time.sleep(pacing_pause)

        # ── Final assembly ────────────────────────────────────────────────────
        if not accumulated:
            raise RuntimeError(
                "No data returned from IBKR. "
                "Check contract, market data subscription, and TWS permissions."
            )

        result = pd.concat(accumulated).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        return result

    # ── Single chunk with retries ─────────────────────────────────────────────

    def _fetch_chunk_with_retry(
        self, ib, contract,
        end_dt: datetime, duration_str: str,
        bar_size: str, use_rth: bool,
        timeout: float, max_retries: int,
        base_backoff: float, max_backoff: float,
    ) -> tuple[list, bool]:
        """
        Try one reqHistoricalData call up to `max_retries+1` times.

        Returns (bars, had_pacing_error).
        `bars` is an empty list if all attempts failed.
        """
        for attempt in range(max_retries + 1):
            error_codes: list[int] = []
            on_error = _make_error_handler(error_codes)
            ib.errorEvent += on_error

            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=2,        # UTC epoch → tz-aware timestamps
                    keepUpToDate=False,
                    timeout=timeout,     # ib_insync cancels and returns [] on timeout
                )
            except Exception as exc:
                log.warning("reqHistoricalData raised exception: %s", exc)
                bars = []
            finally:
                ib.errorEvent -= on_error

            had_pacing = bool(error_codes & _PACING_ERRORS) if not isinstance(error_codes, frozenset) else False
            had_pacing = any(c in _PACING_ERRORS for c in error_codes)

            had_permission = any(c in _PERMISSION_ERRORS for c in error_codes)
            if had_permission:
                log.error(
                    "IBKR permission error (code %s). "
                    "Check your market data subscriptions in TWS / Account Management.",
                    [c for c in error_codes if c in _PERMISSION_ERRORS],
                )
                # Don't retry permission errors — they won't fix themselves
                return [], had_pacing

            if bars:
                return bars, had_pacing

            # Empty response — decide whether to retry
            if attempt < max_retries:
                jitter   = random.random()
                backoff  = min(max_backoff, base_backoff * (2 ** attempt)) + jitter
                reason   = "Error 162 pacing violation" if had_pacing else "empty/timeout response"
                log.warning(
                    "  Attempt %d/%d failed (%s). Retrying in %.1f s …",
                    attempt + 1, max_retries + 1, reason, backoff,
                )
                time.sleep(backoff)
            else:
                log.warning(
                    "  All %d attempts failed for chunk %s ending %s.",
                    max_retries + 1, duration_str,
                    end_dt.strftime("%Y%m%d %H:%M:%S"),
                )

        return [], True   # signal that we hit pacing / failure

    # ── Incremental save ──────────────────────────────────────────────────────

    @staticmethod
    def _save_partial(accumulated: list[pd.DataFrame], out_path: Path) -> None:
        """Merge accumulated chunks and write to disk (for crash recovery)."""
        if not accumulated:
            return
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged = pd.concat(accumulated).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged.to_csv(out_path)
            log.debug("Partial save: %d bars → %s", len(merged), out_path)
        except Exception as exc:
            log.warning("Partial save failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Quick connectivity test  (python -m src.data_engine.ibkr_fetch)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Testing IBKR connectivity …")
    try:
        from ib_insync import IB

        cfg  = _load_config()
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 7497)
        ib   = IB()
        ib.connect(host, port, clientId=99, timeout=5)
        print(f"SUCCESS – connected to {host}:{port}, version {ib.client.serverVersion()}")
        ib.disconnect()
    except ConnectionRefusedError:
        print(f"FAILED – Connection refused on {host}:{port}. Is TWS / Gateway running?")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED – {e}")
        sys.exit(1)
