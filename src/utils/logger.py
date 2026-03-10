"""
Centralised logging setup for TradingEngine.

Call configure_logging() once at application startup (CLI, API, or test runner).
Subsequent calls are no-ops unless force=True.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_CONFIGURED = False

_LOG_DIR = Path(os.environ.get("TE_LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "trading_engine.log"

_FMT_CONSOLE = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
_FMT_FILE    = "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d – %(message)s"
_DATEFMT     = "%Y-%m-%dT%H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
    force: bool = False,
) -> None:
    """
    Set up console + rotating-file logging.

    Parameters
    ----------
    level        : Root log level (default INFO).
    log_file     : Override log file path (default logs/trading_engine.log).
    max_bytes    : Rotate when file exceeds this size.
    backup_count : Keep this many backup files.
    force        : Reconfigure even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers added by basicConfig or earlier calls
    if force:
        root.handlers.clear()

    # ── Console handler ────────────────────────────────────────────────────────
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_FMT_CONSOLE, datefmt=_DATEFMT))
        root.addHandler(console)

    # ── Rotating file handler ──────────────────────────────────────────────────
    file_path = Path(log_file) if log_file else _LOG_FILE
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATEFMT))
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("ib_insync", "asyncio", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
