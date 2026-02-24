"""
FastAPI application for TradingEngine.

Provides REST endpoints for Base44 integration:
  GET  /health           – liveness check
  GET  /status           – engine + training status
  GET  /metrics          – last backtest metrics
  GET  /equity           – equity curve data points
  GET  /trades           – recent trades
  POST /engine/start     – start paper engine in background thread
  POST /engine/stop      – stop paper engine
  POST /train            – trigger training job
  POST /backtest         – trigger backtest

All state is in-memory; the engine thread writes to the shared EngineState object.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(
    title="TradingEngine API",
    version="0.1.0",
    description="Paper trading engine REST API",
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared engine state (thread-safe via a lock)
# ──────────────────────────────────────────────────────────────────────────────

_state_lock = threading.Lock()


class EngineState:
    def __init__(self):
        self.running: bool = False
        self.mode: str = "stopped"          # "paper" | "stopped"
        self.symbol: Optional[str] = None
        self.start_time: Optional[str] = None
        self.stop_time: Optional[str] = None
        self.equity: list[float] = []
        self.trades: list[dict] = []
        self.bar_count: int = 0
        self.last_signal: int = 0
        self.last_confidence: float = 0.0
        self.last_error: Optional[str] = None

        # Training / backtest status
        self.training_status: str = "idle"   # "idle"|"running"|"done"|"error"
        self.training_result: Optional[str] = None
        self.backtest_status: str = "idle"
        self.backtest_metrics: Optional[dict] = None

        # Last loaded backtest report
        self._load_backtest_cache()

    def _load_backtest_cache(self):
        """Try to load the most recent backtest report on startup."""
        report_dir = Path("artifacts/reports")
        if report_dir.exists():
            reports = sorted(report_dir.glob("*_backtest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if reports:
                try:
                    with open(reports[0]) as f:
                        data = json.load(f)
                    self.backtest_metrics = data.get("metrics")
                    self.equity = data.get("equity_curve", [])
                    self.trades = data.get("trades", [])
                    self.backtest_status = "done"
                except Exception:
                    pass


_engine_state = EngineState()
_engine_thread: Optional[threading.Thread] = None


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────


class StartRequest(BaseModel):
    mode: str = "paper"
    symbol: str = "ES"
    csv_input: Optional[str] = None  # path to CSV for paper mode


class TrainRequest(BaseModel):
    symbol: str = "ES"
    trials: int = 20


class BacktestRequest(BaseModel):
    symbol: str = "ES"


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/status")
def status():
    with _state_lock:
        return {
            "engine": {
                "running": _engine_state.running,
                "mode": _engine_state.mode,
                "symbol": _engine_state.symbol,
                "start_time": _engine_state.start_time,
                "stop_time": _engine_state.stop_time,
                "bar_count": _engine_state.bar_count,
                "last_signal": _engine_state.last_signal,
                "last_confidence": _engine_state.last_confidence,
                "last_error": _engine_state.last_error,
            },
            "training": {
                "status": _engine_state.training_status,
                "result": _engine_state.training_result,
            },
            "backtest": {
                "status": _engine_state.backtest_status,
            },
        }


@app.get("/metrics")
def metrics():
    with _state_lock:
        if _engine_state.backtest_metrics is None:
            return {"message": "No backtest results available. Run /backtest first."}
        return _engine_state.backtest_metrics


@app.get("/equity")
def equity():
    with _state_lock:
        eq = _engine_state.equity
    return {"equity": eq, "n_points": len(eq)}


@app.get("/trades")
def trades(limit: int = 100):
    with _state_lock:
        t = _engine_state.trades
    return {"trades": t[-limit:], "total": len(t)}


# ──────────────────────────────────────────────────────────────────────────────
# Engine start / stop
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/engine/start")
def engine_start(req: StartRequest):
    global _engine_thread

    with _state_lock:
        if _engine_state.running:
            raise HTTPException(status_code=409, detail="Engine already running")

    if req.mode != "paper":
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{req.mode}' not supported. Only 'paper' is available.",
        )

    csv_path = req.csv_input or f"data/raw/{req.symbol}_M1.csv"

    def _run():
        global _engine_state
        with _state_lock:
            _engine_state.running = True
            _engine_state.mode = "paper"
            _engine_state.symbol = req.symbol
            _engine_state.start_time = datetime.utcnow().isoformat()
            _engine_state.trades = []
            _engine_state.equity = []
            _engine_state.bar_count = 0
            _engine_state.last_error = None

        try:
            _run_paper_thread(req.symbol, Path(csv_path))
        except Exception as exc:
            with _state_lock:
                _engine_state.last_error = str(exc)
            log.exception("Paper engine error")
        finally:
            with _state_lock:
                _engine_state.running = False
                _engine_state.mode = "stopped"
                _engine_state.stop_time = datetime.utcnow().isoformat()

    _engine_thread = threading.Thread(target=_run, daemon=True, name="paper-engine")
    _engine_thread.start()
    return {"status": "started", "symbol": req.symbol, "mode": req.mode}


@app.post("/engine/stop")
def engine_stop():
    with _state_lock:
        if not _engine_state.running:
            return {"status": "already_stopped"}
        _engine_state.running = False
    return {"status": "stop_requested"}


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/train")
def trigger_train(req: TrainRequest, background_tasks: BackgroundTasks):
    with _state_lock:
        if _engine_state.training_status == "running":
            raise HTTPException(status_code=409, detail="Training already running")
        _engine_state.training_status = "running"
        _engine_state.training_result = None

    def _train_bg():
        try:
            from src.training.train import train
            train(symbol=req.symbol, n_trials=req.trials)
            with _state_lock:
                _engine_state.training_status = "done"
                _engine_state.training_result = f"Training completed for {req.symbol}"
        except SystemExit as e:
            with _state_lock:
                _engine_state.training_status = "error"
                _engine_state.training_result = f"Training failed (exit {e.code})"
        except Exception as exc:
            with _state_lock:
                _engine_state.training_status = "error"
                _engine_state.training_result = str(exc)

    background_tasks.add_task(_train_bg)
    return {"status": "training_started", "symbol": req.symbol, "trials": req.trials}


# ──────────────────────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/backtest")
def trigger_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    with _state_lock:
        if _engine_state.backtest_status == "running":
            raise HTTPException(status_code=409, detail="Backtest already running")
        _engine_state.backtest_status = "running"
        _engine_state.backtest_metrics = None

    def _bt_bg():
        try:
            from src.backtest.engine import run_backtest
            metrics = run_backtest(symbol=req.symbol)
            # Also load equity + trades from saved report
            report_path = Path(f"artifacts/reports/{req.symbol}_backtest.json")
            with _state_lock:
                _engine_state.backtest_status = "done"
                _engine_state.backtest_metrics = metrics
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
                    _engine_state.equity = report.get("equity_curve", [])
                    _engine_state.trades = report.get("trades", [])
        except SystemExit as e:
            with _state_lock:
                _engine_state.backtest_status = "error"
                _engine_state.backtest_metrics = {"error": f"Backtest failed (exit {e.code})"}
        except Exception as exc:
            with _state_lock:
                _engine_state.backtest_status = "error"
                _engine_state.backtest_metrics = {"error": str(exc)}

    background_tasks.add_task(_bt_bg)
    return {"status": "backtest_started", "symbol": req.symbol}


# ──────────────────────────────────────────────────────────────────────────────
# Paper engine thread helper (streaming simulation)
# ──────────────────────────────────────────────────────────────────────────────


def _run_paper_thread(symbol: str, csv_path: Path) -> None:
    """
    Inline paper engine that updates shared state as it runs.
    This is a simplified version of run_paper() that pushes
    equity + trades into the EngineState for the API.
    """
    import json as _json
    from collections import deque
    import numpy as np
    import pandas as pd
    from src.features.builder import build_features, MIN_ROWS
    from src.backtest.engine import (
        COMMISSION_PER_SIDE,
        DEFAULT_CONTRACT_MULTIPLIER,
        DEFAULT_TICK_SIZE,
    )

    CONFIDENCE_THRESHOLD = 0.50
    WARMUP = MIN_ROWS + 10
    SLIPPAGE_PTS = 1 * DEFAULT_TICK_SIZE

    import joblib

    model_path = Path(f"artifacts/models/{symbol}_xgb_best.joblib")
    scaler_path = Path(f"artifacts/scalers/{symbol}_scaler.joblib")
    schema_path = Path(f"artifacts/schema/{symbol}_features.json")

    for p in (model_path, scaler_path, schema_path):
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(schema_path) as f:
        schema = _json.load(f)
    feature_names = schema["feature_names"]
    inv_label_map = {k: int(v) for k, v in schema["inv_label_map"].items()}

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    raw_df.sort_index(inplace=True)

    # Pre-compute ALL features + predictions at once (fast)
    try:
        all_features = build_features(raw_df, min_rows=1)
    except Exception:
        raise RuntimeError("Feature computation failed on input CSV")

    X_all = all_features[feature_names].values
    X_scaled_all = scaler.transform(X_all)
    proba_all = model.predict_proba(X_scaled_all)
    pred_enc_all = np.argmax(proba_all, axis=1)
    conf_all = np.max(proba_all, axis=1)
    signal_all = np.array([inv_label_map[str(e)] for e in pred_enc_all])
    signal_all[conf_all < CONFIDENCE_THRESHOLD] = 0

    feat_ts_set = set(all_features.index)

    position = 0
    entry_price = 0.0
    equity = 0.0
    bar_count = 0
    feat_cursor = 0

    for ts, row in raw_df.iterrows():
        # Check stop signal from API
        with _state_lock:
            if not _engine_state.running:
                break

        bar_count += 1

        if ts not in feat_ts_set:
            continue

        signal = int(signal_all[feat_cursor])
        confidence = float(conf_all[feat_cursor])
        feat_cursor += 1

        px = float(row["close"])

        if position == 0 and signal != 0:
            position = signal
            entry_price = px + signal * SLIPPAGE_PTS
        elif position != 0 and (signal != position or signal == 0):
            fill_px = px - position * SLIPPAGE_PTS
            raw_pnl = position * (fill_px - entry_price) * DEFAULT_CONTRACT_MULTIPLIER
            net_pnl = raw_pnl - 2 * COMMISSION_PER_SIDE
            equity += net_pnl
            with _state_lock:
                _engine_state.equity.append(round(equity, 2))
                _engine_state.trades.append(
                    {
                        "time": str(ts),
                        "dir": "L" if position > 0 else "S",
                        "entry": round(entry_price, 2),
                        "exit": round(fill_px, 2),
                        "pnl": round(net_pnl, 2),
                        "equity": round(equity, 2),
                    }
                )
            position = 0
            if signal != 0:
                position = signal
                entry_price = px + signal * SLIPPAGE_PTS

        with _state_lock:
            _engine_state.bar_count = bar_count
            _engine_state.last_signal = signal
            _engine_state.last_confidence = round(confidence, 4)
