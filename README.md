# TradingEngine

A fully-integrated futures trading engine with IBKR data ingestion, ML signal
generation (XGBoost + Optuna), vectorised backtesting, paper trading simulation,
and a REST API for Base44 integration.

**PAPER TRADING ONLY** – no live order placement.

---

## macOS (Apple Silicon) Setup

```bash
# 1. Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Upgrade build tools
python -m pip install -U pip setuptools wheel

# 3. Install dependencies
python -m pip install -r requirements.txt
```

---

## Workflow

### Step 1 – Fetch Minute Bars from IBKR

Requires TWS or IB Gateway running on **paper port 7497**.

```bash
python -m src.cli fetch --symbol ES --days 10
# Output: data/raw/ES_M1.csv
```

**If TWS is not running**, you will see:
```
Connection refused on 127.0.0.1:7497 – is TWS / IB Gateway running?
  Paper trading: port 7497 (TWS) or 4002 (Gateway)
```

**To test connectivity only:**
```bash
python -m src.data_engine.ibkr_fetch
```

**To use synthetic data for development (no IBKR needed):**
```bash
python scripts/generate_synthetic_data.py --symbol ES --bars 5000
```

### Step 2 – Build Dataset (Features + Labels)

```bash
python -m src.cli build-dataset \
    --symbol ES \
    --input data/raw/ES_M1.csv
# Output:
#   data/processed/ES_features.parquet
#   data/processed/ES_labels.parquet
```

Optional flags:
- `--pt 1.5`       profit-take multiplier (default 1.5)
- `--sl 1.0`       stop-loss multiplier (default 1.0)
- `--max-hold 60`  max holding period in bars (default 60)

### Step 3 – Train Model

```bash
python -m src.cli train --symbol ES --trials 5
# Output:
#   artifacts/models/ES_xgb_best.joblib
#   artifacts/scalers/ES_scaler.joblib
#   artifacts/schema/ES_features.json
```

### Step 4 – Backtest

```bash
python -m src.cli backtest --universe config/universe.yaml
# Output: artifacts/reports/ES_backtest.json
```

### Step 5 – Paper Engine (CSV Stream Simulation)

```bash
python -m src.cli live-paper --symbol ES --input data/raw/ES_M1.csv
```

### Step 6 – Start API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

API endpoints:

| Method | Path            | Description                       |
|--------|-----------------|-----------------------------------|
| GET    | /health         | Liveness check                    |
| GET    | /status         | Engine + training status          |
| GET    | /metrics        | Last backtest metrics             |
| GET    | /equity         | Equity curve                      |
| GET    | /trades         | Recent trades                     |
| POST   | /engine/start   | Start paper engine                |
| POST   | /engine/stop    | Stop paper engine                 |
| POST   | /train          | Trigger training                  |
| POST   | /backtest       | Trigger backtest                  |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `config/ibkr.yaml`    | IBKR host/port/clientId + contract overrides |
| `config/universe.yaml`| Symbols + cost model + contract specs        |

---

## Project Structure

```
TradingEngine/
├── config/
│   ├── ibkr.yaml           # IBKR connection settings
│   └── universe.yaml       # Trading universe
├── data/
│   ├── raw/                # Raw CSV bars from IBKR
│   └── processed/          # Feature + label parquets
├── artifacts/
│   ├── models/             # Trained XGBoost models
│   ├── scalers/            # Feature scalers
│   ├── schema/             # Feature name + label mapping
│   └── reports/            # Backtest reports (JSON)
├── src/
│   ├── cli.py              # Main CLI entry point
│   ├── api/app.py          # FastAPI REST API
│   ├── backtest/engine.py  # Vectorised backtest
│   ├── data_engine/ibkr_fetch.py  # IBKR data fetch
│   ├── features/builder.py # Feature engineering
│   ├── labels/triple_barrier.py   # Triple-barrier labels
│   ├── live/paper_engine.py       # Paper trading simulation
│   └── training/train.py   # Optuna + XGBoost training
├── tests/
│   └── test_labeler.py     # Unit tests
├── requirements.txt
└── README.md
```

---

## IBKR Setup Notes

1. Open TWS → Edit → Global Configuration → API → Settings
2. Enable "Enable ActiveX and Socket Clients"
3. Ensure port is **7497** for paper trading
4. Add `127.0.0.1` to trusted IPs (or use "Allow connections from localhost only")

For IB Gateway (headless), paper port is **4002** – update `config/ibkr.yaml`.

---

## Cost Model

| Item           | Value       |
|----------------|-------------|
| Commission     | $2.05 / contract / side |
| Slippage       | 1 tick / fill (0.25 pts = $12.50 for ES) |
| Tick size (ES) | 0.25 points |
| Multiplier (ES)| $50 / point |