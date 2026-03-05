# Trading Bot — BingX Full-Stack MVP

Production-grade trading bot for BingX supporting **backtest**, **paper trading**, and **live trading** modes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI (port 8000)                         │
│  /health  /symbols  /ohlcv  /strategy  /bot  /trades  /metrics     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    ┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼──────┐
    │  Celery Worker│  │  Celery Worker│  │  Celery Beat│
    │  (data/compute│  │  (trading)    │  │  (scheduler)│
    └───────┬───────┘  └───────┬───────┘  └─────────────┘
            │                  │
   ┌────────▼────────┐ ┌───────▼────────────────────┐
   │  ParquetStore   │ │  ForwardTestEngine / Live   │
   │  (OHLCV files)  │ │  BacktestEngine             │
   └────────┬────────┘ └───────┬────────────────────┘
            │                  │
   ┌────────▼──────────────────▼────────┐
   │           BrokerAdapter            │
   │  BingXAdapter │ PaperAdapter │     │
   │  BacktestAdapter                   │
   └────────────────────┬───────────────┘
                        │
              ┌─────────▼──────────┐
              │  BingXClient       │
              │  (HMAC auth,       │
              │   retry, rate lim) │
              └────────────────────┘

Postgres (trades, orders, fills, bot_state, backtest_runs)
Redis    (Celery broker + results)
Prometheus + Grafana (metrics + dashboards)
```

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- BingX API Key & Secret (for live data; paper mode works with keys for ticker prices)

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your BingX credentials
```

### 3. Start all services

```bash
docker compose up -d
```

Wait ~20 seconds for all services to be healthy, then verify:

```bash
curl http://localhost:8000/health
# → {"status":"ok","checks":{"api":"ok","database":"ok","redis":"ok"}}
```

### 4. Services

| Service      | URL                      | Description              |
|-------------|--------------------------|--------------------------|
| API          | http://localhost:8000    | FastAPI + Swagger docs   |
| Swagger UI   | http://localhost:8000/docs | API documentation      |
| Flower       | http://localhost:5555    | Celery task monitor      |
| Grafana      | http://localhost:3000    | Dashboards (admin/admin) |
| Prometheus   | http://localhost:9090    | Raw metrics              |

---

## Usage Examples

### A. Ingest historical data

```bash
# From Docker
docker compose exec api python scripts/ingest_data.py \
  --symbol BTC-USDT \
  --timeframe 5m \
  --start 2024-01-01 \
  --end 2024-06-01

# Or locally
python scripts/ingest_data.py --symbol BTC-USDT --timeframe 5m --start 2024-01-01 --end 2024-06-01
```

### B. Run a backtest

**Via API:**
```bash
curl -X POST http://localhost:8000/strategy/run_backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USDT",
    "timeframe": "5m",
    "start": "2024-01-01T00:00:00",
    "end": "2024-06-01T00:00:00",
    "strategy_name": "ema_cross",
    "strategy_params": {"ema_fast": 20, "ema_slow": 50, "sma_trend": 200},
    "initial_balance": 10000.0
  }'
# Returns: {"run_id": "...", "status": "submitted"}

# Check results
curl http://localhost:8000/strategy/backtest/{run_id}
```

**Via script (local, no Celery):**
```bash
python scripts/run_backtest.py \
  --symbol BTC-USDT \
  --timeframe 5m \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --output report.json
```

### C. Start paper trading

**Via API:**
```bash
curl -X POST http://localhost:8000/bot/start \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC-USDT", "timeframe": "5m", "mode": "paper"}'

# Check status
curl http://localhost:8000/bot/status

# Stop
curl -X POST http://localhost:8000/bot/stop
```

**Via script (local):**
```bash
python scripts/run_paper.py --symbol BTC-USDT --timeframe 5m --balance 10000
```

### D. View trades and metrics

```bash
# All trades
curl "http://localhost:8000/trades?mode=paper&limit=50"

# Performance summary
curl "http://localhost:8000/metrics/summary?mode=paper"
```

---

## Project Structure

```
trading-bot/
├── app/
│   ├── config.py              # Pydantic Settings (all env vars)
│   ├── main.py                # FastAPI app + lifespan
│   ├── api/routes/            # REST endpoints
│   ├── broker/
│   │   ├── base.py            # BrokerAdapter ABC + domain types
│   │   ├── bingx_client.py    # Raw HTTP client (auth, retry, rate limit)
│   │   ├── bingx_adapter.py   # Real trading adapter
│   │   ├── paper_adapter.py   # Paper trading adapter
│   │   └── backtest_adapter.py
│   ├── data/
│   │   ├── parquet_store.py   # Parquet OHLCV storage + DuckDB queries
│   │   ├── ingestor.py        # Historical + incremental ingestion
│   │   └── feed.py            # LiveFeed / ParquetFeed abstractions
│   ├── strategy/
│   │   ├── base.py            # BaseStrategy ABC
│   │   ├── signals.py         # Signal types
│   │   └── ema_cross.py       # EMA20/50 + SMA200 strategy
│   ├── risk/manager.py        # RiskManager + kill switch
│   ├── engine/
│   │   ├── backtest.py        # Deterministic backtest engine
│   │   ├── forwardtest.py     # Paper/live engine (same code, different adapter)
│   │   └── live.py            # Live engine (extends ForwardTest)
│   ├── tasks/                 # Celery async tasks
│   ├── metrics/prometheus.py  # Prometheus metric definitions
│   └── db/models/             # SQLAlchemy ORM models
├── tests/                     # Pytest unit tests
├── monitoring/                # Prometheus + Grafana configs
├── scripts/                   # CLI tools
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
└── .env.example
```

---

## Configuration Reference

All settings are in `.env`. Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `BINGX_API_KEY` | — | Required for paper/real |
| `BINGX_API_SECRET` | — | Required for paper/real |
| `BINGX_MARKET_TYPE` | `swap` | `swap` (perpetual) or `spot` |
| `BOT_MODE` | `paper` | `paper`, `real`, `backtest` |
| `LEVERAGE` | `1` | Always keep at 1 (no leverage) |
| `RISK_MAX_DAILY_DRAWDOWN_PCT` | `2.0` | Kill switch threshold |
| `RISK_MAX_POSITION_PCT` | `5.0` | Max % of equity per trade |
| `COMMISSION_BPS` | `7.5` | 0.075% per fill |
| `SLIPPAGE_BPS` | `5.0` | 0.05% slippage simulation |

---

## Running Tests

```bash
# Install dev deps
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

---

## Adding a New Strategy

1. Create `app/strategy/my_strategy.py` inheriting `BaseStrategy`
2. Implement `strategy_id`, `min_bars_required`, `on_bar()`
3. Register in `app/strategy/__init__.py` → `STRATEGY_REGISTRY`
4. Use via API: `strategy_name: "my_strategy"`

The same strategy will run identically in backtest, paper, and live modes — no code changes needed.

---

## Risk Manager Kill Switch Conditions

The bot stops automatically if any of these trigger:

| Condition | Threshold (configurable) |
|-----------|--------------------------|
| Daily drawdown | > `RISK_MAX_DAILY_DRAWDOWN_PCT`% |
| Consecutive API errors | > `RISK_MAX_CONSECUTIVE_API_ERRORS` |
| Data delay | > `RISK_DATA_DELAY_THRESHOLD_S` seconds |
| Manual stop | `POST /bot/stop` |

After a kill switch, reset via `risk_manager.reset_kill_switch()` (requires operator action).

---

## Production Notes

- **Never set `LEVERAGE > 1`** without full testing
- Use **paper mode** for at least 2 weeks before going real
- The `BotState` table has a single row (id=1) as a singleton
- All API keys are masked in logs — never printed in plaintext
- Parquet files are partitioned by month — safe to delete old partitions
- Backtest results include equity curve and full trade list in `report` JSON field
