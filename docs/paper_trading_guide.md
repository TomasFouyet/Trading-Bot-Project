# Paper Trading V2 — Guia de Ejecucion

## Que es y para que sirve

`scripts/run_multi_paper_v2.py` es el runner principal de paper trading. Simula trades en tiempo real usando precios live de BingX **sin enviar ordenes reales**. Tambien soporta modo live (`--live-bingx`) que envia ordenes reales con USDT real.

La estrategia que ejecuta es **TrendFollowingV2** (Pine Script v5.1 aligned), definida en `app/strategy/trend_following_v2.py`. La estrategia controla toda la logica de trade internamente (SL, TP1, TP2, trailing, BE, reversal). El runner solo alimenta barras, ejecuta las senales y notifica.

---

## Arquitectura del sistema

```
                    BingX API (precios live)
                          |
                    BingXClient (HTTP)
                     /          \
              PaperAdapter      BingXAdapter
              (simulado)        (ordenes reales)
                     \          /
                 run_multi_paper_v2.py
                          |
              +-----------+-----------+
              |           |           |
          Worker BTC  Worker ETH  Worker SOL ...
              |           |           |
          TrendFollowingV2 (por simbolo)
              |
          +---+---+
          |       |
      HTFContext  RiskManager
      (bias 4H)  (kill switch)
              |
        TelegramNotifier
        (alertas)
```

### Componentes clave

| Componente | Archivo | Funcion |
|---|---|---|
| **Runner** | `scripts/run_multi_paper_v2.py` | Orquesta todo: carga datos, alimenta barras, ejecuta senales, loguea trades |
| **Estrategia** | `app/strategy/trend_following_v2.py` | Logica completa de entrada/salida (EMA+ADX+MACD+pullback, SL/TP1/TP2/trailing) |
| **Paper Adapter** | `app/broker/paper_adapter.py` | Simula fills usando precio live + slippage + comisiones |
| **BingX Client** | `app/broker/bingx_client.py` | Conexion HTTP a BingX API (ticker, OHLCV, ordenes) |
| **Live Feed** | `app/data/feed.py` | Polling de barras cerradas por timeframe (async generator) |
| **HTF Context** | `app/strategy/mtf_context.py` | Calcula bias 4H (EMA50/200 + ADX) para filtro multi-timeframe |
| **Risk Manager** | `app/risk/manager.py` | Kill switch diario (5% DD), limite por posicion, max trades |
| **Telegram** | `app/notify/telegram.py` | Notificaciones de apertura, TP1, cierre, y comando `/equity` |
| **Config** | `app/config.py` + `.env` | API keys, comisiones, slippage, limites de riesgo |

---

## Como ejecutar

### Pre-requisitos

1. Tener `.env` configurado con API keys de BingX:
   ```
   BINGX_API_KEY=tu_api_key
   BINGX_API_SECRET=tu_api_secret
   ```

2. (Opcional) Para notificaciones Telegram:
   ```
   TELEGRAM_BOT_TOKEN=123456:AAxxxxxxx
   TELEGRAM_CHAT_ID=987654321
   ```

3. Entorno Python con dependencias instaladas:
   ```bash
   source .venv/bin/activate
   ```

### Comandos

```bash
# Verificar conexion con BingX
python scripts/run_multi_paper_v2.py --check

# Paper trading (simulado, sin dinero real)
python scripts/run_multi_paper_v2.py

# Solo BTC y ETH, con balance inicial de $1000
python scripts/run_multi_paper_v2.py --symbols BTC ETH --balance 1000

# LIVE — ordenes reales en BingX (CUIDADO: usa USDT real)
python scripts/run_multi_paper_v2.py --live-bingx
```

### Parametros disponibles

| Parametro | Default | Descripcion |
|---|---|---|
| `--symbols` | BTC ETH SOL XRP BNB | Pares a monitorear (se agrega `-USDT`) |
| `--timeframe` | 15m | Timeframe de las barras |
| `--balance` | 500.0 | Balance inicial simulado (solo paper) |
| `--max-positions` | 3 | Maximo de posiciones simultaneas |
| `--alloc-pct` | 20.0 | % del equity maximo por posicion |
| `--lev-a` | 3 | Leverage para senales Tier A (conf >= 0.50) |
| `--lev-b` | 2 | Leverage para senales Tier B |
| `--lev-c` | 1 | Leverage para senales Tier C |
| `--params-file` | None | JSON con override de parametros de estrategia |
| `--check` | — | Solo verifica conexion a BingX y sale |
| `--live-bingx` | — | Envia ordenes REALES a BingX |

---

## Flujo de ejecucion paso a paso

### 1. Inicializacion (al arrancar)

```
main()
  |
  +-- Crea BingXClient con API keys de .env
  +-- Crea PaperAdapter (o BingXAdapter si --live-bingx)
  +-- Crea TelegramNotifier (si configurado)
  +-- Para cada simbolo: lanza run_symbol() como asyncio Task
  +-- Lanza _status_loop() (imprime estado cada 5 min)
  +-- Lanza _telegram_command_loop() (escucha /equity)
```

### 2. Por cada simbolo: run_symbol()

```
run_symbol(BTC-USDT)
  |
  +-- Carga warmup: 300 barras historicas de 15m via BingX API
  +-- Inicializa TrendFollowingV2 con BEST_PARAMS
  +-- Inicializa HTFContext: carga barras 4H y calcula bias
  +-- Inicializa RiskManager con equity actual
  +-- Entra en loop infinito:
       |
       +-- Espera nueva barra de 15m (LiveFeed.stream())
       +-- Agrega barra al buffer circular (300 barras max)
       +-- Refresca HTF bias (cada 4h)
       +-- Llama strategy.on_bar_all(df, htf_bias)
       +-- Procesa senales devueltas:
            |
            +-- HOLD: solo imprime estado
            +-- BUY/SELL: abre posicion
            +-- PARTIAL_CLOSE: cierra 33% en TP1
            +-- CLOSE: cierra posicion (SL/TP2/reversal)
```

### 3. Ciclo de vida de un trade

```
Barra N:   Estrategia emite BUY (conf=0.65, tier=A)
           |
           +-- RiskManager valida: equity suficiente, no kill switch
           +-- Calcula qty: equity * alloc_pct * leverage / price
           +-- PaperAdapter simula fill a precio live + slippage
           +-- Registra open_trade con SL, TP1, TP2
           +-- Si --live-bingx: coloca stop-loss en BingX como proteccion
           +-- Telegram: envia notificacion de apertura
           
Barra N+X: Estrategia emite PARTIAL_CLOSE (TP1 hit)
           |
           +-- Cierra 33% de la posicion al precio de TP1
           +-- Mueve SL a break-even (la estrategia lo maneja internamente)
           +-- Activa trailing stop
           
Barra N+Y: Estrategia emite CLOSE (SL/TP2/trailing/reversal)
           |
           +-- Cierra posicion restante
           +-- Calcula PnL neto (entrada + parciales - comisiones)
           +-- Graba trade en data/paper_trades_v2.csv
           +-- Actualiza W/L counter
           +-- Telegram: envia notificacion de cierre con PnL
```

---

## Parametros de la estrategia (BEST_PARAMS)

Los parametros estan hardcodeados al inicio de `run_multi_paper_v2.py`:

```python
BEST_PARAMS = {
    # Entrada
    "adx_min": 20,              # ADX minimo para generar senal
    "adx_strong": 35,           # ADX para tier A
    "pullback_tolerance_atr": 1.0,  # Proximidad a EMA fast en ATRs
    "allow_short": True,        # Permitir shorts
    "min_confidence": 0.50,     # Solo Tier A (filtro de calidad)
    "sig_cooldown": 5,          # Barras entre senales
    "enable_reversal": True,    # Swap LONG<->SHORT sin cerrar primero
    
    # Stop Loss
    "sl_swing_lookback": 50,    # Barras para buscar swing H/L
    "sl_min_atr": 1.5,          # SL minimo en ATRs
    "sl_max_atr": 3.0,          # SL maximo en ATRs
    "sl_min_pct": 0.015,        # SL minimo 1.5% del precio
    
    # Take Profit (por tier)
    "tp1_r_A": 1.5, "tp2_r_A": 3.0,  # Tier A: TP1=1.5R, TP2=3.0R
    "tp1_r_B": 1.5, "tp2_r_B": 2.5,  # Tier B: TP1=1.5R, TP2=2.5R
    "tp1_r_C": 1.0, "tp2_r_C": 1.5,  # Tier C: TP1=1.0R, TP2=1.5R
}
```

---

## Donde se guardan los resultados

| Archivo | Contenido |
|---|---|
| `data/paper_trades_v2.csv` | Log de cada trade: entry/exit price, PnL, fees, tipo de salida, tier, leverage |
| Console (stdout) | Estado en tiempo real: precio, ADX, posiciones abiertas, uPnL |
| Telegram | Notificaciones push de cada apertura, TP1, y cierre |

### Formato del CSV

```
trade_id, symbol, side, strategy, tier, leverage,
entry_time, entry_price, exit_time, exit_price, exit_type,
qty, pnl, pnl_pct, fees, signal_reason
```

---

## Modos de ejecucion

### Paper (default)

- Usa `PaperAdapter` — simula fills en memoria
- Precios reales de BingX (live ticker)
- Slippage simulado: 0.05% (configurable en `.env` via `SLIPPAGE_BPS`)
- Comisiones simuladas: 0.075% taker por leg
- Balance inicial: $500 (configurable con `--balance`)
- **No toca dinero real. Seguro para probar.**

### Live (`--live-bingx`)

- Usa `BingXAdapter` — envia ordenes reales via API
- Market orders para entrada/salida
- Stop-loss colocado en el exchange como proteccion ante crash del bot
- Leverage configurado por tier (A=3x, B=2x, C=1x por default)
- **USA USDT REAL. Verificar con `--check` antes de activar.**

---

## Protecciones de riesgo

El `RiskManager` aplica estos controles antes de cada trade:

| Control | Valor default | Que hace |
|---|---|---|
| Max daily drawdown | 5% | Kill switch: para de operar si pierde 5% del equity en el dia |
| Max posicion | 20% del equity | No permite mas de 20% de margen en una posicion |
| Max riesgo por trade | 1% del equity | Basado en distancia al SL |
| Max errores API consecutivos | 5 | Kill switch ante problemas de conexion |
| Data delay | 30 min | Kill switch si las barras llegan con mas de 30 min de retraso |

---

## Diferencia con run_multi_paper.py (V1)

| Aspecto | V1 (`run_multi_paper.py`) | V2 (`run_multi_paper_v2.py`) |
|---|---|---|
| Logica SL/TP | En el runner (position_manager) | En la estrategia (TrendFollowingV2) |
| Fuente de verdad | Runner + estrategia (dividido) | Solo la estrategia |
| Session filter | ON | OFF (alineado con Pine Script) |
| Streak adjuster | ON | OFF (Pine Script no lo usa) |
| Patience (soft SL) | ON (48 barras) | OFF (SL inmediato como Pine) |
| min_confidence | 0 (todo pasa) | 0.50 (solo Tier A) |

**V2 es la version recomendada.** Esta alineada 1:1 con Pine Script v5.1 y la estrategia es la unica fuente de verdad para la logica de trade.

---

## Nota sobre la estrategia validada

La validacion estadistica (WFA, Monte Carlo, Permutation test) se hizo sobre una **version simplificada** (`TrendFollowingV2Simple` en `validation/fast_backtest.py`) con estos parametros:

- `rr_ratio=1.5`, `atr_sl_mult=2.0`, HTF=ON (4H EMA50)
- Sin TP1/TP2/trailing/BE — solo SL fijo y TP fijo
- Resultados: WFA 5/5, MC RoR 0.9%, Permutation p=0.001

El paper trading V2 ejecuta la version **completa** (con TP1, TP2, trailing, tiers), que tiene mas complejidad. Los resultados de paper trading pueden diferir del backtest validado por:

1. Exits mas complejos (TP1 parcial + trailing vs TP fijo)
2. Tier sizing (A/B/C) vs posicion fija
3. Reversal swap (V2 puede invertir posicion)
4. Slippage y latencia real vs simulacion perfecta

Para cerrar esta brecha, seria necesario crear una version deployable de `TrendFollowingV2Simple` con la logica exacta que fue validada.
