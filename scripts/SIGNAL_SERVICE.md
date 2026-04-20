# Signal Service TrendBot

## Configuración
1. Crea un bot en Telegram con `@BotFather` usando `/newbot`.
2. Envía al menos un mensaje a tu bot desde tu cuenta.
3. Abre `https://api.telegram.org/bot<TOKEN>/getUpdates` y toma tu `chat_id`.
4. Completa tu `.env` con:
   - `TELEGRAM_BOT_TOKEN=...`
   - `TELEGRAM_CHAT_ID=...`
   - `PAPER_EQUITY_USD=100.0`
   - `SYMBOL=BTC-USDT`
   - `INTERVAL=15m`
   - `LOG_LEVEL=INFO`

`.env` ya está ignorado por git en este repo.

## Cómo ejecutar
Bootstrap local sin enviar Telegram:

```bash
python scripts/run_signal_service.py --once --dry-run
```

Servicio continuo:

```bash
python scripts/run_signal_service.py
```

## Qué esperar
- Al arrancar hace un bootstrap silencioso con histórico reciente para reconstruir el estado del strategy.
- En log verás una línea como:

```text
Service started | BTC-USDT 15m | equity=100.00 USDT | HTF bias=BULL
```

- Luego, en cada cierre de vela 15m:
  - `Bar closed | no signal | adx=... | htf=...`
  - o `Bar closed | signal=LONG|SHORT | ...`

- Si hay señal, Telegram recibe un mensaje con entrada, SL, TP, contexto ADX/body/MACD, sesgo HTF y sizing sugerido.

## Cómo interpretar el log
- `bingx_fetch_retry`: fallo temporal de red o 5xx; el cliente reintenta hasta 3 veces.
- `dropping_in_progress_candle`: la última vela todavía estaba abierta y se descartó.
- `bingx_gap_detected`: faltan varias velas consecutivas en la respuesta pública.
- `Cycle complete`: el loop procesó una o más velas nuevas.

## Archivos generados
- `logs/signal_service.log`: log rotado diariamente.
- `data/live_signals.csv`: registro plano de señales enviadas.
