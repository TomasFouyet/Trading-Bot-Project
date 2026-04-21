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
- Además envía un mensaje Telegram de arranque confirmando símbolo, timeframe, equity y sesgo HTF.
- Luego envía un heartbeat cada 1 hora para confirmar que sigue corriendo.
- Si lo detienes con `Ctrl+C`, hace shutdown ordenado: guarda estado en `data/signal_service_state.json`, flushea logs y envía mensaje de stop por Telegram.
- En log verás una línea como:

```text
Service started | BTC-USDT 15m | equity=100.00 USDT | HTF bias=BULL
```

- Luego, en cada cierre de vela 15m:
  - `Bar closed | no signal | adx=... | htf=...`
  - o `Bar closed | signal=LONG|SHORT | ...`

- Si hay señal, Telegram recibe un mensaje con entrada, SL, TP, contexto ADX/body/MACD, sesgo HTF y sizing sugerido.
- Si un trade virtual toca TP o SL, Telegram también recibe el cierre correspondiente.

## Notificaciones que vas a recibir
- `ENTRY`: apertura LONG o SHORT con entrada, SL, TP y sizing sugerido.
- `EXIT TP`: cierre en take profit con PnL, duración y cooldown.
- `EXIT SL`: cierre en stop loss con PnL, duración y cooldown.
- `START`: mensaje de arranque `🚀` con símbolo, intervalo, equity y bootstrap.
- `STOP`: mensaje de detención `🛑` con velas procesadas, señales, cierres y uptime.
- `DAILY SUMMARY`: resumen UTC del día anterior con señales, cierres, PnL y uptime.
- `API ERROR`: alerta `⚠️` si BingX falla de forma persistente.

## Discrepancias manuales
El modo actual es **notificaciones puras**: no toca BingX, no abre ni cierra órdenes, solo observa y avisa.

Si ves una discrepancia:
- Telegram muestra cierre pero no ejecutaste manualmente: revisa `data/live_signals.csv` y `data/live_closures.csv`; por ahora la reconciliación es manual.
- Ejecutaste manualmente pero no ves cierre en Telegram: compara el timestamp de tu ejecución con la vela 15m cerrada y revisa `logs/signal_service.log`.
- Si necesitas corregir contexto para seguir operando, edita manualmente los CSVs. En una fase futura esto se hará con comandos Telegram.

## Cómo interpretar el log
- `bingx_fetch_retry`: fallo temporal de red o 5xx; el cliente reintenta hasta 3 veces.
- `dropping_in_progress_candle`: la última vela todavía estaba abierta y se descartó.
- `bingx_gap_detected`: faltan varias velas consecutivas en la respuesta pública.
- `Cycle complete`: el loop procesó una o más velas nuevas.

## Archivos generados
- `logs/signal_service.log`: log rotado diariamente.
- `data/live_signals.csv`: registro plano de señales enviadas.
- `data/live_closures.csv`: registro plano de cierres observados.
- `data/signal_service_state.json`: snapshot del estado al cerrar para recuperación ordenada.
