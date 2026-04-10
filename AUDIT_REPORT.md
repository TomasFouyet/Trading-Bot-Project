# AUDIT REPORT — Trading Bot Project
**Fecha:** 2026-04-06  
**Auditores:** TrendBot Inconsistency Inspector + TrendBot Strategy Auditor  
**Scope:** Todos los archivos del proyecto (`app/`, `scripts/`, `data/`, `tests/`)

---

## Resumen Ejecutivo

Se realizó una auditoría completa del proyecto en dos dimensiones complementarias:
- **Inconsistency Inspector**: detecta desajustes entre configuración, documentación, tipos y módulos.
- **Strategy Auditor**: detecta bugs de lógica de trading, cálculos erróneos y riesgos en ejecución.

Se identificaron **28 issues** en total. Los más graves afectan directamente el comportamiento en live trading:
- El cálculo de breakeven para posiciones SHORT está invertido, provocando salidas prematuras.
- Las posiciones SHORT generan PnL negativo cuando deberían ser positivas (fórmula errónea).
- El leverage por defecto en config es 5x en lugar de 1x según la documentación.
- El módulo `app.strategy` no tiene `__init__.py` activo — los imports fallan en runtime.
- Los límites de riesgo por defecto (drawdown 50%, position size 100%) contradicen la documentación y son peligrosos en producción.

**Estado general: NO APTO PARA LIVE TRADING** sin resolver los 5 issues CRÍTICOS.

---

## Tabla de Severidad

| Severidad  | Count | Áreas afectadas                                           |
|------------|-------|-----------------------------------------------------------|
| CRITICAL   | 5     | Module import, leverage, comisiones, BE SHORT, PnL SHORT  |
| HIGH       | 9     | Riesgo por defecto, registry, MACD NaN, EMA NaN, lookahead, precision Decimal, limit expiry, SELL doubling, equity negativa |
| MEDIUM     | 8     | Telegram alerts, on_bar() signature, timeframe duplication, signal metadata, risk coherence, BE constants, min_bars, DB lengths |
| LOW        | 6     | Float conversions, logging level, timezone handling, SQL (mitigated), __init__ exports, timeframe docs |
| **Total**  | **28**|                                                           |

---

## Bugs Críticos

### C-1 — `app.strategy` no tiene `__init__.py` activo
**Archivo:** `app/strategy/` (directorio completo)  
**Impacto:** `from app.strategy import get_strategy` falla en runtime. El bot no arranca.  
**Causa:** Existen `strategy__init__.py` y `not__init__.py` pero ningún `__init__.py` real.  
**Fix:** Renombrar `strategy__init__.py` → `__init__.py` y eliminar `not__init__.py`.

---

### C-2 — Leverage por defecto es 5x (debería ser 1x)
**Archivo:** `app/config.py:63`  
**Impacto:** Usuarios sin `.env` explícito operan con 5x leverage contra la documentación.  
```python
# Actual (peligroso):
leverage: int = Field(default=5, ge=1, le=125)
# Correcto:
leverage: int = Field(default=1, ge=1, le=125)
```

---

### C-3 — Comisión hardcodeada no usa la config
**Archivo:** `app/engine/position_manager.py:23-24`  
**Impacto:** Si se modifica `commission_bps` en config, el position manager no lo refleja. Divergencia backtest/live.  
```python
# Actual:
COMMISSION_RATE = Decimal("0.00075")  # hardcoded
# Correcto: importar desde _settings.commission_rate
```

---

### C-4 — Breakeven SL para SHORT está invertido
**Archivo:** `app/engine/position_manager.py:192-196`  
**Impacto:** En posiciones SHORT, el SL se mueve en dirección errónea tras TP1, cerrando runners prematuramente.  
```python
# Actual (invertido):
if il:  # LONG
    new_sl = float(entry + fee_buffer)
else:   # SHORT — WRONG: sube SL cuando debería bajar
    new_sl = float(entry - fee_buffer)

# Correcto:
if il:  # LONG: SL sube (breakeven)
    new_sl = float(entry + fee_buffer)
else:   # SHORT: SL baja (breakeven)
    new_sl = float(entry - fee_buffer)
# → invertir los dos casos
```

---

### C-5 — Fórmula PnL SHORT incorrecta en ForwardTestEngine
**Archivo:** `app/engine/forwardtest.py:230-235`  
**Impacto:** Trades SHORT rentables se registran como pérdidas. Las notificaciones Telegram reportan PnL erróneo.  
```python
# Actual (fórmula LONG aplicada a SHORT):
gross_pnl = (fill.price - entry_price) * close_qty

# Correcto:
if self._open_trade["side"] == "LONG":
    gross_pnl = (fill.price - entry_price) * close_qty
else:
    gross_pnl = (entry_price - fill.price) * close_qty
```

---

## Issues de Severidad HIGH

### H-1 — Daily drawdown por defecto: 50% (documentado: 2%)
**Archivo:** `app/config.py:66`  
`risk_max_daily_drawdown_pct: Decimal = Decimal("50.0")` → debe ser `Decimal("2.0")`

### H-2 — Position size por defecto: 100% (documentado: 5%)
**Archivo:** `app/config.py:67`  
`risk_max_position_pct: Decimal = Decimal("100")` → debe ser `Decimal("5.0")`

### H-3 — Strategy Registry duplicado e inconsistente
**Archivos:** `app/strategy/strategy__init__.py` vs `app/strategy/not__init__.py`  
El primero tiene 8 estrategias (correcto), el segundo solo 6. Solo uno puede ser `__init__.py`. Eliminar `not__init__.py`.

### H-4 — SELL signal no bloquea posición LONG abierta simultánea
**Archivo:** `app/engine/backtest.py:515`  
La check actual solo evita duplicar el mismo `pid`. Con `max_concurrent=1` y una posición LONG abierta, puede abrirse un SHORT simultáneo, duplicando el capital en riesgo.

### H-5 — MACD prev bar sin validación NaN
**Archivos:** `app/strategy/trend_following_v2.py:261`, `app/strategy/trend_following.py:259`  
`df["macd_hist"].iloc[-2]` puede ser NaN en bars iniciales → señales filtradas incorrectamente.

### H-6 — EMA sin check NaN antes de usar
**Archivos:** `app/strategy/trend_following_v2.py:252-253`, `app/strategy/trend_following.py:250`  
`float(row["ema_fast"])` sin validar NaN → `pb_atr` siempre > 0.5 → todos los setups rechazados en warm-up.

### H-7 — Lookahead bias potencial en HTF data
**Archivo:** `app/engine/backtest.py:381-386`  
Si `bar.ts` es el timestamp de apertura en lugar de cierre, el filtro HTF puede incluir datos del candle en formación.

### H-8 — Pérdida de precisión Decimal en SL/TP
**Archivo:** `app/strategy/trend_following.py:601-602`  
Aritmética float antes de convertir a Decimal acumula errores de redondeo en precios de SL/TP.

### H-9 — Risk Manager no valida equity negativa al abrir posición
**Archivo:** `app/risk/manager.py:104-111`  
No hay guard contra `current_equity <= 0` en `compute_order_qty()`. En condiciones extremas podría abrir posición con equity negativa.

---

## Issues de Severidad MEDIUM

| ID  | Archivo | Descripción |
|-----|---------|-------------|
| M-1 | `app/engine/forwardtest.py` | ForwardTestEngine no llama a TelegramNotifier → sin alertas en paper/live trading |
| M-2 | `app/engine/backtest.py:319-320` | Limit order expiry off-by-one: órdenes se quedan 1 bar más de lo configurado |
| M-3 | Múltiples strategy files | `on_bar()` tiene firmas inconsistentes: `Signal` vs `list[Signal]`, con/sin `htf_bias` |
| M-4 | `app/strategy/trend_following.py:581-594` | Metadata usa `tp1`/`tp2` pero backtest espera `tp1_price`/`tp2_price` |
| M-5 | `app/data/ingestor.py` + `bingx_client.py` | `TIMEFRAME_MAP` y `TIMEFRAME_SECONDS` duplicados sin fuente central |
| M-6 | `app/config.py` | Límites de riesgo incoherentes entre sí (daily 50%, position 100%, trade_risk 3%) |
| M-7 | `app/engine/position_manager.py:161-163` | Cálculo BE no incluye `slippage_bps` de config |
| M-8 | ORM models | `strategy_id: String(64)` puede truncarse con IDs dinámicos largos |

---

## Issues de Severidad LOW

| ID  | Archivo | Descripción |
|-----|---------|-------------|
| L-1 | `app/strategy/trend_following_v2.py:244-291` | Conversiones `float()` redundantes en loop de confidence scoring |
| L-2 | `app/engine/forwardtest.py:165` | Señales rechazadas loggeadas en `INFO` → invisibles con `WARN` level |
| L-3 | `app/engine/backtest.py:381-384` | Timezone handling innecesariamente complejo; asumir siempre UTC |
| L-4 | `app/engine/forwardtest.py:335-346` | `strategy_id` sin sanitizar (mitigado por ORM, riesgo mínimo) |
| L-5 | `app/config.py:62` | `default_timeframe` no documentado en README |
| L-6 | `app/__init__.py` | Empty `__init__.py` — exports explícitos mejorarían soporte IDE |

---

## Roadmap de Correcciones Priorizado

### Sprint 1 — Bloqueante / Antes de cualquier ejecución (1-2 días)

| Prioridad | Issue | Acción |
|-----------|-------|--------|
| 1 | C-1 | `mv app/strategy/strategy__init__.py app/strategy/__init__.py && rm app/strategy/not__init__.py` |
| 2 | C-4 | Invertir lógica LONG/SHORT en `position_manager.py:192-196` |
| 3 | C-5 | Corregir fórmula PnL SHORT en `forwardtest.py:230-235` |
| 4 | C-2 | `config.py:63` → `default=1` en leverage |
| 5 | C-3 | Reemplazar `COMMISSION_RATE` hardcodeado con `_settings.commission_rate` |

### Sprint 2 — Riesgo / Antes de live trading (2-3 días)

| Prioridad | Issue | Acción |
|-----------|-------|--------|
| 6 | H-1 | `config.py:66` → `Decimal("2.0")` para daily drawdown |
| 7 | H-2 | `config.py:67` → `Decimal("5.0")` para position size |
| 8 | H-3 | Eliminar `not__init__.py` (ya hecho en sprint 1) |
| 9 | H-4 | Añadir check `len(open_trades) >= max_concurrent` antes de SELL en backtest |
| 10 | H-9 | Añadir guard `current_equity <= 0` en `compute_order_qty()` |

### Sprint 3 — Calidad de señales (3-5 días)

| Prioridad | Issue | Acción |
|-----------|-------|--------|
| 11 | H-5 | Validar NaN en MACD prev bar (`trend_following.py`, `trend_following_v2.py`) |
| 12 | H-6 | Añadir `if pd.isna(row["ema_fast"]): continue` antes de calcular `pb_atr` |
| 13 | H-8 | Usar `Decimal` desde el inicio en cálculo SL/TP (no float) |
| 14 | H-7 | Documentar/enforzar que `bar.ts` es timestamp de cierre |
| 15 | M-2 | Corregir off-by-one en limit order expiry (`limit_entry_ages[id] = -1`) |
| 16 | M-4 | Unificar naming `tp1` → `tp1_price` en signal metadata |

### Sprint 4 — Observabilidad y mantenimiento (1 semana)

| Prioridad | Issue | Acción |
|-----------|-------|--------|
| 17 | M-1 | Integrar `TelegramNotifier` en `ForwardTestEngine` para todos los eventos |
| 18 | M-3 | Estandarizar firma `on_bar()` en todas las estrategias |
| 19 | M-5 | Extraer `TIMEFRAME_MAP`/`TIMEFRAME_SECONDS` a `app/data/timeframes.py` |
| 20 | M-6 | Hacer coherentes los defaults de riesgo entre sí |
| 21 | M-7 | Incluir `slippage_bps` en cálculo breakeven |
| 22 | L-2 | Cambiar `logger.info` → `logger.warning` en señales rechazadas |
| 23 | L-1 | Refactorizar conversiones float redundantes en confidence scoring |

---

## Archivos con Mayor Densidad de Issues

| Archivo | Issues | Severidad máxima |
|---------|--------|-----------------|
| `app/engine/position_manager.py` | 4 | CRITICAL |
| `app/config.py` | 4 | CRITICAL |
| `app/engine/forwardtest.py` | 3 | CRITICAL |
| `app/strategy/trend_following.py` | 3 | HIGH |
| `app/strategy/trend_following_v2.py` | 3 | HIGH |
| `app/engine/backtest.py` | 3 | HIGH |
| `app/risk/manager.py` | 2 | HIGH |
| `app/strategy/` (directorio) | 2 | CRITICAL |

---

*Generado automáticamente por auditoría paralela — 2026-04-06*
