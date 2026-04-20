# Auditoría Python ↔ Pine Structural — Informe

## Resumen ejecutivo
- Match rate: 35.71%
- Veredicto: RESET_REQUERIDO

## Criterios de veredicto
- MATCH_EXCELENTE: match rate >= 95% Y ningún TYPE_MISMATCH Y diff indicadores todos dentro de threshold. Acción: continuar paper trading.
- MATCH_ACEPTABLE: match rate 85-95%, divergencias explicables por drift numérico en barras iniciales. Acción: continuar paper pero documentar las discrepancias conocidas.
- BUG_REPRODUCIBLE: match rate 50-85% con causa raíz identificada. Acción: pausar paper, arreglar el lado erróneo, re-auditar.
- RESET_REQUERIDO: match rate <50% O cualquier TYPE_MISMATCH. Acción: pausar paper, revisar arquitectura, re-validar desde WFA.

## Números clave

| Métrica | Valor |
| --- | ---: |
| Total señales Python | 6 |
| Total señales Pine | 14 |
| Total MATCH | 5 |
| PYTHON_ONLY | 1 |
| PINE_ONLY | 9 |
| TYPE_MISMATCH | 0 |
| Avg ATR diff % | 0.000 |
| Max ATR diff % | 0.000 |
| P95 ATR diff % | 0.000 |
| Avg ADX diff | 0.000 |
| Max ADX diff | 0.000 |
| P95 ADX diff | 0.000 |
| Avg EMA50 diff % | 0.000 |
| Max EMA50 diff % | 0.000 |
| P95 EMA50 diff % | 0.000 |
| Avg MACD hist diff % | 0.000 |
| Max MACD hist diff % | 0.000 |
| P95 MACD hist diff % | 0.000 |

## Top causas de divergencia
- ghost_trade_after_htf_rejection: 9 casos
- htf_alignment: 1 casos

## Recomendación concreta
Pausar paper trading y revisar arquitectura/contrato de senales antes de seguir.

## Diagnóstico detallado
- Bar 82 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
- Bar 346 - PINE_ONLY SHORT: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BEAR.
- Bar 536 - PINE_ONLY SHORT: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BEAR.
- Bar 749 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
- Bar 1254 - PYTHON_ONLY LONG: Pine rejected because htf_alignment. Pine HTF=BEAR.
- Bar 1443 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
- Bar 1526 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
- Bar 1548 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
- Bar 1622 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
- Bar 1954 - PINE_ONLY LONG: Python rejected because ghost_trade_after_htf_rejection. Python HTF=BULL.
