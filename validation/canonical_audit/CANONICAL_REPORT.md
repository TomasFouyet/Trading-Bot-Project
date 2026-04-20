# Auditoría de Código Canónico — Informe

## Veredicto: BOT_HAS_DRIFTED

## Métricas reproducidas
| Métrica | Documentado | Validador actual | Bot actual |
| --- | ---: | ---: | ---: |
| Total trades | 353 | 353 | 247 |
| Win rate | 36.8% | 36.83% | 38.87% |
| ExpR | +0.335R | +0.335R | +0.383R |
| Sharpe | 3.20 | 3.183 | 3.556 |
| Annual % | 79.4% | 79.40% | 61.36% |
| Max drawdown | — | 11.88% | 12.85% |
| LONG / SHORT | — | 173 / 180 | 117 / 130 |

## Estado del código
- Archivos auditados: validation/fast_backtest.py, validation/structural_stop.py, app/strategy/trend_following_v2_simple.py, scripts/run_simple_paper.py
- Líneas divergentes encontradas: 736
- Impacto funcional estimado: Hay deriva funcional material entre validador canónico y bot deployable.

## Acción recomendada
Revertir o sincronizar el bot al baseline canónico antes de continuar con Pine o live trading.

Concretamente:
- Restaurar `rr_ratio=2.7` si el objetivo es volver al baseline estadísticamente validado.
- Mover el filtro HTF para que ocurra antes de mutar el estado interno del strategy, o integrar el HTF dentro del strategy deployable.
- Volver a correr esta auditoría después de sincronizar esos dos puntos.

## Siguientes pasos
Si BOT_HAS_DRIFTED → revertir/sincronizar el bot al validator antes de continuar.
