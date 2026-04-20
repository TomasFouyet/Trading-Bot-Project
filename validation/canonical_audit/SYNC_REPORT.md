# Sincronización del bot al baseline canónico — Informe

## Estado: CONVERGED

## Cambios aplicados
1. `rr_ratio`: runner `2.5 -> 2.7`, default constructor `2.0 -> 2.7`
2. Filtro HTF movido al strategy (bug de ghost trades eliminado)
3. Banda neutral `±0.2%` eliminada

## Métricas post-sync vs validador canónico

| Métrica | Validador canónico | Bot post-sync | Tolerancia | Estado |
| --- | ---: | ---: | ---: | --- |
| Total trades | 353 | 352 | ±5 trades | OK |
| Win rate | 36.83% | 36.93% | ±1 pp | OK |
| ExpR | +0.3348R | +0.3326R | ±0.02R | OK |
| Sharpe | 3.1828 | 3.1585 | ±0.15 | OK |
| Annual % | 79.40% | 78.30% | ±3.0% | OK |
| LONG trades | 173 | 173 | ±5 | OK |
| SHORT trades | 180 | 179 | ±5 | OK |
| Max drawdown | 11.88% | 11.88% | informativo | OK |

## Divergencias residuales

- No quedaron divergencias materiales frente al baseline canónico.
- La única diferencia visible es `352 vs 353` trades y `179 vs 180` shorts, ambas dentro de tolerancia y consistentes con una diferencia residual mínima en el orden operativo del wrapper barra-a-barra frente al harness vectorizado.
- No hay evidencia de deriva restante en entradas base ni en `structural_stop`.

## Callers de TrendFollowingV2Simple auditados

- `tests/test_simple_paper_runner.py:18` → `params=runner.VALIDATED_PARAMS` → hereda `rr_ratio=2.7`
- `validation/audit/python_runner.py:66` → `params=dict(VALIDATED_PARAMS)` desde `scripts/run_simple_paper.py` → hereda `rr_ratio=2.7`
- `scripts/run_simple_paper.py:1164` → `params=VALIDATED_PARAMS` → usa `rr_ratio=2.7`
- `validation/canonical_audit/run_canonical_audit.py:140` → `params=dict(RUNNER_PARAMS)` desde `scripts/run_simple_paper.py` → hereda `rr_ratio=2.7`

No se encontró ningún caller real del repositorio que pase un `rr_ratio != 2.7`.

## Commits aplicados

1. `c82e5b1` — `fix: sync rr_ratio to canonical 2.7 across runner and default`
2. `3efbd36` — `fix: move HTF filter into strategy to prevent ghost trades`
3. `71747cf` — `fix: remove ±0.2% neutral band in HTF bias for strict comparison`
4. `d06bef8` — `test: cover strict HTF bias after warmup`

## Recomendación

Proceder a escribir Pine Script replicando el estado sincronizado del bot.
