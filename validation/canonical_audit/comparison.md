# Canonical Comparison

## Tabla comparativa
| Métrica | Documentado | Validador actual | Bot actual |
| --- | ---: | ---: | ---: |
| Total trades | 353 | 353 | 247 |
| Win rate | 36.8% | 36.83% | 38.87% |
| ExpR | +0.335R | +0.335R | +0.383R |
| Sharpe | 3.20 | 3.183 | 3.556 |
| Annual % | 79.4% | 79.40% | 61.36% |
| Max drawdown | — | 11.88% | 12.85% |
| LONG / SHORT | — | 173 / 180 | 117 / 130 |

## Diff de código
- Líneas presentes solo en validator: 0
- Líneas presentes solo en bot: 3
- Líneas modificadas: 733
- Raw diff: [validator_vs_bot.diff](/home/tfouyet/projects/Trading-Bot-Project/validation/canonical_audit/validator_vs_bot.diff)

## Análisis por sección
- Condiciones de entrada: MAYORMENTE IDENTICAS. EMA/ADX/MACD/pullback/candle body/cooldown usan la misma formula base en fast_backtest y TrendFollowingV2Simple.
- Filtro HTF: DIFERENTE. El validador aplica HTF dentro de fast_backtest con sesgo binario close>EMA50/close<EMA50. El bot lo aplica afuera, en run_simple_paper.py, con banda neutral de ±0.2%.
- Cálculo de SL: IDENTICO. Ambos caminos llaman a validation.structural_stop.compute_structural_sl con pivots 3/3, buffer 0.25 ATR y min_risk 0.8 ATR.
- Cálculo de TP: DIFERENTE EN PARAMETRO. La fórmula es la misma (risk * rr_ratio), pero el baseline validado usa rr=2.7 y el runner actual pasa rr=2.5.
- Estado y ejecución: DIFERENTE. fast_backtest solo abre trade si el HTF ya aprobó la entrada. El bot actual deja que la estrategia mutile su estado interno antes del filtro HTF del runner, lo que puede crear ghost trades.
- Position sizing: DIFERENTE. El validador mide retornos por trade sin sizing de equity. El runner usa sizing por riesgo y leverage máximo.

## Impacto de divergencias
- El runner actual usa rr_ratio=2.5 en lugar de 2.7, lo que reduce payoff por ganador y normalmente baja ExpR y annual return.
- El filtro HTF del runner tiene una zona neutral ±0.2% alrededor de EMA50 4H, mientras el validador usa comparación estricta. Eso cambia qué señales pasan.
- Como el filtro HTF vive fuera de la estrategia, el bot puede quedar con un trade interno abierto aunque la entrada haya sido rechazada por el runner. Eso filtra señales posteriores de forma no validada.
- fast_backtest soporta modos ATR/HYBRID/trailing/regímenes porque también se usa como harness experimental; TrendFollowingV2Simple no expone todo eso en su interfaz deployable.

## Divergencias concretas y efecto esperado
- `rr_ratio`: el baseline canónico usa `2.7` en [run_structural_validation.py](/home/tfouyet/projects/Trading-Bot-Project/scripts/run_structural_validation.py:388), mientras el runner actual usa `2.5` en [run_simple_paper.py](/home/tfouyet/projects/Trading-Bot-Project/scripts/run_simple_paper.py:59). Impacto esperado: menos payoff por trade ganador, menos annual return, menos trades que alcanzan TP antes de reversión del mercado.
- Filtro HTF: el validador filtra antes de abrir trade en [fast_backtest.py](/home/tfouyet/projects/Trading-Bot-Project/validation/fast_backtest.py:359), pero el runner rechaza después de recibir la señal en [run_simple_paper.py](/home/tfouyet/projects/Trading-Bot-Project/scripts/run_simple_paper.py:1636). Impacto esperado: el strategy puede mutar estado interno antes de que el runner diga “no”.
- Banda neutral HTF: el runner declara `NEUTRAL` si el close 4H está dentro de ±0.2% de EMA50 en [run_simple_paper.py](/home/tfouyet/projects/Trading-Bot-Project/scripts/run_simple_paper.py:320), mientras `fast_backtest` usa comparación estricta `close > ema` / `close < ema` en [fast_backtest.py](/home/tfouyet/projects/Trading-Bot-Project/validation/fast_backtest.py:77). Impacto esperado: cambia qué señales pasan y cuáles se bloquean.
- Mutación de estado del strategy: `TrendFollowingV2Simple` fija `self._last_long_bar`, `self._last_short_bar` y puede abrir `_trade` dentro de [trend_following_v2_simple.py](/home/tfouyet/projects/Trading-Bot-Project/app/strategy/trend_following_v2_simple.py:345) y [trend_following_v2_simple.py](/home/tfouyet/projects/Trading-Bot-Project/app/strategy/trend_following_v2_simple.py:367). Impacto esperado: si luego el runner rechaza esa entrada por HTF, el strategy queda desincronizado con la ejecución real.
- Structural SL: el cálculo del stop sí coincide materialmente entre [fast_backtest.py](/home/tfouyet/projects/Trading-Bot-Project/validation/fast_backtest.py:390) y [trend_following_v2_simple.py](/home/tfouyet/projects/Trading-Bot-Project/app/strategy/trend_following_v2_simple.py:455). Impacto esperado: la deriva no viene del structural stop.
