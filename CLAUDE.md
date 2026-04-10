# TrendBot MTF — Claude Code Configuration

## Descripción del proyecto

Sistema de trading algorítmico multi-timeframe compuesto por:
- **Pine Script** (TradingView): Estrategia de señales y visualización
- **Python Bot**: Ejecución automática de órdenes en BingX
- **Estrategias**: TrendBot MTF (v7.0) + SmartWave v1.0 (SMC)

## Arquitectura del repositorio

```
Trading-Bot-Project/
├── pine_script/          # Estrategias Pine Script (.pine)
├── python/               # Bot de ejecución Python
│   ├── strategies/       # Lógica de estrategias
│   ├── run_multi_paper.py
│   └── ...
├── backtesting/          # Resultados y scripts de backtest
├── config/               # Configuración (parámetros, API keys template)
└── docs/                 # Documentación
```

*(Ajusta las rutas reales del proyecto si difieren)*

## Sub-Agent Routing Rules

### Paralelo (lanzar simultáneamente)
- Cuando se pide auditar el proyecto completo → lanzar ambos agentes en paralelo
- Análisis independientes que no dependen entre sí
- Investigación de un área mientras se trabaja en otra

### Secuencial
- Primero inspector de inconsistencias, luego correcciones → secuencial
- Tasks que requieren output de una para comenzar otra

### Background
- Análisis largos de backtesting
- Búsquedas exhaustivas en el codebase

## Comandos de auditoría frecuentes

### Auditoría completa (usar ambos agentes en paralelo)
```
Lanza trendbot-inconsistency-inspector y trendbot-strategy-auditor en paralelo
sobre todos los archivos del proyecto. Al terminar, consolida los hallazgos
en un reporte final unificado en AUDIT_REPORT.md
```

### Solo inconsistencias
```
Usa trendbot-inconsistency-inspector para revisar los archivos modificados recientemente
```

### Solo calidad
```
Usa trendbot-strategy-auditor para evaluar la lógica de la estrategia actual
```

## Convenciones del proyecto

- Pine Script: versión 5+
- Exchange: BingX (API REST)
- Pares principales: BTC/USDT, posiblemente multi-par
- Timeframes MTF: 1D (tendencia) / 4H (setup) / 15M (entrada)
- Modo paper trading disponible antes de live

## Notas para los agentes

- Los archivos Pine Script pueden tener extensión `.pine` o `.txt`
- Los parámetros clave suelen estar en la sección `input()` de Pine Script
  y en archivos de config o constantes al inicio de los `.py`
- Priorizar hallazgos que afecten la dirección del trade sobre los cosméticos
- El bot se conecta a BingX via API — verificar manejo de errores HTTP
