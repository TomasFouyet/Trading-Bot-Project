from __future__ import annotations

import difflib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.strategy.signals import SignalAction
from app.strategy.trend_following_v2_simple import TrendFollowingV2Simple
from scripts.run_simple_paper import HTF_CONFIG, VALIDATED_PARAMS as RUNNER_PARAMS, WINDOW_SIZE
from scripts.run_structural_validation import BASE_PARAMS, expectancy_r
from validation.data_loader import load_candles
from validation.fast_backtest import _build_metrics, compute_htf_bias, compute_indicators, fast_backtest
from validation.strategy_adapter import TradeRecord


OUT_DIR = ROOT / "validation" / "canonical_audit"
COMPARISON_MD = OUT_DIR / "comparison.md"
REPORT_MD = OUT_DIR / "CANONICAL_REPORT.md"
RAW_DIFF = OUT_DIR / "validator_vs_bot.diff"
SUMMARY_JSON = OUT_DIR / "summary.json"

VALIDATOR_FILES = [
    ROOT / "validation" / "fast_backtest.py",
    ROOT / "validation" / "structural_stop.py",
]
BOT_FILES = [
    ROOT / "app" / "strategy" / "trend_following_v2_simple.py",
    ROOT / "scripts" / "run_simple_paper.py",
]

DOCUMENTED = {
    "total_trades": 353,
    "winrate": 36.8,
    "expr": 0.335,
    "annual_return_pct": 79.4,
    "sharpe_ratio": 3.20,
}

CANONICAL_CFG = dict(
    stop_mode="STRUCTURAL",
    rr_ratio=2.7,
    atr_sl_mult=2.0,
    buffer_atr=0.25,
    min_risk_atr=0.8,
    pivot_left=3,
    pivot_right=3,
)


@dataclass
class MetricBundle:
    total_trades: int
    winrate: float
    expr: float
    sharpe_ratio: float
    max_drawdown_pct: float
    annual_return_pct: float
    long_trades: int
    short_trades: int


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_candles("BTC/USDT", "15m", days=730)

    validator_metrics = run_validator(df)
    validator_reproduces = validator_within_tolerance(validator_metrics)

    if not validator_reproduces:
        git_log = get_validation_git_log()
        verdict = "VALIDATOR_ALSO_DRIFTED"
        write_reports(
            verdict=verdict,
            validator_metrics=validator_metrics,
            bot_metrics=None,
            documented=DOCUMENTED,
            git_log=git_log,
            diff_summary=None,
        )
        return

    bot_metrics = run_bot_wrapper(df)
    diff_summary = build_diff_summary()
    verdict = (
        "CANONICAL_MATCH"
        if metrics_within_5pct(validator_metrics, bot_metrics)
        else "BOT_HAS_DRIFTED"
    )
    write_reports(
        verdict=verdict,
        validator_metrics=validator_metrics,
        bot_metrics=bot_metrics,
        documented=DOCUMENTED,
        git_log=None,
        diff_summary=diff_summary,
    )


def run_validator(df: pd.DataFrame) -> MetricBundle:
    df_ind = compute_indicators(df)
    htf = compute_htf_bias(df_ind)
    metrics = fast_backtest(
        df_ind,
        precomputed=True,
        htf_bias=htf,
        **BASE_PARAMS,
        **CANONICAL_CFG,
    )
    return bundle_metrics(metrics)


def compute_runner_htf_bias(df_15m: pd.DataFrame, ema_period: int = 50) -> pd.Series:
    df_htf = (
        df_15m.set_index("ts")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    ema = df_htf["close"].ewm(span=ema_period, adjust=False).mean()
    bias = pd.Series(0, index=df_htf.index, dtype="int8")
    bias[df_htf["close"] > ema * 1.002] = 1
    bias[df_htf["close"] < ema * 0.998] = -1

    aligned = df_15m[["ts"]].set_index("ts").join(bias.rename("htf_bias"), how="left")
    aligned["htf_bias"] = aligned["htf_bias"].ffill().fillna(0).astype("int8")
    return aligned["htf_bias"]


def run_bot_wrapper(df: pd.DataFrame) -> MetricBundle:
    htf = compute_runner_htf_bias(df, HTF_CONFIG["ema_period"]).to_numpy()
    strategy = TrendFollowingV2Simple("BTC-USDT", params=dict(RUNNER_PARAMS))
    if hasattr(strategy, "force_close"):
        strategy.force_close()

    trades: list[TradeRecord] = []
    open_trade: dict[str, Any] | None = None
    entry_bar_idx: int | None = None

    total = len(df)
    for i in range(WINDOW_SIZE, total):
        if i % 5000 == 0:
            print(f"[bot-wrapper] processed {i}/{total} bars", flush=True)
        window = df.iloc[i - WINDOW_SIZE + 1 : i + 1]
        signals = strategy.on_bar_all(window)
        row = df.iloc[i]
        close = float(row["close"])
        ts = row["ts"]
        bias = int(htf[i])

        for sig in signals:
            if sig.action == SignalAction.CLOSE:
                if open_trade is None or entry_bar_idx is None:
                    continue
                exit_price = float(sig.meta.get("exit_price", close))
                trades.append(
                    TradeRecord(
                        direction=open_trade["direction"],
                        entry_ts=open_trade["entry_ts"],
                        entry_price=open_trade["entry_price"],
                        exit_ts=ts,
                        exit_price=exit_price,
                        pnl_pct=0.0,
                        exit_type=sig.meta.get("exit_type", "close"),
                        confidence=open_trade["confidence"],
                        sl=open_trade["sl"],
                        tp1=open_trade["tp1"],
                        bars_held=i - entry_bar_idx,
                        entry_bar_idx=entry_bar_idx,
                        sl_mode=open_trade["sl_mode"],
                    )
                )
                open_trade = None
                entry_bar_idx = None
                continue

            if sig.action not in (SignalAction.BUY, SignalAction.SELL):
                continue

            direction = "LONG" if sig.action == SignalAction.BUY else "SHORT"
            aligned = bias == 0 or (direction == "LONG" and bias == 1) or (direction == "SHORT" and bias == -1)
            if not aligned:
                continue
            if open_trade is not None:
                continue

            open_trade = {
                "direction": direction,
                "entry_ts": ts,
                "entry_price": close,
                "confidence": float(sig.meta.get("confidence_score", sig.confidence)),
                "sl": float(sig.meta.get("sl", sig.stop_loss or 0.0)),
                "tp1": float(sig.meta.get("tp1", sig.take_profit or 0.0)),
                "sl_mode": sig.meta.get("sl_mode", ""),
            }
            entry_bar_idx = i

    if open_trade is not None and entry_bar_idx is not None:
        last = df.iloc[-1]
        trades.append(
            TradeRecord(
                direction=open_trade["direction"],
                entry_ts=open_trade["entry_ts"],
                entry_price=open_trade["entry_price"],
                exit_ts=last["ts"],
                exit_price=float(last["close"]),
                pnl_pct=0.0,
                exit_type="end_of_data",
                confidence=open_trade["confidence"],
                sl=open_trade["sl"],
                tp1=open_trade["tp1"],
                bars_held=len(df) - 1 - entry_bar_idx,
                entry_bar_idx=entry_bar_idx,
                sl_mode=open_trade["sl_mode"],
            )
        )

    for trade in trades:
        if trade.direction == "LONG":
            trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100

    metrics = _build_metrics(trades, df)
    return bundle_metrics(metrics)


def bundle_metrics(metrics) -> MetricBundle:
    return MetricBundle(
        total_trades=metrics.total_trades,
        winrate=float(metrics.winrate),
        expr=float(expectancy_r(metrics)),
        sharpe_ratio=float(metrics.sharpe_ratio),
        max_drawdown_pct=float(metrics.max_drawdown_pct),
        annual_return_pct=float(metrics.annual_return_pct),
        long_trades=sum(1 for t in metrics.trades if t["direction"] == "LONG"),
        short_trades=sum(1 for t in metrics.trades if t["direction"] == "SHORT"),
    )


def pct_diff(actual: float, expected: float) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(actual - expected) / abs(expected) * 100.0


def validator_within_tolerance(bundle: MetricBundle) -> bool:
    return all(
        pct_diff(getattr(bundle, key), DOCUMENTED[key]) <= 5.0
        for key in ("total_trades", "winrate", "expr", "annual_return_pct")
    )


def metrics_within_5pct(a: MetricBundle, b: MetricBundle) -> bool:
    return all(
        pct_diff(getattr(a, key), getattr(b, key)) <= 5.0
        for key in ("total_trades", "winrate", "expr", "annual_return_pct")
    )


def get_validation_git_log() -> str:
    cmd = [
        "git",
        "log",
        "--oneline",
        "-n",
        "10",
        "--",
        "validation/fast_backtest.py",
        "validation/structural_stop.py",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def build_diff_summary() -> dict[str, Any]:
    validator_text = VALIDATOR_FILES[0].read_text(encoding="utf-8").splitlines()
    bot_text = BOT_FILES[0].read_text(encoding="utf-8").splitlines()
    diff_lines = list(
        difflib.unified_diff(
            validator_text,
            bot_text,
            fromfile="validation/fast_backtest.py",
            tofile="app/strategy/trend_following_v2_simple.py",
            lineterm="",
        )
    )
    RAW_DIFF.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")

    sm = difflib.SequenceMatcher(a=validator_text, b=bot_text)
    validator_only = 0
    bot_only = 0
    modified = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            validator_only += i2 - i1
        elif tag == "insert":
            bot_only += j2 - j1
        elif tag == "replace":
            modified += max(i2 - i1, j2 - j1)

    return {
        "validator_only_lines": validator_only,
        "bot_only_lines": bot_only,
        "modified_lines": modified,
        "files": [str(p.relative_to(ROOT)) for p in VALIDATOR_FILES + BOT_FILES],
        "section_analysis": [
            {
                "section": "Condiciones de entrada",
                "status": "MAYORMENTE IDENTICAS",
                "detail": "EMA/ADX/MACD/pullback/candle body/cooldown usan la misma formula base en fast_backtest y TrendFollowingV2Simple.",
            },
            {
                "section": "Filtro HTF",
                "status": "DIFERENTE",
                "detail": "El validador aplica HTF dentro de fast_backtest con sesgo binario close>EMA50/close<EMA50. El bot lo aplica afuera, en run_simple_paper.py, con banda neutral de ±0.2%.",
            },
            {
                "section": "Cálculo de SL",
                "status": "IDENTICO",
                "detail": "Ambos caminos llaman a validation.structural_stop.compute_structural_sl con pivots 3/3, buffer 0.25 ATR y min_risk 0.8 ATR.",
            },
            {
                "section": "Cálculo de TP",
                "status": "DIFERENTE EN PARAMETRO",
                "detail": "La fórmula es la misma (risk * rr_ratio), pero el baseline validado usa rr=2.7 y el runner actual pasa rr=2.5.",
            },
            {
                "section": "Estado y ejecución",
                "status": "DIFERENTE",
                "detail": "fast_backtest solo abre trade si el HTF ya aprobó la entrada. El bot actual deja que la estrategia mutile su estado interno antes del filtro HTF del runner, lo que puede crear ghost trades.",
            },
            {
                "section": "Position sizing",
                "status": "DIFERENTE",
                "detail": "El validador mide retornos por trade sin sizing de equity. El runner usa sizing por riesgo y leverage máximo.",
            },
        ],
        "impact_analysis": [
            "El runner actual usa rr_ratio=2.5 en lugar de 2.7, lo que reduce payoff por ganador y normalmente baja ExpR y annual return.",
            "El filtro HTF del runner tiene una zona neutral ±0.2% alrededor de EMA50 4H, mientras el validador usa comparación estricta. Eso cambia qué señales pasan.",
            "Como el filtro HTF vive fuera de la estrategia, el bot puede quedar con un trade interno abierto aunque la entrada haya sido rechazada por el runner. Eso filtra señales posteriores de forma no validada.",
            "fast_backtest soporta modos ATR/HYBRID/trailing/regímenes porque también se usa como harness experimental; TrendFollowingV2Simple no expone todo eso en su interfaz deployable.",
        ],
    }


def format_table(documented: dict[str, float], validator: MetricBundle, bot: MetricBundle | None) -> str:
    rows = [
        ("Total trades", f"{documented['total_trades']}", f"{validator.total_trades}", f"{bot.total_trades if bot else '—'}"),
        ("Win rate", f"{documented['winrate']:.1f}%", f"{validator.winrate:.2f}%", f"{bot.winrate:.2f}%" if bot else "—"),
        ("ExpR", f"+{documented['expr']:.3f}R", f"{validator.expr:+.3f}R", f"{bot.expr:+.3f}R" if bot else "—"),
        ("Sharpe", f"{documented['sharpe_ratio']:.2f}", f"{validator.sharpe_ratio:.3f}", f"{bot.sharpe_ratio:.3f}" if bot else "—"),
        ("Annual %", f"{documented['annual_return_pct']:.1f}%", f"{validator.annual_return_pct:.2f}%", f"{bot.annual_return_pct:.2f}%" if bot else "—"),
        ("Max drawdown", "—", f"{validator.max_drawdown_pct:.2f}%", f"{bot.max_drawdown_pct:.2f}%" if bot else "—"),
        ("LONG / SHORT", "—", f"{validator.long_trades} / {validator.short_trades}", f"{bot.long_trades} / {bot.short_trades}" if bot else "—"),
    ]
    lines = [
        "| Métrica | Documentado | Validador actual | Bot actual |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    return "\n".join(lines)


def write_reports(
    *,
    verdict: str,
    validator_metrics: MetricBundle,
    bot_metrics: MetricBundle | None,
    documented: dict[str, float],
    git_log: str | None,
    diff_summary: dict[str, Any] | None,
) -> None:
    comparison_lines = [
        "# Canonical Comparison",
        "",
        "## Tabla comparativa",
        format_table(documented, validator_metrics, bot_metrics),
        "",
    ]

    if git_log:
        comparison_lines.extend(
            [
                "## Estado del validador",
                "El validador actual ya no reproduce el baseline documentado dentro de ±5%.",
                "",
                "### Últimos 10 commits",
                "```text",
                git_log,
                "```",
            ]
        )
    elif diff_summary is not None and bot_metrics is not None:
        comparison_lines.extend(
            [
                "## Diff de código",
                f"- Líneas presentes solo en validator: {diff_summary['validator_only_lines']}",
                f"- Líneas presentes solo en bot: {diff_summary['bot_only_lines']}",
                f"- Líneas modificadas: {diff_summary['modified_lines']}",
                f"- Raw diff: [{RAW_DIFF.name}]({RAW_DIFF})",
                "",
                "## Análisis por sección",
            ]
        )
        for item in diff_summary["section_analysis"]:
            comparison_lines.append(f"- {item['section']}: {item['status']}. {item['detail']}")
        comparison_lines.extend(
            [
                "",
                "## Impacto de divergencias",
            ]
        )
        for item in diff_summary["impact_analysis"]:
            comparison_lines.append(f"- {item}")

    COMPARISON_MD.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    if verdict == "VALIDATOR_ALSO_DRIFTED":
        action = "El validador también ha cambiado, necesitamos decidir cuál commit es la referencia canónica antes de continuar."
        next_steps = "Recuperar la versión canónica desde git o re-validar estadísticamente la versión actual."
        divergent_lines = "N/A"
        impact_desc = "No confiable: la fuente de verdad ya no reproduce las métricas documentadas."
    elif verdict == "CANONICAL_MATCH":
        action = "El bot actual está suficientemente alineado con el código validado. Se puede usar como fuente de verdad para replicación en Pine."
        next_steps = "Proceder a generar Pine Script replicando el código del bot."
        divergent_lines = str(diff_summary["modified_lines"] + diff_summary["validator_only_lines"] + diff_summary["bot_only_lines"]) if diff_summary else "0"
        impact_desc = "Sin deriva material sobre métricas clave."
    else:
        action = "Revertir o sincronizar el bot al baseline canónico antes de continuar con Pine o live trading."
        next_steps = "Si BOT_HAS_DRIFTED → revertir/sincronizar el bot al validator antes de continuar."
        divergent_lines = str(diff_summary["modified_lines"] + diff_summary["validator_only_lines"] + diff_summary["bot_only_lines"]) if diff_summary else "0"
        impact_desc = "Hay deriva funcional material entre validador canónico y bot deployable."

    report_lines = [
        "# Auditoría de Código Canónico — Informe",
        "",
        f"## Veredicto: {verdict}",
        "",
        "## Métricas reproducidas",
        format_table(documented, validator_metrics, bot_metrics),
        "",
        "## Estado del código",
        f"- Archivos auditados: {', '.join(diff_summary['files']) if diff_summary else ', '.join(str(p.relative_to(ROOT)) for p in VALIDATOR_FILES)}",
        f"- Líneas divergentes encontradas: {divergent_lines}",
        f"- Impacto funcional estimado: {impact_desc}",
        "",
        "## Acción recomendada",
        action,
        "",
        "## Siguientes pasos",
    ]
    if verdict == "CANONICAL_MATCH":
        report_lines.append("Si CANONICAL_MATCH → proceder a generar Pine Script replicando el código del bot.")
    elif verdict == "BOT_HAS_DRIFTED":
        report_lines.append("Si BOT_HAS_DRIFTED → revertir/sincronizar el bot al validator antes de continuar.")
    else:
        report_lines.append("Si VALIDATOR_ALSO_DRIFTED → recuperar versión canónica desde git o re-validar estadísticamente la versión actual.")

    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "documented": documented,
                "validator_metrics": asdict(validator_metrics),
                "bot_metrics": asdict(bot_metrics) if bot_metrics else None,
                "git_log": git_log,
                "diff_summary": diff_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
