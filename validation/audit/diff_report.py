from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .common import AUDIT_REPORT_MD, DIFF_SUMMARY_JSON, PLOTS_DIR, relative_diff


THRESHOLDS = {
    "atr_rel_pct": 0.5,
    "adx_abs": 0.3,
    "ema_rel_pct": 0.1,
    "macd_rel_pct": 1.0,
}


def _join_signals(py_signals: pd.DataFrame, pine_signals: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    py = py_signals.rename(columns=lambda c: f"py_{c}" if c not in ("bar_index",) else c)
    pine = pine_signals.rename(columns=lambda c: f"pine_{c}" if c not in ("bar_index",) else c)
    joined = py.merge(pine, on="bar_index", how="outer")

    rows = []
    for _, row in joined.iterrows():
        py_type = row.get("py_signal_type")
        pine_type = row.get("pine_signal_type")
        if pd.notna(py_type) and pd.notna(pine_type):
            bucket = "MATCH" if py_type == pine_type else "TYPE_MISMATCH"
        elif pd.notna(py_type):
            bucket = "PYTHON_ONLY"
        else:
            bucket = "PINE_ONLY"
        row_out = row.to_dict()
        row_out["bucket"] = bucket
        rows.append(row_out)
    return joined, rows


def _conditions_failure_text(bar_row: pd.Series | None, side: str, prefix: str) -> list[str]:
    if bar_row is None:
        return ["missing_bar_context"]
    if bool(bar_row.get("ghost_trade_active", False)):
        return ["ghost_trade_after_htf_rejection"]
    if bool(bar_row.get("in_trade_before", False)):
        return ["existing_open_trade"]
    conds = bar_row.get(f"{side.lower()}_conditions") or {}
    failed = [name for name, ok in conds.items() if not ok]
    aligned = bar_row.get(f"{side.lower()}_htf_aligned")
    if aligned == False:
        failed.append("htf_alignment")
    raw_trigger = bool(bar_row.get(f"{side.lower()}_raw_trigger", False))
    cooldown_ok = bool(bar_row.get(f"{side.lower()}_cooldown_ok", False))
    signal_state = bool(bar_row.get(f"{side.lower()}_signal_state", False))
    trigger_pre_htf = bool(bar_row.get(f"{side.lower()}_trigger_pre_htf", False))
    signal_type = bar_row.get("signal_type")
    if signal_state and not raw_trigger:
        failed.append("edge_detection")
    elif raw_trigger and not cooldown_ok:
        failed.append("cooldown")
    elif trigger_pre_htf and aligned == False:
        failed.append("htf_alignment")
    elif signal_type != side and not failed:
        failed.append("post_filter_state_block")
    return list(dict.fromkeys(failed))


def _summarize_match_diffs(match_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for _, row in match_df.iterrows():
        atr_diff = relative_diff(row["py_atr14"], row["pine_atr14"])
        ema20_diff = relative_diff(row["py_ema20"], row["pine_ema20"])
        ema50_diff = relative_diff(row["py_ema50"], row["pine_ema50"])
        macd_line_diff = relative_diff(row["py_macd_line"], row["pine_macd_line"])
        macd_signal_diff = relative_diff(row["py_macd_signal"], row["pine_macd_signal"])
        macd_hist_diff = relative_diff(row["py_macd_hist"], row["pine_macd_hist"])
        sl_diff = relative_diff(row["py_sl_calculado"], row["pine_sl_calculado"])
        tp_diff = relative_diff(row["py_tp_calculado"], row["pine_tp_calculado"])
        adx_diff = abs(row["py_adx14"] - row["pine_adx14"]) if pd.notna(row["py_adx14"]) and pd.notna(row["pine_adx14"]) else None

        pairs = {
            "atr_rel_pct": atr_diff,
            "adx_abs": adx_diff,
            "ema20_rel_pct": ema20_diff,
            "ema50_rel_pct": ema50_diff,
            "macd_line_rel_pct": macd_line_diff,
            "macd_signal_rel_pct": macd_signal_diff,
            "macd_hist_rel_pct": macd_hist_diff,
            "sl_rel_pct": sl_diff,
            "tp_rel_pct": tp_diff,
            "htf_bias_match": 1.0 if row["py_htf_bias"] == row["pine_htf_bias"] else 0.0,
        }
        for key, value in pairs.items():
            if value is not None:
                metrics[key].append(float(value))

    summary: dict[str, dict[str, float | None]] = {}
    for key, values in metrics.items():
        if not values:
            summary[key] = {"avg": None, "max": None, "p95": None}
            continue
        series = pd.Series(values, dtype=float)
        summary[key] = {
            "avg": float(series.mean()),
            "max": float(series.max()),
            "p95": float(series.quantile(0.95)),
        }
    return summary


def _plot_timeline(df_prices: pd.DataFrame, py_signals: pd.DataFrame, pine_signals: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_prices["timestamp_utc"], df_prices["close"], color="#2c3e50", linewidth=1.0)

    for signal_df, marker, face_alpha, prefix in (
        (py_signals, "o", 1.0, "Python"),
        (pine_signals, "^", 0.0, "Pine"),
    ):
        if signal_df.empty:
            continue
        for signal_type, color in (("LONG", "#1b9e77"), ("SHORT", "#d95f02")):
            subset = signal_df[signal_df["signal_type"] == signal_type]
            if subset.empty:
                continue
            ax.scatter(
                subset["timestamp_utc"],
                subset["close"],
                label=f"{prefix} {signal_type}",
                marker=marker,
                s=50,
                edgecolors=color,
                facecolors=color if face_alpha > 0 else "none",
                linewidths=1.2,
                alpha=0.9,
            )

    ax.set_title("Python vs Pine Signals Timeline")
    ax.set_xlabel("Timestamp UTC")
    ax.set_ylabel("BTC/USDT")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "signals_timeline.png", dpi=160)
    plt.close(fig)


def _plot_indicator_drift(match_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    plots = [
        ("ATR diff %", match_df["atr_rel_pct"], THRESHOLDS["atr_rel_pct"]),
        ("ADX diff abs", match_df["adx_abs"], THRESHOLDS["adx_abs"]),
        ("EMA50 diff %", match_df["ema50_rel_pct"], THRESHOLDS["ema_rel_pct"]),
        ("MACD hist diff %", match_df["macd_hist_rel_pct"], THRESHOLDS["macd_rel_pct"]),
    ]
    for ax, (title, series, threshold) in zip(axes, plots, strict=False):
        ax.plot(match_df["bar_index"], series, color="#34495e", linewidth=1.0)
        ax.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("abs diff")
    axes[-1].set_xlabel("bar_index")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "indicator_drift.png", dpi=160)
    plt.close(fig)


def _plot_htf_alignment(py_per_bar: pd.DataFrame, pine_per_bar: pd.DataFrame) -> None:
    mapping = {"BEAR": -1, "NEUTRAL": 0, "BULL": 1}
    merged = py_per_bar[["bar_index", "timestamp_utc", "htf_bias"]].merge(
        pine_per_bar[["bar_index", "htf_bias"]],
        on="bar_index",
        suffixes=("_py", "_pine"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    axes[0].step(merged["timestamp_utc"], merged["htf_bias_py"].map(mapping), where="post", color="#1f77b4")
    axes[0].set_title("Python HTF Bias")
    axes[1].step(merged["timestamp_utc"], merged["htf_bias_pine"].map(mapping), where="post", color="#ff7f0e")
    axes[1].set_title("Pine HTF Bias")
    for ax in axes:
        ax.set_yticks([-1, 0, 1], labels=["BEAR", "NEUTRAL", "BULL"])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "htf_alignment.png", dpi=160)
    plt.close(fig)


def build_diff_report(
    *,
    dataset_df: pd.DataFrame,
    python_signals: pd.DataFrame,
    pine_signals: pd.DataFrame,
    python_per_bar: pd.DataFrame,
    pine_per_bar: pd.DataFrame,
) -> dict[str, Any]:
    _, rows = _join_signals(python_signals, pine_signals)
    summary_df = pd.DataFrame(rows)

    matches = summary_df[summary_df["bucket"] == "MATCH"].copy()
    if not matches.empty:
        matches["atr_rel_pct"] = matches.apply(lambda r: relative_diff(r["py_atr14"], r["pine_atr14"]), axis=1)
        matches["adx_abs"] = (matches["py_adx14"] - matches["pine_adx14"]).abs()
        matches["ema50_rel_pct"] = matches.apply(lambda r: relative_diff(r["py_ema50"], r["pine_ema50"]), axis=1)
        matches["macd_hist_rel_pct"] = matches.apply(lambda r: relative_diff(r["py_macd_hist"], r["pine_macd_hist"]), axis=1)

    diagnostics: list[str] = []
    causes = Counter()
    for _, row in summary_df[summary_df["bucket"].isin(["PYTHON_ONLY", "PINE_ONLY"])].iterrows():
        if row["bucket"] == "PYTHON_ONLY":
            side = row["py_signal_type"]
            bar_idx = int(row["bar_index"])
            pine_bar = pine_per_bar.loc[pine_per_bar["bar_index"] == bar_idx]
            pine_bar_row = None if pine_bar.empty else pine_bar.iloc[0]
            failed = _conditions_failure_text(pine_bar_row, side, "pine")
            causes.update(failed)
            diagnostics.append(
                f"Bar {bar_idx} - PYTHON_ONLY {side}: Pine rejected because {', '.join(failed)}. "
                f"Pine HTF={pine_bar_row.get('htf_bias', 'NA') if pine_bar_row is not None else 'NA'}."
            )
        else:
            side = row["pine_signal_type"]
            bar_idx = int(row["bar_index"])
            py_bar = python_per_bar.loc[python_per_bar["bar_index"] == bar_idx]
            py_bar_row = None if py_bar.empty else py_bar.iloc[0]
            failed = _conditions_failure_text(py_bar_row, side, "py")
            causes.update(failed)
            diagnostics.append(
                f"Bar {bar_idx} - PINE_ONLY {side}: Python rejected because {', '.join(failed)}. "
                f"Python HTF={py_bar_row.get('htf_bias', 'NA') if py_bar_row is not None else 'NA'}."
            )

    for _, row in summary_df[summary_df["bucket"] == "TYPE_MISMATCH"].iterrows():
        causes.update(["type_mismatch"])
        diagnostics.append(
            f"Bar {int(row['bar_index'])} - TYPE_MISMATCH: Python={row['py_signal_type']} Pine={row['pine_signal_type']}."
        )

    counts = summary_df["bucket"].value_counts().to_dict()
    total_python = int(len(python_signals))
    total_pine = int(len(pine_signals))
    total_match = int(counts.get("MATCH", 0))
    match_rate = (total_match / max(total_python, total_pine) * 100.0) if max(total_python, total_pine) else 0.0
    diff_summary = _summarize_match_diffs(matches)

    _plot_timeline(dataset_df, python_signals, pine_signals)
    _plot_indicator_drift(matches if not matches.empty else pd.DataFrame({"bar_index": [], "atr_rel_pct": [], "adx_abs": [], "ema50_rel_pct": [], "macd_hist_rel_pct": []}))
    _plot_htf_alignment(python_per_bar, pine_per_bar)

    type_mismatch_count = int(counts.get("TYPE_MISMATCH", 0))
    if match_rate >= 95.0 and type_mismatch_count == 0 and (
        (diff_summary.get("atr_rel_pct", {}).get("max") or 0.0) <= THRESHOLDS["atr_rel_pct"]
        and (diff_summary.get("adx_abs", {}).get("max") or 0.0) <= THRESHOLDS["adx_abs"]
        and (diff_summary.get("ema50_rel_pct", {}).get("max") or 0.0) <= THRESHOLDS["ema_rel_pct"]
        and (diff_summary.get("macd_hist_rel_pct", {}).get("max") or 0.0) <= THRESHOLDS["macd_rel_pct"]
    ):
        verdict = "MATCH_EXCELENTE"
        recommendation = "Continuar paper trading. La alineacion entre ejecucion Python y replica Pine es suficientemente alta."
    elif 85.0 <= match_rate < 95.0 and type_mismatch_count == 0:
        verdict = "MATCH_ACEPTABLE"
        recommendation = "Continuar paper trading, pero dejar documentadas las diferencias conocidas antes de pasar a live."
    elif match_rate < 50.0 or type_mismatch_count > 0:
        verdict = "RESET_REQUERIDO"
        recommendation = "Pausar paper trading y revisar arquitectura/contrato de senales antes de seguir."
    else:
        verdict = "BUG_REPRODUCIBLE"
        recommendation = "Pausar paper trading, corregir la causa raiz dominante y volver a correr esta auditoria."

    top_causes = causes.most_common(3)
    result = {
        "total_python": total_python,
        "total_pine": total_pine,
        "total_match": total_match,
        "match_rate": match_rate,
        "divergences": {
            "PYTHON_ONLY": int(counts.get("PYTHON_ONLY", 0)),
            "PINE_ONLY": int(counts.get("PINE_ONLY", 0)),
            "TYPE_MISMATCH": type_mismatch_count,
        },
        "top_causes": top_causes,
        "indicator_diff_summary": diff_summary,
        "diagnostics": diagnostics[:50],
        "verdict": verdict,
        "recommendation": recommendation,
    }
    DIFF_SUMMARY_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(result)
    return result


def _format_metric(value: float | None, precision: int = 3) -> str:
    return "NA" if value is None else f"{value:.{precision}f}"


def _write_report(result: dict[str, Any]) -> None:
    diff = result["indicator_diff_summary"]
    lines = [
        "# Auditoría Python ↔ Pine Structural — Informe",
        "",
        "## Resumen ejecutivo",
        f"- Match rate: {result['match_rate']:.2f}%",
        f"- Veredicto: {result['verdict']}",
        "",
        "## Criterios de veredicto",
        "- MATCH_EXCELENTE: match rate >= 95% Y ningún TYPE_MISMATCH Y diff indicadores todos dentro de threshold. Acción: continuar paper trading.",
        "- MATCH_ACEPTABLE: match rate 85-95%, divergencias explicables por drift numérico en barras iniciales. Acción: continuar paper pero documentar las discrepancias conocidas.",
        "- BUG_REPRODUCIBLE: match rate 50-85% con causa raíz identificada. Acción: pausar paper, arreglar el lado erróneo, re-auditar.",
        "- RESET_REQUERIDO: match rate <50% O cualquier TYPE_MISMATCH. Acción: pausar paper, revisar arquitectura, re-validar desde WFA.",
        "",
        "## Números clave",
        "",
        "| Métrica | Valor |",
        "| --- | ---: |",
        f"| Total señales Python | {result['total_python']} |",
        f"| Total señales Pine | {result['total_pine']} |",
        f"| Total MATCH | {result['total_match']} |",
        f"| PYTHON_ONLY | {result['divergences']['PYTHON_ONLY']} |",
        f"| PINE_ONLY | {result['divergences']['PINE_ONLY']} |",
        f"| TYPE_MISMATCH | {result['divergences']['TYPE_MISMATCH']} |",
        f"| Avg ATR diff % | {_format_metric(diff.get('atr_rel_pct', {}).get('avg'))} |",
        f"| Max ATR diff % | {_format_metric(diff.get('atr_rel_pct', {}).get('max'))} |",
        f"| P95 ATR diff % | {_format_metric(diff.get('atr_rel_pct', {}).get('p95'))} |",
        f"| Avg ADX diff | {_format_metric(diff.get('adx_abs', {}).get('avg'))} |",
        f"| Max ADX diff | {_format_metric(diff.get('adx_abs', {}).get('max'))} |",
        f"| P95 ADX diff | {_format_metric(diff.get('adx_abs', {}).get('p95'))} |",
        f"| Avg EMA50 diff % | {_format_metric(diff.get('ema50_rel_pct', {}).get('avg'))} |",
        f"| Max EMA50 diff % | {_format_metric(diff.get('ema50_rel_pct', {}).get('max'))} |",
        f"| P95 EMA50 diff % | {_format_metric(diff.get('ema50_rel_pct', {}).get('p95'))} |",
        f"| Avg MACD hist diff % | {_format_metric(diff.get('macd_hist_rel_pct', {}).get('avg'))} |",
        f"| Max MACD hist diff % | {_format_metric(diff.get('macd_hist_rel_pct', {}).get('max'))} |",
        f"| P95 MACD hist diff % | {_format_metric(diff.get('macd_hist_rel_pct', {}).get('p95'))} |",
        "",
        "## Top causas de divergencia",
    ]
    if result["top_causes"]:
        for cause, count in result["top_causes"]:
            lines.append(f"- {cause}: {count} casos")
    else:
        lines.append("- No se detectaron causas de divergencia porque no hubo divergencias.")

    lines.extend(
        [
            "",
            "## Recomendación concreta",
            f"{result['recommendation']}",
            "",
            "## Diagnóstico detallado",
        ]
    )
    if result["diagnostics"]:
        for line in result["diagnostics"]:
            lines.append(f"- {line}")
    else:
        lines.append("- No hubo divergencias para detallar.")

    AUDIT_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
