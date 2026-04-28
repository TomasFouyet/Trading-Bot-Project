"""
Forward Trade Analysis — Lag & Quality
BTC/USDT 15m | TrendBot V2 Structural | Apr 2026
Multi-symbol paper_trades_v2.csv

Produces:
  data/forward_trades_enriched.csv
  validation/output/forward_lag_quality_BTCUSDT.png
  validation/output/forward_metrics_detail_BTCUSDT.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from validation.fast_backtest import compute_indicators

PARQUET_DIR = ROOT / "data" / "parquet"
OUTPUT_DIR = ROOT / "validation" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Canonical parameters ────────────────────────────────────────────────────
RR_RATIO     = 2.7
STOP_MODE    = "STRUCTURAL"
BUFFER_ATR   = 0.25
MIN_RISK_ATR = 0.8
PIVOT_LEFT   = 3
PIVOT_RIGHT  = 3
EMA_FAST     = 20
EMA_SLOW     = 50
ADX_MIN      = 20
HTF_EMA      = 50

# EMA8/13 for fast-cross analysis
EMA_FAST2 = 8
EMA_SLOW2 = 13

DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GREEN = "#00d4aa"
RED = "#ff4757"
GOLD = "#ffd700"
BLUE = "#4a90d9"
GRAY = "#888888"


# ─── Load OHLCV parquet (multi-month) ────────────────────────────────────────
def load_parquet(symbol: str, tf: str, months: list[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        p = PARQUET_DIR / symbol / tf / f"{m}.parquet"
        if p.exists():
            frames.append(pq.read_table(str(p)).to_pandas())
    if not frames:
        raise FileNotFoundError(f"No parquet for {symbol}/{tf} months={months}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def load_ohlcv_with_indicators(symbol: str) -> pd.DataFrame:
    months = ["2026-03", "2026-04"]
    # Also include 2026-02 for warm-up
    warmup_months = ["2025-11", "2025-12", "2026-01", "2026-02"] + months
    df = load_parquet(symbol, "15m", warmup_months)
    df = compute_indicators(df, ema_fast_p=EMA_FAST, ema_slow_p=EMA_SLOW)
    # Add EMA8/EMA13
    df["ema8"]  = df["close"].ewm(span=EMA_FAST2, adjust=False).mean()
    df["ema13"] = df["close"].ewm(span=EMA_SLOW2, adjust=False).mean()
    # Stochastic RSI (14,3,3,14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    stoch_k_raw = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    df["stoch_k"] = stoch_k_raw.rolling(3).mean()
    # Body ratio
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-10)
    # Pullback dist ATR (distance from EMA_FAST in ATR units)
    df["pb_dist_atr"] = (df["close"] - df["ema_fast"]).abs() / df["atr"].replace(0, np.nan)
    # MACD hist direction
    df["macd_hist_increasing"] = df["macd_hist"].diff() > 0

    df = df.reset_index(drop=True)
    df["bar_idx"] = df.index
    return df


# ─── EMA cross "move start" detection ────────────────────────────────────────
def find_move_start(df_sym: pd.DataFrame, entry_bar: int, direction: str, lookback: int = 20) -> int:
    """
    Find the bar index where the move that triggered entry began.
    Heuristic: last bar (before entry) where price crossed EMA20 in the entry direction.
    """
    start = max(0, entry_bar - lookback)
    sub = df_sym.loc[start:entry_bar].copy()
    close = sub["close"].values
    ema = sub["ema_fast"].values

    # Find crossover bars
    if direction == "LONG":
        crosses = np.where((close[1:] > ema[1:]) & (close[:-1] <= ema[:-1]))[0]
    else:
        crosses = np.where((close[1:] < ema[1:]) & (close[:-1] >= ema[:-1]))[0]

    if len(crosses) > 0:
        last_cross_local = crosses[-1] + 1  # bar after the cross
        return sub.index[last_cross_local]
    # Fallback: entry bar itself (lag=0)
    return entry_bar


# ─── Enrich trades ────────────────────────────────────────────────────────────
def enrich_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for symbol, grp in trades_df.groupby("symbol"):
        try:
            df_sym = load_ohlcv_with_indicators(symbol)
        except FileNotFoundError:
            print(f"  [SKIP] No OHLCV for {symbol}")
            continue

        ts_arr = df_sym["ts"].values

        for _, row in grp.iterrows():
            entry_ts = pd.Timestamp(row["entry_time"]).tz_convert("UTC")
            exit_ts  = pd.Timestamp(row["exit_time"]).tz_convert("UTC")

            # Find closest bar
            diffs = np.abs(ts_arr - np.datetime64(entry_ts))
            entry_bar = int(np.argmin(diffs))
            diffs_exit = np.abs(ts_arr - np.datetime64(exit_ts))
            exit_bar = int(np.argmin(diffs_exit))

            erow = df_sym.iloc[entry_bar]
            direction = row["side"]  # LONG or SHORT

            # Move start
            move_start_bar = find_move_start(df_sym, entry_bar, direction)
            lag_bars = entry_bar - move_start_bar
            price_at_move_start = df_sym.iloc[move_start_bar]["close"]

            if direction == "LONG":
                lag_price_pct = (row["entry_price"] - price_at_move_start) / price_at_move_start * 100
            else:
                lag_price_pct = (price_at_move_start - row["entry_price"]) / price_at_move_start * 100

            # Duration
            duration_bars = exit_bar - entry_bar

            # Outcome
            exit_type = row["exit_type"]
            if exit_type in ["sl", "be_sl", "reversal_swap"]:
                outcome = "sl"
            elif exit_type in ["tp", "tp1", "tp2", "trailing_sl"]:
                outcome = "tp"
            else:
                outcome = "other"

            # PnL in R
            sl_dist_pct = abs(row.get("pnl_pct", 0))  # approximate
            pnl_pct = row["pnl_pct"]

            # SL distance from signal_reason (rough) — use entry vs sl if available
            risk_pct = abs(erow["atr"] / row["entry_price"] * 100) * 1.5  # ATR fallback
            pnl_r = pnl_pct / risk_pct if risk_pct > 0 else np.nan

            # MAE/MFE (approximate from bars held)
            if duration_bars > 0 and exit_bar <= len(df_sym) - 1:
                sub = df_sym.iloc[entry_bar:exit_bar + 1]
                if direction == "LONG":
                    mae = -(sub["low"].min() - row["entry_price"]) / row["entry_price"] * 100
                    mfe =  (sub["high"].max() - row["entry_price"]) / row["entry_price"] * 100
                else:
                    mae = -(row["entry_price"] - sub["high"].max()) / row["entry_price"] * 100
                    mfe =  (row["entry_price"] - sub["low"].min()) / row["entry_price"] * 100
                mae_r = mae / risk_pct if risk_pct > 0 else np.nan
                mfe_r = mfe / risk_pct if risk_pct > 0 else np.nan
            else:
                mae_r = mfe_r = np.nan

            # EMA8/13 cross
            ema8_above_ema13 = erow["ema8"] > erow["ema13"]
            # Check cross in last 3 bars
            if entry_bar >= 3:
                pre3 = df_sym.iloc[entry_bar - 3:entry_bar + 1]
                if direction == "LONG":
                    ema8_cross_recent = any(
                        (pre3["ema8"].values[i] > pre3["ema13"].values[i] and
                         pre3["ema8"].values[i-1] <= pre3["ema13"].values[i-1])
                        for i in range(1, len(pre3))
                    )
                else:
                    ema8_cross_recent = any(
                        (pre3["ema8"].values[i] < pre3["ema13"].values[i] and
                         pre3["ema8"].values[i-1] >= pre3["ema13"].values[i-1])
                        for i in range(1, len(pre3))
                    )
            else:
                ema8_cross_recent = False

            # MACD hist direction
            macd_hist_dir = "increasing" if erow["macd_hist_increasing"] else "decreasing"

            records.append({
                "trade_id": row.get("trade_id", ""),
                "symbol": symbol,
                "direction": direction,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "exit_type": exit_type,
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "pnl_r": pnl_r,
                "mae_r": mae_r,
                "mfe_r": mfe_r,
                "entry_bar_idx": entry_bar,
                "duration_bars": duration_bars,
                "duration_minutes": duration_bars * 15,
                "move_start_bar": move_start_bar,
                "lag_bars": lag_bars,
                "lag_minutes": lag_bars * 15,
                "price_at_move_start": price_at_move_start,
                "lag_price_pct": lag_price_pct,
                "adx_at_entry": erow["adx"],
                "body_ratio": erow["body_ratio"],
                "pullback_dist_atr": erow["pb_dist_atr"],
                "stoch_k_at_entry": erow["stoch_k"],
                "confidence_score": erow.get("confidence", np.nan),
                "ema8_above_ema13": ema8_above_ema13,
                "ema8_cross_recent": ema8_cross_recent,
                "macd_hist_dir": macd_hist_dir,
                "sl_mode": row.get("sl_mode", "unknown"),
            })

    enriched = pd.DataFrame(records)
    return enriched


# ─── Stats helpers ────────────────────────────────────────────────────────────
def wr_ci(n_tp, n_total, z=1.96):
    """Wilson 95% CI for win rate."""
    if n_total == 0:
        return 0, 0, 0
    p = n_tp / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p * 100, max(0, (center - margin) * 100), min(100, (center + margin) * 100)


# ─── Classification of SL trades ─────────────────────────────────────────────
def classify_sl_trades(enriched: pd.DataFrame) -> pd.DataFrame:
    sl_trades = enriched[enriched["outcome"] == "sl"].copy()
    sl_trades["type_a"] = (sl_trades["lag_bars"] > 5) & (sl_trades["lag_price_pct"] > 1.5)
    sl_trades["type_b"] = (sl_trades["adx_at_entry"] < 25) | (sl_trades["body_ratio"] < 0.35)
    sl_trades["type_c"] = sl_trades["mae_r"].fillna(99) < 1.1
    sl_trades["type_d"] = sl_trades["mae_r"].fillna(99) < 0.3
    return sl_trades


# ─── Print Step 0 ─────────────────────────────────────────────────────────────
def print_step0(trades_df, enriched):
    print("\n" + "="*65)
    print("STEP 0 — DATOS DISPONIBLES")
    print("="*65)
    print(f"Archivo: data/paper_trades_v2.csv")
    print(f"Columnas: {trades_df.columns.tolist()}")
    print(f"Total trades en CSV: {len(trades_df)}")
    completed = trades_df[~trades_df["exit_type"].isin(["shutdown"])]
    print(f"Trades completados (non-shutdown): {len(completed)}")
    print(f"Período: {pd.to_datetime(completed['entry_time']).min().strftime('%Y-%m-%d')} "
          f"to {pd.to_datetime(completed['exit_time']).max().strftime('%Y-%m-%d')}")
    print(f"Símbolos: {sorted(completed['symbol'].unique().tolist())}")
    print()
    print(f"Trades enriquecidos: {len(enriched)}")
    completed_e = enriched[enriched["outcome"].isin(["sl", "tp"])]
    print(f"  Completados (sl+tp): {len(completed_e)}")
    print(f"  TP: {(completed_e['outcome']=='tp').sum()}")
    print(f"  SL: {(completed_e['outcome']=='sl').sum()}")
    print()

    missing = []
    for col in ["adx_at_entry", "body_ratio", "stoch_k_at_entry", "pullback_dist_atr"]:
        n_miss = enriched[col].isna().sum()
        if n_miss > 0:
            missing.append(f"  {col}: {n_miss} missing")
    if missing:
        print("Campos con missing values (recalculados desde OHLCV):")
        for m in missing:
            print(m)
    else:
        print("Sin campos faltantes — todos calculados desde OHLCV.")


# ─── Print Step 2 (Lag) ───────────────────────────────────────────────────────
def print_step2(enriched: pd.DataFrame):
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    lag = e["lag_bars"]
    outcome_num = (e["outcome"] == "tp").astype(int)
    corr = lag.corr(outcome_num)

    print("\n" + "="*65)
    print("STEP 2 — ANÁLISIS DE LAG")
    print("="*65)
    print(f"Mean lag:   {lag.mean():.1f} barras ({lag.mean()*15:.0f} min)")
    print(f"Median lag: {lag.median():.1f} barras")
    print(f"Std lag:    {lag.std():.1f} barras")
    print(f"P25:        {lag.quantile(0.25):.1f} barras")
    print(f"P75:        {lag.quantile(0.75):.1f} barras")
    print()
    n = len(e)
    print(f"% lag <=1b: {(lag<=1).sum()/n*100:.1f}%")
    print(f"% lag <=3b: {(lag<=3).sum()/n*100:.1f}%")
    print(f"% lag <=5b: {(lag<=5).sum()/n*100:.1f}%")
    print(f"% lag  >5b: {(lag>5).sum()/n*100:.1f}%")
    print(f"Mean lag_price_pct: {e['lag_price_pct'].mean():.2f}%")
    print()

    print(f"2b. Lag por outcome:")
    for oc in ["tp", "sl"]:
        sub = e[e["outcome"] == oc]
        print(f"  {oc.upper()} N={len(sub)}: avg_lag={sub['lag_bars'].mean():.1f}b "
              f"lag_price={sub['lag_price_pct'].mean():.2f}%")
    print(f"  Correlación lag vs outcome: r={corr:.3f}")
    if corr < -0.15:
        verdict = "LAG CONTRIBUYE A PERDER TRADES"
    elif corr > -0.05:
        verdict = "LAG NO ES EL PROBLEMA PRINCIPAL"
    else:
        verdict = "LAG TIENE EFECTO MODERADO"
    print(f"  Veredicto lag: {verdict}")
    print()

    print("2c. Lag por dirección:")
    for d in ["LONG", "SHORT"]:
        sub = e[e["direction"] == d]
        if len(sub) > 0:
            print(f"  {d}: avg_lag={sub['lag_bars'].mean():.1f} barras (N={len(sub)})")
    return corr


# ─── Print Step 3 (Calidad) ───────────────────────────────────────────────────
def print_step3(enriched: pd.DataFrame):
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    tp = e[e["outcome"] == "tp"]
    sl = e[e["outcome"] == "sl"]
    n_total = len(e)
    wr_global, _, _ = wr_ci(len(tp), n_total)

    print("\n" + "="*65)
    print("STEP 3 — ANÁLISIS DE CALIDAD DEL SETUP")
    print("="*65)

    # 3a
    print(f"{'Métrica':<22} {'TP':>8} {'SL':>8} {'Delta':>8} {'Signal?':>10}")
    print("-"*60)
    metrics = [
        ("ADX al entrar",     "adx_at_entry",       3),
        ("Body ratio",        "body_ratio",          0.05),
        ("Pullback dist ATR", "pullback_dist_atr",   0.1),
        ("Stoch K al entrar", "stoch_k_at_entry",    5),
        ("Confidence",        "confidence_score",    0.03),
    ]
    for label, col, threshold in metrics:
        v_tp = tp[col].dropna().mean()
        v_sl = sl[col].dropna().mean()
        delta = v_tp - v_sl
        signal = "SÍ" if abs(delta) >= threshold else "no"
        print(f"  {label:<20} {v_tp:>8.2f} {v_sl:>8.2f} {delta:>+8.2f} {signal:>10}")

    # Boolean metrics
    bool_metrics = [
        ("EMA8>EMA13",       "ema8_above_ema13"),
        ("EMA8 cruce reciente", "ema8_cross_recent"),
        ("MACD increasing",  None),
    ]
    for label, col in bool_metrics:
        if col is None:
            v_tp = (tp["macd_hist_dir"] == "increasing").mean() * 100
            v_sl = (sl["macd_hist_dir"] == "increasing").mean() * 100
        else:
            v_tp = tp[col].mean() * 100 if col in tp else np.nan
            v_sl = sl[col].mean() * 100 if col in sl else np.nan
        delta = v_tp - v_sl
        signal = "SÍ" if abs(delta) >= 5 else "no"
        print(f"  {label:<20} {v_tp:>7.1f}% {v_sl:>7.1f}% {delta:>+7.1f}pp {signal:>10}")

    # 3b. WR by ADX quartile
    print()
    print("3b. WR por cuartil ADX:")
    adx_bins = [(20, 24), (24, 28), (28, 33), (33, 999)]
    for lo, hi in adx_bins:
        sub = e[(e["adx_at_entry"] >= lo) & (e["adx_at_entry"] < hi)]
        n_sub = len(sub)
        if n_sub == 0:
            print(f"  ADX {lo}-{hi}: N=0")
            continue
        n_tp = (sub["outcome"] == "tp").sum()
        wr, ci_lo, ci_hi = wr_ci(n_tp, n_sub)
        print(f"  ADX {lo:2d}-{hi if hi<999 else '+'}: WR={wr:.0f}%  [{ci_lo:.0f}-{ci_hi:.0f}% CI]  N={n_sub}")

    # 3c. WR by body ratio
    print()
    print("3c. WR por body ratio:")
    body_bins = [(0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.0)]
    for lo, hi in body_bins:
        sub = e[(e["body_ratio"] >= lo) & (e["body_ratio"] < hi)]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        n_tp = (sub["outcome"] == "tp").sum()
        wr, ci_lo, ci_hi = wr_ci(n_tp, n_sub)
        print(f"  body {lo:.2f}-{hi:.2f}: WR={wr:.0f}%  [{ci_lo:.0f}-{ci_hi:.0f}% CI]  N={n_sub}")

    # 3d. EMA8/13 filter
    print()
    print("3d. EMA8/13 impacto:")
    for col, label in [("ema8_above_ema13", "EMA8>EMA13"), ("ema8_cross_recent", "Cruce reciente")]:
        for val, lbl in [(True, "CON"), (False, "SIN")]:
            sub = e[e[col] == val]
            n_sub = len(sub)
            if n_sub == 0:
                continue
            n_tp = (sub["outcome"] == "tp").sum()
            wr, ci_lo, ci_hi = wr_ci(n_tp, n_sub)
            print(f"  {label} {lbl}: WR={wr:.0f}%  [{ci_lo:.0f}-{ci_hi:.0f}% CI]  N={n_sub}")


# ─── Print Step 4 ────────────────────────────────────────────────────────────
def print_step4(enriched: pd.DataFrame) -> pd.DataFrame:
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    sl_trades = classify_sl_trades(e)
    n_sl = len(sl_trades)

    print("\n" + "="*65)
    print("STEP 4 — CLASIFICACIÓN DE TRADES PERDEDORES")
    print("="*65)
    if n_sl == 0:
        print("Sin trades SL para clasificar.")
        return sl_trades

    for typ, label in [("type_a", "A — entrada tardía"), ("type_b", "B — setup débil"),
                        ("type_c", "C — stop hunt"),    ("type_d", "D — contra-tendencia")]:
        n_typ = sl_trades[typ].sum()
        print(f"  Tipo {label}: {n_typ}/{n_sl} ({n_typ/n_sl*100:.0f}%)")

    # Top 5 worst trades
    print()
    print("Top 5 peores trades (por pnl_pct):")
    worst = sl_trades.nsmallest(5, "pnl_pct")
    for _, r in worst.iterrows():
        typ_str = "".join([
            "A" if r["type_a"] else "",
            "B" if r["type_b"] else "",
            "C" if r["type_c"] else "",
            "D" if r["type_d"] else "",
        ]) or "—"
        print(f"  {r['symbol']:10s} {r['direction']:5s} lag={r['lag_bars']:2.0f}b "
              f"adx={r['adx_at_entry']:.0f} body={r['body_ratio']:.2f} "
              f"stoch={r['stoch_k_at_entry']:.0f} tipo={typ_str} "
              f"pnl={r['pnl_pct']:.2f}%")

    return sl_trades


# ─── Print Step 5 ────────────────────────────────────────────────────────────
def print_step5(enriched: pd.DataFrame):
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    n_total = len(e)
    n_tp_all = (e["outcome"] == "tp").sum()
    wr_all, _, _ = wr_ci(n_tp_all, n_total)
    pnl_all = e["pnl_pct"].sum()

    print("\n" + "="*65)
    print("STEP 5 — TEST EMA8/13 COMO FILTRO")
    print("="*65)
    print(f"Sin filtro EMA8/13: N={n_total}, WR={wr_all:.1f}%, P&L={pnl_all:.2f}%")

    # With EMA8>13 filter
    with_filter = e[
        ((e["direction"] == "LONG")  & (e["ema8_above_ema13"] == True)) |
        ((e["direction"] == "SHORT") & (e["ema8_above_ema13"] == False))
    ]
    n_f = len(with_filter)
    n_tp_f = (with_filter["outcome"] == "tp").sum()
    wr_f, ci_lo_f, ci_hi_f = wr_ci(n_tp_f, n_f) if n_f > 0 else (0, 0, 0)
    pnl_f = with_filter["pnl_pct"].sum()
    print(f"Con filtro EMA8>13: N={n_f}, WR={wr_f:.1f}% [{ci_lo_f:.0f}-{ci_hi_f:.0f}%], "
          f"P&L={pnl_f:.2f}%")
    print(f"  Trades eliminados: {n_total - n_f} ({(n_total-n_f)/n_total*100:.0f}%)")
    print(f"  WR delta: {wr_f - wr_all:+.1f}pp")

    # With recent cross filter
    with_cross = e[
        ((e["direction"] == "LONG")  & (e["ema8_cross_recent"] == True)) |
        ((e["direction"] == "SHORT") & (e["ema8_cross_recent"] == True))
    ]
    n_c = len(with_cross)
    if n_c > 0:
        n_tp_c = (with_cross["outcome"] == "tp").sum()
        wr_c, ci_lo_c, ci_hi_c = wr_ci(n_tp_c, n_c)
        pnl_c = with_cross["pnl_pct"].sum()
        print(f"Con cruce EMA8/13 reciente: N={n_c}, WR={wr_c:.1f}% [{ci_lo_c:.0f}-{ci_hi_c:.0f}%], "
              f"P&L={pnl_c:.2f}%")
    return wr_all, wr_f, n_total, n_f


# ─── Figures ──────────────────────────────────────────────────────────────────
def plot_figure1(enriched: pd.DataFrame, sl_trades: pd.DataFrame,
                 corr_lag: float, wr_no_filter: float, wr_filter: float,
                 n_no_filter: int, n_filter: int):
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    tp = e[e["outcome"] == "tp"]
    sl = e[e["outcome"] == "sl"]
    n_total = len(e)
    wr_global = len(tp) / n_total * 100 if n_total > 0 else 0

    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle("Análisis Forward Trades — Lag & Calidad | Multi-símbolo Apr 2026",
                 color="white", fontsize=14, y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def setup_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    # Panel 1 — Lag distribution
    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, f"Lag de Entrada (barras 15m) — TP vs SL\nn={n_total}")
    bins = range(0, int(e["lag_bars"].max()) + 3)
    ax1.hist(e["lag_bars"], bins=bins, color=GRAY, alpha=0.4, label="Todos")
    ax1.hist(tp["lag_bars"], bins=bins, color=GREEN, alpha=0.7, label=f"TP (n={len(tp)})")
    ax1.hist(sl["lag_bars"], bins=bins, color=RED, alpha=0.7, label=f"SL (n={len(sl)})")
    ax1.axvline(3, color=GOLD, linestyle="--", alpha=0.8, label="3 barras")
    ax1.axvline(5, color="orange", linestyle="--", alpha=0.8, label="5 barras")
    ax1.set_xlabel("Lag (barras)")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="white")

    # Panel 2 — Lag vs PnL scatter
    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, f"Lag vs PnL (%) | r={corr_lag:.3f}\n(positivo = lag ayuda a ganar)")
    colors_sc = [GREEN if o == "tp" else RED for o in e["outcome"]]
    ax2.scatter(e["lag_bars"], e["pnl_pct"], c=colors_sc, alpha=0.7, s=60, edgecolors="none")
    if len(e) > 2:
        m, b = np.polyfit(e["lag_bars"], e["pnl_pct"], 1)
        x_line = np.linspace(e["lag_bars"].min(), e["lag_bars"].max(), 100)
        ax2.plot(x_line, m * x_line + b, color=GOLD, linestyle="--", alpha=0.8)
    ax2.axhline(0, color=GRAY, alpha=0.5)
    ax2.set_xlabel("Lag (barras)")
    ax2.set_ylabel("PnL (%)")
    tp_patch = mpatches.Patch(color=GREEN, label="TP")
    sl_patch = mpatches.Patch(color=RED, label="SL")
    ax2.legend(handles=[tp_patch, sl_patch], fontsize=8, facecolor=PANEL_BG, labelcolor="white")

    # Panel 3 — WR by ADX quartile
    ax3 = fig.add_subplot(gs[1, 0])
    setup_ax(ax3, "WR por ADX al entrar (cuartiles)")
    adx_bins = [(20, 24, "20-24"), (24, 28, "24-28"), (28, 33, "28-33"), (33, 999, "33+")]
    x_labels, wrs, ci_los, ci_his, ns = [], [], [], [], []
    for lo, hi, lbl in adx_bins:
        sub = e[(e["adx_at_entry"] >= lo) & (e["adx_at_entry"] < hi)]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        n_tp = (sub["outcome"] == "tp").sum()
        wr, ci_lo, ci_hi = wr_ci(n_tp, n_sub)
        x_labels.append(f"{lbl}\n(n={n_sub})")
        wrs.append(wr)
        ci_los.append(wr - ci_lo)
        ci_his.append(ci_hi - wr)
        ns.append(n_sub)
    x = np.arange(len(x_labels))
    colors_bar = [RED if w < wr_global else GREEN for w in wrs]
    bars = ax3.bar(x, wrs, color=colors_bar, alpha=0.8, width=0.6,
                   yerr=[ci_los, ci_his], capsize=5, error_kw={"color": "white", "alpha": 0.6})
    ax3.axhline(wr_global, color=GOLD, linestyle="--", alpha=0.9, label=f"WR global {wr_global:.0f}%")
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, fontsize=8)
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")

    # Panel 4 — WR by body ratio
    ax4 = fig.add_subplot(gs[1, 1])
    setup_ax(ax4, "WR por Body Ratio de vela de entrada")
    body_bins = [(0, 0.30, "<0.30"), (0.30, 0.50, "0.30-0.50"),
                 (0.50, 0.70, "0.50-0.70"), (0.70, 1.0, ">0.70")]
    x_labels2, wrs2, ci_los2, ci_his2 = [], [], [], []
    for lo, hi, lbl in body_bins:
        sub = e[(e["body_ratio"] >= lo) & (e["body_ratio"] < hi)]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        n_tp = (sub["outcome"] == "tp").sum()
        wr, ci_lo, ci_hi = wr_ci(n_tp, n_sub)
        x_labels2.append(f"{lbl}\n(n={n_sub})")
        wrs2.append(wr)
        ci_los2.append(wr - ci_lo)
        ci_his2.append(ci_hi - wr)
    x2 = np.arange(len(x_labels2))
    colors_bar2 = [RED if w < wr_global else GREEN for w in wrs2]
    ax4.bar(x2, wrs2, color=colors_bar2, alpha=0.8, width=0.6,
            yerr=[ci_los2, ci_his2], capsize=5, error_kw={"color": "white", "alpha": 0.6})
    ax4.axhline(wr_global, color=GOLD, linestyle="--", alpha=0.9, label=f"WR global {wr_global:.0f}%")
    ax4.set_xticks(x2)
    ax4.set_xticklabels(x_labels2, fontsize=8)
    ax4.set_ylabel("Win Rate (%)")
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")

    # Panel 5 — SL trade classification
    ax5 = fig.add_subplot(gs[2, 0])
    setup_ax(ax5, f"Clasificación trades perdedores (N_SL={len(sl_trades)})")
    if len(sl_trades) > 0:
        types = ["A\n(lag tardío)", "B\n(setup débil)", "C\n(stop hunt)", "D\n(contra-tend)"]
        counts = [sl_trades["type_a"].sum(), sl_trades["type_b"].sum(),
                  sl_trades["type_c"].sum(), sl_trades["type_d"].sum()]
        pcts = [c / len(sl_trades) * 100 for c in counts]
        colors_type = [RED, "orange", BLUE, GRAY]
        x5 = np.arange(len(types))
        bars5 = ax5.bar(x5, pcts, color=colors_type, alpha=0.85, width=0.6)
        for bar, cnt, pct in zip(bars5, counts, pcts):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{cnt} ({pct:.0f}%)", ha="center", va="bottom", color="white", fontsize=9)
        ax5.set_xticks(x5)
        ax5.set_xticklabels(types, fontsize=8)
        ax5.set_ylabel("% de SL trades")
        ax5.set_ylim(0, 110)
        ax5.text(0.5, 0.92, "Un trade puede clasificar en múltiples tipos",
                 transform=ax5.transAxes, ha="center", fontsize=7, color=GRAY)

    # Panel 6 — EMA8/13 filter impact
    ax6 = fig.add_subplot(gs[2, 1])
    setup_ax(ax6, "Impacto filtro EMA8/13 en forward trades")
    labels_6 = [f"Sin filtro\n(N={n_no_filter})", f"Con EMA8/13\n(N={n_filter})"]
    wrs_6 = [wr_no_filter, wr_filter]
    colors_6 = [GRAY, GREEN if wr_filter >= wr_no_filter else RED]
    x6 = np.arange(2)
    bars6 = ax6.bar(x6, wrs_6, color=colors_6, alpha=0.85, width=0.5)
    for bar, wr in zip(bars6, wrs_6):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{wr:.1f}%", ha="center", va="bottom", color="white", fontsize=11)
    ax6.set_xticks(x6)
    ax6.set_xticklabels(labels_6)
    ax6.set_ylabel("Win Rate (%)")
    ax6.set_ylim(0, 100)
    ax6.axhline(wr_global, color=GOLD, linestyle="--", alpha=0.8, label=f"WR global {wr_global:.0f}%")
    ax6.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")

    fig.savefig(OUTPUT_DIR / "forward_lag_quality_BTCUSDT.png", dpi=130,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"\n  Figura 1 guardada: validation/output/forward_lag_quality_BTCUSDT.png")


def plot_figure2(enriched: pd.DataFrame):
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    tp = e[e["outcome"] == "tp"]
    sl = e[e["outcome"] == "sl"]
    n_total = len(e)
    wr_global = len(tp) / n_total * 100 if n_total > 0 else 0

    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle("Deep Dive Métricas — Forward Trades | Multi-símbolo Apr 2026",
                 color="white", fontsize=14, y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def setup_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    # Panel 1 — Stoch K box plot
    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, "Stoch RSI K al entrar — TP vs SL")
    data_box = [tp["stoch_k_at_entry"].dropna().values, sl["stoch_k_at_entry"].dropna().values]
    bp = ax1.boxplot(data_box, labels=["TP", "SL"], patch_artist=True,
                     medianprops={"color": GOLD, "linewidth": 2})
    bp["boxes"][0].set_facecolor(GREEN + "88")
    bp["boxes"][1].set_facecolor(RED + "88")
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("white")
    ax1.set_ylabel("Stoch K")

    # Panel 2 — Pullback dist ATR box plot
    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "Pullback dist ATR al entrar — TP vs SL")
    data_box2 = [tp["pullback_dist_atr"].dropna().values, sl["pullback_dist_atr"].dropna().values]
    bp2 = ax2.boxplot(data_box2, labels=["TP", "SL"], patch_artist=True,
                      medianprops={"color": GOLD, "linewidth": 2})
    bp2["boxes"][0].set_facecolor(GREEN + "88")
    bp2["boxes"][1].set_facecolor(RED + "88")
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp2[element]:
            item.set_color("white")
    ax2.set_ylabel("Pullback dist (ATR units)")

    # Panel 3 — Duration bars histogram
    ax3 = fig.add_subplot(gs[1, 0])
    setup_ax(ax3, "Duration (barras) — TP vs SL")
    bins3 = np.linspace(0, e["duration_bars"].max() + 5, 25)
    ax3.hist(tp["duration_bars"], bins=bins3, color=GREEN, alpha=0.7, label=f"TP (n={len(tp)})")
    ax3.hist(sl["duration_bars"], bins=bins3, color=RED, alpha=0.7, label=f"SL (n={len(sl)})")
    ax3.set_xlabel("Barras de 15m")
    ax3.set_ylabel("Frecuencia")
    ax3.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")

    # Panel 4 — MAE/MFE scatter
    ax4 = fig.add_subplot(gs[1, 1])
    setup_ax(ax4, "MAE vs MFE (en R) — TP=verde, SL=rojo")
    colors_sc = [GREEN if o == "tp" else RED for o in e["outcome"]]
    e_clean = e.dropna(subset=["mae_r", "mfe_r"])
    ax4.scatter(e_clean["mae_r"], e_clean["mfe_r"],
                c=[GREEN if o == "tp" else RED for o in e_clean["outcome"]],
                alpha=0.7, s=60)
    ax4.axhline(0, color=GRAY, alpha=0.4)
    ax4.axvline(0, color=GRAY, alpha=0.4)
    ax4.set_xlabel("MAE (R)")
    ax4.set_ylabel("MFE (R)")

    # Panel 5 — EMA8/13 impact bar chart
    ax5 = fig.add_subplot(gs[2, 0])
    setup_ax(ax5, "EMA8>EMA13 — Impacto en WR")
    groups = []
    for col, label in [("ema8_above_ema13", "EMA8>13"), ("ema8_cross_recent", "Cruce reciente")]:
        for val, lbl in [(True, "Con"), (False, "Sin")]:
            sub = e[e[col] == val]
            n_sub = len(sub)
            if n_sub == 0:
                continue
            n_tp = (sub["outcome"] == "tp").sum()
            wr, ci_lo, ci_hi = wr_ci(n_tp, n_sub)
            groups.append((f"{label}\n{lbl}\n(n={n_sub})", wr, wr - ci_lo, ci_hi - wr))
    if groups:
        x5 = np.arange(len(groups))
        wrs5 = [g[1] for g in groups]
        ci_l5 = [g[2] for g in groups]
        ci_h5 = [g[3] for g in groups]
        labels5 = [g[0] for g in groups]
        colors5 = [GREEN if w >= wr_global else RED for w in wrs5]
        ax5.bar(x5, wrs5, color=colors5, alpha=0.85, width=0.6,
                yerr=[ci_l5, ci_h5], capsize=4, error_kw={"color": "white", "alpha": 0.6})
        ax5.axhline(wr_global, color=GOLD, linestyle="--", alpha=0.8)
        ax5.set_xticks(x5)
        ax5.set_xticklabels(labels5, fontsize=7)
        ax5.set_ylabel("Win Rate (%)")
        ax5.set_ylim(0, 100)

    # Panel 6 — Waterfall top 10 trades by entry_time
    ax6 = fig.add_subplot(gs[2, 1])
    setup_ax(ax6, "Top 10 trades por PnL (% bruto) | ordenado por fecha")
    top10 = pd.concat([e.nlargest(5, "pnl_pct"), e.nsmallest(5, "pnl_pct")]) if len(e) >= 10 else e
    top10 = top10.sort_values("entry_time")
    x6 = np.arange(len(top10))
    colors6 = [GREEN if p > 0 else RED for p in top10["pnl_pct"]]
    bars6 = ax6.bar(x6, top10["pnl_pct"], color=colors6, alpha=0.85)
    ax6.axhline(0, color=GRAY, alpha=0.5)
    ax6.set_xticks(x6)
    short_labels = [f"{r['symbol'].split('-')[0]}\n{r['direction'][0]}" for _, r in top10.iterrows()]
    ax6.set_xticklabels(short_labels, fontsize=7)
    ax6.set_ylabel("PnL (%)")
    ax6.set_title("Top 10 trades seleccionados | por fecha de entrada", color="white", fontsize=10)

    fig.savefig(OUTPUT_DIR / "forward_metrics_detail_BTCUSDT.png", dpi=130,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Figura 2 guardada: validation/output/forward_metrics_detail_BTCUSDT.png")


# ─── Final report ────────────────────────────────────────────────────────────
def print_final_report(enriched: pd.DataFrame, sl_trades: pd.DataFrame,
                        corr_lag: float, wr_no_filter: float, wr_filter: float):
    e = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    tp = e[e["outcome"] == "tp"]
    sl = e[e["outcome"] == "sl"]
    n_total = len(e)
    wr_global = len(tp) / n_total * 100 if n_total > 0 else 0
    exp_r = e["pnl_r"].mean() if "pnl_r" in e else np.nan

    lag = e["lag_bars"]
    pct_lag_gt5 = (lag > 5).sum() / n_total * 100

    adx_delta = tp["adx_at_entry"].mean() - sl["adx_at_entry"].mean()
    body_delta = tp["body_ratio"].mean() - sl["body_ratio"].mean()
    ema_delta = wr_filter - wr_global

    # Classification counts
    n_sl = len(sl_trades)
    pct_a = sl_trades["type_a"].sum() / max(n_sl, 1) * 100
    pct_b = sl_trades["type_b"].sum() / max(n_sl, 1) * 100
    pct_c = sl_trades["type_c"].sum() / max(n_sl, 1) * 100
    pct_d = sl_trades["type_d"].sum() / max(n_sl, 1) * 100

    # Determine scenario
    lag_problem = corr_lag < -0.15 and pct_a > 40
    quality_problem = (abs(adx_delta) > 3 or abs(body_delta) > 0.05) and pct_b > 40
    stop_hunt_problem = pct_c > 30

    if lag_problem:
        scenario = "A"; causa = "TIMING"; accion = "Agregar cruce EMA8/13 como gatillo de entrada"
    elif quality_problem:
        scenario = "B"; causa = "CALIDAD"; accion = "Elevar adx_min a 25 o exigir body_ratio >= 0.40"
    elif stop_hunt_problem:
        scenario = "C"; causa = "STOP HUNT"; accion = "Aumentar buffer_atr de 0.25 a 0.40"
    else:
        scenario = "D"; causa = "RUIDO"; accion = "Acumular más trades (objetivo 60+) antes de cambiar nada"

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║    ANÁLISIS FORWARD TRADES — Multi-símbolo Apr 2026         ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║ DATOS                                                        ║")
    print(f"  ║  Trades analizados:    {n_total:<38d}║")
    print(f"  ║  Período:              Apr 3 - Apr 10, 2026                 ║")
    print(f"  ║  WR real:              {wr_global:.1f}%                                    ║")
    print(f"  ║  ExpR real:            {exp_r:+.3f}R (aprox)                          ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║ ANÁLISIS DE LAG                                              ║")
    print(f"  ║  Lag promedio:         {lag.mean():.1f} barras ({lag.mean()*15:.0f} min){'':>22s}║")
    print(f"  ║  % trades lag > 5b:    {pct_lag_gt5:.1f}%{'':>42s}║")
    print(f"  ║  Correlación lag/WR:   r={corr_lag:.3f}{'':>40s}║")
    lag_verdict = "SÍ" if corr_lag < -0.15 else ("NO" if corr_lag > -0.05 else "PARCIAL")
    print(f"  ║  Lag es el problema:   {lag_verdict:<39s}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║ ANÁLISIS DE CALIDAD                                          ║")
    print(f"  ║  ADX delta TP vs SL:   {adx_delta:+.1f} pts{'':>41s}║")
    print(f"  ║  Body delta TP vs SL:  {body_delta:+.3f}{'':>41s}║")
    print(f"  ║  EMA8/13 mejora WR:    {ema_delta:+.1f}pp ({wr_global:.1f}→{wr_filter:.1f}%){'':>28s}║")
    quality_verdict = "SÍ" if quality_problem else ("NO" if not quality_problem else "PARCIAL")
    print(f"  ║  Calidad es problema:  {quality_verdict:<39s}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║ CLASIFICACIÓN SL TRADES (N={n_sl}){'':>33s}║")
    print(f"  ║  Tipo A (lag tardío):  {pct_a:.0f}%{'':>42s}║")
    print(f"  ║  Tipo B (setup débil): {pct_b:.0f}%{'':>42s}║")
    print(f"  ║  Tipo C (stop hunt):   {pct_c:.0f}%{'':>42s}║")
    print(f"  ║  Tipo D (contra-tend): {pct_d:.0f}%{'':>42s}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║ VEREDICTO                                                    ║")
    print(f"  ║  Causa principal:      {causa:<39s}║")
    print(f"  ║  Escenario:            {scenario:<39s}║")
    print(f"  ║                                                              ║")
    accion_wrapped = accion[:55]
    print(f"  ║  Acción recomendada:   {accion_wrapped:<39s}║")
    print(f"  ║  Prioridad:            {'BAJA (n=34, ampliar muestra)':<39s}║")
    print(f"  ║                                                              ║")
    print(f"  ║  NO cambiar hasta:     60+ trades completados en paper       ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    print()
    print("─"*65)
    print("LÍNEA DE ACCIÓN:")
    print(f'  "El problema principal es {causa}.')
    print(f'   Acción: {accion}.')
    print(f'   Implementar: cuando se acumulen 60+ trades paper."')
    print("─"*65)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("ANÁLISIS DE LAG Y CALIDAD — Forward Test Trades")
    print("Multi-símbolo | TrendBot V2 | Apr 2026")
    print("="*65)

    # Load trades
    trades_df = pd.read_csv(ROOT / "data" / "paper_trades_v2.csv")
    completed = trades_df[~trades_df["exit_type"].isin(["shutdown"])].copy()

    n_completed = len(completed)
    print(f"\nTrades completados (non-shutdown): {n_completed}")

    if n_completed < 20:
        print("\n⚠  MUESTRA INSUFICIENTE — esperar más trades (n < 20)")
        print("   No se pueden sacar conclusiones estadísticas.")
        return

    # Enrich
    print("\nEnriqueciendo trades con métricas OHLCV...")
    enriched = enrich_trades(completed)
    print(f"Enriquecidos: {len(enriched)} trades")

    # Filter to sl/tp only
    e_final = enriched[enriched["outcome"].isin(["sl", "tp"])].copy()
    n_final = len(e_final)
    print(f"Completados sl+tp: {n_final}")

    if n_final < 20:
        print(f"\n⚠  Solo {n_final} trades con outcome sl/tp — muestra pequeña, CI amplios.")

    # Save enriched CSV
    out_path = ROOT / "data" / "forward_trades_enriched.csv"
    enriched.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}")

    # Steps
    print_step0(trades_df, enriched)
    corr = print_step2(e_final)
    print_step3(e_final)
    sl_trades = print_step4(e_final)
    wr_no_f, wr_f, n_no_f, n_f = print_step5(e_final)

    # Plots
    print("\nGenerando figuras...")
    plot_figure1(e_final, sl_trades, corr, wr_no_f, wr_f, n_no_f, n_f)
    plot_figure2(e_final)

    # Final report
    print_final_report(e_final, sl_trades, corr, wr_no_f, wr_f)


if __name__ == "__main__":
    main()
