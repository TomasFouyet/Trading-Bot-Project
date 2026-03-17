"""
Multi-objective Bayesian optimizer for RSI Divergence strategy — Optuna / NSGA-II edition.

Objectives (all maximized simultaneously):
  1. total_pnl_pct  — total return as % of initial balance
  2. winrate        — win rate (% of profitable trades)
  3. total_trades   — number of trades taken (rewards active strategies)

Optuna stores every trial in a SQLite file → stop and resume at any time.

──────────────────────────────────────────────────────────────────────────────
HOW TO CONFIGURE
──────────────────────────────────────────────────────────────────────────────
Edit SEARCH_SPACE at the top of this file to control which parameters Optuna
explores and their ranges.  You do NOT need to touch the objective() function.

  type "int"         → trial.suggest_int(name, low, high)
  type "float"       → trial.suggest_float(name, low, high, step=step)
  type "categorical" → trial.suggest_categorical(name, choices)

Parameters NOT listed in SEARCH_SPACE fall back to FIXED_PARAMS (or the
strategy's own default if not present there either).  Adding or removing a
strategy parameter only requires editing SEARCH_SPACE / FIXED_PARAMS.

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────

  # Run 300 trials (sequential, recommended for NSGA-II):
  python scripts/optimize_optuna.py \\
      --symbol BTC-USDT --timeframe 5m \\
      --start 2025-11-20 --end 2026-03-06 \\
      --n-trials 300 --output pareto.csv

  # Resume a previous study (same study-name → loads from SQLite):
  python scripts/optimize_optuna.py --study-name rsi_nsga2_BTC-USDT_5m \\
      --n-trials 200 --output pareto.csv

  # Parallel: run two terminals pointing at the same SQLite storage
  python scripts/optimize_optuna.py --study-name my_study --n-trials 150 &
  python scripts/optimize_optuna.py --study-name my_study --n-trials 150 &

  # Change objective weights (see --pnl-weight / --wr-weight / --trades-weight):
  python scripts/optimize_optuna.py --pnl-weight 2.0 --wr-weight 1.0 --trades-weight 0.5

──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import optuna
import pandas as pd


# ── Search space ───────────────────────────────────────────────────────────────
#
# Edit THIS dict to change what Optuna explores.  The objective() function reads
# this automatically — no other code needs to change.
#
# Keys must match the parameter names accepted by RSIDivergenceStrategy.__init__.
# Any key NOT listed here will use its value from FIXED_PARAMS (or strategy default).
#
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # ── RSI / EMA indicators ──────────────────────────────────────────────────
    "rsi_period": {
        "type": "int",
        "low": 8, "high": 14,
        "description": "RSI lookback period",
    },
    "ema_period": {
        "type": "int",
        "low": 9, "high": 22,
        "description": "EMA trigger period",
    },
    # ── Swing detection ───────────────────────────────────────────────────────
    "swing_window": {
        "type": "int",
        "low": 2, "high": 15,
        "description": "Bars on each side to confirm a pivot",
    },
    "swing_separation": {
        "type": "int",
        "low": 3, "high": 20,
        "description": "Min bars between two compared swings",
    },
    # ── RSI thresholds ────────────────────────────────────────────────────────
    "rsi_oversold": {
        "type": "float",
        "low": 20.0, "high": 30.0, "step": 2.5,
        "description": "RSI level for bullish divergence",
    },
    "rsi_overbought": {
        "type": "float",
        "low": 70.0, "high": 80.0, "step": 2.5,
        "description": "RSI level for bearish divergence",
    },
    # ── Risk / reward ─────────────────────────────────────────────────────────
    "rr_ratio": {
        "type": "float",
        "low": 1.0, "high": 3.5, "step": 0.05,
        "description": "TP1 risk:reward (closes 70% of position)",
    },
    "tp2_ratio": {
        "type": "float",
        "low": 1.2, "high": 5.0, "step": 0.05,
        "description": "TP2 risk:reward (closes remaining 30%)",
    },
    "sl_buffer_pct": {
        "type": "float",
        "low": 0.001, "high": 0.015, "step": 0.001,
        "description": "SL placed this % beyond swing extreme",
    },
    # ── Trend filter ──────────────────────────────────────────────────────────
    "trend_ema_period": {
        "type": "categorical",
        "choices": [0, 20, 50],
        "description": "HTF EMA period (0 = disabled)",
    },
    "min_trend_coeff": {
        "type": "float",
        "low": 0.0, "high": 0.9, "step": 0.1,
        "description": "Min HTF trend coefficient to allow entry",
    },
    # ── Entry timing ──────────────────────────────────────────────────────────
    "entry_cooldown_bars": {
        "type": "int",
        "low": 0, "high": 30,
        "description": "Min bars between consecutive entry signals",
    },
}

# ── Fixed parameters ────────────────────────────────────────────────────────────
#
# Parameters held constant across all trials.
# These are merged with the sampled params before each backtest.
# Values here take lower priority than SEARCH_SPACE when both define the same key.
#
FIXED_PARAMS: dict[str, Any] = {
    "swing_lookback":   100,
    "trigger_window":   10,
    "allow_short":      True,
    "trend_slope_bars": 5,
    "entry_window":     2,
    "tp1_close_pct":    0.70,
}

# Minimum trades for a trial to appear in the Pareto front (not a hard constraint —
# just used when printing the "best" single-objective results at the end)
MIN_TRADES_DISPLAY = 5


# ── Indicator cache ─────────────────────────────────────────────────────────────
#
# Computing RSI/EMA is fast but calling it 300 times for the same rsi_period+ema_period
# is wasteful.  We cache by (rsi_period, ema_period) → full precomputed DataFrame.
# The cache lives in the main process across all sequential trials.
#
_INDICATOR_CACHE: dict[tuple, pd.DataFrame] = {}
_BARS_DF: pd.DataFrame | None = None          # full raw bars (loaded once)
_TS_MS_TO_POS: dict[int, int] | None = None   # int(ms) → row index


def _get_precomputed(rsi_period: int, ema_period: int, strategy) -> pd.DataFrame:
    """Return (and cache) precomputed indicator DataFrame for this rsi/ema combo."""
    key = (rsi_period, ema_period)
    if key not in _INDICATOR_CACHE:
        _INDICATOR_CACHE[key] = strategy._compute_indicators(_BARS_DF)
    return _INDICATOR_CACHE[key]


# ── Parameter suggestion ────────────────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Build a parameter dict by calling the appropriate suggest_* method for each
    entry in SEARCH_SPACE, then merging with FIXED_PARAMS.

    To add a new parameter: add an entry to SEARCH_SPACE — nothing else changes.
    """
    sampled: dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec["type"]
        if kind == "int":
            sampled[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "float":
            kwargs: dict[str, Any] = {"low": spec["low"], "high": spec["high"]}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            sampled[name] = trial.suggest_float(name, **kwargs)
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type '{kind}' for param '{name}'")

    # FIXED_PARAMS fill in any gaps (sampled takes priority if same key)
    return {**FIXED_PARAMS, **sampled}


# ── Single backtest runner ──────────────────────────────────────────────────────

async def _run_backtest_async(params: dict[str, Any], cfg: dict) -> dict | None:
    """
    Run one backtest with the given params.  Returns the metrics dict, or None on error.

    This function is intentionally decoupled from strategy internals: it only reads
    the standard 'metrics' dict from result.to_report() — strategy changes don't break it.
    """
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.data.parquet_store import ParquetStore
    from app.engine.backtest import BacktestEngine
    from app.strategy import get_strategy
    from app.strategy.base import BaseStrategy

    configure_logging(log_level="ERROR", log_format="console")
    get_settings()

    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    start     = cfg["start"]
    end       = cfg["end"]
    balance   = Decimal(str(cfg["balance"]))
    comm_rate = Decimal(str(cfg["commission_bps"])) / 10000
    slip_rate = Decimal(str(cfg["slippage_bps"])) / 10000

    try:
        store    = ParquetStore()
        strategy = get_strategy("rsi_divergence", symbol=symbol, params=params)

        # ── Inject indicator cache ──────────────────────────────────────────
        # Avoid recomputing RSI/EMA when only non-indicator params changed.
        rsi_p = params.get("rsi_period", 9)
        ema_p = params.get("ema_period", 14)

        if _BARS_DF is not None:
            precomputed = _get_precomputed(rsi_p, ema_p, strategy)
            ts_ms_to_pos = _TS_MS_TO_POS
            orig_compute = strategy._compute_indicators

            def _cached_compute(df: pd.DataFrame) -> pd.DataFrame:
                if "rsi" in df.columns:
                    return df
                ts_ms = int(df["ts"].iloc[-1].timestamp() * 1_000)
                end_pos = ts_ms_to_pos.get(ts_ms)
                if end_pos is None:
                    return orig_compute(df)
                return precomputed.iloc[end_pos - len(df) + 1 : end_pos + 1]

            strategy._compute_indicators = _cached_compute

            # Patch bars_to_df to skip Decimal→float conversion per bar
            _orig_bars_to_df = BaseStrategy.__dict__["bars_to_df"].__func__
            _raw_df = _BARS_DF[["ts", "open", "high", "low", "close", "volume"]]

            def _fast_bars_to_df(bars_list):
                if not bars_list:
                    return _orig_bars_to_df(bars_list)
                ts_ms = int(bars_list[-1].ts.timestamp() * 1_000)
                pos = ts_ms_to_pos.get(ts_ms)
                if pos is None:
                    return _orig_bars_to_df(bars_list)
                n = len(bars_list)
                return _raw_df.iloc[pos - n + 1 : pos + 1]

            BaseStrategy.bars_to_df = staticmethod(_fast_bars_to_df)
        # ───────────────────────────────────────────────────────────────────

        engine = BacktestEngine(
            strategy=strategy,
            store=store,
            initial_balance=balance,
            commission_rate=comm_rate,
            slippage_rate=slip_rate,
            verbose=False,
            max_daily_drawdown_pct=Decimal("100"),  # disable kill switch during optimization
        )
        result = await engine.run(symbol, timeframe, start, end, params)
        m = result.to_report()["metrics"]
        return m
    except Exception as exc:
        return None


# ── Objective function ──────────────────────────────────────────────────────────

def _make_objective(cfg: dict):
    """
    Returns the objective function closed over the backtest config.

    Returns THREE values, all to be MAXIMIZED:
      (total_pnl_pct, winrate, total_trades)

    Trials with < MIN_TRADES are penalised to guide NSGA-II away from
    degenerate strategies that barely trade.
    """
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = _suggest_params(trial)
        m = asyncio.run(_run_backtest_async(params, cfg))

        if m is None or m.get("total_trades", 0) == 0:
            # Hard failure: return worst possible values
            return (-1000.0, 0.0, 0.0)

        pnl_pct     = float(m.get("total_pnl_pct", -1000.0))
        winrate     = float(m.get("winrate", 0.0))
        n_trades    = float(m.get("total_trades", 0))
        sharpe      = float(m.get("sharpe_ratio", 0.0))
        max_dd      = float(m.get("max_drawdown_pct", 100.0))

        # Store extra metrics as trial attributes for the CSV report
        trial.set_user_attr("total_pnl",     round(float(m.get("total_pnl", 0)), 2))
        trial.set_user_attr("total_pnl_pct", round(pnl_pct, 2))
        trial.set_user_attr("winrate",        round(winrate, 1))
        trial.set_user_attr("total_trades",   int(n_trades))
        trial.set_user_attr("sharpe",         round(sharpe, 3))
        trial.set_user_attr("max_drawdown",   round(max_dd, 2))
        trial.set_user_attr("avg_trade_pnl",  round(float(m.get("avg_trade_pnl", 0)), 2))

        # Penalise very low trade count — we want active strategies
        trade_score = n_trades if n_trades >= MIN_TRADES_DISPLAY else n_trades * 0.1

        return (pnl_pct, winrate, trade_score)

    return objective


# ── Progress callback ───────────────────────────────────────────────────────────

_BAR_WIDTH = 36   # characters for the filled/empty bar


def _progress_bar(done: int, total: int, width: int = _BAR_WIDTH) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _eta_label(eta_secs: float) -> str:
    """Human-readable ETA with colour hints (green < 5 min, yellow < 30 min, red ≥ 30 min)."""
    s = int(eta_secs)
    text = _fmt_dur(s)
    if not sys.stdout.isatty():
        return text
    # ANSI colours
    if s < 300:
        return f"\033[32m{text}\033[0m"    # green
    if s < 1800:
        return f"\033[33m{text}\033[0m"    # yellow
    return f"\033[31m{text}\033[0m"        # red


def _make_callback(total_trials: int, t0: float):
    """
    Prints a two-line live progress display (TTY) or periodic status blocks (non-TTY).

    TTY output (rewrites same two lines every trial):
    ──────────────────────────────────────────────────────────────
      [████████████████░░░░░░░░░░░░░░░░░░░░]  80/300  26.7%
      elapsed=1m05s   ETA=3m02s   rate=1.2 t/s   pareto=14
      trial pnl=+3.21%  wr=58.3%  trades=22  rr=1.55  rsi=9  ema=14
    ──────────────────────────────────────────────────────────────

    Non-TTY (Docker / pipe): prints a compact block every 10 trials.
    """
    _best_pnl    = [-1e9]
    _best_wr     = [0.0]
    _best_trades = [0]
    _best_sharpe = [-1e9]
    _printed_lines = [0]   # how many lines were written last time (for ANSI erase)
    _local_done  = [0]     # trials completed by THIS worker (not global study count)

    # ANSI: move cursor up N lines and clear to end of screen
    def _erase_prev(n: int) -> None:
        if n > 0:
            sys.stdout.write(f"\033[{n}A\033[J")

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        done    = trial.number + 1
        elapsed = time.monotonic() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (total_trials - done) / rate if rate > 0 else 0
        pct     = done / total_trials * 100

        # ── Track session bests ────────────────────────────────────────────
        ua = trial.user_attrs
        trial_pnl    = ua.get("total_pnl_pct", -1e9)
        trial_wr     = ua.get("winrate", 0.0)
        trial_trades = ua.get("total_trades", 0)
        trial_sharpe = ua.get("sharpe", -1e9)

        if trial_pnl    > _best_pnl[0]:    _best_pnl[0]    = trial_pnl
        if trial_wr     > _best_wr[0]:     _best_wr[0]     = trial_wr
        if trial_trades > _best_trades[0]: _best_trades[0] = trial_trades
        if trial_sharpe > _best_sharpe[0]: _best_sharpe[0] = trial_sharpe

        n_pareto = len(study.best_trials)
        p = trial.params   # sampled params for this trial

        # ── Build display lines ────────────────────────────────────────────
        bar   = _progress_bar(done, total_trials)
        eta_s = _eta_label(eta)

        line1 = (
            f"  [{bar}]  {done}/{total_trials}  ({pct:.1f}%)"
        )
        line2 = (
            f"  elapsed={_fmt_dur(elapsed)}   ETA={eta_s}   "
            f"rate={rate:.2f} t/s   pareto_front={n_pareto}"
        )
        line3 = (
            f"  this trial → "
            f"pnl={trial_pnl:+.2f}%  wr={trial_wr:.1f}%  "
            f"trades={trial_trades}  "
            f"rr={p.get('rr_ratio', '?')}  "
            f"rsi={p.get('rsi_period', '?')}  "
            f"ema={p.get('ema_period', '?')}  "
            f"sw={p.get('swing_window', '?')}"
        )
        line4 = (
            f"  session best → "
            f"pnl={_best_pnl[0]:+.2f}%  wr={_best_wr[0]:.1f}%  "
            f"trades={_best_trades[0]}  sharpe={_best_sharpe[0]:.3f}"
        )

        if sys.stdout.isatty():
            _erase_prev(_printed_lines[0])
            output = "\n".join([line1, line2, line3, line4]) + "\n"
            sys.stdout.write(output)
            sys.stdout.flush()
            _printed_lines[0] = 4   # 4 lines written

        else:
            # Non-TTY (Docker, pipe, CI): one line per trial
            print(
                f"[{done:>4}/{total_trials}  {pct:5.1f}%]"
                f"  ETA={_fmt_dur(eta):<8}"
                f"  trial: pnl={trial_pnl:+6.2f}%  wr={trial_wr:5.1f}%  trades={trial_trades:>3}"
                f"  best:  pnl={_best_pnl[0]:+6.2f}%  pareto={n_pareto}",
                flush=True,
            )

    return callback


def _fmt_dur(secs: float) -> str:
    s = int(secs)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


# ── Pareto front reporting ──────────────────────────────────────────────────────

def _print_pareto(study: optuna.Study, top_n: int = 20) -> None:
    """Print the Pareto-optimal trials sorted by PnL descending."""
    best = study.best_trials
    if not best:
        print("  No Pareto-optimal trials found.")
        return

    # Sort by pnl_pct descending
    best_sorted = sorted(
        best,
        key=lambda t: t.user_attrs.get("total_pnl_pct", -1e9),
        reverse=True,
    )

    print(f"\n{'='*85}")
    print(f"  PARETO FRONT  —  {len(best)} non-dominated solutions  (top {min(top_n, len(best))} by PnL)")
    print(f"{'='*85}")
    print(
        f"  {'#':>4}  {'PnL%':>7}  {'WR%':>6}  {'Trades':>7}  {'Sharpe':>7}  {'MaxDD%':>7}"
        f"  {'rsi':>4}  {'ema':>4}  {'sw':>4}  {'sep':>4}  {'rr':>5}  {'tp2':>5}"
        f"  {'ovs':>5}  {'obg':>5}  {'cool':>5}"
    )
    print(f"  {'-'*81}")

    for i, t in enumerate(best_sorted[:top_n]):
        ua = t.user_attrs
        p  = t.params
        print(
            f"  {i+1:>4}  {ua.get('total_pnl_pct',0):>+7.2f}  "
            f"{ua.get('winrate',0):>6.1f}  {ua.get('total_trades',0):>7}  "
            f"{ua.get('sharpe',0):>7.3f}  {ua.get('max_drawdown',0):>7.2f}  "
            f"{p.get('rsi_period','?'):>4}  {p.get('ema_period','?'):>4}  "
            f"{p.get('swing_window','?'):>4}  {p.get('swing_separation','?'):>4}  "
            f"{p.get('rr_ratio',0):>5.2f}  {p.get('tp2_ratio',0):>5.2f}  "
            f"{p.get('rsi_oversold',0):>5.1f}  {p.get('rsi_overbought',0):>5.1f}  "
            f"{p.get('entry_cooldown_bars','?'):>5}"
        )

    print(f"{'='*85}\n")


def _save_results(study: optuna.Study, output_path: str) -> None:
    """Save ALL trials to CSV and Pareto front to JSON."""
    # All trials → CSV
    records = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number}
        row.update(t.params)
        row.update(t.user_attrs)
        if t.values:
            row["obj_pnl_pct"]   = t.values[0]
            row["obj_winrate"]   = t.values[1]
            row["obj_trades"]    = t.values[2]
        records.append(row)

    if records:
        df = pd.DataFrame(records)
        df.sort_values("total_pnl_pct", ascending=False, inplace=True)
        df.to_csv(output_path, index=False)
        print(f"  All trials saved → {output_path}  ({len(df)} rows)")

    # Pareto front → JSON (best params for each solution)
    pareto_path = output_path.replace(".csv", "_pareto.json")
    pareto = []
    for t in sorted(
        study.best_trials,
        key=lambda x: x.user_attrs.get("total_pnl_pct", -1e9),
        reverse=True,
    ):
        pareto.append({
            "trial":  t.number,
            "params": {**FIXED_PARAMS, **t.params},
            "metrics": t.user_attrs,
        })
    with open(pareto_path, "w") as f:
        json.dump(pareto, f, indent=2, default=str)
    print(f"  Pareto front saved → {pareto_path}  ({len(pareto)} solutions)")


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.data.parquet_store import ParquetStore
    from app.strategy import get_strategy
    from app.strategy.base import BaseStrategy

    configure_logging(log_level="ERROR", log_format="console")
    get_settings()

    # ── Validate data ──────────────────────────────────────────────────────────
    store = ParquetStore()
    min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
    if min_ts is None:
        print(f"[ERROR] No data for {args.symbol}/{args.timeframe}. Run ingest_data.py first.")
        sys.exit(1)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # ── Pre-load raw bars into module-level cache ──────────────────────────────
    # Workers reuse this across all trials — avoids re-reading Parquet every trial.
    global _BARS_DF, _TS_MS_TO_POS
    print("\n  Loading bars into memory...", end=" ", flush=True)
    _t0_load = time.monotonic()
    all_bars  = store.read_bars(args.symbol, args.timeframe, start, end)
    _BARS_DF  = BaseStrategy.bars_to_df(all_bars)
    _ts_ms_arr = (pd.DatetimeIndex(_BARS_DF["ts"]).asi8 // 1_000_000).tolist()
    _TS_MS_TO_POS = dict(zip(_ts_ms_arr, range(len(_ts_ms_arr))))
    print(f"{len(_BARS_DF):,} bars in {_fmt_dur(time.monotonic() - _t0_load)}")

    # ── Study name / storage ───────────────────────────────────────────────────
    study_name = args.study_name or f"rsi_nsga2_{args.symbol}_{args.timeframe}"
    db_path    = args.db_path or os.path.join(
        os.path.dirname(__file__), f"{study_name}.db"
    )
    storage    = f"sqlite:///{db_path}"

    # ── Print header ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RSI Divergence — Multi-Objective Optimizer (Optuna / NSGA-II)")
    print(f"{'='*70}")
    print(f"  Symbol:      {args.symbol} {args.timeframe}")
    print(f"  Period:      {args.start} → {args.end}")
    print(f"  Balance:     ${args.balance:,.2f}")
    print(f"  Trials:      {args.n_trials}")
    print(f"  Population:  {args.population_size} (NSGA-II)")
    print(f"  Objectives:  PnL%  |  Win rate  |  Trades")
    print(f"  Study name:  {study_name}")
    print(f"  Storage:     {db_path}")
    print(f"  Data range:  {min_ts} → {max_ts}")
    print(f"{'='*70}")
    print(f"\n  Search space ({len(SEARCH_SPACE)} parameters):")
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "categorical":
            val_str = str(spec["choices"])
        elif spec["type"] == "int":
            val_str = f"[{spec['low']} .. {spec['high']}]"
        else:
            step_str = f"  step={spec['step']}" if "step" in spec else ""
            val_str = f"[{spec['low']} .. {spec['high']}]{step_str}"
        print(f"    {name:<25} {val_str}")
    print()

    # ── Create / load study ────────────────────────────────────────────────────
    sampler = optuna.samplers.NSGAIISampler(
        population_size=args.population_size,
        seed=args.seed,
    )

    # Suppress Optuna's own logging (we have our own progress callback)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize", "maximize"],
        sampler=sampler,
        load_if_exists=True,  # resume automatically if study already exists
    )

    existing_trials = len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE])
    if existing_trials > 0:
        print(f"  Resuming study: {existing_trials} trials already completed.\n")

    # ── Backtest config (passed to every trial) ────────────────────────────────
    cfg = {
        "symbol":         args.symbol,
        "timeframe":      args.timeframe,
        "start":          start,
        "end":            end,
        "balance":        args.balance,
        "commission_bps": args.commission_bps,
        "slippage_bps":   args.slippage_bps,
    }

    # ── Run optimization ───────────────────────────────────────────────────────
    objective  = _make_objective(cfg)
    t0         = time.monotonic()
    callback   = _make_callback(args.n_trials, t0)

    print(f"  Running {args.n_trials} trials...\n")
    # Print 4 blank lines so the first callback erase+rewrite works correctly
    if sys.stdout.isatty():
        sys.stdout.write("\n" * 4)
        sys.stdout.flush()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[callback],
        show_progress_bar=False,
        gc_after_trial=False,    # avoid GC pauses between trials
    )

    if sys.stdout.isatty():
        print()  # blank line after the 4-line progress block

    elapsed = time.monotonic() - t0
    total_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  Done!  {args.n_trials} new trials in {_fmt_dur(elapsed)}"
          f"  ({args.n_trials / elapsed:.1f} trials/s)"
          f"  |  {total_done} total in study")

    # ── Print Pareto front ─────────────────────────────────────────────────────
    _print_pareto(study, top_n=args.top_n)

    # ── Save results ───────────────────────────────────────────────────────────
    if args.output:
        _save_results(study, args.output)
        print()

    # ── Print recommended params (best by PnL, min MIN_TRADES_DISPLAY trades) ──
    candidates = [
        t for t in study.best_trials
        if t.user_attrs.get("total_trades", 0) >= MIN_TRADES_DISPLAY
    ]
    if candidates:
        best = max(candidates, key=lambda t: t.user_attrs.get("total_pnl_pct", -1e9))
        recommended = {**FIXED_PARAMS, **best.params}
        print("  Recommended params (best PnL from Pareto front, ≥ min trades):")
        print(f"  {json.dumps(recommended, indent=4, default=str)}\n")
    else:
        print("  No Pareto trial met the minimum trade count. Lower MIN_TRADES_DISPLAY"
              " or run more trials.\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-objective Optuna optimizer for RSI Divergence strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backtest config
    parser.add_argument("--symbol",    default="BTC-USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start",     default="2025-11-20")
    parser.add_argument("--end",       default="2026-03-06")
    parser.add_argument("--balance",   type=float, default=10_000.0)
    parser.add_argument("--commission-bps", type=float, default=7.5,  dest="commission_bps")
    parser.add_argument("--slippage-bps",   type=float, default=5.0,  dest="slippage_bps")

    # Optuna config
    parser.add_argument("--n-trials",        type=int, default=300, dest="n_trials",
                        help="Number of new trials to run in this session")
    parser.add_argument("--population-size", type=int, default=50,  dest="population_size",
                        help="NSGA-II population size per generation (≥ 2*objectives = 6)")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--study-name",      default=None, dest="study_name",
                        help="Study name (auto-generated from symbol+timeframe if omitted)")
    parser.add_argument("--db-path",         default=None, dest="db_path",
                        help="SQLite DB file path (default: scripts/<study_name>.db)")

    # Output
    parser.add_argument("--output", default="pareto_results.csv",
                        help="CSV file to save all trial results")
    parser.add_argument("--top-n", type=int, default=20, dest="top_n",
                        help="Number of Pareto-optimal solutions to print")
    parser.add_argument("--export-only", action="store_true", dest="export_only",
                        help="Load existing study from --db-path and export results without running new trials")

    args = parser.parse_args()

    if args.export_only:
        import os
        study_name = args.study_name or f"rsi_nsga2_{args.symbol}_{args.timeframe}"
        db_path    = args.db_path or os.path.join(os.path.dirname(__file__), f"{study_name}.db")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"\n  Loaded study '{study_name}' from {db_path}")
        print(f"  Completed trials: {len(complete)}  |  Pareto front: {len(study.best_trials)}\n")
        _print_pareto(study, top_n=args.top_n)
        if args.output:
            _save_results(study, args.output)
            print()
        candidates = [t for t in study.best_trials if t.user_attrs.get("total_trades", 0) >= MIN_TRADES_DISPLAY]
        if candidates:
            best = max(candidates, key=lambda t: t.user_attrs.get("total_pnl_pct", -1e9))
            print("  Recommended params (best PnL from Pareto front):")
            print(f"  {json.dumps({**FIXED_PARAMS, **best.params}, indent=4, default=str)}\n")
    else:
        main(args)
