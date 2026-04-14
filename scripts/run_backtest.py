"""
Run a backtest locally without Celery.

Usage:
    python scripts/run_backtest.py \
        --symbol BTC-USDT \
        --timeframe 5m \
        --start 2024-01-01 \
        --end 2024-06-01 \
        --balance 10000

Requires data to already be ingested in Parquet.
To ingest data first, run scripts/ingest_data.py or POST /strategy/run_backtest.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal


async def main(args: argparse.Namespace) -> None:
    # Configure logging
    from app.config import get_settings
    from app.core.logging import configure_logging
    settings = get_settings()
    configure_logging(log_level="INFO", log_format="console")

    from app.data.parquet_store import ParquetStore
    from app.engine.backtest import BacktestEngine
    from app.strategy import get_strategy

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # Build strategy params depending on chosen strategy
    if args.strategy == "ema_cross":
        strategy_params = {
            "ema_fast": args.ema_fast,
            "ema_slow": args.ema_slow,
            "sma_trend": args.sma_trend,
        }
    elif args.strategy in ("rsi_divergence", "hybrid_rsi_pivot"):
        strategy_params = {
            "rsi_period": args.rsi_period,
            "ema_period": args.ema_period,
            "swing_window": args.swing_window,
            "swing_separation": args.swing_separation,
            "swing_lookback": args.swing_lookback,
            "trigger_window": args.trigger_window,
            "rsi_oversold": args.rsi_oversold,
            "rsi_overbought": args.rsi_overbought,
            "allow_short": args.allow_short,
            "sl_buffer_pct": args.sl_buffer_pct,
            "rr_ratio": args.rr_ratio,
            "tp2_ratio": args.tp2_ratio,
            "min_trend_coeff": args.min_trend_coeff,
            "trend_ema_period": args.trend_ema_period,
            "trend_slope_bars": args.trend_slope_bars,
            "entry_window": args.entry_window,
            "entry_cooldown_bars": args.entry_cooldown_bars,
            "max_concurrent_positions": args.max_concurrent,
            # Cali technique filters
            "use_ema_cross_filter": args.use_ema_cross_filter,
            "ema_fast_filter": args.ema_fast_filter,
            "ema_slow_filter": args.ema_slow_filter,
            "vol_confirm_enabled": args.vol_confirm_enabled,
            "vol_confirm_period": args.vol_confirm_period,
            "sr_zone_enabled": args.sr_zone_enabled,
            "sr_zone_pct": args.sr_zone_pct,
            "tp1_close_pct": args.tp1_close_pct,
            # Hybrid-only params
            "atr_period": args.atr_period,
            "atr_sl_mult": args.atr_sl_mult,
            "pivot_atr_proximity": args.pivot_atr_proximity,
            "vol_avg_period": args.vol_avg_period,
            "vol_min_ratio": args.vol_min_ratio,
            "session_start_utc": args.session_start_utc,
            "session_end_utc": args.session_end_utc,
            "bars_per_day": args.bars_per_day,
        }
    elif args.strategy == "mean_reversion":
        strategy_params = {
            "rsi_oversold":  args.rsi_oversold,
            "rsi_overbought": args.rsi_overbought,
            "adx_max":       args.adx_max,
            "min_rr":        args.min_rr,
            "allow_short":   args.allow_short,
            "swing_lookback": args.swing_lookback,
        }
    elif args.strategy in ("trend_following", "trend_following_v2"):
        strategy_params = {
            # Core
            "adx_min":               args.adx_min,
            "min_rr":                args.min_rr,
            "pullback_tolerance_atr": args.pullback_tolerance_atr,
            "allow_short":           args.allow_short,
            "ema_fast":              args.ema_fast_tf,
            "ema_slow":              args.ema_slow_tf,
            "slope_bars":            args.slope_bars,
            # Layer 1: Confidence
            "use_confidence":        args.tf_use_confidence,
            "adx_strong":            args.tf_adx_strong,
            "tight_pb_atr":          args.tf_tight_pb_atr,
            "min_confidence":        args.tf_min_confidence,
            # Layer 2: Session
            "use_session_filter":    args.tf_use_session,
            "us_session_start":      args.tf_us_start,
            "us_session_end":        args.tf_us_end,
            "session_mult_us":       args.tf_sess_us,
            "session_mult_eu":       args.tf_sess_eu,
            "session_mult_other":    args.tf_sess_other,
            # Layer 3: Streak
            "use_streak_adj":        args.tf_use_streak,
            "streak_euphoria_after": args.tf_streak_after,
            "streak_euphoria_mult":  args.tf_streak_mult,
            # Layer 4: Patience (SL triggers on wick — keep disabled)
            "use_patience":          args.tf_use_patience,
            "soft_sl_bars":          args.tf_soft_sl_bars,
            # ATR-based SL + price floor
            "sl_min_atr":            args.sl_min_atr,
            "sl_max_atr":            args.sl_max_atr,
            "sl_min_pct":            args.sl_min_pct,
            # Reversal swap
            "enable_reversal":       not args.no_reversal,
        }
    elif args.strategy == "event_driven":
        strategy_params = {
            "vol_multiplier": args.vol_multiplier,
            "lookback":       args.event_lookback,
            "min_rr":         args.min_rr,
            "max_hold_bars":  args.max_hold_bars,
            "allow_short":    args.allow_short,
            "tp_rr":          args.tp_rr,
        }
    else:
        strategy_params = {}

    strategy = get_strategy(args.strategy, symbol=args.symbol, params=strategy_params)
    store = ParquetStore()

    min_ts, max_ts = store.get_date_range(args.symbol, args.timeframe)
    if min_ts is None:
        print(f"[ERROR] No data found for {args.symbol}/{args.timeframe}")
        print("       Run data ingestion first:")
        print(f"       python scripts/ingest_data.py --symbol {args.symbol} --timeframe {args.timeframe} --start {args.start} --end {args.end}")
        return

    print(f"\nData available: {min_ts} -> {max_ts}")
    print(f"Running backtest: {args.symbol} {args.timeframe} | {start.date()} -> {end.date()}")
    print(f"Strategy: {strategy.strategy_id}")
    print(f"Initial balance: ${args.balance:,.2f}\n")

    engine = BacktestEngine(
        strategy=strategy,
        store=store,
        initial_balance=Decimal(str(args.balance)),
        commission_rate=Decimal(str(args.commission_bps)) / 10000,
        slippage_rate=Decimal(str(args.slippage_bps)) / 10000,
        verbose=bool(args.output),
        leverage=args.leverage,
        max_daily_drawdown_pct=Decimal(str(args.max_dd)),
    )

    result = await engine.run(args.symbol, args.timeframe, start, end, strategy_params,
                              htf_timeframe=args.htf_timeframe)
    report = result.to_report()

    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    m = report["metrics"]
    print(f"  Total bars:       {m['total_bars']}")
    print(f"  Total trades:     {m['total_trades']}")
    print(f"  Win rate:         {m['winrate']:.1f}%")
    print(f"  Total PnL:        ${m['total_pnl']:.2f} ({m['total_pnl_pct']:.2f}%)")
    print(f"  Max drawdown:     {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio:     {m['sharpe_ratio']:.3f}")
    print(f"  Avg trade PnL:    ${m['avg_trade_pnl']:.2f}")
    print(f"  Exposure:         {m['exposure_pct']:.1f}%")
    print("=" * 60)

    if args.output:
        report["initial_balance"] = args.balance
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {args.output}")
        print(f"Viewer: open scripts/backtest_viewer.html in your browser and load the JSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trading strategy backtest")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--start", default="2025-05-18")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--balance", type=float, default=500.0)
    parser.add_argument("--strategy", default="trend_following_v2",
                        choices=["ema_cross", "rsi_divergence", "hybrid_rsi_pivot",
                                 "mean_reversion", "trend_following", "trend_following_v2",
                                 "event_driven"],
                        help="Strategy to backtest")
    parser.add_argument("--commission-bps", type=float, default=7.5, dest="commission_bps")
    parser.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    parser.add_argument("--leverage", type=int, default=3,
                        help="Futures leverage (default: 3)")
    parser.add_argument("--max-dd", type=float, default=100.0, dest="max_dd",
                        help="Max daily drawdown %% kill switch (default: 100 = disabled for backtests)")
    parser.add_argument("--output", default=None, help="Save full JSON report to file")

    # EMA Cross params
    parser.add_argument("--ema-fast", type=int, default=20, dest="ema_fast")
    parser.add_argument("--ema-slow", type=int, default=50, dest="ema_slow")
    parser.add_argument("--sma-trend", type=int, default=200, dest="sma_trend")

    # RSI Divergence params
    parser.add_argument("--rsi-period", type=int, default=9, dest="rsi_period")
    parser.add_argument("--ema-period", type=int, default=14, dest="ema_period")
    parser.add_argument("--swing-window", type=int, default=5, dest="swing_window")
    parser.add_argument("--swing-separation", type=int, default=10, dest="swing_separation")
    parser.add_argument("--swing-lookback", type=int, default=100, dest="swing_lookback")
    parser.add_argument("--trigger-window", type=int, default=10, dest="trigger_window")
    parser.add_argument("--rsi-oversold", type=float, default=30.0, dest="rsi_oversold")
    parser.add_argument("--rsi-overbought", type=float, default=70.0, dest="rsi_overbought")
    parser.add_argument("--allow-short", action="store_true", default=True, dest="allow_short")
    parser.add_argument("--sl-buffer-pct", type=float, default=0.003, dest="sl_buffer_pct")
    parser.add_argument("--rr-ratio", type=float, default=1.5, dest="rr_ratio",
                        help="TP1 risk:reward ratio (optimizer default: 1.5–1.75)")
    parser.add_argument("--tp2-ratio", type=float, default=1.75, dest="tp2_ratio",
                        help="TP2 risk:reward ratio (optimizer FIXED default: 1.75)")
    parser.add_argument("--min-trend-coeff", type=float, default=0.5, dest="min_trend_coeff")
    parser.add_argument("--trend-ema-period", type=int, default=50, dest="trend_ema_period",
                        help="Slow EMA period for LTF slope filter (0 = disabled)")
    parser.add_argument("--trend-slope-bars", type=int, default=5, dest="trend_slope_bars",
                        help="Bars back to measure trend EMA slope")
    parser.add_argument("--entry-window", type=int, default=2, dest="entry_window",
                        help="Bars to wait for limit fill after EMA confirmation")
    parser.add_argument("--entry-cooldown-bars", type=int, default=5, dest="entry_cooldown_bars",
                        help="Min bars between consecutive entry signals (default: 5)")
    parser.add_argument("--max-concurrent", type=int, default=1, dest="max_concurrent",
                        help="Max simultaneous open positions (1 = classic single-trade mode)")
    parser.add_argument("--htf-timeframe", default="1h", dest="htf_timeframe",
                        help="Higher timeframe for macro trend filter (default: 1h)")

    # Cali technique filters
    parser.add_argument("--use-ema-cross-filter", action="store_true", default=False,
                        dest="use_ema_cross_filter",
                        help="Require EMA_fast > EMA_slow for LONG (EMA7/EMA25 by default)")
    parser.add_argument("--ema-fast-filter", type=int, default=7, dest="ema_fast_filter",
                        help="Fast EMA for cross filter (default: 7)")
    parser.add_argument("--ema-slow-filter", type=int, default=25, dest="ema_slow_filter",
                        help="Slow EMA for cross filter (default: 25)")
    parser.add_argument("--vol-confirm-enabled", action="store_true", default=False,
                        dest="vol_confirm_enabled",
                        help="Require volume >= 80%% of vol_ma before entry")
    parser.add_argument("--vol-confirm-period", type=int, default=20, dest="vol_confirm_period",
                        help="Rolling window for volume moving average (default: 20)")
    parser.add_argument("--sr-zone-enabled", action="store_true", default=False,
                        dest="sr_zone_enabled",
                        help="Require price near an HTF S/R zone (Pupupu zones)")
    parser.add_argument("--sr-zone-pct", type=float, default=0.005, dest="sr_zone_pct",
                        help="Proximity threshold for S/R zones as fraction of price (default: 0.005 = 0.5%%)")
    parser.add_argument("--tp1-close-pct", type=float, default=0.33, dest="tp1_close_pct",
                        help="Fraction of position closed at TP1 (Cali recommends 0.75)")

    # Hybrid RSI Pivot params
    parser.add_argument("--atr-period", type=int, default=14, dest="atr_period")
    parser.add_argument("--atr-sl-mult", type=float, default=1.5, dest="atr_sl_mult")
    parser.add_argument("--pivot-atr-proximity", type=float, default=1.5, dest="pivot_atr_proximity")
    parser.add_argument("--vol-avg-period", type=int, default=20, dest="vol_avg_period")
    parser.add_argument("--vol-min-ratio", type=float, default=0.5, dest="vol_min_ratio")
    parser.add_argument("--session-start-utc", type=int, default=8, dest="session_start_utc")
    parser.add_argument("--session-end-utc", type=int, default=21, dest="session_end_utc")
    parser.add_argument("--bars-per-day", type=int, default=288, dest="bars_per_day",
                        help="Bars per calendar day: 288 for 5m, 96 for 15m, 24 for 1h")

    # ── Mean Reversion params ──────────────────────────────────────────────────
    parser.add_argument("--adx-max", type=float, default=25.0, dest="adx_max",
                        help="ADX máximo para mean reversion (encima = tendencia, no operar)")
    parser.add_argument("--min-rr", type=float, default=1.5, dest="min_rr",
                        help="R:R mínimo para mean_reversion / trend_following / event_driven")

    # ── Trend Following params ─────────────────────────────────────────────────
    parser.add_argument("--adx-min", type=float, default=25.0, dest="adx_min",
                        help="ADX mínimo para trend following (tendencia activa)")
    parser.add_argument("--pullback-tolerance-atr", type=float, default=1.5,
                        dest="pullback_tolerance_atr",
                        help="Tolerancia del pullback a EMA rápida en ATRs")
    parser.add_argument("--ema-fast-tf", type=int, default=20, dest="ema_fast_tf",
                        help="EMA rápida para trend following (proxy pullback 4H)")
    parser.add_argument("--ema-slow-tf", type=int, default=50, dest="ema_slow_tf",
                        help="EMA lenta para trend following (proxy trend diario)")
    parser.add_argument("--slope-bars", type=int, default=5, dest="slope_bars",
                        help="Barras para medir pendiente de EMA lenta")

    # ── Trend Following: Layer 1 — Confidence Scoring ──────────────────────
    parser.add_argument("--tf-use-confidence", action="store_true", default=True,
                        dest="tf_use_confidence",
                        help="Enable confidence scoring (Layer 1)")
    parser.add_argument("--tf-no-confidence", action="store_false", dest="tf_use_confidence",
                        help="Disable confidence scoring")
    parser.add_argument("--tf-adx-strong", type=float, default=35.0, dest="tf_adx_strong",
                        help="ADX threshold for 'strong trend' bonus (default: 35)")
    parser.add_argument("--tf-tight-pb-atr", type=float, default=0.5, dest="tf_tight_pb_atr",
                        help="ATR multiplier for 'tight pullback' bonus (default: 0.5)")
    parser.add_argument("--tf-min-confidence", type=float, default=0.40, dest="tf_min_confidence",
                        help="Minimum confidence score to allow entry (default: 0.40)")

    # ── Trend Following: Layer 2 — Session-Aware Sizing ────────────────────
    parser.add_argument("--tf-use-session", action="store_true", default=True,
                        dest="tf_use_session",
                        help="Enable session-aware sizing (Layer 2)")
    parser.add_argument("--tf-no-session", action="store_false", dest="tf_use_session",
                        help="Disable session-aware sizing")
    parser.add_argument("--tf-us-start", type=int, default=14, dest="tf_us_start",
                        help="US session start hour UTC (default: 14)")
    parser.add_argument("--tf-us-end", type=int, default=21, dest="tf_us_end",
                        help="US session end hour UTC (default: 21)")
    parser.add_argument("--tf-sess-us", type=float, default=1.0, dest="tf_sess_us",
                        help="Size multiplier for US session (default: 1.0)")
    parser.add_argument("--tf-sess-eu", type=float, default=0.75, dest="tf_sess_eu",
                        help="Size multiplier for EU session (default: 0.75)")
    parser.add_argument("--tf-sess-other", type=float, default=0.50, dest="tf_sess_other",
                        help="Size multiplier for Asia/night (default: 0.50)")

    # ── Trend Following: Layer 3 — Streak Adjuster ─────────────────────────
    parser.add_argument("--tf-use-streak", action="store_true", default=True,
                        dest="tf_use_streak",
                        help="Enable anti-euphoria streak adjuster (Layer 3)")
    parser.add_argument("--tf-no-streak", action="store_false", dest="tf_use_streak",
                        help="Disable streak adjuster")
    parser.add_argument("--tf-streak-after", type=int, default=2, dest="tf_streak_after",
                        help="Reduce size after N consecutive wins (default: 2)")
    parser.add_argument("--tf-streak-mult", type=float, default=0.75, dest="tf_streak_mult",
                        help="Size multiplier after win streak (default: 0.75)")

    # ── Trend Following: Layer 4 — Patience Timer ──────────────────────────
    parser.add_argument("--tf-use-patience", action="store_true", default=False,
                        dest="tf_use_patience",
                        help="Enable soft SL patience timer (Layer 4)")
    parser.add_argument("--tf-no-patience", action="store_false", dest="tf_use_patience",
                        help="Disable patience timer")
    parser.add_argument("--tf-soft-sl-bars", type=int, default=0, dest="tf_soft_sl_bars",
                        help="Bars for soft SL (0 = SL triggers on wick, default: 0)")

    # ── Trend Following: ATR-based SL range ────────────────────────────────
    parser.add_argument("--sl-min-atr", type=float, default=1.5, dest="sl_min_atr",
                        help="Minimum SL distance in ATRs (default: 1.5)")
    parser.add_argument("--sl-max-atr", type=float, default=3.0, dest="sl_max_atr",
                        help="Maximum SL distance in ATRs (default: 3.0)")
    parser.add_argument("--sl-min-pct", type=float, default=0.015, dest="sl_min_pct",
                        help="Minimum SL distance as fraction of price, floor for gap risk (default: 0.015 = 1.5%%)")
    parser.add_argument("--no-reversal", action="store_true", default=False, dest="no_reversal",
                        help="Disable reversal swap (open opposite position on signal flip)")

    # ── Event Driven params ────────────────────────────────────────────────────
    parser.add_argument("--vol-multiplier", type=float, default=2.0, dest="vol_multiplier",
                        help="Múltiplo de volumen para detectar el evento")
    parser.add_argument("--event-lookback", type=int, default=20, dest="event_lookback",
                        help="Barras para definir el rango y vol promedio (event_driven)")
    parser.add_argument("--max-hold-bars", type=int, default=20, dest="max_hold_bars",
                        help="Barras máximas sin confirmación antes de cierre por tiempo")
    parser.add_argument("--tp-rr", type=float, default=2.5, dest="tp_rr",
                        help="Target en múltiplos de R para event_driven")

    args = parser.parse_args()
    asyncio.run(main(args))