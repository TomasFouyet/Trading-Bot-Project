// ══════════════════════════════════════════════════════════════════════════════
// TrendBot MTF v5.1 — STRATEGY VERSION (con panel de rendimiento)
// ══════════════════════════════════════════════════════════════════════════════
// CAMBIO PRINCIPAL: indicator() → strategy()
//   Trading View muestra automáticamente:
//     ✓ Equity curve (gráfico de rentabilidad)
//     ✓ Lista de trades con entrada/salida/PnL
//     ✓ Net Profit, Win Rate, Profit Factor, Max Drawdown
//     ✓ Sharpe Ratio, Sortino Ratio
//     ✓ Average trade, Best/Worst trade
//
// Para ver las métricas: click en "Strategy Tester" (pestaña abajo del gráfico)
//
// NOTAS:
//   - Usa strategy.entry/exit/close en vez de labels manuales
//   - TP1 parcial: strategy.exit con qty_percent
//   - Reversal swap: strategy.close + strategy.entry en la misma barra
//   - Comisión y slippage configurables en los inputs
// ══════════════════════════════════════════════════════════════════════════════
//@version=6
strategy("TrendBot MTF v5.1 [Strategy]",
         shorttitle="TBot5 Strat",
         overlay=true,
         // ── Configuración del backtester ──
         initial_capital=200,
         default_qty_type=strategy.percent_of_equity,
         default_qty_value=10,
         commission_type=strategy.commission.percent,
         commission_value=0.075,           // 7.5 bps taker
         slippage=2,                       // 2 ticks slippage
         process_orders_on_close=false,    // ordenes se ejecutan en apertura siguiente
         calc_on_every_tick=false,
         pyramiding=0,                     // 1 posición a la vez
         max_labels_count=100,
         max_lines_count=20)

// ─────────────────────────────────────────────────────────────────────────────
// INPUTS
// ─────────────────────────────────────────────────────────────────────────────
var g_core = "═══ Estrategia 15M ═══"
adx_min      = input.int(20,    "ADX mínimo",             group=g_core, minval=10, maxval=50)
adx_strong   = input.int(35,    "ADX fuerte (★)",          group=g_core, minval=20, maxval=60)
ema_fast     = input.int(20,    "EMA rápida (15M)",        group=g_core, minval=5,  maxval=50)
ema_slow     = input.int(50,    "EMA lenta (15M)",         group=g_core, minval=20, maxval=200)
pb_tol_atr   = input.float(1.0, "Pullback tolerancia ATR", group=g_core, minval=0.2, maxval=3.0, step=0.1)
min_conf     = input.float(0.0, "Confianza mínima",        group=g_core, minval=0.0, maxval=1.0, step=0.05)
allow_short  = input.bool(true,  "Permitir SHORTS",        group=g_core)
sig_cooldown = input.int(5,     "Cooldown entre señales (barras)", group=g_core, minval=1, maxval=50)

var g_htf = "═══ Bias 4H ═══"
htf_tf       = input.timeframe("240", "Timeframe HTF",    group=g_htf)
htf_ema_f    = input.int(50,   "HTF EMA rápida",          group=g_htf, minval=10, maxval=100)
htf_ema_s    = input.int(200,  "HTF EMA lenta",            group=g_htf, minval=50, maxval=500)
show_htf_bg  = input.bool(true,  "Fondo bias 4H",         group=g_htf)
show_htf_ema = input.bool(true,  "Mostrar EMAs 4H",       group=g_htf)

var g_ltf = "═══ Entrada 5M ═══"
ltf_tf       = input.timeframe("5", "Timeframe LTF",      group=g_ltf)
show_ltf_ema = input.bool(true,  "Mostrar EMA9 (5M)",     group=g_ltf)

var g_sl = "═══ SL / TP ═══"
sl_swing_lookback = input.int(50, "SL swing lookback (barras)", group=g_sl, minval=10, maxval=200)
sl_swing_window   = input.int(3,  "SL swing window (barras c/lado)", group=g_sl, minval=1, maxval=10)
sl_min_atr   = input.float(1.0, "SL mínimo (ATR)",        group=g_sl, step=0.1)
sl_max_atr   = input.float(2.5, "SL máximo (ATR)",        group=g_sl, step=0.1)
sl_buf_atr   = input.float(0.3, "SL buffer (ATR)",        group=g_sl, step=0.1)
tp1_r_A      = input.float(1.5, "TP1 R (Tier A)",         group=g_sl, step=0.1)
tp2_r_A      = input.float(3.0, "TP2 R (Tier A)",         group=g_sl, step=0.1)
tp1_r_B      = input.float(1.5, "TP1 R (Tier B)",         group=g_sl, step=0.1)
tp2_r_B      = input.float(2.5, "TP2 R (Tier B)",         group=g_sl, step=0.1)
tp1_r_C      = input.float(1.0, "TP1 R (Tier C)",         group=g_sl, step=0.1)
tp2_r_C      = input.float(1.5, "TP2 R (Tier C)",         group=g_sl, step=0.1)

var g_rev = "═══ Reversal Swap ═══"
enable_reversal = input.bool(true, "Habilitar swap de reversal", group=g_rev)

var g_sizing = "═══ Position Sizing ═══"
use_tier_sizing = input.bool(true,  "Usar sizing por tier",  group=g_sizing)
tier_a_pct      = input.float(100,  "Tier A: % equity",     group=g_sizing, minval=10, maxval=200)
tier_b_pct      = input.float(75,   "Tier B: % equity",     group=g_sizing, minval=10, maxval=200)
tier_c_pct      = input.float(25,   "Tier C: % equity",     group=g_sizing, minval=10, maxval=200)

var g_vis = "═══ Visual ═══"
show_trade_zones = input.bool(true, "Zonas SL/TP del trade",  group=g_vis)
show_trade_bg    = input.bool(true, "Fondo color durante trade", group=g_vis)
show_session     = input.bool(true, "Fondo de sesión",         group=g_vis)
show_htf_bg_inp  = input.bool(true, "Fondo bias 4H",          group=g_vis)
show_labels      = input.bool(true, "Labels de entrada/salida", group=g_vis)

// ─────────────────────────────────────────────────────────────────────────────
// COLORES
// ─────────────────────────────────────────────────────────────────────────────
C_BULL     = color.new(#1D9E75, 0)
C_BEAR     = color.new(#E24B4A, 0)
C_NEUT     = color.new(#888780, 0)
C_AMBER    = color.new(#EF9F27, 0)
C_BLUE     = color.new(#3B8BD4, 0)
C_PURPLE   = color.new(#7F77DD, 0)
C_BG_BULL  = color.new(#1D9E75, 92)
C_BG_BEAR  = color.new(#E24B4A, 92)
C_BG_US    = color.new(#3B8BD4, 95)
C_BG_EU    = color.new(#EF9F27, 95)

// ─────────────────────────────────────────────────────────────────────────────
// 15M INDICATORS
// ─────────────────────────────────────────────────────────────────────────────
atr14                          = ta.atr(14)
[diplus, diminus, adx_val]     = ta.dmi(14, 14)
ema_f_line                     = ta.ema(close, ema_fast)
ema_s_line                     = ta.ema(close, ema_slow)
ema_s_slope                    = ema_s_line - ema_s_line[5]
ema_f_slope                    = ema_f_line - ema_f_line[5]
[macd_line, macd_sig_line, macd_hist] = ta.macd(close, 12, 26, 9)
vol_sma                        = ta.sma(volume, 20)
vol_ratio                      = vol_sma > 0 ? volume / vol_sma : 0.0

// ─────────────────────────────────────────────────────────────────────────────
// 4H HTF DATA
// ─────────────────────────────────────────────────────────────────────────────
f_htf_adx() =>
    [_p, _m, _adx] = ta.dmi(14, 14)
    _adx

htf_close = request.security(syminfo.tickerid, htf_tf, close,                    lookahead=barmerge.lookahead_off)
htf_ef    = request.security(syminfo.tickerid, htf_tf, ta.ema(close, htf_ema_f), lookahead=barmerge.lookahead_off)
htf_es    = request.security(syminfo.tickerid, htf_tf, ta.ema(close, htf_ema_s), lookahead=barmerge.lookahead_off)
htf_adx   = request.security(syminfo.tickerid, htf_tf, f_htf_adx(),              lookahead=barmerge.lookahead_off)
htf_c1    = request.security(syminfo.tickerid, htf_tf, close[1],                 lookahead=barmerge.lookahead_off)
htf_c2    = request.security(syminfo.tickerid, htf_tf, close[2],                 lookahead=barmerge.lookahead_off)

htf_score   = 0.0
htf_score  := htf_ef > htf_es ? htf_score + 0.35 : htf_ef < htf_es ? htf_score - 0.35 : htf_score
htf_score  := htf_close > htf_ef and htf_ef > htf_es ? htf_score + 0.25 :
              htf_close < htf_ef and htf_ef < htf_es ? htf_score - 0.25 :
              htf_close > htf_es                     ? htf_score + 0.10 :
              htf_close < htf_es                     ? htf_score - 0.10 : htf_score
htf_adx_bonus = htf_adx >= 30 ? 0.20 : htf_adx >= 20 ? 0.10 : 0.0
htf_score  := htf_score > 0 ? htf_score + htf_adx_bonus :
              htf_score < 0 ? htf_score - htf_adx_bonus : htf_score
htf_rising  = htf_c2 <= htf_c1 and htf_c1 <= htf_close
htf_falling = htf_c2 >= htf_c1 and htf_c1 >= htf_close
htf_score  := htf_rising ? htf_score + 0.10 : htf_falling ? htf_score - 0.10 : htf_score

htf_score_c  = math.max(-1.0, math.min(1.0, htf_score))
htf_bias     = htf_score_c >= 0.35 ? 1 : htf_score_c <= -0.35 ? -1 : 0
htf_strength = math.abs(htf_score_c)

htf_conf_mod(dir) =>
    if htf_bias == 0
        0.0
    else
        _aligned = (dir == "LONG" and htf_bias == 1) or (dir == "SHORT" and htf_bias == -1)
        if _aligned
            htf_strength >= 0.6 ? 0.15 : 0.05
        else
            htf_strength >= 0.6 ? -0.35 : -0.20

// ─────────────────────────────────────────────────────────────────────────────
// 5M LTF DATA
// ─────────────────────────────────────────────────────────────────────────────
ltf_ema9        = request.security(syminfo.tickerid, ltf_tf, ta.ema(close, 9),  lookahead=barmerge.lookahead_off)
ltf_ema21       = request.security(syminfo.tickerid, ltf_tf, ta.ema(close, 21), lookahead=barmerge.lookahead_off)
ltf_close       = request.security(syminfo.tickerid, ltf_tf, close,             lookahead=barmerge.lookahead_off)
ltf_open        = request.security(syminfo.tickerid, ltf_tf, open,              lookahead=barmerge.lookahead_off)
ltf_is_green    = ltf_close > ltf_open
ltf_pb_dist_pct = ltf_ema9 > 0 ? math.abs(close - ltf_ema9) / close * 100 : 999.0
ltf_at_ema9     = ltf_pb_dist_pct < 0.15

// ─────────────────────────────────────────────────────────────────────────────
// CONFIDENCE SCORING
// ─────────────────────────────────────────────────────────────────────────────
compute_conf(dir) =>
    sc = 0.0
    sc := adx_val >= adx_strong ? sc + 0.20 : adx_val >= 30 ? sc + 0.10 : sc
    pb_atr_d = atr14 > 0 ? math.abs(close - ema_f_line) / atr14 : 999.0
    sc := pb_atr_d <= 0.5 ? sc + 0.20 : pb_atr_d <= 1.0 ? sc + 0.10 : sc
    if dir == "LONG" and macd_hist > 0 and macd_hist > macd_hist[1]
        sc := sc + 0.15
    else if dir == "SHORT" and macd_hist < 0 and macd_hist < macd_hist[1]
        sc := sc + 0.15
    else if (dir == "LONG" and macd_hist > 0) or (dir == "SHORT" and macd_hist < 0)
        sc := sc + 0.05
    _body  = math.abs(close - open)
    _rng   = high - low
    _ratio = _rng > 0 ? _body / _rng : 0.0
    sc := _ratio >= 0.60 ? sc + 0.15 : _ratio >= 0.40 ? sc + 0.07 : sc
    if dir == "LONG" and ema_f_slope > 0
        sc := sc + 0.15
    else if dir == "SHORT" and ema_f_slope < 0
        sc := sc + 0.15
    sc := vol_ratio >= 1.2 ? sc + 0.15 : vol_ratio >= 0.8 ? sc + 0.05 : sc
    math.min(sc, 1.0)

// ─────────────────────────────────────────────────────────────────────────────
// SETUP CONDITIONS
// ─────────────────────────────────────────────────────────────────────────────
pb_zone    = math.abs(close - ema_f_line) < atr14 * pb_tol_atr
sl_rising  = ema_s_slope > 0
sl_falling = ema_s_slope < 0
p_above    = close > ema_s_line
p_below    = close < ema_s_line
m_bull     = macd_line > macd_sig_line
m_bear     = macd_line < macd_sig_line
c_bull_bar = close > open
c_bear_bar = close < open
adx_ok     = adx_val >= adx_min

long_base  = adx_ok and sl_rising  and p_above and m_bull and pb_zone and c_bull_bar
short_base = adx_ok and sl_falling and p_below and m_bear and pb_zone and c_bear_bar and allow_short

raw_conf_l = long_base  ? compute_conf("LONG")  : 0.0
raw_conf_s = short_base ? compute_conf("SHORT") : 0.0
conf_long  = math.max(0.0, math.min(1.0, raw_conf_l + htf_conf_mod("LONG")))
conf_short = math.max(0.0, math.min(1.0, raw_conf_s + htf_conf_mod("SHORT")))

long_signal  = long_base  and conf_long  >= min_conf
short_signal = short_base and conf_short >= min_conf

get_tier(c) => c >= 0.65 ? "A" : c >= 0.40 ? "B" : "C"
get_size_mult(c) => c >= 0.65 ? 2.0 : c >= 0.40 ? 1.5 : 0.5
get_tp1_close_pct(c) => c >= 0.65 ? 0.33 : c >= 0.40 ? 0.50 : 0.70

// Cooldown
var int last_long_bar  = -999
var int last_short_bar = -999
long_trigger_raw  = long_signal  and not long_signal[1]
short_trigger_raw = short_signal and not short_signal[1]
long_trigger  = long_trigger_raw  and (bar_index - last_long_bar)  >= sig_cooldown
short_trigger = short_trigger_raw and (bar_index - last_short_bar) >= sig_cooldown
if long_trigger
    last_long_bar  := bar_index
if short_trigger
    last_short_bar := bar_index

// Session
utc_h     = hour(time, "UTC")
is_us     = utc_h >= 14 and utc_h < 21
is_eu     = utc_h >= 8  and utc_h < 14

// ─────────────────────────────────────────────────────────────────────────────
// SL CALCULATION (identical to indicator version)
// ─────────────────────────────────────────────────────────────────────────────
find_swing(dir, lookback, w) =>
    _found = false
    _level = 0.0
    _lb = math.min(lookback, bar_index)
    if _lb > 2 * w
        if dir == "LONG"
            for i = w to _lb - w
                _is_pivot = true
                for j = 1 to w
                    if i - j < 0
                        _is_pivot := false
                        break
                    if low[i] > low[i - j] or low[i] > low[i + j]
                        _is_pivot := false
                        break
                if _is_pivot
                    _level := low[i]
                    _found := true
                    break
        else
            for i = w to _lb - w
                _is_pivot = true
                for j = 1 to w
                    if i - j < 0
                        _is_pivot := false
                        break
                    if high[i] < high[i - j] or high[i] < high[i + j]
                        _is_pivot := false
                        break
                if _is_pivot
                    _level := high[i]
                    _found := true
                    break
    [_level, _found]

calc_sl(dir, conf) =>
    _lb = math.min(sl_swing_lookback, bar_index)
    [_swing, _swing_found] = find_swing(dir, sl_swing_lookback, sl_swing_window)
    _method = _swing_found ? "swing" : "lookback"
    if dir == "LONG"
        _fallback = ta.lowest(low, _lb > 0 ? _lb : 1)
        _base = _swing_found ? _swing : _fallback
        _struct_sl = math.min(_base, ema_s_line) - atr14 * sl_buf_atr
        _dist = close - _struct_sl
        _dist := math.max(_dist, atr14 * sl_min_atr)
        _dist := math.min(_dist, atr14 * sl_max_atr)
        _sl = close - _dist
        _risk = math.abs(close - _sl)
        _rr1 = conf >= 0.65 ? tp1_r_A : conf >= 0.40 ? tp1_r_B : tp1_r_C
        _rr2 = conf >= 0.65 ? tp2_r_A : conf >= 0.40 ? tp2_r_B : tp2_r_C
        _tp1 = close + _risk * _rr1
        _tp2 = close + _risk * _rr2
        [_sl, _tp1, _tp2, _method]
    else
        _fallback = ta.highest(high, _lb > 0 ? _lb : 1)
        _base = _swing_found ? _swing : _fallback
        _struct_sl = math.max(_base, ema_s_line) + atr14 * sl_buf_atr
        _dist = _struct_sl - close
        _dist := math.max(_dist, atr14 * sl_min_atr)
        _dist := math.min(_dist, atr14 * sl_max_atr)
        _sl = close + _dist
        _risk = math.abs(close - _sl)
        _rr1 = conf >= 0.65 ? tp1_r_A : conf >= 0.40 ? tp1_r_B : tp1_r_C
        _rr2 = conf >= 0.65 ? tp2_r_A : conf >= 0.40 ? tp2_r_B : tp2_r_C
        _tp1 = close - _risk * _rr1
        _tp2 = close - _risk * _rr2
        [_sl, _tp1, _tp2, _method]

[sl_l, tp1_l, tp2_l, sl_method_l] = calc_sl("LONG",  conf_long)
[sl_s, tp1_s, tp2_s, sl_method_s] = calc_sl("SHORT", conf_short)

// ═════════════════════════════════════════════════════════════════════════════
// TRADE STATE MACHINE — strategy version
// ═════════════════════════════════════════════════════════════════════════════
// We track state manually AND use strategy.* calls so TV's backtester
// records the trades for the performance panel.

var int   trade_state   = 0
var float trade_entry   = na
var float trade_sl      = na
var float trade_tp1     = na
var float trade_tp2     = na
var int   trade_start   = na
var bool  trade_tp1_hit = false
var string trade_tier   = "X"
var string trade_sl_method = ""

// Position sizing by tier
get_qty_pct(conf) =>
    if not use_tier_sizing
        100.0
    else
        _tier = get_tier(conf)
        _tier == "A" ? tier_a_pct : _tier == "B" ? tier_b_pct : tier_c_pct

// ─── Event detection ────────────────────────────────────────────────────────
reversal_to_long  = enable_reversal and trade_state == -1 and long_trigger
reversal_to_short = enable_reversal and trade_state == 1  and short_trigger and not trade_tp1_hit
is_reversal       = reversal_to_long or reversal_to_short
normal_long_entry  = trade_state == 0 and long_trigger
normal_short_entry = trade_state == 0 and short_trigger
new_long  = normal_long_entry  or reversal_to_long
new_short = normal_short_entry or reversal_to_short

// Exit detection (SL checked before TP — conservative)
sl_hit_long   = trade_state == 1  and not na(trade_sl) and low  <= trade_sl and not is_reversal
sl_hit_short  = trade_state == -1 and not na(trade_sl) and high >= trade_sl and not is_reversal
tp1_hit_long  = trade_state == 1  and not trade_tp1_hit and not na(trade_tp1) and high >= trade_tp1 and not sl_hit_long
tp1_hit_short = trade_state == -1 and not trade_tp1_hit and not na(trade_tp1) and low  <= trade_tp1 and not sl_hit_short
tp2_hit_long  = trade_state == 1  and trade_tp1_hit and not na(trade_tp2) and high >= trade_tp2 and not sl_hit_long
tp2_hit_short = trade_state == -1 and trade_tp1_hit and not na(trade_tp2) and low  <= trade_tp2 and not sl_hit_short

trade_sl_exit  = sl_hit_long or sl_hit_short
trade_tp2_exit = tp2_hit_long or tp2_hit_short
trade_exit     = (trade_sl_exit or trade_tp2_exit) and not is_reversal

// ─── Process TP1 hit → partial close + move SL to BE ────────────────────────
if tp1_hit_long and not trade_sl_exit and not is_reversal
    trade_tp1_hit := true
    _tp1_close_pct = get_tp1_close_pct(conf_long) * 100
    // Partial close at TP1
    strategy.close("Long", comment="TP1 ✓", qty_percent=_tp1_close_pct)
    trade_sl := trade_entry  // Move SL to breakeven

if tp1_hit_short and not trade_sl_exit and not is_reversal
    trade_tp1_hit := true
    _tp1_close_pct = get_tp1_close_pct(conf_short) * 100
    strategy.close("Short", comment="TP1 ✓", qty_percent=_tp1_close_pct)
    trade_sl := trade_entry

// ─── Process trade exit (SL or TP2) ─────────────────────────────────────────
if trade_exit
    _exit_type = trade_sl_exit ? "SL" : "TP2"
    if trade_state == 1
        strategy.close("Long", comment=_exit_type)
    else
        strategy.close("Short", comment=_exit_type)
    trade_state   := 0
    trade_entry   := na
    trade_sl      := na
    trade_tp1     := na
    trade_tp2     := na
    trade_tp1_hit := false

// ─── Process reversal swap ──────────────────────────────────────────────────
if is_reversal
    _old_dir = trade_state == 1 ? "L" : "S"
    _new_dir = reversal_to_long ? "L" : "S"
    // Close current position
    if trade_state == 1
        strategy.close("Long", comment="⟳ " + _old_dir + "→" + _new_dir)
    else
        strategy.close("Short", comment="⟳ " + _old_dir + "→" + _new_dir)
    // Reset — new entry opens below
    trade_state   := 0
    trade_entry   := na
    trade_sl      := na
    trade_tp1     := na
    trade_tp2     := na
    trade_tp1_hit := false

// ─── Open new trade ─────────────────────────────────────────────────────────
if new_long and trade_state == 0
    _qty_pct = get_qty_pct(conf_long)
    _tier    = get_tier(conf_long)
    _htf_tag = htf_bias == 1 ? "↑4H" : htf_bias == -1 ? "↓4H!" : "~4H"

    strategy.entry("Long", strategy.long,
                   qty=strategy.equity * (_qty_pct / 100) / close,
                   comment="▲ T" + _tier + " " + _htf_tag)

    trade_state     := 1
    trade_entry     := close
    trade_sl        := sl_l
    trade_tp1       := tp1_l
    trade_tp2       := tp2_l
    trade_tp1_hit   := false
    trade_start     := bar_index
    trade_tier      := _tier
    trade_sl_method := sl_method_l

if new_short and trade_state == 0
    _qty_pct = get_qty_pct(conf_short)
    _tier    = get_tier(conf_short)
    _htf_tag = htf_bias == -1 ? "↓4H" : htf_bias == 1 ? "↑4H!" : "~4H"

    strategy.entry("Short", strategy.short,
                   qty=strategy.equity * (_qty_pct / 100) / close,
                   comment="▼ T" + _tier + " " + _htf_tag)

    trade_state     := -1
    trade_entry     := close
    trade_sl        := sl_s
    trade_tp1       := tp1_s
    trade_tp2       := tp2_s
    trade_tp1_hit   := false
    trade_start     := bar_index
    trade_tier      := _tier
    trade_sl_method := sl_method_s

// ═════════════════════════════════════════════════════════════════════════════
// TRADE VISUAL — plot() + fill() (same as indicator version)
// ═════════════════════════════════════════════════════════════════════════════
plot_entry = trade_state != 0 ? trade_entry : na
plot_sl    = trade_state != 0 ? trade_sl    : na
plot_tp1   = trade_state != 0 and not trade_tp1_hit ? trade_tp1 : na
plot_tp2   = trade_state != 0 ? trade_tp2   : na

p_entry = plot(show_trade_zones ? plot_entry : na, "Entry",
               color=color.new(color.white, 40), linewidth=2, style=plot.style_linebr,
               display=display.pane)
p_sl    = plot(show_trade_zones ? plot_sl : na, "Stop loss",
               color=color.new(#E24B4A, 20), linewidth=1, style=plot.style_linebr,
               display=display.pane)
p_tp1   = plot(show_trade_zones ? plot_tp1 : na, "Take profit 1",
               color=color.new(#3B8BD4, 20), linewidth=1, style=plot.style_linebr,
               display=display.pane)
p_tp2   = plot(show_trade_zones ? plot_tp2 : na, "Take profit 2",
               color=color.new(#1D9E75, 20), linewidth=1, style=plot.style_linebr,
               display=display.pane)

fill(p_entry, p_sl,  color=show_trade_zones and trade_state != 0 ? color.new(#E24B4A, 85) : na, title="Risk zone")
fill(p_entry, p_tp1, color=show_trade_zones and trade_state != 0 and not trade_tp1_hit ? color.new(#3B8BD4, 88) : na, title="TP1 zone")
fill(p_tp1,   p_tp2, color=show_trade_zones and trade_state != 0 and not trade_tp1_hit ? color.new(#1D9E75, 90) : na, title="TP2 zone")
fill(p_entry, p_sl,  color=show_trade_zones and trade_state != 0 and trade_tp1_hit ? color.new(#EF9F27, 85) : na, title="BE zone")

trade_bg_col = trade_state == 1 ? color.new(#1D9E75, 95) :
               trade_state == -1 ? color.new(#E24B4A, 95) : na
bgcolor(show_trade_bg ? trade_bg_col : na, title="Trade background")

// ─────────────────────────────────────────────────────────────────────────────
// PLOTS — EMAs
// ─────────────────────────────────────────────────────────────────────────────
plot(ema_f_line, "EMA20",     color=color.new(C_AMBER, 0), linewidth=1)
plot(ema_s_line, "EMA50",     color=color.new(C_BLUE,  0), linewidth=2)

htf_vis = timeframe.in_seconds() <= 14400
plot(show_htf_ema and htf_vis ? htf_ef : na, "EMA50 4H",  color=color.new(C_BULL, 25), linewidth=2, style=plot.style_circles)
plot(show_htf_ema and htf_vis ? htf_es : na, "EMA200 4H", color=color.new(C_BEAR, 25), linewidth=2, style=plot.style_circles)

ltf_vis = timeframe.in_seconds() <= 900
plot(show_ltf_ema and ltf_vis ? ltf_ema9  : na, "EMA9 5M",  color=color.new(#D4537E, 45), linewidth=1, style=plot.style_stepline)
plot(show_ltf_ema and ltf_vis ? ltf_ema21 : na, "EMA21 5M", color=color.new(#D4537E, 70), linewidth=1, style=plot.style_stepline)

// ─────────────────────────────────────────────────────────────────────────────
// BACKGROUNDS
// ─────────────────────────────────────────────────────────────────────────────
bgcolor(show_session and is_us ? C_BG_US   : na, title="Sesión US")
bgcolor(show_session and is_eu ? C_BG_EU   : na, title="Sesión EU")
bgcolor(show_htf_bg_inp and htf_bias ==  1 ? C_BG_BULL : na, title="4H Bull")
bgcolor(show_htf_bg_inp and htf_bias == -1 ? C_BG_BEAR : na, title="4H Bear")

// ─────────────────────────────────────────────────────────────────────────────
// ALERTS (same as indicator)
// ─────────────────────────────────────────────────────────────────────────────
alertcondition(new_long and not is_reversal,   title="LONG setup",    message="{{ticker}} LONG — revisar gráfico")
alertcondition(new_short and not is_reversal,  title="SHORT setup",   message="{{ticker}} SHORT — revisar gráfico")
alertcondition(is_reversal,                    title="REVERSAL SWAP", message="{{ticker}} ⟳ SWAP — trend reversed")
alertcondition(tp1_hit_long or tp1_hit_short,  title="TP1 hit",       message="{{ticker}} TP1 hit — trailing to BE")
alertcondition(trade_exit,                     title="Trade exit",     message="{{ticker}} trade exited")
