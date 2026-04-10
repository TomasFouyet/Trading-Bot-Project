"""
Telegram trade notifier.

Setup (one time):
  1. Open Telegram → search @BotFather → /newbot → copy the token
  2. Send any message to your new bot
  3. Run: python -m app.notify.telegram --setup
     (this will print your chat_id)
  4. Add to .env:
       TELEGRAM_BOT_TOKEN=123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TELEGRAM_CHAT_ID=987654321

Usage in code:
    from app.notify.telegram import TelegramNotifier
    notifier = TelegramNotifier.from_env()
    await notifier.trade_opened(...)
    await notifier.tp1_hit(...)
    await notifier.trade_closed(...)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import httpx


@dataclass
class TelegramNotifier:
    token: str
    chat_id: str
    _client: httpx.AsyncClient | None = None

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "TelegramNotifier | None":
        """
        Returns a notifier if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set,
        otherwise None (so the rest of the code can just do `if notifier: ...`).
        Reads from pydantic Settings (which loads .env automatically).
        """
        try:
            from app.config import get_settings
            s = get_settings()
            token   = s.telegram_bot_token.strip()
            chat_id = s.telegram_chat_id.strip()
        except Exception:
            # Fallback to raw os.getenv (e.g. when running outside the app)
            token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

        if not token or not chat_id:
            return None
        return cls(token=token, chat_id=chat_id)

    # ── Low-level send ────────────────────────────────────────────────────────

    async def send(self, text: str) -> bool:
        """Send a message. Returns True on success, False on failure (never raises)."""
        url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, json=data)
                return r.status_code == 200
        except Exception:
            return False

    # ── Trade event helpers ───────────────────────────────────────────────────

    async def trade_opened(
        self,
        symbol: str,
        side: str,           # "LONG" or "SHORT"
        tier: str,           # "A", "B", "C"
        leverage: int,
        entry_price: float,
        qty: float,
        stop_loss: float | None,
        tp1: float | None,
        tp2: float | None,
        session: str = "",
        reason: str = "",
        equity: float = 0,
    ) -> bool:
        direction_emoji = "🟢" if side == "LONG" else "🔴"
        sl_pct  = abs(entry_price - stop_loss) / entry_price * 100 if stop_loss else 0
        tp1_pct = abs(tp1 - entry_price) / entry_price * 100 if tp1 else 0
        tp2_pct = abs(tp2 - entry_price) / entry_price * 100 if tp2 else 0
        notional = entry_price * qty
        sl_str  = f"${stop_loss:,.2f}  (-{sl_pct:.1f}%)" if stop_loss else "—"
        tp1_str = f"${tp1:,.2f}  (+{tp1_pct:.1f}%)"      if tp1       else "—"
        tp2_str = f"${tp2:,.2f}  (+{tp2_pct:.1f}%)"      if tp2       else "—"

        msg = (
            f"{direction_emoji} <b>TRADE OPENED</b>\n"
            f"\n"
            f"📊 <b>{symbol}</b>  {side}  Tier {tier}  {leverage}x\n"
            f"\n"
            f"💵 Entry:      <code>${entry_price:,.2f}</code>\n"
            f"📦 Qty:        <code>{qty:.4f}  (${notional:,.2f} notional)</code>\n"
            f"\n"
            f"🛡 Stop Loss:  <code>{sl_str}</code>\n"
            f"🎯 TP1:        <code>{tp1_str}</code>\n"
            f"🏆 TP2:        <code>{tp2_str}</code>\n"
        )
        if session:
            msg += f"🕐 Session:    {session}\n"
        if reason:
            msg += f"💡 Signal:     {reason[:60]}\n"
        if equity:
            msg += f"💰 Equity:     <code>${equity:,.2f}</code>\n"
        msg += f"\n⏰ {_now_str()}"
        return await self.send(msg)

    async def tp1_hit(
        self,
        symbol: str,
        exit_price: float,
        close_pct: float,
        partial_pnl: float,
        remaining_qty: float,
        entry_price: float,
    ) -> bool:
        pnl_emoji = "🟡" if partial_pnl >= 0 else "🟠"
        pnl_sign  = "+" if partial_pnl >= 0 else ""
        pnl_pct   = abs(exit_price - entry_price) / entry_price * 100

        msg = (
            f"{pnl_emoji} <b>TP1 HIT — {symbol}</b>\n"
            f"\n"
            f"💵 Exit:         <code>${exit_price:,.2f}</code>  ({pnl_pct:.1f}%)\n"
            f"📦 Closed:       {close_pct*100:.0f}% of position\n"
            f"💰 Partial PnL:  <code>{pnl_sign}${partial_pnl:,.2f}</code>\n"
            f"📌 Remaining:    {remaining_qty:.4f} units\n"
            f"✅ SL moved to breakeven\n"
            f"\n⏰ {_now_str()}"
        )
        return await self.send(msg)

    async def trade_closed(
        self,
        symbol: str,
        side: str,
        exit_type: str,       # "tp2", "sl", "be_sl", "signal_close", "shutdown"
        exit_price: float,
        entry_price: float,
        total_pnl: float,
        qty: float,
        total_trades: int = 0,
        wins: int = 0,
        losses: int = 0,
    ) -> bool:
        # Emoji & label by exit type
        labels = {
            "tp2":           ("🏆", "TP2 HIT"),
            "sl":            ("🔴", "STOP LOSS HIT"),
            "be_sl":         ("🟡", "BREAKEVEN STOP"),
            "trailing_sl":   ("🔶", "TRAILING STOP HIT"),
            "reversal_swap": ("🔄", "REVERSAL SWAP"),
            "signal_close":  ("⚪", "SIGNAL CLOSE"),
            "shutdown":      ("⛔", "SHUTDOWN CLOSE"),
        }
        emoji, label = labels.get(exit_type, ("⚪", exit_type.upper()))

        pnl_pct  = total_pnl / (entry_price * qty) * 100 if (entry_price * qty) else 0
        pnl_sign = "+" if total_pnl >= 0 else ""
        wr_str   = f"{wins/total_trades*100:.0f}%" if total_trades else "—"

        msg = (
            f"{emoji} <b>{label}</b>\n"
            f"📊 {symbol}  {side}\n"
            f"\n"
            f"💵 Exit:      <code>${exit_price:,.2f}</code>\n"
            f"📥 Entry:     <code>${entry_price:,.2f}</code>\n"
            f"💰 Net PnL:   <code>{pnl_sign}${total_pnl:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)</code>\n"
        )
        if total_trades:
            msg += (
                f"\n📈 Session:   {total_trades} trades  "
                f"W={wins}  L={losses}  WR={wr_str}\n"
            )
        msg += f"\n⏰ {_now_str()}"
        return await self.send(msg)

    async def daily_summary(
        self,
        equity: float,
        starting_equity: float,
        total_trades: int,
        wins: int,
        losses: int,
        top_pnl: float,
        worst_pnl: float,
    ) -> bool:
        net     = equity - starting_equity
        net_pct = net / starting_equity * 100 if starting_equity else 0
        wr      = wins / total_trades * 100 if total_trades else 0
        emoji   = "📈" if net >= 0 else "📉"

        msg = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"\n"
            f"💰 Equity:   <code>${equity:,.2f}  ({'+' if net>=0 else ''}{net_pct:.2f}%)</code>\n"
            f"📊 Trades:   {total_trades}  (W={wins}  L={losses}  WR={wr:.0f}%)\n"
            f"🏆 Best:     <code>+${top_pnl:,.2f}</code>\n"
            f"🔴 Worst:    <code>${worst_pnl:,.2f}</code>\n"
            f"\n⏰ {_now_str()}"
        )
        return await self.send(msg)

    async def error_alert(self, symbol: str, message: str) -> bool:
        msg = (
            f"⚠️ <b>BOT ALERT — {symbol}</b>\n"
            f"\n{message[:300]}\n"
            f"\n⏰ {_now_str()}"
        )
        return await self.send(msg)

    async def bot_started(self, symbols: list[str], mode: str, balance: float) -> bool:
        sym_str = ", ".join(symbols)
        msg = (
            f"🤖 <b>BOT STARTED</b>\n"
            f"\n"
            f"📊 Symbols:  {sym_str}\n"
            f"⚙️ Mode:     {mode}\n"
            f"💰 Balance:  <code>${balance:,.2f}</code>\n"
            f"\n⏰ {_now_str()}"
        )
        return await self.send(msg)

    async def poll_commands(self, offset: int = 0) -> tuple[list[dict], int]:
        """
        Poll for new Telegram updates (messages/commands).
        Returns (list of messages, next_offset).
        """
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, params={"offset": offset, "timeout": 1})
            data = r.json()
            if not data.get("ok"):
                return [], offset
            updates = data.get("result", [])
            messages = []
            next_offset = offset
            for u in updates:
                next_offset = u["update_id"] + 1
                msg = u.get("message") or u.get("channel_post")
                if msg and msg.get("text"):
                    messages.append({"text": msg["text"], "chat_id": str(msg["chat"]["id"])})
            return messages, next_offset
        except Exception:
            return [], offset

    async def bot_stopped(self, total_trades: int, net_pnl: float, wins: int, losses: int) -> bool:
        wr    = wins / total_trades * 100 if total_trades else 0
        emoji = "🟢" if net_pnl >= 0 else "🔴"
        msg = (
            f"{emoji} <b>BOT STOPPED</b>\n"
            f"\n"
            f"📊 Trades:   {total_trades}  (W={wins}  L={losses}  WR={wr:.0f}%)\n"
            f"💰 Session PnL: <code>{'+' if net_pnl>=0 else ''}${net_pnl:,.2f}</code>\n"
            f"\n⏰ {_now_str()}"
        )
        return await self.send(msg)


# ── Util ──────────────────────────────────────────────────────────────────────

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── CLI setup helper ──────────────────────────────────────────────────────────

async def _cli_setup(token: str) -> None:
    """Fetch updates to find the chat_id after you send a message to the bot."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
    data = r.json()
    if not data.get("ok"):
        print(f"Error: {data}")
        return
    updates = data.get("result", [])
    if not updates:
        print("No messages found. Send any message to your bot first, then re-run.")
        return
    for u in updates:
        msg = u.get("message") or u.get("channel_post", {})
        chat = msg.get("chat", {})
        print(f"chat_id = {chat.get('id')}  ({chat.get('type')}  {chat.get('title') or chat.get('username') or chat.get('first_name')})")

    print("\nAdd to .env:")
    print(f"  TELEGRAM_BOT_TOKEN={token}")
    print(f"  TELEGRAM_CHAT_ID=<chat_id from above>")


if __name__ == "__main__":
    import argparse
    import asyncio

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    setup_p = sub.add_parser("setup", help="Find your chat_id")
    setup_p.add_argument("--token", required=True, help="Bot token from @BotFather")

    test_p = sub.add_parser("test", help="Send a test trade notification")

    args = p.parse_args()

    if args.cmd == "setup":
        asyncio.run(_cli_setup(args.token))

    elif args.cmd == "test":
        n = TelegramNotifier.from_env()
        if not n:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")
        else:
            async def _test():
                print("Sending test trade_opened...")
                ok = await n.trade_opened(
                    symbol="BTC-USDT", side="LONG", tier="A", leverage=5,
                    entry_price=83245.00, qty=0.1250,
                    stop_loss=81890.00, tp1=85280.00, tp2=88620.00,
                    session="US", reason="trend_long|T=A|c=0.78", equity=10500.0,
                )
                print(f"  trade_opened: {'OK' if ok else 'FAILED'}")

                print("Sending test tp1_hit...")
                ok = await n.tp1_hit(
                    symbol="BTC-USDT", exit_price=85280.00, close_pct=0.33,
                    partial_pnl=253.75, remaining_qty=0.0838, entry_price=83245.00,
                )
                print(f"  tp1_hit: {'OK' if ok else 'FAILED'}")

                print("Sending test trade_closed (TP2)...")
                ok = await n.trade_closed(
                    symbol="BTC-USDT", side="LONG", exit_type="tp2",
                    exit_price=88620.00, entry_price=83245.00,
                    total_pnl=687.50, qty=0.1250,
                    total_trades=7, wins=5, losses=2,
                )
                print(f"  trade_closed: {'OK' if ok else 'FAILED'}")

            asyncio.run(_test())

    else:
        p.print_help()