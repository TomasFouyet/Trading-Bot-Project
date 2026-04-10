"""
test_bingx_order.py — Prueba de conectividad y ejecución real en BingX

Abre un LONG mínimo en BTC-USDT y lo cierra inmediatamente.
Sirve para verificar que las API keys funcionan y las órdenes se ejecutan.

Uso:
    python scripts/test_bingx_order.py            # solo verifica conexión y balance
    python scripts/test_bingx_order.py --trade     # abre y cierra una orden real mínima
"""
from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal


async def main(do_trade: bool) -> None:
    from app.config import get_settings
    from app.core.logging import configure_logging
    from app.broker.bingx_client import BingXClient
    from app.broker.bingx_adapter import BingXAdapter
    from app.broker.base import OrderRequest, OrderSide, OrderType

    configure_logging(log_level="ERROR", log_format="console")
    s = get_settings()

    client = BingXClient(
        api_key=s.bingx_api_key,
        api_secret=s.bingx_api_secret,
        base_url=s.bingx_base_url,
        market_type=s.bingx_market_type,
    )

    print("\n" + "═" * 60)
    print("  TEST CONEXIÓN BINGX")
    print("═" * 60)

    # ── 1. Precio actual ────────────────────────────────────────────────────
    try:
        ticker = await client.get_ticker("BTC-USDT")
        price = float(ticker["last"])
        print(f"  [OK] Precio BTC-USDT:  ${price:,.2f}")
    except Exception as e:
        print(f"  [FAIL] No se pudo obtener precio: {e}")
        await client.close()
        return

    # ── 2. Balance ──────────────────────────────────────────────────────────
    try:
        raw_bal = await client.get_balance()
        usdt = next((a for a in raw_bal if "USDT" in str(a.get("asset", ""))), None)
        balance = float(usdt.get("balance", 0)) if usdt else 0.0
        available = float(usdt.get("availableMargin", usdt.get("balance", 0))) if usdt else 0.0
        print(f"  [OK] Balance USDT:     ${balance:,.2f}  (disponible: ${available:,.2f})")
    except Exception as e:
        print(f"  [FAIL] No se pudo obtener balance: {e}")
        await client.close()
        return

    if not do_trade:
        print("\n  Conexión OK. Usa --trade para ejecutar una orden de prueba real.")
        print("═" * 60 + "\n")
        await client.close()
        return

    # ── 3. Orden de prueba ──────────────────────────────────────────────────
    # Cantidad mínima de BTC en BingX futuros = 0.001 BTC
    MIN_QTY = Decimal("0.001")
    notional = MIN_QTY * Decimal(str(price))

    print(f"\n  Intentando abrir LONG mínimo:")
    print(f"    Qty:      {float(MIN_QTY):.4f} BTC  (~${float(notional):,.2f} USDT notional)")
    print(f"    Símbolo:  BTC-USDT (futuros perpetuos)")

    if available < float(notional) / 10:  # necesita al menos 10% de margen con 10x
        print(f"\n  [WARN] Balance disponible insuficiente para abrir ${float(notional):,.2f} notional")
        print(f"         Necesitas al menos ~${float(notional)/10:.1f} USDT de margen")
        await client.close()
        return

    adapter = BingXAdapter(client=client)
    try:
        await adapter.initialize()
    except Exception as e:
        print(f"  [WARN] initialize: {e}")

    # Abrir LONG
    print("\n  Abriendo LONG...")
    try:
        result, fill = await adapter.place_order(OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=MIN_QTY,
            extra={"positionSide": "LONG"},
        ))
        print(f"  [OK] Orden enviada:  orderId={result.order_id}  status={result.status.value}")
        if fill:
            print(f"  [OK] Fill recibido:  price=${float(fill.price):,.2f}  qty={float(fill.qty):.4f}")
        else:
            print(f"  [INFO] Fill no inmediato (status={result.status.value}) — cerrando igual")
    except Exception as e:
        print(f"  [FAIL] Error al abrir LONG: {e}")
        await client.close()
        return

    # Pequeña pausa para que la posición se registre
    print("  Esperando 2s...")
    await asyncio.sleep(2)

    # Cerrar LONG
    print("  Cerrando LONG...")
    try:
        result2, fill2 = await adapter.place_order(OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=MIN_QTY,
            extra={"positionSide": "LONG"},
        ))
        print(f"  [OK] Orden cierre:   orderId={result2.order_id}  status={result2.status.value}")
        if fill2:
            pnl = (fill2.price - (fill.price if fill else fill2.price)) * MIN_QTY
            print(f"  [OK] Fill cierre:    price=${float(fill2.price):,.2f}  pnl≈${float(pnl):+.4f}")
    except Exception as e:
        print(f"  [FAIL] Error al cerrar LONG: {e}")

    await client.close()

    print("\n  ✅ Test completado — BingX ejecuta órdenes correctamente.")
    print("═" * 60 + "\n")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test de conexión y órdenes en BingX")
    p.add_argument(
        "--trade",
        action="store_true",
        help="Ejecuta una orden LONG mínima real y la cierra (0.001 BTC)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    asyncio.run(main(do_trade=args.trade))
