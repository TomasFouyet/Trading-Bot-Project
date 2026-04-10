"""
Raw BingX HTTP client.

Handles:
- HMAC-SHA256 authentication
- Retry with exponential backoff + jitter
- Rate limit detection and sleep
- Response validation
- Symbol normalization

BingX API reference:
  Perpetual swap: https://bingx-api.github.io/docs/#/en-us/swapV2/
  Spot:           https://bingx-api.github.io/docs/#/en-us/spot/
"""
from __future__ import annotations

import hashlib
import hmac
import random
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from app.config import MarketType, get_settings
from app.core.exceptions import (
    AuthenticationError,
    BrokerError,
    OrderNotFoundError,
    RateLimitError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
_settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Symbol normalizer
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
}


def normalize_symbol(symbol: str) -> str:
    """
    BingX perpetual uses 'BTC-USDT', spot may use 'BTC-USDT' as well.
    This normalizer handles common inputs like 'BTCUSDT' -> 'BTC-USDT'.
    """
    s = symbol.upper().replace("/", "-").replace("_", "-")
    if "-" not in s:
        # Try to split BTCUSDT -> BTC-USDT (simple heuristic for USDT pairs)
        for quote in ("USDT", "USDC", "BTC", "ETH", "BNB"):
            if s.endswith(quote):
                base = s[: -len(quote)]
                s = f"{base}-{quote}"
                break
    return s


# ─────────────────────────────────────────────────────────────────────────────
# BingX client
# ─────────────────────────────────────────────────────────────────────────────

class BingXClient:
    """
    Low-level BingX REST client.

    Exposes clean methods:
      fetch_ohlcv, get_ticker, place_order, get_order,
      cancel_order, get_positions, get_balance

    Both spot and perpetual swap are supported via market_type config.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://open-api.bingx.com",
        market_type: MarketType = MarketType.SWAP,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._market_type = market_type
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"X-BX-APIKEY": self._api_key},
        )

    # ── Auth helpers ─────────────────────────────────────────────────────────

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def _build_query_string(self, params: dict[str, Any]) -> str:
        """
        Replica exacta de parseParam() del ejemplo oficial BingX:
        1. Ordena las claves alfabéticamente
        2. Construye key=value&key=value
        3. Agrega &timestamp=... al final
        La firma se calcula sobre esta cadena completa.
        """
        filtered = {k: v for k, v in params.items() if v is not None}
        sorted_str = "&".join(f"{k}={v}" for k, v in sorted(filtered.items()))
        ts = self._timestamp_ms()
        if sorted_str:
            return f"{sorted_str}&timestamp={ts}", ts
        return f"timestamp={ts}", ts

    def _sign(self, query_string: str) -> str:
        """HMAC-SHA256 sobre el query string ya construido (sin signature)."""
        return hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _build_signed_url(self, path: str, params: dict[str, Any]) -> str:
        """Construye la URL final: base + path + ?query_string&signature=..."""
        query_string, _ = self._build_query_string(params)
        signature = self._sign(query_string)
        return f"{self._base_url}{path}?{query_string}&signature={signature}"

    # ── Route helpers ─────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _kline_path(self) -> str:
        if self._market_type == MarketType.SWAP:
            return "/openApi/swap/v2/quote/klines"
        return "/openApi/spot/v1/market/kline"

    def _order_path(self) -> str:
        if self._market_type == MarketType.SWAP:
            return "/openApi/swap/v2/trade/order"
        return "/openApi/spot/v1/trade/order"

    def _positions_path(self) -> str:
        return "/openApi/swap/v2/user/positions"

    def _balance_path(self) -> str:
        if self._market_type == MarketType.SWAP:
            return "/openApi/swap/v2/user/balance"
        return "/openApi/spot/v1/account/balance"

    def _ticker_path(self) -> str:
        if self._market_type == MarketType.SWAP:
            return "/openApi/swap/v2/quote/ticker"
        return "/openApi/spot/v1/market/ticker"

    def _leverage_path(self) -> str:
        return "/openApi/swap/v2/trade/leverage"

    # ── HTTP core ─────────────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        signed: bool = False,
    ) -> Any:
        # Merge all parameters (query + body) into one dict for signing
        all_params = {**(params or {}), **(json or {})}

        if signed:
            # BingX: ALL params go in query string (signed URL), body is empty
            url = self._build_signed_url(path, all_params)
            req_params = None
        else:
            # Unsigned: normal query string via httpx params=
            url = self._url(path)
            req_params = all_params if all_params else None

        try:
            response = await self._client.request(method, url, params=req_params)
        except httpx.TimeoutException as e:
            raise BrokerError(f"Request timeout: {path}") from e
        except httpx.NetworkError as e:
            raise BrokerError(f"Network error: {path}") from e

        return self._validate_response(response)

    def _validate_response(self, response: httpx.Response) -> Any:
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            raise RateLimitError(f"Rate limit hit. Retry after {retry_after}s")

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key or signature")

        if response.status_code >= 500:
            raise BrokerError(f"BingX server error: {response.status_code}")

        try:
            data = response.json()
        except Exception as e:
            raise BrokerError(f"Non-JSON response: {response.text[:200]}") from e

        code = data.get("code", 0)
        if code != 0:
            msg = data.get("msg", "Unknown error")
            if code in (100500, 100404):
                raise OrderNotFoundError(msg)
            if code == 100401:
                raise AuthenticationError(msg)
            if code in (100429, 100410):
                raise RateLimitError(msg)
            raise BrokerError(f"BingX API error {code}: {msg}")

        return data.get("data", data)

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        signed: bool = False,
    ) -> Any:
        last_exc: Exception | None = None
        delays = [0.5, 1.0, 2.0, 4.0, 8.0]
        for attempt, delay in enumerate(delays, start=1):
            try:
                return await self._request(method, path, params=params, json=json, signed=signed)
            except RateLimitError as e:
                # "disabled period" = endpoint banned; retrying makes it worse — give up fast
                if "disabled period" in str(e):
                    logger.warning("endpoint_disabled", path=path, error=str(e)[:120])
                    raise
                sleep = delay + random.uniform(0, 0.5)
                logger.warning("rate_limit_backoff", attempt=attempt, sleep_s=sleep, path=path)
                time.sleep(sleep)
                last_exc = RateLimitError(f"Rate limited after {attempt} attempts")
            except BrokerError as e:
                if attempt == len(delays):
                    raise
                sleep = delay + random.uniform(0, 0.3)
                logger.warning("broker_retry", attempt=attempt, error=str(e), path=path)
                time.sleep(sleep)
                last_exc = e
        raise last_exc  # type: ignore[misc]

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Fetch OHLCV bars.
        Returns list of dicts: {ts, open, high, low, close, volume}
        """
        sym = normalize_symbol(symbol)
        tf = TIMEFRAME_MAP.get(timeframe, timeframe)

        params: dict[str, Any] = {
            "symbol": sym,
            "interval": tf,
            "limit": min(limit, 1440),
        }
        if start:
            params["startTime"] = int(start.timestamp() * 1000)
        if end:
            params["endTime"] = int(end.timestamp() * 1000)

        data = await self._request_with_retry("GET", self._kline_path(), params=params)

        bars = []
        # BingX returns a list; each element is [openTime, open, high, low, close, volume, closeTime, ...]
        # or a dict depending on endpoint version
        raw_list = data if isinstance(data, list) else data.get("data", [])
        for item in raw_list:
            if isinstance(item, list):
                ts_ms, o, h, l, c, v = item[0], item[1], item[2], item[3], item[4], item[5]
            elif isinstance(item, dict):
                ts_ms = item.get("time", item.get("openTime", item.get("t")))
                o = item.get("open", item.get("o"))
                h = item.get("high", item.get("h"))
                l = item.get("low", item.get("l"))
                c = item.get("close", item.get("c"))
                v = item.get("volume", item.get("v"))
            else:
                continue

            bars.append(
                {
                    "ts": datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
            )

        bars.sort(key=lambda x: x["ts"])
        return bars

    async def get_ticker(self, symbol: str) -> dict:
        """Return latest ticker data."""
        sym = normalize_symbol(symbol)
        data = await self._request_with_retry(
            "GET", self._ticker_path(), params={"symbol": sym}
        )
        # Normalize to common format
        if isinstance(data, list):
            data = data[0]
        return {
            "symbol": sym,
            "last": float(data.get("lastPrice", data.get("c", data.get("last", 0)))),
            "bid": float(data.get("bidPrice", data.get("b", 0))),
            "ask": float(data.get("askPrice", data.get("a", 0))),
            "volume": float(data.get("volume", data.get("v", 0))),
            "ts": datetime.now(timezone.utc),
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: float | None = None,
        client_order_id: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """
        Place an order. Returns raw broker response dict.

        side: BUY | SELL
        order_type: MARKET | LIMIT
        """
        sym = normalize_symbol(symbol)
        payload: dict[str, Any] = {
            "symbol": sym,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(qty),
        }
        if price is not None:
            payload["price"] = str(price)
        if client_order_id:
            # BingX requires alphanumeric clientOrderID (no hyphens)
            payload["clientOrderID"] = client_order_id.replace("-", "")[:36]
        if extra:
            payload.update(extra)

        data = await self._request_with_retry(
            "POST", self._order_path(), json=payload, signed=True
        )
        return data if isinstance(data, dict) else {"order": data}

    async def get_order(self, order_id: str, symbol: str) -> dict:
        sym = normalize_symbol(symbol)
        data = await self._request_with_retry(
            "GET",
            self._order_path(),
            params={"orderId": order_id, "symbol": sym},
            signed=True,
        )
        return data if isinstance(data, dict) else {}

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        sym = normalize_symbol(symbol)
        data = await self._request_with_retry(
            "DELETE",
            self._order_path(),
            params={"orderId": order_id, "symbol": sym},
            signed=True,
        )
        return data if isinstance(data, dict) else {}

    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = normalize_symbol(symbol)
        data = await self._request_with_retry(
            "GET", self._positions_path(), params=params, signed=True
        )
        return data if isinstance(data, list) else data.get("positions", [])

    async def get_balance(self) -> list[dict]:
        data = await self._request_with_retry(
            "GET", self._balance_path(), params={}, signed=True
        )
        if isinstance(data, dict) and "balance" in data:
            return [data["balance"]] if isinstance(data["balance"], dict) else data["balance"]
        return data if isinstance(data, list) else []

    async def get_contracts(self) -> list[dict]:
        """
        Fetch all available perpetual swap contracts from BingX.
        Endpoint: GET /openApi/swap/v2/quote/contracts
        Returns list of contract info dicts (symbol, size, tickSize, etc.)
        """
        data = await self._request_with_retry(
            "GET", "/openApi/swap/v2/quote/contracts", params={}
        )
        return data if isinstance(data, list) else []

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a perpetual swap symbol."""
        sym = normalize_symbol(symbol)
        data = await self._request_with_retry(
            "POST",
            self._leverage_path(),
            json={"symbol": sym, "side": "LONG", "leverage": leverage},
            signed=True,
        )
        return data if isinstance(data, dict) else {}

    async def place_stop_order(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        qty: float,
        pos_side: str,
        client_order_id: str | None = None,
    ) -> dict:
        """
        Place a STOP_MARKET order on BingX perpetual futures.

        This is used to set a hard stop-loss on the exchange so that if the bot
        crashes or goes offline, open positions are protected automatically.

        Args:
            symbol:           e.g. "BTC-USDT"
            side:             "BUY" to close a SHORT, "SELL" to close a LONG
            stop_price:       trigger price for the stop-market order
            qty:              position quantity to close
            pos_side:         "LONG" or "SHORT" (hedge-mode positionSide)
            client_order_id:  optional idempotency key (alphanumeric, max 36 chars)
        """
        sym = normalize_symbol(symbol)
        payload: dict[str, Any] = {
            "symbol": sym,
            "side": side.upper(),
            "type": "STOP_MARKET",
            "quantity": str(qty),
            "stopPrice": str(stop_price),
            "positionSide": pos_side.upper(),
        }
        if client_order_id:
            payload["clientOrderID"] = client_order_id.replace("-", "")[:36]

        data = await self._request_with_retry(
            "POST", self._order_path(), json=payload, signed=True
        )
        return data if isinstance(data, dict) else {"order": data}

    async def cancel_all_open_orders(self, symbol: str) -> dict:
        """Cancel all open orders for a symbol (e.g. stale SL orders on position close)."""
        sym = normalize_symbol(symbol)
        data = await self._request_with_retry(
            "DELETE",
            "/openApi/swap/v2/trade/allOpenOrders",
            params={"symbol": sym},
            signed=True,
        )
        return data if isinstance(data, dict) else {}

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BingXClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
