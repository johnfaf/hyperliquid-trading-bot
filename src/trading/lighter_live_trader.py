"""
Optional Lighter live execution bridge.

The existing Lighter adapter is a public-data adapter.  This module adds a
fail-closed private execution path for operators who explicitly select
``LIVE_EXECUTION_VENUE=lighter`` and provide Lighter API credentials.

Private Lighter order APIs are SDK-backed and async.  The bridge keeps the
same small surface used by the rest of the bot: ``execute_signal()``,
``place_market_order()``, ``place_trigger_order()``, ``cancel_all_orders()``,
``get_positions()``, and ``get_stats()``.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import config
from src.exchanges.lighter_adapter import LighterAdapter
from src.signals.signal_schema import TradeSignal, signal_from_execution_dict

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async SDK call from the synchronous trading cycle."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if not loop.is_running():
        return loop.run_until_complete(coro)

    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - surfaced below
            result["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=30)
    if thread.is_alive():
        raise TimeoutError("lighter_sdk_call_timeout")
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


def _tuple_result(value) -> tuple[Any, Any, Any]:
    """Normalize common Lighter SDK return shapes."""
    if isinstance(value, tuple):
        if len(value) >= 3:
            return value[0], value[1], value[2]
        if len(value) == 2:
            return value[0], None, value[1]
        if len(value) == 1:
            return value[0], None, None
    return value, None, None


def _side_value(side: Any) -> str:
    raw = getattr(side, "value", side)
    return str(raw or "").strip().lower()


@dataclass
class _MarketPrecision:
    order_book_index: int
    size_decimals: int
    price_decimals: int


class LighterLiveTrader:
    """Private execution bridge for Lighter, disabled unless fully configured."""

    exchange = "lighter"

    def __init__(
        self,
        *,
        dry_run: bool = True,
        base_url: Optional[str] = None,
        account_index: Optional[int] = None,
        api_key_index: Optional[int] = None,
        private_key: Optional[str] = None,
        l1_address: Optional[str] = None,
        max_order_usd: Optional[float] = None,
        default_leverage: Optional[float] = None,
        max_slippage_bps: Optional[float] = None,
    ) -> None:
        self.dry_run = bool(dry_run)
        self.base_url = (base_url or config.LIGHTER_BASE_URL).rstrip("/")
        self.account_index = int(account_index if account_index is not None else config.LIGHTER_ACCOUNT_INDEX)
        self.api_key_index = int(api_key_index if api_key_index is not None else config.LIGHTER_API_KEY_INDEX)
        self.private_key = private_key if private_key is not None else config.LIGHTER_PRIVATE_KEY
        self.l1_address = l1_address if l1_address is not None else config.LIGHTER_L1_ADDRESS
        self.min_order_usd = float(getattr(config, "LIGHTER_MIN_ORDER_USD", 1.0))
        self.max_order_usd = float(max_order_usd if max_order_usd is not None else config.LIGHTER_MAX_ORDER_USD)
        self.default_leverage = float(
            default_leverage if default_leverage is not None else config.LIGHTER_DEFAULT_LEVERAGE
        )
        self.max_slippage_bps = float(
            max_slippage_bps if max_slippage_bps is not None else config.LIGHTER_MAX_SLIPPAGE_BPS
        )
        self.market_adapter = LighterAdapter(config={"base_url": self.base_url})
        self._sdk = None
        self._client = None
        self._last_error = ""
        self._submitted_orders = 0
        self._submitted_triggers = 0
        self._failed_orders = 0
        self._tracked_orders: list[dict[str, Any]] = []
        self.status_reason = ""
        self._client_order_counter = 0
        self._load_sdk()

    def _load_sdk(self) -> None:
        try:
            self._sdk = importlib.import_module("lighter")
        except Exception as exc:
            self._sdk = None
            self.status_reason = "lighter_sdk_missing"
            self._last_error = f"lighter SDK import failed: {exc}"

    def is_live_enabled(self) -> bool:
        return bool(config.LIGHTER_LIVE_TRADING_ENABLED and not self.dry_run)

    def is_deployable(self) -> bool:
        if not self.is_live_enabled():
            self.status_reason = "lighter_live_disabled_or_dry_run"
            return False
        if not bool(getattr(config, "LIGHTER_LIVE_TRADING_DUAL_CONTROL_CONFIRM", False)):
            self.status_reason = "lighter_dual_control_confirmation_missing"
            return False
        if self.account_index < 0:
            self.status_reason = "LIGHTER_ACCOUNT_INDEX_missing"
            return False
        if not self.private_key:
            self.status_reason = "LIGHTER_PRIVATE_KEY_missing"
            return False
        if self._sdk is None:
            self.status_reason = "lighter_sdk_missing"
            return False
        return True

    def _sdk_constant(self, *names: str, default: Any = None) -> Any:
        for name in names:
            if self._client is not None and hasattr(self._client, name):
                return getattr(self._client, name)
            if self._sdk is not None and hasattr(self._sdk, name):
                return getattr(self._sdk, name)
        return default

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._sdk is None:
            raise RuntimeError("lighter SDK is not installed")
        signer_cls = getattr(self._sdk, "SignerClient", None)
        if signer_cls is None:
            raise RuntimeError("lighter.SignerClient is unavailable")
        try:
            self._client = signer_cls(
                url=self.base_url,
                account_index=self.account_index,
                api_private_keys={self.api_key_index: self.private_key},
            )
        except TypeError:
            self._client = signer_cls(
                url=self.base_url,
                private_key=self.private_key,
                account_index=self.account_index,
                api_key_index=self.api_key_index,
            )
        return self._client

    def _client_order_index(self) -> int:
        self._client_order_counter = (self._client_order_counter + 1) % 1_000_000
        raw = (int(time.time() * 1000) * 1_000_000) + self._client_order_counter
        return raw % (2**48 - 1)

    def _market_precision(self, coin: str) -> Optional[_MarketPrecision]:
        coin = str(coin or "").upper()
        self.market_adapter._ensure_markets_loaded()
        order_book_id = self.market_adapter._reverse_symbol_map.get(coin)
        if order_book_id is None:
            self.status_reason = f"lighter_market_missing:{coin}"
            return None
        row = self.market_adapter._market_cache.get(str(order_book_id), {})

        def _int_field(*names: str, default: int) -> int:
            for name in names:
                value = row.get(name)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass
            return default

        return _MarketPrecision(
            order_book_index=int(order_book_id),
            size_decimals=_int_field(
                "size_decimals", "sizeDecimals", "base_decimals", "baseDecimals",
                default=int(config.LIGHTER_SIZE_DECIMALS_DEFAULT),
            ),
            price_decimals=_int_field(
                "price_decimals", "priceDecimals", "quote_decimals", "quoteDecimals",
                default=int(config.LIGHTER_PRICE_DECIMALS_DEFAULT),
            ),
        )

    def _market_mid(self, coin: str) -> float:
        try:
            markets = self.market_adapter.get_market_data([str(coin).upper()])
            if markets:
                return float(markets[0].mid_price or markets[0].mark_price or 0.0)
        except Exception as exc:
            logger.debug("Lighter market mid failed for %s: %s", coin, exc)
        return 0.0

    def _to_int_amount(self, value: float, decimals: int) -> int:
        return int(round(float(value) * (10 ** int(decimals))))

    def _from_int_amount(self, value: Any, decimals: int) -> float:
        try:
            return float(value) / (10 ** int(decimals))
        except Exception:
            return 0.0

    def get_account_value(self) -> Optional[float]:
        snapshot = self._account_snapshot()
        for key in ("account_value", "accountValue", "collateral", "total_collateral"):
            try:
                value = float(snapshot.get(key, 0) or 0)
                if value > 0:
                    return value
            except Exception:
                continue
        return None

    def get_free_margin(self) -> Optional[float]:
        snapshot = self._account_snapshot()
        for key in ("available_balance", "availableBalance", "free_collateral", "freeCollateral"):
            try:
                value = float(snapshot.get(key, 0) or 0)
                if value >= 0:
                    return value
            except Exception:
                continue
        return self.get_account_value()

    def _account_snapshot(self) -> Dict[str, Any]:
        params = None
        if self.account_index >= 0:
            params = {"by": "index", "value": self.account_index}
        elif self.l1_address:
            params = {"by": "l1_address", "value": self.l1_address}
        if not params:
            return {}
        data = self.market_adapter._get("/account", params=params, quiet=True)
        if isinstance(data, dict):
            return data
        return {}

    def get_positions(self, *_, force_fresh: bool = False) -> list[dict[str, Any]]:
        if self.l1_address:
            positions = self.market_adapter.get_trader_positions(self.l1_address)
            return [
                {
                    "exchange": "lighter",
                    "coin": p.coin,
                    "side": p.side,
                    "size": p.size,
                    "szi": p.size if p.side == "long" else -p.size,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": p.leverage,
                    "margin_used": p.margin_used,
                }
                for p in positions
            ]
        return []

    def get_open_orders(self, *_, force_fresh: bool = False) -> list[dict[str, Any]]:
        return list(self._tracked_orders)

    def place_market_order(
        self,
        coin: str,
        side: str,
        size: float,
        leverage: float = 1,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        return self._place_order(
            coin=coin,
            side=side,
            size=size,
            leverage=leverage,
            reduce_only=reduce_only,
            order_type="market",
        )

    def place_trigger_order(
        self,
        coin: str,
        side: str,
        size: float,
        trigger_price: float,
        tp_or_sl: str = "sl",
    ) -> Dict[str, Any]:
        return self._place_order(
            coin=coin,
            side=side,
            size=size,
            leverage=1,
            reduce_only=True,
            order_type=str(tp_or_sl or "sl").lower(),
            trigger_price=trigger_price,
        )

    def _place_order(
        self,
        *,
        coin: str,
        side: str,
        size: float,
        leverage: float,
        reduce_only: bool,
        order_type: str,
        trigger_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        coin = str(coin or "").upper()
        side_value = _side_value(side)
        is_ask = side_value in {"sell", "short"}
        if side_value in {"long", "buy"}:
            is_ask = False
        elif side_value in {"short", "sell"}:
            is_ask = True
        else:
            return {"status": "rejected", "reason": "invalid_side", "venue": "lighter"}

        precision = self._market_precision(coin)
        if precision is None:
            return {"status": "rejected", "reason": self.status_reason, "venue": "lighter"}
        size = abs(float(size or 0.0))
        if size <= 0 or not math.isfinite(size):
            return {"status": "rejected", "reason": "invalid_size", "venue": "lighter"}

        reference_price = float(trigger_price or 0.0) or self._market_mid(coin)
        if reference_price <= 0:
            return {"status": "rejected", "reason": "price_unavailable", "venue": "lighter"}
        notional = size * reference_price
        if not reduce_only and notional > self.max_order_usd:
            size = self.max_order_usd / reference_price
            notional = size * reference_price

        amount_int = self._to_int_amount(size, precision.size_decimals)
        slip = self.max_slippage_bps / 10_000.0
        exec_price = reference_price * (1 - slip if is_ask else 1 + slip)
        price_int = self._to_int_amount(float(trigger_price or exec_price), precision.price_decimals)
        if amount_int <= 0 or price_int <= 0:
            return {"status": "rejected", "reason": "rounded_order_zero", "venue": "lighter"}

        if not self.is_deployable():
            return {
                "status": "dry_run" if self.dry_run else "rejected",
                "reason": self.status_reason,
                "venue": "lighter",
                "coin": coin,
                "side": side_value,
                "size": size,
                "price": reference_price,
                "notional": notional,
            }

        try:
            client = self._get_client()
            sdk_order_type = self._resolve_order_type(order_type)
            is_trigger = order_type in {"sl", "tp", "stop_loss", "take_profit"}
            tif = self._sdk_constant(
                "ORDER_TIME_IN_FORCE_GOOD_TILL_TIME" if is_trigger else "ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                "ORDER_TIME_IN_FORCE_GOOD_TIL_TIME" if is_trigger else "ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                default=1 if is_trigger else 0,
            )
            trigger_price_int = (
                self._to_int_amount(float(trigger_price), precision.price_decimals)
                if is_trigger and trigger_price
                else self._sdk_constant("NIL_TRIGGER_PRICE", default=0)
            )
            order_expiry = self._sdk_constant("DEFAULT_28_DAY_ORDER_EXPIRY", default=0)

            async def _submit():
                response = await _maybe_await(client.create_order(
                    market_index=precision.order_book_index,
                    client_order_index=self._client_order_index(),
                    base_amount=amount_int,
                    price=price_int,
                    is_ask=is_ask,
                    order_type=sdk_order_type,
                    time_in_force=tif,
                    reduce_only=bool(reduce_only),
                    trigger_price=trigger_price_int,
                    order_expiry=order_expiry,
                ))
                return response

            tx, tx_hash, err = _tuple_result(_run_async(_submit()))
            if err:
                self._failed_orders += 1
                self._last_error = str(err)
                return {"status": "rejected", "reason": "lighter_sdk_error", "error": str(err), "venue": "lighter"}
            payload = {
                "status": "submitted",
                "venue": "lighter",
                "coin": coin,
                "side": side_value,
                "size": size,
                "price": reference_price,
                "notional": notional,
                "order_type": order_type,
                "reduce_only": bool(reduce_only),
                "tx": tx,
                "tx_hash": tx_hash,
            }
            if order_type in {"sl", "tp", "stop_loss", "take_profit"}:
                self._submitted_triggers += 1
                self._tracked_orders.append(payload)
            else:
                self._submitted_orders += 1
            return payload
        except Exception as exc:
            self._failed_orders += 1
            self._last_error = str(exc)
            logger.error("Lighter order failed for %s %s: %s", coin, side_value, exc)
            return {"status": "error", "reason": type(exc).__name__, "message": str(exc), "venue": "lighter"}

    def _resolve_order_type(self, order_type: str) -> Any:
        value = str(order_type or "market").lower()
        if value in {"sl", "stop", "stop_loss"}:
            return self._sdk_constant("ORDER_TYPE_STOP_LOSS", "ORDER_TYPE_STOP_MARKET", default=3)
        if value in {"tp", "take_profit"}:
            return self._sdk_constant("ORDER_TYPE_TAKE_PROFIT", "ORDER_TYPE_TAKE_PROFIT_MARKET", default=4)
        return self._sdk_constant("ORDER_TYPE_MARKET", default=1)

    def execute_signal(self, signal: TradeSignal | Dict[str, Any], bypass_firewall: bool = False) -> Optional[Dict[str, Any]]:
        trade_signal = signal if isinstance(signal, TradeSignal) else signal_from_execution_dict(signal)
        side = _side_value(trade_signal.side)
        size = abs(float(getattr(trade_signal, "size", 0.0) or 0.0))
        entry_price = float(getattr(trade_signal, "entry_price", 0.0) or 0.0) or self._market_mid(trade_signal.coin)
        leverage = max(1.0, float(getattr(trade_signal, "leverage", self.default_leverage) or self.default_leverage))
        if size <= 0 and entry_price > 0:
            account_value = self.get_free_margin() or self.get_account_value() or 0.0
            margin_budget = account_value * float(getattr(trade_signal, "position_pct", 0.0) or 0.0)
            size = (margin_budget * leverage) / entry_price if entry_price > 0 else 0.0

        entry = self.place_market_order(
            trade_signal.coin,
            "buy" if side == "long" else "sell",
            size,
            leverage=leverage,
            reduce_only=False,
        )
        if entry.get("status") not in {"submitted", "dry_run"}:
            return entry

        sl_price, tp_price = trade_signal.risk.resolve_trigger_prices(entry_price, side, leverage)
        close_side = "sell" if side == "long" else "buy"
        sl = self.place_trigger_order(trade_signal.coin, close_side, size, sl_price, tp_or_sl="sl")
        tp = self.place_trigger_order(trade_signal.coin, close_side, size, tp_price, tp_or_sl="tp")
        return {
            "status": entry.get("status"),
            "venue": "lighter",
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
        }

    def close_position(self, coin: str) -> Dict[str, Any]:
        coin = str(coin or "").upper()
        positions = [p for p in self.get_positions(force_fresh=True) if str(p.get("coin") or "").upper() == coin]
        if not positions:
            return {"status": "skipped", "reason": "no_position", "venue": "lighter", "coin": coin}
        pos = positions[0]
        side = "sell" if str(pos.get("side")).lower() == "long" else "buy"
        return self.place_market_order(coin, side, abs(float(pos.get("size", 0) or 0)), reduce_only=True)

    def cancel_all_orders(self, coin: Optional[str] = None) -> int:
        target = str(coin or "").upper()
        before = len(self._tracked_orders)
        if target:
            self._tracked_orders = [o for o in self._tracked_orders if str(o.get("coin") or "").upper() != target]
        else:
            self._tracked_orders.clear()
        return before - len(self._tracked_orders)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "venue": "lighter",
            "live_enabled": self.is_live_enabled(),
            "deployable": self.is_deployable(),
            "dry_run": self.dry_run,
            "status_reason": self.status_reason,
            "account_index_configured": self.account_index >= 0,
            "l1_address_configured": bool(self.l1_address),
            "sdk_loaded": self._sdk is not None,
            "submitted_orders": self._submitted_orders,
            "submitted_triggers": self._submitted_triggers,
            "failed_orders": self._failed_orders,
            "tracked_orders": len(self._tracked_orders),
            "last_error": self._last_error,
            "max_order_usd": self.max_order_usd,
        }
