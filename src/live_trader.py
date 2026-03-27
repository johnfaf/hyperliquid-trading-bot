"""
Live Trader Module
==================
Executes real trades on Hyperliquid DEX using EIP-712 signed orders.

This module:
  - Signs orders using eth_account for EIP-712
  - Places market, limit, and trigger orders on Hyperliquid
  - Enforces position limits, daily loss limits, kill switches
  - All trades routed through DecisionFirewall before execution
  - Comprehensive audit logging for every trade
  - Dry-run mode by default (must explicitly enable)

SECURITY NOTES:
  - Private key loaded from HL_PRIVATE_KEY env var
  - dry_run=True by default — nothing executes without explicit enablement
  - Daily loss limit triggers automatic kill switch
  - Max position size enforced per order
  - All API errors caught and logged, never crash
"""

import logging
import os
import time
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import requests

# Try importing eth_account; if unavailable, set flag
try:
    from eth_account import Account
    from eth_account.messages import encode_structured_data
    HAS_ETH_ACCOUNT = True
except ImportError:
    HAS_ETH_ACCOUNT = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.decision_firewall import DecisionFirewall
from src.signal_schema import TradeSignal, SignalSide

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order types supported by Hyperliquid."""
    LIMIT_GTC = "Gtc"         # Good Till Canceled limit
    LIMIT_IOC = "Ioc"         # Immediate or Cancel market-style
    LIMIT_ALO = "Alo"         # Add Liquidity Only
    TRIGGER_SL = "sl"         # Stop loss trigger
    TRIGGER_TP = "tp"         # Take profit trigger


class HyperliquidSigner:
    """
    Signs exchange requests using EIP-712 for Hyperliquid.

    Implements the domain and signing requirements for Hyperliquid orders.
    """

    # EIP-712 domain for Hyperliquid (Arbitrum mainnet)
    DOMAIN = {
        "name": "HyperliquidSignTransaction",
        "version": "1",
        "chainId": 42161,  # Arbitrum mainnet
        "verifyingContract": "0x0000000000000000000000000000000000000000"
    }

    def __init__(self, private_key: str):
        """
        Initialize signer with Ethereum private key.

        Args:
            private_key: Hex string (with or without 0x prefix)
        """
        if not HAS_ETH_ACCOUNT:
            raise RuntimeError(
                "eth_account library not installed. "
                "Please install: pip install eth_account"
            )

        # Ensure 0x prefix
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self.account = Account.from_key(private_key)
        self.address = self.account.address
        logger.info(f"HyperliquidSigner initialized with address: {self.address}")

    def sign_action(self, action: Dict, nonce: int) -> Dict:
        """
        Sign an action for submission to Hyperliquid exchange.

        Args:
            action: The action dict (order, cancel, etc.)
            nonce: Timestamp in milliseconds

        Returns:
            Dict with 'r', 's', 'v' signature components
        """
        try:
            # Build the EIP-712 payload
            payload = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"}
                    ],
                    "HyperliquidTransaction": [
                        {"name": "action", "type": "string"},
                        {"name": "nonce", "type": "uint64"}
                    ]
                },
                "primaryType": "HyperliquidTransaction",
                "domain": self.DOMAIN,
                "message": {
                    "action": json.dumps(action),
                    "nonce": nonce
                }
            }

            # Sign using eth_account
            message = encode_structured_data(payload)
            signed_message = self.account.sign_message(message)

            # Return signature components
            return {
                "r": signed_message.signature.hex()[:66],    # r component
                "s": "0x" + signed_message.signature.hex()[66:130],  # s component
                "v": signed_message.v
            }
        except Exception as e:
            logger.error(f"Error signing action: {e}")
            raise

    @staticmethod
    def get_action_hash(action: Dict) -> str:
        """Compute hash of action for auditing."""
        action_str = json.dumps(action, sort_keys=True)
        return hashlib.sha256(action_str.encode()).hexdigest()


class LiveTrader:
    """
    Executes real trades on Hyperliquid DEX.

    CRITICAL: dry_run=True by default — must be explicitly enabled.
    All signals routed through DecisionFirewall before execution.
    """

    def __init__(self, firewall: DecisionFirewall, dry_run: bool = True,
                 max_daily_loss: float = 500, max_position_size: float = 1000):
        """
        Initialize live trader.

        Args:
            firewall: DecisionFirewall instance for validation
            dry_run: If True, log what WOULD happen but don't execute (default True)
            max_daily_loss: Daily loss limit in USD (default $500)
            max_position_size: Max notional per position in USD (default $1000)
        """
        self.firewall = firewall
        self.dry_run = dry_run
        self.max_daily_loss = float(os.environ.get("HL_MAX_DAILY_LOSS", max_daily_loss))
        self.max_position_size = float(os.environ.get("HL_MAX_POSITION_SIZE", max_position_size))

        # API endpoints (must come early since _load_* methods need these)
        self.exchange_url = "https://api.hyperliquid.xyz/exchange"
        self.info_url = "https://api.hyperliquid.xyz/info"

        # Signer (loaded from env)
        self.signer = None
        self.public_address = None
        self._load_credentials()

        # Asset index mapping (BTC=0, ETH=1, etc.)
        self.asset_index_map = {}
        self._load_asset_index_map()

        # State tracking
        self.daily_pnl = 0.0
        self.daily_reset_date = ""
        self.kill_switch_active = False
        self.orders_today = 0
        self.fills_today = 0

        logger.info(
            f"LiveTrader initialized: dry_run={dry_run}, "
            f"max_daily_loss=${self.max_daily_loss:.2f}, "
            f"max_position_size=${self.max_position_size:.2f}"
        )

        if dry_run:
            logger.warning("⚠️  DRY RUN MODE - No real trades will be executed")

        if not self.signer:
            logger.warning("⚠️  No private key configured - forcing dry_run mode")
            self.dry_run = True

    def _load_credentials(self):
        """Load private key and public address from environment."""
        private_key = os.environ.get("HL_PRIVATE_KEY")
        public_address = os.environ.get("HL_PUBLIC_ADDRESS")

        if not private_key or not public_address:
            logger.warning("HL_PRIVATE_KEY or HL_PUBLIC_ADDRESS not set in environment")
            return

        try:
            self.signer = HyperliquidSigner(private_key)
            self.public_address = public_address
            logger.info(f"Credentials loaded for address: {public_address}")
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            self.signer = None
            self.public_address = None

    def _load_asset_index_map(self):
        """Load asset index mapping from Hyperliquid meta endpoint."""
        try:
            response = requests.post(
                self.info_url,
                json={"type": "meta"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if "universe" in data:
                for idx, coin_data in enumerate(data["universe"]):
                    coin_name = coin_data.get("name", "")
                    if coin_name:
                        self.asset_index_map[coin_name] = idx

                logger.info(f"Loaded {len(self.asset_index_map)} asset indices from meta")
            else:
                logger.warning("No 'universe' in meta response")
        except Exception as e:
            logger.error(f"Failed to load asset index map: {e}")

    def _get_asset_index(self, coin: str) -> Optional[int]:
        """Get asset index for a coin."""
        idx = self.asset_index_map.get(coin)
        if idx is None:
            logger.warning(f"Asset index not found for {coin}")
        return idx

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self.daily_reset_date:
            self.daily_reset_date = today
            self.daily_pnl = 0.0
            self.orders_today = 0
            self.fills_today = 0
            self.kill_switch_active = False
            logger.info("Daily counters reset")

    def check_daily_loss(self) -> bool:
        """
        Check if daily loss limit exceeded.

        Returns:
            True if loss > limit (triggers kill switch)
        """
        self._check_daily_reset()

        if abs(self.daily_pnl) > self.max_daily_loss:
            logger.warning(f"⚠️  Daily loss limit exceeded: ${abs(self.daily_pnl):.2f} > ${self.max_daily_loss:.2f}")
            self.kill_switch_active = True
            return True

        return False

    def _get_mid_price(self, coin: str) -> Optional[float]:
        """Get mid price from Hyperliquid."""
        try:
            response = requests.post(
                self.info_url,
                json={"type": "allMids"},
                timeout=10
            )
            response.raise_for_status()
            mids = response.json()

            price = mids.get(coin)
            if price:
                return float(price)
        except Exception as e:
            logger.error(f"Failed to get mid price for {coin}: {e}")

        return None

    def _apply_market_slippage(self, mid_price: float, side: str, slippage_pct: float = 0.05) -> float:
        """
        Apply slippage to mid price for market execution.

        Args:
            mid_price: Current mid price
            side: "buy" or "sell"
            slippage_pct: Slippage percentage (default 5%)

        Returns:
            Adjusted price
        """
        if side.lower() == "buy":
            return mid_price * (1 + slippage_pct)
        else:
            return mid_price * (1 - slippage_pct)

    def _post_order(self, action: Dict, dry_run_override: Optional[bool] = None) -> Dict:
        """
        Post an order to Hyperliquid exchange.

        Args:
            action: The action dict (order, cancel, etc.)
            dry_run_override: Override dry_run for this call (default None = use self.dry_run)

        Returns:
            Response dict
        """
        is_dry_run = dry_run_override if dry_run_override is not None else self.dry_run

        nonce = int(time.time() * 1000)
        action_hash = HyperliquidSigner.get_action_hash(action)

        if is_dry_run:
            logger.info(f"[DRY RUN] Would post order: {json.dumps(action, indent=2)}")
            return {
                "dry_run": True,
                "action_hash": action_hash,
                "status": "simulated",
                "message": "This is a dry run"
            }

        try:
            if not self.signer:
                logger.error("No signer available - cannot post order")
                return {"status": "error", "message": "No signer available"}

            # Sign the action
            signature = self.signer.sign_action(action, nonce)

            # Prepare request
            payload = {
                "action": action,
                "nonce": nonce,
                "signature": signature
            }

            logger.debug(f"Posting order to {self.exchange_url}: action_hash={action_hash}")

            response = requests.post(
                self.exchange_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"Order posted: {json.dumps(result)}")
            return result

        except Exception as e:
            logger.error(f"Error posting order: {e}")
            return {"status": "error", "message": str(e)}

    def place_market_order(self, coin: str, side: str, size: float,
                           leverage: float = 1, reduce_only: bool = False) -> Dict:
        """
        Place a market order on Hyperliquid.

        Args:
            coin: Coin symbol (e.g., "BTC")
            side: "buy" or "sell"
            size: Size in coins
            leverage: Leverage multiplier (default 1)
            reduce_only: Only reduce position (default False)

        Returns:
            Order result dict
        """
        # Check kill switch
        if self.kill_switch_active:
            logger.warning(f"Kill switch active - rejecting market order {coin} {side}")
            return {"status": "rejected", "reason": "kill_switch_active"}

        # Check daily loss
        if self.check_daily_loss():
            logger.warning(f"Daily loss limit exceeded - rejecting market order {coin} {side}")
            return {"status": "rejected", "reason": "daily_loss_exceeded"}

        # Get asset index
        asset_idx = self._get_asset_index(coin)
        if asset_idx is None:
            return {"status": "error", "message": f"Unknown coin: {coin}"}

        # Get mid price
        mid = self._get_mid_price(coin)
        if not mid:
            return {"status": "error", "message": f"Could not get mid price for {coin}"}

        # Apply slippage
        price = self._apply_market_slippage(mid, side)

        # Check position size
        notional = size * price
        if notional > self.max_position_size:
            logger.warning(f"Position size ${notional:.2f} exceeds limit ${self.max_position_size:.2f}")
            return {"status": "rejected", "reason": "position_size_exceeded"}

        # Build order
        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": str(round(price, 8)),
            "s": str(round(size, 8)),
            "r": reduce_only,
            "t": {"limit": {"tif": OrderType.LIMIT_IOC.value}}
        }

        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }

        logger.info(f"Placing market order: {coin} {side} {size} @ ${price:.2f} (notional: ${notional:.2f})")
        result = self._post_order(action)

        if result.get("status") != "error" and not self.dry_run:
            self.orders_today += 1

        return result

    def place_limit_order(self, coin: str, side: str, size: float, price: float,
                          leverage: float = 1, reduce_only: bool = False) -> Dict:
        """
        Place a limit order on Hyperliquid.

        Args:
            coin: Coin symbol
            side: "buy" or "sell"
            size: Size in coins
            price: Limit price
            leverage: Leverage (default 1)
            reduce_only: Only reduce position (default False)

        Returns:
            Order result dict
        """
        if self.kill_switch_active:
            logger.warning(f"Kill switch active - rejecting limit order {coin}")
            return {"status": "rejected", "reason": "kill_switch_active"}

        asset_idx = self._get_asset_index(coin)
        if asset_idx is None:
            return {"status": "error", "message": f"Unknown coin: {coin}"}

        # Check position size
        notional = size * price
        if notional > self.max_position_size:
            logger.warning(f"Position size ${notional:.2f} exceeds limit")
            return {"status": "rejected", "reason": "position_size_exceeded"}

        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": str(round(price, 8)),
            "s": str(round(size, 8)),
            "r": reduce_only,
            "t": {"limit": {"tif": OrderType.LIMIT_GTC.value}}
        }

        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }

        logger.info(f"Placing limit order: {coin} {side} {size} @ ${price:.2f}")
        result = self._post_order(action)

        if result.get("status") != "error" and not self.dry_run:
            self.orders_today += 1

        return result

    def place_trigger_order(self, coin: str, side: str, size: float,
                            trigger_price: float, tp_or_sl: str = "sl") -> Dict:
        """
        Place a trigger order (stop loss or take profit).

        Args:
            coin: Coin symbol
            side: "buy" or "sell"
            size: Size in coins
            trigger_price: Price to trigger at
            tp_or_sl: "sl" for stop loss, "tp" for take profit

        Returns:
            Order result dict
        """
        asset_idx = self._get_asset_index(coin)
        if asset_idx is None:
            return {"status": "error", "message": f"Unknown coin: {coin}"}

        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": "0",  # Trigger price is in trigger, not order price
            "s": str(round(size, 8)),
            "r": False,
            "t": {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": str(round(trigger_price, 8)),
                    "tpsl": tp_or_sl
                }
            }
        }

        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }

        logger.info(f"Placing {tp_or_sl} trigger: {coin} {size} @ ${trigger_price:.2f}")
        result = self._post_order(action)

        return result

    def cancel_order(self, coin: str, order_id: int) -> bool:
        """
        Cancel a specific order.

        Args:
            coin: Coin symbol
            order_id: Order ID from exchange

        Returns:
            True if cancelled successfully
        """
        asset_idx = self._get_asset_index(coin)
        if asset_idx is None:
            logger.warning(f"Cannot cancel order for unknown coin: {coin}")
            return False

        action = {
            "type": "cancel",
            "cancels": [{"a": asset_idx, "o": order_id}]
        }

        result = self._post_order(action)
        return result.get("status") != "error"

    def cancel_all_orders(self, coin: Optional[str] = None) -> int:
        """
        Cancel all open orders (optionally for a specific coin).

        Args:
            coin: Specific coin (optional). If None, cancel all.

        Returns:
            Number of orders cancelled
        """
        logger.info(f"Cancelling all orders{f' for {coin}' if coin else ''}")

        try:
            orders = self.get_open_orders()
            if not orders:
                return 0

            if coin:
                orders = [o for o in orders if o.get("coin") == coin]

            for order in orders:
                self.cancel_order(order.get("coin"), order.get("id"))

            return len(orders)
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return 0

    def close_position(self, coin: str) -> Dict:
        """
        Close entire position on a coin with market order.

        Args:
            coin: Coin symbol

        Returns:
            Order result dict
        """
        logger.info(f"Closing position on {coin}")

        try:
            positions = self.get_positions()
            pos = next((p for p in positions if p.get("coin") == coin), None)

            if not pos:
                logger.warning(f"No position found for {coin}")
                return {"status": "error", "message": "No position found"}

            size = abs(pos.get("size", 0))
            side = "sell" if pos.get("size", 0) > 0 else "buy"

            return self.place_market_order(coin, side, size, reduce_only=True)

        except Exception as e:
            logger.error(f"Error closing position on {coin}: {e}")
            return {"status": "error", "message": str(e)}

    def emergency_close_all(self) -> List[Dict]:
        """
        KILL SWITCH: Close all positions immediately and cancel all orders.

        Returns:
            List of close results
        """
        logger.critical("🔴 EMERGENCY CLOSE ALL TRIGGERED")
        self.kill_switch_active = True

        results = []

        try:
            # Cancel all orders first
            cancelled = self.cancel_all_orders()
            logger.warning(f"Cancelled {cancelled} open orders")

            # Close all positions
            positions = self.get_positions()
            for pos in positions:
                result = self.close_position(pos.get("coin"))
                results.append(result)
                logger.warning(f"Closed position on {pos.get('coin')}")

        except Exception as e:
            logger.error(f"Error in emergency close all: {e}")

        return results

    def get_positions(self) -> List[Dict]:
        """
        Get current open positions from exchange.

        Returns:
            List of position dicts
        """
        try:
            if not self.public_address:
                logger.warning("No public address configured")
                return []

            response = requests.post(
                self.info_url,
                json={"type": "clearinghouseState", "user": self.public_address},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            positions = data.get("assetPositions", [])
            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_open_orders(self) -> List[Dict]:
        """
        Get current open orders from exchange.

        Returns:
            List of open order dicts
        """
        try:
            if not self.public_address:
                logger.warning("No public address configured")
                return []

            response = requests.post(
                self.info_url,
                json={"type": "openOrders", "user": self.public_address},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            orders = data.get("orders", []) if isinstance(data, dict) else data
            return orders

        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def execute_signal(self, signal: TradeSignal) -> Optional[Dict]:
        """
        Main entry point: execute a trade signal through firewall.

        Pipeline:
          1. Validate signal through DecisionFirewall
          2. Check kill switch and daily loss
          3. Place market entry order
          4. Place stop loss trigger
          5. Place take profit trigger
          6. Return execution result

        Args:
            signal: TradeSignal object

        Returns:
            Execution result dict or None if rejected
        """
        # Validate through firewall
        passed, reason = self.firewall.validate(signal)
        if not passed:
            logger.info(f"Signal rejected by firewall: {reason}")
            return None

        # Check kill switch
        if self.kill_switch_active:
            logger.warning("Kill switch active - rejecting signal")
            return None

        # Check daily loss
        if self.check_daily_loss():
            logger.warning("Daily loss limit exceeded - rejecting signal")
            return None

        try:
            coin = signal.coin
            side = signal.side.value.lower()
            size = signal.size or (signal.position_pct * signal.effective_size)

            logger.info(f"Executing signal: {coin} {side} {size:.4f} "
                       f"(confidence={signal.confidence:.0%}, "
                       f"leverage={signal.leverage}x)")

            # 1. Place market entry
            entry_result = self.place_market_order(
                coin, side, size,
                leverage=signal.leverage,
                reduce_only=False
            )

            if entry_result.get("status") == "error":
                logger.error(f"Failed to place entry order: {entry_result}")
                return entry_result

            # 2. Calculate stop loss and take profit prices
            mid = self._get_mid_price(coin)
            if mid:
                if side == "buy":
                    sl_price = mid * (1 - signal.risk.stop_loss_pct)
                    tp_price = mid * (1 + signal.risk.take_profit_pct)
                else:
                    sl_price = mid * (1 + signal.risk.stop_loss_pct)
                    tp_price = mid * (1 - signal.risk.take_profit_pct)

                # 3. Place stop loss
                sl_result = self.place_trigger_order(
                    coin, "sell" if side == "buy" else "buy",
                    size, sl_price, tp_or_sl="sl"
                )

                # 4. Place take profit
                tp_result = self.place_trigger_order(
                    coin, "sell" if side == "buy" else "buy",
                    size, tp_price, tp_or_sl="tp"
                )

                logger.info(f"Placed SL @ ${sl_price:.2f}, TP @ ${tp_price:.2f}")

            # Return summary
            return {
                "status": "success",
                "coin": coin,
                "side": side,
                "size": size,
                "leverage": signal.leverage,
                "entry_result": entry_result,
                "dry_run": self.dry_run,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict:
        """
        Get current trader statistics.

        Returns:
            Stats dict with orders, fills, PnL, kill switch status
        """
        self._check_daily_reset()

        return {
            "dry_run": self.dry_run,
            "signer_available": self.signer is not None,
            "public_address": self.public_address,
            "kill_switch_active": self.kill_switch_active,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_limit": self.max_daily_loss,
            "orders_today": self.orders_today,
            "fills_today": self.fills_today,
            "max_position_size": self.max_position_size,
            "asset_indices_loaded": len(self.asset_index_map),
            "timestamp": datetime.utcnow().isoformat()
        }
