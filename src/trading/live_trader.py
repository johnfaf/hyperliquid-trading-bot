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
  - Agent private key loaded from HL_AGENT_PRIVATE_KEY or secret manager
  - Trading account address loaded from HL_PUBLIC_ADDRESS
  - Agent signer wallet must be distinct from trading account wallet
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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
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
from src.core.secret_manager import SecretManagerError, load_agent_private_key
from src.signals.decision_firewall import DecisionFirewall
from src.signals.signal_schema import TradeSignal, signal_from_execution_dict

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

    # EIP-712 domain for Hyperliquid L1 mainnet
    # NOTE: Hyperliquid migrated from Arbitrum (42161) to its own L1.
    # Mainnet uses chainId 1337; testnet uses 421614.
    # Env override available for future chain changes.
    DOMAIN = {
        "name": "HyperliquidSignTransaction",
        "version": "1",
        "chainId": int(os.environ.get("HL_CHAIN_ID", 1337)),
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

            # Hyperliquid expects 32-byte hex components. Zero-pad r/s so
            # signatures remain valid when either integer has leading zeros.
            return {
                "r": f"0x{signed_message.r:064x}",
                "s": f"0x{signed_message.s:064x}",
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
                 max_daily_loss: float = 500, max_position_size: float = 1000,
                 max_order_usd: Optional[float] = None,
                 regime_forecaster: Optional[object] = None):
        """
        Initialize live trader.

        Args:
            firewall: DecisionFirewall instance for validation
            dry_run: If True, log what WOULD happen but don't execute (default True)
            max_daily_loss: Daily loss limit in USD (default $500)
            max_position_size: Max notional per position in USD (default $1000)
            max_order_usd: Hard cap on $ notional for any single live order.
                Acts as a safety net on top of max_position_size — even if the
                computed size would be larger, nothing above this value is sent
                to the exchange.  When None, falls back to config.LIVE_MAX_ORDER_USD
                and finally to max_position_size.
            regime_forecaster: Optional PredictiveRegimeForecaster for regime-aware position sizing
        """
        self.firewall = firewall
        self.live_requested = not dry_run
        self.dry_run = dry_run
        self.max_daily_loss = float(os.environ.get("HL_MAX_DAILY_LOSS", max_daily_loss))
        self.max_position_size = float(os.environ.get("HL_MAX_POSITION_SIZE", max_position_size))
        # Hard per-order $ cap (safety net during live bootstrap).
        env_max_order = os.environ.get("LIVE_MAX_ORDER_USD")
        if env_max_order is not None:
            self.max_order_usd = float(env_max_order)
        elif max_order_usd is not None:
            self.max_order_usd = float(max_order_usd)
        else:
            self.max_order_usd = float(
                getattr(config, "LIVE_MAX_ORDER_USD", self.max_position_size)
            )
        self.regime_forecaster = regime_forecaster
        self.status_reason = "dry_run_requested" if dry_run else "initializing"

        # API endpoints (must come early since _load_* methods need these)
        self.exchange_url = "https://api.hyperliquid.xyz/exchange"
        self.info_url = "https://api.hyperliquid.xyz/info"

        # Signer (loaded from env)
        self.signer = None
        self.agent_wallet_address = None
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

        # Track realized PnL from closed positions for daily loss enforcement
        self._last_known_positions: Dict[str, Dict] = {}  # coin -> position snapshot

        # Order idempotency: prevent duplicate orders from timeout/retry
        # Maps action_hash -> (timestamp, result) for recent orders
        self._recent_order_hashes: Dict[str, Tuple[float, Dict]] = {}
        self._ORDER_DEDUP_WINDOW = 30  # seconds to remember recent orders

        # Wallet balance tracking (last-known snapshots for dashboards/cycles).
        self._last_balance_snapshot: Dict[str, Optional[float]] = {
            "perps_margin": None,
            "spot_usdc": None,
            "total": None,
            "timestamp": None,
        }
        self._balance_log_interval_s = 300  # log balance at most once per 5 min
        self._last_balance_log_ts: float = 0.0

        logger.info(
            f"LiveTrader initialized: dry_run={dry_run}, "
            f"max_daily_loss=${self.max_daily_loss:.2f}, "
            f"max_position_size=${self.max_position_size:.2f}, "
            f"max_order_usd=${self.max_order_usd:.2f}"
        )

        if dry_run:
            logger.warning("DRY RUN MODE - No real trades will be executed")

        if not self.signer:
            logger.warning("No agent wallet signer configured - forcing dry_run mode")
            self.dry_run = True
            self.status_reason = "missing_agent_wallet_signer"
        elif self.live_requested and self.dry_run:
            self.status_reason = "dry_run_forced"
        elif self.live_requested:
            self.status_reason = "live_ready"
        else:
            self.status_reason = "dry_run_requested"

        # Reconcile positions on startup
        if not self.dry_run and self.signer:
            self.reconcile_positions()

    def _load_credentials(self):
        """
        Load agent-wallet signer credentials and trading account address.

        Security model:
        - Agent private key comes from secret manager or HL_AGENT_PRIVATE_KEY.
        - HL_PUBLIC_ADDRESS is the trading account (master/vault) to manage.
        - Agent signer address must differ from HL_PUBLIC_ADDRESS.
        """
        trading_address = os.environ.get("HL_PUBLIC_ADDRESS", "").strip()
        configured_agent_address = os.environ.get("HL_AGENT_WALLET_ADDRESS", "").strip()
        secret_provider = str(getattr(config, "SECRET_MANAGER_PROVIDER", "none")).lower().strip()

        if not trading_address:
            logger.warning("HL_PUBLIC_ADDRESS not set; live execution remains disabled")
            self.status_reason = "missing_trading_address"
            return
        if os.environ.get("HL_PRIVATE_KEY"):
            logger.error("HL_PRIVATE_KEY is disallowed in agent-wallet-only mode")
            self.status_reason = "legacy_private_key_blocked"
            return

        try:
            private_key = load_agent_private_key(secret_provider)
            if not private_key:
                logger.warning(
                    "No agent private key loaded (provider=%s). "
                    "Set HL_AGENT_PRIVATE_KEY or configure secret manager.",
                    secret_provider,
                )
                return

            self.signer = HyperliquidSigner(private_key)
            derived_agent_address = self.signer.address
            if configured_agent_address and configured_agent_address.lower() != derived_agent_address.lower():
                raise RuntimeError(
                    "HL_AGENT_WALLET_ADDRESS does not match signer address derived from secret"
                )
            if derived_agent_address.lower() == trading_address.lower():
                raise RuntimeError(
                    "Agent wallet must be different from HL_PUBLIC_ADDRESS trading wallet"
                )

            self.agent_wallet_address = configured_agent_address or derived_agent_address
            self.public_address = trading_address
            self.status_reason = "credentials_loaded"
            logger.info(
                "Agent-wallet credentials loaded (provider=%s agent=%s trading=%s)",
                secret_provider,
                self.agent_wallet_address,
                self.public_address,
            )
        except (SecretManagerError, Exception) as e:
            logger.error(f"Failed to load agent-wallet credentials: {e}")
            self.signer = None
            self.agent_wallet_address = None
            self.public_address = None
            self.status_reason = f"credential_error:{e}"

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

    def is_live_enabled(self) -> bool:
        """Return True when the operator explicitly requested live execution."""
        return bool(self.live_requested)

    def is_deployable(self) -> bool:
        """Return True when the trader can actually submit live orders."""
        return bool(self.live_requested and not self.dry_run and self.signer and self.public_address)

    def _coerce_signal(self, signal: Union[TradeSignal, Dict[str, Any]]) -> TradeSignal:
        """Accept either TradeSignal or execution dict for safer live mirroring."""
        if isinstance(signal, TradeSignal):
            return signal
        if isinstance(signal, dict):
            return signal_from_execution_dict(signal)
        raise TypeError(f"Unsupported signal type for live execution: {type(signal)}")

    def get_account_state(self) -> Dict[str, Any]:
        """Fetch raw clearinghouse state for the trading account."""
        if not self.public_address:
            return {}
        try:
            response = requests.post(
                self.info_url,
                json={"type": "clearinghouseState", "user": self.public_address},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error(f"Error getting account state: {e}")
            return {}

    def get_account_value(self) -> Optional[float]:
        """
        Best-effort extraction of live account value.

        Checks perps margin first (clearinghouseState), then spot wallet
        (spotClearinghouseState).  If USDC is in spot but not perps,
        logs a clear message telling the operator to transfer.
        """
        state = self.get_account_state()
        candidates = [
            state.get("marginSummary", {}).get("accountValue"),
            state.get("crossMarginSummary", {}).get("accountValue"),
            state.get("accountValue"),
        ]
        perps_value = None
        for value in candidates:
            try:
                if value is not None:
                    perps_value = float(value)
                    break
            except (TypeError, ValueError):
                continue

        if perps_value and perps_value > 0:
            return perps_value

        # Perps margin is $0 or unavailable — check spot wallet
        spot_usdc = self._get_spot_usdc_balance()
        if spot_usdc is not None and spot_usdc > 0:
            logger.warning(
                "Perps margin is $0 but spot wallet has $%.2f USDC. "
                "Transfer USDC from Spot to Perps in the Hyperliquid UI "
                "(Portfolio → Transfer → Spot to Perps) to enable live trading.",
                spot_usdc,
            )
            # Return 0.0 (not None) — the account exists, it's just unfunded for perps
            return 0.0

        # Both are empty/failed
        return perps_value  # 0.0 or None

    def _get_spot_usdc_balance(self) -> Optional[float]:
        """Fetch USDC balance from the spot wallet."""
        if not self.public_address:
            return None
        try:
            response = requests.post(
                self.info_url,
                json={"type": "spotClearinghouseState", "user": self.public_address},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            for bal in data.get("balances", []):
                if bal.get("coin") == "USDC":
                    return float(bal.get("total", 0) or 0)
            return 0.0
        except Exception as e:
            logger.debug("Error checking spot balance: %s", e)
            return None

    def snapshot_balance(self, log: bool = True) -> Dict[str, Optional[float]]:
        """
        Refresh the cached wallet balance snapshot (perps + spot + total).

        This is the canonical place the bot fetches and caches the live
        wallet state.  Called periodically from the fast cycle so the
        dashboard, logs, and audit trail all see a consistent view.

        Args:
            log: When True, emit an INFO log at most once every
                 self._balance_log_interval_s (default 5 min) so operators
                 can see balance updates without spamming the log.

        Returns:
            Dict with keys: perps_margin, spot_usdc, total, timestamp.
        """
        perps = None
        try:
            if self.public_address:
                response = requests.post(
                    self.info_url,
                    json={"type": "clearinghouseState", "user": self.public_address},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                margin_summary = data.get("marginSummary", {}) or {}
                perps = float(margin_summary.get("accountValue", 0) or 0)
        except Exception as e:
            logger.debug("snapshot_balance: perps fetch error: %s", e)
            perps = None

        spot = self._get_spot_usdc_balance()
        total: Optional[float]
        if perps is None and spot is None:
            total = None
        else:
            total = float(perps or 0) + float(spot or 0)

        now = time.time()
        self._last_balance_snapshot = {
            "perps_margin": perps,
            "spot_usdc": spot,
            "total": total,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if log and (now - self._last_balance_log_ts) >= self._balance_log_interval_s:
            self._last_balance_log_ts = now
            logger.info(
                "Wallet balance: perps=$%s, spot=$%s, total=$%s",
                f"{perps:.2f}" if perps is not None else "n/a",
                f"{spot:.2f}" if spot is not None else "n/a",
                f"{total:.2f}" if total is not None else "n/a",
            )

        return self._last_balance_snapshot

    def get_balance_snapshot(self) -> Dict[str, Optional[float]]:
        """Return the most recent balance snapshot (cached, no API call)."""
        return dict(self._last_balance_snapshot)

    def _normalize_position(self, pos: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten Hyperliquid position payload into a consistent dict shape."""
        pos_info = pos.get("position", pos) if isinstance(pos, dict) else {}
        size = float(pos_info.get("szi", pos_info.get("size", 0)) or 0)
        entry_px = float(pos_info.get("entryPx", pos_info.get("entry_price", 0)) or 0)
        unrealized = float(pos_info.get("unrealizedPnl", pos_info.get("unrealized_pnl", 0)) or 0)
        return {
            "coin": pos_info.get("coin", ""),
            "size": abs(size),
            "szi": size,
            "side": "long" if size > 0 else "short",
            "entry_price": entry_px,
            "entryPx": entry_px,
            "unrealized_pnl": unrealized,
            "unrealizedPnl": unrealized,
            "leverage": float(pos_info.get("leverage", pos_info.get("lev", 1)) or 1),
            "raw": pos,
        }

    def get_firewall_positions(self) -> List[Dict[str, Any]]:
        """Return normalized live positions in the format the firewall expects."""
        return self.get_positions()

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

        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"⚠️  Daily loss limit exceeded: ${abs(self.daily_pnl):.2f} > ${self.max_daily_loss:.2f}")
            self.kill_switch_active = True
            return True

        return False

    def reconcile_positions(self):
        """
        Compare exchange positions vs local state on startup.

        Logs discrepancies so the operator knows if the bot's view
        of the world differs from reality. Triggers kill switch if
        unexpected positions are found.
        """
        logger.info("Reconciling positions with exchange...")
        try:
            exchange_positions = self.get_positions()
            active_coins = []

            for pos in exchange_positions:
                coin = pos.get("coin", "")
                size = float(pos.get("szi", pos.get("size", 0)) or 0)
                entry_px = float(pos.get("entry_price", pos.get("entryPx", 0)) or 0)
                unrealized_pnl = float(pos.get("unrealized_pnl", pos.get("unrealizedPnl", 0)) or 0)

                if abs(size) > 0:
                    side = "long" if size > 0 else "short"
                    active_coins.append(coin)
                    self._last_known_positions[coin] = {
                        "coin": coin,
                        "side": side,
                        "size": abs(size),
                        "entry_price": entry_px,
                        "unrealized_pnl": unrealized_pnl,
                    }
                    logger.warning(
                        f"EXISTING POSITION found: {side.upper()} {coin} "
                        f"size={abs(size)} entry=${entry_px:,.2f} "
                        f"uPnL=${unrealized_pnl:+,.2f}"
                    )

            if active_coins:
                logger.warning(
                    f"Reconciliation: {len(active_coins)} open positions found on "
                    f"exchange: {', '.join(active_coins)}. These will be tracked."
                )
            else:
                logger.info("Reconciliation: no open positions on exchange (clean state)")

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
            logger.warning(
                "Could not verify exchange state on startup. "
                "Proceeding with caution — monitor positions manually."
            )

    def update_daily_pnl_from_fills(self):
        """
        Fetch recent fills from the exchange and update daily_pnl.

        This is the CRITICAL missing piece: without this, the daily loss
        circuit breaker never fires. Call this after every trade and
        periodically from the trading cycle.
        """
        self._check_daily_reset()

        try:
            if not self.public_address:
                return

            response = requests.post(
                self.info_url,
                json={"type": "userFills", "user": self.public_address},
                timeout=10,
            )
            response.raise_for_status()
            fills = response.json()

            if not isinstance(fills, list):
                return

            # Sum closed PnL from today's fills
            today = datetime.utcnow().strftime("%Y-%m-%d")
            today_pnl = 0.0
            for fill in fills:
                fill_time = fill.get("time", "")
                if isinstance(fill_time, (int, float)):
                    from datetime import timezone
                    fill_date = datetime.fromtimestamp(
                        fill_time / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                elif isinstance(fill_time, str):
                    fill_date = fill_time[:10]
                else:
                    continue

                if fill_date == today:
                    closed_pnl = float(fill.get("closedPnl", 0))
                    today_pnl += closed_pnl

            old_pnl = self.daily_pnl
            self.daily_pnl = today_pnl

            if abs(self.daily_pnl) > 0 and abs(self.daily_pnl - old_pnl) > 0.01:
                logger.info(f"Daily PnL updated: ${self.daily_pnl:+,.2f}")

            # Keep the firewall on the same realized-loss snapshot instead of
            # re-adding the full day's losses on every refresh.
            if hasattr(self.firewall, "set_daily_losses"):
                self.firewall.set_daily_losses(abs(min(self.daily_pnl, 0.0)))

            # Check if kill switch should trigger
            self.check_daily_loss()

        except Exception as e:
            logger.error(f"Failed to update daily PnL from fills: {e}")

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

    # Price sanity: track recent mids to detect garbage prices
    _price_history: Dict[str, float] = {}  # coin -> last known good mid

    def _validate_price(self, coin: str, price: float) -> bool:
        """
        Validate that a price is reasonable.

        Rejects:
        - Zero, negative, NaN, Infinity
        - Prices that deviate >10% from the last known good price
          (protects against corrupt API responses)

        Returns True if price is sane.
        """
        if not price or price <= 0:
            logger.error(f"PRICE REJECTED: {coin} price={price} (zero/negative)")
            return False

        import math
        if math.isnan(price) or math.isinf(price):
            logger.error(f"PRICE REJECTED: {coin} price={price} (NaN/Inf)")
            return False

        last_good = self._price_history.get(coin)
        if last_good:
            deviation = abs(price - last_good) / last_good
            if deviation > 0.10:  # >10% move since last check
                logger.error(
                    f"PRICE REJECTED: {coin} price=${price:,.2f} deviates "
                    f"{deviation:.1%} from last known ${last_good:,.2f}. "
                    f"Possible corrupt data — blocking order."
                )
                return False

        # Price is sane — update history
        self._price_history[coin] = price
        return True

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
        normalized_side = self._normalize_order_side(side)
        if normalized_side == "buy":
            return mid_price * (1 + slippage_pct)
        else:
            return mid_price * (1 - slippage_pct)

    @staticmethod
    def _normalize_order_side(side: str) -> str:
        """Normalize long/short or buy/sell inputs to buy/sell."""
        value = str(side or "").strip().lower()
        if value in {"buy", "long"}:
            return "buy"
        if value in {"sell", "short"}:
            return "sell"
        raise ValueError(f"Unsupported order side: {side}")

    @staticmethod
    def _is_order_result_success(result: Optional[Dict]) -> bool:
        """Best-effort classification of exchange responses into success/failure."""
        if not result:
            return False
        if not isinstance(result, dict):
            return bool(result)
        status = str(result.get("status", "")).strip().lower()
        if not status:
            # Hyperliquid returns {"status": "ok", "response": {...}} on success.
            # Missing/empty status means malformed response — treat as failure.
            logger.warning("Order result has no status field: %s", result)
            return False
        return status in {"success", "simulated", "verified", "filled", "accepted", "ok"}

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

        # ── Idempotency check: reject duplicate orders within dedup window ──
        now = time.time()
        # Evict expired entries
        expired = [h for h, (ts, _) in self._recent_order_hashes.items()
                   if now - ts > self._ORDER_DEDUP_WINDOW]
        for h in expired:
            del self._recent_order_hashes[h]

        if action_hash in self._recent_order_hashes:
            prev_ts, prev_result = self._recent_order_hashes[action_hash]
            age = now - prev_ts
            logger.warning(
                f"DUPLICATE ORDER blocked: action_hash={action_hash[:16]}... "
                f"(identical order placed {age:.1f}s ago)"
            )
            return {
                "status": "rejected",
                "reason": "duplicate_order",
                "original_result": prev_result,
                "age_seconds": round(age, 1),
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
            if self.public_address and self.signer:
                if self.public_address.lower() != self.signer.address.lower():
                    payload["vaultAddress"] = self.public_address

            logger.debug(f"Posting order to {self.exchange_url}: action_hash={action_hash}")

            response = requests.post(
                self.exchange_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # Record successful order for dedup
            self._recent_order_hashes[action_hash] = (now, result)

            logger.info(f"Order posted: {json.dumps(result)}")
            return result

        except requests.exceptions.Timeout:
            # CRITICAL: On timeout, the order may have been accepted on-chain.
            # Record the hash to BLOCK retries — caller must verify via fills.
            self._recent_order_hashes[action_hash] = (now, {"status": "timeout"})
            logger.error(
                f"Order TIMEOUT: action_hash={action_hash[:16]}... "
                f"Order may have executed on-chain. Blocking retries. "
                f"Verify position via get_positions() before retrying."
            )
            return {"status": "error", "message": "timeout — order may have executed, check positions"}

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
        side = self._normalize_order_side(side)

        # Kill switch and daily loss only block NEW positions (not closes).
        # reduce_only orders MUST always go through so the bot can exit
        # positions during emergencies and when protective orders fail.
        if not reduce_only:
            if self.kill_switch_active:
                logger.warning(f"Kill switch active - rejecting market order {coin} {side}")
                return {"status": "rejected", "reason": "kill_switch_active"}

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

        # Validate price sanity before proceeding
        if not self._validate_price(coin, mid):
            return {"status": "rejected", "reason": "price_sanity_failed",
                    "message": f"Price ${mid} for {coin} failed sanity check"}

        # Apply slippage
        price = self._apply_market_slippage(mid, side)

        # Position size limit only applies to new positions, not closes.
        # A reduce_only order can never increase exposure, so blocking it
        # would leave the position stuck open (un-closeable).
        notional = size * price
        if not reduce_only:
            if notional > self.max_position_size:
                logger.warning(f"Position size ${notional:.2f} exceeds limit ${self.max_position_size:.2f}")
                return {"status": "rejected", "reason": "position_size_exceeded"}
            # Hard $ cap on any single live order (bootstrap safety net).
            if self.max_order_usd and notional > self.max_order_usd:
                capped_size = self.max_order_usd / price
                logger.warning(
                    "place_market_order: capping %s %s size %.6f → %.6f "
                    "to honor LIVE_MAX_ORDER_USD=$%.2f (was $%.2f notional)",
                    coin, side, size, capped_size, self.max_order_usd, notional,
                )
                size = capped_size
                notional = size * price
                if size <= 0:
                    return {"status": "rejected", "reason": "order_cap_zero_size"}

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

        if self._is_order_result_success(result) and not self.dry_run:
            self.orders_today += 1

        return result

    def verify_fill(self, coin: str, expected_side: str, expected_size: float,
                    timeout: float = 10.0, poll_interval: float = 1.0) -> Optional[Dict]:
        """
        Poll exchange positions to verify an order actually filled.

        Args:
            coin: Expected coin
            expected_side: "buy" or "sell"
            expected_size: Expected position size change
            timeout: Max seconds to wait
            poll_interval: Seconds between polls

        Returns:
            Position dict if verified, None if not found after timeout
        """
        if self.dry_run:
            return {"status": "verified", "dry_run": True}

        deadline = time.time() + timeout
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            try:
                positions = self.get_positions()
                for pos in positions:
                    pos_coin = pos.get("coin", "")
                    pos_size = float(pos.get("szi", 0) or 0)

                    if pos_coin != coin or abs(pos_size) == 0:
                        continue

                    # Verify side matches (buy → positive szi, sell → negative)
                    if expected_side == "buy" and pos_size < 0:
                        continue
                    if expected_side == "sell" and pos_size > 0:
                        continue

                    fill_size = abs(pos_size)
                    if fill_size < expected_size * 0.5:
                        logger.warning(
                            f"Fill partial: {coin} got {abs(pos_size):.6f} "
                            f"vs expected {expected_size:.6f}"
                        )
                        continue

                    matched_size = min(fill_size, expected_size)
                    partial_fill = matched_size < (expected_size * 0.99)

                    logger.info(
                        f"Fill VERIFIED: {coin} size={pos_size} "
                        f"(expected={expected_size:.6f}, attempt {attempt}, "
                        f"{time.time() - (deadline - timeout):.1f}s)"
                    )
                    return {
                        "status": "verified",
                        "coin": coin,
                        "size": matched_size,
                        "position_size": pos_size,
                        "partial_fill": partial_fill,
                        "attempt": attempt,
                    }
            except Exception as e:
                logger.debug(f"Fill verification poll error: {e}")

            time.sleep(poll_interval)

        logger.warning(
            f"Fill NOT verified after {timeout}s: {coin} {expected_side} {expected_size}"
        )
        return None

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
        side = self._normalize_order_side(side)

        # Kill switch and position size limit only block NEW positions.
        # reduce_only orders MUST go through so the bot can exit positions.
        if not reduce_only:
            if self.kill_switch_active:
                logger.warning(f"Kill switch active - rejecting limit order {coin}")
                return {"status": "rejected", "reason": "kill_switch_active"}

        asset_idx = self._get_asset_index(coin)
        if asset_idx is None:
            return {"status": "error", "message": f"Unknown coin: {coin}"}

        # Position size check only for new positions
        notional = size * price
        if not reduce_only:
            if notional > self.max_position_size:
                logger.warning(f"Position size ${notional:.2f} exceeds limit")
                return {"status": "rejected", "reason": "position_size_exceeded"}
            # Hard $ cap on any single live order (bootstrap safety net).
            if self.max_order_usd and notional > self.max_order_usd:
                capped_size = self.max_order_usd / price
                logger.warning(
                    "place_limit_order: capping %s %s size %.6f → %.6f "
                    "to honor LIVE_MAX_ORDER_USD=$%.2f (was $%.2f notional)",
                    coin, side, size, capped_size, self.max_order_usd, notional,
                )
                size = capped_size
                notional = size * price
                if size <= 0:
                    return {"status": "rejected", "reason": "order_cap_zero_size"}

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

        logger.info(f"Placing limit order: {coin} {side} {size} @ ${price:.2f} (notional: ${notional:.2f})")
        result = self._post_order(action)

        if self._is_order_result_success(result) and not self.dry_run:
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
        side = self._normalize_order_side(side)

        # Validate trigger price
        import math
        if not trigger_price or trigger_price <= 0 or math.isnan(trigger_price) or math.isinf(trigger_price):
            logger.error("Invalid trigger price for %s %s: %s", coin, tp_or_sl, trigger_price)
            return {"status": "error", "message": f"Invalid trigger price: {trigger_price}"}

        asset_idx = self._get_asset_index(coin)
        if asset_idx is None:
            return {"status": "error", "message": f"Unknown coin: {coin}"}

        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": "0",  # Trigger price is in trigger, not order price
            "s": str(round(size, 8)),
            "r": True,  # SL/TP must be reduce_only to close the position
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
            side = "sell" if pos.get("szi", 0) > 0 else "buy"

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

            data = self.get_account_state()
            positions = data.get("assetPositions", []) if isinstance(data, dict) else []
            normalized = [self._normalize_position(pos) for pos in positions]
            return [pos for pos in normalized if pos.get("coin")]

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

    def apply_regime_overlay(self, signal: TradeSignal, regime_data: Optional[Dict] = None) -> TradeSignal:
        """
        Apply regime-based position sizing and risk adjustments to a signal.

        Fetches regime prediction from self.regime_forecaster if available,
        or uses provided regime_data. Applies dynamic sizing and stop loss adjustments.

        Args:
            signal: TradeSignal to modify
            regime_data: Optional pre-computed regime data dict with 'regime' and 'confidence' keys

        Returns:
            Modified signal with adjusted size and risk parameters
        """
        regime_info = None

        # Fetch regime data from forecaster if available
        if self.regime_forecaster and not regime_data:
            try:
                regime_info = self.regime_forecaster.predict_regime(signal.coin)
            except Exception as e:
                logger.debug(f"Failed to fetch regime data for {signal.coin}: {e}")
        elif regime_data:
            regime_info = regime_data

        if not regime_info:
            return signal

        regime = regime_info.get("regime", "neutral")
        confidence = regime_info.get("confidence", 0.0)

        # CRIT-FIX CRIT-4 (continued): remove the position_pct * effective_size fallback.
        # execute_signal now guarantees signal.size > 0 before calling this method.
        # Use signal.size directly; log an error if somehow it's still unset here.
        base_size = signal.size or 0.0
        if base_size <= 0:
            logger.error(
                f"apply_regime_overlay: signal.size not set for {signal.coin} — "
                f"skipping size adjustment (execute_signal should have computed it first)"
            )
            return signal

        # 1. CRASH regime: reduce size 60%, tighten stop loss to 3%
        if regime == "crash" and confidence > 0.4:
            signal.size = base_size * 0.4  # 60% reduction = multiply by 0.4
            signal.risk.stop_loss_pct = 0.03
            logger.warning(
                f"REGIME OVERLAY: crash detected (conf={confidence:.2f}), "
                f"reducing size 60%, tightening SL to 3% for {signal.coin}"
            )

        # 2. VOLATILE regime: reduce size 30%, widen stop loss to 8%
        elif regime == "volatile":
            signal.size = base_size * 0.7  # 30% reduction = multiply by 0.7
            signal.risk.stop_loss_pct = 0.08
            logger.info(
                f"REGIME OVERLAY: volatile detected, "
                f"reducing size 30%, widening SL to 8% for {signal.coin}"
            )

        # 3. BULLISH regime: allow full size, boost by 10%
        elif regime == "bullish" and confidence > 0.6:
            signal.size = base_size * 1.1  # 10% boost
            logger.info(
                f"REGIME OVERLAY: bullish confirmed (conf={confidence:.2f}), "
                f"boosting size 10% for {signal.coin}"
            )

        return signal

    def _apply_order_usd_cap(self, signal: TradeSignal) -> Optional[TradeSignal]:
        """
        Clamp signal.size so its notional never exceeds self.max_order_usd.

        This is a *hard* safety net for the live-trading bootstrap phase.
        Even when paper sizing, rescaling, or regime overlay produce a larger
        size, nothing above max_order_usd ($3 by default) is sent to the
        exchange.  If the required minimum coin quantity would round to zero,
        the trade is dropped entirely (returns None).

        Args:
            signal: TradeSignal with signal.size already computed.

        Returns:
            Same signal (possibly with shrunken size), or None if the cap
            would require a zero-sized order.
        """
        if self.max_order_usd is None or self.max_order_usd <= 0:
            return signal

        size = float(signal.size or 0)
        if size <= 0:
            return signal

        mid = self._get_mid_price(signal.coin)
        if not mid or mid <= 0:
            # Cannot evaluate notional — err on the side of caution and drop
            logger.warning(
                "Cannot apply $%.2f cap to %s: mid price unavailable — dropping trade",
                self.max_order_usd,
                signal.coin,
            )
            return None

        notional = size * mid
        if notional <= self.max_order_usd:
            return signal

        capped_size = self.max_order_usd / mid
        if capped_size <= 0:
            logger.warning(
                "Order cap $%.2f produces zero size for %s @ $%.4f — dropping trade",
                self.max_order_usd,
                signal.coin,
                mid,
            )
            return None

        logger.info(
            "Capping %s size to honor LIVE_MAX_ORDER_USD=$%.2f: "
            "%.6f → %.6f (notional $%.2f → $%.2f)",
            signal.coin,
            self.max_order_usd,
            size,
            capped_size,
            notional,
            capped_size * mid,
        )
        signal.size = capped_size
        return signal

    def execute_signal(
        self,
        signal: Union[TradeSignal, Dict[str, Any]],
        bypass_firewall: bool = False,
    ) -> Optional[Dict]:
        """
        Main entry point: execute a trade signal through firewall.

        Pipeline:
          1. Validate signal through DecisionFirewall (unless bypassed)
          2. Check kill switch and daily loss
          3. Place market entry order
          4. Place stop loss trigger
          5. Place take profit trigger
          6. Return execution result

        Args:
            signal: TradeSignal object
            bypass_firewall: If True, skip DecisionFirewall validation.  This
                is used by the paper→live mirror path — the paper trade has
                already passed the firewall, so re-running it would always
                fail on the cooldown check ("COIN traded Ns ago") because the
                paper execution just happened.  Kill-switch, daily loss, and
                per-order caps still apply.

        Returns:
            Execution result dict or None if rejected
        """
        signal = self._coerce_signal(signal)

        if not bypass_firewall:
            live_positions = self.get_firewall_positions() if self.is_deployable() else None
            live_account_value = self.get_account_value() if self.is_deployable() else None

            # Validate through firewall
            passed, reason = self.firewall.validate(
                signal,
                open_positions=live_positions,
                account_balance=live_account_value,
            )
            if not passed:
                logger.info(f"Signal rejected by firewall: {reason}")
                return None
        else:
            logger.debug(
                "Firewall bypass active for %s (mirror path — paper trade already validated)",
                signal.coin,
            )

        # Check kill switch
        if self.kill_switch_active:
            logger.warning("Kill switch active - rejecting signal")
            return None

        # Check daily loss
        if self.check_daily_loss():
            logger.warning("Daily loss limit exceeded - rejecting signal")
            return None

        # CRIT-FIX CRIT-4: resolve canonical coin quantity BEFORE the regime overlay
        # so the overlay can apply a correct proportional multiplier.  The original
        # fallback `position_pct * effective_size` = `position_pct²` which produced
        # micro-orders (~0.48% of intended size) silently posted to the exchange.
        coin = signal.coin
        signal_side = signal.side.value.lower()
        entry_side = self._normalize_order_side(signal_side)
        side = entry_side

        if not (signal.size and signal.size > 0):
            # Compute coin quantity from USD notional: position_pct × max_position_size / mid
            mid = self._get_mid_price(coin)
            if not mid or mid <= 0:
                logger.warning(
                    f"Cannot compute order size for {coin}: no mid price available — skipping"
                )
                return None
            position_usd = signal.position_pct * self.max_position_size
            signal.size = position_usd / mid
            logger.debug(
                f"Computed size for {coin}: ${position_usd:.2f} / ${mid:.4f} = {signal.size:.6f} coins"
            )

        # Apply regime overlay for dynamic position sizing (signal.size is now set)
        signal = self.apply_regime_overlay(signal)

        # Hard per-order $ cap — applied AFTER regime overlay and any paper→live
        # rescaling so nothing above max_order_usd ever hits the exchange.
        capped_signal = self._apply_order_usd_cap(signal)
        if capped_signal is None:
            return None
        signal = capped_signal

        try:
            size = signal.size

            if not size or size <= 0:
                logger.warning(f"Calculated size is 0 or negative for {coin} — skipping")
                return None

            logger.info(f"Executing signal: {coin} {signal_side} {size:.4f} "
                       f"(confidence={signal.confidence:.0%}, "
                       f"leverage={signal.leverage}x)")

            # 1. Place market entry
            entry_result = self.place_market_order(
                coin, entry_side, size,
                leverage=signal.leverage,
                reduce_only=False
            )

            if not self._is_order_result_success(entry_result):
                logger.error(f"Failed to place entry order: {entry_result}")
                return entry_result

            # Update daily PnL after every execution to keep loss tracking current
            self.update_daily_pnl_from_fills()

            actual_fill_size = size

            # 1b. Verify the fill actually happened before placing SL/TP
            if not self.dry_run:
                fill_check = self.verify_fill(coin, entry_side, size, timeout=10.0)
                if not fill_check:
                    logger.error(
                        f"FILL NOT VERIFIED for {coin} {side} {size} — "
                        f"skipping SL/TP placement. Manual intervention required."
                    )
                    return {
                        "status": "warning",
                        "message": "order posted but fill not verified",
                        "coin": coin,
                        "entry_result": entry_result,
                    }
                actual_fill_size = float(fill_check.get("size", size) or size)
                if actual_fill_size <= 0:
                    logger.error(
                        "Fill verification returned invalid size for %s: %s",
                        coin,
                        fill_check,
                    )
                    close_result = self.close_position(coin)
                    return {
                        "status": "error",
                        "message": "invalid_fill_size",
                        "coin": coin,
                        "entry_result": entry_result,
                        "close_result": close_result,
                    }
                if fill_check.get("partial_fill"):
                    logger.warning(
                        "Partial fill accepted for %s: protecting %.6f of requested %.6f",
                        coin,
                        actual_fill_size,
                        size,
                    )

            # 2. Calculate stop loss and take profit prices
            mid = self._get_mid_price(coin)
            if not mid:
                logger.error(
                    "Cannot place SL/TP for %s: mid price unavailable. "
                    "Position is UNPROTECTED — closing immediately.", coin
                )
                close_result = None
                if not self.dry_run:
                    close_result = self.close_position(coin)
                return {
                    "status": "error",
                    "message": "mid_price_unavailable_for_sl_tp",
                    "coin": coin,
                    "entry_result": entry_result,
                    "close_result": close_result,
                }

            # mid is guaranteed non-None here (early return above)
            if side == "buy":
                sl_price = mid * (1 - signal.risk.stop_loss_pct)
                tp_price = mid * (1 + signal.risk.take_profit_pct)
            else:
                sl_price = mid * (1 + signal.risk.stop_loss_pct)
                tp_price = mid * (1 - signal.risk.take_profit_pct)

            # 3. Place stop loss
            sl_result = self.place_trigger_order(
                coin, "sell" if side == "buy" else "buy",
                actual_fill_size, sl_price, tp_or_sl="sl"
            )

            # 4. Place take profit
            tp_result = self.place_trigger_order(
                coin, "sell" if side == "buy" else "buy",
                actual_fill_size, tp_price, tp_or_sl="tp"
            )

            if not self._is_order_result_success(sl_result) or not self._is_order_result_success(tp_result):
                logger.error(
                    "Protective order placement failed for %s: sl=%s tp=%s",
                    coin,
                    sl_result,
                    tp_result,
                )
                close_result = None
                if not self.dry_run:
                    close_result = self.close_position(coin)
                return {
                    "status": "error",
                    "message": "protective_order_failed",
                    "coin": coin,
                    "entry_result": entry_result,
                    "stop_loss_result": sl_result,
                    "take_profit_result": tp_result,
                    "close_result": close_result,
                }

            logger.info(f"Placed SL @ ${sl_price:.2f}, TP @ ${tp_price:.2f}")

            # Return summary
            return {
                "status": "success",
                "coin": coin,
                "side": signal_side,
                "size": actual_fill_size,
                "requested_size": size,
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
            "live_enabled": self.live_requested,
            "deployable": self.is_deployable(),
            "dry_run": self.dry_run,
            "signer_available": self.signer is not None,
            "agent_wallet_address": self.agent_wallet_address,
            "public_address": self.public_address,
            "status_reason": self.status_reason,
            "kill_switch_active": self.kill_switch_active,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_limit": self.max_daily_loss,
            "orders_today": self.orders_today,
            "fills_today": self.fills_today,
            "max_position_size": self.max_position_size,
            "max_order_usd": self.max_order_usd,
            "wallet_balance": dict(self._last_balance_snapshot),
            "asset_indices_loaded": len(self.asset_index_map),
            "timestamp": datetime.utcnow().isoformat()
        }
