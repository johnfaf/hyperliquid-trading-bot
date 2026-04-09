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
import copy
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import requests

# Try importing eth_account; if unavailable, set flag
try:
    from eth_account import Account
    try:
        # eth_account >= 0.8 provides encode_typed_data
        from eth_account.messages import encode_typed_data as _encode_typed_data
        _USE_TYPED_DATA = True
    except ImportError:  # pragma: no cover - fallback for very old eth_account
        from eth_account.messages import encode_structured_data as _encode_typed_data  # type: ignore
        _USE_TYPED_DATA = False
    HAS_ETH_ACCOUNT = True
except ImportError:
    HAS_ETH_ACCOUNT = False
    _USE_TYPED_DATA = False

try:
    import msgpack  # type: ignore
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    from eth_utils import keccak as _keccak  # type: ignore
    HAS_KECCAK = True
except ImportError:
    HAS_KECCAK = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.core.api_manager import Priority, get_manager
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


# ────────────────────────────────────────────────────────────────────────
# Hyperliquid wire format helpers
# ────────────────────────────────────────────────────────────────────────
#
# Hyperliquid enforces strict rules on the string form of price and size
# fields in the signed action payload.  Violating them produces
# ``{"error": "Order has invalid price."}`` — the outer response is still
# ``{"status": "ok"}`` so the order looks accepted but never fills.
#
#   Prices (perps):
#     • Integer prices are always allowed.
#     • Otherwise: max 5 SIGNIFICANT FIGURES and max (6 - szDecimals)
#       decimal places.
#     • Canonical form: no trailing zeros (Decimal.normalize).
#
#   Sizes (perps):
#     • Rounded to the asset's szDecimals.
#     • Canonical form: no trailing zeros.
#
# These helpers mirror ``float_to_wire`` / ``price_to_wire`` in the
# official ``hyperliquid-python-sdk`` so we emit byte-identical payloads.

def _hl_normalize_decimal(value: Decimal) -> str:
    """Render a Decimal in canonical (no trailing zeros) string form."""
    # Decimal("0.100").normalize() → Decimal("0.1"), but "1E+2" for "100".
    # Use format string to force plain-notation for large integers.
    normalized = value.normalize()
    as_str = format(normalized, "f")
    if as_str == "-0":
        return "0"
    return as_str


def _hl_format_price(price: float, sz_decimals: int) -> str:
    """
    Format a price for the Hyperliquid wire protocol.

    Applies the 5-significant-figure limit and the
    ``max_decimals = 6 - sz_decimals`` perps constraint, then renders in
    canonical no-trailing-zeros form.  Returns ``"0"`` for non-positive
    or non-finite input (trigger orders use "0" as a placeholder).
    """
    try:
        px = float(price)
    except (TypeError, ValueError):
        return "0"
    if not px or px != px or px in (float("inf"), float("-inf")):
        return "0"
    if px <= 0:
        return "0"

    # Integer prices are always allowed and skip the 5-sig-fig round.
    if px == int(px):
        return _hl_normalize_decimal(Decimal(int(px)))

    # 1. Round to 5 significant figures.
    sig_fig_rounded = float(f"{px:.5g}")

    # 2. Clamp to max allowed decimal places for perps.
    max_decimals = max(0, 6 - int(sz_decimals or 0))
    final = round(sig_fig_rounded, max_decimals)

    # 3. Canonical string via Decimal normalize.
    return _hl_normalize_decimal(Decimal(repr(final)))


def _hl_format_size(size: float, sz_decimals: int) -> str:
    """
    Format a size for the Hyperliquid wire protocol.

    Rounds to the asset's ``szDecimals`` and renders in canonical
    no-trailing-zeros form.
    """
    try:
        sz = float(size)
    except (TypeError, ValueError):
        return "0"
    if not sz or sz != sz or sz in (float("inf"), float("-inf")):
        return "0"

    decimals = max(0, int(sz_decimals or 0))
    rounded = round(sz, decimals)
    # Avoid "-0" for very small negative rounding artifacts.
    if rounded == 0:
        return "0"
    return _hl_normalize_decimal(Decimal(repr(rounded)))


class HyperliquidSigner:
    """
    Signs L1 actions for the Hyperliquid exchange.

    Hyperliquid uses a specific signing scheme that is NOT just an EIP-712
    wrap of the raw action JSON.  The canonical steps (matching the
    official ``hyperliquid-python-sdk``) are:

      1. ``action_bytes = msgpack.packb(action)``
      2. Append 8-byte big-endian nonce.
      3. Append vault-address flag byte + 20-byte vault address, or a single
         ``\\x00`` byte if no vault is used (the "expiresAfter" flag is a
         second optional byte, appended after the vault block).
      4. ``connection_id = keccak256(bytes)`` (this is a bytes32 value).
      5. EIP-712 sign a struct::

             Agent { source: string, connectionId: bytes32 }

         under domain::

             { name: "Exchange", version: "1",
               chainId: 1337, verifyingContract: 0x0000…0000 }

      6. ``source`` is ``"a"`` on mainnet and ``"b"`` on testnet.

    Historical bug: an earlier version of this class signed
    ``{ "action": json.dumps(action), "nonce": nonce }`` under a made-up
    ``HyperliquidTransaction`` struct.  Hyperliquid's verifier could not
    reproduce that hash, so ``ecrecover`` returned a garbage address on
    every order and the exchange replied ``"User or API Wallet 0x... does
    not exist"``.  See commit history for details.
    """

    # Signing-domain chain ID is ALWAYS 1337 on mainnet and 421614 on
    # testnet regardless of where orders are routed.  This is distinct from
    # Arbitrum/native L1 chain IDs.  Override via HL_CHAIN_ID only if
    # Hyperliquid changes the signing domain.
    CHAIN_ID = int(os.environ.get("HL_CHAIN_ID", 1337))
    DOMAIN = {
        "name": "Exchange",
        "version": "1",
        "chainId": CHAIN_ID,
        "verifyingContract": "0x0000000000000000000000000000000000000000",
    }
    # "a" = mainnet, "b" = testnet.  Hyperliquid uses the signing chain ID
    # 1337 for mainnet and 421614 for testnet, so derive source from it.
    SOURCE = "a" if CHAIN_ID == 1337 else "b"

    def __init__(self, private_key: str):
        """
        Initialize signer with an Ethereum private key.

        Args:
            private_key: Hex string (with or without ``0x`` prefix)
        """
        if not HAS_ETH_ACCOUNT:
            raise RuntimeError(
                "eth_account library not installed. "
                "Please install: pip install eth_account"
            )
        if not HAS_MSGPACK:
            raise RuntimeError(
                "msgpack library not installed. Hyperliquid L1 action signing "
                "requires msgpack to encode actions canonically. "
                "Please install: pip install msgpack"
            )
        if not HAS_KECCAK:
            raise RuntimeError(
                "eth_utils keccak not available — required for Hyperliquid "
                "action hashing.  Reinstall eth_account to pull eth_utils."
            )

        # Ensure 0x prefix
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self.account = Account.from_key(private_key)
        self.address = self.account.address
        masked = f"{self.address[:6]}...{self.address[-4:]}" if self.address else "unknown"
        logger.info("HyperliquidSigner initialized with address: %s", masked)

    @staticmethod
    def _action_hash(action: Dict, vault_address: Optional[str], nonce: int,
                     expires_after: Optional[int] = None) -> bytes:
        """
        Compute the Hyperliquid L1 ``connectionId`` for an action.

        This mirrors ``hyperliquid.utils.signing.action_hash`` in the
        official Python SDK.
        """
        data = msgpack.packb(action)
        data += nonce.to_bytes(8, "big")
        if vault_address is None:
            data += b"\x00"
        else:
            data += b"\x01"
            data += bytes.fromhex(vault_address.removeprefix("0x"))
        if expires_after is not None:
            data += b"\x00"
            data += expires_after.to_bytes(8, "big")
        return _keccak(data)

    def sign_action(self, action: Dict, nonce: int,
                    vault_address: Optional[str] = None,
                    expires_after: Optional[int] = None) -> Dict:
        """
        Sign an L1 action (order, cancel, modify, etc.) for Hyperliquid.

        Args:
            action: The action dict exactly as it will be sent to
                ``/exchange`` in the request body ``"action"`` field.
            nonce: Millisecond timestamp used as nonce.
            vault_address: When trading on behalf of another account (agent
                wallet mode), pass the *trading account* address here.  It
                MUST be baked into the signed hash or Hyperliquid will
                recover a different address than the signer's.
            expires_after: Optional expiry timestamp (ms since epoch).

        Returns:
            ``{"r": "0x…", "s": "0x…", "v": int}`` — zero-padded to 32
            bytes each so signatures with leading zeros remain valid.
        """
        try:
            connection_id = self._action_hash(
                action, vault_address, nonce, expires_after,
            )

            payload = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    "Agent": [
                        {"name": "source", "type": "string"},
                        {"name": "connectionId", "type": "bytes32"},
                    ],
                },
                "primaryType": "Agent",
                "domain": self.DOMAIN,
                "message": {
                    "source": self.SOURCE,
                    "connectionId": connection_id,
                },
            }

            if _USE_TYPED_DATA:
                message = _encode_typed_data(full_message=payload)
            else:  # pragma: no cover
                message = _encode_typed_data(payload)
            signed_message = self.account.sign_message(message)

            # Hyperliquid expects 32-byte hex components. Zero-pad r/s so
            # signatures with leading zeros remain valid.
            return {
                "r": f"0x{signed_message.r:064x}",
                "s": f"0x{signed_message.s:064x}",
                "v": signed_message.v,
            }
        except Exception as e:
            logger.error(f"Error signing action: {e}")
            raise

    @staticmethod
    def get_action_hash(action: Dict) -> str:
        """Compute a stable hash of an action for dedup/auditing.

        This is NOT the signing hash — it is only used locally for order
        deduplication.  It intentionally ignores nonce and vault so that two
        functionally-identical orders collide in the dedup cache.
        """
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
        # Exchange-enforced minimum notional per order.  Hyperliquid silently
        # drops any order below $10 — we keep a small buffer (default $11)
        # so rounding and price drift do not push us under the floor.
        self.min_order_usd = float(
            getattr(config, "LIVE_MIN_ORDER_USD", 11.0)
        )
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
        # Guard: the exchange enforces a hard minimum notional, so any cap
        # below that would make live trading physically impossible (every
        # order would be dropped by the matching engine, then fail fill
        # verification).  Raise the cap to the floor and warn loudly.
        if self.max_order_usd < self.min_order_usd:
            logger.warning(
                "LIVE_MAX_ORDER_USD=$%.2f is below Hyperliquid's $%.2f minimum "
                "notional per order.  Raising cap to $%.2f so orders can "
                "actually execute.  Set LIVE_MAX_ORDER_USD to a higher value "
                "on Railway to give the bot more headroom.",
                self.max_order_usd,
                self.min_order_usd,
                self.min_order_usd,
            )
            self.max_order_usd = self.min_order_usd

        # Canary rollout guardrails for live deployment.
        self.canary_mode = bool(getattr(config, "LIVE_CANARY_MODE", False))
        self.canary_max_order_usd = float(
            getattr(config, "LIVE_CANARY_MAX_ORDER_USD", self.max_order_usd)
        )
        self.canary_max_signals_per_day = int(
            getattr(config, "LIVE_CANARY_MAX_SIGNALS_PER_DAY", 25)
        )
        if self.canary_mode and self.canary_max_order_usd > 0:
            self.max_order_usd = max(
                self.min_order_usd,
                min(self.max_order_usd, self.canary_max_order_usd),
            )
            logger.warning(
                "LIVE_CANARY_MODE enabled: max_order_usd tightened to $%.2f",
                self.max_order_usd,
            )

        # Optional per-source/day entry cap (0 disables this guardrail).
        self.max_orders_per_source_per_day = int(
            getattr(config, "LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 0)
        )

        # Optional external kill-switch controls.
        self.external_kill_switch_file = str(
            getattr(config, "LIVE_EXTERNAL_KILL_SWITCH_FILE", "")
        ).strip()
        self.external_kill_switch_env = "LIVE_EXTERNAL_KILL_SWITCH"
        self._kill_switch_reason: str = ""
        self.regime_forecaster = regime_forecaster
        self.status_reason = "dry_run_requested" if dry_run else "initializing"

        # API endpoints (must come early since _load_* methods need these)
        self.exchange_url = config.HYPERLIQUID_EXCHANGE_URL
        self.info_url = config.HYPERLIQUID_INFO_URL
        self.api_manager = get_manager()

        # Signer (loaded from env)
        self.signer = None
        self.agent_wallet_address = None
        self.public_address = None
        self._load_credentials()

        # Asset index + szDecimals mapping (BTC=0/szDecimals=5, ETH=1/4, etc.)
        # szDecimals drives price rounding: perps allow max
        # (6 - szDecimals) decimal places, so we MUST know it per-coin to
        # emit valid wire-format prices/sizes.
        self.asset_index_map: Dict[str, int] = {}
        self.sz_decimals_map: Dict[str, int] = {}
        self._load_asset_index_map()

        # State tracking
        self.daily_pnl = 0.0
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = 0.0
        self.daily_reset_date = ""
        self.kill_switch_active = False
        self.orders_today = 0
        self.fills_today = 0
        self._source_orders_today: Dict[str, int] = defaultdict(int)

        # Track realized PnL from closed positions for daily loss enforcement
        self._last_known_positions: Dict[str, Dict] = {}  # coin -> position snapshot

        # Order idempotency: prevent duplicate orders from timeout/retry
        # Maps action_hash -> (timestamp, result) for recent orders
        self._recent_order_hashes: Dict[str, Tuple[float, Dict]] = {}
        self._ORDER_DEDUP_WINDOW = max(
            1.0,
            float(os.environ.get("LIVE_ORDER_DEDUP_WINDOW_S", "30")),
        )
        self._last_hash_cleanup_ts = 0.0
        self._HASH_CLEANUP_INTERVAL = 60.0  # periodic cleanup every 60s

        # Price sanity state (instance-level; class-level cache leaks across traders).
        self._price_history: Dict[str, float] = {}  # coin -> last known good mid
        self._price_seed_candidates: Dict[str, float] = {}  # coin -> untrusted first sample

        # Fill verification defaults:
        # - verify_fill() API remains blocking by default for backward compatibility.
        # - execute_signal() uses a separate non-blocking default to avoid cycle stalls.
        self._fill_verify_blocking = os.environ.get(
            "LIVE_FILL_VERIFY_BLOCKING", "true"
        ).strip().lower() in {"1", "true", "yes"}
        # execute_signal uses this mode explicitly (default non-blocking).
        self._execute_fill_verify_blocking = os.environ.get(
            "LIVE_EXECUTE_FILL_VERIFY_BLOCKING", "false"
        ).strip().lower() in {"1", "true", "yes"}
        self._fill_verify_timeout_s = max(
            0.0,
            float(os.environ.get("LIVE_FILL_VERIFY_TIMEOUT_S", "10.0")),
        )
        self._fill_verify_poll_s = max(
            0.1,
            float(os.environ.get("LIVE_FILL_VERIFY_POLL_S", "1.0")),
        )

        # Emergency exit retries for transient API/network failures.
        self._emergency_close_retries = max(
            1,
            int(os.environ.get("LIVE_EMERGENCY_CLOSE_RETRIES", "3")),
        )
        self._emergency_close_retry_delay_s = max(
            0.1,
            float(os.environ.get("LIVE_EMERGENCY_CLOSE_RETRY_DELAY_S", "1.5")),
        )

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
        - HL_AGENT_WALLET_ADDRESS, when set, is the AUTHORITATIVE expected
          address of the agent wallet approved on Hyperliquid.  The private
          key loaded from secrets MUST derive to this address.  If it does
          not, the mismatch is logged prominently and live execution is
          disabled — this prevents silently submitting orders signed by the
          wrong key (which Hyperliquid rejects with
          "User or API Wallet 0x... does not exist").
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

            # HL_AGENT_WALLET_ADDRESS is authoritative when set.  Validate that
            # the private key actually corresponds to the expected wallet.
            if configured_agent_address:
                if configured_agent_address.lower() != derived_agent_address.lower():
                    logger.error(
                        "AGENT WALLET MISMATCH: HL_AGENT_WALLET_ADDRESS=%s but "
                        "HL_AGENT_PRIVATE_KEY derives address %s. The private key "
                        "does not belong to the configured agent wallet. Update "
                        "HL_AGENT_PRIVATE_KEY on Railway to the key for %s, or "
                        "update HL_AGENT_WALLET_ADDRESS to %s. Live trading DISABLED.",
                        configured_agent_address,
                        derived_agent_address,
                        configured_agent_address,
                        derived_agent_address,
                    )
                    self.signer = None
                    self.agent_wallet_address = configured_agent_address
                    self.public_address = None
                    self.status_reason = "agent_wallet_address_mismatch"
                    return
                logger.info(
                    "HL_AGENT_WALLET_ADDRESS matches signer-derived address (%s)",
                    configured_agent_address,
                )
            else:
                logger.warning(
                    "HL_AGENT_WALLET_ADDRESS not set on Railway. Using signer-derived "
                    "address %s. Set HL_AGENT_WALLET_ADDRESS to enable startup "
                    "validation and catch private-key/wallet mismatches before trading.",
                    derived_agent_address,
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

            # Verify the agent wallet is actually approved on Hyperliquid.
            # Catches the "User or API Wallet 0x... does not exist" case at
            # startup instead of at first order placement.
            self._verify_agent_wallet_registered()
        except (SecretManagerError, Exception) as e:
            logger.error(f"Failed to load agent-wallet credentials: {e}")
            self.signer = None
            self.agent_wallet_address = None
            self.public_address = None
            self.status_reason = f"credential_error:{e}"

    def _verify_agent_wallet_registered(self) -> None:
        """
        Check with Hyperliquid whether the loaded agent wallet is actually
        registered as an API wallet for HL_PUBLIC_ADDRESS.

        Queries the `extraAgents` field on the user's clearinghouse state —
        each registered API wallet appears there by address.  If the signer
        address is not in that list, log a prominent warning telling the
        operator to approve the wallet in the Hyperliquid UI before trading.
        This is best-effort: on network errors or unexpected payloads we log
        a debug message and continue (the order path will still surface the
        "User or API Wallet does not exist" error if the check is wrong).
        """
        if not self.signer or not self.public_address:
            return
        try:
            response = requests.post(
                self.info_url,
                json={"type": "extraAgents", "user": self.public_address},
                timeout=10,
            )
            if response.status_code != 200:
                logger.debug(
                    "extraAgents check returned HTTP %s — skipping validation",
                    response.status_code,
                )
                return
            data = response.json()
            agents = data if isinstance(data, list) else data.get("agents", []) if isinstance(data, dict) else []
            signer_addr = self.signer.address.lower()
            approved = [
                str(a.get("address", "")).lower()
                for a in agents
                if isinstance(a, dict) and a.get("address")
            ]
            if signer_addr in approved:
                logger.info(
                    "Agent wallet %s is registered on Hyperliquid for %s",
                    self.signer.address,
                    self.public_address,
                )
                return
            if not approved:
                logger.warning(
                    "Could not confirm agent wallet registration (empty extraAgents "
                    "response). Signer=%s. Trading account=%s. If orders fail with "
                    "'User or API Wallet does not exist', approve the agent wallet "
                    "in the Hyperliquid UI (Account → API → Generate/Approve).",
                    self.signer.address,
                    self.public_address,
                )
                return
            logger.error(
                "AGENT WALLET NOT APPROVED: signer %s is NOT in the approved API "
                "wallets for %s. Approved wallets: %s. Live trading DISABLED. "
                "Approve the wallet in the Hyperliquid UI (Account → API → Approve) "
                "or update HL_AGENT_PRIVATE_KEY to one that IS approved.",
                self.signer.address,
                self.public_address,
                approved,
            )
            self.signer = None
            self.status_reason = "agent_wallet_not_approved_on_exchange"
        except Exception as exc:
            logger.debug("extraAgents check failed: %s — skipping validation", exc)

    def _load_asset_index_map(self):
        """Load asset index and szDecimals mapping from Hyperliquid meta endpoint.

        Capturing ``szDecimals`` is essential — it drives both price rounding
        (``max_decimals = 6 - szDecimals`` on perps) and size rounding, and
        without it we emit prices like ``"33.95015"`` that Hyperliquid rejects
        as ``{"error": "Order has invalid price."}``.
        """
        try:
            data = self.api_manager.post(
                {"type": "meta"},
                priority=Priority.LOW,
                timeout=10,
            )
            if isinstance(data, dict) and "universe" in data:
                for idx, coin_data in enumerate(data["universe"]):
                    coin_name = coin_data.get("name", "")
                    if coin_name:
                        self.asset_index_map[coin_name] = idx
                        try:
                            self.sz_decimals_map[coin_name] = int(
                                coin_data.get("szDecimals", 0) or 0
                            )
                        except (TypeError, ValueError):
                            self.sz_decimals_map[coin_name] = 0

                logger.info(
                    "Loaded %d asset indices + szDecimals from meta",
                    len(self.asset_index_map),
                )
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
            data = self.api_manager.post(
                {"type": "clearinghouseState", "user": self.public_address},
                priority=Priority.HIGH,
                timeout=10,
            )
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

    def get_wallet_balance(self) -> Optional[float]:
        """Backward-compatible alias for callers expecting wallet balance API."""
        return self.get_account_value()

    def _get_spot_usdc_balance(self) -> Optional[float]:
        """Fetch USDC balance from the spot wallet."""
        if not self.public_address:
            return None
        try:
            data = self.api_manager.post(
                {"type": "spotClearinghouseState", "user": self.public_address},
                priority=Priority.HIGH,
                timeout=10,
            )
            if not isinstance(data, dict):
                return None
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
                data = self.api_manager.post(
                    {"type": "clearinghouseState", "user": self.public_address},
                    priority=Priority.HIGH,
                    timeout=10,
                )
                if isinstance(data, dict):
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if log and (now - self._last_balance_log_ts) >= self._balance_log_interval_s:
            self._last_balance_log_ts = now
            logger.info(
                "Wallet balance: perps=$%s, spot=$%s, total=$%s",
                f"{perps:.2f}" if perps is not None else "n/a",
                f"{spot:.2f}" if spot is not None else "n/a",
                f"{total:.2f}" if total is not None else "n/a",
            )

        # Periodic cleanup of stale order dedup hashes (runs during idle periods
        # when no orders trigger the at-order-time cleanup).
        if now - self._last_hash_cleanup_ts > self._HASH_CLEANUP_INTERVAL:
            self._last_hash_cleanup_ts = now
            stale = [h for h, (ts, _) in self._recent_order_hashes.items()
                     if now - ts > self._ORDER_DEDUP_WINDOW]
            for h in stale:
                del self._recent_order_hashes[h]

        return self._last_balance_snapshot

    def get_balance_snapshot(self) -> Dict[str, Optional[float]]:
        """Return the most recent balance snapshot (cached, no API call)."""
        return dict(self._last_balance_snapshot)

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Safely coerce a Hyperliquid API field to float.

        Hyperliquid returns many numeric fields as strings ("0.123"),
        sometimes as nested dicts ({"value": 5}), and occasionally as
        ``null``.  Plain ``float(x)`` crashes on dicts and None — which
        caused ``Error getting positions: float() argument must be a
        string or a real number, not 'dict'`` to fire 72 times in one
        log, bringing down ``verify_fill`` and leaving real positions
        unprotected on the exchange.
        """
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value) if value else default
            except ValueError:
                return default
        if isinstance(value, dict):
            # Hyperliquid encodes leverage as {"type": "cross", "value": 5}
            # and cum funding as {"allTime": "0.123", ...}.  Prefer "value"
            # then "allTime" then any numeric-looking entry.
            for key in ("value", "allTime", "sinceOpen", "sinceChange"):
                if key in value:
                    return LiveTrader._coerce_float(value[key], default)
            return default
        return default

    def _normalize_position(self, pos: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten Hyperliquid position payload into a consistent dict shape.

        The clearinghouseState endpoint returns positions shaped like::

            {
              "position": {
                "coin": "BTC",
                "szi": "0.001",
                "entryPx": "67000",
                "unrealizedPnl": "1.23",
                "leverage": {"type": "cross", "value": 5},  # DICT, not scalar
                "positionValue": "67",
                ...
              },
              "type": "oneWay"
            }

        Every numeric field must be coerced through ``_coerce_float`` to
        handle the string-vs-number-vs-dict polymorphism without
        crashing.
        """
        pos_info = pos.get("position", pos) if isinstance(pos, dict) else {}
        size = self._coerce_float(pos_info.get("szi", pos_info.get("size", 0)))
        entry_px = self._coerce_float(
            pos_info.get("entryPx", pos_info.get("entry_price", 0))
        )
        unrealized = self._coerce_float(
            pos_info.get("unrealizedPnl", pos_info.get("unrealized_pnl", 0))
        )
        leverage = self._coerce_float(
            pos_info.get("leverage", pos_info.get("lev", 1)),
            default=1.0,
        )
        if leverage <= 0:
            leverage = 1.0
        return {
            "coin": pos_info.get("coin", ""),
            "size": abs(size),
            "szi": size,
            "side": "long" if size > 0 else "short",
            "entry_price": entry_px,
            "entryPx": entry_px,
            "unrealized_pnl": unrealized,
            "unrealizedPnl": unrealized,
            "leverage": leverage,
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

    @staticmethod
    def _signal_source_key(signal: TradeSignal) -> str:
        source = getattr(signal, "source", None)
        if hasattr(source, "value"):
            source = source.value
        key = str(source or "unknown").strip().lower()
        return key or "unknown"

    def _refresh_external_kill_switch(self) -> bool:
        """
        External kill-switch hook.

        Triggers when either:
          - env LIVE_EXTERNAL_KILL_SWITCH is truthy (1/true/on/yes)
          - LIVE_EXTERNAL_KILL_SWITCH_FILE exists and is non-empty/truthy
        """
        reason = ""

        env_val = os.environ.get(self.external_kill_switch_env, "").strip().lower()
        if env_val in {"1", "true", "yes", "on"}:
            reason = f"env:{self.external_kill_switch_env}"

        if not reason and self.external_kill_switch_file:
            try:
                if os.path.exists(self.external_kill_switch_file):
                    with open(self.external_kill_switch_file, "r", encoding="utf-8") as handle:
                        content = handle.read().strip().lower()
                    if not content or content in {"1", "true", "yes", "on", "kill"}:
                        reason = f"file:{self.external_kill_switch_file}"
            except Exception as exc:
                logger.warning(
                    "Failed to evaluate external kill switch file %s: %s",
                    self.external_kill_switch_file,
                    exc,
                )

        if reason:
            if not self.kill_switch_active or self._kill_switch_reason != reason:
                logger.critical("External kill switch ACTIVATED (%s)", reason)
            self.kill_switch_active = True
            self._kill_switch_reason = reason
            self.status_reason = "external_kill_switch"
            return True
        return False

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_reset_date:
            self.daily_reset_date = today
            self.daily_pnl = 0.0
            self.daily_realized_pnl = 0.0
            self.daily_unrealized_pnl = 0.0
            self.orders_today = 0
            self.fills_today = 0
            self._source_orders_today.clear()
            self.kill_switch_active = False
            self._kill_switch_reason = ""
            logger.info("Daily counters reset")
            # Keep external kill-switch state sticky across day boundaries.
            self._refresh_external_kill_switch()

    def check_daily_loss(self) -> bool:
        """
        Check if daily loss limit exceeded.

        Returns:
            True if loss > limit (triggers kill switch)
        """
        self._check_daily_reset()
        if self._refresh_external_kill_switch():
            return True

        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"⚠️  Daily loss limit exceeded: ${abs(self.daily_pnl):.2f} > ${self.max_daily_loss:.2f}")
            self.kill_switch_active = True
            return True

        return False

    def reconcile_positions(self):
        """
        Compare exchange positions vs local state and protect orphans.

        Logs discrepancies so the operator knows if the bot's view
        of the world differs from reality, then places SL/TP on any
        position that lacks protective orders (e.g., positions opened
        by a previous run that crashed during verify_fill, leaving SL/TP
        un-placed).
        """
        logger.info("Reconciling positions with exchange...")
        try:
            exchange_positions = self.get_positions()
            active_coins = []
            unprotected = []

            for pos in exchange_positions:
                coin = pos.get("coin", "")
                # _normalize_position already coerces everything; these
                # are safe scalars now.
                size = self._coerce_float(pos.get("szi", pos.get("size", 0)))
                entry_px = self._coerce_float(
                    pos.get("entry_price", pos.get("entryPx", 0))
                )
                unrealized_pnl = self._coerce_float(
                    pos.get("unrealized_pnl", pos.get("unrealizedPnl", 0))
                )

                if abs(size) > 0 and coin:
                    side = "long" if size > 0 else "short"
                    active_coins.append(coin)
                    self._last_known_positions[coin] = {
                        "coin": coin,
                        "side": side,
                        "size": abs(size),
                        "entry_price": entry_px,
                        "unrealized_pnl": unrealized_pnl,
                    }
                    logger.info(
                        f"EXISTING POSITION found: {side.upper()} {coin} "
                        f"size={abs(size)} entry=${entry_px:,.2f} "
                        f"uPnL=${unrealized_pnl:+,.2f}"
                    )
                    unprotected.append(pos)

            if active_coins:
                logger.info(
                    f"Reconciliation: {len(active_coins)} open positions found on "
                    f"exchange: {', '.join(active_coins)}. These will be tracked."
                )
                # Place SL/TP for any position that doesn't already have
                # protective reduce-only orders.  This catches the class
                # of bugs where verify_fill crashed (float(dict) leverage)
                # and SL/TP placement was skipped — leaving real money
                # on the exchange with no downside protection.
                self.protect_orphaned_positions(unprotected)
            else:
                logger.info("Reconciliation: no open positions on exchange (clean state)")

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
            logger.warning(
                "Could not verify exchange state on startup. "
                "Proceeding with caution — monitor positions manually."
            )

    def protect_orphaned_positions(
        self,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Place SL/TP on any open position that lacks reduce-only orders.

        Designed to be safe to call repeatedly — checks open-order state
        first and skips positions that already have protective orders.
        Uses conservative default SL/TP percentages (3% SL / 6% TP) when
        no signal context is available.

        Args:
            positions: Optional pre-fetched list of normalized positions.
                When omitted, fetches via ``get_positions()``.

        Returns:
            Summary dict with counts of protected / skipped / failed.
        """
        if self.dry_run:
            return {"status": "skipped", "reason": "dry_run"}

        if positions is None:
            positions = self.get_positions()

        if not positions:
            return {"status": "ok", "protected": 0, "skipped": 0, "failed": 0}

        # One fetch of open orders, then index by coin for O(1) lookup.
        try:
            open_orders = self.get_open_orders()
        except Exception as exc:
            logger.warning("protect_orphaned_positions: open-orders fetch failed: %s", exc)
            open_orders = []

        # Index protective orders by coin. Track order IDs for stale cleanup.
        _MAX_PROTECTIVE_ORDERS_PER_COIN = 2  # 1 SL + 1 TP expected
        protected_coins: Dict[str, List[str]] = {}
        protective_order_ids: Dict[str, List[int]] = {}  # coin → [oid, ...]
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            coin = order.get("coin", "")
            if not coin:
                continue
            reduce_only = bool(order.get("reduceOnly") or order.get("reduce_only"))
            order_type_raw = order.get("orderType") or order.get("type") or ""
            order_type = str(order_type_raw).lower()
            # Trigger orders from place_trigger_order show up with
            # orderType containing "stop" or "take" — both protect the
            # position even without an explicit reduceOnly flag.
            if reduce_only or "stop" in order_type or "take" in order_type or "trigger" in order_type:
                protected_coins.setdefault(coin, []).append(order_type or "reduce_only")
                oid = order.get("oid") or order.get("order_id") or order.get("id")
                if oid is not None:
                    protective_order_ids.setdefault(coin, []).append(int(oid))

        # Clean up stale protective orders: if a coin has >2 reduce_only orders,
        # cancel the oldest extras to prevent accumulation over position changes.
        stale_cancelled = 0
        for coin, oids in protective_order_ids.items():
            if len(oids) > _MAX_PROTECTIVE_ORDERS_PER_COIN:
                excess = sorted(oids)[:-_MAX_PROTECTIVE_ORDERS_PER_COIN]  # Keep newest 2
                logger.warning(
                    "Cleaning %d stale protective orders for %s (had %d, keeping %d)",
                    len(excess), coin, len(oids), _MAX_PROTECTIVE_ORDERS_PER_COIN,
                )
                for oid in excess:
                    try:
                        if self.cancel_order(coin, oid):
                            stale_cancelled += 1
                    except Exception as exc:
                        logger.debug("Failed to cancel stale order %s/%s: %s", coin, oid, exc)

        protected = 0
        skipped = 0
        failed = 0
        for pos in positions:
            coin = pos.get("coin", "")
            size = abs(self._coerce_float(pos.get("size", pos.get("szi", 0))))
            szi = self._coerce_float(pos.get("szi", pos.get("size", 0)))
            entry_price = self._coerce_float(
                pos.get("entry_price", pos.get("entryPx", 0))
            )
            if not coin or size <= 0 or entry_price <= 0:
                continue

            if coin in protected_coins:
                logger.info(
                    "protect_orphaned_positions: %s already has protective "
                    "orders (%s), skipping",
                    coin, protected_coins[coin],
                )
                skipped += 1
                continue

            side = "long" if szi > 0 else "short"
            # Conservative defaults: 3% stop loss, 6% take profit
            sl_pct = 0.03
            tp_pct = 0.06
            if side == "long":
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
                protect_side = "sell"
            else:
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)
                protect_side = "buy"

            logger.warning(
                "UNPROTECTED POSITION: %s %s size=%.6f entry=$%.6f — "
                "placing default 3%%/6%% SL/TP (SL=$%.6f TP=$%.6f)",
                side.upper(), coin, size, entry_price, sl_price, tp_price,
            )

            sl_result = self.place_trigger_order(
                coin, protect_side, size, sl_price, tp_or_sl="sl"
            )
            tp_result = self.place_trigger_order(
                coin, protect_side, size, tp_price, tp_or_sl="tp"
            )

            sl_ok = self._is_order_result_success(sl_result)
            tp_ok = self._is_order_result_success(tp_result)
            if sl_ok and tp_ok:
                protected += 1
                logger.info(
                    "Orphan protection placed for %s: SL=$%.6f TP=$%.6f",
                    coin, sl_price, tp_price,
                )
            else:
                failed += 1
                logger.error(
                    "Orphan protection FAILED for %s: sl=%s tp=%s — "
                    "position remains unprotected, MANUAL INTERVENTION REQUIRED",
                    coin, sl_result, tp_result,
                )

        summary = {
            "status": "ok",
            "protected": protected,
            "skipped": skipped,
            "failed": failed,
            "stale_cancelled": stale_cancelled,
        }
        if protected or failed or stale_cancelled:
            logger.warning("Orphan protection summary: %s", summary)
        return summary

    _PROTECTIVE_RESIZE_DELAY_S = 30.0  # Wait for remaining fills before resizing

    def _deferred_protective_resize(
        self,
        coin: str,
        protect_side: str,
        original_protective_size: float,
        sl_price: float,
        tp_price: float,
        tp_or_sl_type: tuple = ("sl", "tp"),
    ) -> None:
        """Background task: after a delay, check actual position size and resize
        SL/TP if they're oversized relative to the real position.

        This handles the case where a market order partially fills (e.g. 140 of
        1902 units) and we initially place SL/TP for the full requested size.
        After the delay, if the remaining quantity never filled, we cancel and
        re-place protective orders sized to the actual position.
        """
        try:
            time.sleep(self._PROTECTIVE_RESIZE_DELAY_S)

            # Check actual position size
            positions = self.get_positions()
            actual_size = 0.0
            for pos in positions:
                if pos.get("coin") == coin:
                    actual_size = abs(self._coerce_float(pos.get("size", pos.get("szi", 0))))
                    break

            if actual_size <= 0:
                logger.info(
                    "Deferred resize for %s: position gone (closed or liquidated), skipping",
                    coin,
                )
                return

            # Only resize if the protective orders are significantly oversized (>10%)
            if original_protective_size <= actual_size * 1.10:
                logger.debug(
                    "Deferred resize for %s: protective size %.4f within 10%% of actual %.4f, no action",
                    coin, original_protective_size, actual_size,
                )
                return

            logger.warning(
                "PROTECTIVE RESIZE %s: actual position=%.4f but SL/TP sized at %.4f (%.0f%% over). "
                "Cancelling and re-placing at correct size.",
                coin, actual_size, original_protective_size,
                (original_protective_size / actual_size - 1) * 100,
            )

            # Cancel all existing protective orders for this coin
            try:
                open_orders = self.get_open_orders()
                for order in open_orders:
                    if not isinstance(order, dict):
                        continue
                    if order.get("coin") != coin:
                        continue
                    reduce_only = bool(order.get("reduceOnly") or order.get("reduce_only"))
                    otype = str(order.get("orderType") or order.get("type") or "").lower()
                    if reduce_only or "stop" in otype or "take" in otype or "trigger" in otype:
                        oid = order.get("oid") or order.get("order_id") or order.get("id")
                        if oid is not None:
                            self.cancel_order(coin, int(oid))
            except Exception as exc:
                logger.warning("Deferred resize cancel phase failed for %s: %s", coin, exc)
                return

            # Re-place SL and TP at correct actual size
            sl_result = self.place_trigger_order(coin, protect_side, actual_size, sl_price, tp_or_sl="sl")
            tp_result = self.place_trigger_order(coin, protect_side, actual_size, tp_price, tp_or_sl="tp")

            sl_ok = self._is_order_result_success(sl_result)
            tp_ok = self._is_order_result_success(tp_result)
            if sl_ok and tp_ok:
                logger.info(
                    "Deferred resize for %s complete: SL/TP resized from %.4f to %.4f",
                    coin, original_protective_size, actual_size,
                )
            else:
                logger.error(
                    "Deferred resize for %s FAILED: sl=%s tp=%s — position may be unprotected!",
                    coin, sl_result, tp_result,
                )

        except Exception as exc:
            logger.error("Deferred protective resize error for %s: %s", coin, exc)

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

            fills = self.api_manager.post(
                {"type": "userFills", "user": self.public_address},
                priority=Priority.NORMAL,
                timeout=10,
            )
            if not isinstance(fills, list):
                return

            # Sum realized closed PnL from today's fills.
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_realized = 0.0
            for fill in fills:
                fill_time = fill.get("time", "")
                if isinstance(fill_time, (int, float)):
                    fill_date = datetime.fromtimestamp(
                        fill_time / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d")
                elif isinstance(fill_time, str):
                    fill_date = fill_time[:10]
                else:
                    continue

                if fill_date == today:
                    closed_pnl = float(fill.get("closedPnl", 0))
                    today_realized += closed_pnl

            # Include current unrealized PnL so daily loss controls react to
            # open-position drawdowns, not only realized closes.
            unrealized = 0.0
            try:
                for pos in self.get_positions():
                    unrealized += self._coerce_float(
                        pos.get("unrealized_pnl", pos.get("unrealizedPnl", 0))
                    )
            except Exception as exc:
                logger.debug("Could not include unrealized PnL in daily tracking: %s", exc)

            today_pnl = today_realized + unrealized

            old_pnl = self.daily_pnl
            self.daily_realized_pnl = today_realized
            self.daily_unrealized_pnl = unrealized
            self.daily_pnl = today_pnl

            if abs(self.daily_pnl) > 0 and abs(self.daily_pnl - old_pnl) > 0.01:
                logger.info(
                    "Daily PnL updated: total=%+.2f (realized=%+.2f, unrealized=%+.2f)",
                    self.daily_pnl,
                    self.daily_realized_pnl,
                    self.daily_unrealized_pnl,
                )

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
            mids = self.api_manager.post(
                {"type": "allMids"},
                priority=Priority.HIGH,
                timeout=10,
            )
            if not isinstance(mids, dict):
                return None

            price = mids.get(coin)
            if price:
                return float(price)
        except Exception as e:
            logger.error(f"Failed to get mid price for {coin}: {e}")

        return None

    def _validate_price(self, coin: str, price: float) -> bool:
        """
        Validate that a price is reasonable.

        Rejects:
        - Zero, negative, NaN, Infinity
        - Prices that deviate >10% from the last known good price
          (protects against corrupt API responses)
        - First-sample baseline poisoning after restart (requires two
          reasonably-close samples before trusting the initial baseline)

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
        if last_good and last_good > 0:
            deviation = abs(price - last_good) / last_good
            if deviation > 0.10:  # >10% move since last check
                logger.error(
                    f"PRICE REJECTED: {coin} price=${price:,.2f} deviates "
                    f"{deviation:.1%} from last known ${last_good:,.2f}. "
                    f"Possible corrupt data — blocking order."
                )
                return False
            self._price_history[coin] = price
            return True

        # No trusted baseline yet: require two close samples.
        seed = self._price_seed_candidates.get(coin)
        if not seed or seed <= 0:
            self._price_seed_candidates[coin] = price
            logger.info(
                "Price sanity baseline warm-up for %s: captured first sample %.6f; "
                "waiting for confirmation sample before strict deviation checks.",
                coin,
                price,
            )
            return True

        seed_deviation = abs(price - seed) / seed
        if seed_deviation > 0.05:
            logger.error(
                "PRICE REJECTED during baseline warm-up: %s second sample %.6f "
                "deviates %.1f%% from initial %.6f. Re-seeding baseline.",
                coin,
                price,
                seed_deviation * 100.0,
                seed,
            )
            self._price_seed_candidates[coin] = price
            return False

        # Promote the confirmed baseline.
        confirmed = (seed + price) / 2.0
        self._price_history[coin] = confirmed
        self._price_seed_candidates.pop(coin, None)
        logger.info(
            "Price sanity baseline established for %s at %.6f (seed %.6f / %.6f).",
            coin,
            confirmed,
            seed,
            price,
        )
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
    def _extract_inner_order_statuses(result: Optional[Dict]) -> List[Dict[str, Any]]:
        """Pull the per-order ``statuses`` list out of a Hyperliquid response.

        Hyperliquid wraps successful requests in
        ``{"status": "ok", "response": {"type": "order", "data": {"statuses": [...]}}}``
        where each entry is one of:
          - ``{"resting": {"oid": int}}``     — posted, waiting to match
          - ``{"filled": {"oid": int, "totalSz": str, "avgPx": str}}``
          - ``{"error": "..."}``              — per-order rejection, outer
                                                status is STILL ``"ok"``

        Returns ``[]`` if the shape doesn't match.
        """
        if not isinstance(result, dict):
            return []
        response = result.get("response")
        if not isinstance(response, dict):
            return []
        data = response.get("data")
        if not isinstance(data, dict):
            return []
        statuses = data.get("statuses")
        if isinstance(statuses, list):
            return [s for s in statuses if isinstance(s, dict)]
        return []

    @classmethod
    def _extract_reported_fill_size(cls, result: Optional[Dict]) -> Optional[float]:
        """Return the exchange-reported filled size from an order response, if any."""
        total = 0.0
        found = False
        for entry in cls._extract_inner_order_statuses(result):
            filled = entry.get("filled")
            if not isinstance(filled, dict):
                continue
            total_sz = cls._coerce_float(filled.get("totalSz"), 0.0)
            if total_sz > 0:
                total += total_sz
                found = True
        return total if found else None

    @classmethod
    def _extract_reported_fill_price(cls, result: Optional[Dict]) -> Optional[float]:
        """Return size-weighted average fill price from an order response, if any."""
        weighted_notional = 0.0
        total_size = 0.0
        for entry in cls._extract_inner_order_statuses(result):
            filled = entry.get("filled")
            if not isinstance(filled, dict):
                continue
            total_sz = cls._coerce_float(filled.get("totalSz"), 0.0)
            avg_px = cls._coerce_float(filled.get("avgPx"), 0.0)
            if total_sz <= 0 or avg_px <= 0:
                continue
            weighted_notional += total_sz * avg_px
            total_size += total_sz
        if total_size > 0:
            return weighted_notional / total_size
        return None

    @classmethod
    def _is_order_result_success(cls, result: Optional[Dict]) -> bool:
        """Best-effort classification of exchange responses into success/failure.

        Hyperliquid uses a two-level success model: the outer request can
        be ``status: ok`` while the inner per-order ``statuses`` list
        contains ``{"error": "..."}`` rejections.  We must inspect the
        inner list — otherwise a wire-format rejection (e.g. "Order has
        invalid price.") looks successful and fill verification polls
        pointlessly for 10 seconds before reporting a phantom failure.
        """
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
        if status not in {"success", "simulated", "verified", "filled", "accepted", "ok"}:
            return False

        # Outer ok — but drill into inner statuses and fail if any entry
        # carries an error string.
        inner = cls._extract_inner_order_statuses(result)
        if inner and any("error" in entry for entry in inner):
            errors = [entry.get("error") for entry in inner if "error" in entry]
            logger.warning(
                "Order outer status was 'ok' but per-order statuses contain "
                "errors: %s",
                errors,
            )
            return False
        return True

    @classmethod
    def _is_insufficient_margin_rejection(cls, result: Optional[Dict]) -> bool:
        """Return True when an order rejection is caused by insufficient margin."""
        if not isinstance(result, dict):
            return False

        reason = str(result.get("reason", "")).strip().lower()
        if "insufficient_margin" in reason:
            return True

        messages: List[str] = []
        errors = result.get("errors")
        if isinstance(errors, list):
            messages.extend(str(err) for err in errors if err is not None)
        elif errors is not None:
            messages.append(str(errors))

        message = result.get("message")
        if message:
            messages.append(str(message))

        return any("insufficient margin" in msg.lower() for msg in messages)

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

            # Vault vs agent-wallet master-account semantics:
            # -------------------------------------------------
            # Hyperliquid has TWO distinct "trade on behalf of" modes:
            #
            # 1. Agent wallet → master account (the common case, and our case):
            #    HL_PUBLIC_ADDRESS is a regular user account.  The signer is
            #    an approved API wallet for that account.  Orders must NOT
            #    include ``vaultAddress`` in the request payload, and the
            #    signed hash must NOT include a vault byte.  Hyperliquid
            #    looks up the agent→master mapping from its approved-agents
            #    registry and routes the order to the master automatically.
            #
            # 2. Trading on behalf of an actual Vault (or sub-account):
            #    HL_VAULT_ADDRESS is set and refers to a real Hyperliquid
            #    Vault.  The signer is authorized to trade that vault.
            #    ``vaultAddress`` is sent in the payload AND baked into the
            #    signed hash.
            #
            # The previous implementation treated case 1 as case 2 and sent
            # HL_PUBLIC_ADDRESS as ``vaultAddress``, producing the error
            # "Vault not registered: 0x…" (because a regular user account
            # is not a vault).  We now only send vaultAddress when an
            # explicit HL_VAULT_ADDRESS env var is set.
            vault_address: Optional[str] = (
                os.environ.get("HL_VAULT_ADDRESS", "").strip() or None
            )

            # Sign the action.  vault_address MUST be included in the hash
            # iff we are going to echo it in the request body — and MUST
            # NOT be included otherwise.
            signature = self.signer.sign_action(
                action, nonce, vault_address=vault_address,
            )

            # Prepare request
            payload = {
                "action": action,
                "nonce": nonce,
                "signature": signature,
            }
            if vault_address:
                payload["vaultAddress"] = vault_address

            logger.debug(f"Posting order to {self.exchange_url}: action_hash={action_hash}")

            result = self.api_manager.post(
                payload,
                priority=Priority.CRITICAL,
                retries=1,
                endpoint_url=self.exchange_url,
                cache_response=False,
                req_type="exchange",
                timeout=30,
                raise_on_timeout=True,
            )
            if result is None:
                logger.error("Exchange request failed for action_hash=%s", action_hash[:16])
                return {"status": "error", "message": "exchange request failed"}

            logger.info(f"Order posted: {json.dumps(result)}")

            # Hyperliquid wraps per-order errors under an outer
            # ``status: ok`` response.  Promote any inner-error into a
            # clear rejection so downstream callers stop waiting on
            # verify_fill for a fill that will never come.
            inner_statuses = self._extract_inner_order_statuses(result)
            if inner_statuses and any("error" in s for s in inner_statuses):
                errors = [s["error"] for s in inner_statuses if "error" in s]
                rejection_payload = {
                    "status": "rejected",
                    "reason": "exchange_inner_error",
                    "errors": errors,
                    "raw_response": result,
                }
                if self._is_insufficient_margin_rejection(rejection_payload):
                    logger.warning(
                        "Exchange rejected order for insufficient margin: %s (action_hash=%s)",
                        errors,
                        action_hash[:16],
                    )
                else:
                    logger.error(
                        "Exchange rejected order at inner level: %s (action_hash=%s)",
                        errors, action_hash[:16],
                    )
                # Do NOT cache inner-error rejections — the order did not
                # execute, so retries (e.g. from the next orphan-protection
                # sweep after the caller has fixed the price) must be
                # allowed to reach the exchange instead of being silently
                # swallowed by the dedup cache.
                return rejection_payload

            # Only cache successfully accepted orders.  Rejected orders
            # are not "placed" and must remain retryable.
            self._recent_order_hashes[action_hash] = (now, result)
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
        self._refresh_external_kill_switch()

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
        requested_size = float(size)

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
            # Exchange-enforced minimum notional: Hyperliquid silently drops
            # orders below $10.  Reject here with a clear reason instead of
            # sending a no-op order and waiting for fill verification to
            # time out.
            if self.min_order_usd and notional < self.min_order_usd:
                # Tiny price drift between _rescale_size_for_live (signal
                # entry price) and here (live mid) can leave us a few cents
                # below the $11 floor.  Floor UP to ~min*1.02 in-place as
                # long as the bump stays under max_order_usd.  This is
                # cheaper than rejecting the order: the caller has already
                # paid for signal generation, firewall, rescale, and
                # signing.
                bumped = False
                if self.max_order_usd and self.min_order_usd > 0:
                    target_notional = min(
                        self.max_order_usd,
                        self.min_order_usd * 1.02,
                    )
                    candidate_size = target_notional / price if price > 0 else 0
                    candidate_notional = candidate_size * price
                    # Only bump if the new size is within 25% of the
                    # original — anything larger means the rescale
                    # computed the wrong size entirely and should fail
                    # loudly instead of silently scaling up.
                    if (
                        candidate_notional >= self.min_order_usd
                        and candidate_notional <= self.max_order_usd
                        and candidate_size <= size * 1.25
                    ):
                        logger.info(
                            "place_market_order: floor-up %s %s %.6f → "
                            "%.6f (notional $%.2f → $%.2f) to clear "
                            "Hyperliquid's $%.2f minimum after price drift",
                            coin, side, size, candidate_size,
                            notional, candidate_notional, self.min_order_usd,
                        )
                        size = candidate_size
                        notional = candidate_notional
                        bumped = True
                if not bumped:
                    logger.warning(
                        "place_market_order: rejecting %s %s size %.6f — "
                        "notional $%.2f is below Hyperliquid's $%.2f "
                        "minimum.  Raise LIVE_MAX_ORDER_USD or fund the "
                        "live wallet so rescaling produces a larger order.",
                        coin, side, size, notional, self.min_order_usd,
                    )
                    return {
                        "status": "rejected",
                        "reason": "below_exchange_minimum_notional",
                        "notional": round(notional, 4),
                        "min_notional": self.min_order_usd,
                    }

        # Format price and size for Hyperliquid's strict wire protocol
        # (5 sig figs, (6 - szDecimals) decimals for perps, no trailing
        # zeros).  Emitting plain ``round(x, 8)`` produces strings like
        # ``"33.95015"`` which the exchange rejects as "Order has invalid
        # price." — outer response is still ``status:ok`` so the order
        # looks accepted but silently never fills.
        sz_decimals = self.sz_decimals_map.get(coin, 0)
        wire_price = _hl_format_price(price, sz_decimals)
        wire_size = _hl_format_size(size, sz_decimals)
        if wire_size == "0":
            logger.warning(
                "place_market_order: %s %s rounds to zero at szDecimals=%d "
                "(raw size %.10f). Rejecting.",
                coin, side, sz_decimals, size,
            )
            return {"status": "rejected", "reason": "size_rounds_to_zero"}

        # Build order
        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": wire_price,
            "s": wire_size,
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
        if isinstance(result, dict):
            result = dict(result)
            result["requested_size"] = requested_size
            result["submitted_size"] = float(size)
            result["submitted_notional"] = float(notional)
            result["wire_size"] = wire_size
            result["wire_price"] = wire_price
            reported_fill_size = self._extract_reported_fill_size(result)
            if reported_fill_size is not None:
                result["exchange_reported_fill_size"] = reported_fill_size
            reported_fill_price = self._extract_reported_fill_price(result)
            if reported_fill_price is not None:
                result["exchange_reported_fill_price"] = reported_fill_price

        if self._is_order_result_success(result) and not self.dry_run:
            self.orders_today += 1

        return result

    def verify_fill(self, coin: str, expected_side: str, expected_size: float,
                    timeout: float = 10.0, poll_interval: float = 1.0,
                    blocking: Optional[bool] = None) -> Optional[Dict]:
        """
        Verify an order fill against exchange positions.

        Non-blocking by default (single snapshot poll) to avoid stalling
        trading-cycle threads. Set ``blocking=True`` to keep the legacy
        poll-with-timeout behavior.

        Args:
            coin: Expected coin
            expected_side: "buy" or "sell"
            expected_size: Expected position size change
            timeout: Max seconds to wait
            poll_interval: Seconds between polls
            blocking: Override instance default verification mode

        Returns:
            Position dict if verified, None if not found after timeout
        """
        if self.dry_run:
            return {"status": "verified", "dry_run": True}

        expected_side = self._normalize_order_side(expected_side)
        blocking_mode = self._fill_verify_blocking if blocking is None else bool(blocking)

        def _poll_once(attempt: int) -> Optional[Dict]:
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
                position_entry_price = self._coerce_float(
                    pos.get("entry_price", pos.get("entryPx", 0)),
                    0.0,
                )

                logger.info(
                    f"Fill VERIFIED: {coin} size={pos_size} "
                    f"(expected={expected_size:.6f}, attempt {attempt})"
                )
                return {
                    "status": "verified",
                    "coin": coin,
                    "size": matched_size,
                    "position_size": pos_size,
                    "partial_fill": partial_fill,
                    "attempt": attempt,
                    "entry_price": position_entry_price if position_entry_price > 0 else None,
                }
            return None

        if not blocking_mode:
            try:
                return _poll_once(attempt=1)
            except Exception as e:
                logger.debug(f"Fill verification snapshot error: {e}")
                return None

        deadline = time.time() + timeout
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            try:
                match = _poll_once(attempt=attempt)
                if match:
                    match["elapsed_s"] = round(time.time() - (deadline - timeout), 2)
                    return match
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
        self._refresh_external_kill_switch()

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
            # Exchange-enforced minimum notional — Hyperliquid silently
            # drops orders below $10.  Reject here with a clear reason.
            if self.min_order_usd and notional < self.min_order_usd:
                logger.warning(
                    "place_limit_order: rejecting %s %s size %.6f — notional "
                    "$%.2f is below Hyperliquid's $%.2f minimum.",
                    coin, side, size, notional, self.min_order_usd,
                )
                return {
                    "status": "rejected",
                    "reason": "below_exchange_minimum_notional",
                    "notional": round(notional, 4),
                    "min_notional": self.min_order_usd,
                }

        # Format for Hyperliquid wire protocol — see place_market_order
        # for the full rationale.  Without szDecimals-aware rounding the
        # exchange returns "Order has invalid price" under a
        # ``status: ok`` wrapper and our fill verification times out.
        sz_decimals = self.sz_decimals_map.get(coin, 0)
        wire_price = _hl_format_price(price, sz_decimals)
        wire_size = _hl_format_size(size, sz_decimals)
        if wire_size == "0":
            logger.warning(
                "place_limit_order: %s %s rounds to zero at szDecimals=%d "
                "(raw size %.10f). Rejecting.",
                coin, side, sz_decimals, size,
            )
            return {"status": "rejected", "reason": "size_rounds_to_zero"}

        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": wire_price,
            "s": wire_size,
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

        # Hyperliquid's trigger orders need BOTH a trigger price AND a
        # limit price in the ``p`` field.  ``p="0"`` was wrong — the
        # exchange rejects it with "Order has invalid price" under a
        # ``status: ok`` wrapper, and the position sits unprotected.
        #
        # ``p`` is the limit cap applied once the trigger fires, so we
        # pad the trigger_price by a slippage buffer in the direction
        # that lets the resulting market order match:
        #   buy  (closing short) → p = trigger * (1 + slippage)
        #   sell (closing long)  → p = trigger * (1 - slippage)
        # 5% slippage is the Hyperliquid SDK default for market_close.
        slippage = 0.05
        if side.lower() == "buy":
            limit_px = trigger_price * (1 + slippage)
        else:
            limit_px = trigger_price * (1 - slippage)

        sz_decimals = self.sz_decimals_map.get(coin, 0)
        wire_size = _hl_format_size(size, sz_decimals)
        wire_trigger_px = _hl_format_price(trigger_price, sz_decimals)
        wire_limit_px = _hl_format_price(limit_px, sz_decimals)
        if wire_size == "0":
            logger.warning(
                "place_trigger_order: %s %s rounds to zero at szDecimals=%d "
                "(raw size %.10f). Rejecting.",
                coin, side, sz_decimals, size,
            )
            return {"status": "rejected", "reason": "size_rounds_to_zero"}
        if wire_trigger_px == "0" or wire_limit_px == "0":
            logger.error(
                "place_trigger_order: formatted price rounded to 0 for %s "
                "(trigger=%s, limit=%s, raw_trigger=%s, raw_limit=%s). Rejecting.",
                coin, wire_trigger_px, wire_limit_px, trigger_price, limit_px,
            )
            return {
                "status": "rejected",
                "reason": "price_rounds_to_zero",
            }

        order = {
            "a": asset_idx,
            "b": side.lower() == "buy",
            "p": wire_limit_px,
            "s": wire_size,
            "r": True,  # SL/TP must be reduce_only to close the position
            "t": {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": wire_trigger_px,
                    "tpsl": tp_or_sl,
                }
            }
        }

        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }

        logger.info(
            "Placing %s trigger: %s %s size=%s trigger=%s limit=%s",
            tp_or_sl, coin, side, wire_size, wire_trigger_px, wire_limit_px,
        )
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
        ok = self._is_order_result_success(result)
        if not ok:
            logger.warning(
                "Cancel rejected for %s oid=%s: %s",
                coin,
                order_id,
                result,
            )
        return ok

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

            cancelled = 0
            for order in orders:
                if self.cancel_order(order.get("coin"), order.get("id")):
                    cancelled += 1

            if cancelled < len(orders):
                logger.warning(
                    "cancel_all_orders: confirmed %d/%d cancellations%s",
                    cancelled,
                    len(orders),
                    f" for {coin}" if coin else "",
                )
            return cancelled
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
            if size <= 0:
                logger.warning("close_position(%s): position size is zero — may have been closed concurrently", coin)
                return {"status": "error", "message": "Position size is zero (may have been closed by SL/TP)"}
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
                coin = pos.get("coin")
                if not coin:
                    continue
                result = None
                for attempt in range(1, self._emergency_close_retries + 1):
                    result = self.close_position(coin)
                    if self._is_order_result_success(result):
                        logger.warning(
                            "Closed position on %s (attempt %d/%d)",
                            coin,
                            attempt,
                            self._emergency_close_retries,
                        )
                        break
                    if attempt < self._emergency_close_retries:
                        wait = self._emergency_close_retry_delay_s * attempt
                        logger.warning(
                            "Emergency close retry for %s in %.1fs (attempt %d/%d): %s",
                            coin,
                            wait,
                            attempt,
                            self._emergency_close_retries,
                            result,
                        )
                        time.sleep(wait)
                if result is not None:
                    results.append(result)
                    if not self._is_order_result_success(result):
                        logger.error(
                            "FAILED to close %s after %d attempts. Manual intervention required.",
                            coin,
                            self._emergency_close_retries,
                        )

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

            data = self.api_manager.post(
                {"type": "openOrders", "user": self.public_address},
                priority=Priority.HIGH,
                timeout=10,
            )

            orders = data.get("orders", []) if isinstance(data, dict) else data
            return orders if isinstance(orders, list) else []

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

        # Avoid mutating caller-owned signal objects in-place. This keeps
        # upstream audit logs and strategy traces immutable.
        adjusted = copy.deepcopy(signal)

        regime = regime_info.get("regime", "neutral")
        confidence = regime_info.get("confidence", 0.0)

        # CRIT-FIX CRIT-4 (continued): remove the position_pct * effective_size fallback.
        # execute_signal now guarantees signal.size > 0 before calling this method.
        # Use signal.size directly; log an error if somehow it's still unset here.
        base_size = adjusted.size or 0.0
        if base_size <= 0:
            logger.error(
                f"apply_regime_overlay: signal.size not set for {adjusted.coin} — "
                f"skipping size adjustment (execute_signal should have computed it first)"
            )
            return adjusted

        # 1. CRASH regime: reduce size 60%, tighten stop loss to 3%
        if regime == "crash" and confidence > 0.4:
            adjusted.size = base_size * 0.4  # 60% reduction = multiply by 0.4
            adjusted.risk.stop_loss_pct = 0.03
            logger.warning(
                f"REGIME OVERLAY: crash detected (conf={confidence:.2f}), "
                f"reducing size 60%, tightening SL to 3% for {adjusted.coin}"
            )

        # 2. VOLATILE regime: reduce size 30%, widen stop loss to 8%
        elif regime == "volatile":
            adjusted.size = base_size * 0.7  # 30% reduction = multiply by 0.7
            adjusted.risk.stop_loss_pct = 0.08
            logger.info(
                f"REGIME OVERLAY: volatile detected, "
                f"reducing size 30%, widening SL to 8% for {adjusted.coin}"
            )

        # 3. BULLISH regime: allow full size, boost by 10%
        elif regime == "bullish" and confidence > 0.6:
            adjusted.size = base_size * 1.1  # 10% boost
            logger.info(
                f"REGIME OVERLAY: bullish confirmed (conf={confidence:.2f}), "
                f"boosting size 10% for {adjusted.coin}"
            )

        return adjusted

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
        if notional > self.max_order_usd:
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
            notional = capped_size * mid

        # Reject anything below the exchange minimum — Hyperliquid silently
        # drops sub-$10 orders and fill verification times out.
        if self.min_order_usd and notional < self.min_order_usd:
            logger.warning(
                "Dropping %s trade: notional $%.2f is below Hyperliquid's "
                "$%.2f minimum per order.  Raise LIVE_MAX_ORDER_USD or fund "
                "the live wallet so the rescaled size clears the minimum.",
                signal.coin,
                notional,
                self.min_order_usd,
            )
            return None

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
        source_key = self._signal_source_key(signal)

        # External kill-switch takes precedence over any other gate.
        self._refresh_external_kill_switch()

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

        # Canary/source caps apply to NEW live entries only.
        total_signals_today = sum(self._source_orders_today.values())
        if self.canary_mode and self.canary_max_signals_per_day > 0:
            if total_signals_today >= self.canary_max_signals_per_day:
                logger.warning(
                    "Canary signal cap reached (%d/%d) - rejecting %s (source=%s)",
                    total_signals_today,
                    self.canary_max_signals_per_day,
                    signal.coin,
                    source_key,
                )
                return None
        if self.max_orders_per_source_per_day > 0:
            used = self._source_orders_today.get(source_key, 0)
            if used >= self.max_orders_per_source_per_day:
                logger.warning(
                    "Source/day cap reached for %s (%d/%d) - rejecting %s",
                    source_key,
                    used,
                    self.max_orders_per_source_per_day,
                    signal.coin,
                )
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
            import copy as _copy
            signal = _copy.deepcopy(signal)  # Avoid mutating caller's reference
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
            requested_entry_size = float(size)

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
                if self._is_insufficient_margin_rejection(entry_result):
                    logger.warning(
                        "Skipping %s %s entry due to insufficient margin: %s",
                        coin,
                        signal_side,
                        entry_result,
                    )
                else:
                    logger.error(f"Failed to place entry order: {entry_result}")
                return entry_result

            # Update daily PnL after every execution to keep loss tracking current
            self.update_daily_pnl_from_fills()

            submitted_entry_size = float(entry_result.get("submitted_size", size) or size)
            expected_fill_size = submitted_entry_size
            exchange_reported_fill_size = self._coerce_float(
                entry_result.get("exchange_reported_fill_size"),
                0.0,
            )
            exchange_reported_fill_price = self._coerce_float(
                entry_result.get("exchange_reported_fill_price"),
                0.0,
            )
            if exchange_reported_fill_size > 0:
                expected_fill_size = exchange_reported_fill_size
            actual_fill_size = expected_fill_size
            verified_fill_price = 0.0
            fill_check = None

            # 1b. Verify the fill actually happened before placing SL/TP
            if not self.dry_run:
                try:
                    fill_check = self.verify_fill(
                        coin,
                        entry_side,
                        expected_fill_size,
                        timeout=self._fill_verify_timeout_s,
                        poll_interval=self._fill_verify_poll_s,
                        blocking=self._execute_fill_verify_blocking,
                    )
                except TypeError:
                    # Backward-compatibility for monkeypatched/tests that still
                    # use the legacy verify_fill signature.
                    fill_check = self.verify_fill(
                        coin,
                        entry_side,
                        expected_fill_size,
                        timeout=self._fill_verify_timeout_s,
                        poll_interval=self._fill_verify_poll_s,
                    )
                if not fill_check:
                    if self._execute_fill_verify_blocking:
                        logger.error(
                            f"FILL NOT VERIFIED for {coin} {side} {expected_fill_size} — "
                            f"skipping SL/TP placement. Manual intervention required."
                        )
                        return {
                            "status": "warning",
                            "message": "order posted but fill not verified",
                            "coin": coin,
                            "expected_fill_size": expected_fill_size,
                            "submitted_size": submitted_entry_size,
                            "entry_result": entry_result,
                        }
                    logger.info(
                        "Fill verification pending for %s (non-blocking mode). "
                        "Proceeding with conservative protection sizing.",
                        coin,
                    )
                else:
                    actual_fill_size = float(
                        fill_check.get("size", expected_fill_size) or expected_fill_size
                    )
                    verified_fill_price = self._coerce_float(fill_check.get("entry_price"), 0.0)
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
                            "Partial fill accepted for %s: protecting at least %.6f of requested %.6f",
                            coin,
                            actual_fill_size,
                            submitted_entry_size,
                        )

            # Protective sizing:
            # - If we verified a concrete position, protect that observed size.
            # - If verification is pending/non-blocking, protect intended size.
            observed_position_size = 0.0
            if fill_check:
                observed_position_size = abs(
                    self._coerce_float(fill_check.get("position_size"), 0.0)
                )
                protective_size = max(actual_fill_size, observed_position_size)
            else:
                protective_size = max(expected_fill_size, submitted_entry_size)
            if protective_size <= 0:
                logger.error(
                    "Cannot place protective orders for %s: computed size is invalid "
                    "(actual=%.6f expected=%.6f submitted=%.6f)",
                    coin,
                    actual_fill_size,
                    expected_fill_size,
                    submitted_entry_size,
                )
                close_result = self.close_position(coin) if not self.dry_run else None
                return {
                    "status": "error",
                    "message": "invalid_protective_size",
                    "coin": coin,
                    "entry_result": entry_result,
                    "close_result": close_result,
                }
            if protective_size > actual_fill_size * 1.01:
                logger.info(
                    "Protective size for %s expanded from %.6f to %.6f to cover potential "
                    "partial-fill remainders.",
                    coin,
                    actual_fill_size,
                    protective_size,
                )

            # 2. Calculate stop loss and take profit prices from actual fill price when available.
            entry_anchor_price = exchange_reported_fill_price or verified_fill_price
            if entry_anchor_price <= 0:
                mid = self._get_mid_price(coin)
                if not mid:
                    logger.error(
                        "Cannot place SL/TP for %s: neither fill price nor mid price is available. "
                        "Position is UNPROTECTED — closing immediately.",
                        coin,
                    )
                    close_result = None
                    if not self.dry_run:
                        close_result = self.close_position(coin)
                    return {
                        "status": "error",
                        "message": "entry_price_unavailable_for_sl_tp",
                        "coin": coin,
                        "entry_result": entry_result,
                        "close_result": close_result,
                    }
                entry_anchor_price = mid
                logger.warning(
                    "SL/TP for %s falling back to mid price %.6f (fill price unavailable).",
                    coin,
                    entry_anchor_price,
                )

            if side == "buy":
                sl_price = entry_anchor_price * (1 - signal.risk.stop_loss_pct)
                tp_price = entry_anchor_price * (1 + signal.risk.take_profit_pct)
            else:
                sl_price = entry_anchor_price * (1 + signal.risk.stop_loss_pct)
                tp_price = entry_anchor_price * (1 - signal.risk.take_profit_pct)

            # 3. Place stop loss
            sl_result = self.place_trigger_order(
                coin, "sell" if side == "buy" else "buy",
                protective_size, sl_price, tp_or_sl="sl"
            )

            # 4. Place take profit
            tp_result = self.place_trigger_order(
                coin, "sell" if side == "buy" else "buy",
                protective_size, tp_price, tp_or_sl="tp"
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

            # Schedule deferred protective order resize if we over-protected
            # due to pending fill verification (protective_size >> actual_fill).
            if protective_size > actual_fill_size * 1.10 and not self.dry_run:
                import threading as _threading
                _protect_side = "sell" if side == "buy" else "buy"
                _threading.Thread(
                    target=self._deferred_protective_resize,
                    args=(coin, _protect_side, protective_size, sl_price, tp_price),
                    kwargs={"tp_or_sl_type": ("sl", "tp")},
                    daemon=True,
                    name=f"protect-resize-{coin}",
                ).start()

            # Return summary
            if self._is_order_result_success(entry_result) and not self.dry_run:
                self._source_orders_today[source_key] += 1
            return {
                "status": "success",
                "coin": coin,
                "side": signal_side,
                "size": actual_fill_size,
                "protected_size": protective_size,
                "entry_fill_price": entry_anchor_price,
                "requested_size": requested_entry_size,
                "submitted_size": submitted_entry_size,
                "exchange_reported_fill_size": (
                    exchange_reported_fill_size if exchange_reported_fill_size > 0 else None
                ),
                "exchange_reported_fill_price": (
                    exchange_reported_fill_price if exchange_reported_fill_price > 0 else None
                ),
                "fill_verified": bool(fill_check),
                "leverage": signal.leverage,
                "entry_result": entry_result,
                "dry_run": self.dry_run,
                "timestamp": datetime.now(timezone.utc).isoformat()
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
            "kill_switch_reason": self._kill_switch_reason or None,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_realized_pnl": round(self.daily_realized_pnl, 2),
            "daily_unrealized_pnl": round(self.daily_unrealized_pnl, 2),
            "daily_pnl_limit": self.max_daily_loss,
            "orders_today": self.orders_today,
            "fills_today": self.fills_today,
            "source_orders_today": dict(self._source_orders_today),
            "total_entry_signals_today": int(sum(self._source_orders_today.values())),
            "max_orders_per_source_per_day": self.max_orders_per_source_per_day,
            "canary_mode": self.canary_mode,
            "canary_max_order_usd": self.canary_max_order_usd,
            "canary_max_signals_per_day": self.canary_max_signals_per_day,
            "external_kill_switch_file": self.external_kill_switch_file or None,
            "max_position_size": self.max_position_size,
            "max_order_usd": self.max_order_usd,
            "order_dedup_window_s": self._ORDER_DEDUP_WINDOW,
            "fill_verify_blocking": self._fill_verify_blocking,
            "execute_fill_verify_blocking": self._execute_fill_verify_blocking,
            "wallet_balance": dict(self._last_balance_snapshot),
            "asset_indices_loaded": len(self.asset_index_map),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
