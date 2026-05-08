"""
Hyperliquid L1 Signing & Wire-Format Helpers
=============================================
Extracted from live_trader.py to keep that file focused on the order-flow
state machine. This module owns:

  - OrderType enum
  - Wire-format helpers (_hl_normalize_decimal, _hl_format_price,
    _hl_format_size) that mirror float_to_wire / price_to_wire from the
    official hyperliquid-python-sdk.
  - HyperliquidSigner — EIP-712 + msgpack-keccak signing for L1 actions.

Public symbols are re-exported from src.trading.live_trader so existing
imports (tests, downstream callers) keep working unchanged.
"""

import hashlib
import json
import logging
import os
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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
