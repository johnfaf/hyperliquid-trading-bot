"""
Secret-manager adapters for agent-wallet private key loading.

Supported providers:
  - none:        read HL_AGENT_PRIVATE_KEY directly from env
  - aws_kms:     decrypt AWS_KMS_CIPHERTEXT_B64 via AWS KMS
  - hashicorp:   read key from HashiCorp Vault KV
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class SecretManagerError(RuntimeError):
    """Raised when secure secret loading fails."""


def _normalize_private_key(raw: str) -> str:
    """
    Normalize secret payload into a hex private key string.

    Accepts plain hex key or JSON payload with known key fields.
    """
    value = (raw or "").strip()
    if not value:
        raise SecretManagerError("Empty private-key payload from secret manager")

    if value.startswith("{"):
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SecretManagerError(f"Invalid JSON secret payload: {exc}") from exc
        for field in ("hl_agent_private_key", "private_key", "agent_private_key", "key"):
            candidate = str(data.get(field, "")).strip()
            if candidate:
                value = candidate
                break

    if value.startswith("0x"):
        value = value[2:]
    if len(value) != 64:
        raise SecretManagerError("Private key must be 64 hex chars (excluding 0x)")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SecretManagerError("Private key contains non-hex characters") from exc
    return "0x" + value


def _load_from_env() -> Optional[str]:
    key = os.environ.get("HL_AGENT_PRIVATE_KEY", "").strip()
    if not key:
        return None
    return _normalize_private_key(key)


def _load_from_aws_kms() -> str:
    ciphertext_b64 = os.environ.get("AWS_KMS_CIPHERTEXT_B64", "").strip()
    region = os.environ.get("AWS_KMS_REGION", "").strip()
    if not ciphertext_b64:
        raise SecretManagerError("AWS_KMS_CIPHERTEXT_B64 is required for aws_kms provider")
    if not region:
        raise SecretManagerError("AWS_KMS_REGION is required for aws_kms provider")

    try:
        import boto3
    except ImportError as exc:
        raise SecretManagerError("boto3 is required for aws_kms provider") from exc

    try:
        blob = base64.b64decode(ciphertext_b64)
    except Exception as exc:
        raise SecretManagerError(f"Invalid AWS_KMS_CIPHERTEXT_B64: {exc}") from exc

    client = boto3.client("kms", region_name=region)
    decrypt_kwargs = {"CiphertextBlob": blob}
    key_id = os.environ.get("AWS_KMS_KEY_ID", "").strip()
    if key_id:
        decrypt_kwargs["KeyId"] = key_id
    response = client.decrypt(**decrypt_kwargs)
    plaintext = response.get("Plaintext")
    if not plaintext:
        raise SecretManagerError("AWS KMS decrypt returned empty plaintext")
    return _normalize_private_key(plaintext.decode("utf-8"))


def _load_from_hashicorp_vault() -> str:
    addr = os.environ.get("VAULT_ADDR", "").strip().rstrip("/")
    token = os.environ.get("VAULT_TOKEN", "").strip()
    path = os.environ.get("VAULT_SECRET_PATH", "").strip().lstrip("/")
    secret_key = os.environ.get("VAULT_SECRET_KEY", "hl_agent_private_key").strip()
    kv_version = int(os.environ.get("VAULT_KV_VERSION", "2"))

    if not addr or not token or not path:
        raise SecretManagerError(
            "VAULT_ADDR, VAULT_TOKEN, and VAULT_SECRET_PATH are required for hashicorp provider"
        )

    url = f"{addr}/v1/{path}"
    headers = {"X-Vault-Token": token}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SecretManagerError(
            f"Vault request failed for path '{path}': {exc.__class__.__name__}"
        ) from exc
    payload = response.json()

    data = payload.get("data", {})
    if kv_version == 2 and isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    if not isinstance(data, dict):
        raise SecretManagerError("Unexpected Vault response structure for secret data")

    secret = str(data.get(secret_key, "")).strip()
    if not secret:
        raise SecretManagerError(
            f"Vault secret key '{secret_key}' not found at '{path}'"
        )
    return _normalize_private_key(secret)


def load_agent_private_key(provider: str = "none") -> Optional[str]:
    """
    Load agent private key according to selected provider.

    Returns normalized 0x-prefixed key or None if provider is "none" and env
    key is not set.
    """
    provider = (provider or "none").strip().lower()
    if provider == "none":
        return _load_from_env()
    if provider == "aws_kms":
        return _load_from_aws_kms()
    if provider in ("hashicorp", "hashicorp_vault", "vault"):
        return _load_from_hashicorp_vault()
    raise SecretManagerError(f"Unsupported SECRET_MANAGER_PROVIDER '{provider}'")
