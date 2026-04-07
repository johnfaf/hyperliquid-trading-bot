"""UTC time helpers for runtime-safe timestamp handling."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def utc_now_naive() -> datetime:
    """Return the current UTC time as a naive datetime for legacy call sites."""
    return utc_now().replace(tzinfo=None)


def utc_now_iso(*, aware: bool = False, suffix_z: bool = False) -> str:
    """Return the current UTC time as ISO text, preserving legacy naive mode by default."""
    current = utc_now() if aware else utc_now_naive()
    text = current.isoformat()
    if suffix_z:
        if aware:
            return text.replace("+00:00", "Z")
        return f"{text}Z"
    return text


def utc_from_timestamp(value: float) -> datetime:
    """Return a timezone-aware UTC datetime from a UNIX timestamp."""
    return datetime.fromtimestamp(float(value), UTC)


def utc_from_timestamp_naive(value: float) -> datetime:
    """Return a naive UTC datetime from a UNIX timestamp for legacy storage."""
    return utc_from_timestamp(value).replace(tzinfo=None)


def utc_today_str() -> str:
    """Return the current UTC date in YYYY-MM-DD form."""
    return utc_now().strftime("%Y-%m-%d")


def utc_date_str(fmt: str) -> str:
    """Format the current UTC time with *fmt*."""
    return utc_now().strftime(fmt)


def utc_timestamp() -> float:
    """Return the current UNIX timestamp in UTC seconds."""
    return utc_now().timestamp()


def utc_timestamp_ms() -> int:
    """Return the current UNIX timestamp in UTC milliseconds."""
    return int(utc_timestamp() * 1000)
