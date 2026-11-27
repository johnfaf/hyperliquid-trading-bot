"""Lossless, bounded snapshot payloads that remain valid SQLite JSON and PostgreSQL JSONB."""
import base64
import json
import zlib

BLOB = "_hl_snapshot_zlib_v1"
SIZE = "_hl_snapshot_original_bytes"
MAX_BYTES = 4 * 1024**2


def encode(payload: str) -> str:
    raw = payload.encode("utf-8")
    if not 256 <= len(raw) <= MAX_BYTES:
        return payload
    blob = base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")
    packed = json.dumps({BLOB:blob, SIZE:len(raw)}, separators=(",", ":"))
    return packed if len(packed) < len(raw) else payload


def decode(payload: str | dict) -> dict:
    body = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(body, dict):
        raise ValueError("market snapshot is not an object")
    if body.keys() != {BLOB, SIZE}:
        return body
    size, blob = body[SIZE], body[BLOB]
    if type(size) is not int or not 0 < size <= MAX_BYTES or not isinstance(blob, str) or len(blob) > 2*MAX_BYTES:
        raise ValueError("invalid compressed snapshot envelope")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(blob, validate=True), size+1)
    except (ValueError, zlib.error) as error:
        raise ValueError("invalid compressed snapshot stream") from error
    if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("compressed snapshot length or stream boundary mismatch")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("decoded snapshot is not an object")
    return result
