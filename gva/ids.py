"""ULID generator. Crockford base32, 48-bit time plus 80 bits of randomness."""

from __future__ import annotations

import os
import time

_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(timestamp_ms: int | None = None) -> str:
    ts = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    ts &= (1 << 48) - 1
    value = (ts << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_ENCODING[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def utc_now_ms() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
