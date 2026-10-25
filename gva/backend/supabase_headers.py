"""PostgREST headers for new Supabase secret keys and legacy service-role JWTs.

A key that starts with ``sb_`` (``sb_secret_`` or ``sb_publishable_``) is not a
JWT. PostgREST accepts it in the ``apikey`` header and rejects
``Authorization: Bearer``. A legacy service-role JWT still needs both.
"""

from __future__ import annotations


def supabase_rest_headers(
    key: str,
    *,
    prefer: str | None = None,
    content_type: bool = True,
) -> dict[str, str]:
    headers = {
        "apikey": key,
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = "application/json"
    if not key.startswith("sb_"):
        headers["Authorization"] = f"Bearer {key}"
    if prefer:
        headers["Prefer"] = prefer
    return headers
