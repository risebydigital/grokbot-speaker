"""Latest work digest from public.milo_context.

Brenda posts the row. Milo only mentions items that are actually in a fresh
row. A missing table, a stale row, or an empty list means no suggestions.
Item text is data for the model, never written to the event log.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gva.backend.supabase_headers import supabase_rest_headers
from gva.config import Config, env_value

_TABLE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_DEVICE_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_KINDS = {"task", "report", "meeting", "other"}
_CACHE_S = 60
_ITEM_LIMIT = 8
_UNSET = object()


def waiting_items(cfg: Config, http: Any, *, now: datetime | None = None) -> list[dict[str, str]]:
    """Clean items from the latest digest for this device, or an empty list."""
    if not cfg.bot_bridge.chat_while_waiting:
        return []
    moment = now or datetime.now(timezone.utc)
    cached = _read_cache(cfg, moment)
    if cached is not _UNSET:
        return _fresh_items(cached if isinstance(cached, dict) else None, cfg, moment)
    row = _fetch(cfg, http)
    _write_cache(cfg, row)
    return _fresh_items(row, cfg, moment)


def _fresh_items(row: dict[str, Any] | None, cfg: Config, now: datetime) -> list[dict[str, str]]:
    if not row:
        return []
    created = _parse_time(row.get("created_at"))
    if created is None:
        return []
    age = (now - created).total_seconds()
    if age > float(cfg.bot_bridge.context_max_age_s):
        return []
    return _clean_items(row.get("items"))


def _fetch(cfg: Config, http: Any) -> dict[str, Any] | None:
    table = str(cfg.bot_bridge.context_table or "").strip()
    device = str(cfg.device.id or "").strip()
    url = env_value("SUPABASE_URL")
    key = env_value("SUPABASE_SERVICE_KEY")
    if http is None or not url or not key or _TABLE_NAME.fullmatch(table) is None:
        return None
    if _DEVICE_ID.fullmatch(device) is None:
        return None
    endpoint = url.rstrip("/") + f"/rest/v1/{table}"
    try:
        response = http.get(
            endpoint,
            params={
                "device_id": f"eq.{device}",
                "select": "items,created_at",
                "order": "created_at.desc",
                "limit": "1",
            },
            headers=supabase_rest_headers(key, content_type=False),
            timeout=5.0,
        )
    except Exception:
        return None
    code = int(getattr(response, "status_code", 0) or 0)
    if code != 200:
        # 404 (table not created yet) and any other status: no suggestions.
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return None
    row = payload[0]
    return {"created_at": row.get("created_at"), "items": row.get("items")}


def _clean_items(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    items: list[dict[str, str]] = []
    for entry in raw:
        if len(items) >= _ITEM_LIMIT:
            break
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "other").strip().lower()
        if kind not in _KINDS:
            kind = "other"
        title = " ".join(str(entry.get("title") or "").split())[:120]
        if not title:
            continue
        cleaned = {"kind": kind, "title": title}
        due = entry.get("due")
        if isinstance(due, str) and due.strip():
            cleaned["due"] = " ".join(due.split())[:80]
        note = entry.get("note")
        if isinstance(note, str) and note.strip():
            cleaned["note"] = " ".join(note.split())[:160]
        items.append(cleaned)
    return items


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cache_path(cfg: Config) -> Path:
    return Path(cfg.telemetry.db_path).with_name("milo-context.json")


def _read_cache(cfg: Config, now: datetime) -> dict[str, Any] | None | object:
    """A cached row, None when a recent fetch found nothing, or `_UNSET` to fetch."""
    path = _cache_path(cfg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _UNSET
    if not isinstance(data, dict):
        return _UNSET
    fetched = data.get("fetched_at")
    try:
        fetched_at = float(fetched)
    except (TypeError, ValueError):
        return _UNSET
    if now.timestamp() - fetched_at > _CACHE_S:
        return _UNSET
    row = data.get("row")
    if row is None:
        return None
    if not isinstance(row, dict):
        return _UNSET
    return row


def _write_cache(cfg: Config, row: dict[str, Any] | None) -> None:
    path = _cache_path(cfg)
    payload = {"fetched_at": time.time(), "row": row}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return
