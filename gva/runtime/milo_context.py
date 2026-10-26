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
_DETAILS_LIMIT = 800
_OWNER_LIMIT = 80
_STATUS_LIMIT = 40
_LINK_LIMIT = 300
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_UNSET = object()
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def waiting_items(cfg: Config, http: Any, *, now: datetime | None = None) -> list[dict[str, str]]:
    """Clean items from the latest digest for this device, or an empty list."""
    if not cfg.bot_bridge.chat_while_waiting:
        return []
    status, items = load_digest(cfg, http, now=now)
    if status != "fresh":
        return []
    return items


def load_digest(cfg: Config, http: Any, *, now: datetime | None = None) -> tuple[str, list[dict[str, str]]]:
    """The latest digest, without the chat-while-waiting gate.

    Status is ``fresh``, ``empty``, ``stale``, or ``unavailable``. A failed
    read is unavailable and is not cached, so a blip does not wipe a schedule.
    """
    status, items, _created = load_digest_row(cfg, http, now=now)
    return status, items


def load_digest_row(
    cfg: Config, http: Any, *, now: datetime | None = None
) -> tuple[str, list[dict[str, str]], datetime | None]:
    """Same as ``load_digest``, plus the row time used for "as of"."""
    moment = now or datetime.now(timezone.utc)
    cached = _read_cache(cfg, moment)
    if cached is _UNSET:
        status, row = _fetch_row(cfg, http)
        if status == "unavailable":
            return "unavailable", [], None
        _write_cache(cfg, row)
    else:
        row = cached if isinstance(cached, dict) else None
    return _classify(row, cfg, moment)


def _classify(
    row: dict[str, Any] | None, cfg: Config, now: datetime
) -> tuple[str, list[dict[str, str]], datetime | None]:
    if not row:
        return "empty", [], None
    created = _parse_time(row.get("created_at"))
    if created is None:
        return "empty", [], None
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age = (now - created).total_seconds()
    if age > float(cfg.bot_bridge.context_max_age_s):
        return "stale", [], created
    return "fresh", _clean_items(row.get("items")), created


def _fetch_row(cfg: Config, http: Any) -> tuple[str, dict[str, Any] | None]:
    """``ok`` with a row or None, ``empty`` when the table has no digest, ``unavailable`` on a blip."""
    table = str(cfg.bot_bridge.context_table or "").strip()
    device = str(cfg.device.id or "").strip()
    url = env_value("SUPABASE_URL")
    key = env_value("SUPABASE_SERVICE_KEY")
    if http is None or not url or not key or _TABLE_NAME.fullmatch(table) is None:
        return "unavailable", None
    if _DEVICE_ID.fullmatch(device) is None:
        return "unavailable", None
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
        return "unavailable", None
    code = int(getattr(response, "status_code", 0) or 0)
    if code != 200:
        # 404 (table not created yet) and any other status: no digest.
        return "empty", None
    try:
        payload = response.json()
    except Exception:
        return "empty", None
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return "empty", None
    row = payload[0]
    return "ok", {"created_at": row.get("created_at"), "items": row.get("items")}


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
        details = _plain(entry.get("details"), _DETAILS_LIMIT)
        if details:
            cleaned["details"] = details
        owner = _plain(entry.get("owner"), _OWNER_LIMIT)
        if owner:
            cleaned["owner"] = owner
        status = _plain(entry.get("status"), _STATUS_LIMIT)
        if status:
            cleaned["status"] = status
        link = _https_link(entry.get("link"))
        if link:
            cleaned["link"] = link
        items.append(cleaned)
    return items


def offer_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    """Titles only. Details stay out of the offer the model hears first."""
    cap = max(0, int(limit))
    offered: list[dict[str, str]] = []
    for item in items:
        if len(offered) >= cap:
            break
        title = " ".join(str(item.get("title") or "").split())
        if not title:
            continue
        kind = str(item.get("kind") or "other").strip().lower()
        if kind not in _KINDS:
            kind = "other"
        offered.append({"kind": kind, "title": title})
    return offered


def match_item(items: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """The digest row whose title the question is about, if one matches."""
    folded = " ".join((query or "").casefold().split())
    if not folded:
        return None
    best: dict[str, Any] | None = None
    best_len = 0
    for item in items:
        title = " ".join(str(item.get("title") or "").split())
        key = title.casefold()
        if not key:
            continue
        if key in folded or folded in key:
            if len(key) > best_len:
                best = item
                best_len = len(key)
    return best


def title_mentioned(question: str, titles: list[str]) -> str:
    """Longest digest title named in the question, or an empty string."""
    folded = " ".join((question or "").casefold().split())
    if not folded:
        return ""
    found = [title.strip() for title in titles if title and title.casefold().strip() in folded]
    if not found:
        return ""
    return max(found, key=len)


def spoken_update(item: dict[str, Any], created: datetime | None, zone_name: str) -> str | None:
    """One sentence from the pre-fetched update. The link is never included."""
    details = _without_urls(str(item.get("details") or ""))
    details = " ".join(details.split())
    if not details:
        return None
    if details[-1] not in ".!?":
        details += "."
    if len(details) > 1 and details[0].isupper() and not details[1].isupper():
        details = details[0].lower() + details[1:]
    extras: list[str] = []
    blob = details.casefold()
    status = " ".join(str(item.get("status") or "").split())
    owner = " ".join(str(item.get("owner") or "").split())
    if status and status.casefold() not in blob:
        extras.append(f"The status is {status}.")
    if owner and owner.casefold() not in blob:
        extras.append(f"{owner} owns it.")
    tail = (" " + " ".join(extras)) if extras else ""
    line = f"As of {update_phrase(created, zone_name)}, {details}{tail}"
    line = " ".join(_without_urls(line).split())
    link = str(item.get("link") or "")
    if link and link in line:
        line = " ".join(line.replace(link, " ").split())
    return line or None


def update_phrase(created: datetime | None, zone_name: str) -> str:
    """How Milo attributes the digest. Brenda posts it."""
    if created is None:
        return "the latest update"
    try:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(zone_name or "Europe/London")
    except Exception:
        zone = timezone.utc
    local = created.astimezone(zone)
    return f"Brenda's {_spoken_clock(local)} update"


def ordinal_word(index: int) -> str:
    if 0 <= index < len(_ORDINALS):
        return _ORDINALS[index]
    return "next"


def labelled_line(base: str, *, label: str, ordinal: int, concurrent: bool) -> str:
    """Point a team answer at the question it belongs to."""
    if label:
        return f"On {label}: {base}"
    if concurrent:
        return f"On your {ordinal_word(ordinal)} question: {base}"
    return base


def _plain(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.replace("\x00", " ").split())
    return text[:limit]


def _https_link(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = "".join(value.split())
    if not text.lower().startswith("https://") or len(text) > _LINK_LIMIT:
        return ""
    if any(mark in text for mark in " <>\"'"):
        return ""
    return text


def _without_urls(text: str) -> str:
    return _URL.sub(" ", text)


def _spoken_clock(local: datetime) -> str:
    hour = local.hour % 12 or 12
    suffix = "am" if local.hour < 12 else "pm"
    if local.minute == 0:
        return f"{hour}{suffix}"
    return f"{hour}:{local.minute:02d}{suffix}"


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
