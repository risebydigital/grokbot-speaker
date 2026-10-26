"""Tool handlers. Results stay in memory; logs receive redacted arguments only."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from gva.ids import new_ulid


def _zone(ctx) -> ZoneInfo:
    try:
        return ZoneInfo(ctx.cfg.device.timezone)
    except Exception:
        return ZoneInfo("UTC")


def set_timer(args: dict[str, Any], ctx) -> dict[str, Any]:
    now = ctx.now().astimezone(_zone(ctx))
    duration = int(args["duration_seconds"])
    ends = now + timedelta(seconds=duration)
    timer_id = new_ulid()
    ctx.store.upsert_timer(
        {
            "id": timer_id,
            "label": args.get("label"),
            "duration_seconds": duration,
            "started_at": now.isoformat(),
            "ends_at": ends.isoformat(),
            "status": "running",
        }
    )
    return {"ok": True, "id": timer_id, "label": args.get("label"), "duration_seconds": duration, "ends_at": ends.isoformat()}


def _running(ctx) -> list[dict[str, Any]]:
    return ctx.store.list_timers("running")


def _pick(timers: list[dict[str, Any]], label: str | None) -> dict[str, Any] | None:
    if label:
        folded = label.casefold()
        for timer in timers:
            if (timer.get("label") or "").casefold() == folded:
                return timer
        return None
    if len(timers) == 1:
        return timers[0]
    return None


def cancel_timer(args: dict[str, Any], ctx) -> dict[str, Any]:
    timers = _running(ctx)
    chosen = _pick(timers, args.get("label"))
    if chosen is None:
        return {"ok": False, "error": "not_found", "running": len(timers)}
    chosen["status"] = "cancelled"
    ctx.store.upsert_timer(chosen)
    return {"ok": True, "cancelled": chosen.get("label") or chosen["id"]}


def update_timer(args: dict[str, Any], ctx) -> dict[str, Any]:
    timers = _running(ctx)
    chosen = _pick(timers, args.get("label"))
    if chosen is None:
        return {"ok": False, "error": "not_found", "running": len(timers)}
    now = ctx.now().astimezone(_zone(ctx))
    duration = int(args["duration_seconds"])
    chosen["duration_seconds"] = duration
    chosen["ends_at"] = (now + timedelta(seconds=duration)).isoformat()
    ctx.store.upsert_timer(chosen)
    return {"ok": True, "label": chosen.get("label"), "duration_seconds": duration}


def list_timers(args: dict[str, Any], ctx) -> dict[str, Any]:
    now = ctx.now().astimezone(_zone(ctx))
    items = []
    for timer in _running(ctx):
        ends = datetime.fromisoformat(timer["ends_at"])
        if ends.tzinfo is None:
            ends = ends.replace(tzinfo=_zone(ctx))
        left = max(0, int((ends - now).total_seconds()))
        items.append({"label": timer.get("label"), "seconds_left": left})
    return {"ok": True, "timers": items}


def set_alarm(args: dict[str, Any], ctx) -> dict[str, Any]:
    zone = _zone(ctx)
    now = ctx.now().astimezone(zone)
    hour, minute = (int(part) for part in args["time"].split(":"))
    day = args.get("day")
    fires = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day == "tomorrow" or fires <= now:
        fires = fires + timedelta(days=1)
    if day == "today" and fires <= now:
        return {"ok": False, "error": "that time today has already passed"}
    alarm_id = new_ulid()
    ctx.store.upsert_alarm(
        {
            "id": alarm_id,
            "label": args.get("label"),
            "time_hhmm": args["time"],
            "day": day,
            "fires_at": fires.isoformat(),
            "status": "set",
        }
    )
    return {"ok": True, "id": alarm_id, "time": args["time"], "fires_at": fires.isoformat()}


def get_weather(args: dict[str, Any], ctx) -> dict[str, Any]:
    location = args.get("location") or ctx.cfg.device.default_location
    if ctx.http is None:
        return {"ok": False, "error": "weather_unavailable"}
    try:
        found = ctx.http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=5,
        )
        found.raise_for_status()
        results = (found.json() or {}).get("results") or []
        if not results:
            return {"ok": False, "error": "unknown_place", "location": location}
        place = results[0]
        forecast = ctx.http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": ctx.cfg.device.timezone,
                "forecast_days": 3,
            },
            timeout=5,
        )
        forecast.raise_for_status()
        body = forecast.json()
    except Exception:
        return {"ok": False, "error": "weather_unavailable", "location": location}
    current = body.get("current") or {}
    daily = body.get("daily") or {}
    return {
        "ok": True,
        "location": place.get("name") or location,
        "range": args.get("range") or "today",
        "temperature_c": current.get("temperature_2m"),
        "today_max_c": (daily.get("temperature_2m_max") or [None])[0],
        "today_min_c": (daily.get("temperature_2m_min") or [None])[0],
        "rain_probability_pct": (daily.get("precipitation_probability_max") or [None])[0],
    }


def switch_device(args: dict[str, Any], ctx) -> dict[str, Any]:
    known = {device.id: device for device in ctx.cfg.home.devices}
    device_id = args["device"]
    if device_id not in known:
        return {"ok": False, "error": "unknown_device"}
    ctx.store.set_device(device_id, args["state"], None)
    return {"ok": True, "device": device_id, "name": known[device_id].name, "state": args["state"]}


def set_device_level(args: dict[str, Any], ctx) -> dict[str, Any]:
    known = {device.id: device for device in ctx.cfg.home.devices if "level" in device.kinds}
    device_id = args["device"]
    if device_id not in known:
        return {"ok": False, "error": "unknown_device"}
    level = int(args["level"])
    ctx.store.set_device(device_id, "on" if level else "off", level)
    return {"ok": True, "device": device_id, "name": known[device_id].name, "level": level}


def defer_until_answer(args: dict[str, Any], ctx) -> dict[str, Any]:
    """Leave the call. Milo nudges when the team answer arrives."""
    bridge = getattr(ctx, "bridge", None)
    if bridge is None or not hasattr(bridge, "defer"):
        return {"status": "nothing_pending"}
    try:
        result = bridge.defer()
    except Exception:
        return {"status": "nothing_pending"}
    status = str(result.get("status") or "nothing_pending")
    if status != "deferred":
        return {"status": "nothing_pending"}
    return {"status": "deferred"}


def check_team_answer(args: dict[str, Any], ctx) -> dict[str, Any]:
    """Return a team answer that arrived after Milo had already given up waiting."""
    bridge = getattr(ctx, "bridge", None)
    if bridge is None:
        return {"status": "waiting"}
    try:
        answers = bridge.recall(deliver=False)
    except Exception:
        return {"status": "waiting"}
    if not answers:
        return {"status": "waiting"}
    return {"status": "ready", "answer": " ".join(answers)}


def ask_rbd_team(args: dict[str, Any], ctx) -> dict[str, Any]:
    """Hand a Rise By Digital work question to the bot bridge. No polling here."""
    question = str(args.get("question") or "").strip()
    if not question:
        return {"status": "error", "error": "missing question"}
    bridge = getattr(ctx, "bridge", None)
    if bridge is None:
        return {"status": "error", "error": "could not reach the team"}
    try:
        bridge.submit(question)
    except Exception:
        return {"status": "error", "error": "could not reach the team"}
    return {"status": "sent"}


def get_calendar_events(args: dict[str, Any], ctx) -> dict[str, Any]:
    """No calendar backend is configured. Say so. Do not invent events."""
    return {"ok": False, "error": "calendar not connected", "range": args.get("range") or "today"}


def get_time(args: dict[str, Any], ctx) -> dict[str, Any]:
    now = ctx.now().astimezone(_zone(ctx))
    what = args.get("what") or "time"
    return {
        "ok": True,
        "what": what,
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%A ") + f"{now.day} " + now.strftime("%B %Y"),
    }


HANDLERS = {
    "set_timer": set_timer,
    "cancel_timer": cancel_timer,
    "update_timer": update_timer,
    "list_timers": list_timers,
    "set_alarm": set_alarm,
    "get_weather": get_weather,
    "switch_device": switch_device,
    "set_device_level": set_device_level,
    "get_calendar_events": get_calendar_events,
    "get_time": get_time,
    "ask_rbd_team": ask_rbd_team,
    "check_team_answer": check_team_answer,
    "defer_until_answer": defer_until_answer,
}
