"""Hourly and daily rollups from the local event table. Exact percentiles; volume is small."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from gva.telemetry.store import Store


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return int(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return int(ordered[low])
    weight = rank - low
    return int(round(ordered[low] + (ordered[high] - ordered[low]) * weight))


def _usage(event: dict[str, Any], field: str) -> int:
    usage = event.get("usage") or {}
    value = usage.get(field) or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rebuild(store: Store) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = store.all_events()
    hours: dict[tuple[str, str], dict[str, Any]] = defaultdict(_blank_hour)
    days: dict[tuple[str, str], dict[str, Any]] = defaultdict(_blank_day)
    for event in events:
        ts = event.get("ts") or ""
        if len(ts) < 13:
            continue
        device = event["device_id"]
        hour_key = (ts[:13] + ":00:00Z", device)
        day_key = (ts[:10], device)
        hour = hours[hour_key]
        day = days[day_key]
        etype = event.get("type")
        if etype == "turn.completed":
            for bucket in (hour, day):
                bucket["turns"] += 1
                if event.get("status") == "ok":
                    bucket["turns_ok"] += 1
                else:
                    bucket["errors"] += 1
                if event.get("e2e_first_audio_ms") is not None:
                    bucket["e2e"].append(int(event["e2e_first_audio_ms"]))
            timings = event.get("stage_timings") or {}
            if timings.get("stt_ms") is not None:
                hour["stt"].append(int(timings["stt_ms"]))
                day["stt"].append(int(timings["stt_ms"]))
        elif etype == "llm.completed":
            for bucket in (hour, day):
                bucket["tokens_prompt"] += _usage(event, "prompt_tokens")
                bucket["tokens_completion"] += _usage(event, "completion_tokens")
                bucket["tokens_reasoning"] += _usage(event, "reasoning_tokens")
                bucket["tokens_cached"] += _usage(event, "cached_tokens")
                cost = event.get("cost_usd")
                if isinstance(cost, (int, float)):
                    bucket["cost_usd"] += float(cost)
                if event.get("ttft_ms") is not None:
                    bucket["ttft"].append(int(event["ttft_ms"]))
        elif etype == "wake.detected":
            hour["wake_events"] += 1
            if event.get("false_wake"):
                hour["false_wakes"] += 1
        elif etype == "heartbeat":
            hour["heartbeat_minutes"] += 1
            day["heartbeats"] += 1

    hourly_rows = [_finish_hour(key, value) for key, value in sorted(hours.items())]
    daily_rows = [_finish_day(key, value) for key, value in sorted(days.items())]
    store.replace_rollups(hourly_rows, daily_rows)
    return hourly_rows, daily_rows


def _blank_hour() -> dict[str, Any]:
    return {
        "turns": 0,
        "turns_ok": 0,
        "errors": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_reasoning": 0,
        "tokens_cached": 0,
        "cost_usd": 0.0,
        "e2e": [],
        "ttft": [],
        "stt": [],
        "wake_events": 0,
        "false_wakes": 0,
        "heartbeat_minutes": 0,
    }


def _blank_day() -> dict[str, Any]:
    blank = _blank_hour()
    blank["heartbeats"] = 0
    return blank


def _finish_hour(key: tuple[str, str], value: dict[str, Any]) -> dict[str, Any]:
    bucket, device = key
    return {
        "bucket": bucket,
        "device_id": device,
        "turns": value["turns"],
        "turns_ok": value["turns_ok"],
        "errors": value["errors"],
        "tokens_prompt": value["tokens_prompt"],
        "tokens_completion": value["tokens_completion"],
        "tokens_reasoning": value["tokens_reasoning"],
        "tokens_cached": value["tokens_cached"],
        "cost_usd": round(value["cost_usd"], 6),
        "p50_e2e_ms": _percentile(value["e2e"], 50),
        "p95_e2e_ms": _percentile(value["e2e"], 95),
        "p50_ttft_ms": _percentile(value["ttft"], 50),
        "p95_ttft_ms": _percentile(value["ttft"], 95),
        "p50_stt_ms": _percentile(value["stt"], 50),
        "p95_stt_ms": _percentile(value["stt"], 95),
        "wake_events": value["wake_events"],
        "false_wakes": value["false_wakes"],
        "heartbeat_minutes": min(60, value["heartbeat_minutes"]),
    }


def _finish_day(key: tuple[str, str], value: dict[str, Any]) -> dict[str, Any]:
    bucket, device = key
    heartbeats = value.get("heartbeats", 0)
    uptime = round(min(100.0, heartbeats / 1440.0 * 100.0), 2)
    return {
        "bucket": bucket,
        "device_id": device,
        "turns": value["turns"],
        "turns_ok": value["turns_ok"],
        "errors": value["errors"],
        "tokens_prompt": value["tokens_prompt"],
        "tokens_completion": value["tokens_completion"],
        "tokens_reasoning": value["tokens_reasoning"],
        "tokens_cached": value["tokens_cached"],
        "cost_usd": round(value["cost_usd"], 6),
        "p50_e2e_ms": _percentile(value["e2e"], 50),
        "p95_e2e_ms": _percentile(value["e2e"], 95),
        "p50_ttft_ms": _percentile(value["ttft"], 50),
        "p95_ttft_ms": _percentile(value["ttft"], 95),
        "p50_stt_ms": _percentile(value["stt"], 50),
        "p95_stt_ms": _percentile(value["stt"], 95),
        "uptime_pct": uptime,
    }
