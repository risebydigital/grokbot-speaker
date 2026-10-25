"""Evaluate local alert rules. Channel secrets stay in the environment."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from gva.config import env_value
from gva.ids import new_ulid, utc_now_ms
from gva.telemetry.store import Store

OPERATORS = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
    "=": lambda value, threshold: value == threshold,
}


def seed_defaults(store: Store) -> None:
    if store.alert_rule_count() > 0:
        return
    rules = [
        ("Temperature", "temp_c", ">", 80, 15, "warning"),
        ("Disk", "disk_pct", ">", 90, 15, "critical"),
        ("API down", "api_reachable", "=", 0, 5, "critical"),
    ]
    for name, metric, operator, threshold, window, severity in rules:
        store.insert_alert_rule(
            {
                "id": new_ulid(),
                "name": name,
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "window_minutes": window,
                "severity": severity,
                "channel": "log",
                "enabled": 1,
            }
        )


def evaluate(store: Store, device_id: str) -> list[dict[str, Any]]:
    opened = []
    now = datetime.now(timezone.utc)
    for rule in store.alert_rules():
        if not rule["enabled"]:
            continue
        start = (now - timedelta(minutes=int(rule["window_minutes"]))).isoformat()
        value = _metric(store, rule["metric"], start)
        if value is None:
            continue
        compare = OPERATORS.get(rule["operator"])
        if compare is None or not compare(value, float(rule["threshold"])):
            continue
        alert = {
            "id": new_ulid(),
            "rule_id": rule["id"],
            "device_id": device_id,
            "severity": rule["severity"],
            "title": f"{rule['name']} is {value}",
            "status": "active",
            "first_at": utc_now_ms(),
            "last_at": utc_now_ms(),
            "acknowledged_by": None,
            "muted_until": None,
        }
        store.insert_alert(alert)
        opened.append(alert)
        _notify(alert)
    return opened


def _metric(store: Store, name: str, start: str) -> float | None:
    beats = store.events_between(start, "9999-01-01T00:00:00Z", "heartbeat")
    if not beats:
        return None
    latest = beats[-1]
    if name == "api_reachable":
        value = latest.get("api_reachable")
        if value is None:
            return None
        return 1.0 if value else 0.0
    raw = latest.get(name)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _notify(alert: dict[str, Any]) -> None:
    webhook = env_value("GVA_ALERT_SLACK_WEBHOOK")
    if not webhook:
        return
    try:
        httpx.post(webhook, json={"text": alert["title"]}, timeout=5)
    except httpx.HTTPError:
        return


def render_alert(alert: dict[str, Any]) -> str:
    return json.dumps({key: alert[key] for key in ("severity", "title", "status")})
